"""Walk-forward validation of the hybrid architecture (gates -> decision-focused).

Each year (expanding window, no look-ahead) we RE-RUN the full validity gates on past data only
(FDR -> redundancy -> incremental-value, the actual pipeline functions) to get the validated factor
set, then RE-FIT decision-focused weights on those validated factors, and form the long-only MVO.
Tests how much of the static hybrid edge survives genuine OOS.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from factor_research.factor_validity_pipeline import (settled_history, split_period_by_settlement,
                                     load_config, single_factor_screen, factor_correlation,
                                      redundancy_filter, incremental_value_filter)
from factor_research.portfolio_construction import (ledoit_wolf_identity, month_is_tradable,
                                    solve_long_only_mvo, _sharpe, _ann_return)
from factor_research.decision_focused import fit_df
import factor_research.solve_unified_v2 as v2

TRAIN_END = pd.Timestamp("2019-12-31")
VAL_END = pd.Timestamp("2022-12-31")
TRAIN_END_P = pd.Period("2019-12", "M")
VAL_END_P = pd.Period("2022-12", "M")


def run_v2_selector(train_frame, pcfg, cfg_raw, factors, K, kappa, fdr_mode, beta_min_frac,
                    gate3_mode, time_limit, solver="auto", mip_gap=1e-2):
    """One-shot v2 MISOCP selector: returns (selected_factors, L2-normalized b dict). Replaces the
    sequential run_gates + fit_df with a single certified solve on the SAME train window."""
    tf = train_frame.copy()
    if "ym" not in tf.columns:
        tf["ym"] = pd.to_datetime(tf["date"]).dt.to_period("M")
    fam_map = cfg_raw["family_constraints"]["factor_families"]
    fam_caps = cfg_raw["family_constraints"]["max_per_family"]
    rho = float(cfg_raw["redundancy"]["max_abs_corr"])
    scr = cfg_raw.get("screening", {})
    std = v2.per_date_standardize(tf, factors, "forward_return")
    G_months, months = v2.monthly_factor_returns(std, factors, "forward_return")
    if len(months) < 6:
        return [], {}, {"selector": "one_shot_v2", "status": "too_few_months"}
    mu_f = G_months.mean(0)
    Sigma_f, _ = ledoit_wolf_identity(G_months)
    p = len(factors)
    # p-values on this train window via the pipeline's own screen (combined_p_value)
    screen = single_factor_screen(tf, pcfg)
    pv_map = dict(zip(screen["factor"].astype(str), screen["combined_p_value"].astype(float)))
    p_values = np.array([pv_map.get(f, 1.0) for f in factors])
    p_values = np.where(np.isfinite(p_values), p_values, 1.0)
    if fdr_mode == "prescreen":
        pass_map = dict(zip(screen["factor"].astype(str), screen["passes_stat_screen"].astype(bool)))
        elig = np.array([1.0 if pass_map.get(f, False) else 0.0 for f in factors])
    else:
        cov_map = dict(zip(screen["factor"].astype(str), screen["coverage"].astype(float)))
        ic_map = dict(zip(screen["factor"].astype(str), screen["mean_ic"].astype(float)))
        mc = float(scr.get("min_coverage", 0.45)); mi = float(scr.get("min_abs_ic", 0.005))
        elig = np.array([1.0 if (cov_map.get(f, 0) >= mc and abs(ic_map.get(f, 0)) > mi) else 0.0
                         for f in factors])
    Xexp = std[factors].to_numpy(float); corr = np.corrcoef(Xexp, rowvar=False)
    forbidden = [(i, j) for i in range(p) for j in range(i + 1, p) if abs(corr[i, j]) > rho]
    fam_of = [fam_map.get(f, "other") for f in factors]
    G_s = v2.build_survival_scenarios(std, factors, "forward_return", 60, 6.0, 123)
    res = v2.solve(mu_f, Sigma_f, np.zeros(p), factors, elig, forbidden, fam_of, fam_caps, G_s,
                   kappa, 0.0, K, 0.0, p_values=p_values, fdr_mode=fdr_mode,
                   fdr_q=float(scr.get("q_value_max", 0.25)), gate3_mode=gate3_mode,
                   beta_min_frac=beta_min_frac, time_limit=time_limit, solver=solver,
                   mip_gap=mip_gap)
    info = {"selector": "one_shot_v2", "status": res["status"],
            "outcome": res.get("outcome"), "incumbent": res.get("incumbent"),
            "solve_s": res["solve_s"], "time_limit": res["time_limit"],
            "time_limit_enforced": res["time_limit_enforced"],
            "time_limit_hit": res["time_limit_hit"]}
    sel_idx = res["selected_idx"]
    if not sel_idx:
        return [], {}, info
    bvec = np.array([res["b"][j] for j in sel_idx])
    nrm = np.linalg.norm(bvec) or 1.0                     # L2-normalize so score scale matches gamma tuning
    sel_factors = [factors[j] for j in sel_idx]
    return (sel_factors,
            {f: float(res["b"][j] / nrm) for f, j in zip(sel_factors, sel_idx)}, info)


def _cash_row(period, prev_weights, cost_rate, index_fwd, reason):
    """Keep an untradable month in the calendar and liquidate any existing book.

    Gross return is zero, net return includes the one-way sale cost, and the
    reported turnover equals the liquidated position mass."""
    traded = float(sum(abs(w) for w in prev_weights.values()))
    idx_ret = float(index_fwd.get(period, np.nan)) if len(index_fwd) else np.nan
    return {"ym": str(period), "split": split_of(period), "net": -cost_rate * traded,
            "ew": 0.0, "vw": 0.0, "idx": idx_ret, "turnover": traded,
            "n_validated": 0, "n_tradable": 0, "cash_month": True, "cash_reason": reason,
            "cov_n_months": 0, "cohort_label_end_max": str(_SETTLE_MAX.get(period, ""))}


_SETTLE_MAX: dict = {}


def split_of(period, settle_max=None):
    """Assign a cohort by its label settlement under the shared split boundaries."""
    return split_period_by_settlement(period, settle_max or _SETTLE_MAX,
                                     TRAIN_END, VAL_END)


class BenchmarkError(ValueError):
    """A present-but-unusable benchmark file. Distinct from an absent file: absence is a
    declared no-benchmark run, while a wrong file silently hashing to the empty string
    (e3b0c442...) and still being reported as CSI-300 is a false claim."""


_CSI300_SECID = 1.0003
# Compared after lowercasing and stripping everything but letters, digits and CJK.
# These names do NOT identify the index -- identification is the exact secid -- they
# exist to catch mislabeled files: a row that NAMES itself CSI-300 while its secid says
# another index is a conflict, refused rather than resolved.
_CSI300_NAMES = {"csi300", "csi300index", "沪深300", "沪深300指数",
                 "hs300", "hs300index", "000300", "sh000300", "shse000300", "000300sh",
                 "000300xshg"}


def _normalize_benchmark_name(name) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]", "", str(name).lower())


def _select_csi300_rows(idx: pd.DataFrame, source) -> pd.DataFrame:
    """Identify CSI-300 by exact numeric secid 1.0003.

    The name is descriptive only. Refuse conflicting canonical names, nearby numeric
    IDs and name-only identification so metrics and provenance use the same index."""
    secid_num = pd.to_numeric(idx["secid"], errors="coerce")
    secid_mask = secid_num == _CSI300_SECID
    name_mask = idx["benchmark"].map(_normalize_benchmark_name).isin(_CSI300_NAMES)
    if bool(secid_mask.any()):
        stray = name_mask & ~secid_mask
        if bool(stray.any()):
            raise BenchmarkError(
                f"benchmark file {source}: {int(stray.sum())} rows are named CSI-300 "
                f"but do not carry secid {_CSI300_SECID} -- name/secid conflict")
        return idx[secid_mask]
    if bool((name_mask & secid_num.notna()).any()):
        raise BenchmarkError(
            f"benchmark file {source}: rows are named CSI-300 but their secid is "
            f"not {_CSI300_SECID} -- name/secid conflict")
    raise BenchmarkError(
        f"benchmark file {source} contains no CSI-300 rows: no row carries the exact "
        f"secid {_CSI300_SECID}, and identification is by secid -- a CSI-300 name "
        "alone (or a name merely containing '300') is not accepted")


def _load_benchmark_rows(path, source=None) -> pd.DataFrame:
    """Load benchmark rows shared by the return series and canonical hash.

    Require the complete schema and nonempty identifiable CSI-300 rows so metrics
    and the manifest describe the same benchmark."""
    source = path if source is None else source
    idx = pd.read_csv(path)
    missing = {"date", "benchmark", "secid", "close"} - set(idx.columns)
    if missing:
        raise BenchmarkError(f"benchmark file {source} is missing columns {sorted(missing)}")
    rows = _select_csi300_rows(idx, source).copy()
    rows["date"] = pd.to_datetime(rows["date"], errors="coerce")
    rows["close"] = pd.to_numeric(rows["close"], errors="coerce")
    bad = rows["date"].isna() | rows["close"].isna() | ~np.isfinite(rows["close"]) \
        | (rows["close"] <= 0)
    if bool(bad.any()):
        raise BenchmarkError(
            f"benchmark file {source}: {int(bad.sum())} CSI-300 rows have unparseable "
            "dates or non-finite/non-positive closes")
    rows = rows.sort_values("date")
    # keep="last" silently picked whichever row the vendor export happened to order
    # last; identical duplicates are a legitimate export artifact, conflicting closes
    # for one date are not resolvable.
    n_closes = rows.groupby("date")["close"].nunique()
    conflicted = n_closes[n_closes > 1]
    if len(conflicted):
        raise BenchmarkError(
            f"benchmark file {source}: {len(conflicted)} dates carry conflicting "
            f"closes (first: {conflicted.index[0].date()}); refusing to pick one")
    rows = rows.drop_duplicates(subset="date", keep="last")
    if rows["date"].dt.to_period("M").nunique() < 12:
        raise BenchmarkError(
            f"benchmark file {source} covers fewer than 12 months of CSI-300 closes; "
            "too short to benchmark a monthly walk-forward")
    return rows[["date", "close"]]


def _series_sha256_of_rows(rows: pd.DataFrame) -> str:
    # repr is the shortest text that round-trips the IEEE double. Hashing a rounding
    # (.10g) made 4005.0 and 4005.0000001 collide, so "equal hashes" did not actually
    # mean equal prices.
    payload = "\n".join(
        f"{d:%Y-%m-%d},{float(c)!r}" for d, c in zip(rows["date"], rows["close"]))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _index_fwd_of_rows(rows: pd.DataFrame) -> pd.Series:
    ym = rows["date"].dt.to_period("M")
    me = rows.groupby(ym)["close"].last()
    return (me.shift(-1) / me - 1.0)


class BenchmarkSnapshot(NamedTuple):
    raw_sha256: str      # SHA-256 of the file bytes as read at startup
    series_sha256: str   # canonical hash of the CSI-300 series IN THOSE BYTES
    fwd: pd.Series       # forward monthly returns the run trades against


def load_benchmark_snapshot(path) -> BenchmarkSnapshot:
    """Read the benchmark file ONCE and derive everything from that byte snapshot.

    The returns the run trades against and the two hashes the manifest records must
    describe the SAME bytes: re-reading the file at manifest time, hours into a long
    run, would hash whatever the file had become while the metrics still came from the
    startup read."""
    try:
        data = Path(path).read_bytes()
        rows = _load_benchmark_rows(io.BytesIO(data), source=path)
    except BenchmarkError:
        raise
    except (OSError, ValueError) as exc:
        # EmptyDataError, ParserError and UnicodeDecodeError are ValueErrors; a
        # directory or unreadable path is an OSError. All of them are present-but-
        # unusable files, and the contract for that is a controlled refusal, not a
        # traceback that happens to have the right exit code.
        raise BenchmarkError(
            f"benchmark file {path} cannot be read as CSV "
            f"({exc.__class__.__name__}: {exc})") from exc
    return BenchmarkSnapshot(raw_sha256=hashlib.sha256(data).hexdigest(),
                             series_sha256=_series_sha256_of_rows(rows),
                             fwd=_index_fwd_of_rows(rows))


def benchmark_series_sha256(path) -> str:
    """Canonical hash of the CSI-300 price SERIES, independent of file layout.

    Equal hashes mean the same doubles, whatever the container's columns, float text
    formatting or line endings looked like. Raises BenchmarkError on an unusable file
    instead of hashing an empty series."""
    return load_benchmark_snapshot(path).series_sha256


def load_index_fwd(path):
    """Official CSI-300 forward monthly return, indexed by Period(M). Benchmark for the
    strategy. Raises BenchmarkError through the shared loader on an unusable file."""
    return load_benchmark_snapshot(path).fwd


def run_gates(train_frame, pcfg):
    screen = single_factor_screen(train_frame, pcfg)
    corr = factor_correlation(train_frame, pcfg)
    red = redundancy_filter(screen, corr, pcfg)
    red_passed = red.loc[red["passes_redundancy_screen"] == True, "factor"].astype(str).tolist()  # noqa: E712
    incr = incremental_value_filter(train_frame, red_passed, pcfg)
    return incr.loc[incr["passes_incremental"] == True, "factor"].astype(str).tolist()  # noqa: E712


def main():
    """Refit factors and weights on settled history, then evaluate each month.

    Training, covariance history and benchmark coverage use explicit availability
    checks. Save returns, weights and refit diagnostics even when the run ends with
    INSUFFICIENT_DATA; benchmark-required reports also refuse incomplete test coverage.
    """
    ap = argparse.ArgumentParser(description="Hybrid (gates->decision-focused) walk-forward.")
    ap.add_argument("--panel-csv", default="data/a_share_factor_panel.csv")
    ap.add_argument("--config", default="a_share_baseline_config.json")
    ap.add_argument("--output-dir", default="outputs/hybrid")
    ap.add_argument("--cov-window", type=int, default=24)
    ap.add_argument("--gamma", type=float, default=10.0)
    ap.add_argument("--w-max", type=float, default=0.02)
    ap.add_argument("--cost-rate", type=float, default=0.0015)
    ap.add_argument("--index-csv", default="data/public_benchmark_indices.csv")
    ap.add_argument("--no-tradable-filter", action="store_true", help="ignore the tradability flag")
    ap.add_argument("--selector", default="hybrid",
                    choices=["hybrid", "one_shot_v2", "one_shot_v2_dfweight"],
                    help="hybrid=sequential gates->decision-focused; one_shot_v2=single MISOCP (select+weight); "
                         "one_shot_v2_dfweight=MISOCP SELECTION only, then decision-focused weights (isolates selection)")
    ap.add_argument("--fdr-mode", default="by", choices=["prescreen", "bh", "by"])
    ap.add_argument("--overwrite-output", action="store_true",
                    help="delete existing walkforward_* artifacts in --output-dir")
    ap.add_argument("--min-test-months", type=int, default=1,
                    help="a formal OOS report needs a test segment; below this the run "
                         "reports INSUFFICIENT_TEST_DATA and exits non-zero")
    ap.add_argument("--require-full-test-benchmark", action="store_true",
                    help="formal-report mode: refuse (controlled non-zero exit, status "
                         "INCOMPLETE_TEST_BENCHMARK) when any test month lacks a "
                         "benchmark return or no benchmark file was provided; without "
                         "the flag, coverage gaps only warn")
    ap.add_argument("--v2-mip-gap", type=float, default=1e-2)
    ap.add_argument("--label-embargo-days", type=int, default=None,
                    help="fallback purge for panels without forward_return_end_date")
    ap.add_argument("--solver", default="auto",
                    choices=["auto", "MOSEK", "SCIP", "ECOS_BB"],
                    help="auto = first installed of MOSEK, SCIP, ECOS_BB")
    ap.add_argument("--gate3-mode", default="off", choices=["off", "betamin"])
    ap.add_argument("--beta-min-frac", type=float, default=0.0)
    ap.add_argument("--v2-K", type=int, default=6)
    ap.add_argument("--v2-kappa", type=float, default=0.6)
    ap.add_argument("--v2-time-limit", type=float, default=20.0)
    args = ap.parse_args()

    pcfg = load_config(args.config)
    cfg_raw = json.loads(Path(args.config).read_text(encoding="utf-8"))
    panel = pd.read_csv(args.panel_csv); panel["ym"] = pd.to_datetime(panel["date"]).dt.to_period("M")
    factors = [f for f in pcfg.factor_cols if f in panel.columns]
    use_tradable = (not args.no_tradable_filter) and ("tradable" in panel.columns)
    import math as _math
    if args.min_test_months < 1:
        raise SystemExit("--min-test-months must be >= 1; a formal OOS report needs a "
                         "test segment and a negative floor disables the check")
    if args.cov_window < 1:
        raise SystemExit(f"--cov-window must be >= 1, got {args.cov_window}")
    if args.label_embargo_days is not None and args.label_embargo_days < 1:
        raise SystemExit(
            f"--label-embargo-days must be a positive integer, got "
            f"{args.label_embargo_days}: a non-positive embargo cancels the label "
            "isolation it exists to provide")
    for name, value, positive in (("--gamma", args.gamma, True),
                                  ("--w-max", args.w_max, True),
                                  ("--cost-rate", args.cost_rate, False)):
        if not _math.isfinite(value) or (value <= 0 if positive else value < 0):
            raise SystemExit(
                f"{name} must be a finite {'positive' if positive else 'non-negative'} "
                f"number, got {value}")
    from factor_research.factor_validity_pipeline import label_end_series
    try:
        _wf_label_end = label_end_series(
            panel, context="factor_research.hybrid_walkforward", date_col="date",
            target_col="forward_return", allow_missing=True)
    except ValueError as exc:
        # a corrupt panel is a controlled refusal, not a Python traceback
        raise SystemExit(str(exc)) from None
    if _wf_label_end is not None:
        panel["_label_end"] = _wf_label_end
        label_availability_mode = "settlement_column"
    elif args.label_embargo_days is None:
        raise SystemExit(
            "panel has no forward_return_end_date column and no --label-embargo-days was "
            "given; refit windows would train on labels made of post-cutoff prices. "
            "Rebuild the panel with the current builder or pass an explicit embargo.")
    else:
        # Conservative proxy: assume every label needs the full embargo to settle. This
        # feeds the SAME _label_end column the refit purge, the cohort split and the
        # covariance all read, ensuring a shared eligibility rule.
        panel["_label_end"] = (pd.to_datetime(panel["date"])
                               + pd.Timedelta(days=int(args.label_embargo_days)))
        label_availability_mode = "embargo_proxy"
    if "_label_end" in panel.columns:
        _SETTLE_MAX.update(panel.groupby("ym")["_label_end"].max().to_dict())
    benchmark_available = Path(args.index_csv).exists()
    bench = None
    if benchmark_available:
        try:
            # ONE byte read: the metrics, the raw hash and the series hash must all
            # describe the same bytes, however long the run takes after this line.
            bench = load_benchmark_snapshot(args.index_csv)
        except BenchmarkError as exc:
            raise SystemExit(str(exc)) from None
        index_fwd = bench.fwd
    else:
        if args.require_full_test_benchmark:
            raise SystemExit(
                f"--require-full-test-benchmark: no benchmark file at {args.index_csv}; "
                "a formal benchmark-relative run needs one")
        index_fwd = pd.Series(dtype=float)
    solver_requested = args.solver
    args.solver = v2.resolve_solver(args.solver)   # resolve once; record both
    returns_wide = panel.pivot_table(index="ym", columns="asset", values="forward_return")
    months = list(returns_wide.index)

    b_arr, val_factors, mean, std, last_year = None, [], None, None, None
    rows, prev, refit_log, weight_rows, skip_log = [], {}, [], [], []
    for idx in range(args.cov_window, len(months)):
        t = months[idx]
        # ONE attempt per year, success or failure: `b_arr is None or ...` retried
        # every month after a failure -- 60 refits over five years instead of five.
        if t.year != last_year:
            train_frame = panel[panel["ym"] < t]
            refit_cutoff = t.to_timestamp(how="start")
            train_frame = train_frame[train_frame["_label_end"] < refit_cutoff]
            if train_frame["ym"].nunique() >= args.cov_window:
                if args.selector in ("one_shot_v2", "one_shot_v2_dfweight"):   # MISOCP selection
                    val_factors, bd, selector_info = run_v2_selector(
                        train_frame, pcfg, cfg_raw, factors,
                        args.v2_K, args.v2_kappa, args.fdr_mode,
                        args.beta_min_frac, args.gate3_mode,
                        args.v2_time_limit, solver=args.solver, mip_gap=args.v2_mip_gap)
                    if args.selector == "one_shot_v2_dfweight" and val_factors:
                        bd = fit_df(train_frame, val_factors)         # re-weight selection w/ decision-focused
                else:                                                 # sequential gates -> decision-focused
                    val_factors = [f for f in run_gates(train_frame, pcfg) if f in panel.columns]
                    bd = fit_df(train_frame, val_factors) if val_factors else {}
                    selector_info = {"selector": "gates"}
                train_cutoff = str(train_frame["ym"].max()) if len(train_frame) else None
                refit_entry = {"year": t.year, "refit_month": str(t),
                               "train_cutoff": train_cutoff,
                               "success": bool(val_factors),
                               "n_validated": len(val_factors), "validated": val_factors,
                               "weights": {k: float(v) for k, v in bd.items()},
                               "selector_info": selector_info}
                refit_log.append(refit_entry)
                last_year = t.year          # one attempt per year, success or not
                if val_factors:
                    b_arr = np.array([bd[f] for f in val_factors])
                    Xv = train_frame[val_factors].apply(pd.to_numeric, errors="coerce").to_numpy(float)
                    mean = np.nanmean(Xv, 0); std = np.nanstd(Xv, 0); std = np.where(std > 0, std, 1.0)
        if b_arr is None or not val_factors:
            skip_log.append({"ym": str(t), "reason": "no_validated_factors",
                             "n_eligible": None})
            rows.append(_cash_row(t, prev, args.cost_rate, index_fwd,
                                  "no_validated_factors"))
            prev = {}
            continue
        cross = panel[panel["ym"] == t].dropna(subset=["forward_return"]).copy()
        Z = np.nan_to_num((cross[val_factors].apply(pd.to_numeric, errors="coerce").to_numpy(float) - mean) / std)
        cross["score"] = Z @ b_arr
        # Filter settled months before taking the requested trailing covariance window.
        formation = t.to_timestamp(how="start")
        usable = settled_history(_SETTLE_MAX, formation, args.cov_window)
        if len(usable) < args.cov_window:
            skip_log.append({"ym": str(t), "reason": "covariance_window_unsettled",
                             "n_eligible": None, "cov_n_months": len(usable),
                             "cov_window_required": args.cov_window})
            rows.append(_cash_row(t, prev, args.cost_rate, index_fwd,
                                  "covariance_window_unsettled"))
            prev = {}
            continue
        hist = returns_wide.loc[usable]
        elig = [a for a in cross["asset"] if a in hist.columns and hist[a].notna().sum() >= args.cov_window]
        cross = cross[cross["asset"].isin(elig)]
        if use_tradable:                                          # only hold names actually tradable at t
            cross = cross[pd.to_numeric(cross["tradable"], errors="coerce").fillna(0) > 0]
        if not month_is_tradable(len(cross), args.w_max):
            skip_log.append({"ym": str(t), "reason": "month_not_tradable",
                             "n_eligible": int(len(cross))})
            rows.append(_cash_row(t, prev, args.cost_rate, index_fwd,
                                  "month_not_tradable"))
            prev = {}
            continue
        assets = cross["asset"].tolist()
        sigma, _ = ledoit_wolf_identity(hist[assets].fillna(0.0).to_numpy())
        w = solve_long_only_mvo(cross["score"].to_numpy(float), sigma, args.gamma, args.w_max)
        w_map = dict(zip(assets, w))
        realized = cross["forward_return"].to_numpy(float)
        gross = float(np.dot(w, realized))
        traded = sum(abs(w_map.get(a, 0.0) - prev.get(a, 0.0)) for a in set(w_map) | set(prev))
        net = gross - args.cost_rate * traded
        ew = float(np.mean(realized))
        caps = pd.to_numeric(cross["float_mktcap"], errors="coerce").to_numpy()
        vw = (float(np.dot(np.where(np.isfinite(caps), caps, 0) / np.nansum(caps), realized))
              if np.nansum(caps) > 0 else np.nan)
        idx_ret = float(index_fwd.get(t, np.nan)) if len(index_fwd) else np.nan
        rows.append({"ym": str(t), "split": split_of(t), "net": net, "ew": ew, "vw": vw, "idx": idx_ret,
                     "turnover": traded, "n_validated": len(val_factors),
                     "n_tradable": len(cross), "cash_month": False, "cash_reason": None,
                     "cov_n_months": len(usable),
                     "cohort_label_end_max": str(_SETTLE_MAX.get(t, ""))})
        # ALL solved weights, including zero/tiny ones: dropping w <= 1e-6 left the
        # monthly sums at ~0.999986 and turnover could not be recomputed exactly.
        for asset, weight in w_map.items():
            weight_rows.append({"ym": str(t), "asset": asset, "weight": float(weight)})
        prev = w_map

    df = pd.DataFrame(rows)
    # Twelve months of benchmark IN TOTAL says nothing about WHICH months: a file
    # ending in 2020 is a legal CSI-300 file, yet every later month's idx is NaN and
    # the benchmark-relative metrics quietly skip those months. Coverage is measured
    # on the idx column the metrics actually consume.
    if bench is not None and not df.empty:
        covered = df["idx"].notna()
        benchmark_missing_months = df.loc[~covered, "ym"].tolist()
        benchmark_n_months, benchmark_missing_by_split, benchmark_coverage_by_split = {}, {}, {}
        for s in ("train", "validation", "test"):
            in_split = df["split"] == s
            miss = df.loc[in_split & ~covered, "ym"].tolist()
            benchmark_n_months[s] = int((in_split & covered).sum())
            benchmark_missing_by_split[s] = miss
            # a single overall status could not be audited per segment: a formal claim
            # cares about the TEST months specifically, not the union
            benchmark_coverage_by_split[s] = (
                "NO_MONTHS" if not int(in_split.sum())
                else ("FULL" if not miss else "INCOMPLETE_BENCHMARK"))
        benchmark_coverage = ("FULL" if not benchmark_missing_months
                              else "INCOMPLETE_BENCHMARK")
        if benchmark_missing_months:
            print(f"WARNING: benchmark covers {int(covered.sum())} of {len(df)} report "
                  f"months (missing {', '.join(benchmark_missing_months[:6])}"
                  + ("..." if len(benchmark_missing_months) > 6 else "")
                  + "); benchmark-relative metrics skip the missing months")
    else:
        benchmark_missing_months, benchmark_n_months, benchmark_coverage = [], None, None
        benchmark_missing_by_split, benchmark_coverage_by_split = None, None
    od = Path(args.output_dir)
    stale = [p.name for p in od.glob("walkforward_*")] if od.exists() else []
    if stale and not args.overwrite_output:
        raise SystemExit(
            f"{od} already holds walk-forward artifacts {sorted(stale)[:4]}; reusing a "
            "directory leaves last run's returns/weights beside this run's report. Use a "
            "fresh --output-dir or pass --overwrite-output.")
    for name in stale:
        (od / name).unlink()
    od.mkdir(parents=True, exist_ok=True)
    fname = "walkforward_one_shot_v2.json" if args.selector == "one_shot_v2" else "hybrid_walkforward.json"
    from factor_research.experiment_manifest import build_manifest, write_strict_json
    manifest = build_manifest(
        panel_csv=args.panel_csv, config_path=args.config,
        extra={"selector": args.selector, "cov_window": args.cov_window,
               "gamma": args.gamma, "w_max": args.w_max, "cost_rate": args.cost_rate,
               "solver_requested": solver_requested, "solver": args.solver,
               "train_end": str(TRAIN_END.date()), "val_end": str(VAL_END.date()),
               "benchmark_available": benchmark_available,
               "benchmark_csv_sha256": (bench.raw_sha256 if bench else None),
               "benchmark_series_sha256": (bench.series_sha256 if bench else None),
               "benchmark_coverage": benchmark_coverage,
               "benchmark_coverage_by_split": benchmark_coverage_by_split,
               "benchmark_n_months": benchmark_n_months,
               "benchmark_missing_months": benchmark_missing_months,
               "benchmark_missing_months_by_split": benchmark_missing_by_split,
               "require_full_test_benchmark": bool(args.require_full_test_benchmark),
               "label_availability_mode": label_availability_mode,
               "label_embargo_days": args.label_embargo_days,
               "v2_K": args.v2_K, "v2_kappa": args.v2_kappa,
               "v2_time_limit": args.v2_time_limit, "v2_mip_gap": args.v2_mip_gap,
               "min_test_months": args.min_test_months,
               "missing_month_policy": ("a month with no tradable universe, no "
                                        "validated factors, or no settled covariance "
                                        "window is LIQUIDATED TO CASH. It appears in the "
                                        "return series as a cash row -- zero gross, the "
                                        "one-way cost of selling the book, calendar "
                                        "contiguous -- and is listed in skipped_months "
                                        "with its reason. A run whose every month is "
                                        "cash reports INSUFFICIENT_DATA.")})
    # Cash rows keep the calendar contiguous for a strategy that traded SOMETIMES.
    # A run that never traded a single month has no result to report, however
    # many cash rows it accumulated.
    if df.empty or not bool((~df["cash_month"]).any()):
        # Refuse an all-cash run as insufficient data and preserve explicit refit
        # and skip evidence explaining why no portfolio could be formed.
        from factor_research.experiment_manifest import file_sha256
        refits_json = od / "walkforward_refits.json"
        write_strict_json(refits_json, {"refits": refit_log, "skipped_months": skip_log})
        manifest["artifacts"] = {
            "walkforward_refits_json_sha256": file_sha256(refits_json)}
        write_strict_json(od / fname, {"status": "INSUFFICIENT_DATA", "manifest": manifest,
                                       "metrics": None, "refits": refit_log,
                                       "skipped_months": skip_log})
        print("no tradable months -- wrote INSUFFICIENT_DATA report, exiting non-zero")
        raise SystemExit(1)
    out = {}
    for s in ("train", "validation", "test"):
        sub = df[df["split"] == s]
        active = (sub["net"] - sub["idx"]).dropna()               # active return over CSI-300
        ir = (float(active.mean() / active.std(ddof=1) * np.sqrt(12)) if len(active) > 1 and active.std(ddof=1) > 0
              else float("nan"))
        out[s] = {"n": int(len(sub)), "sharpe_net": _sharpe(sub["net"], 12), "sharpe_ew": _sharpe(sub["ew"], 12),
                  "sharpe_vw": _sharpe(sub["vw"], 12), "sharpe_index": _sharpe(sub["idx"], 12),
                  "info_ratio_vs_index": ir, "ann_net": _ann_return(sub["net"], 12),
                  "ann_index": _ann_return(sub["idx"], 12),
                  "avg_n_validated": float(sub["n_validated"].mean()) if len(sub) else float("nan"),
                  "avg_n_tradable": float(sub["n_tradable"].mean()) if len(sub) else float("nan")}
    # Per-period artifacts: the summary Sharpe must be independently recomputable, and
    # cost/capacity/outlier-month audits need the monthly series and holdings, not a
    # digest of them. Their hashes go into the manifest so the report pins its inputs.
    from factor_research.experiment_manifest import file_sha256
    returns_csv = od / "walkforward_returns.csv"
    weights_csv = od / "walkforward_weights.csv"
    refits_json = od / "walkforward_refits.json"
    df.to_csv(returns_csv, index=False, encoding="utf-8")
    pd.DataFrame(weight_rows).to_csv(weights_csv, index=False, encoding="utf-8")
    write_strict_json(refits_json, {"refits": refit_log, "skipped_months": skip_log})
    manifest["artifacts"] = {
        "walkforward_returns_csv_sha256": file_sha256(returns_csv),
        "walkforward_weights_csv_sha256": file_sha256(weights_csv),
        "walkforward_refits_json_sha256": file_sha256(refits_json)}
    if out["test"]["n"] < args.min_test_months:
        write_strict_json(od / fname, {
            "status": "INSUFFICIENT_TEST_DATA", "manifest": manifest, "metrics": out,
            "refits": refit_log, "skipped_months": skip_log,
            "why": (f"test segment has {out['test']['n']} months, below "
                    f"--min-test-months {args.min_test_months}; a formal OOS claim "
                    "needs a test window")})
        print(f"test segment {out['test']['n']} < min {args.min_test_months} months -- "
              "INSUFFICIENT_TEST_DATA, exiting non-zero")
        raise SystemExit(1)
    if args.require_full_test_benchmark and benchmark_missing_by_split["test"]:
        test_missing = benchmark_missing_by_split["test"]
        why = (f"--require-full-test-benchmark: {len(test_missing)} test months have no "
               f"benchmark return ({', '.join(test_missing[:6])}"
               + ("..." if len(test_missing) > 6 else "")
               + "); a formal benchmark-relative claim needs the full test window covered")
        write_strict_json(od / fname, {
            "status": "INCOMPLETE_TEST_BENCHMARK", "manifest": manifest, "metrics": out,
            "refits": refit_log, "skipped_months": skip_log, "why": why})
        print(why + " -- INCOMPLETE_TEST_BENCHMARK, exiting non-zero")
        raise SystemExit(1)
    tag = "one_shot_v2 (single MISOCP)" if args.selector == "one_shot_v2" else "hybrid gates->decision-focused"
    write_strict_json(od / fname, {
        "status": "OK", "method": f"{tag}, yearly walk-forward",
        "selector": args.selector, "fdr_mode": args.fdr_mode, "gate3_mode": args.gate3_mode,
        "beta_min_frac": args.beta_min_frac, "tradable_filter": use_tradable,
        "benchmark_index": "CSI-300" if benchmark_available else None,
        "manifest": manifest, "metrics": out, "refits": refit_log,
        "skipped_months": skip_log})
    print(f"=== WALK-FORWARD [{tag}] (yearly, no look-ahead) | tradable_filter={use_tradable} ===")
    for s in ("train", "validation", "test"):
        m = out[s]
        print(f"  [{s:>10}] n={m['n']:>2}  net={m['sharpe_net']:.3f}  EW={m['sharpe_ew']:.3f}  "
              f"VW={m['sharpe_vw']:.3f}  CSI300={m['sharpe_index']:.3f}  IR={m['info_ratio_vs_index']:.3f}  "
              f"ann={m['ann_net']:.2%}(idx {m['ann_index']:.2%})  #val={m['avg_n_validated']:.1f}  "
              f"#hold={m['avg_n_tradable']:.0f}")
    print("  refit history: " + "; ".join(f"{r['year']}:{r['n_validated']}" for r in refit_log))


if __name__ == "__main__":
    main()
