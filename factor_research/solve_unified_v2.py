"""Factor-space mixed-integer conic selection with explicit solver diagnostics.

Optimize factor-portfolio mean return minus a turnover proxy under a quadratic
risk normalization, sparsity, family and redundancy constraints. Factor returns
are per-date cross-sectional managed-portfolio returns; their time-series risk
matrix differs from the exposure Gram matrix in a prediction objective.

The CLI supports prescreen or BH/BY self-consistency constraints, optional beta-min
and survival/CVaR constraints, and pair/clique redundancy formulations. These
constraints are implemented optimization rules, not an unconditional proof of
post-selection FDR control. With costs or additional constraints the objective is
not literally the unconstrained maximum Sharpe problem.

Covariance shrinkage stabilizes short training windows but cannot remove selection
bias. Use settled-label training cuts and report solver status/gap; feasible and
certified-optimal are different outcomes. Historical method rankings are not
expected outputs of this release. See docs/REPRODUCIBILITY.md for data/OOS limitations.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("MOSEKLM_LICENSE_FILE", os.path.expanduser(r"~/mosek/mosek.lic"))
warnings.filterwarnings("ignore", message=".*Incorrect array format.*")

import numpy as np
import pandas as pd
import cvxpy as cp

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from factor_research.portfolio_construction import ledoit_wolf_identity  # reuse LW shrinkage (LW 2004)


def maximal_cliques(n, edges):
    """Bron-Kerbosch (no pivot) maximal cliques of the undirected graph on {0..n-1} with `edges`.
    A clique C of the redundancy graph (|corr|>rho within C) admits at most one selected factor, so
    sum_{j in C} z_j <= 1 -- strictly tighter than the pairwise constraints (which only give
    sum <= |C|/2 in the LP relaxation), tightening the branch-and-bound."""
    adj = {i: set() for i in range(n)}
    for (i, j) in edges:
        adj[i].add(j); adj[j].add(i)
    cliques = []

    def bk(R, P, X):
        if not P and not X:
            if len(R) >= 2:
                cliques.append(sorted(R))
            return
        for v in list(P):
            bk(R | {v}, P & adj[v], X & adj[v])
            P = P - {v}; X = X | {v}

    bk(set(), set(range(n)), set())
    return cliques


def stationary_bootstrap_months(months, mean_block, rng):
    n = len(months)
    p_restart = 1.0 / max(mean_block, 1.0001)
    out, cur = [], int(rng.integers(0, n))
    for i in range(n):
        if i > 0 and rng.random() < p_restart:
            cur = int(rng.integers(0, n))
        out.append(months[cur]); cur = (cur + 1) % n
    return out


def per_date_standardize(df, factors, target_col, date_col="ym"):
    """Per-rebalance-date cross-sectional z-score of each factor (demean -> long-short weights sum to
    zero; divide by cross-sectional std -> comparable scale). Returns a copy with standardized factor
    columns and a demeaned target. NaNs -> 0 (= the cross-sectional mean after standardization)."""
    carry = ["asset"] if "asset" in df.columns else []
    out = df[[date_col, target_col] + carry + list(factors)].copy()
    for f in list(factors) + [target_col]:
        out[f] = pd.to_numeric(out[f], errors="coerce")
    g = out.groupby(date_col)
    for f in factors:
        mu = g[f].transform("mean")
        sd = g[f].transform("std")
        out[f] = ((out[f] - mu) / sd).where(sd > 0, 0.0)
    # demean the target cross-sectionally so g_t is a pure long-short cross-moment
    out[target_col] = out[target_col] - g[target_col].transform("mean")
    out[list(factors)] = out[list(factors)].fillna(0.0)
    out[target_col] = out[target_col].fillna(0.0)
    return out


def monthly_factor_returns(std_df, factors, target_col, date_col="ym"):
    """g_t = X_t' y_t / n_t for each month t. Returns (G  [T x p],  month_index)."""
    g_rows, months = [], []
    for m, block in std_df.groupby(date_col, sort=True):
        Xt = block[list(factors)].to_numpy(float)
        yt = block[target_col].to_numpy(float)
        nt = len(yt)
        if nt < 5:
            continue
        g_rows.append(Xt.T @ yt / nt)
        months.append(m)
    return np.array(g_rows), months


def turnover_proxy(std_df, factors, date_col="ym"):
    """TO_j = 1 - mean consecutive-month cross-sectional correlation of factor j (on common stocks).
    High autocorrelation of the standardized exposure -> low portfolio turnover -> small TO_j."""
    if "asset" not in std_df.columns:
        return np.zeros(len(factors))
    months = list(pd.unique(std_df[date_col]))
    to = np.zeros(len(factors))
    for j, f in enumerate(factors):
        wide = std_df.pivot_table(index="asset", columns=date_col, values=f)
        cols = [m for m in months if m in wide.columns]
        corrs = []
        for a, b in zip(cols[:-1], cols[1:]):
            pair = wide[[a, b]].dropna()
            if len(pair) >= 10 and pair[a].std() > 0 and pair[b].std() > 0:
                corrs.append(pair[a].corr(pair[b]))
        to[j] = float(np.clip(1.0 - (np.mean(corrs) if corrs else 0.0), 0.0, 2.0))
    return to


def build_survival_scenarios(std_df, factors, target_col, n_scenarios, mean_block, seed, date_col="ym"):
    """g_s on stationary-bootstrap-resampled months, same estimator as g_t. Returns G_s [S x p]."""
    months = list(pd.unique(std_df[date_col]))
    idx_by = {m: np.where(std_df[date_col].to_numpy() == m)[0] for m in months}
    Xall = std_df[list(factors)].to_numpy(float)
    yall = std_df[target_col].to_numpy(float)
    rng = np.random.default_rng(seed)
    gs = []
    for _ in range(n_scenarios):
        ms = stationary_bootstrap_months(months, mean_block, rng)
        rows = np.concatenate([idx_by[m] for m in ms])
        gs.append(Xall[rows].T @ yall[rows] / len(rows))
    return np.array(gs)


def resolve_solver(choice: str) -> str:
    """'auto' picks the first installed of MOSEK, SCIP, ECOS_BB. The default must not be
    a commercial solver the [opt] extra does not install."""
    if choice != "auto":
        return choice
    installed = set(cp.installed_solvers())
    for name in ("MOSEK", "SCIP", "ECOS_BB"):
        if name in installed:
            return name
    raise SystemExit("no MI-capable solver installed; `pip install .[opt]` provides SCIP")


def classify_outcome(status: str | None, has_incumbent: bool,
                     native_status: str | None = None) -> tuple[str, bool]:
    """Map solver status and incumbent availability to an explicit outcome.

    Only OPTIMAL and FEASIBLE_INACCURATE are usable zero-exit results. Preserve
    infeasible, unbounded and missing-incumbent diagnoses without presenting
    an unavailable solution as a portfolio."""
    # Definitive model verdicts first: they say WHY there is nothing to use, and
    # collapsing them into NO_INCUMBENT would throw that diagnosis away.
    if status in ("infeasible", "infeasible_inaccurate"):
        return "INFEASIBLE", False
    if status in ("unbounded", "unbounded_inaccurate"):
        return "UNBOUNDED", False
    if not has_incumbent:
        # cvxpy can report optimal with no usable variable values; "optimal" is a claim
        # about the model, an incumbent is what the caller actually consumes
        return "NO_INCUMBENT", False
    if native_status and "timelimit" in str(native_status).lower():
        # SCIP reports its own termination reason; a solution found under a hit limit is
        # usable but is NOT a certified optimum and must not be reported as one
        return "FEASIBLE_LIMIT", True
    if status == "optimal":
        return "OPTIMAL", True
    if status == "optimal_inaccurate":
        return "FEASIBLE_INACCURATE", True
    return "NO_INCUMBENT", False


def solver_options(solver: str, time_limit: float, mip_gap: float) -> tuple[dict, bool]:
    """Per-solver limit/tolerance plumbing. MOSEK-only limits meant the default `auto`
    environment (SCIP) ran unbounded -- an 800x240 sweep could run for hours with no
    ceiling. Returns (solve kwargs, whether a wall-clock limit is actually enforced);
    ECOS_BB has no time limit, so callers must know the ceiling is iterations only."""
    import math
    if not math.isfinite(time_limit) or time_limit <= 0:
        raise ValueError(f"time_limit must be a finite positive number, got {time_limit}")
    if not math.isfinite(mip_gap) or mip_gap < 0:
        raise ValueError(f"mip_gap must be finite and >= 0, got {mip_gap}")
    kwargs = {"solver": getattr(cp, solver)}
    if solver == "MOSEK":
        # Record time and MIP-gap limits so reports distinguish feasible incumbents
        # from certified optima. A zero gap requests exact optimality.
        kwargs["mosek_params"] = {"MSK_DPAR_MIO_TOL_REL_GAP": mip_gap,
                                  "MSK_DPAR_MIO_MAX_TIME": time_limit,
                                  "MSK_DPAR_OPTIMIZER_MAX_TIME": time_limit}
        return kwargs, True
    if solver == "SCIP":
        kwargs["scip_params"] = {"limits/time": float(time_limit),
                                 "limits/gap": float(mip_gap)}
        return kwargs, True
    if solver == "ECOS_BB":
        kwargs["mi_max_iters"] = 10000
        kwargs["mi_rel_eps"] = float(mip_gap)
        return kwargs, False
    return kwargs, False


def solve(mu_f, Sigma_f, TO, factors, elig, forbidden, fam_of, fam_caps, G_s,
          kappa, tau, max_card, lambda_cost, solver="auto", bigM=None,
          p_values=None, fdr_mode="prescreen", fdr_q=0.25,
          gate3_mode="off", beta_min_frac=0.0, mip_gap=1e-2, time_limit=60.0,
          redundancy_mode="clique"):
    """Solve the factor-space mixed-integer conic allocation with the declared gate constraints.
    Return solver diagnostics as well as coefficients; a time-limited feasible solution is not an optimality proof."""
    p = len(factors)
    # big-M so the SOC ball (not the box) is the binding scale: max|b_j| under b'Sigma_f b<=1 is
    # 1/sqrt(lambda_min(Sigma_f)); give 10% headroom.
    lam_min = float(np.linalg.eigvalsh(Sigma_f)[0])
    max_abs_b = 1.0 / np.sqrt(max(lam_min, 1e-10))
    if bigM is None:
        bigM = 1.1 * max_abs_b
    # beta-min threshold in b-space, expressed as a scale-free fraction of the max achievable |b_j|
    eps = float(beta_min_frac) * max_abs_b

    b = cp.Variable(p)
    z = cp.Variable(p, boolean=True)
    a = cp.Variable(p, nonneg=True)                       # a_j = |b_j|
    cons = [z <= elig,
            b <= bigM * z, b >= -bigM * z,
            a >= b, a >= -b, a <= bigM * z,
            cp.sum(z) <= max_card,
            cp.quad_form(b, cp.psd_wrap(Sigma_f)) <= 1.0]  # SOC normalization

    # --- Gate 3 endogenized: incremental-value via beta-min disjunction ---
    # A selected factor must carry a weight of magnitude >= eps in the joint MVE direction, else it is
    # dropped. This is the in-solver analogue of the Fama-MacBeth joint-regression gate: a correlated
    # fake factor, once real factors absorb its signal, gets |b_j| < eps and cannot be selected.
    #   z_j=1, s_j=1 -> b_j >= eps  (long) ;  z_j=1, s_j=0 -> b_j <= -eps  (short)
    if gate3_mode == "betamin" and eps > 0:
        s = cp.Variable(p, boolean=True)                  # sign indicator (1=long, 0=short)
        cons += [b >= cp.multiply(eps, z) - bigM * (1 - s),
                 b <= -cp.multiply(eps, z) + bigM * s]

    # --- Gate 1 endogenized: BH/BY self-consistency (Blanchard-Roquain 2008) ---
    # A rejection set R is self-consistent w.r.t. threshold beta(r) if p_j <= beta(|R|) for all j in R.
    # ANY self-consistent set controls FDR (BY shape-fn -> arbitrary dependence; BH linear -> PRDS),
    # so we let the max-Sharpe objective pick the set inside the self-consistent feasible region.
    # Linear encoding valid for all j:  p_j <= beta(sum z) + (1 - z_j)   [the (1-z_j) deactivates it
    # when z_j=0 since p_j<=1<=RHS].  beta(r) = q*r/m (BH) or q*r/(m*H_m) (BY).
    if fdr_mode in ("bh", "by") and p_values is not None:
        m = p
        Hm = float(np.sum(1.0 / np.arange(1, m + 1)))
        denom = m if fdr_mode == "bh" else m * Hm
        slope = fdr_q / denom
        pv = np.asarray(p_values, float)
        for j in range(p):
            cons.append(pv[j] <= slope * cp.sum(z) + (1.0 - z[j]))

    # Gate 2 redundancy: maximal-clique constraints (tighter) or raw pairwise
    if redundancy_mode == "clique":
        for C in maximal_cliques(p, forbidden):
            cons.append(cp.sum(cp.hstack([z[k] for k in C])) <= 1)
    else:
        for (i, j) in forbidden:
            cons.append(z[i] + z[j] <= 1)
    for fam, cap in fam_caps.items():
        members = [k for k in range(p) if fam_of[k] == fam]
        if members:
            cons.append(cp.sum(cp.hstack([z[k] for k in members])) <= cap)
    if G_s is not None:
        S = G_s.shape[0]
        beta = cp.Variable()
        u = cp.Variable(S, nonneg=True)
        cons += [u >= tau - G_s @ b - beta,
                 beta + cp.sum(u) / ((1.0 - kappa) * S) <= 0]

    obj = cp.Maximize(mu_f @ b - lambda_cost * (TO @ a))
    prob = cp.Problem(obj, cons)
    solver = resolve_solver(solver)
    solve_kwargs, time_limit_enforced = solver_options(solver, time_limit, mip_gap)
    t0 = time.time()
    try:
        prob.solve(**solve_kwargs)
    except cp.error.SolverError as exc:
        # a solver crash is an outcome, not a traceback
        return {"status": f"solver_error: {exc}", "solve_s": time.time() - t0,
                "time_limit": time_limit, "time_limit_enforced": time_limit_enforced,
                "time_limit_hit": None, "outcome": "NO_INCUMBENT", "incumbent": False,
                "objective": None, "selected_idx": [], "b": np.zeros(p), "bigM": bigM,
                "lam_min_Sigma_f": lam_min, "insample_monthly_sharpe": 0.0,
                "insample_ann_sharpe": 0.0}
    dt = time.time() - t0
    has_incumbent = z.value is not None and b.value is not None
    if has_incumbent:
        has_incumbent = bool(np.isfinite(np.asarray(b.value, dtype=float)).all())
    native_status = None
    try:                       # SCIP exposes its own termination reason; others may not
        native_status = (prob.solver_stats.extra_stats or {}).get("scip_status")
    except Exception:          # noqa: BLE001 - diagnostics must never break the solve
        native_status = None
    outcome, _usable = classify_outcome(prob.status, has_incumbent, native_status)
    # The wall clock includes CVXPY canonicalization and is NOT the solver's own clock:
    # it marked an optimal SCIP finish as hit=true. Native termination reasons are not
    # exposed portably through cvxpy, so anything other than a clean optimal is unknown,
    # never guessed.
    if native_status is not None:
        time_limit_hit = "timelimit" in str(native_status).lower()
    elif not time_limit_enforced:
        time_limit_hit = None
    elif prob.status == "optimal":
        time_limit_hit = False
    else:
        time_limit_hit = None

    zval = np.round(np.asarray(z.value).flatten()).astype(int) if z.value is not None else np.zeros(p, int)
    bval = np.asarray(b.value).flatten() if b.value is not None else np.zeros(p)
    sel = [k for k in range(p) if zval[k] == 1]
    # realized in-sample Sharpe of the chosen b (annualized) as a sanity read
    gross = float(mu_f @ bval)
    return {"status": prob.status, "outcome": outcome, "incumbent": has_incumbent,
            "native_status": native_status,
            "solve_s": dt, "time_limit": time_limit, "time_limit_enforced": time_limit_enforced, "time_limit_hit": time_limit_hit, "objective": float(prob.value) if prob.value is not None else None,
            "selected_idx": sel, "b": bval, "bigM": bigM, "lam_min_Sigma_f": lam_min,
            "insample_monthly_sharpe": gross, "insample_ann_sharpe": gross * np.sqrt(12.0)}


def main():
    """Validate selection provenance, purge unsettled labels and solve once.

    The CLI writes weights and a strict-JSON diagnostic under the output directory.
    Missing provenance, incompatible inputs or an unusable solver outcome return
    nonzero status instead of presenting an invalid solution as a portfolio.
    """
    ap = argparse.ArgumentParser(description="Unified v2 MISOCP factor selection (max-Sharpe objective).")
    ap.add_argument("--panel-csv", default="data/a_share_factor_panel.csv")
    ap.add_argument("--selection-csv", default="outputs/a_share_baseline/final_factor_selection.csv")
    ap.add_argument("--config", default="a_share_baseline_config.json")
    ap.add_argument("--output-dir", default="outputs/unified_v2")
    ap.add_argument("--train-end", default="2019-12")
    ap.add_argument("--max-cardinality", type=int, default=6)
    ap.add_argument("--lambda-cost", type=float, default=0.0, help="turnover penalty coefficient")
    ap.add_argument("--n-scenarios", type=int, default=100)
    ap.add_argument("--label-embargo-days", type=int, default=None,
                    help="fallback purge for panels without forward_return_end_date")
    ap.add_argument("--overwrite-output", action="store_true",
                    help="delete existing unified_v2_* / v2_weights.csv in --output-dir")
    ap.add_argument("--time-limit", type=float, default=60.0,
                    help="wall-clock ceiling passed to MOSEK/SCIP (ECOS_BB caps "
                         "iterations instead and cannot honor this)")
    ap.add_argument("--mip-gap", type=float, default=1e-2)
    ap.add_argument("--mean-block", type=float, default=6.0)
    ap.add_argument("--kappa", type=float, default=0.6)
    ap.add_argument("--tau", type=float, default=0.0)
    ap.add_argument("--no-survival", action="store_true")
    ap.add_argument("--fdr-mode", default="prescreen", choices=["prescreen", "bh", "by"],
                    help="prescreen restricts z to passes_stat_screen; bh/by=self-consistency constraint (BY default-safe)")
    ap.add_argument("--fdr-q", type=float, default=None, help="FDR level q (default: config screening.q_value_max)")
    ap.add_argument("--gate3-mode", default="off", choices=["off", "betamin"],
                    help="off=no incremental gate; betamin=|b_j|>=eps for selected (eps=frac*max|b|)")
    ap.add_argument("--beta-min-frac", type=float, default=0.0,
                    help="beta-min threshold as fraction of max achievable |b_j|")
    ap.add_argument("--redundancy-mode", default="clique", choices=["clique", "pairwise"],
                    help="clique=maximal-clique constraints (tighter); pairwise imposes z_i+z_j<=1")
    ap.add_argument("--solver", default="auto",
                    choices=["auto", "MOSEK", "SCIP", "ECOS_BB"],
                    help="auto = first installed of MOSEK, SCIP, ECOS_BB; MOSEK is "
                         "commercial and not part of the opt extra")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    solver_requested = args.solver
    args.solver = resolve_solver(args.solver)   # record the solver actually used, not "auto"
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    factors = cfg["factor_cols"]
    fam_map = cfg["family_constraints"]["factor_families"]
    fam_caps = cfg["family_constraints"]["max_per_family"]
    rho = float(cfg["redundancy"]["max_abs_corr"])
    target_col = cfg.get("target_col", "forward_return")

    panel = pd.read_csv(args.panel_csv)
    panel["ym"] = pd.to_datetime(panel["date"]).dt.to_period("M")
    factors = [f for f in factors if f in panel.columns]
    p = len(factors)
    from factor_research.factor_validity_pipeline import purge_label_overlap
    train = panel[panel["ym"] <= pd.Period(args.train_end, "M")].copy()
    train, label_purge = purge_label_overlap(
        train, pd.Period(args.train_end, "M").end_time.normalize(), "date",
        embargo_days=args.label_embargo_days, context="unified-v2 train window")

    # ---- Objective inputs: per-date standardize -> monthly factor returns g_t -> mu_f, Sigma_f
    std_df = per_date_standardize(train, factors, target_col)
    G_months, months = monthly_factor_returns(std_df, factors, target_col)
    T = len(months)
    mu_f = G_months.mean(axis=0)
    Sigma_f, shrink = ledoit_wolf_identity(G_months)
    TO = turnover_proxy(std_df, factors) if "asset" in std_df.columns else np.zeros(p)

    # ---- gate inputs: FDR domain / self-consistency, redundancy pairs, families
    scr = cfg.get("screening", {})
    min_cov = float(scr.get("min_coverage", 0.45))
    min_abs_ic = float(scr.get("min_abs_ic", 0.005))
    min_ls = float(scr.get("min_long_short", 0.0))
    fdr_q = args.fdr_q if args.fdr_q is not None else float(scr.get("q_value_max", 0.25))

    from factor_research.experiment_manifest import build_manifest, file_sha256, write_strict_json
    sel_manifest_path = Path(args.selection_csv).parent / "selection_manifest.json"
    if not sel_manifest_path.is_file():
        raise SystemExit(
            f"selection manifest missing at {sel_manifest_path}. Regenerate the selection "
            "with `python -m factor_research.factor_validity_pipeline --config <cfg> --train-end <date>`; "
            "an unverifiable (possibly full-sample) selection is refused because its "
            "p-values may already contain the evaluation window.")
    sel_manifest = json.loads(sel_manifest_path.read_text(encoding="utf-8"))
    if sel_manifest.get("panel_sha256") != file_sha256(args.panel_csv):
        raise SystemExit(
            "selection was computed on a different panel than --panel-csv "
            f"(manifest {str(sel_manifest.get('panel_sha256'))[:12]} != panel "
            f"{str(file_sha256(args.panel_csv))[:12]}); regenerate the selection")
    if sel_manifest.get("config_sha256") != file_sha256(args.config):
        raise SystemExit(
            "selection was computed under a different config than --config "
            f"(manifest {str(sel_manifest.get('config_sha256'))[:12]} != config "
            f"{str(file_sha256(args.config))[:12]}); regenerate the selection or pass "
            "the matching config")
    if sel_manifest.get("selection_csv_sha256") != file_sha256(args.selection_csv):
        raise SystemExit(
            "final_factor_selection.csv does not match the hash its manifest recorded; "
            "the file was replaced or edited after the pipeline wrote it")
    sel_train_end = sel_manifest.get("train_end")
    purge_rec = sel_manifest.get("label_purge")
    if not isinstance(purge_rec, dict) or "label_end_col_present" not in purge_rec:
        raise SystemExit(
            "selection manifest predates the label-boundary purge (no usable label_purge "
            "record), so its p-values may come from rows whose labels were built from "
            "post-cutoff prices; regenerate the selection with the current pipeline")
    if not purge_rec.get("label_end_col_present") and not purge_rec.get("embargo_days"):
        raise SystemExit(
            "the selection was fitted on a panel without settlement dates and without an "
            "embargo; its training window cannot be shown to exclude post-cutoff labels")
    if not isinstance(purge_rec.get("n_purged_label_overlap"), int) or \
            purge_rec["n_purged_label_overlap"] < 0:
        raise SystemExit(
            f"label_purge.n_purged_label_overlap is {purge_rec.get('n_purged_label_overlap')!r}; "
            "a purge count must be a non-negative integer")
    max_end = purge_rec.get("max_train_label_end")
    if purge_rec.get("label_end_col_present") and not max_end:
        raise SystemExit("label_purge records settlement dates but no max_train_label_end")
    # Compare against the SELECTION's own cutoff: checking it against this solve's window
    # would let a later solver window hide an earlier selection's crossing labels.
    if max_end and sel_train_end and pd.Timestamp(max_end) > pd.Timestamp(sel_train_end):
        raise SystemExit(
            f"the selection's latest training label settles {max_end}, after its own "
            f"train_end {sel_train_end}")
    if not sel_train_end or pd.Period(sel_train_end, "M") > pd.Period(args.train_end, "M"):
        raise SystemExit(
            f"selection train_end {sel_train_end!r} exceeds (or is missing vs) the "
            f"solver's --train-end {args.train_end}; the selection saw data this solve "
            "treats as out-of-sample")
    sel = pd.read_csv(args.selection_csv).set_index("factor")
    sel = sel.reindex(factors)
    # p-values for the self-consistency constraint (all m candidates)
    p_values = sel["combined_p_value"].to_numpy(float)
    p_values = np.where(np.isfinite(p_values), p_values, 1.0)
    if args.fdr_mode == "prescreen":
        # Prescreen domain: passes_stat_screen bundles q_value AND coverage AND ic
        elig = sel["passes_stat_screen"].fillna(False).astype(bool).to_numpy().astype(float)
    else:
        # bh/by: FDR handled by self-consistency constraint; domain = EXOGENOUS DATA QUALITY only
        # (coverage + |IC| / long-short), which is NOT statistical inference and stays a pre-filter.
        cov = sel["coverage"].to_numpy(float)
        mic = sel["mean_ic"].abs().to_numpy(float)
        mls = sel["mean_long_short"].to_numpy(float)
        elig_dq = (cov >= min_cov) & ((mic > min_abs_ic) | (mls > min_ls))
        elig = elig_dq.astype(float)
    # redundancy uses cross-sectional exposure correlation
    Xexp = std_df[factors].to_numpy(float)
    corr_exp = np.corrcoef(Xexp, rowvar=False)
    forbidden = [(i, j) for i in range(p) for j in range(i + 1, p) if abs(corr_exp[i, j]) > rho]
    fam_of = [fam_map.get(f, "other") for f in factors]

    G_s = None if args.no_survival else build_survival_scenarios(
        std_df, factors, target_col, args.n_scenarios, args.mean_block, args.seed)

    # feasible-region pre-check for the self-consistency constraint (max self-consistent cardinality)
    fdr_precheck = None
    if args.fdr_mode in ("bh", "by"):
        m = p
        Hm = float(np.sum(1.0 / np.arange(1, m + 1)))
        denom = m if args.fdr_mode == "bh" else m * Hm
        pv_dom = np.sort(p_values[elig > 0])
        maxcard = max([r for r in range(1, len(pv_dom) + 1) if pv_dom[r - 1] <= fdr_q * r / denom] + [0])
        fdr_precheck = {"mode": args.fdr_mode, "q": fdr_q, "Hm": round(Hm, 4),
                        "threshold_at_K": round(fdr_q * args.max_cardinality / denom, 6),
                        "max_self_consistent_cardinality": maxcard, "n_domain": int((elig > 0).sum())}
        print(f"[FDR pre-check] mode={args.fdr_mode} q={fdr_q} thr@K={fdr_precheck['threshold_at_K']:.5f} "
              f"max-self-consistent-card={maxcard} (domain {fdr_precheck['n_domain']})")

    res = solve(mu_f, Sigma_f, TO, factors, elig, forbidden, fam_of, fam_caps, G_s,
                args.kappa, args.tau, args.max_cardinality, args.lambda_cost, args.solver,
                p_values=p_values, fdr_mode=args.fdr_mode, fdr_q=fdr_q,
                gate3_mode=args.gate3_mode, beta_min_frac=args.beta_min_frac,
                redundancy_mode=args.redundancy_mode,
                mip_gap=args.mip_gap, time_limit=args.time_limit)

    selected = [factors[j] for j in res["selected_idx"]]
    out = Path(args.output_dir)
    stale = ([p.name for p in out.glob("unified_v2_*")]
             + [p.name for p in out.glob("v2_weights.csv")]) if out.exists() else []
    if stale and not args.overwrite_output:
        raise SystemExit(
            f"{out} already holds {sorted(stale)[:3]}; a failed rerun would leave last "
            "run's weights beside this run's report. Use a fresh --output-dir or pass "
            "--overwrite-output.")
    for name in stale:
        (out / name).unlink()
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "method": "joint_factor_selection",
        "fdr_mode": args.fdr_mode, "fdr_q": fdr_q, "fdr_precheck": fdr_precheck,
        "gate3_mode": args.gate3_mode, "beta_min_frac": args.beta_min_frac,
        "solver": args.solver, "solver_requested": solver_requested,
        "status": res["status"], "solve_seconds": round(res["solve_s"], 3),
        "mip_gap": args.mip_gap, "redundancy_mode": args.redundancy_mode,
        "tau": args.tau, "n_scenarios": args.n_scenarios,
        "time_limit": res["time_limit"], "time_limit_enforced": res["time_limit_enforced"],
        "time_limit_hit": res["time_limit_hit"],
        "outcome": res["outcome"], "incumbent": res["incumbent"],
        "native_status": res.get("native_status"),
        "objective_is_monthly_sharpe": True, "objective": res["objective"],
        "insample_ann_sharpe": round(res["insample_ann_sharpe"], 4),
        "train_months": T, "n_factors": p, "ledoit_wolf_shrink": round(float(shrink), 4),
        "lam_min_Sigma_f": res["lam_min_Sigma_f"], "bigM": res["bigM"],
        "n_domain_eligible": int(elig.sum()), "n_redundancy_pairs": len(forbidden),
        "max_cardinality": args.max_cardinality, "lambda_cost": args.lambda_cost,
        "survival": (not args.no_survival), "kappa": args.kappa,
        "selected_factors": selected,
        "selected_p_values": {factors[j]: float(p_values[j]) for j in res["selected_idx"]},
        "b": {factors[j]: float(res["b"][j]) for j in res["selected_idx"]},
    }
    result["manifest"] = build_manifest(
        panel_csv=args.panel_csv, config_path=args.config,
        extra={"selection_csv": str(args.selection_csv),
               "selection_manifest": sel_manifest,
               "train_end": args.train_end, "seed": args.seed,
               "label_purge": label_purge,
               "cvxpy_version": cp.__version__})
    write_strict_json(out / "unified_v2_solution.json", result)
    if res["outcome"] not in ("OPTIMAL", "FEASIBLE_INACCURATE", "FEASIBLE_LIMIT"):
        print(f"SOLVE FAILED: outcome={res['outcome']} status={res['status']} -- "
              "diagnostic JSON written; exiting non-zero so callers cannot mistake this "
              "for a clean solve")
        sys.exit(2)
    # weights csv for Variant A OOS evaluation
    pd.DataFrame({"factor": selected, "weight": [result["b"][f] for f in selected]}).to_csv(
        out / "v2_weights.csv", index=False)

    print(f"=== Unified v2 (single {args.solver} solve, max-Sharpe objective) ===")
    print(f"status={res['status']}  solve={res['solve_s']:.2f}s  T={T} months  LW shrink={shrink:.3f}")
    print(f"objective (in-sample monthly Sharpe)={res['objective']:.4f}  -> annualized {res['insample_ann_sharpe']:.3f}")
    print(f"factors total={p}, domain-eligible={int(elig.sum())}, fdr-mode={args.fdr_mode}, "
          f"redundancy pairs={len(forbidden)}, survival={'on' if G_s is not None else 'off'}")
    print(f"selected ({len(selected)}): {selected}")
    print(f"b: {result['b']}")
    print(f"Wrote: {out/'unified_v2_solution.json'} and {out/'v2_weights.csv'}")


if __name__ == "__main__":
    main()
