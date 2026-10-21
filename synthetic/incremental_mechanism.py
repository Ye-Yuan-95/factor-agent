"""Compare marginal and survivor-conditioned FM evidence for correlated proxies.

Track false factors that pass marginal screening and bin their statistics by
correlation with injected truth. Multivariate controls are the surviving set,
not all configured factors. Removal includes earlier gates and unavailable FM
statistics; descriptive associations do not by themselves identify causation.

Run as python -m synthetic.incremental_mechanism; --help lists inputs and output paths.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Support both `python -m synthetic.NAME` and direct-script invocation.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from synthetic.research_runtime import prepare_run, run_experiment

import numpy as np
import pandas as pd

from synthetic.stage_a_fpr_power import calibrate, generate_panel, bh_fdr
from synthetic.four_gate_ablation import per_month_ic, gate_redundancy, gate_survival


class FMStats(dict):
    """Dictionary-compatible FM statistics with per-fit effective sample metadata."""

    def __init__(self, values, diagnostics):
        super().__init__(values)
        self.diagnostics = diagnostics


def fm_tstats(panel, factors, sel_idx):
    """Return absolute IID FM t statistics controlling only for sel_idx factors.

    Fit standardized cross-sections on complete rows. With fewer than five
    usable dates return NaN; callers must not convert unavailable evidence to a pass.
    """
    diagnostics = {
        "n_date_groups": int(panel["date"].nunique()), "n_periods_used": 0,
        "required_periods": 5, "n_complete_rows_used": 0,
        "cross_section_sizes": [], "n_skipped_insufficient_obs": 0,
        "n_skipped_fit_failed": 0, "n_controls": len(sel_idx),
        "status": "no_survivors" if not sel_idx else "insufficient_cross_sections",
        "standard_error": "iid across usable date coefficients",
    }
    if not sel_idx:
        return FMStats({}, diagnostics)
    sel = [factors[j] for j in sel_idx]
    coefs = []
    for _, g in panel.groupby("date", sort=True):
        Z = g[sel].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        y = pd.to_numeric(g["forward_return"], errors="coerce").to_numpy(float)
        ok = np.isfinite(y) & np.all(np.isfinite(Z), 1)
        if ok.sum() < len(sel) + 2:
            diagnostics["n_skipped_insufficient_obs"] += 1
            continue
        Z, y = Z[ok], y[ok]
        Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
        try:
            beta = np.linalg.lstsq(
                np.column_stack([np.ones(len(y)), Z]), y, rcond=None
            )[0][1:]
        except np.linalg.LinAlgError:
            diagnostics["n_skipped_fit_failed"] += 1
            continue
        if not np.isfinite(beta).all():
            diagnostics["n_skipped_fit_failed"] += 1
            continue
        coefs.append(beta)
        diagnostics["cross_section_sizes"].append(int(ok.sum()))
    diagnostics["n_periods_used"] = len(coefs)
    diagnostics["n_complete_rows_used"] = sum(diagnostics["cross_section_sizes"])
    if len(coefs) < 5:
        return FMStats({j: np.nan for j in sel_idx}, diagnostics)
    C = np.array(coefs)
    t = C.mean(0) / (C.std(0, ddof=1) / np.sqrt(len(C)) + 1e-12)
    diagnostics["status"] = "estimated"
    diagnostics["n_finite_statistics"] = int(np.isfinite(t).sum())
    return FMStats({sel_idx[i]: abs(float(t[i])) if np.isfinite(t[i]) else np.nan
                    for i in range(len(sel_idx))}, diagnostics)


def univariate_t(ic):
    """Compute absolute IID t statistics of date-wise IC; unavailable observations are excluded."""
    n = np.sum(np.isfinite(ic), 0)
    mean = np.nanmean(ic, 0)
    sd = np.nanstd(ic, 0, ddof=1)
    return np.abs(mean / (sd / np.sqrt(np.maximum(n, 1)) + 1e-12))


def main():
    """Compare marginal and multivariate FM evidence for correlated proxy factors.

    Build the declared truth scenarios, collect survivor statistics and record
    usable-fit diagnostics. Grouped summaries describe estimable populations
    and do not convert unavailable evidence into a passing statistic.
    """
    ap = argparse.ArgumentParser(
        description="Empirical mechanism of the incremental-value gate."
    )
    ap.add_argument(
        "--panel-csv", default=str(ROOT / "data_big_v2" / "a_share_factor_panel.csv")
    )
    ap.add_argument("--config", default=str(ROOT / "a_share_baseline_config.json"))
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "outputs" / "research" / "incremental_mechanism"),
    )
    ap.add_argument("--n-assets", type=int, default=150)
    ap.add_argument("--n-months", type=int, default=150)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--q-max", type=float, default=0.25)
    ap.add_argument("--kappa", type=float, default=0.6)
    ap.add_argument("--survival-B", type=int, default=40)
    ap.add_argument("--t-thresh", type=float, default=2.0)
    ap.add_argument("--strength", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    prepare_run(args, __file__)

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    rho = float(cfg["redundancy"]["max_abs_corr"])
    fam_map = cfg["family_constraints"]["factor_families"]
    calib = calibrate(Path(args.panel_csv), cfg["factor_cols"])
    factors = calib["factors"]
    p = len(factors)
    cross_idx, seen = [], set()
    for j in range(p):
        fam = fam_map.get(factors[j])
        if fam not in seen:
            seen.add(fam)
            cross_idx.append(j)
        if len(cross_idx) >= 4:
            break
    true_set = set(cross_idx)
    sample_diagnostics = []
    print(
        f"true factors: {[factors[j] for j in cross_idx]} (cross-family, IC={args.strength})"
    )

    # accumulate per FAKE factor that PASSES FDR: (corr-with-truth, univariate |t|, multivariate |t|, killed?)
    recs = []
    fdr_fake, killed_fake = 0, 0
    for r in range(args.reps):
        rng = np.random.default_rng(args.seed + r)
        panel, truth = generate_panel(
            rng,
            calib,
            args.n_assets,
            args.n_months,
            cross_idx,
            [args.strength] * len(cross_idx),
            tails="t",
        )
        ic, mean_ic, pval = per_month_ic(panel, factors)
        uni_t = univariate_t(ic)
        q = bh_fdr(pval)
        s1 = [j for j in range(p) if q[j] <= args.q_max]  # FDR survivors
        Z = np.nan_to_num(
            panel[factors].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        )
        corr = np.corrcoef(Z, rowvar=False)
        s2 = gate_redundancy(s1, corr, mean_ic, rho)
        s3 = gate_survival(s2, ic, mean_ic, args.kappa, args.survival_B, rng)
        mt = fm_tstats(panel, factors, s3)  # multivariate t on survivors
        sample_diagnostics.append(dict(mt.diagnostics))
        for j in s1:
            if j in true_set:
                continue
            fdr_fake += 1
            cmax = max((abs(corr[j, k]) for k in cross_idx), default=0.0)
            survived = j in s3
            mtj = mt.get(j, np.nan) if survived else np.nan
            killed = survived and (np.isnan(mtj) or mtj < args.t_thresh)
            if killed or not survived:
                killed_fake += 1
            recs.append(
                {
                    "corr_truth": cmax,
                    "uni_t": float(uni_t[j]),
                    "mult_t": float(mtj) if survived else np.nan,
                    "survived_to_incr": survived,
                }
            )

    df = pd.DataFrame(
        recs, columns=["corr_truth", "uni_t", "mult_t", "survived_to_incr"]
    )
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.01]
    labels = ["[0,0.2)", "[0.2,0.4)", "[0.4,0.6)", "[0.6,0.8)", "[0.8,1]"]
    df["bin"] = pd.cut(df["corr_truth"], bins=bins, labels=labels, right=False)
    rows = []
    print(
        f"\nFake factors passing FDR: {fdr_fake} | removed by/before incremental: {killed_fake} "
        f"({killed_fake / max(fdr_fake, 1):.1%})"
    )
    print(
        "\n  corr-with-truth | n  | mean univariate|t| (FDR sees) | mean multivariate|t| (incr sees) | killed%"
    )
    for lab in labels:
        sub = df[df["bin"] == lab]
        if not len(sub):
            continue
        surv = sub.dropna(subset=["mult_t"])
        killed_pct = (sub["mult_t"].isna() | (sub["mult_t"] < args.t_thresh)).mean()
        rows.append(
            {
                "corr_truth_bin": lab,
                "n_fake_pass_fdr": int(len(sub)),
                "mean_univariate_t": round(float(sub["uni_t"].mean()), 2),
                "mean_multivariate_t": round(float(surv["mult_t"].mean()), 2)
                if len(surv)
                else None,
                "killed_pct": round(float(killed_pct), 3),
            }
        )
        mm = f"{surv['mult_t'].mean():.2f}" if len(surv) else "  -  "
        print(
            f"   {lab:>10}     | {len(sub):>3} |         {sub['uni_t'].mean():>5.2f}            |"
            f"          {mm:>6}              | {killed_pct:.0%}"
        )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        out / "incremental_mechanism.csv", index=False, encoding="utf-8-sig"
    )
    (out / "incremental_mechanism.json").write_text(
        json.dumps(
            {
                "true_factors": [factors[j] for j in cross_idx],
                "strength_IC": args.strength,
                "fake_passing_fdr": fdr_fake,
                "fake_removed": killed_fake,
                "removed_fraction": killed_fake / max(fdr_fake, 1),
                "by_corr_bin": rows,
                "fm_sample_diagnostics": sample_diagnostics,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote: {out / 'incremental_mechanism.csv'}")


if __name__ == "__main__":
    run_experiment(main)
