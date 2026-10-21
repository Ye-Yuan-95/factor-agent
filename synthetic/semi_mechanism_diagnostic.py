"""Compare supplied versus generated panel geometry, dependence and tail moments.

Reports date counts, cross-sectional size, off-diagonal factor correlation,
true-factor R-squared, null IC autocorrelation and return kurtosis. The supplied calibration determines the diagnostic output.
These diagnostics motivate mechanisms but do not identify causal effects.
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

from synthetic.stage_a_fpr_power import calibrate, generate_panel
from synthetic.four_gate_ablation import per_month_ic
from synthetic.semi_synthetic import make_semi_synthetic
from synthetic.research_metrics import ic_sample_accounting


def main():
    """Compare panel geometry and null diagnostics; results are descriptive."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--panel-csv", default=str(ROOT / "data_big_v2/a_share_factor_panel.csv")
    )
    ap.add_argument("--config", default=str(ROOT / "a_share_baseline_config.json"))
    ap.add_argument(
        "--output-dir", default=str(ROOT / "outputs/research/semi_mechanism_diagnostic")
    )
    ap.add_argument("--n-assets", type=int, default=150)
    ap.add_argument("--n-months", type=int, default=150)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    prepare_run(args, __file__)
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    panel = pd.read_csv(args.panel_csv)
    factors = [f for f in cfg["factor_cols"] if f in panel.columns]
    p = len(factors)
    fam = cfg["family_constraints"]["factor_families"]
    calib = calibrate(args.panel_csv, cfg["factor_cols"])
    cross, seen = [], set()
    for j in range(p):
        f = fam.get(factors[j])
        if f not in seen:
            seen.add(f)
            cross.append(j)
        if len(cross) >= 4:
            break
    rng = np.random.default_rng(args.seed)

    print(
        "1) n date-groups: real/semi =",
        panel["date"].nunique(),
        " vs Gen-1 =",
        args.n_months,
    )

    Xr = panel[factors].apply(pd.to_numeric, errors="coerce")
    # CoW (the default in pandas 3) can expose read-only NumPy views. Own a
    # writable copy before changing the diagonal; leave the DataFrame untouched.
    cr = Xr.corr().to_numpy(copy=True)
    np.fill_diagonal(cr, 0)
    g1, _ = generate_panel(rng, calib, args.n_assets, args.n_months, [], [], tails="t")
    cg = g1[factors].corr().to_numpy(copy=True)
    np.fill_diagonal(cg, 0)
    print(
        f"2) mean|offdiag corr|: real={np.nanmean(np.abs(cr[~np.eye(p, dtype=bool)])):.3f}  gen1={np.nanmean(np.abs(cg[~np.eye(p, dtype=bool)])):.3f} ; "
        f"P(|corr|>0.5): real={np.mean(np.abs(cr[~np.eye(p, dtype=bool)]) > 0.5):.3f} gen1={np.mean(np.abs(cg[~np.eye(p, dtype=bool)]) > 0.5):.3f}"
    )

    def r2s(df):
        out = []
        X = df[factors].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        X = np.nan_to_num((X - np.nanmean(X, 0)) / (np.nanstd(X, 0) + 1e-12))
        for j in cross:
            others = [k for k in range(p) if k != j]
            beta, _, _, _ = np.linalg.lstsq(X[:, others], X[:, j], rcond=None)
            resid = X[:, j] - X[:, others] @ beta
            out.append(1 - resid.var() / X[:, j].var())
        return out

    print("3) R2 of true factors on the other factors:")
    print("   real :", [f"{v:.3f}" for v in r2s(panel)])
    print("   gen1 :", [f"{v:.3f}" for v in r2s(g1)])

    semi_null, _ = make_semi_synthetic(panel, factors, [], 0.0, rng)
    ic_s, _, _ = per_month_ic(semi_null, factors)
    g1_null, _ = generate_panel(
        np.random.default_rng(args.seed + 1),
        calib,
        args.n_assets,
        args.n_months,
        [],
        [],
        tails="t",
    )
    ic_g, _, _ = per_month_ic(g1_null, factors)

    def ac1(ic):
        acs = []
        for j in range(ic.shape[1]):
            s = pd.Series(ic[:, j]).dropna()
            if len(s) > 20:
                acs.append(s.autocorr(1))
        return np.nanmean(acs)

    print(
        f"4) null-market IC lag-1 autocorr: semi={ac1(ic_s):.3f}  gen1={ac1(ic_g):.3f}"
    )

    ks = (
        semi_null.groupby("date")["forward_return"]
        .apply(lambda x: pd.to_numeric(x, errors="coerce").kurt())
        .mean()
    )
    kg = (
        g1_null.groupby("date")["forward_return"]
        .apply(lambda x: pd.to_numeric(x, errors="coerce").kurt())
        .mean()
    )
    print(f"5) per-date XS return excess kurt: semi={ks:.1f}  gen1={kg:.1f}")
    stocks_per_date = len(panel) / panel["date"].nunique()
    # Exclude diagonal entries from both the mean correlation and denominator.
    offdiag = ~np.eye(p, dtype=bool)
    result = {
        "schema_version": 2,
        "sample_accounting_semi": ic_sample_accounting(semi_null, factors, ic_s),
        "sample_accounting_gen1": ic_sample_accounting(g1_null, factors, ic_g),
        "real_dates": int(panel["date"].nunique()),
        "gen1_dates": args.n_months,
        "real_stocks_per_date": stocks_per_date,
        "gen1_stocks_per_date": args.n_assets,
        "n_factors": p,
        "true_factors": [factors[j] for j in cross],
        "mean_abs_corr_real": float(np.nanmean(np.abs(cr[offdiag]))),
        "mean_abs_corr_gen1": float(np.nanmean(np.abs(cg[offdiag]))),
        "true_r2_real": r2s(panel),
        "true_r2_gen1": r2s(g1),
        "null_ic_ac1_semi": float(ac1(ic_s)),
        "null_ic_ac1_gen1": float(ac1(ic_g)),
        "xs_kurt_semi": float(ks),
        "xs_kurt_gen1": float(kg),
        "interpretation": "Descriptive diagnostics only; differences do not identify a unique causal explanation.",
    }
    from factor_research.experiment_manifest import write_strict_json

    write_strict_json(Path(args.output_dir) / "semi_mechanism_diagnostic.json", result)
    print(
        f"Average stocks per date: supplied={stocks_per_date:.1f}, Gen-1={args.n_assets}"
    )


if __name__ == "__main__":
    run_experiment(main)
