"""Compare same-sample and split-sample conditional FM threshold exceedances.

Screen on all dates, on the first half, or screen on the first half and test on
the second. Mode names same150/same75 mean full/half for any --n-months.
Reported conditional power excludes true factors that never reached a finite
FM test; it is not whole-pipeline recall. Temporal dependence may persist across
the split, so nominal normal size is a reference, not a guaranteed result.

Run as python -m synthetic.split_sample_incremental; --help lists inputs and output paths.
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
from scipy.stats import norm

from synthetic.stage_a_fpr_power import calibrate, generate_panel, bh_fdr
from synthetic.four_gate_ablation import per_month_ic, gate_redundancy, gate_survival
from synthetic.incremental_mechanism import fm_tstats
from synthetic.research_metrics import ThresholdPopulation, write_sample_accounting


def screen(panel, factors, q_max, rho, kappa, B, rng):
    """FDR -> redundancy -> survival on `panel`; returns surviving indices s3."""
    p = len(factors)
    ic, mean_ic, pval = per_month_ic(panel, factors)
    q = bh_fdr(pval)
    s1 = [j for j in range(p) if q[j] <= q_max]
    Z = np.nan_to_num(
        panel[factors].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    )
    corr = np.corrcoef(Z, rowvar=False)
    s2 = gate_redundancy(s1, corr, mean_ic, rho)
    return gate_survival(s2, ic, mean_ic, kappa, B, rng)


def main():
    """Compare full-sample, half-sample and split-screen/FM populations.

    Use date-based halves for the split arm and preserve both conditional and
    full-pipeline denominators. Save unavailable-statistic counts and per-fit
    sample diagnostics with the threshold results.
    """
    ap = argparse.ArgumentParser(
        description="Sample-splitting vs same-sample incremental gate size."
    )
    ap.add_argument(
        "--panel-csv", default=str(ROOT / "data_big_v2" / "a_share_factor_panel.csv")
    )
    ap.add_argument("--config", default=str(ROOT / "a_share_baseline_config.json"))
    ap.add_argument(
        "--output-dir", default=str(ROOT / "outputs" / "research" / "split_sample")
    )
    ap.add_argument("--n-assets", type=int, default=150)
    ap.add_argument("--n-months", type=int, default=150)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--q-max", type=float, default=0.25)
    ap.add_argument("--kappa", type=float, default=0.6)
    ap.add_argument("--survival-B", type=int, default=40)
    ap.add_argument("--taus", default="1.0,1.5,2.0,2.5,3.0")
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
    taus = [float(x) for x in args.taus.split(",")]
    print(
        f"true: {[factors[j] for j in cross_idx]} (IC={args.strength}); modes: same150 / same75 / split"
    )

    acc = {m: ThresholdPopulation() for m in ("same150", "same75", "split")}
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
        dates = np.sort(panel["date"].unique())
        half1 = panel[panel["date"].isin(dates[: len(dates) // 2])]
        half2 = panel[panel["date"].isin(dates[len(dates) // 2 :])]

        s3_full = screen(
            panel,
            factors,
            args.q_max,
            rho,
            args.kappa,
            args.survival_B,
            np.random.default_rng(args.seed + r),
        )
        s3_half = screen(
            half1,
            factors,
            args.q_max,
            rho,
            args.kappa,
            args.survival_B,
            np.random.default_rng(args.seed + r),
        )

        for mode, s3, test_panel in (
            ("same150", s3_full, panel),
            ("same75", s3_half, half1),
            ("split", s3_half, half2),
        ):
            mt = fm_tstats(test_panel, factors, s3)
            acc[mode].add(truth, s3, mt)

    rows = []
    print(
        f"\n{'tau':>4} | {'theory':>7} | {'same150':>8} | {'same75':>7} | {'split':>7} |  conditional Power(same150/split)"
    )
    for tau in taus:
        row = {"tau": tau, "size_theory": round(2 * (1 - norm.cdf(tau)), 4)}
        for mode in ("same150", "same75", "split"):
            ft, tt = acc[mode]
            metrics = acc[mode].at(tau)
            row.update({f"{key}_{mode}": value for key, value in metrics.items()})
            row[f"size_{mode}"] = (
                round(float(np.mean(ft >= tau)), 4) if len(ft) else float("nan")
            )
            row[f"conditional_power_{mode}"] = (
                round(float(np.mean(tt >= tau)), 4) if len(tt) else float("nan")
            )
        rows.append(row)
        print(
            f"{tau:>4} | {row['size_theory']:>7.4f} | {row['size_same150']:>8.4f} | "
            f"{row['size_same75']:>7.4f} | {row['size_split']:>7.4f} |  "
            f"{row['conditional_power_same150']:.3f} / {row['conditional_power_split']:.3f}"
        )

    n_reach = {m: len(tuple(acc[m])[0]) for m in acc}
    print(f"\nfakes reaching gate: {n_reach}")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        out / "split_sample.csv", index=False, encoding="utf-8-sig"
    )
    write_sample_accounting(out / "sample_accounting.json", acc)
    print(f"Wrote: {out / 'split_sample.csv'}")


if __name__ == "__main__":
    run_experiment(main)
