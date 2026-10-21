"""Estimate an FM cutoff on calibration seeds and evaluate it on disjoint seeds.

The cutoff is the empirical quantile of finite false-factor statistics reaching
the split-sample gate. Size and power are conditional on that gate population.
Reject an empty calibration population; do not silently fabricate a threshold.
No universal finite-sample guarantee follows from the estimated quantile.

Run as python -m synthetic.calibrated_tau; --help lists inputs and output paths.
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

from synthetic.stage_a_fpr_power import calibrate, generate_panel
from synthetic.split_sample_incremental import screen
from synthetic.incremental_mechanism import fm_tstats
from synthetic.research_metrics import ThresholdPopulation, write_sample_accounting


def collect_split_tstats(calib, cross_idx, args, seed0, reps):
    """Run the split protocol (screen half1 -> FM test half2) for `reps` panels; return fake/true |t| arrays."""
    factors = calib["factors"]
    population = ThresholdPopulation()
    for r in range(reps):
        rng = np.random.default_rng(seed0 + r)
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
        s3 = screen(
            half1,
            factors,
            args.q_max,
            args.rho,
            args.kappa,
            args.survival_B,
            np.random.default_rng(seed0 + r),
        )
        mt = fm_tstats(half2, factors, s3)
        population.add(truth, s3, mt)
    return population


def main():
    """Estimate an FM cutoff from calibration seeds and evaluate disjoint seeds.

    Require finite false-factor calibration statistics before choosing a cutoff.
    Write conditional/full-pipeline rates with sample accounting and keep the
    calibration population separate from validation evidence.
    """
    ap = argparse.ArgumentParser(
        description="Synthetic-calibrated tau for the incremental gate."
    )
    ap.add_argument(
        "--panel-csv", default=str(ROOT / "data_big_v2" / "a_share_factor_panel.csv")
    )
    ap.add_argument("--config", default=str(ROOT / "a_share_baseline_config.json"))
    ap.add_argument(
        "--output-dir", default=str(ROOT / "outputs" / "research" / "calibrated_tau")
    )
    ap.add_argument("--n-assets", type=int, default=150)
    ap.add_argument("--n-months", type=int, default=150)
    ap.add_argument("--reps-cal", type=int, default=30)
    ap.add_argument("--reps-val", type=int, default=20)
    ap.add_argument("--q-max", type=float, default=0.25)
    ap.add_argument("--kappa", type=float, default=0.6)
    ap.add_argument("--survival-B", type=int, default=40)
    ap.add_argument("--targets", default="0.10,0.0455,0.012")
    ap.add_argument("--strength", type=float, default=0.10)
    ap.add_argument("--seed-cal", type=int, default=7)
    ap.add_argument("--seed-val", type=int, default=1007)
    args = ap.parse_args()
    prepare_run(args, __file__)

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    args.rho = float(cfg["redundancy"]["max_abs_corr"])
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
    targets = [float(x) for x in args.targets.split(",")]
    print(
        f"true: {[factors[j] for j in cross_idx]} (IC={args.strength}); split protocol; "
        f"cal reps={args.reps_cal} (seed {args.seed_cal}) / val reps={args.reps_val} (seed {args.seed_val})"
    )

    cal_population = collect_split_tstats(
        calib, cross_idx, args, args.seed_cal, args.reps_cal
    )
    val_population = collect_split_tstats(
        calib, cross_idx, args, args.seed_val, args.reps_val
    )
    fake_cal, _ = cal_population
    fake_val, true_val = val_population
    print(
        f"calibration fakes reaching gate: {len(fake_cal)} | validation fakes: {len(fake_val)}, "
        f"trues: {len(true_val)}"
    )

    if not len(fake_cal):
        raise ValueError(
            "no finite false-factor statistics reached calibration; cannot estimate tau"
        )
    rows = []
    print(
        f"\n{'target':>7} | {'tau* (calib)':>12} | {'tau (normal)':>12} | "
        f"{'size@tau* (val)':>15} | {'size@normal (val)':>17} | {'CondPower@tau*':>10} | {'CondPower@normal':>12}"
    )
    for tgt in targets:
        tau_star = float(np.quantile(fake_cal, 1 - tgt))
        tau_norm = float(norm.ppf(1 - tgt / 2))
        size_star = float(np.mean(fake_val >= tau_star))
        size_norm = float(np.mean(fake_val >= tau_norm))
        pow_star = float(np.mean(true_val >= tau_star))
        pow_norm = float(np.mean(true_val >= tau_norm))
        rows.append(
            {
                "target_size": tgt,
                "tau_calibrated": round(tau_star, 3),
                "tau_normal": round(tau_norm, 3),
                "size_at_calibrated": round(size_star, 4),
                "size_at_normal": round(size_norm, 4),
                "conditional_power_at_calibrated": round(pow_star, 4),
                "conditional_power_at_normal": round(pow_norm, 4),
            }
        )
        primary_metrics = val_population.at(tau_star)
        normal_metrics = val_population.at(tau_norm)
        rows[-1].update(primary_metrics)
        rows[-1]["full_power_at_calibrated"] = primary_metrics["full_power"]
        rows[-1]["full_power_at_normal"] = normal_metrics["full_power"]
        rows[-1]["full_fpr_at_calibrated"] = primary_metrics["full_fpr"]
        rows[-1]["full_fpr_at_normal"] = normal_metrics["full_fpr"]
        print(
            f"{tgt:>7} | {tau_star:>12.3f} | {tau_norm:>12.3f} | {size_star:>15.4f} | "
            f"{size_norm:>17.4f} | {pow_star:>10.3f} | {pow_norm:>12.3f}"
        )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        out / "calibrated_tau.csv", index=False, encoding="utf-8-sig"
    )
    (out / "calibrated_tau.json").write_text(
        json.dumps(
            {
                "protocol": "split-sample; calibrate quantile on cal seeds, validate on fresh seeds",
                "results": rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    write_sample_accounting(out / "sample_accounting.json", {"calibration": cal_population, "validation": val_population})
    print(f"\nWrote: {out / 'calibrated_tau.csv'}")


if __name__ == "__main__":
    run_experiment(main)
