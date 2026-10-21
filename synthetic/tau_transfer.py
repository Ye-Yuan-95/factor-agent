"""Test transfer of one calibrated FM threshold to other synthetic settings.

Calibrate once on the baseline, then use disjoint evaluation seeds for different
strengths, truth families and Gen-2 structure. Use the same evaluation seeds across
settings for comparability. Size and power are conditional on finite gate-reaching
statistics; the output is not whole-pipeline recall or a universal size guarantee.

Run as python -m synthetic.tau_transfer; --help lists inputs and output paths.
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
from synthetic.dgp_v2 import calibrate_v2, generate_panel_v2
from synthetic.split_sample_incremental import screen
from synthetic.incremental_mechanism import fm_tstats
from synthetic.research_metrics import ThresholdPopulation, write_sample_accounting


def collect(gen, factors, args, seed0, reps):
    """Split protocol over `reps` panels from generator `gen(rng)`; returns fake/true FM |t| arrays."""
    population = ThresholdPopulation()
    for r in range(reps):
        rng = np.random.default_rng(seed0 + r)
        panel, truth = gen(rng)
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
    """Calibrate one FM cutoff and evaluate its transfer to other settings.

    Calibration and validation seed ranges must be disjoint. Transfer reports
    retain conditional/full-pipeline rates and missing-evidence accounting;
    a calibrated cutoff is not a universal finite-sample guarantee.
    """
    ap = argparse.ArgumentParser(
        description="Does the calibrated tau* transfer across settings/DGPs?"
    )
    ap.add_argument(
        "--panel-csv", default=str(ROOT / "data_big_v2" / "a_share_factor_panel.csv")
    )
    ap.add_argument("--config", default=str(ROOT / "a_share_baseline_config.json"))
    ap.add_argument(
        "--output-dir", default=str(ROOT / "outputs" / "research" / "tau_transfer")
    )
    ap.add_argument("--n-assets", type=int, default=150)
    ap.add_argument("--n-months", type=int, default=150)
    ap.add_argument("--reps-cal", type=int, default=30)
    ap.add_argument("--reps-val", type=int, default=15)
    ap.add_argument("--q-max", type=float, default=0.25)
    ap.add_argument("--kappa", type=float, default=0.6)
    ap.add_argument("--survival-B", type=int, default=40)
    ap.add_argument("--target", type=float, default=0.0455)
    ap.add_argument("--seed-cal", type=int, default=7)
    ap.add_argument("--seed-val", type=int, default=1007)
    args = ap.parse_args()
    prepare_run(args, __file__)

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    args.rho = float(cfg["redundancy"]["max_abs_corr"])
    fam_map = cfg["family_constraints"]["factor_families"]
    calib = calibrate(Path(args.panel_csv), cfg["factor_cols"])
    calib_v2 = calibrate_v2(Path(args.panel_csv), cfg["factor_cols"], fam_map)
    factors = calib["factors"]
    p = len(factors)
    cross, seen = [], set()
    for j in range(p):
        f = fam_map.get(factors[j])
        if f not in seen:
            seen.add(f)
            cross.append(j)
    momentum = [j for j in range(p) if fam_map.get(factors[j]) == "momentum"]

    def gen_thin(true_idx, ic):
        return lambda rng: generate_panel(
            rng,
            calib,
            args.n_assets,
            args.n_months,
            true_idx,
            [ic] * len(true_idx),
            tails="t",
        )

    def gen_v2(true_idx, ic):
        return lambda rng: generate_panel_v2(
            rng,
            calib_v2,
            args.n_assets,
            args.n_months,
            true_idx,
            [ic] * len(true_idx),
            coverage=calib_v2["coverage"],
        )

    # ---- calibrate ONCE on the baseline ----
    cal_population = collect(
        gen_thin(cross[:4], 0.10), factors, args, args.seed_cal, args.reps_cal
    )
    fake_cal, _ = cal_population
    if not len(fake_cal):
        raise ValueError(
            "no finite false-factor statistics reached calibration; cannot estimate tau"
        )
    tau_star = float(np.quantile(fake_cal, 1 - args.target))
    tau_norm = float(norm.ppf(1 - args.target / 2))
    print(
        f"baseline calibration (cross4 IC=0.10, thin-tail): tau* = {tau_star:.3f} "
        f"(normal quantile = {tau_norm:.3f}, target size = {args.target})"
    )

    # ---- transfer targets (fresh seeds) ----
    settings = [
        ("baseline fresh (cross4 IC=0.10)", gen_thin(cross[:4], 0.10)),
        ("weaker signal (cross4 IC=0.06)", gen_thin(cross[:4], 0.06)),
        ("weak signal (cross4 IC=0.03)", gen_thin(cross[:4], 0.03)),
        ("same-family mom x4 IC=0.10", gen_thin(momentum[:4], 0.10)),
        ("single true (cross1 IC=0.10)", gen_thin(cross[:1], 0.10)),
        ("dgp_v2 realistic (cross4 IC=0.10)", gen_v2(cross[:4], 0.10)),
    ]
    populations = {"calibration": cal_population}
    rows = []
    print(
        f"\n{'setting':>34} | {'n_fake':>6} | {'size@tau*':>9} | {'size@normal':>8} | "
        f"{'target':>6} | {'CondPower@tau*':>10}"
    )
    for name, gen in settings:
        population = collect(gen, factors, args, args.seed_val, args.reps_val)
        populations[name] = population
        fake_v, true_v = population
        s_star = float(np.mean(fake_v >= tau_star)) if len(fake_v) else float("nan")
        s_norm = float(np.mean(fake_v >= tau_norm)) if len(fake_v) else float("nan")
        pw = float(np.mean(true_v >= tau_star)) if len(true_v) else float("nan")
        rows.append(
            {
                "setting": name,
                "n_fake_reaching": int(len(fake_v)),
                "size_at_tau_star": round(s_star, 4),
                "size_at_normal": round(s_norm, 4),
                "conditional_power_at_tau_star": round(pw, 4),
            }
        )
        rows[-1].update(population.at(tau_star))
        rows[-1]["full_power_at_tau_star"] = population.at(tau_star)["full_power"]
        rows[-1]["full_fpr_at_normal"] = population.at(tau_norm)["full_fpr"]
        print(
            f"{name:>34} | {len(fake_v):>6} | {s_star:>9.4f} | {s_norm:>8.4f} | "
            f"{args.target:>6} | {pw:>10.3f}"
        )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        out / "tau_transfer.csv", index=False, encoding="utf-8-sig"
    )
    (out / "tau_transfer.json").write_text(
        json.dumps(
            {
                "tau_star": round(tau_star, 4),
                "tau_normal": round(tau_norm, 4),
                "target": args.target,
                "calibrated_on": "thin-tail cross4 IC=0.10 split protocol",
                "results": rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    write_sample_accounting(out / "sample_accounting.json", populations)
    print(f"\nWrote: {out / 'tau_transfer.csv'}")


if __name__ == "__main__":
    run_experiment(main)
