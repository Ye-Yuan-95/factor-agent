"""Sweep injected strength, truth families and noise tails for sequential selection.

Compare marginal screening with the final gate on paired panels. Different
families are not assumed uncorrelated. The output records actual truth counts
when fewer than eight distinct families are available.

Run as python -m synthetic.incremental_robustness; --help lists inputs and output paths.
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
from synthetic.four_gate_ablation import four_gate, metrics


def main():
    """Sweep injected strengths, truth-family layouts and return-tail settings.

    Compare selection before/after the incremental gate and save rates for the
    actual supplied calibration. Use the run seed and recorded arguments to
    identify each simulation population.
    """
    ap = argparse.ArgumentParser(
        description="Incremental-gate decisiveness across market settings."
    )
    ap.add_argument(
        "--panel-csv", default=str(ROOT / "data_big_v2" / "a_share_factor_panel.csv")
    )
    ap.add_argument("--config", default=str(ROOT / "a_share_baseline_config.json"))
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "outputs" / "research" / "incremental_robustness"),
    )
    ap.add_argument("--n-assets", type=int, default=150)
    ap.add_argument("--n-months", type=int, default=150)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--q-max", type=float, default=0.25)
    ap.add_argument("--kappa", type=float, default=0.6)
    ap.add_argument("--survival-B", type=int, default=40)
    ap.add_argument("--t-thresh", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    prepare_run(args, __file__)

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    rho = float(cfg["redundancy"]["max_abs_corr"])
    fam_map = cfg["family_constraints"]["factor_families"]
    calib = calibrate(Path(args.panel_csv), cfg["factor_cols"])
    factors = calib["factors"]
    p = len(factors)

    cross, seen = [], set()
    for j in range(p):
        f = fam_map.get(factors[j])
        if f not in seen:
            seen.add(f)
            cross.append(j)
    momentum = [j for j in range(p) if fam_map.get(factors[j]) == "momentum"]

    # (label, true_idx, strength, tails)
    settings = [
        ("cross4 IC=0.03 (weak)", cross[:4], 0.03, "t"),
        ("cross4 IC=0.06 (med)", cross[:4], 0.06, "t"),
        ("cross4 IC=0.10 (strong)", cross[:4], 0.10, "t"),
        ("cross1 IC=0.10", cross[:1], 0.10, "t"),
        (f"cross{len(cross[:8])} IC=0.06 (up to 8)", cross[:8], 0.06, "t"),
        ("SAME-family mom x4 IC=0.10", momentum[:4], 0.10, "t"),
        ("SAME-family mom x4 IC=0.06", momentum[:4], 0.06, "t"),
        ("cross4 IC=0.06 GAUSSIAN", cross[:4], 0.06, "gaussian"),
    ]

    rows = []
    print(f"{'setting':>28} | FPR FDR -> +Incr (drop) | Power FDR -> +Incr")
    for label, true_idx, ic, tails in settings:
        acc = {"fpr_fdr": [], "fpr_inc": [], "pow_fdr": [], "pow_inc": []}
        for r in range(args.reps):
            rng = np.random.default_rng(args.seed + r)
            panel, truth = generate_panel(
                rng,
                calib,
                args.n_assets,
                args.n_months,
                true_idx,
                [ic] * len(true_idx),
                tails=tails,
            )
            sel = four_gate(
                panel,
                factors,
                args.q_max,
                rho,
                args.kappa,
                args.survival_B,
                args.t_thresh,
                rng,
            )
            for stage, key in (("FDR", "fdr"), ("+Incremental", "inc")):
                mask = np.zeros(p, dtype=bool)
                mask[sel[stage]] = True
                m = metrics(mask, truth)
                acc[f"fpr_{key}"].append(m["FPR"])
                if truth.any():
                    acc[f"pow_{key}"].append(m["Power"])
        fpr0, fpr1 = np.mean(acc["fpr_fdr"]), np.mean(acc["fpr_inc"])
        pow0 = np.mean(acc["pow_fdr"]) if acc["pow_fdr"] else float("nan")
        pow1 = np.mean(acc["pow_inc"]) if acc["pow_inc"] else float("nan")
        drop = fpr0 / fpr1 if fpr1 > 1e-9 else float("inf")
        rows.append(
            {
                "setting": label,
                "n_true": len(true_idx),
                "true_factors": ",".join(factors[j] for j in true_idx),
                "FPR_FDR": round(float(fpr0), 3),
                "FPR_incr": round(float(fpr1), 3),
                "FPR_drop_x": (round(float(drop), 1) if np.isfinite(drop) else None),
                "Power_FDR": round(float(pow0), 3),
                "Power_incr": round(float(pow1), 3),
                "Power_loss": round(float(pow0 - pow1), 3),
            }
        )
        dx = f"{drop:.0f}x" if np.isfinite(drop) else ">inf"
        print(
            f"{label:>28} | {fpr0:.3f} -> {fpr1:.3f} ({dx:>4}) | {pow0:.2f} -> {pow1:.2f} "
            f"(loss {pow0 - pow1:+.2f})"
        )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        out / "incremental_robustness.csv", index=False, encoding="utf-8-sig"
    )
    print(f"\nWrote: {out / 'incremental_robustness.csv'}")


if __name__ == "__main__":
    run_experiment(main)
