"""Sensitivity of selection to tails, clustering and panel dimensions.

Vary one axis at a time around the declared center and compare the full stack
with a no-survival ablation. The finite grid yields descriptive observations,
not confidence bounds or guarantees for all DGPs. Pooled kurtosis is measured
on the first null replicate in each cell."""

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

from synthetic.dgp_v2 import calibrate_v2, generate_panel_v2, report_moments
from synthetic.four_gate_ablation import four_gate, metrics
from synthetic.p1_validate import no_survival_stack


def eval_cell(
    calib,
    factors,
    cross,
    rho,
    args,
    *,
    n_assets,
    n_months,
    sigma_h,
    rho_h,
    reps,
    ic=0.06,
):
    """Return FPR(null), FPR/Power(alt_med) for the no-survival ablation and the full four-gate,
    plus the achieved pooled kurtosis, at one DGP setting."""
    p = len(factors)
    out = {
        "noSurv": {"nullFPR": [], "FPR": [], "Pow": []},
        "four": {"nullFPR": [], "FPR": [], "Pow": []},
    }
    kurts = []
    for r in range(reps):
        rng = np.random.default_rng(args.seed + r)
        # null
        pn, tn = generate_panel_v2(
            rng,
            calib,
            n_assets,
            n_months,
            [],
            [],
            sigma_h=sigma_h,
            rho_h=rho_h,
            coverage=calib["coverage"],
        )
        if r == 0:
            kurts.append(report_moments(pn, factors)["pooled_excess_kurt"])
        ns = no_survival_stack(pn, factors, args.q_max, rho, args.t_thresh)
        mns = np.zeros(p, bool)
        mns[ns] = True
        out["noSurv"]["nullFPR"].append(metrics(mns, tn)["FPR"])
        fg = four_gate(
            pn,
            factors,
            args.q_max,
            rho,
            args.kappa,
            args.survival_B,
            args.t_thresh,
            rng,
        )
        mfg = np.zeros(p, bool)
        mfg[fg["+Incremental"]] = True
        out["four"]["nullFPR"].append(metrics(mfg, tn)["FPR"])
        # alt_med
        pa, ta = generate_panel_v2(
            rng,
            calib,
            n_assets,
            n_months,
            cross,
            [ic] * len(cross),
            sigma_h=sigma_h,
            rho_h=rho_h,
            coverage=calib["coverage"],
        )
        ns = no_survival_stack(pa, factors, args.q_max, rho, args.t_thresh)
        mns = np.zeros(p, bool)
        mns[ns] = True
        mm = metrics(mns, ta)
        out["noSurv"]["FPR"].append(mm["FPR"])
        out["noSurv"]["Pow"].append(mm["Power"])
        fg = four_gate(
            pa,
            factors,
            args.q_max,
            rho,
            args.kappa,
            args.survival_B,
            args.t_thresh,
            rng,
        )
        mfg = np.zeros(p, bool)
        mfg[fg["+Incremental"]] = True
        mm = metrics(mfg, ta)
        out["four"]["FPR"].append(mm["FPR"])
        out["four"]["Pow"].append(mm["Power"])
    return {
        "pooled_kurt": round(float(np.mean(kurts)), 1),
        "noSurv_nullFPR": round(float(np.mean(out["noSurv"]["nullFPR"])), 3),
        "noSurv_altFPR": round(float(np.mean(out["noSurv"]["FPR"])), 3),
        "noSurv_altPow": round(float(np.mean(out["noSurv"]["Pow"])), 3),
        "four_altFPR": round(float(np.mean(out["four"]["FPR"])), 3),
        "four_altPow": round(float(np.mean(out["four"]["Pow"])), 3),
    }


def main():
    """Evaluate the fixed tails, volatility and dimension grid using the input calibration.

    Run repeated null/alternative panels for each setting, then write measured
    gate rates and moment diagnostics. The grid is descriptive, not a bound over
    all possible data-generating processes.
    """
    ap = argparse.ArgumentParser(description="DGP sensitivity grid.")
    ap.add_argument(
        "--panel-csv", default=str(ROOT / "data_big_v2" / "a_share_factor_panel.csv")
    )
    ap.add_argument("--config", default=str(ROOT / "a_share_baseline_config.json"))
    ap.add_argument(
        "--output-dir", default=str(ROOT / "outputs" / "research" / "dgp_sensitivity")
    )
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--q-max", type=float, default=0.25)
    ap.add_argument("--kappa", type=float, default=0.6)
    ap.add_argument("--survival-B", type=int, default=40)
    ap.add_argument("--t-thresh", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    prepare_run(args, __file__)

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    rho = float(cfg["redundancy"]["max_abs_corr"])
    fam = cfg["family_constraints"]["factor_families"]
    calib = calibrate_v2(Path(args.panel_csv), cfg["factor_cols"], fam)
    factors = calib["factors"]
    p = len(factors)
    cross, seen = [], set()
    for j in range(p):
        f = fam.get(factors[j])
        if f not in seen:
            seen.add(f)
            cross.append(j)
        if len(cross) >= 4:
            break

    C = dict(n_assets=150, n_months=150, sigma_h=0.55, rho_h=0.35)  # calibrated center
    axes = [
        ("tail sigma_h", "sigma_h", [0.0, 0.3, 0.55]),
        ("volclust rho_h", "rho_h", [0.0, 0.35, 0.7]),
        ("n_months", "n_months", [60, 120, 240]),
        ("n_assets", "n_assets", [80, 150, 300]),
    ]
    rows = []
    print(
        f"{'axis':>16} {'value':>8} | kurt | noSurv: nullFPR altFPR altPow | four: altFPR altPow"
    )
    for axis_name, key, vals in axes:
        for v in vals:
            setting = dict(C)
            setting[key] = v
            res = eval_cell(calib, factors, cross, rho, args, reps=args.reps, **setting)
            rows.append({"axis": axis_name, "value": v, **res})
            print(
                f"{axis_name:>16} {str(v):>8} | {res['pooled_kurt']:>4} | "
                f"{res['noSurv_nullFPR']:>7} {res['noSurv_altFPR']:>6} {res['noSurv_altPow']:>6} | "
                f"{res['four_altFPR']:>6} {res['four_altPow']:>6}"
            )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "dgp_sensitivity.csv", index=False, encoding="utf-8-sig")

    # Summarize the measured grid without assuming monotonicity or equivalence.
    ns_nullfpr = df["noSurv_nullFPR"].max()
    ns_altfpr = df["noSurv_altFPR"].max()
    L = [
        "# DGP sensitivity results\n",
        f"日期 · {args.reps} reps/cell · 中心=calibrated dgp_v2 · 每次只变一个轴 · 主报告=消融栈(FDR>冗余>增量,无 survival)\n",
        "| 轴 | 值 | pooled峰度 | noSurv nullFPR | noSurv altFPR | noSurv altPow | four altFPR | four altPow |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        L.append(
            f"| {r['axis']} | {r['value']} | {r['pooled_kurt']} | {r['noSurv_nullFPR']} | "
            f"{r['noSurv_altFPR']} | {r['noSurv_altPow']} | {r['four_altFPR']} | {r['four_altPow']} |"
        )
    ns_pow_lo = df["noSurv_altPow"].min()
    ns_pow_hi = df["noSurv_altPow"].max()
    surv_gap = float((df["noSurv_altPow"] - df["four_altPow"]).abs().max())
    L += [
        "\n## Observed grid diagnostics\n",
        f"Maximum no-survival null FPR: {ns_nullfpr:.4f}; alternative FPR: {ns_altfpr:.4f}.",
        f"No-survival power range: [{ns_pow_lo:.4f}, {ns_pow_hi:.4f}].",
        f"Maximum absolute power difference from the four-gate arm: {surv_gap:.4f}.",
        "These are observed values, not assertions of monotonicity, equivalence or universal FPR control.",
    ]
    (out / "dgp_sensitivity.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"Wrote {out / 'dgp_sensitivity.csv'} and dgp_sensitivity.md")


if __name__ == "__main__":
    run_experiment(main)
