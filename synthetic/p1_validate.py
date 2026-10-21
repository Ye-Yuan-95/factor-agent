"""Compare sequential selection gates on a calibrated Gen-2 synthetic market.

Generate null and three injected-strength scenarios. CSV and Markdown tables
report measurements from this run. See docs/RESEARCH_EXPERIMENTS.md for assumptions.
Run as python -m synthetic.p1_validate; --help lists inputs and output paths."""

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
from synthetic.four_gate_ablation import (
    four_gate,
    metrics,
    per_month_ic,
    bh_fdr,
    gate_redundancy,
    gate_incremental,
)


def no_survival_stack(panel, factors, q_max, rho, t_thresh):
    """Ablation omitting survival, using the same survivor-conditioned FM test."""
    p = len(factors)
    ic, mean_ic, pval = per_month_ic(panel, factors)
    q = bh_fdr(pval)
    s1 = [j for j in range(p) if q[j] <= q_max]
    Z = np.nan_to_num(
        panel[factors].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    )
    corr = np.corrcoef(Z, rowvar=False)
    s2 = gate_redundancy(s1, corr, mean_ic, rho)
    return gate_incremental(panel, s2, factors, t_thresh)


def main():
    """Compare sequential gate variants on calibrated Gen-2 truth scenarios.

    Run null and alternative replications, save rates and sample accounting,
    and report measured panel moments. The outputs describe these settings
    and do not assert universal FPR or power behavior.
    """
    ap = argparse.ArgumentParser(
        description="Four-gate ablation on calibrated Gen-2 panels."
    )
    ap.add_argument(
        "--panel-csv", default=str(ROOT / "data_big_v2" / "a_share_factor_panel.csv")
    )
    ap.add_argument("--config", default=str(ROOT / "a_share_baseline_config.json"))
    ap.add_argument(
        "--output-dir", default=str(ROOT / "outputs" / "research" / "p1_validate")
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
    calib = calibrate_v2(Path(args.panel_csv), cfg["factor_cols"], fam_map)
    factors = calib["factors"]
    p = len(factors)
    cross, seen = [], set()
    for j in range(p):
        f = fam_map.get(factors[j])
        if f not in seen:
            seen.add(f)
            cross.append(j)
        if len(cross) >= 4:
            break
    # confirm the realism moments once
    mrep = report_moments(
        generate_panel_v2(
            np.random.default_rng(0),
            calib,
            args.n_assets,
            args.n_months,
            [],
            [],
            coverage=calib["coverage"],
        )[0],
        factors,
    )
    print(
        f"dgp_v2 moments: pooled_kurt={mrep['pooled_excess_kurt']:.1f}, "
        f"xs_kurt={mrep['xs_excess_kurt']:.1f}, vol_acf={mrep['vol_acf1']:.2f}, "
        f"std={mrep['ret_std']:.3f}"
    )
    print(
        f"sectors={calib['n_sectors']}, exposure_phi={calib['exposure_phi']:.2f}, coverage={calib['coverage']:.3f}\n"
    )

    gates = ["FDR", "+Redundancy", "+Survival", "+Incremental", "noSurv(FDR>Red>Incr)"]
    configs = [("null", [], 0.0)] + [
        (f"alt_{lbl}", cross, ic)
        for lbl, ic in [("weak", 0.03), ("med", 0.06), ("strong", 0.10)]
    ]
    rows = []
    for name, true_idx, strength in configs:
        agg = {g: {"FPR": [], "Power": [], "n": []} for g in gates}
        for r in range(args.reps):
            rng = np.random.default_rng(args.seed + r)
            panel, truth = generate_panel_v2(
                rng,
                calib,
                args.n_assets,
                args.n_months,
                true_idx,
                [strength] * len(true_idx),
                coverage=calib["coverage"],
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
            sel["noSurv(FDR>Red>Incr)"] = no_survival_stack(
                panel, factors, args.q_max, rho, args.t_thresh
            )
            for g in gates:
                mask = np.zeros(p, dtype=bool)
                mask[sel[g]] = True
                m = metrics(mask, truth)
                agg[g]["FPR"].append(m["FPR"])
                agg[g]["n"].append(m["n_selected"])
                if truth.any():
                    agg[g]["Power"].append(m["Power"])
        for g in gates:
            rows.append(
                {
                    "experiment": name,
                    "strength_IC": strength,
                    "gate": g,
                    "FPR": float(np.mean(agg[g]["FPR"])),
                    "Power": (
                        float(np.mean(agg[g]["Power"])) if agg[g]["Power"] else np.nan
                    ),
                    "n_selected": float(np.mean(agg[g]["n"])),
                }
            )
        f_pow = (
            np.mean(agg["+Survival"]["Power"])
            if agg["+Survival"]["Power"]
            else float("nan")
        )
        ns_fpr = np.mean(agg["noSurv(FDR>Red>Incr)"]["FPR"])
        ns_pow = (
            np.mean(agg["noSurv(FDR>Red>Incr)"]["Power"])
            if agg["noSurv(FDR>Red>Incr)"]["Power"]
            else float("nan")
        )
        print(
            f"[{name:>10}] +Survival Power={f_pow:.2f} || noSurv FPR={ns_fpr:.3f} Power={ns_pow:.2f}"
        )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "p1_four_gate_v2.csv", index=False, encoding="utf-8-sig")

    # Summarize only measurements computed for this run.
    L = [
        "# Gen-2 sequential-gate results\n",
        f"Repetitions: {args.reps}; calibrated heavy tails, sectors, missingness and exposure persistence.\n",
        f"Generated moments: pooled kurtosis {mrep['pooled_excess_kurt']:.1f}; "
        f"cross-sectional kurtosis {mrep['xs_excess_kurt']:.1f}; "
        f"volatility ACF {mrep['vol_acf1']:.2f}; return std {mrep['ret_std']:.3f}; "
        f"sectors {calib['n_sectors']}; exposure phi {calib['exposure_phi']:.2f}.\n",
        "## Sequential-gate comparison\n",
        "| Scenario | Gate | FPR | Whole-pipeline power | Mean selections |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        pw = "-" if pd.isna(r["Power"]) else f"{r['Power']:.3f}"
        L.append(
            f"| {r['experiment']} | {r['gate']} | {r['FPR']:.3f} | {pw} | {r['n_selected']:.1f} |"
        )
    alt = df[df["experiment"].str.startswith("alt")]
    fdr_fpr = alt[alt["gate"] == "FDR"]["FPR"].mean()
    red_pow = alt[alt["gate"] == "+Redundancy"]["Power"].mean()
    surv_pow = alt[alt["gate"] == "+Survival"]["Power"].mean()
    inc_fpr = alt[alt["gate"] == "+Incremental"]["FPR"].mean()
    inc_pow = alt[alt["gate"] == "+Incremental"]["Power"].mean()
    L.append("\n## Descriptive diagnostics (not a preregistered confirmation)\n")
    L.append(
        f"Alternative mean FPR: {fdr_fpr:.4f} -> {inc_fpr:.4f}; power: {inc_pow:.4f}."
    )
    L.append(f"Survival-minus-redundancy mean power: {surv_pow - red_pow:.4f}.")
    L.append(
        "No universal robustness claim follows from this finite calibration-dependent run."
    )
    (out / "p1_validate.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"Wrote {out / 'p1_four_gate_v2.csv'} and p1_validate.md")


if __name__ == "__main__":
    run_experiment(main)
