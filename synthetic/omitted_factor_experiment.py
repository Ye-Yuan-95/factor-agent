"""Measure how removing an injected true factor changes proxy selection.

For each seeded panel, compare the full and reduced candidate libraries and bin
false-factor selections by correlation with the omitted factor. Controls are
screened survivors, so even the full library does not guarantee a complete
conditioning set. Power is measured on the remaining true factors in both arms.

Run as python -m synthetic.omitted_factor_experiment; --help lists inputs and output paths.
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
from synthetic.incremental_mechanism import fm_tstats


def run_stack(panel, flist, q_max, rho, kappa, B, t_thresh, rng):
    """Full gate stack on the candidate list `flist` -> set of selected factor NAMES."""
    p = len(flist)
    ic, mean_ic, pval = per_month_ic(panel, flist)
    q = bh_fdr(pval)
    s1 = [j for j in range(p) if q[j] <= q_max]
    Z = np.nan_to_num(
        panel[flist].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    )
    corr = np.corrcoef(Z, rowvar=False)
    s2 = gate_redundancy(s1, corr, mean_ic, rho)
    s3 = gate_survival(s2, ic, mean_ic, kappa, B, rng)
    mt = fm_tstats(panel, flist, s3)
    return {flist[j] for j in s3 if mt.get(j, 0.0) >= t_thresh}


def main():
    """Compare proxy selection with an injected truth column present or omitted.

    Use matched simulated panels and record full/omitted selection outcomes.
    The control set still depends on earlier screening, so the comparison does
    not assume every true factor survives as a control.
    """
    ap = argparse.ArgumentParser(
        description="Omitted-true-factor boundary of the incremental gate."
    )
    ap.add_argument(
        "--panel-csv", default=str(ROOT / "data_big_v2" / "a_share_factor_panel.csv")
    )
    ap.add_argument("--config", default=str(ROOT / "a_share_baseline_config.json"))
    ap.add_argument(
        "--output-dir", default=str(ROOT / "outputs" / "research" / "omitted_factor")
    )
    ap.add_argument("--n-assets", type=int, default=150)
    ap.add_argument("--n-months", type=int, default=150)
    ap.add_argument("--reps", type=int, default=20)
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
    true_names = {factors[j] for j in cross_idx}
    print(
        f"true factors: {sorted(true_names)} (IC={args.strength}); omit one per rep, cycling"
    )

    recs = []  # one row per (rep, fake factor)
    pow_full, pow_omit = [], []
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
        m = cross_idx[r % len(cross_idx)]  # omitted true factor this rep
        m_name = factors[m]
        Z = np.nan_to_num(
            panel[factors].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        )
        corr = np.corrcoef(Z, rowvar=False)

        sel_full = run_stack(
            panel,
            factors,
            args.q_max,
            rho,
            args.kappa,
            args.survival_B,
            args.t_thresh,
            np.random.default_rng(args.seed + r),
        )
        reduced = [f for k, f in enumerate(factors) if k != m]
        sel_omit = run_stack(
            panel,
            reduced,
            args.q_max,
            rho,
            args.kappa,
            args.survival_B,
            args.t_thresh,
            np.random.default_rng(args.seed + r),
        )

        remaining_true = true_names - {m_name}
        pow_full.append(len(sel_full & remaining_true) / len(remaining_true))
        pow_omit.append(len(sel_omit & remaining_true) / len(remaining_true))
        for j in range(p):
            if truth[j]:
                continue
            recs.append(
                {
                    "corr_omitted": abs(float(corr[j, m])),
                    "sel_full": factors[j] in sel_full,
                    "sel_omit": factors[j] in sel_omit,
                }
            )

    df = pd.DataFrame(recs)
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.01]
    labels = ["[0,0.2)", "[0.2,0.4)", "[0.4,0.6)", "[0.6,0.8)", "[0.8,1]"]
    df["bin"] = pd.cut(df["corr_omitted"], bins=bins, labels=labels, right=False)
    rows = []
    print(
        f"\nfakes tracked: {len(df)} | overall FPR: full={df['sel_full'].mean():.4f}  "
        f"omitted={df['sel_omit'].mean():.4f}"
    )
    print(
        f"{'corr w/ omitted':>16} | {'n':>4} | {'P(sel) full lib':>15} | {'P(sel) omitted':>14}"
    )
    for lab in labels:
        sub = df[df["bin"] == lab]
        if not len(sub):
            continue
        rows.append(
            {
                "corr_bin": lab,
                "n_fake": int(len(sub)),
                "p_selected_full": round(float(sub["sel_full"].mean()), 4),
                "p_selected_omitted": round(float(sub["sel_omit"].mean()), 4),
            }
        )
        print(
            f"{lab:>16} | {len(sub):>4} | {sub['sel_full'].mean():>15.4f} | {sub['sel_omit'].mean():>14.4f}"
        )
    print(
        f"\nPower on remaining 3 true factors: full={np.mean(pow_full):.3f}  omitted={np.mean(pow_omit):.3f}"
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        out / "omitted_factor.csv", index=False, encoding="utf-8-sig"
    )
    (out / "omitted_factor.json").write_text(
        json.dumps(
            {
                "true_factors": sorted(true_names),
                "strength_IC": args.strength,
                "t_thresh": args.t_thresh,
                "overall_fpr_full": round(float(df["sel_full"].mean()), 4),
                "overall_fpr_omitted": round(float(df["sel_omit"].mean()), 4),
                "power_remaining_full": round(float(np.mean(pow_full)), 4),
                "power_remaining_omitted": round(float(np.mean(pow_omit)), 4),
                "by_corr_bin": rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Wrote: {out / 'omitted_factor.csv'}")


if __name__ == "__main__":
    run_experiment(main)
