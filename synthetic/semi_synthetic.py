"""Semi-synthetic experiment: retain supplied exposures and permute returns by date.

Permutation preserves the observed return multiset before injection and breaks
systematic alignment in expectation; a finite draw can retain random correlation.
Premium coefficients are target_ic * pooled_return_std, NOT calibrated realized
rank ICs. Truth denotes nonzero injected coefficients, not absence of marginal
predictability. With dummy input this is a workflow check, not real-market evidence.
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

from synthetic.four_gate_ablation import four_gate, metrics
from synthetic.p1_validate import no_survival_stack


def make_semi_synthetic(real_panel, factors, true_idx, target_ic, rng):
    """Return a panel with real exposures + within-date-permuted returns + injected premium on true_idx."""
    df = real_panel[["date", "asset", "forward_return"] + factors].copy()
    for c in ["forward_return"] + factors:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    s_r = df["forward_return"].std()
    parts = []
    for _, g in df.groupby("date", sort=True):
        g = g.copy()
        y = g["forward_return"].to_numpy()
        obs = np.isfinite(y)
        yp = y.copy()
        idx_obs = np.where(obs)[0]
        perm = rng.permutation(idx_obs)  # randomize alignment among observed stocks
        yp[idx_obs] = y[perm]
        # per-date standardized real exposures -> injected premium on the true factors
        prem = np.zeros(len(g))
        for j in true_idx:
            x = g[factors[j]].to_numpy()
            mu, sd = np.nanmean(x), np.nanstd(x)
            xs = (x - mu) / sd if sd > 0 else np.zeros_like(x)
            xs = np.nan_to_num(xs)
            prem += target_ic * s_r * xs
        g["forward_return"] = yp + prem
        parts.append(g)
    out = pd.concat(parts, ignore_index=True)
    truth = np.zeros(len(factors), dtype=bool)
    truth[list(true_idx)] = True
    return out, truth


def main():
    """Retain supplied exposures and compare gates on permuted/injected returns.

    Permutation is within each date; injected coefficients identify truth.
    Save measured gate rates and usable IC counts for the actual input geometry,
    without treating dummy exposures as historical market evidence.
    """
    ap = argparse.ArgumentParser(
        description="5.2 semi-synthetic (real exposures + controlled returns)."
    )
    ap.add_argument(
        "--panel-csv", default=str(ROOT / "data_big_v2" / "a_share_factor_panel.csv")
    )
    ap.add_argument("--config", default=str(ROOT / "a_share_baseline_config.json"))
    ap.add_argument(
        "--output-dir", default=str(ROOT / "outputs" / "research" / "semi_synthetic")
    )
    ap.add_argument("--reps", type=int, default=15)
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
    panel = pd.read_csv(args.panel_csv)
    factors = [f for f in cfg["factor_cols"] if f in panel.columns]
    p = len(factors)
    cross, seen = [], set()
    for j in range(p):
        f = fam.get(factors[j])
        if f not in seen:
            seen.add(f)
            cross.append(j)
        if len(cross) >= 4:
            break
    print(
        f"semi-synthetic: real exposures ({panel['asset'].nunique()} assets, {panel['date'].nunique()} dates), "
        f"true={[factors[j] for j in cross]}"
    )

    gates = ["FDR", "+Redundancy", "+Survival", "+Incremental", "noSurv(FDR>Red>Incr)"]
    rows = []
    for name, true_idx, ic in [
        ("null", [], 0.0),
        ("alt_weak", cross, 0.03),
        ("alt_med", cross, 0.06),
        ("alt_strong", cross, 0.10),
    ]:
        agg = {g: {"FPR": [], "Power": [], "n": []} for g in gates}
        for r in range(args.reps):
            rng = np.random.default_rng(args.seed + r)
            sp, truth = make_semi_synthetic(panel, factors, true_idx, ic, rng)
            sel = four_gate(
                sp,
                factors,
                args.q_max,
                rho,
                args.kappa,
                args.survival_B,
                args.t_thresh,
                rng,
            )
            sel["noSurv(FDR>Red>Incr)"] = no_survival_stack(
                sp, factors, args.q_max, rho, args.t_thresh
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
                    "strength_IC": ic,
                    "gate": g,
                    "FPR": float(np.mean(agg[g]["FPR"])),
                    "Power": (
                        float(np.mean(agg[g]["Power"])) if agg[g]["Power"] else np.nan
                    ),
                    "n_selected": float(np.mean(agg[g]["n"])),
                }
            )
        inc = [
            r for r in rows if r["experiment"] == name and r["gate"] == "+Incremental"
        ][0]
        fdr = [r for r in rows if r["experiment"] == name and r["gate"] == "FDR"][0]
        print(
            f"[{name:>10}] FDR FPR={fdr['FPR']:.3f} -> +Incr FPR={inc['FPR']:.3f} "
            f"Power={inc['Power'] if not np.isnan(inc['Power']) else float('nan'):.2f} (n={inc['n_selected']:.1f})"
        )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "semi_synthetic.csv", index=False, encoding="utf-8-sig")
    L = [
        "# Semi-synthetic: supplied exposures + controlled returns\n",
        f"日期 · {args.reps} reps · 保留所提供面板的因子暴露 · 收益 = 按日期置换(注入前保持截面分布)+ 注入 premium\n",
        "| 市场 | 关卡 | FPR | Power | n |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        pw = "-" if np.isnan(r["Power"]) else f"{r['Power']:.3f}"
        L.append(
            f"| {r['experiment']} | {r['gate']} | {r['FPR']:.3f} | {pw} | {r['n_selected']:.1f} |"
        )
    alt = df[df["experiment"].str.startswith("alt")]
    fdr_fpr = alt[alt["gate"] == "FDR"]["FPR"].mean()
    inc_fpr = alt[alt["gate"] == "+Incremental"]["FPR"].mean()
    inc_pow = alt[alt["gate"] == "+Incremental"]["Power"].mean()
    L.append("\n## Descriptive measurements\n")
    L.append(
        f"**所提供暴露**下,alternative 平均:FDR-only FPR={fdr_fpr:.3f} → +增量 FPR={inc_fpr:.3f}(Power={inc_pow:.2f})。"
    )
    L.append(
        "This is a calibration-dependent diagnostic; permutation and linear injection do not reproduce all market mechanisms."
    )
    (out / "semi_synthetic.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"Wrote {out / 'semi_synthetic.csv'} and semi_synthetic.md")


if __name__ == "__main__":
    run_experiment(main)
