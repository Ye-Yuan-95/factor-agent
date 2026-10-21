"""Four-gate FPR/Power ablation on synthetic ground-truth markets.

Applies the selection gates SEQUENTIALLY and measures FPR/Power after each, so we can read off
how much each gate reduces false positives:

  Gate 1  FDR           : rank-IC -> normal p-value -> BH-FDR q <= q_max
  Gate 2  + Redundancy  : greedy drop of factors with |corr| > rho to an already-kept factor
  Gate 3  + Survival    : keep only factors whose IC sign survives month bootstrap (>= kappa of draws)
  Gate 4  + Incremental : Fama-MacBeth on the survivors -> keep only factors with |t| >= t_thresh
                          (incremental value: significant after controlling for the other kept factors)

Run on null (all fake -> FPR) and alternative (a few injected-true -> Power), R reps, aggregate.
Reuses the calibrated generator from stage_a_fpr_power.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage_a_fpr_power import calibrate, generate_panel, bh_fdr, metrics

_REPO_ROOT = Path(__file__).resolve().parents[1]
from synthetic.research_runtime import prepare_run, run_experiment  # noqa: E402


def per_month_ic(panel, factors):
    """Return per-date rank IC, its time mean and IID-normal approximate p-values.
    Missing observations are removed pairwise; this estimator does not apply HAC standard errors."""
    ic_rows = []
    for _, grp in panel.groupby("date", sort=True):
        y = pd.to_numeric(grp["forward_return"], errors="coerce").to_numpy()
        row = np.full(len(factors), np.nan)
        for j, f in enumerate(factors):
            x = pd.to_numeric(grp[f], errors="coerce").to_numpy()
            ok = np.isfinite(x) & np.isfinite(y)          # NaN-robust: rank-IC on observed pairs only
            if ok.sum() >= 10:
                xr = pd.Series(x[ok]).rank().to_numpy(); yr = pd.Series(y[ok]).rank().to_numpy()
                if xr.std() > 0 and yr.std() > 0:
                    row[j] = np.corrcoef(xr, yr)[0, 1]
        ic_rows.append(row)
    ic = np.array(ic_rows)
    mean_ic = np.nanmean(ic, 0)
    sd = np.nanstd(ic, 0, ddof=1)
    n = np.sum(np.isfinite(ic), 0)
    t = mean_ic / (sd / np.sqrt(np.maximum(n, 1)) + 1e-12)
    pval = 2 * (1 - norm.cdf(np.abs(t)))
    return ic, mean_ic, pval


def gate_redundancy(kept, corr, mean_ic, rho):
    """Greedily keep the strongest marginal-IC factors below the pairwise correlation cap."""
    order = sorted(kept, key=lambda j: -abs(mean_ic[j]))
    keep = []
    for j in order:
        if all(abs(corr[j, k]) <= rho for k in keep):
            keep.append(j)
    return keep


def gate_survival(kept, ic, mean_ic, kappa, B, rng, floor=0.005):
    """Keep factors with sufficiently frequent same-sign bootstrap means above the IC floor."""
    out = []
    for j in kept:
        col = ic[:, j][np.isfinite(ic[:, j])]
        if len(col) < 5:
            continue
        rs = np.sign(mean_ic[j])
        means = [col[rng.integers(0, len(col), len(col))].mean() for _ in range(B)]
        cnt = sum(1 for m in means if np.sign(m) == rs and abs(m) > floor)
        if cnt / B >= kappa:
            out.append(j)
    return out


def gate_incremental(panel, kept, factors, t_thresh):
    """Filter surviving factors by absolute IID Fama-MacBeth t statistics.
    Controls are the gate survivors, not necessarily all true factors. Fewer than
    five usable cross-sections is insufficient evidence and rejects all survivors."""
    if not kept:
        return []
    sel = [factors[j] for j in kept]
    coefs = []
    for _, grp in panel.groupby("date", sort=True):
        Z = grp[sel].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        y = pd.to_numeric(grp["forward_return"], errors="coerce").to_numpy(float)
        ok = np.isfinite(y) & np.all(np.isfinite(Z), 1)
        if ok.sum() < len(sel) + 2:
            continue
        Z, y = Z[ok], y[ok]
        Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
        try:
            beta = np.linalg.lstsq(np.column_stack([np.ones(len(y)), Z]), y, rcond=None)[0][1:]
        except np.linalg.LinAlgError:
            continue
        coefs.append(beta)
    if len(coefs) < 5:
        return []
    C = np.array(coefs)
    t = C.mean(0) / (C.std(0, ddof=1) / np.sqrt(len(C)) + 1e-12)
    return [kept[i] for i in range(len(kept)) if np.isfinite(t[i]) and abs(t[i]) >= t_thresh]


def four_gate(panel, factors, q_max, rho, kappa, B, t_thresh, rng):
    """Return candidate indices after each sequential gate using one supplied RNG stream."""
    ic, mean_ic, pval = per_month_ic(panel, factors)
    q = bh_fdr(pval)
    s1 = [j for j in range(len(factors)) if q[j] <= q_max]
    Z = np.nan_to_num(panel[factors].apply(pd.to_numeric, errors="coerce").to_numpy(float))
    corr = np.corrcoef(Z, rowvar=False)
    s2 = gate_redundancy(s1, corr, mean_ic, rho)
    s3 = gate_survival(s2, ic, mean_ic, kappa, B, rng)
    s4 = gate_incremental(panel, s3, factors, t_thresh)
    return {"FDR": s1, "+Redundancy": s2, "+Survival": s3, "+Incremental": s4}


def main():
    """Compare sequential gate subsets on calibrated Gen-1 null/alternative panels.

    Aggregate selections against injected truth and write measured FPR/power.
    False factors may correlate with truth; the metrics retain the stated
    injection-based truth definition.
    """
    ap = argparse.ArgumentParser(description="Four-gate FPR/Power ablation on synthetic markets.")
    ap.add_argument("--panel-csv", default=str(_REPO_ROOT / "data" / "a_share_factor_panel.csv"))
    ap.add_argument("--config", default=str(_REPO_ROOT / "a_share_baseline_config.json"))
    ap.add_argument("--output-dir", default=str(_REPO_ROOT / "outputs" / "synthetic" / "four_gate"))
    ap.add_argument("--n-assets", type=int, default=150)
    ap.add_argument("--n-months", type=int, default=150)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--q-max", type=float, default=0.25)
    ap.add_argument("--kappa", type=float, default=0.6)
    ap.add_argument("--survival-B", type=int, default=40)
    ap.add_argument("--t-thresh", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    prepare_run(args, __file__)

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    rho = float(cfg["redundancy"]["max_abs_corr"])
    calib = calibrate(Path(args.panel_csv), cfg["factor_cols"])
    factors = calib["factors"]
    p = len(factors)
    print(f"calibrated: {p} factors, n_assets={args.n_assets}, n_months={args.n_months}, reps={args.reps}")

    gates = ["FDR", "+Redundancy", "+Survival", "+Incremental"]
    # cross-family true factors (4 distinct families) for alternative
    fam_map = cfg["family_constraints"]["factor_families"]
    cross_idx, seen = [], set()
    for j in range(p):
        fam = fam_map.get(factors[j])
        if fam not in seen:
            seen.add(fam); cross_idx.append(j)
        if len(cross_idx) >= 4:
            break

    rows = []
    configs = [("null", [], 0.0)] + [(f"alt_{lbl}", cross_idx, ic) for lbl, ic in
                                     [("weak", 0.03), ("med", 0.06), ("strong", 0.10)]]
    for name, true_idx, strength in configs:
        agg = {g: {"FPR": [], "Power": [], "n": []} for g in gates}
        for r in range(args.reps):
            rng = np.random.default_rng(args.seed + r)
            panel, truth = generate_panel(rng, calib, args.n_assets, args.n_months,
                                          true_idx, [strength] * len(true_idx), tails="t")
            sel = four_gate(panel, factors, args.q_max, rho, args.kappa, args.survival_B, args.t_thresh, rng)
            for g in gates:
                mask = np.zeros(p, dtype=bool); mask[sel[g]] = True
                m = metrics(mask, truth)
                agg[g]["FPR"].append(m["FPR"]); agg[g]["n"].append(m["n_selected"])
                if truth.any():
                    agg[g]["Power"].append(m["Power"])
        for g in gates:
            rows.append({"experiment": name, "strength_IC": strength, "gate": g,
                         "FPR": float(np.mean(agg[g]["FPR"])),
                         "Power": (float(np.mean(agg[g]["Power"])) if agg[g]["Power"] else np.nan),
                         "n_selected": float(np.mean(agg[g]["n"]))})
        line = " | ".join(f"{g}: FPR={np.mean(agg[g]['FPR']):.3f}"
                          f"{('/Pow='+format(np.mean(agg[g]['Power']),'.2f')) if agg[g]['Power'] else ''}"
                          f"(n={np.mean(agg[g]['n']):.1f})" for g in gates)
        print(f"[{name:>11}] {line}")

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "four_gate_ablation.csv", index=False, encoding="utf-8-sig")
    print(f"\nWrote: {out/'four_gate_ablation.csv'}")


if __name__ == "__main__":
    run_experiment(main)
