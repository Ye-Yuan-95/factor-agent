"""Compare conditional FM test size and full-pipeline recovery over t thresholds.

Generate paired panels and compute each surviving factor statistic only once.
Normal tail probabilities are reference sizes; same-sample screening can inflate
them. The power field divides by all injected true factors, while the separate
conditional_power field divides only by true factors with finite gate statistics.

Run as python -m synthetic.tau_size_sweep; --help lists inputs and output paths.
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


def main():
    """Sweep incremental FM thresholds using one recorded survivor population.

    Track screened-out, finite-statistic and selected factors separately so
    conditional size/power and full-pipeline recovery use explicit denominators.
    Write fit diagnostics beside the measured threshold table.
    """
    ap = argparse.ArgumentParser(
        description="tau sweep: incremental-gate size vs 2(1-Phi(tau))."
    )
    ap.add_argument(
        "--panel-csv", default=str(ROOT / "data_big_v2" / "a_share_factor_panel.csv")
    )
    ap.add_argument("--config", default=str(ROOT / "a_share_baseline_config.json"))
    ap.add_argument(
        "--output-dir", default=str(ROOT / "outputs" / "research" / "tau_size_sweep")
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
        f"true factors: {[factors[j] for j in cross_idx]} (IC={args.strength}); taus={taus}"
    )

    population = ThresholdPopulation()
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
        ic, mean_ic, pval = per_month_ic(panel, factors)
        q = bh_fdr(pval)
        s1 = [j for j in range(p) if q[j] <= args.q_max]
        Z = np.nan_to_num(
            panel[factors].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        )
        corr = np.corrcoef(Z, rowvar=False)
        s2 = gate_redundancy(s1, corr, mean_ic, rho)
        s3 = gate_survival(s2, ic, mean_ic, args.kappa, args.survival_B, rng)
        mt = fm_tstats(
            panel, factors, s3
        )  # multivariate |t| for gate survivors, tau-independent
        population.add(truth, s3, mt)

    fake_t, true_t = population
    n_fake_total, n_true_total = population.n_fake_total, population.n_true_total
    rows = []
    print(
        f"\nfakes reaching gate: {len(fake_t)} (of {n_fake_total} total) | trues reaching: {len(true_t)}"
    )
    print(
        f"{'tau':>5} | {'cond. size (obs)':>16} | {'theory 2(1-Phi)':>15} | {'uncond FPR':>10} | {'Power':>6}"
    )
    for tau in taus:
        metrics = population.at(tau)
        size_obs = float(np.mean(fake_t >= tau)) if len(fake_t) else float("nan")
        size_theory = 2 * (1 - norm.cdf(tau))
        fpr_uncond = float((fake_t >= tau).sum() / max(n_fake_total, 1))
        # Full-pipeline power includes true factors rejected before this gate.
        power = (
            float((true_t >= tau).sum() / n_true_total)
            if n_true_total
            else float("nan")
        )
        conditional_power = (
            float(np.mean(true_t >= tau)) if len(true_t) else float("nan")
        )
        rows.append(
            {
                "tau": tau,
                "size_observed": round(size_obs, 4),
                "size_theory": round(size_theory, 4),
                "fpr_unconditional": round(fpr_uncond, 4),
                "power": round(power, 4),
                "n_fake_reaching": len(population.fake),
                "n_true_total": n_true_total,
                "conditional_power": conditional_power,
            }
        )
        rows[-1].update(metrics)
        print(
            f"{tau:>5} | {size_obs:>16.4f} | {size_theory:>15.4f} | {fpr_uncond:>10.4f} | {power:>6.3f}"
        )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        out / "tau_size_sweep.csv", index=False, encoding="utf-8-sig"
    )
    write_sample_accounting(out / "sample_accounting.json", {"sweep": population})
    print(f"\nWrote: {out / 'tau_size_sweep.csv'}")


if __name__ == "__main__":
    run_experiment(main)
