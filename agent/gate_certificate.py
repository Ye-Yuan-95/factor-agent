"""Estimate a source-bound process certificate under Gen-1 and Gen-2 markets.

The base-column selector is the sequential four-gate selector plus sparse
weights; it is not the complete adaptive DSL search process. The certificate
compares conservative cross-DGP null-FPR and alternative-power bounds at a stated
signal strength. It does not establish alternative-panel FDR or profitability.

Certification binds panel, configuration and listed source bytes. Any relevant
change, including a docstring change in a bound source, requires a new matching
certificate. Missing or mismatched certificates leave otherwise passing real-data
candidates as DRAFT; factor cards can still be written. Synthetic in-memory runs
are not certifiable. Semi-synthetic research is separate from this routine loop.

Run: python -m agent.gate_certificate --panel-csv PATH --help
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_DIR = Path(__file__).resolve().parents[1]
for p in (str(_PROJECT_DIR), str(_PROJECT_DIR / "synthetic")):
    if p not in sys.path:
        sys.path.insert(0, p)

from factor_research.factor_validity_pipeline import load_config, fit_sparse_linear_weights  # noqa: E402
from stage_a_fpr_power import calibrate, generate_panel, metrics  # noqa: E402
from four_gate_ablation import four_gate  # noqa: E402
from dgp_v2 import calibrate_v2, generate_panel_v2  # noqa: E402


def five_gate_select(panel: pd.DataFrame, factors: list[str], *, q_max: float, rho: float,
                     kappa: float, survival_b: int, t_thresh: float, pipe_cfg, rng) -> np.ndarray:
    """Four ablation gates + gate 5 (sparse L1/L2 weight non-zero). Returns a boolean mask."""
    sel = four_gate(panel, factors, q_max, rho, kappa, survival_b, t_thresh, rng)
    kept4 = sel["+Incremental"]
    mask = np.zeros(len(factors), dtype=bool)
    if not kept4:
        return mask
    names = [factors[j] for j in kept4]
    weights = fit_sparse_linear_weights(panel.dropna(subset=["forward_return"]), names, pipe_cfg)
    chosen = set(weights.loc[weights["selected_by_weight"], "factor"].astype(str))
    for j in kept4:
        mask[j] = factors[j] in chosen
    return mask


def run_dgp(gen_name: str, generator, calib, factors, reps: int, *, n_assets: int, n_months: int,
            true_idx: list[int], alphas: list[float], gate_args: dict, seed: int) -> dict:
    """Monte-Carlo FPR (null) and Power (alternative at each alpha) for one DGP."""
    out: dict = {"reps": reps}
    for label, ic in [("null", 0.0)] + [(f"alt_ic{ic:g}", ic) for ic in alphas]:
        idx, strengths = ([], []) if ic == 0.0 else (true_idx, [ic] * len(true_idx))
        fprs, powers, nsel = [], [], []
        for r in range(reps):
            rng = np.random.default_rng(seed + r)
            panel, truth = generator(rng, calib, n_assets, n_months, idx, strengths)
            mask = five_gate_select(panel, factors, rng=rng, **gate_args)
            m = metrics(mask, truth)
            fprs.append(m["FPR"]); nsel.append(m["n_selected"])
            if truth.any():
                powers.append(m["Power"])
        entry = {"FPR": float(np.mean(fprs)), "FPR_sd": float(np.std(fprs, ddof=1)) if reps > 1 else 0.0,
                 "n_selected": float(np.mean(nsel)), "FPR_reps": [round(v, 5) for v in fprs]}
        if powers:
            entry["Power"] = float(np.mean(powers))
            entry["Power_reps"] = [round(v, 5) for v in powers]
        out[label] = entry
        pw = f"  Power={entry.get('Power', float('nan')):.3f}" if powers else ""
        print(f"  [{gen_name}:{label:>9}] FPR={entry['FPR']:.3f}{pw}  avg#sel={entry['n_selected']:.1f}")
    return out


def build_certificate(*, panel_csv: str, config_path: str, gen1_reps: int, gen2_reps: int,
                      n_assets: int, n_months: int, alpha_ic: float, extra_ic: float,
                      seed: int, output_path: Path, target_fpr: float = 0.1,
                      target_power: float = 0.3, conf: float = 0.95) -> dict:
    """Estimate process FPR and power under configured Gen-1/Gen-2 experiments.
    Bind the result to panel, configuration and gate-source hashes; this is not a factor profitability certificate."""
    cfg_raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
    pipe_cfg = load_config(config_path)
    factors_cfg = cfg_raw["factor_cols"]
    fam_map = cfg_raw["family_constraints"]["factor_families"]

    gate_args = {
        "q_max": pipe_cfg.q_value_max,
        "rho": pipe_cfg.max_abs_corr,
        "kappa": 0.6,
        "survival_b": 40,
        "t_thresh": pipe_cfg.min_incremental_tstat,
        "pipe_cfg": pipe_cfg,
    }

    calib1 = calibrate(Path(panel_csv), factors_cfg)
    factors = calib1["factors"]
    # cross-family true set (4 distinct families), same convention as the ablation studies
    cross_idx, seen = [], set()
    for j, f in enumerate(factors):
        fam = fam_map.get(f)
        if fam not in seen:
            seen.add(fam); cross_idx.append(j)
        if len(cross_idx) >= 4:
            break
    alphas = sorted({alpha_ic, extra_ic})
    print(f"certificate: {len(factors)} factors, true(cross-family)={[factors[j] for j in cross_idx]}, "
          f"alphas={alphas}, Gen-1 reps={gen1_reps}, Gen-2 reps={gen2_reps} (Gen-3 excluded)")

    def gen1(rng, calib, n_a, n_m, idx, strengths):
        return generate_panel(rng, calib, n_a, n_m, idx, strengths, tails="t")

    res_gen1 = run_dgp("Gen-1", gen1, calib1, factors, gen1_reps,
                       n_assets=n_assets, n_months=n_months, true_idx=cross_idx,
                       alphas=alphas, gate_args=gate_args, seed=seed)

    calib2 = calibrate_v2(panel_csv, factors_cfg, fam_map)

    def gen2(rng, calib, n_a, n_m, idx, strengths):
        return generate_panel_v2(rng, calib, n_a, n_m, idx, strengths,
                                 coverage=calib.get("coverage", 0.973))

    res_gen2 = run_dgp("Gen-2", gen2, calib2, factors, gen2_reps,
                       n_assets=n_assets, n_months=n_months, true_idx=cross_idx,
                       alphas=alphas, gate_args=gate_args, seed=seed + 10_000)

    alpha_key = f"alt_ic{alpha_ic:g}"
    gate_fpr = max(res_gen1["null"]["FPR"], res_gen2["null"]["FPR"])
    gate_power = min(res_gen1[alpha_key].get("Power", float("nan")),
                     res_gen2[alpha_key].get("Power", float("nan")))
    # acceptance binds FPR, Power AND uncertainty (conservative across DGPs)
    from agent.certification import compute_binding, content_hash, one_sided_lower, one_sided_upper
    from agent.validator_oracle import ValidatorOracle, _resolve_path
    fpr_ucbs = [v for v in (one_sided_upper(res_gen1["null"]["FPR_reps"], conf),
                            one_sided_upper(res_gen2["null"]["FPR_reps"], conf)) if v is not None]
    power_lcbs = [v for v in (one_sided_lower(res_gen1[alpha_key]["Power_reps"], conf),
                              one_sided_lower(res_gen2[alpha_key]["Power_reps"], conf)) if v is not None]
    fpr_ucb = max(fpr_ucbs) if fpr_ucbs else None          # a legitimate 0.0 bound must survive
    power_lcb = min(power_lcbs) if power_lcbs else None
    acceptance = {"target_fpr": target_fpr, "target_power": target_power, "conf": conf,
                  "rule": "upper_CI(null FPR) <= target_fpr AND lower_CI(Power@alpha) >= "
                          "target_power, conservative over both DGPs"}
    accepted = bool(fpr_ucb is not None and power_lcb is not None
                    and fpr_ucb <= target_fpr and power_lcb >= target_power)
    panel_resolved = _resolve_path(panel_csv)
    oracle_params = ValidatorOracle(config_path).oracle_gate_params()
    binding = compute_binding(
        pipeline_config_raw=cfg_raw, oracle_gate_params=oracle_params,
        panel_content_hash=content_hash(panel_resolved), acceptance=acceptance)
    gate_config = {k: gate_args[k] for k in ("q_max", "rho", "kappa", "survival_b", "t_thresh")}
    gate_config["weight_gate"] = {"l1_alpha": pipe_cfg.l1_alpha, "l2_alpha": pipe_cfg.l2_alpha,
                                  "min_abs_weight": pipe_cfg.min_abs_weight}
    certificate = {
        "gate_fpr": round(gate_fpr, 4),
        "gate_fpr_upper_ci": round(fpr_ucb, 4) if fpr_ucb is not None else None,
        "gate_power": round(gate_power, 4) if np.isfinite(gate_power) else None,
        "gate_power_lower_ci": round(power_lcb, 4) if power_lcb is not None else None,
        "gate_power_alpha": f"IC={alpha_ic:g}",
        "accepted": accepted,
        "acceptance_rule": acceptance["rule"],
        "acceptance": acceptance,
        "binding": binding,
        "selector_note": "certificate selector = base-column five-gate stack (four_gate + sparse "
                         "weights); the oracle additionally applies ingredient/neighbour controls "
                         "to DERIVED candidates -- closed-loop coverage is the realized-FDR harness",
        "selector": "five_gate(FDR->redundancy->survival->incremental->sparse_weight)",
        "dgp_policy": {"gen1_reps": gen1_reps, "gen2_reps": gen2_reps,
                       "gen3": "excluded from agent trials (paper-defense DGP only)"},
        "per_dgp": {"gen1_stage_a_t": res_gen1, "gen2_dgp_v2": res_gen2},
        "n_assets": n_assets, "n_months": n_months, "seed": seed,
        "true_factors_cross_family": [factors[j] for j in cross_idx],
        "gate_config": gate_config,
        "gate_config_hash": hashlib.sha1(
            json.dumps(gate_config, sort_keys=True).encode()).hexdigest()[:12],
        "calibration_panel": str(panel_resolved),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(certificate, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")
    print(f"\ngate_fpr={certificate['gate_fpr']}  gate_power={certificate['gate_power']} "
          f"@ {certificate['gate_power_alpha']}  accepted={certificate['accepted']}")
    print(f"Wrote {output_path}")
    return certificate


def main() -> None:
    """Calibrate the supplied panel and estimate a source-bound process certificate.

    Acceptance depends on the configured Monte Carlo bounds; successfully writing
    a certificate does not imply the acceptance rule passed.
    """
    ap = argparse.ArgumentParser(description="Issue the process-level gate certificate (Gen-1 + Gen-2).")
    ap.add_argument("--panel-csv", default=str(_PROJECT_DIR / "data_big_v2" / "a_share_factor_panel.csv"))
    ap.add_argument("--config", default=str(_PROJECT_DIR / "a_share_big_config.json"))
    ap.add_argument("--output", default=str(_PROJECT_DIR / "outputs" / "agent" / "gate_certificate.json"))
    ap.add_argument("--gen1-reps", type=int, default=60,
                    help="Gen-1 is cheap; it carries the sampling load (default 2x Gen-2).")
    ap.add_argument("--gen2-reps", type=int, default=30)
    ap.add_argument("--n-assets", type=int, default=150)
    ap.add_argument("--n-months", type=int, default=150)
    ap.add_argument("--alpha-ic", type=float, default=0.03,
                    help="Signal strength the headline gate_power is quoted at.")
    ap.add_argument("--extra-ic", type=float, default=0.06, help="Second strength, reported per-DGP.")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--target-fpr", type=float, default=0.1,
                    help="acceptance: one-sided upper CI of null FPR must be <= this")
    ap.add_argument("--target-power", type=float, default=0.3,
                    help="acceptance: one-sided lower CI of Power@alpha must be >= this")
    args = ap.parse_args()
    build_certificate(panel_csv=args.panel_csv, config_path=args.config,
                      gen1_reps=args.gen1_reps, gen2_reps=args.gen2_reps,
                      n_assets=args.n_assets, n_months=args.n_months,
                      alpha_ic=args.alpha_ic, extra_ic=args.extra_ic,
                      seed=args.seed, output_path=Path(args.output),
                      target_fpr=args.target_fpr, target_power=args.target_power)


if __name__ == "__main__":
    main()
