"""Empirical FDP of the discovery loop on synthetic ground truth.

Report pooled FDP, mean FDP over all alternative runs (empty discoveries = 0)
and the mean conditional on discoveries. Candidate-level Wilson bounds assume
independent candidates; they are not seed-cluster bounds or a population FDR
certificate. Threshold decisions use unrounded integer ratios.

Each run has a fresh market, memory and generator with a fixed trial budget.
Generation uses allowed gate feedback; accepted factors grow the control library.
Truth is evaluated through the declared loading criterion, not string matching.
Gen-1 carries the main sampling budget and Gen-2 provides robustness checks.
Every null-market accept counts as a false discovery.

Run: python -m agent.realized_fdr --gen1-seeds 10 --gen1-null-seeds 4
     --gen2-seeds 3 --budget 12"""
from __future__ import annotations

import argparse
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

from agent.baseline_generator import BaselineGenerator  # noqa: E402
from agent.factor_dsl import compile_expression, DSLError  # noqa: E402
from agent.memory import MemoryStore  # noqa: E402
from agent.validator_oracle import ValidatorOracle  # noqa: E402
from stage_a_fpr_power import calibrate, generate_panel  # noqa: E402
from dgp_v2 import calibrate_v2, generate_panel_v2  # noqa: E402

DEFAULT_OUT = _PROJECT_DIR / "outputs" / "agent" / "realized_fdr"


def _truth_r2(values: pd.Series, panel: pd.DataFrame, true_names: list[str]) -> float:
    """Rank-space R^2 of the candidate on the joint injected-truth set (0.0 if not computable)."""
    if not true_names:
        return 0.0
    df = pd.DataFrame({"y": values,
                       **{t: pd.to_numeric(panel[t], errors="coerce") for t in true_names}}).dropna()
    if len(df) < 50:
        return 0.0
    ranks = df.rank()
    X = np.column_stack([np.ones(len(ranks)), ranks[true_names].to_numpy(float)])
    y = ranks["y"].to_numpy(float)
    if y.var() <= 0:
        return 0.0
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return float(1.0 - resid.var() / y.var())


def cross_family_true_idx(factors: list[str], fam_map: dict[str, str], k: int = 4) -> list[int]:
    idx, seen = [], set()
    for j, f in enumerate(factors):
        fam = fam_map.get(f)
        if fam not in seen:
            seen.add(fam)
            idx.append(j)
        if len(idx) >= k:
            break
    return idx


def run_one(*, run_id: str, gen_kind: str, calib, config_path: str, seed: int, inject_ic: float,
            budget: int, generator_mode: str, fam_map: dict[str, str], truth_corr: float,
            n_assets: int, n_months: int, out_dir: Path) -> dict:
    """Generate one seeded market, run proposals and return an empirical scorecard.

    Refuse an existing run directory before spending its trial budget, and
    evaluates uncertified would-be decisions separately from process certification.
    Joint-truth loading is scored after generation, never fed back to proposals.
    Save candidate history with zero FDP for an empty discovery set.
    """
    run_dir = out_dir / "runs" / run_id
    if run_dir.exists():
        raise ValueError(f"run directory already exists; choose a new output path: {run_dir}")
    rng = np.random.default_rng(seed)
    factors = calib["factors"]
    true_idx = cross_family_true_idx(factors, fam_map) if inject_ic > 0 else []
    strengths = [inject_ic] * len(true_idx)
    if gen_kind == "gen1":
        panel, _ = generate_panel(rng, calib, n_assets, n_months, true_idx, strengths, tails="t")
    else:
        panel, _ = generate_panel_v2(rng, calib, n_assets, n_months, true_idx, strengths,
                                     coverage=calib.get("coverage", 0.973))
    true_names = [factors[j] for j in true_idx]

    oracle = ValidatorOracle(config_path)
    dates = pd.to_datetime(sorted(pd.unique(panel["date"])))
    oracle.val_end = pd.Timestamp(dates[-1])
    oracle.train_end = pd.Timestamp(dates[int(len(dates) * 0.7) - 1])
    universe = run_id
    oracle.set_panel(panel, universe, label_semantics="direct")

    memory = MemoryStore(run_dir / "memory")
    generator = BaselineGenerator(list(factors), fam_map, np.random.default_rng(seed + 777),
                                  mode=generator_mode, name_prefix=f"{gen_kind}s{seed}_")
    history: list[dict] = []
    existing: list[str] = []
    attempts = 0
    while memory.n_trials() < budget and attempts < budget * 6:
        attempts += 1
        cand = generator.propose(history)
        if cand is None:
            break
        try:
            card = oracle.validate(name=cand["name"], expression=cand["expression"],
                                   rationale=cand["rationale"], family=cand["family"],
                                   existing=existing, memory=memory, seed=seed)
        except DSLError as exc:                    # coverage/eval failure on this panel: no trial
            memory.record_failure(name=cand["name"], expression=cand["expression"],
                                  error=str(exc), universe=universe)
            history.append({"name": cand["name"], "outcome": "sandbox_failure"})
            continue
        decision = card.decision
        if decision == "DRAFT":   # harness measures the loop's WOULD-BE decisions when the
            reason = str(card.reject_reason or "")            # process certificate is not bound
            if reason.startswith("uncertified_would_be:ACCEPT"):
                decision = "ACCEPT"
            elif reason.startswith("uncertified_would_be:SHELVED"):
                decision = "SHELVED"
        if decision == "ACCEPT":
            existing.append(cand["name"])
        history.append({"name": cand["name"], "expression": cand["expression"],
                        "outcome": decision, "raw_decision": card.decision,
                        "reject_reason": card.reject_reason})

    # ---- truth mapping (MULTIVARIATE): rank-space R^2 on the joint injected-truth set must be
    # >= truth_corr^2 (univariate reduces exactly to |Spearman| >= truth_corr); a max-single-
    # column loading would undercount additive composites of half-truths.
    lockboxed = oracle.load_panel() if oracle._panel is None else oracle._panel
    accepted = [h for h in history if h["outcome"] == "ACCEPT"]
    accepted_true: list[str] = []
    recovered: set[str] = set()
    for h in accepted:
        values = compile_expression(h["expression"]).evaluate(lockboxed)
        best, best_t = 0.0, None
        for t in true_names:
            c = values.corr(pd.to_numeric(lockboxed[t], errors="coerce"), method="spearman")
            if pd.notna(c) and abs(c) > best:
                best, best_t = abs(float(c)), t
        h["truth_loading"] = round(best, 3)
        h["truth_r2"] = _truth_r2(values, lockboxed, true_names)
        if h["truth_r2"] >= truth_corr ** 2:
            accepted_true.append(h["name"])
            if best_t is not None and best >= 0.3:   # attribute recovery to the dominant column
                recovered.add(best_t)

    n_acc = len(accepted)
    n_false = n_acc - len(accepted_true)
    row = {
        "run_id": run_id, "dgp": gen_kind, "seed": seed, "inject_ic": inject_ic,
        "budget": budget, "n_trials": memory.n_trials(), "n_proposals": attempts,
        "n_accept": n_acc, "n_accept_true": len(accepted_true), "n_accept_false": n_false,
        # Per-run FDP is zero for an empty discovery set; do not exclude such
        # runs when averaging over the repeated experiment.
        "run_fdp": n_false / n_acc if n_acc else 0.0,
        "realized_fdr": round(n_false / n_acc, 4) if n_acc else 0.0,
        "power_recovered": round(len(recovered) / len(true_names), 3) if true_names else None,
        "decisions": {d: sum(1 for h in history if h["outcome"] == d)
                      for d in ("ACCEPT", "SHELVED", "REJECT", "sandbox_failure")},
    }
    (out_dir / "runs" / run_id / "history.json").write_text(
        json.dumps({"true_factors": true_names, "history": history}, indent=1, ensure_ascii=False),
        encoding="utf-8")
    flag = "NULL " if inject_ic == 0 else ""
    print(f"  [{run_id:>14}] {flag}trials={row['n_trials']:>2} accept={n_acc} "
          f"false={n_false} rFDR={row['realized_fdr']} power={row['power_recovered']}")
    return row


def summarize_fdp(rows: list[dict], target_q: float) -> dict:
    """Distinguish pooled FDP, all-run mean FDP and acceptance-conditional mean.

    Recompute from integer counts, including zero-discovery runs as FDP=0. Rates
    are descriptive; candidates and runs sharing a panel seed are not independent.
    The threshold decision uses the unrounded pooled ratio, not displayed values.
    """
    alt = [r for r in rows if r["inject_ic"] > 0]
    total = sum(r["n_accept"] for r in alt)
    false = sum(r["n_accept_false"] for r in alt)
    fdps = [r["n_accept_false"] / r["n_accept"] if r["n_accept"] else 0.0 for r in alt]
    positive = [fdp for r, fdp in zip(alt, fdps) if r["n_accept"] > 0]
    pooled = false / total if total else None
    return {
        "fdp_schema_version": 2,
        "alt_runs": len(alt), "alt_total_accepts": total,
        "alt_total_false_accepts": false,
        "pooled_empirical_fdp": pooled,
        "mean_run_fdp": float(np.mean(fdps)) if fdps else None,
        "mean_run_fdp_given_acceptance": float(np.mean(positive)) if positive else None,
        "n_zero_accept_alt_runs": len(alt) - len(positive),
        "all_run_fdp_denominator": len(alt),
        "conditional_run_fdp_denominator": len(positive),
        "zero_discovery_run_fdp": 0.0,
        "point_rule_passed": pooled is not None and pooled <= target_q,
        "uncertainty_note": "Descriptive ratios; pooled candidate Wilson bounds assume independent candidates and are not seed-cluster confidence bounds or a population FDR guarantee.",
    }


def main() -> None:
    """Evaluate the discovery loop over declared null and alternative seed runs.

    Save run scorecards and empirical FDP/recovery summaries. Truth is used only
    for evaluation, and pooled candidate bounds do not establish population FDR.
    """
    ap = argparse.ArgumentParser(description="L2 research: empirical FDP with explicit run denominators.")
    ap.add_argument("--panel-csv", default=str(_PROJECT_DIR / "data_big_v2" / "a_share_factor_panel.csv"))
    ap.add_argument("--config", default=str(_PROJECT_DIR / "a_share_big_config.json"))
    ap.add_argument("--output-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--gen1-seeds", type=int, default=10, help="Gen-1 alternative-market runs")
    ap.add_argument("--gen1-null-seeds", type=int, default=4, help="Gen-1 null-market runs")
    ap.add_argument("--gen2-seeds", type=int, default=3, help="Gen-2 alternative-market spot checks")
    ap.add_argument("--budget", type=int, default=12, help="trial budget (DSR's N) per run")
    ap.add_argument("--inject-ic", type=float, default=0.06)
    ap.add_argument("--generator", choices=["random", "adaptive"], default="adaptive")
    ap.add_argument("--truth-corr", type=float, default=0.5,
                    help="truth strictness: candidate counts true if rank-space R2 on the joint "
                         "injected-truth set >= truth_corr^2 (univariate == |Spearman| >= truth_corr)")
    ap.add_argument("--target-q", type=float, default=0.25,
                    help="point acceptance: pooled empirical FDP <= target q; not a population FDR guarantee")
    ap.add_argument("--n-assets", type=int, default=150)
    ap.add_argument("--n-months", type=int, default=150)
    ap.add_argument("--seed0", type=int, default=100)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        ap.error(f"output directory is not empty; choose a new path: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_raw = json.loads(Path(args.config).read_text(encoding="utf-8"))
    fam_map = cfg_raw["family_constraints"]["factor_families"]
    # harness-level manifest + config snapshot + certificate refs
    _oracle_probe = ValidatorOracle(args.config)
    _cert, _cert_refs = _oracle_probe._load_certificate()
    (out_dir / "manifest.json").write_text(json.dumps(
        {"created_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "purpose": "realized_fdr_harness",
         "cli_args": vars(args)}, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    (out_dir / "config.snapshot.json").write_text(json.dumps(
        {"pipeline_config": cfg_raw, "oracle_gate_params": _oracle_probe.oracle_gate_params()},
        indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    (out_dir / "certificate_refs.json").write_text(json.dumps(
        _cert_refs, indent=1, ensure_ascii=False, default=str), encoding="utf-8")

    print(f"calibrating Gen-1/Gen-2 to {args.panel_csv} ...")
    calib1 = calibrate(Path(args.panel_csv), cfg_raw["factor_cols"])
    calib2 = calibrate_v2(args.panel_csv, cfg_raw["factor_cols"], fam_map) if args.gen2_seeds else None

    rows: list[dict] = []
    plan = ([("gen1", calib1, args.seed0 + i, args.inject_ic) for i in range(args.gen1_seeds)]
            + [("gen1", calib1, args.seed0 + 500 + i, 0.0) for i in range(args.gen1_null_seeds)]
            + [("gen2", calib2, args.seed0 + 900 + i, args.inject_ic) for i in range(args.gen2_seeds)])
    print(f"{len(plan)} runs x budget {args.budget} trials, generator={args.generator}, "
          f"truth |rho|>={args.truth_corr}, target q={args.target_q}")
    for gen_kind, calib, seed, ic in plan:
        run_id = f"{gen_kind}_ic{ic:g}_s{seed}"
        rows.append(run_one(run_id=run_id, gen_kind=gen_kind, calib=calib, config_path=args.config,
                            seed=seed, inject_ic=ic, budget=args.budget,
                            generator_mode=args.generator, fam_map=fam_map,
                            truth_corr=args.truth_corr, n_assets=args.n_assets,
                            n_months=args.n_months, out_dir=out_dir))
        # crash-safe: persist the scoreboard after every run
        pd.DataFrame([{k: v for k, v in r.items() if k != "decisions"} for r in rows]).to_csv(
            out_dir / "runs.csv", index=False, encoding="utf-8-sig")

    alt = [r for r in rows if r["inject_ic"] > 0]
    null = [r for r in rows if r["inject_ic"] == 0]
    tot_acc = sum(r["n_accept"] for r in alt)
    tot_false = sum(r["n_accept_false"] for r in alt)
    pooled_fdr = round(tot_false / tot_acc, 4) if tot_acc else None
    fdp_summary = summarize_fdp(rows, args.target_q)
    powers = [r["power_recovered"] for r in alt if r["power_recovered"] is not None]
    null_accepts = sum(r["n_accept"] for r in null)
    null_trials = sum(r["n_trials"] for r in null)

    from agent.certification import wilson_upper
    fdp_ucb = wilson_upper(tot_false, tot_acc) if tot_acc else None
    certificate = {
        "accepted": fdp_summary["point_rule_passed"],
        "accepted_upper_ci": bool(fdp_ucb is not None and fdp_ucb <= args.target_q),
        "acceptance_rule": f"point rule: pooled empirical FDP of ACCEPTed factors <= {args.target_q} "
                           "on alternative synthetic markets; the CI rule (accepted_upper_ci) "
                           "additionally requires the one-sided Wilson upper bound <= target",
        "terminology_note": "pooled_realized_fdr is a pooled empirical FDP point estimate "
                            "(FDR = E[FDP]); see pooled_fdp_wilson_upper for uncertainty",
        "pooled_realized_fdr": pooled_fdr,
        "pooled_empirical_fdp": pooled_fdr,
        "pooled_fdp_wilson_upper": round(fdp_ucb, 4) if fdp_ucb is not None else None,
        # Display alias for the all-alternative-run mean, including empty discoveries.
        "mean_run_realized_fdr": fdp_summary["mean_run_fdp"],
        "mean_power_recovered": round(float(np.mean(powers)), 4) if powers else None,
        "alt_runs": len(alt), "alt_total_accepts": tot_acc, "alt_total_false_accepts": tot_false,
        "null_runs": len(null), "null_total_trials": null_trials,
        "null_total_accepts": null_accepts,
        "null_accept_rate_per_trial": round(null_accepts / null_trials, 4) if null_trials else None,
        "generator": args.generator, "budget_per_run": args.budget,
        "inject_ic": args.inject_ic, "truth_corr_threshold": args.truth_corr,
        "truth_criterion": f"rank-space R2 on joint truth set >= {args.truth_corr ** 2:.4g} "
                           f"(univariate == |Spearman| >= {args.truth_corr:g})",
        "dgp_policy": {"gen1_alt": args.gen1_seeds, "gen1_null": args.gen1_null_seeds,
                       "gen2_alt": args.gen2_seeds, "gen3": "excluded (agent-trial policy)"},
        "n_assets": args.n_assets, "n_months": args.n_months, "seed0": args.seed0,
        "config": str(args.config), "calibration_panel": str(args.panel_csv),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "note": "agent-level certificate: measures the closed loop incl. feedback adaptation; "
                "complements the process-level gate certificate (gate_certificate.json)",
    }
    certificate.update(fdp_summary)
    (out_dir / "agent_certificate.json").write_text(
        json.dumps(certificate, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\npooled realized FDR = {pooled_fdr} (target <= {args.target_q}) "
          f"-> accepted={certificate['accepted']}")
    print(f"mean power(recovered truths) = {certificate['mean_power_recovered']}; "
          f"null accepts = {null_accepts}/{null_trials} trials")
    print(f"Wrote {out_dir / 'agent_certificate.json'} and runs.csv")


if __name__ == "__main__":
    main()
