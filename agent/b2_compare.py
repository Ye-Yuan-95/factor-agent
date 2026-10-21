"""Pre-registered, paired multi-seed B2 comparison for factor generators.

The harness compares random, adaptive and GLM generators on exactly the same synthetic panel
for each seed.  The search vocabulary is restricted to one frozen family, while Oracle logic,
Memory accounting and trial budgets are shared.  Synthetic certificate downgrades remain DRAFT
in FactorCards; only the offline B2 scorecard restores ``uncertified_would_be:*`` so generator
quality can be compared without pretending that a synthetic card is investable.

The protocol is written before results and hashed into the output manifest.  Runs are resumable:
a completed arm is loaded, never silently overwritten.  GLM traces exclude raw responses and
credentials.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_PROJECT_DIR = Path(__file__).resolve().parents[1]
for _path in (str(_PROJECT_DIR), str(_PROJECT_DIR / "synthetic")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from agent.baseline_generator import BaselineGenerator  # noqa: E402
from agent.certification import GATE_SOURCE_FILES, wilson_upper  # noqa: E402
from agent.factor_dsl import DSLError, compile_expression  # noqa: E402
from agent.generator_contract import (  # noqa: E402
    GeneratorSpec,
    ResearchRequest,
    assign_expression_family,
)
from agent.llm_generator import GLMClient, GenerationError, LLMGenerator  # noqa: E402
from agent.memory import MemoryStore  # noqa: E402
from agent.validator_oracle import ValidatorOracle  # noqa: E402
from stage_a_fpr_power import calibrate, generate_panel  # noqa: E402


DEFAULT_PROTOCOL = (
    _PROJECT_DIR / "experiments" / "b2_momentum_preregistered_2026-07-20_v2.json")
_SOURCE_FILES = tuple(dict.fromkeys((
    _PROJECT_DIR / "agent" / "b2_compare.py",
    _PROJECT_DIR / "agent" / "baseline_generator.py",
    _PROJECT_DIR / "agent" / "llm_generator.py",
    _PROJECT_DIR / "agent" / "generator_contract.py",
    # DSR, certification and Gen-2 changes can alter decisions or paired panels
    # even when the harness itself is unchanged.
    *GATE_SOURCE_FILES,
)))


def _canonical_hash(value: Any, length: int = 16) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                         default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


def _source_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in _SOURCE_FILES:
        digest.update(str(path.relative_to(_PROJECT_DIR)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def _git_state() -> dict[str, Any]:
    """Record best-effort commit and dirty state for provenance; unavailable Git is reported as unknown."""
    def run(*args: str) -> str:
        if not (_PROJECT_DIR / ".git").exists():
            return "unknown"
        try:
            return subprocess.check_output(
                ["git", *args], cwd=_PROJECT_DIR, text=True,
                stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.SubprocessError):
            return "unknown"

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "--short", "HEAD"),
        "worktree_dirty": bool(status and status != "unknown"),
        "porcelain_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest()[:16]
        if status != "unknown" else None,
    }


def load_protocol(path: str | Path) -> dict[str, Any]:
    """Validate required B2 protocol fields and refuse configurations explicitly gated against execution."""
    protocol = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"protocol_id", "generators", "query", "family", "dgp", "budgets",
                "oracle", "glm", "go_no_go", "output_dir"}
    missing = sorted(required - set(protocol))
    if protocol.get("runnable") is False:
        raise SystemExit(
            f"protocol {protocol.get('protocol_id')} is gated and not runnable: "
            f"{protocol.get('gate_reason', 'no reason recorded')}")
    if missing:
        raise ValueError(f"protocol missing fields: {missing}")
    if protocol["family"] != "momentum":
        raise ValueError("This comparison protocol requires family=momentum")
    if protocol["dgp"]["kind"] not in ("gen1", "gen2"):
        raise ValueError("protocol dgp.kind must be gen1 or gen2")
    generation_budget = protocol["budgets"]["generation_attempts"]
    oracle_budget = protocol["budgets"]["oracle_trials"]
    if type(generation_budget) is not int or type(oracle_budget) is not int:
        raise ValueError("protocol budgets must be exact integers")
    if oracle_budget <= 0 or generation_budget < oracle_budget:
        raise ValueError("generation budget must be >= positive Oracle budget")
    alt = protocol["dgp"]["alternative_seeds"]
    null = protocol["dgp"]["null_seeds"]
    if not alt or not null or any(type(seed) is not int for seed in [*alt, *null]):
        raise ValueError("protocol requires integer alternative and null seeds")
    if set(alt) & set(null):
        raise ValueError("alternative and null seeds must not overlap")
    if protocol["dgp"]["true_factor"] != "mom1m":
        raise ValueError("This comparison protocol's truth is frozen to mom1m")
    return protocol


def _panel_hash(panel: pd.DataFrame) -> str:
    """Content hash for paired-arm verification without persisting the generated panel."""
    digest = hashlib.sha256()
    digest.update(json.dumps(list(panel.columns), ensure_ascii=False).encode("utf-8"))
    digest.update(json.dumps([str(dtype) for dtype in panel.dtypes]).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(panel, index=True, categorize=True).values.tobytes())
    return digest.hexdigest()[:16]


def _generate_scenario_panel(*, protocol: dict[str, Any], calib: dict[str, Any],
                             seed: int, scenario: str) -> tuple[pd.DataFrame, list[str]]:
    """Generate paired null/alternative panels using the declared truth factor and seed."""
    factors = list(calib["factors"])
    true_factor = str(protocol["dgp"]["true_factor"])
    if true_factor not in factors:
        raise ValueError(f"true factor {true_factor!r} is absent from calibration factors")
    true_idx = [factors.index(true_factor)] if scenario == "alternative" else []
    strength = float(protocol["dgp"]["inject_ic"])
    if protocol["dgp"]["kind"] == "gen2":
        from dgp_v2 import generate_panel_v2
        panel, _ = generate_panel_v2(
            np.random.default_rng(seed), calib,
            int(protocol["dgp"]["n_assets"]), int(protocol["dgp"]["n_months"]),
            true_idx, [strength] * len(true_idx),
            coverage=calib.get("coverage", 0.973))
    else:
        panel, _ = generate_panel(
            np.random.default_rng(seed), calib,
            int(protocol["dgp"]["n_assets"]), int(protocol["dgp"]["n_months"]),
            true_idx, [strength] * len(true_idx), tails="t")
    return panel, ([true_factor] if true_idx else [])


def _normalise_decision(card: dict[str, Any]) -> tuple[str, str | None]:
    decision = str(card.get("decision", "REJECT"))
    reason = card.get("reject_reason")
    text = str(reason or "")
    if decision == "DRAFT" and text.startswith("uncertified_would_be:ACCEPT"):
        return "ACCEPT", None
    if decision == "DRAFT" and text.startswith("uncertified_would_be:SHELVED"):
        suffix = text.split(";", 1)[1] if ";" in text else None
        return "SHELVED", suffix
    return decision, str(reason) if reason is not None else None


def _failed_gates(card: dict[str, Any]) -> list[str]:
    checks = (
        ("fdr", card.get("fdr_pass")),
        ("redundancy", card.get("redundancy_pass")),
        ("survival", card.get("survival_pass")),
        ("incremental", card.get("incremental_pass")),
        ("sparse_weight", card.get("selected_by_weight")),
        ("dsr", card.get("dsr_pass")),
    )
    return [name for name, passed in checks if passed is not True]


def _truth_loading(expression: str, panel: pd.DataFrame, truth_names: list[str]) -> float | None:
    if not truth_names:
        return None
    values = compile_expression(expression).evaluate(panel)
    best = 0.0
    for name in truth_names:
        corr = values.corr(pd.to_numeric(panel[name], errors="coerce"), method="spearman")
        if pd.notna(corr):
            best = max(best, abs(float(corr)))
    return best


def _append_jsonl(path: Path, event: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")


def _sum_trace(records: list[dict[str, Any]], key: str) -> int:
    return int(sum(int(record.get(key) or 0) for record in records))


def _proposal_quality(*, generated_proposals: int, family_adherent: int,
                      unique_signatures: int) -> tuple[float | None, float | None]:
    """Rates over proposals returned by the generator, including later coverage failures."""
    if generated_proposals <= 0:
        return None, None
    return (round(family_adherent / generated_proposals, 6),
            round(unique_signatures / generated_proposals, 6))


def run_arm(*, protocol: dict[str, Any], config_path: Path, config_raw: dict[str, Any],
            panel: pd.DataFrame, panel_content_hash: str, truth_names: list[str],
            scenario: str, seed: int, generator_name: str, out_dir: Path) -> dict[str, Any]:
    """Execute or resume one scenario/seed/generator arm with isolated cards and trial memory.
    A completed arm is reusable only with the expected panel hash; incomplete directories are refused."""
    run_dir = out_dir / "runs" / f"{scenario}_s{seed}" / generator_name
    result_path = run_dir / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("panel_content_hash") != panel_content_hash:
            raise RuntimeError(f"completed arm has wrong panel hash: {run_dir}")
        print(f"  [resume] {scenario} seed={seed} generator={generator_name}")
        return result
    if run_dir.exists():
        raise RuntimeError(
            f"incomplete arm exists at {run_dir}; refusing to overwrite pre-registered evidence")
    cards_dir = run_dir / "cards"
    memory_dir = run_dir / "memory"
    cards_dir.mkdir(parents=True)
    memory_dir.mkdir(parents=True)
    trace_path = run_dir / "trace.jsonl"

    family = str(protocol["family"])
    family_map = config_raw["family_constraints"]["factor_families"]
    vocab_families = set(protocol.get("request_allowed_families", [family]))
    columns = sorted(column for column in config_raw["factor_cols"]
                     if family_map.get(column) in vocab_families)
    generator_spec_cfg = protocol.get("generator_spec", {})
    if generator_spec_cfg.get("vocabulary", "base") == "extended":
        from agent.generator_contract import PIT_RAW_COLUMNS
        columns = columns + [c for c in PIT_RAW_COLUMNS if c in panel.columns]
    spec = GeneratorSpec.from_config(config_raw, columns=columns,
                                     functions=generator_spec_cfg.get("operators"))
    generation_budget = int(protocol["budgets"]["generation_attempts"])
    oracle_budget = int(protocol["budgets"]["oracle_trials"])
    universe = f"b2_{protocol['dgp']['kind']}_{scenario}_seed{seed}_ic{protocol['dgp']['inject_ic'] if truth_names else 0}"
    request = ResearchRequest.from_dict({
        "original_query": protocol["query"],
        "objective": protocol["query"],
        "universe": universe,
        "allowed_families": list(protocol.get("request_allowed_families", [family])),
        "excluded_families": [],
        "preferences": {
            "low_redundancy": True,
            "low_turnover": False,
            "validation_stability": True,
        },
        "existing_factors": [],
        "generation_attempt_budget": generation_budget,
        "oracle_trial_budget": oracle_budget,
        "assumptions": [],
        "unsupported_requirements": [],
    })

    oracle = ValidatorOracle(
        config_path, dsr_min=float(protocol["oracle"]["dsr_min"]))
    dates = pd.to_datetime(sorted(pd.unique(panel["date"])))
    oracle.train_end = pd.Timestamp(dates[int(len(dates) * 0.7) - 1])
    oracle.val_end = pd.Timestamp(dates[-1])
    oracle.set_panel(panel, universe, label_semantics="direct")
    memory = MemoryStore(memory_dir)

    if generator_name in {"random", "adaptive"}:
        generator: Any = BaselineGenerator(
            columns, family_map, np.random.default_rng(seed + 777),
            mode=generator_name, name_prefix=f"{generator_name}_s{seed}_")
        client = None
    elif generator_name == "glm":
        glm = protocol["glm"]
        client = GLMClient(
            model=glm["model"], max_tokens=int(glm["max_tokens"]),
            max_retries=int(glm["max_retries"]))
        generator = LLMGenerator(
            client=client, spec=spec, prompt_version=glm["prompt_version"],
            mechanism_mode=bool(generator_spec_cfg.get("mechanism_mode", False)))
    else:
        raise ValueError(f"unknown generator: {generator_name}")

    _append_jsonl(trace_path, {
        "event": "arm_started", "scenario": scenario, "seed": seed,
        "generator": generator_name, "panel_content_hash": panel_content_hash,
        "generator_spec_hash": spec.fingerprint(), "request": request.to_dict(),
    })
    attempts = 0
    response_local_failures = 0
    runtime_coverage_failures = 0
    history: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    existing: list[str] = []
    signatures: set[str] = set()
    family_adherent = 0
    generation_traces: list[dict[str, Any]] = []
    generated_proposals = 0

    while attempts < generation_budget and memory.n_trials() < oracle_budget:
        attempts += 1
        if generator_name == "glm":
            try:
                proposal = generator.propose(request, history=history)
            except GenerationError as exc:
                response_local_failures += 1
                safe = {
                    **exc.trace,
                    "event": "generation_failure",
                    "failure_stage": "response_or_local_constraint",
                    "attempt": attempts,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "provider": client.provider,
                    "model": client.model,
                }
                generation_traces.append(safe)
                _append_jsonl(trace_path, safe)
                continue
            name = f"glm_s{seed}_{attempts:03d}"
            candidate = {
                "name": name,
                "expression": proposal["expression"],
                "rationale": proposal["rationale"],
                "family": proposal["family"],
            }
            signature = proposal["expression_signature"]
            generation_trace = dict(proposal["generation"])
            generation_traces.append(generation_trace)
            _append_jsonl(trace_path, {
                "event": "candidate_generated", "attempt": attempts, "name": name,
                "expression_signature": signature, "family": proposal["family"],
                "family_exposures": proposal["family_exposures"], **generation_trace,
            })
        else:
            candidate = generator.propose(history)
            if candidate is None:
                response_local_failures += 1
                _append_jsonl(trace_path, {
                    "event": "generation_failure", "attempt": attempts,
                    "failure_stage": "response_or_local_constraint",
                    "error_type": "SamplerExhausted", "error": "no novel valid proposal",
                })
                continue
            signature = compile_expression(candidate["expression"]).signature()
            _append_jsonl(trace_path, {
                "event": "candidate_generated", "attempt": attempts,
                "name": candidate["name"], "expression_signature": signature,
                "family": candidate["family"],
            })

        generated_proposals += 1
        assignment = assign_expression_family(candidate["expression"], spec)
        adherent = set(assignment.exposures).issubset(request.allowed_families)
        family_adherent += int(adherent)
        if not adherent:
            raise RuntimeError("locally valid candidate escaped the frozen family allowlist")
        if signature in signatures:
            raise RuntimeError("generator returned a duplicate signature as a valid proposal")
        signatures.add(signature)

        try:
            card = oracle.validate(
                name=candidate["name"], expression=candidate["expression"],
                rationale=candidate["rationale"], family=assignment.primary,
                existing=existing, memory=memory, seed=seed)
        except DSLError as exc:
            memory.record_failure(
                name=candidate["name"], expression=candidate["expression"],
                error=str(exc), universe=universe)
            runtime_coverage_failures += 1
            _append_jsonl(trace_path, {
                "event": "sandbox_failure", "attempt": attempts,
                "failure_stage": "runtime_or_coverage",
                "name": candidate["name"], "error": str(exc),
            })
            continue

        card_dict = card.to_dict()
        (cards_dir / f"{candidate['name']}.json").write_text(
            json.dumps(card_dict, indent=2, ensure_ascii=False), encoding="utf-8")
        score_decision, score_reason = _normalise_decision(card_dict)
        if score_decision == "ACCEPT":
            existing.append(candidate["name"])
        feedback = {
            "name": candidate["name"],
            "columns_used": sorted(compile_expression(candidate["expression"]).columns),
            "expression_signature": signature,
            "outcome": score_decision,
            "failed_gates": _failed_gates(card_dict),
            "reject_reason": score_reason,
        }
        history.append(feedback)
        candidates.append({
            "name": candidate["name"],
            "expression": candidate["expression"],
            "expression_signature": signature,
            "family": assignment.primary,
            "family_exposures": list(assignment.exposures),
            "raw_decision": card_dict["decision"],
            "score_decision": score_decision,
            "score_reason": score_reason,
            "failed_gates": feedback["failed_gates"],
            "n_trials_used": card_dict["n_trials_used"],
            "n_eff_used": card_dict["n_eff_used"],
        })
        _append_jsonl(trace_path, {
            "event": "validated", "attempt": attempts, "name": candidate["name"],
            "raw_decision": card_dict["decision"], "score_decision": score_decision,
            "certified": card_dict["certified"], "reject_reason": card_dict["reject_reason"],
        })

    threshold = float(protocol["oracle"]["truth_loading_abs_spearman"])
    for candidate in candidates:
        loading = _truth_loading(candidate["expression"], panel, truth_names)
        candidate["truth_loading"] = loading
        candidate["is_true_candidate"] = bool(loading is not None and loading >= threshold)
    accepted = [candidate for candidate in candidates
                if candidate["score_decision"] == "ACCEPT"]
    true_accepts = [candidate for candidate in accepted if candidate["is_true_candidate"]]
    false_accepts = len(accepted) - len(true_accepts)
    trials = memory.n_trials()
    family_adherence, nonduplicate_rate = _proposal_quality(
        generated_proposals=generated_proposals, family_adherent=family_adherent,
        unique_signatures=len(signatures))
    result = {
        "protocol_id": protocol["protocol_id"],
        "scenario": scenario,
        "seed": seed,
        "generator": generator_name,
        "universe": universe,
        "panel_content_hash": panel_content_hash,
        "generator_spec_hash": spec.fingerprint(),
        "allowed_columns": columns,
        "true_factors": truth_names,
        "generation_attempt_budget": generation_budget,
        "oracle_trial_budget": oracle_budget,
        "generation_attempts": attempts,
        "generation_failures": response_local_failures + runtime_coverage_failures,
        "response_local_failures": response_local_failures,
        "runtime_coverage_failures": runtime_coverage_failures,
        "generated_proposals": generated_proposals,
        "oracle_trials": trials,
        "n_eff": memory.stats()["n_eff"],
        "valid_candidate_rate": round(trials / attempts, 6) if attempts else 0.0,
        "family_adherence": family_adherence,
        "nonduplicate_rate": nonduplicate_rate,
        "n_true_candidates": sum(int(candidate["is_true_candidate"])
                                 for candidate in candidates),
        "true_candidate_rate": round(
            sum(int(candidate["is_true_candidate"]) for candidate in candidates) / len(candidates),
            6) if candidates else None,
        "n_would_be_accept": len(accepted),
        "n_true_accept": len(true_accepts),
        "n_false_accept": false_accepts,
        "empirical_fdp": round(false_accepts / len(accepted), 6) if accepted else None,
        "power_recovered": int(bool(true_accepts)) if truth_names else None,
        "raw_decisions": dict(Counter(candidate["raw_decision"] for candidate in candidates)),
        "score_decisions": dict(Counter(candidate["score_decision"] for candidate in candidates)),
        "input_tokens": _sum_trace(generation_traces, "input_tokens"),
        "output_tokens": _sum_trace(generation_traces, "output_tokens"),
        "total_tokens": _sum_trace(generation_traces, "total_tokens"),
        "latency_ms": _sum_trace(generation_traces, "latency_ms"),
        "candidates": candidates,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    _append_jsonl(trace_path, {
        "event": "arm_completed", "oracle_trials": trials,
        "generation_attempts": attempts, "n_would_be_accept": len(accepted),
        "n_true_accept": len(true_accepts),
    })
    print(f"  [{generator_name:>8}] {scenario} s{seed}: attempts={attempts} trials={trials} "
          f"would_accept={len(accepted)} true_accept={len(true_accepts)}")
    return result


def _generator_summary(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate attempts, trials, accepted candidates and truth recovery with explicit denominators.
    Pooled candidate FDP is distinct from the mean per-run FDP and from null false-positive rates."""
    selected = [row for row in rows if row["generator"] == name]
    alt = [row for row in selected if row["scenario"] == "alternative"]
    null = [row for row in selected if row["scenario"] == "null"]
    attempts = sum(row["generation_attempts"] for row in selected)
    trials = sum(row["oracle_trials"] for row in selected)
    accepts = sum(row["n_would_be_accept"] for row in alt)
    false_accepts = sum(row["n_false_accept"] for row in alt)
    null_accepts = sum(row["n_would_be_accept"] for row in null)
    null_trials = sum(row["oracle_trials"] for row in null)
    all_candidates = sum(len(row["candidates"]) for row in selected)
    true_candidates = sum(row["n_true_candidates"] for row in alt)
    alt_candidates = sum(len(row["candidates"]) for row in alt)
    family_values = [row["family_adherence"] for row in selected
                     if row["family_adherence"] is not None]
    duplicate_values = [row["nonduplicate_rate"] for row in selected
                        if row["nonduplicate_rate"] is not None]
    powers = [row["power_recovered"] for row in alt if row["power_recovered"] is not None]
    fdp = false_accepts / accepts if accepts else None
    return {
        "generator": name,
        "completed_alternative_seeds": len(alt),
        "completed_null_seeds": len(null),
        "generation_attempts": attempts,
        "response_local_failures": sum(row.get("response_local_failures", 0)
                                       for row in selected),
        "runtime_coverage_failures": sum(row.get("runtime_coverage_failures", 0)
                                         for row in selected),
        "oracle_trials": trials,
        "valid_candidate_rate": round(trials / attempts, 6) if attempts else 0.0,
        "mean_family_adherence": round(float(np.mean(family_values)), 6)
        if family_values else None,
        "mean_nonduplicate_rate": round(float(np.mean(duplicate_values)), 6)
        if duplicate_values else None,
        "alternative_candidates": alt_candidates,
        "true_candidates": true_candidates,
        "true_candidate_rate": round(true_candidates / alt_candidates, 6)
        if alt_candidates else None,
        "would_be_accepts": accepts,
        "true_accepts": sum(row["n_true_accept"] for row in alt),
        "false_accepts": false_accepts,
        "pooled_empirical_fdp": round(fdp, 6) if fdp is not None else None,
        "pooled_fdp_wilson_upper": round(wilson_upper(false_accepts, accepts), 6)
        if accepts else None,
        "mean_power_recovered": round(float(np.mean(powers)), 6) if powers else None,
        "null_would_be_accepts": null_accepts,
        "null_trials": null_trials,
        "null_accept_rate_per_trial": round(null_accepts / null_trials, 6)
        if null_trials else None,
        "mean_n_eff": round(float(np.mean([row["n_eff"] for row in selected])), 6)
        if selected else None,
        "input_tokens": sum(row["input_tokens"] for row in selected),
        "output_tokens": sum(row["output_tokens"] for row in selected),
        "total_tokens": sum(row["total_tokens"] for row in selected),
        "latency_ms": sum(row["latency_ms"] for row in selected),
        "total_candidates": all_candidates,
    }


def _go_no_go(protocol: dict[str, Any], summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the protocol's minimum-evidence and performance rules; insufficient acceptance is inconclusive."""
    by_name = {summary["generator"]: summary for summary in summaries}
    if "glm" not in by_name:
        return {"status": "PENDING_GLM", "checks": {},
                "reason": "GLM arm has not been run yet"}
    glm = by_name["glm"]
    rule = protocol["go_no_go"]
    baseline_powers = [by_name[name]["mean_power_recovered"]
                       for name in ("random", "adaptive")
                       if name in by_name and by_name[name]["mean_power_recovered"] is not None]
    enough_accepts = glm["would_be_accepts"] >= int(
        rule["minimum_glm_would_be_accepts_for_fdp_decision"])
    fdp_ok = (glm["pooled_empirical_fdp"] is not None and
              glm["pooled_empirical_fdp"] <= float(
                  rule["maximum_glm_pooled_empirical_fdp"]))
    power_ok = (glm["mean_power_recovered"] is not None and baseline_powers and
                glm["mean_power_recovered"] >= max(baseline_powers))
    checks = {
        "alternative_seeds_complete": glm["completed_alternative_seeds"] >= int(
            rule["minimum_completed_alternative_seeds"]),
        "null_seeds_complete": glm["completed_null_seeds"] >= int(
            rule["minimum_completed_null_seeds"]),
        "valid_candidate_rate": glm["valid_candidate_rate"] >= float(
            rule["minimum_glm_valid_candidate_rate"]),
        "family_adherence": glm["mean_family_adherence"] == float(
            rule["required_family_adherence"]),
        "sufficient_accept_denominator": enough_accepts,
        "pooled_empirical_fdp": fdp_ok if enough_accepts else None,
        "null_accept_rate": (
            glm["null_accept_rate_per_trial"] is not None and
            glm["null_accept_rate_per_trial"] <= float(
                rule["maximum_glm_null_accept_rate_per_trial"])),
        "power_at_least_best_baseline": bool(power_ok),
    }
    if not enough_accepts:
        status = str(rule["insufficient_denominator_result"])
        reason = "would-be ACCEPT denominator is below the pre-registered minimum"
    else:
        status = "GO" if all(value is True for value in checks.values()) else "NO_GO"
        reason = "all pre-registered checks passed" if status == "GO" else (
            "one or more pre-registered checks failed")
    return {"status": status, "checks": checks, "reason": reason}


def _write_outputs(*, protocol: dict[str, Any], rows: list[dict[str, Any]], out_dir: Path) -> dict:
    """Write deterministic comparison artifacts from completed arms, retaining would-be decision semantics."""
    names = [name for name in protocol["generators"]
             if any(row["generator"] == name for row in rows)]
    summaries = [_generator_summary(name, rows) for name in names]
    go_no_go = _go_no_go(protocol, summaries)
    output = {
        "protocol_id": protocol["protocol_id"],
        "protocol_hash": _canonical_hash(protocol),
        "generators_completed": names,
        "summaries": summaries,
        "go_no_go": go_no_go,
        "interpretation": (
            "All ACCEPT figures are offline would-be decisions on synthetic panels. "
            "Original FactorCards remain uncertified DRAFT where applicable."),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    flat_rows = [{key: value for key, value in row.items() if key != "candidates"} for row in rows]
    pd.DataFrame(flat_rows).sort_values(["scenario", "seed", "generator"]).to_csv(
        out_dir / "runs.csv", index=False, encoding="utf-8-sig")
    lines = [
        f"# B2 Momentum Comparison — {protocol['protocol_id']}", "",
        f"- Protocol hash: `{_canonical_hash(protocol)}`",
        f"- Status: **{go_no_go['status']}** — {go_no_go['reason']}",
        "- ACCEPT metrics below are synthetic-only would-be decisions; FactorCards remain DRAFT.",
        "", "## Generator summary", "",
        "| Generator | Valid/attempt | Fail local/runtime | Family | True candidate rate | Would-ACCEPT | FDP | Power | Null accept/trial | Tokens | Latency ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        def show(value: Any) -> str:
            return "n/a" if value is None else str(value)
        lines.append(
            f"| {summary['generator']} | {summary['valid_candidate_rate']:.3f} | "
            f"{summary['response_local_failures']}/{summary['runtime_coverage_failures']} | "
            f"{show(summary['mean_family_adherence'])} | "
            f"{show(summary['true_candidate_rate'])} | {summary['would_be_accepts']} | "
            f"{show(summary['pooled_empirical_fdp'])} | "
            f"{show(summary['mean_power_recovered'])} | "
            f"{summary['null_would_be_accepts']}/{summary['null_trials']} | "
            f"{summary['total_tokens']} | {summary['latency_ms']} |")
    lines.extend(["", "## Go/no-go checks", ""])
    if go_no_go["checks"]:
        for name, value in go_no_go["checks"].items():
            lines.append(f"- `{name}`: {value}")
    else:
        lines.append("- GLM results pending.")
    lines.extend(["", "## Scope", "",
                  "This pilot does not authorize a real-panel run, test-lockbox opening, or production portfolio use.", ""])
    (out_dir / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    return output


def main() -> None:
    """Run paired family comparisons under the declared protocol and source identity.

    Check provider availability and resume compatibility before executing the
    selected generator/seed arms. Aggregate completed artifacts into comparison
    reports; missing evidence keeps the protocol decision inconclusive.
    """
    parser = argparse.ArgumentParser(
        description="Run the pre-registered paired multi-seed B2 momentum comparison")
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--config", default=str(_PROJECT_DIR / "a_share_big_config.json"))
    parser.add_argument("--panel-csv", default=str(
        _PROJECT_DIR / "data_big_v2" / "a_share_factor_panel.csv"))
    parser.add_argument("--generators", default="",
                        help="comma-separated subset for resumable execution")
    args = parser.parse_args()

    protocol_path = Path(args.protocol)
    protocol = load_protocol(protocol_path)
    requested = ([item.strip() for item in args.generators.split(",") if item.strip()]
                 or list(protocol["generators"]))
    unknown = sorted(set(requested) - set(protocol["generators"]))
    if unknown:
        parser.error(f"generators are not in the frozen protocol: {unknown}")
    if "glm" in requested and not os.environ.get("GLM_API_KEY"):
        raise SystemExit(
            "GLM_API_KEY is not present in this process. Run baseline arms here, then execute "
            "--generators glm in the PowerShell session that owns the temporary key.")

    config_path = Path(args.config).resolve()
    panel_csv = Path(args.panel_csv).resolve()
    config_raw = json.loads(config_path.read_text(encoding="utf-8"))
    out_dir = (_PROJECT_DIR / protocol["output_dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    protocol_hash = _canonical_hash(protocol)
    config_hash = _canonical_hash(config_raw)
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("protocol_hash") != protocol_hash:
            raise RuntimeError("output directory belongs to a different protocol hash")
        if manifest.get("config_hash") != config_hash:
            raise RuntimeError("config hash changed or is missing; use a new protocol/output dir")
        if manifest.get("source_fingerprint") != _source_fingerprint():
            raise RuntimeError(
                "B2 source fingerprint changed after execution began; use a new protocol/output dir")
    else:
        manifest = {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "purpose": "paired_multiseed_b2_generator_comparison",
            "protocol_path": str(protocol_path.resolve()),
            "protocol_hash": protocol_hash,
            "source_fingerprint": _source_fingerprint(),
            "config_hash": config_hash,
            "calibration_panel": str(panel_csv),
            "git": _git_state(),
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        (out_dir / "protocol.snapshot.json").write_text(
            json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")
        (out_dir / "config.snapshot.json").write_text(
            json.dumps(config_raw, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"protocol={protocol['protocol_id']} hash={protocol_hash}")
    print(f"source={manifest['source_fingerprint']} generators={requested}")
    dgp_kind = protocol["dgp"]["kind"]
    print(f"calibrating {dgp_kind} to {panel_csv} ...")
    if dgp_kind == "gen2":
        from dgp_v2 import calibrate_v2
        calib = calibrate_v2(panel_csv, config_raw["factor_cols"],
                             config_raw["family_constraints"]["factor_families"])
    else:
        calib = calibrate(panel_csv, config_raw["factor_cols"])
    all_rows: list[dict[str, Any]] = []
    for scenario, seeds in (
        ("alternative", protocol["dgp"]["alternative_seeds"]),
        ("null", protocol["dgp"]["null_seeds"]),
    ):
        for seed in seeds:
            panel, truth_names = _generate_scenario_panel(
                protocol=protocol, calib=calib, seed=int(seed), scenario=scenario)
            panel_content_hash = _panel_hash(panel)
            paired_hashes: set[str] = set()
            for generator_name in requested:
                result = run_arm(
                    protocol=protocol, config_path=config_path, config_raw=config_raw,
                    panel=panel, panel_content_hash=panel_content_hash,
                    truth_names=truth_names, scenario=scenario, seed=int(seed),
                    generator_name=generator_name, out_dir=out_dir)
                all_rows.append(result)
                paired_hashes.add(result["panel_content_hash"])
                existing_results = list((out_dir / "runs").glob("*/*/result.json"))
                rows_so_far = [json.loads(path.read_text(encoding="utf-8"))
                               for path in existing_results]
                _write_outputs(protocol=protocol, rows=rows_so_far, out_dir=out_dir)
            if len(paired_hashes) != 1:
                raise RuntimeError(f"paired panel hash mismatch for {scenario} seed={seed}")

    all_results = [json.loads(path.read_text(encoding="utf-8"))
                   for path in (out_dir / "runs").glob("*/*/result.json")]
    summary = _write_outputs(protocol=protocol, rows=all_results, out_dir=out_dir)
    print(f"summary: {out_dir / 'comparison.md'}")
    print(f"go/no-go: {summary['go_no_go']['status']}")


if __name__ == "__main__":
    main()
