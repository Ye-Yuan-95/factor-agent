"""Protocol-driven generator comparisons on mechanism-truth synthetic panels.

The DGP injects a cross-family mechanism. Factorial cells vary vocabulary,
recipe prompting and hard cross-family constraints; local controls sample random
expressions, recipes or canonical forms. Budget and paired-prefix protocols use
their own frozen arm definitions and interpretation rules.

Positive controls evaluate known expressions separately. They do not establish
end-to-end power or attribute every discovery failure to the generator. Report
seed-level outcomes alongside pooled rates: trials within a seed are dependent,
and a protocol's interval rule does not itself provide a cluster-aware comparison.
Prompt/truth alignment limits causal attribution to model capability."""
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

from agent.baseline_generator import (  # noqa: E402
    BaselineGenerator,
    EnumeratorBaselineGenerator,
    FormBlindEnumeratorGenerator,
    RecipeBaselineGenerator,
)
from agent.certification import GATE_SOURCE_FILES, wilson_upper  # noqa: E402
from agent.factor_dsl import DSLError, compile_expression  # noqa: E402
from agent.generator_contract import (  # noqa: E402
    GeneratorSpec,
    ResearchRequest,
    assign_expression_family,
    check_recipe_conformance,
)
from agent.llm_generator import GLMClient, GenerationError, LLMGenerator  # noqa: E402
from agent.memory import MemoryStore  # noqa: E402
from agent.validator_oracle import ValidatorOracle  # noqa: E402
from dgp_mechanism import MECHANISM_KINDS, _mech_signal, calibrate, generate_mechanism_panel  # noqa: E402


_SOURCE_FILES = tuple(dict.fromkeys((
    _PROJECT_DIR / "agent" / "factorial_ablation.py",
    _PROJECT_DIR / "agent" / "baseline_generator.py",
    _PROJECT_DIR / "agent" / "llm_generator.py",
    _PROJECT_DIR / "agent" / "generator_contract.py",
    # Bind the exploration policy that controls hybrid-arm candidate generation.
    _PROJECT_DIR / "agent" / "exploration.py",
    _PROJECT_DIR / "agent" / "hybrid_generator.py",
    # Reuse the gate dependency list: DSR, certificate/CI logic and their source
    # dependencies affect decisions even when the caller's source is unchanged.
    *GATE_SOURCE_FILES,
    _PROJECT_DIR / "synthetic" / "dgp_mechanism.py",
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
    """Record the commit and worktree state, including an explicit unavailable-Git fallback."""
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
        "commit": run("rev-parse", "HEAD"),
        "worktree_dirty": bool(status and status != "unknown"),
        "porcelain_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest()[:16]
        if status != "unknown" else None,
    }


def _git_input_provenance(protocol_path: Path) -> dict[str, Any]:
    """Fail closed unless the protocol and every run-affecting source equal ``HEAD``.

    Unrelated local edits are allowed: a personal note or Obsidian workspace file cannot
    change the run. This is stricter and more useful than a global clean-worktree rule because
    it freezes the exact inputs while preserving unrelated user work.
    """

    def output(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=_PROJECT_DIR, text=True,
                stderr=subprocess.STDOUT).strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            detail = getattr(exc, "output", "") or str(exc)
            raise RuntimeError(f"git provenance check failed: {detail.strip()}") from exc

    repo_root = Path(output("rev-parse", "--show-toplevel")).resolve()
    paths = [protocol_path.resolve(), *[path.resolve() for path in _SOURCE_FILES]]
    blobs: dict[str, str] = {}
    for path in paths:
        try:
            relative = path.relative_to(repo_root).as_posix()
        except ValueError as exc:
            raise RuntimeError(f"frozen input is outside the git repository: {path}") from exc
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", relative],
            cwd=repo_root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if tracked.returncode != 0:
            raise RuntimeError(f"frozen input is not tracked by git: {relative}")
        unchanged = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", relative],
            cwd=repo_root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if unchanged.returncode != 0:
            raise RuntimeError(
                f"frozen input differs from HEAD; commit it before running: {relative}")
        blobs[relative] = output("rev-parse", f"HEAD:{relative}")
    protocol_relative = protocol_path.resolve().relative_to(repo_root).as_posix()
    return {
        "commit": output("rev-parse", "HEAD"),
        "protocol_path": protocol_relative,
        "protocol_blob": blobs[protocol_relative],
        "source_blobs": {key: value for key, value in blobs.items()
                         if key != protocol_relative},
        "relevant_paths_clean": True,
    }


def _requires_glm_key(arms: list[dict[str, Any]]) -> bool:
    return any(arm["kind"] in ("glm", "hybrid_glm") for arm in arms)


def wilson_lower(k: int, n: int, conf: float = 0.95) -> float | None:
    if n <= 0:
        return None
    upper_of_complement = wilson_upper(n - k, n, conf)
    return round(max(0.0, 1.0 - upper_of_complement), 6)


def _rate_ci(k: int, n: int) -> dict[str, Any]:
    return {
        "k": int(k), "n": int(n),
        "rate": round(k / n, 6) if n else None,
        "wilson_lower": wilson_lower(k, n) if n else None,
        "wilson_upper": round(wilson_upper(k, n), 6) if n else None,
    }


def load_protocol(path: str | Path) -> dict[str, Any]:
    """Validate the mechanism, seed sets, budgets, vocabularies and declared arms.

    Refuse runnable=false protocols, overlapping null/alternative seeds and invalid
    budget or search-space definitions before outputs or provider calls are made.
    Return the checked protocol without changing its frozen scientific settings.
    """
    protocol = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"protocol_id", "query", "dgp", "budgets", "oracle", "glm", "vocabularies",
                "cells", "baseline_arms", "interpretation", "output_dir"}
    if protocol.get("runnable") is False:
        raise SystemExit(
            f"protocol {protocol.get('protocol_id')} is gated and not runnable: "
            f"{protocol.get('gate_reason', 'no reason recorded')}")
    missing = sorted(required - set(protocol))
    if missing:
        raise ValueError(f"protocol missing fields: {missing}")
    dgp = protocol["dgp"]
    if dgp["kind"] != "mechanism":
        raise ValueError("factorial protocol requires dgp.kind == 'mechanism'")
    if dgp["mech_kind"] not in MECHANISM_KINDS:
        raise ValueError(f"dgp.mech_kind must be one of {MECHANISM_KINDS}")
    alt, null = dgp["alternative_seeds"], dgp["null_seeds"]
    if not alt or not null or any(type(seed) is not int for seed in [*alt, *null]):
        raise ValueError("protocol requires integer alternative and null seeds")
    if set(alt) & set(null):
        raise ValueError("alternative and null seeds must not overlap")
    generation_budget = protocol["budgets"]["generation_attempts"]
    oracle_budget = protocol["budgets"]["oracle_trials"]
    if type(generation_budget) is not int or type(oracle_budget) is not int:
        raise ValueError("protocol budgets must be exact integers")
    if oracle_budget <= 0 or generation_budget < oracle_budget:
        raise ValueError("generation budget must be >= positive Oracle budget")
    vocabularies = protocol["vocabularies"]
    optional = {"oracle_trials", "generation_attempts", "scenarios"}

    def _check_overrides(entry: dict[str, Any], label: str) -> None:
        for key in ("oracle_trials", "generation_attempts"):
            if key in entry and (type(entry[key]) is not int or entry[key] <= 0):
                raise ValueError(f"{label}: {key} override must be a positive integer")
        if "scenarios" in entry:
            if (not isinstance(entry["scenarios"], list) or not entry["scenarios"]
                    or not set(entry["scenarios"]) <= {"alternative", "null"}):
                raise ValueError(f"{label}: scenarios must be a non-empty subset of "
                                 "['alternative', 'null']")

    cell_ids: set[str] = set()
    for cell in protocol["cells"]:
        required = {"cell_id", "vocabulary", "recipe_prompt", "hard_constraint"}
        switches = {"use_scheduler", "use_coverage", "use_novelty", "generator"}
        if not (required <= set(cell) and set(cell) <= required | optional | switches):
            raise ValueError(f"malformed cell: {cell}")
        if cell["vocabulary"] not in vocabularies:
            raise ValueError(f"cell {cell['cell_id']} uses unknown vocabulary")
        if type(cell["recipe_prompt"]) is not bool or type(cell["hard_constraint"]) is not bool:
            raise ValueError(f"cell {cell['cell_id']} flags must be booleans")
        if cell["cell_id"] in cell_ids:
            raise ValueError(f"duplicate cell_id {cell['cell_id']}")
        cell_ids.add(cell["cell_id"])
        _check_overrides(cell, f"cell {cell['cell_id']}")
    for arm in protocol["baseline_arms"]:
        required = {"arm_id", "generator", "vocabulary"}
        if not (required <= set(arm) and set(arm) <= required | optional):
            raise ValueError(f"malformed baseline arm: {arm}")
        if arm["generator"] not in (
                "random", "recipe_random", "enumerator", "form_blind",
                "hybrid_random", "hybrid_form_aware"):
            raise ValueError(f"unknown baseline generator {arm['generator']}")
        if arm["vocabulary"] not in vocabularies:
            raise ValueError(f"baseline arm {arm['arm_id']} uses unknown vocabulary")
        _check_overrides(arm, f"baseline arm {arm['arm_id']}")
    return protocol


def _vocabulary_columns(protocol: dict[str, Any], config_raw: dict[str, Any],
                        vocabulary: str) -> list[str]:
    family_map = config_raw["family_constraints"]["factor_families"]
    families = protocol["vocabularies"][vocabulary]
    if families is None:
        return sorted(config_raw["factor_cols"])
    return sorted(column for column in config_raw["factor_cols"]
                  if family_map.get(column) in set(families))


def _vocabulary_families(protocol: dict[str, Any], config_raw: dict[str, Any],
                         vocabulary: str) -> list[str]:
    families = protocol["vocabularies"][vocabulary]
    if families is not None:
        return list(families)
    family_map = config_raw["family_constraints"]["factor_families"]
    return sorted({family_map.get(column, "unclassified")
                   for column in config_raw["factor_cols"]})


def _panel_hash(panel: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(list(panel.columns), ensure_ascii=False).encode("utf-8"))
    digest.update(json.dumps([str(dtype) for dtype in panel.dtypes]).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(panel, index=True, categorize=True).values.tobytes())
    return digest.hexdigest()[:16]


def _truth_series(panel: pd.DataFrame, *, mech_kind: str, x_col: str, y_col: str) -> pd.Series:
    """Reconstruct the injected per-month-standardized mechanism signal from panel columns."""
    signal = pd.Series(
        _mech_signal(mech_kind, panel[x_col].to_numpy(dtype=float),
                     panel[y_col].to_numpy(dtype=float)),
        index=panel.index)
    return signal.groupby(panel["date"]).transform(
        lambda values: (values - values.mean()) / (values.std() + 1e-12))


def _truth_loading(expression: str, panel: pd.DataFrame,
                   truth: pd.Series | None) -> float | None:
    if truth is None:
        return None
    values = compile_expression(expression).evaluate(panel)
    corr = values.corr(truth, method="spearman")
    return abs(float(corr)) if pd.notna(corr) else None


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


def _normalise_decision(card: dict[str, Any]) -> tuple[str, str | None]:
    decision = str(card.get("decision", "REJECT"))
    reason = card.get("reject_reason")
    text = str(reason or "")
    if decision == "DRAFT" and text.startswith("uncertified_would_be:ACCEPT"):
        return "ACCEPT", None
    if decision == "DRAFT" and text.startswith("uncertified_would_be:SHELVED"):
        return "SHELVED", text.split(";", 1)[1] if ";" in text else None
    return decision, str(reason) if reason is not None else None


def _append_jsonl(path: Path, event: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")


def _arm_plan(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand declared baseline and model cells into arms while retaining per-arm budget overrides."""
    def spec(entry: dict[str, Any], arm_id: str, kind: str,
             cell: dict[str, Any] | None) -> dict[str, Any]:
        return {"arm_id": arm_id, "kind": kind, "vocabulary": entry["vocabulary"],
                "cell": cell,
                "oracle_trials": entry.get("oracle_trials"),
                "generation_attempts": entry.get("generation_attempts"),
                "scenarios": list(entry.get("scenarios") or ("alternative", "null"))}

    plan = [spec(arm, arm["arm_id"], arm["generator"], None)
            for arm in protocol["baseline_arms"]]
    plan.extend(spec(cell, f"{cell.get('generator', 'glm')}_{cell['cell_id']}",
                     str(cell.get("generator", "glm")), cell)
                for cell in protocol["cells"])
    return plan


def run_arm(*, protocol: dict[str, Any], config_path: Path, config_raw: dict[str, Any],
            panel: pd.DataFrame, panel_content_hash: str, truth: pd.Series | None,
            scenario: str, seed: int, arm: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    """Run an isolated paired arm or resume its hash-matched completed result.
    Truth is used for offline scoring, not generator feedback; trials and proposals have separate budgets."""
    arm_id = arm["arm_id"]
    run_dir = out_dir / "runs" / f"{scenario}_s{seed}" / arm_id
    result_path = run_dir / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("panel_content_hash") != panel_content_hash:
            raise RuntimeError(f"completed arm has wrong panel hash: {run_dir}")
        print(f"  [resume] {scenario} seed={seed} arm={arm_id}")
        return result
    if run_dir.exists():
        raise RuntimeError(
            f"incomplete arm exists at {run_dir}; refusing to overwrite pre-registered evidence")
    cards_dir = run_dir / "cards"
    memory_dir = run_dir / "memory"
    cards_dir.mkdir(parents=True)
    memory_dir.mkdir(parents=True)
    trace_path = run_dir / "trace.jsonl"

    family_map = config_raw["family_constraints"]["factor_families"]
    columns = _vocabulary_columns(protocol, config_raw, arm["vocabulary"])
    allowed_families = _vocabulary_families(protocol, config_raw, arm["vocabulary"])
    spec = GeneratorSpec.from_config(config_raw, columns=columns)
    generation_budget = int(arm.get("generation_attempts")
                            or protocol["budgets"]["generation_attempts"])
    oracle_budget = int(arm.get("oracle_trials") or protocol["budgets"]["oracle_trials"])
    if generation_budget < oracle_budget:
        raise ValueError(f"arm {arm_id}: generation budget must be >= oracle budget")
    universe = f"factorial_{arm_id}_{scenario}_s{seed}"
    request = ResearchRequest.from_dict({
        "original_query": protocol["query"],
        "objective": protocol["query"],
        "universe": universe,
        "allowed_families": allowed_families,
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

    oracle = ValidatorOracle(config_path, dsr_min=float(protocol["oracle"]["dsr_min"]))
    dates = pd.to_datetime(sorted(pd.unique(panel["date"])))
    oracle.train_end = pd.Timestamp(dates[int(len(dates) * 0.7) - 1])
    oracle.val_end = pd.Timestamp(dates[-1])
    oracle.set_panel(panel, universe, label_semantics="direct")
    memory = MemoryStore(memory_dir)

    client = None
    if arm["kind"] == "random":
        generator: Any = BaselineGenerator(
            columns, family_map, np.random.default_rng(seed + 777),
            mode="random", name_prefix=f"random_s{seed}_")
    elif arm["kind"] == "recipe_random":
        generator = RecipeBaselineGenerator(
            columns, family_map, np.random.default_rng(seed + 888),
            name_prefix=f"rr_s{seed}_")
    elif arm["kind"] == "enumerator":
        generator = EnumeratorBaselineGenerator(
            columns, family_map, np.random.default_rng(seed + 999),
            name_prefix=f"enum_s{seed}_")
    elif arm["kind"] == "form_blind":
        generator = FormBlindEnumeratorGenerator(
            columns, family_map, np.random.default_rng(seed + 1010),
            name_prefix=f"fbe_s{seed}_")
    elif arm["kind"] == "hybrid_random":
        from agent.hybrid_generator import HybridRandomGenerator
        generator = HybridRandomGenerator(
            columns, family_map, np.random.default_rng(seed + 1111),
            name_prefix=f"hr_s{seed}_")
    elif arm["kind"] == "hybrid_form_aware":
        from agent.hybrid_generator import FormAwareHybridRandomGenerator
        generator = FormAwareHybridRandomGenerator(
            # Common random numbers with hybrid_random: paths are identical until the
            # form-completion intervention first fires.
            columns, family_map, np.random.default_rng(seed + 1111),
            name_prefix=f"hfa_s{seed}_",
            form_completion_reward_min=float(
                protocol["interpretation"]["form_completion_reward_min"]))
    elif arm["kind"] in ("glm", "hybrid_glm"):
        glm = protocol["glm"]
        cell = arm["cell"]
        client = GLMClient(
            model=glm["model"], max_tokens=int(glm["max_tokens"]),
            max_retries=int(glm["max_retries"]))
        generator = LLMGenerator(
            client=client, spec=spec, prompt_version=glm["prompt_version"],
            recipe_prompt=bool(cell["recipe_prompt"]),
            cross_family_required=bool(cell["hard_constraint"]))
        if arm["kind"] == "hybrid_glm":
            from agent.hybrid_generator import HybridLLMGenerator
            generator = HybridLLMGenerator(
                llm_generator=generator, columns=columns, families=family_map,
                rng=np.random.default_rng(seed + 2222),
                use_scheduler=bool(cell.get("use_scheduler", True)),
                use_coverage=bool(cell.get("use_coverage", True)),
                use_novelty=bool(cell.get("use_novelty", True)))
    else:
        raise ValueError(f"unknown arm kind: {arm['kind']}")

    _append_jsonl(trace_path, {
        "event": "arm_started", "scenario": scenario, "seed": seed, "arm": arm_id,
        "cell": arm["cell"], "panel_content_hash": panel_content_hash,
        "generator_spec_hash": spec.fingerprint(), "request": request.to_dict(),
    })
    attempts = 0
    response_local_failures = 0
    runtime_coverage_failures = 0
    generated_proposals = 0
    history: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    existing: list[str] = []
    signatures: set[str] = set()
    generation_traces: list[dict[str, Any]] = []
    threshold = float(protocol["oracle"]["truth_loading_abs_spearman"])
    first_true_accept_trial: int | None = None

    while attempts < generation_budget and memory.n_trials() < oracle_budget:
        attempts += 1
        recipe_name: str | None = None
        recipe_conformant: bool | None = None
        if arm["kind"] in ("glm", "hybrid_glm"):
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
            name = f"{arm_id}_s{seed}_{attempts:03d}"
            if proposal.get("pending_key") is not None:
                # close the hybrid feedback loop: the generator keyed this candidate by its
                # expression, the harness owns names, and gate feedback arrives keyed by name
                generator.register_name(str(proposal["pending_key"]), name)
            candidate = {
                "name": name,
                "expression": proposal["expression"],
                "rationale": proposal["rationale"],
                "family": proposal["family"],
            }
            signature = proposal["expression_signature"]
            recipe_name = proposal.get("recipe_name")
            recipe_conformant = proposal.get("recipe_conformant")
            generation_trace = dict(proposal["generation"])
            generation_traces.append(generation_trace)
            _append_jsonl(trace_path, {
                "event": "candidate_generated", "attempt": attempts, "name": name,
                "expression_signature": signature, "family": proposal["family"],
                "family_exposures": proposal["family_exposures"],
                "recipe_name": recipe_name, "recipe_conformant": recipe_conformant,
                "recipe_violation": proposal.get("recipe_violation"), **generation_trace,
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
            recipe_name = candidate.get("recipe_name")
            if recipe_name is not None:
                recipe_conformant, _violation = check_recipe_conformance(
                    candidate["expression"], recipe_name, spec)
            _append_jsonl(trace_path, {
                "event": "candidate_generated", "attempt": attempts,
                "name": candidate["name"], "expression_signature": signature,
                "family": candidate["family"], "recipe_name": recipe_name,
                "recipe_conformant": recipe_conformant,
            })

        generated_proposals += 1
        assignment = assign_expression_family(candidate["expression"], spec)
        if not set(assignment.exposures).issubset(request.allowed_families):
            raise RuntimeError("locally valid candidate escaped the vocabulary allowlist")
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
        loading = _truth_loading(candidate["expression"], panel, truth)
        is_true = bool(loading is not None and loading >= threshold)
        if score_decision == "ACCEPT" and is_true and first_true_accept_trial is None:
            first_true_accept_trial = memory.n_trials()
        feedback = {
            "name": candidate["name"],
            "columns_used": sorted(compile_expression(candidate["expression"]).columns),
            "expression_signature": signature,
            "outcome": score_decision,
            "failed_gates": _failed_gates(card_dict),
            "reject_reason": score_reason,
        }
        if recipe_name is not None:
            feedback["recipe_name"] = recipe_name
        history.append(feedback)
        candidates.append({
            "name": candidate["name"],
            "expression": candidate["expression"],
            "expression_signature": signature,
            "family": assignment.primary,
            "family_exposures": list(assignment.exposures),
            "recipe_name": recipe_name,
            "recipe_conformant": recipe_conformant,
            "gate_form": candidate.get("gate_form"),
            "form_priority": bool(candidate.get("form_priority", False)),
            "raw_decision": card_dict["decision"],
            "score_decision": score_decision,
            "score_reason": score_reason,
            "failed_gates": feedback["failed_gates"],
            "truth_loading": loading,
            "is_true_candidate": is_true,
            "n_trials_used": card_dict["n_trials_used"],
            "n_eff_used": card_dict["n_eff_used"],
        })
        _append_jsonl(trace_path, {
            "event": "validated", "attempt": attempts, "name": candidate["name"],
            "raw_decision": card_dict["decision"], "score_decision": score_decision,
            "truth_loading": loading, "reject_reason": card_dict["reject_reason"],
        })

    accepted = [c for c in candidates if c["score_decision"] == "ACCEPT"]
    true_accepts = [c for c in accepted if c["is_true_candidate"]]
    declared = [c for c in candidates if c["recipe_name"] is not None]
    conformant = [c for c in declared if c["recipe_conformant"] is True]
    trials = memory.n_trials()
    if hasattr(generator, "finalize"):
        generator.finalize(history)

    def _sum_trace(key: str) -> int:
        return int(sum(int(record.get(key) or 0) for record in generation_traces))

    result = {
        "protocol_id": protocol["protocol_id"],
        "scenario": scenario,
        "seed": seed,
        "arm_id": arm_id,
        "arm_kind": arm["kind"],
        "vocabulary": arm["vocabulary"],
        "cell": arm["cell"],
        "universe": universe,
        "panel_content_hash": panel_content_hash,
        "generator_spec_hash": spec.fingerprint(),
        "allowed_columns": columns,
        "generation_attempt_budget": generation_budget,
        "oracle_trial_budget": oracle_budget,
        "generation_attempts": attempts,
        "response_local_failures": response_local_failures,
        "runtime_coverage_failures": runtime_coverage_failures,
        "generated_proposals": generated_proposals,
        "oracle_trials": trials,
        "n_eff": memory.stats()["n_eff"],
        "valid_candidate_rate": round(trials / attempts, 6) if attempts else 0.0,
        "n_recipe_declared": len(declared),
        "n_recipe_conformant": len(conformant),
        "recipe_names": dict(Counter(c["recipe_name"] for c in declared)),
        "n_true_candidates": sum(int(c["is_true_candidate"]) for c in candidates),
        "n_would_be_accept": len(accepted),
        "n_true_accept": len(true_accepts),
        "n_false_accept": len(accepted) - len(true_accepts),
        "discovery": int(bool(true_accepts)) if truth is not None else None,
        "first_true_accept_trial": first_true_accept_trial,
        "score_decisions": dict(Counter(c["score_decision"] for c in candidates)),
        "input_tokens": _sum_trace("input_tokens"),
        "output_tokens": _sum_trace("output_tokens"),
        "total_tokens": _sum_trace("total_tokens"),
        "latency_ms": _sum_trace("latency_ms"),
        "generator_diagnostics": (
            generator.diagnostics() if hasattr(generator, "diagnostics") else None),
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
    print(f"  [{arm_id:>16}] {scenario} s{seed}: attempts={attempts} trials={trials} "
          f"accepts={len(accepted)} true={len(true_accepts)}")
    return result


def _pool(rows: list[dict[str, Any]], arm_ids: set[str]) -> dict[str, Any]:
    alt = [row for row in rows if row["arm_id"] in arm_ids and row["scenario"] == "alternative"]
    k = sum(row["n_true_accept"] for row in alt)
    n = sum(row["oracle_trials"] for row in alt)
    return {**_rate_ci(k, n), "arms": sorted(arm_ids), "n_seed_runs": len(alt),
            "discoveries": sum(row["discovery"] or 0 for row in alt)}


def _compare(name: str, side_a: dict[str, Any], side_b: dict[str, Any],
             label_a: str, label_b: str, minimum_trials: int) -> dict[str, Any]:
    """Apply the protocol-defined pooled interval-separation rule after the minimum trial check.
    Candidates within a seed are dependent; use the seed-paired F4b analysis for its formal comparisons."""
    status = "INSUFFICIENT_DATA"
    if side_a["n"] >= minimum_trials and side_b["n"] >= minimum_trials:
        if side_a["wilson_lower"] is not None and side_b["wilson_upper"] is not None \
                and side_a["wilson_lower"] > side_b["wilson_upper"]:
            status = f"CONFIRMED_{label_a}"
        elif side_b["wilson_lower"] is not None and side_a["wilson_upper"] is not None \
                and side_b["wilson_lower"] > side_a["wilson_upper"]:
            status = f"CONFIRMED_{label_b}"
        else:
            status = "INCONCLUSIVE"
    return {"comparison": name, label_a: side_a, label_b: side_b, "status": status}


def _glm_status(plan: list[dict[str, Any]], rows: list[dict[str, Any]]) -> str:
    if any(row["arm_kind"] in ("glm", "hybrid_glm") for row in rows):
        return "COMPLETE"
    if any(arm["kind"] in ("glm", "hybrid_glm") for arm in plan):
        return "PENDING_GLM"
    return "NOT_APPLICABLE"


def _write_outputs(*, protocol: dict[str, Any], rows: list[dict[str, Any]],
                   out_dir: Path) -> dict[str, Any]:
    """Pool completed arms and save protocol-defined comparisons plus flat run rows.

    Choose factorial or budget-ladder aggregation from the protocol, keep pending
    model arms explicit, and apply the minimum trial count before comparison
    decisions. Pooled Wilson intervals describe candidate rates; trials within a
    seed are dependent and do not become independent replications by pooling.
    """
    plan = _arm_plan(protocol)
    minimum_trials = int(protocol["interpretation"]["minimum_pooled_trials"])
    arm_summaries = []
    for arm in plan:
        selected = [row for row in rows if row["arm_id"] == arm["arm_id"]]
        if not selected:
            continue
        alt = [row for row in selected if row["scenario"] == "alternative"]
        null = [row for row in selected if row["scenario"] == "null"]
        declared = sum(row["n_recipe_declared"] for row in selected)
        conformant = sum(row["n_recipe_conformant"] for row in selected)
        arm_summaries.append({
            "arm_id": arm["arm_id"],
            "arm_kind": arm["kind"],
            "vocabulary": arm["vocabulary"],
            "cell": arm["cell"],
            "alt_seed_runs": len(alt),
            "null_seed_runs": len(null),
            "true_accept_per_trial": _rate_ci(
                sum(row["n_true_accept"] for row in alt),
                sum(row["oracle_trials"] for row in alt)),
            "true_candidate_per_proposal": _rate_ci(
                sum(row["n_true_candidates"] for row in alt),
                sum(row["generated_proposals"] for row in alt)),
            "discoveries": sum(row["discovery"] or 0 for row in alt),
            "false_accepts_alt": sum(row["n_false_accept"] for row in alt),
            "null_accept_per_trial": _rate_ci(
                sum(row["n_would_be_accept"] for row in null),
                sum(row["oracle_trials"] for row in null)),
            "recipe_conformance": _rate_ci(conformant, declared),
            "valid_candidate_rate": round(
                sum(row["oracle_trials"] for row in selected)
                / max(sum(row["generation_attempts"] for row in selected), 1), 6),
            "total_tokens": sum(row["total_tokens"] for row in selected),
        })
    glm_rows = [row for row in rows if row["arm_kind"] in ("glm", "hybrid_glm")]
    comparisons = []
    budget_curve: list[dict[str, Any]] = []
    mode = str(protocol["interpretation"].get("mode", "factorial"))
    if mode == "budget_ladder":
        budgets = sorted({row["oracle_trial_budget"] for row in rows})
        # Derive generator identities from completed arms so every measured curve is included.
        lines = sorted({row["arm_kind"] for row in rows})
        for budget in budgets:
            for line in lines:
                arm_ids = {row["arm_id"] for row in rows
                           if row["arm_kind"] == line and row["oracle_trial_budget"] == budget}
                if arm_ids:
                    budget_curve.append(
                        {"budget": budget, "generator": line, **_pool(rows, arm_ids)})
        for budget in budgets:
            pooled = {entry["generator"]: entry for entry in budget_curve
                      if entry["budget"] == budget}
            for side_a, side_b in (("glm", "enumerator"), ("glm", "recipe_random"),
                                   ("enumerator", "recipe_random")):
                if side_a in pooled and side_b in pooled:
                    comparisons.append(_compare(
                        f"b{budget}_{side_a}_vs_{side_b}", pooled[side_a], pooled[side_b],
                        side_a.upper(), side_b.upper(), minimum_trials))
    elif glm_rows:
        def cells(predicate) -> set[str]:
            return {f"glm_{cell['cell_id']}" for cell in protocol["cells"] if predicate(cell)}

        recipe_random_arms = {arm["arm_id"] for arm in protocol["baseline_arms"]
                              if arm["generator"] == "recipe_random"}
        random_arms = {arm["arm_id"] for arm in protocol["baseline_arms"]
                       if arm["generator"] == "random"}
        comparisons = [
            _compare("Q1_model_beyond_template",
                     _pool(rows, cells(lambda c: c["recipe_prompt"])),
                     _pool(rows, recipe_random_arms),
                     "GLM_RECIPE_PROMPTED", "RECIPE_RANDOM", minimum_trials),
            _compare("Q2_recipe_prompt_effect",
                     _pool(rows, cells(lambda c: c["recipe_prompt"])),
                     _pool(rows, cells(lambda c: not c["recipe_prompt"])),
                     "RECIPE_ON", "RECIPE_OFF", minimum_trials),
            _compare("Q3_vocabulary_effect",
                     _pool(rows, cells(lambda c: c["vocabulary"] == "narrow")),
                     _pool(rows, cells(lambda c: c["vocabulary"] == "wide")),
                     "NARROW", "WIDE", minimum_trials),
            _compare("Q4_hard_constraint_effect",
                     _pool(rows, cells(lambda c: c["hard_constraint"])),
                     _pool(rows, cells(lambda c: not c["hard_constraint"])),
                     "HARD_ON", "HARD_OFF", minimum_trials),
            _compare("floor_glm_vs_random",
                     _pool(rows, cells(lambda c: True)),
                     _pool(rows, random_arms),
                     "GLM_ALL", "RANDOM", minimum_trials),
        ]
    output = {
        "protocol_id": protocol["protocol_id"],
        "protocol_hash": _canonical_hash(protocol),
        "arms_completed": sorted({row["arm_id"] for row in rows}),
        "glm_status": _glm_status(plan, rows),
        "arm_summaries": arm_summaries,
        "budget_curve": budget_curve,
        "preregistered_comparisons": comparisons,
        "interpretation_rules": protocol["interpretation"],
        "note": ("All ACCEPT figures are offline would-be decisions on synthetic mechanism "
                 "panels; comparisons use pooled per-trial rates with Wilson 95% intervals "
                 "and read INCONCLUSIVE when intervals overlap. Trials cluster within seeds; "
                 "seed count is the independent-unit caveat."),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    flat = [{key: value for key, value in row.items() if key != "candidates"} for row in rows]
    pd.DataFrame(flat).sort_values(["scenario", "seed", "arm_id"]).to_csv(
        out_dir / "runs.csv", index=False, encoding="utf-8-sig")
    return output


def main() -> None:
    """Freeze protocol inputs, run requested mechanism arms or summarize artifacts.

    Relevant source and protocol files must match committed HEAD. Resume only
    when fingerprints and paired panel identities match; summarization can be
    deferred until independently executed workers have finished.
    """
    parser = argparse.ArgumentParser(
        description="Run the pre-registered factorial ablation on mechanism-truth panels")
    parser.add_argument("--protocol", default=str(
        _PROJECT_DIR / "experiments" / "factorial_mechanism_2026-07-24.json"))
    parser.add_argument("--config", default=str(_PROJECT_DIR / "a_share_big_config.json"))
    parser.add_argument("--panel-csv", default=str(
        _PROJECT_DIR / "data_big_v2" / "a_share_factor_panel.csv"))
    parser.add_argument("--arms", default="",
                        help="comma-separated arm_id subset for resumable execution "
                             "(baseline arm ids and glm_<cell_id>)")
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument(
        "--defer-summary", action="store_true",
        help="write only disjoint run artifacts; use for parallel arm workers, then call "
             "--summarize-only once all workers finish")
    execution.add_argument(
        "--summarize-only", action="store_true",
        help="rebuild summary/runs.csv from completed immutable result artifacts without "
             "running any arm")
    args = parser.parse_args()

    protocol_path = Path(args.protocol)
    protocol = load_protocol(protocol_path)
    plan = _arm_plan(protocol)
    known = [arm["arm_id"] for arm in plan]
    requested_ids = ([item.strip() for item in args.arms.split(",") if item.strip()] or known)
    unknown = sorted(set(requested_ids) - set(known))
    if unknown:
        parser.error(f"arms are not in the frozen protocol: {unknown}")
    requested = [arm for arm in plan if arm["arm_id"] in set(requested_ids)]
    if not args.summarize_only and _requires_glm_key(requested) \
            and not os.environ.get("GLM_API_KEY"):
        raise SystemExit(
            "GLM_API_KEY is not present in this process. Run baseline arms here, then execute "
            "--arms glm_... in the PowerShell session that owns the temporary key.")

    config_path = Path(args.config).resolve()
    panel_csv = Path(args.panel_csv).resolve()
    config_raw = json.loads(config_path.read_text(encoding="utf-8"))
    frozen_inputs = _git_input_provenance(protocol_path)
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
            raise RuntimeError(
                "config hash changed or is missing; use a new protocol/output dir")
        if manifest.get("source_fingerprint") != _source_fingerprint():
            raise RuntimeError(
                "source fingerprint changed after execution began; use a new protocol/output dir")
    else:
        manifest = {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "purpose": "factorial_mechanism_ablation",
            "protocol_path": str(protocol_path.resolve()),
            "protocol_hash": protocol_hash,
            "source_fingerprint": _source_fingerprint(),
            "config_hash": config_hash,
            "calibration_panel": str(panel_csv),
            "git": _git_state(),
            "frozen_inputs": frozen_inputs,
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        (out_dir / "protocol.snapshot.json").write_text(
            json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")
        (out_dir / "config.snapshot.json").write_text(
            json.dumps(config_raw, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"protocol={protocol['protocol_id']} hash={protocol_hash}")
    print(f"source={manifest['source_fingerprint']} arms={[a['arm_id'] for a in requested]}")
    if args.summarize_only:
        rows = [json.loads(path.read_text(encoding="utf-8"))
                for path in (out_dir / "runs").glob("*/*/result.json")]
        if not rows:
            raise RuntimeError("no completed result artifacts to summarize")
        summary = _write_outputs(protocol=protocol, rows=rows, out_dir=out_dir)
        print(f"summary: {out_dir / 'summary.json'}")
        print(f"glm_status: {summary['glm_status']}")
        return
    dgp = protocol["dgp"]
    print(f"calibrating mechanism DGP to {panel_csv} ...")
    calib = calibrate(panel_csv, config_raw["factor_cols"])
    for scenario, seeds, ic in (
        ("alternative", dgp["alternative_seeds"], float(dgp["inject_ic"])),
        ("null", dgp["null_seeds"], 0.0),
    ):
        for seed in seeds:
            panel = generate_mechanism_panel(
                np.random.default_rng(int(seed)), calib,
                int(dgp["n_assets"]), int(dgp["n_months"]),
                kind=dgp["mech_kind"], x_col=dgp["x_col"], y_col=dgp["y_col"], ic=ic)
            panel_content_hash = _panel_hash(panel)
            truth = (_truth_series(panel, mech_kind=dgp["mech_kind"],
                                   x_col=dgp["x_col"], y_col=dgp["y_col"])
                     if scenario == "alternative" else None)
            paired_hashes: set[str] = set()
            for arm in requested:
                if scenario not in arm["scenarios"]:
                    continue
                result = run_arm(
                    protocol=protocol, config_path=config_path, config_raw=config_raw,
                    panel=panel, panel_content_hash=panel_content_hash, truth=truth,
                    scenario=scenario, seed=int(seed), arm=arm, out_dir=out_dir)
                paired_hashes.add(result["panel_content_hash"])
                if not args.defer_summary:
                    rows = [json.loads(path.read_text(encoding="utf-8"))
                            for path in (out_dir / "runs").glob("*/*/result.json")]
                    _write_outputs(protocol=protocol, rows=rows, out_dir=out_dir)
            if len(paired_hashes) > 1:
                raise RuntimeError(f"paired panel hash mismatch for {scenario} seed={seed}")

    if args.defer_summary:
        print("summary deferred; run --summarize-only after all workers finish")
    else:
        rows = [json.loads(path.read_text(encoding="utf-8"))
                for path in (out_dir / "runs").glob("*/*/result.json")]
        summary = _write_outputs(protocol=protocol, rows=rows, out_dir=out_dir)
        print(f"summary: {out_dir / 'summary.json'}")
        print(f"glm_status: {summary['glm_status']}")
        for comparison in summary["preregistered_comparisons"]:
            print(f"  {comparison['comparison']}: {comparison['status']}")


if __name__ == "__main__":
    main()
