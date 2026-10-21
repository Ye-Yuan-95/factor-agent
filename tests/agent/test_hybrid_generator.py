"""Exploration policy and hybrid generators: reward isolation, scheduling, novelty, coverage."""
import json

import numpy as np
import pytest

from agent.exploration import (
    CoverageTable,
    ExplorationScheduler,
    ast_distance,
    classify_gate_form,
    gate_progress_reward,
    min_ast_distance,
)
from agent.generator_contract import GeneratorSpec, ResearchRequest
from agent.hybrid_generator import (
    FormAwareHybridRandomGenerator,
    HybridLLMGenerator,
    HybridRandomGenerator,
)
from agent.llm_generator import LLMCallResult, LLMGenerator, ProposalConstraintError

FAMILIES = {"ep": "value", "bm": "value", "illiq": "liquidity", "turn": "liquidity"}
COLUMNS = ["ep", "bm", "illiq", "turn"]


class FakeClient:
    provider, model = "fake", "fake-model"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate_json(self, *, system_prompt, user_prompt):
        self.calls.append((system_prompt, user_prompt))
        return LLMCallResult(raw_text=self.responses.pop(0), provider="fake",
                             model="fake-model", input_tokens=1, output_tokens=1,
                             total_tokens=2, latency_ms=1, request_id="r", parameters={})


def _request():
    return ResearchRequest.from_dict({
        "original_query": "寻找机制因子", "objective": "寻找机制因子", "universe": "u",
        "allowed_families": ["value", "liquidity"], "excluded_families": [],
        "preferences": {}, "existing_factors": [],
        "generation_attempt_budget": 8, "oracle_trial_budget": 4,
    })


def _spec():
    return GeneratorSpec(allowed_columns=tuple(sorted(COLUMNS)), column_families=dict(FAMILIES))


# --------------------------------------------------------------------------- #
# reward: gate feedback only                                                   #
# --------------------------------------------------------------------------- #
def test_gate_progress_reward_is_graded_and_truth_free():
    assert gate_progress_reward([], "ACCEPT") == 1.0
    assert gate_progress_reward([], "SHELVED") == 0.9
    early = gate_progress_reward(["fdr", "incremental"], "REJECT")
    late = gate_progress_reward(["incremental"], "REJECT")
    deep = gate_progress_reward(["dsr"], "REJECT")
    assert early == 0.0 < late < deep < 0.9          # further down the funnel scores higher
    assert gate_progress_reward(None, "REJECT") == 0.0
    # the signature accepts nothing that could carry truth information
    import inspect
    assert list(inspect.signature(gate_progress_reward).parameters) == ["failed_gates",
                                                                       "decision"]


def test_gate_form_classification_separates_the_f3_failure_mode():
    assert classify_gate_form("zscore(ep) * sign(illiq)") == "level_gate"
    assert classify_gate_form("zscore(ep) * sign(delta(illiq, 3))") == "delta_gate"
    assert classify_gate_form("rank(ep) * clip(zscore(illiq), 0, 3)") == "clip_gate"
    assert classify_gate_form("zscore(ep) - zscore(illiq)") == "spread"
    assert classify_gate_form("rank(ts_mean(ep, 6))") == "other"


# --------------------------------------------------------------------------- #
# scheduler                                                                    #
# --------------------------------------------------------------------------- #
def test_scheduler_covers_every_arm_before_exploiting():
    scheduler = ExplorationScheduler(["conditional_signal", "regime_gated"],
                                     ["value", "liquidity"], np.random.default_rng(0))
    assert len(scheduler.arms) == 4                  # 2 recipes x 2 ordered family pairs
    seen = set()
    for _ in range(len(scheduler.arms)):
        arm = scheduler.select_arm()
        assert arm.key not in seen
        seen.add(arm.key)
        scheduler.update(arm, 0.0)
    assert seen == {arm.key for arm in scheduler.arms}


def test_scheduler_concentrates_on_the_rewarding_arm():
    scheduler = ExplorationScheduler(["conditional_signal"], ["value", "liquidity"],
                                     np.random.default_rng(3))
    good, bad = scheduler.arms[0], scheduler.arms[1]
    for _ in range(30):
        scheduler.update(good, 0.9)
        scheduler.update(bad, 0.0)
    picks = [scheduler.select_arm().key for _ in range(40)]
    assert picks.count(good.key) > picks.count(bad.key) * 3
    stats = {row["arm"]: row for row in scheduler.stats()}
    assert stats[good.key]["mean_reward"] > stats[bad.key]["mean_reward"]


def test_scheduler_rejects_impossible_configurations():
    with pytest.raises(ValueError, match="two families"):
        ExplorationScheduler(["conditional_signal"], ["value"], np.random.default_rng(0))
    scheduler = ExplorationScheduler(["conditional_signal"], ["value", "liquidity"],
                                     np.random.default_rng(0))
    with pytest.raises(ValueError, match="reward"):
        scheduler.update(scheduler.arms[0], 1.5)


# --------------------------------------------------------------------------- #
# novelty                                                                      #
# --------------------------------------------------------------------------- #
def test_ast_distance_catches_cosmetic_rewrites():
    assert ast_distance("zscore(ep) * sign(illiq)", "zscore(ep) * sign(illiq)") == 0.0
    close = ast_distance("zscore(ep) * sign(illiq)", "rank(ep) * sign(illiq)")
    far = ast_distance("zscore(ep) * sign(illiq)", "zscore(bm) - zscore(ts_mean(turn, 6))")
    assert 0 < close < far
    assert min_ast_distance("zscore(ep) * sign(illiq)", []) == 1.0


# --------------------------------------------------------------------------- #
# coverage table                                                               #
# --------------------------------------------------------------------------- #
def test_coverage_table_reports_untried_forms():
    table = CoverageTable()
    table.record(("ep", "illiq"), "zscore(ep) * sign(delta(illiq, 3))", 0.4)
    untried = table.untried_forms(("ep", "illiq"))
    assert "delta_gate" not in untried and "level_gate" in untried
    payload = table.prompt_payload()
    assert payload["n_distinct_column_pairs_tried"] == 1
    row = payload["tried_column_pairs"][0]
    assert row["columns"] == ["ep", "illiq"] and row["tried"] == 1
    json.dumps(payload, ensure_ascii=False)          # must be prompt-serialisable


# --------------------------------------------------------------------------- #
# hybrid random (the floor)                                                    #
# --------------------------------------------------------------------------- #
def test_hybrid_random_proposes_cross_family_and_learns_from_history():
    generator = HybridRandomGenerator(COLUMNS, FAMILIES, np.random.default_rng(1))
    first = generator.propose([])
    assert first is not None and first["recipe_name"]
    families = {FAMILIES[c] for c in ("ep", "bm", "illiq", "turn") if c in first["expression"]}
    assert len(families) >= 2
    history = [{"name": first["name"], "outcome": "REJECT", "failed_gates": ["incremental"]}]
    generator.propose(history)
    pulled = [row for row in generator.scheduler_stats() if row["pulls"] > 0]
    assert pulled, "gate feedback from history must reach the bandit"
    assert generator.coverage.cells, "coverage table must record the attempt"


def test_hybrid_random_novelty_blocks_structural_repeats():
    generator = HybridRandomGenerator(COLUMNS, FAMILIES, np.random.default_rng(2),
                                      min_novelty=0.9)
    proposals = []
    for _ in range(6):
        candidate = generator.propose([])
        if candidate is None:
            break
        proposals.append(candidate["expression"])
    for i, a in enumerate(proposals):
        for b in proposals[i + 1:]:
            assert ast_distance(a, b) >= 0.9


def test_form_aware_hybrid_prioritises_an_untried_form_on_a_promising_pair():
    generator = FormAwareHybridRandomGenerator(
        COLUMNS, FAMILIES, np.random.default_rng(4),
        form_completion_reward_min=0.4)
    generator.coverage.record(
        ("ep", "illiq"), "zscore(ep) * clip(sign(illiq), 0, 3)", 0.6)
    candidate = generator.propose([])
    assert candidate is not None
    assert candidate["form_priority"] is True
    assert candidate["gate_form"] in {"level_gate", "delta_gate", "spread"}
    assert candidate["gate_form"] != "clip_gate"
    from agent.factor_dsl import compile_expression
    assert set(compile_expression(candidate["expression"]).columns) == {"ep", "illiq"}
    assert generator.form_priority_picks == 1


# --------------------------------------------------------------------------- #
# hybrid LLM                                                                   #
# --------------------------------------------------------------------------- #
def test_hybrid_llm_injects_region_and_coverage_into_prompt():
    client = FakeClient([json.dumps({"expression": "zscore(ep) * sign(illiq)",
                                     "rationale": "条件价值", "recipe_name": "conditional_signal"})])
    llm = LLMGenerator(client=client, spec=_spec(), recipe_prompt=True)
    hybrid = HybridLLMGenerator(llm_generator=llm, columns=COLUMNS, families=FAMILIES,
                                rng=np.random.default_rng(0),
                                recipes=["conditional_signal"])
    proposal = hybrid.propose(_request(), history=[])
    payload = json.loads(client.calls[0][1])
    assert payload["assigned_region"]["recipe_name"] in ("conditional_signal", "regime_gated",
                                                         "cross_family_spread")
    assert "coverage" in payload
    assert proposal["assigned_arm"]


def test_hybrid_llm_novelty_pressure_rejects_repeat():
    # NOT an exact repeat -- that is already caught by the duplicate-signature check upstream.
    # Novelty pressure exists for the cosmetic rewrite: same structure, one wrapper swapped.
    first = json.dumps({"expression": "zscore(ep) * sign(illiq)", "rationale": "r",
                        "recipe_name": "conditional_signal"})
    cosmetic = json.dumps({"expression": "rank(ep) * sign(illiq)", "rationale": "r",
                           "recipe_name": "conditional_signal"})
    client = FakeClient([first, cosmetic])
    llm = LLMGenerator(client=client, spec=_spec(), recipe_prompt=True)
    # 0.4 sits above the measured wrapper-swap distance (0.333) and below a column swap (0.444)
    hybrid = HybridLLMGenerator(llm_generator=llm, columns=COLUMNS, families=FAMILIES,
                                rng=np.random.default_rng(0), min_novelty=0.4,
                                recipes=["conditional_signal"])
    hybrid.propose(_request(), history=[])
    with pytest.raises(ProposalConstraintError, match="structurally too close"):
        hybrid.propose(_request(), history=[])
    assert hybrid.novelty_rejections == 1


def test_hybrid_llm_switches_are_separable():
    payload_seen = []

    def run(**switches):
        client = FakeClient([json.dumps({"expression": "zscore(bm) * sign(turn)",
                                         "rationale": "r",
                                         "recipe_name": "conditional_signal"})])
        llm = LLMGenerator(client=client, spec=_spec(), recipe_prompt=True)
        hybrid = HybridLLMGenerator(llm_generator=llm, columns=COLUMNS, families=FAMILIES,
                                    rng=np.random.default_rng(0),
                                    recipes=["conditional_signal"], **switches)
        hybrid.propose(_request(), history=[])
        payload_seen.append(json.loads(client.calls[0][1]))

    run(use_scheduler=False, use_coverage=False, use_novelty=False)
    assert "assigned_region" not in payload_seen[0] and "coverage" not in payload_seen[0]
    run(use_scheduler=True, use_coverage=False)
    assert "assigned_region" in payload_seen[1] and "coverage" not in payload_seen[1]


def test_off_arm_proposal_is_hard_rejected_not_miscredited():
    """An off-region proposal must not be validated and must not be credited to the arm the
    scheduler chose; otherwise the bandit posterior learns from a reward that arm never
    earned and the ablation measures a scheduler that does not schedule."""
    columns = ["ep", "bm", "illiq", "turn", "roa"]
    families = {**FAMILIES, "roa": "profitability"}
    spec = GeneratorSpec(allowed_columns=tuple(sorted(columns)), column_families=families)
    request = ResearchRequest.from_dict({
        "original_query": "q", "objective": "q", "universe": "u",
        "allowed_families": ["value", "liquidity", "profitability"], "excluded_families": [],
        "preferences": {}, "existing_factors": [],
        "generation_attempt_budget": 8, "oracle_trial_budget": 4})
    client = FakeClient([json.dumps({"expression": "zscore(roa) * sign(turn)",
                                     "rationale": "r"})])
    llm = LLMGenerator(client=client, spec=spec)
    hybrid = HybridLLMGenerator(llm_generator=llm, columns=columns, families=families,
                                rng=np.random.default_rng(0))
    arm = hybrid.scheduler.select_arm()
    with pytest.raises(ProposalConstraintError, match="leaves the assigned arm"):
        hybrid.propose(request, history=[])
    assert hybrid.off_arm_proposals == 1
    assert hybrid.scheduler.state[arm.key].pulls == 0        # nothing credited
    assert not hybrid._pending                                # nothing queued for feedback


def test_hybrid_llm_feedback_loop_closes_through_register_name():
    """The harness names candidates after generation, so the arm bookkeeping is keyed by the
    generator's placeholder until register_name re-keys it. Without that call the bandit
    never sees a single reward."""
    client = FakeClient([json.dumps({"expression": "zscore(ep) * sign(illiq)",
                                     "rationale": "r", "recipe_name": "conditional_signal"})])
    llm = LLMGenerator(client=client, spec=_spec(), recipe_prompt=True)
    hybrid = HybridLLMGenerator(llm_generator=llm, columns=COLUMNS, families=FAMILIES,
                                rng=np.random.default_rng(0),
                                recipes=["conditional_signal"])
    proposal = hybrid.propose(_request(), history=[])
    assert proposal["pending_key"] == proposal["expression"]
    hybrid.register_name(proposal["pending_key"], "arm_s1_001")
    hybrid._absorb_history([{"name": "arm_s1_001", "outcome": "REJECT",
                             "failed_gates": ["incremental"]}])
    pulled = [row for row in hybrid.scheduler_stats() if row["pulls"] > 0]
    assert len(pulled) == 1 and pulled[0]["mean_reward"] > 0
    assert hybrid.coverage.cells
    hybrid.finalize([{"name": "arm_s1_001", "outcome": "REJECT",
                      "failed_gates": ["incremental"]}])
    assert hybrid.diagnostics()["n_pending_feedback"] == 0


def test_assigned_recipe_mismatch_is_hard_rejected_not_miscredited():
    """Matching families are insufficient: the scheduled recipe itself must be obeyed."""
    client = FakeClient([json.dumps({
        "expression": "zscore(ep) * sign(illiq)",
        "rationale": "r",
        "recipe_name": "conditional_signal",
    })])
    llm = LLMGenerator(client=client, spec=_spec(), recipe_prompt=True)
    hybrid = HybridLLMGenerator(
        llm_generator=llm, columns=COLUMNS, families=FAMILIES,
        rng=np.random.default_rng(0), recipes=["regime_gated"])
    arm = hybrid.scheduler.select_arm()
    with pytest.raises(ProposalConstraintError, match="declared recipe='conditional_signal'"):
        hybrid.propose(_request(), history=[])
    assert arm.recipe == "regime_gated"
    assert hybrid.off_arm_proposals == 1
    assert hybrid.scheduler.state[arm.key].pulls == 0
    assert not hybrid._pending


def test_hybrid_glm_arm_is_wired_into_the_harness():
    from agent.factorial_ablation import _SOURCE_FILES, _arm_plan
    names = {path.name for path in _SOURCE_FILES}
    # the exploration policy is the generator under test; it must be fingerprinted
    assert {"exploration.py", "hybrid_generator.py"} <= names
    protocol = {
        "cells": [{"cell_id": "h1", "vocabulary": "narrow", "recipe_prompt": True,
                   "hard_constraint": True, "generator": "hybrid_glm",
                   "use_coverage": False}],
        "baseline_arms": [],
    }
    plan = _arm_plan(protocol)
    assert plan[0]["arm_id"] == "hybrid_glm_h1" and plan[0]["kind"] == "hybrid_glm"
    assert plan[0]["cell"]["use_coverage"] is False


def test_extra_payload_cannot_override_prompt_fields():
    client = FakeClient([json.dumps({"expression": "rank(ep)", "rationale": "r"})])
    llm = LLMGenerator(client=client, spec=_spec())
    with pytest.raises(ValueError, match="may not override"):
        llm.propose(_request(), history=[], extra_payload={"generator_spec": {"hacked": True}})
