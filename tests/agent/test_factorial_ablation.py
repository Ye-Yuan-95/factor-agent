"""Factorial-ablation building blocks: recipe conformance checking, the recipe_name proposal
schema, the recipe-following random baseline, protocol loading and Wilson interval logic."""
import json

import numpy as np
import pytest

from agent.baseline_generator import RecipeBaselineGenerator
from agent.factor_dsl import compile_expression
from agent.generator_contract import (
    RECIPE_NAMES,
    GeneratorSpec,
    ResearchRequest,
    check_recipe_conformance,
    sanitize_generator_history,
)
from agent.llm_generator import (
    CandidateProposal,
    LLMCallResult,
    LLMGenerator,
    ProposalConstraintError,
    ProposalFormatError,
)


def _spec():
    return GeneratorSpec(
        allowed_columns=("a", "b", "c"),
        column_families={"a": "f1", "b": "f2", "c": "f3"},
    )


def _call(raw_text):
    return LLMCallResult(
        raw_text=raw_text, provider="fake", model="fake-model",
        input_tokens=10, output_tokens=5, total_tokens=15, latency_ms=1,
        request_id="req-1", parameters={"response_format": "json_object"})


class FakeClient:
    provider = "fake"
    model = "fake-model"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate_json(self, *, system_prompt, user_prompt):
        self.calls.append((system_prompt, user_prompt))
        return _call(self.responses.pop(0))


def _request():
    return ResearchRequest.from_dict({
        "original_query": "寻找具有增量信息的因子",
        "objective": "寻找具有增量信息的因子",
        "universe": "factorial_test",
        "allowed_families": ["f1", "f2", "f3"],
        "excluded_families": [],
        "preferences": {},
        "existing_factors": [],
        "generation_attempt_budget": 4,
        "oracle_trial_budget": 2,
    })


# --------------------------------------------------------------------------- #
# check_recipe_conformance                                                    #
# --------------------------------------------------------------------------- #
def test_conditional_signal_conformance():
    spec = _spec()
    ok, reason = check_recipe_conformance("zscore(a) * sign(b)", "conditional_signal", spec)
    assert ok, reason
    ok, _ = check_recipe_conformance("-1 * (zscore(a) * sign(b))", "conditional_signal", spec)
    assert ok  # constant wrappers are stripped
    ok, reason = check_recipe_conformance("zscore(a) * sign(a)", "conditional_signal", spec)
    assert not ok and "share families" in reason
    ok, reason = check_recipe_conformance("zscore(a) + sign(b)", "conditional_signal", spec)
    assert not ok and "product" in reason
    ok, reason = check_recipe_conformance("zscore(a) * 2", "conditional_signal", spec)
    assert not ok  # constant side has no data column


def test_regime_gated_conformance():
    spec = _spec()
    ok, reason = check_recipe_conformance(
        "rank(a) * clip(zscore(b), 0, 3)", "regime_gated", spec)
    assert ok, reason
    ok, _ = check_recipe_conformance("sign(b) * rank(a)", "regime_gated", spec)
    assert ok  # gate side order does not matter
    ok, reason = check_recipe_conformance("rank(a) * zscore(b)", "regime_gated", spec)
    assert not ok and "gate" in reason


def test_cross_family_spread_conformance():
    spec = _spec()
    ok, reason = check_recipe_conformance(
        "zscore(a) - zscore(ts_mean(b, 6))", "cross_family_spread", spec)
    assert ok, reason
    ok, reason = check_recipe_conformance("a - zscore(b)", "cross_family_spread", spec)
    assert not ok and "rank/zscore" in reason
    ok, reason = check_recipe_conformance("zscore(a) - zscore(a)", "cross_family_spread", spec)
    assert not ok and "share families" in reason
    ok, reason = check_recipe_conformance("zscore(a) * sign(b)", "no_such_recipe", spec)
    assert not ok and "unknown recipe" in reason


# --------------------------------------------------------------------------- #
# proposal schema with recipe_name                                            #
# --------------------------------------------------------------------------- #
def test_proposal_recipe_name_schema_is_strict():
    base = {"expression": "zscore(a) * sign(b)", "rationale": "条件动量假设"}
    with_recipe = {**base, "recipe_name": "conditional_signal"}
    # not required -> the field is rejected (schema stays exact)
    with pytest.raises(ProposalFormatError, match="exactly"):
        CandidateProposal.from_json(json.dumps(with_recipe))
    # required -> the field must be present
    with pytest.raises(ProposalFormatError, match="exactly"):
        CandidateProposal.from_json(json.dumps(base), require_recipe_name=True)
    proposal = CandidateProposal.from_json(json.dumps(with_recipe), require_recipe_name=True)
    assert proposal.recipe_name == "conditional_signal"
    assert CandidateProposal.from_json(json.dumps(base)).recipe_name is None


def test_llm_generator_recipe_prompt_records_conformance():
    spec = _spec()
    good = json.dumps({"expression": "zscore(a) * sign(b)", "rationale": "条件信号",
                       "recipe_name": "conditional_signal"})
    nonconformant = json.dumps({"expression": "zscore(b) - zscore(c)", "rationale": "价差",
                                "recipe_name": "conditional_signal"})
    bad_name = json.dumps({"expression": "zscore(a) * sign(c)", "rationale": "x",
                           "recipe_name": "invented_recipe"})
    generator = LLMGenerator(client=FakeClient([good, nonconformant, bad_name]), spec=spec,
                             recipe_prompt=True, cross_family_required=False)
    first = generator.propose(_request())
    assert first["recipe_name"] == "conditional_signal"
    assert first["recipe_conformant"] is True
    second = generator.propose(_request())
    assert second["recipe_conformant"] is False    # recorded, NOT blocked
    assert second["recipe_violation"]
    with pytest.raises(ProposalConstraintError, match="not one of"):
        generator.propose(_request())


def test_llm_generator_flags_are_separable():
    spec = _spec()
    single_family = json.dumps({"expression": "rank(ts_mean(a, 6))", "rationale": "平滑动量"})
    # cross_family off + recipe off: single-family proposal passes, no recipe fields
    generator = LLMGenerator(client=FakeClient([single_family]), spec=spec)
    proposal = generator.propose(_request())
    assert proposal["recipe_name"] is None and proposal["recipe_conformant"] is None
    # cross_family on alone rejects single-family output without requiring recipe_name
    generator = LLMGenerator(client=FakeClient([single_family]), spec=spec,
                             cross_family_required=True)
    with pytest.raises(ProposalConstraintError, match="cross-family"):
        generator.propose(_request())
    # The combined mechanism switch displays recipes and enforces cross-family
    # output without requiring a recipe_name declaration.
    legacy = LLMGenerator(client=FakeClient([]), spec=spec, mechanism_mode=True)
    assert legacy.show_recipes and legacy.cross_family_required
    assert not legacy.recipe_prompt


def test_recipe_prompt_appears_only_when_enabled():
    spec = _spec()
    on = LLMGenerator(client=FakeClient(
        [json.dumps({"expression": "zscore(a) * sign(b)", "rationale": "r",
                     "recipe_name": "conditional_signal"})]),
        spec=spec, recipe_prompt=True)
    on.propose(_request())
    system_prompt, user_prompt = on.client.calls[0]
    assert "recipe_name" in system_prompt and "conditional_signal" in user_prompt
    off = LLMGenerator(client=FakeClient(
        [json.dumps({"expression": "rank(a)", "rationale": "r"})]), spec=spec)
    off.propose(_request())
    system_prompt, user_prompt = off.client.calls[0]
    assert "recipe_name" not in system_prompt
    assert json.loads(user_prompt)["mechanism_recipes"] == []


def test_recipe_name_survives_history_sanitization():
    history = [{"name": "x", "expression": "zscore(a) * sign(b)", "outcome": "REJECT",
                "failed_gates": ["incremental"], "recipe_name": "conditional_signal"}]
    safe = sanitize_generator_history(history)
    assert safe[0]["recipe_name"] == "conditional_signal"


# --------------------------------------------------------------------------- #
# recipe-following random baseline                                            #
# --------------------------------------------------------------------------- #
def test_recipe_baseline_generator_is_conformant_by_construction():
    spec = _spec()
    generator = RecipeBaselineGenerator(
        ["a", "b", "c"], {"a": "f1", "b": "f2", "c": "f3"},
        np.random.default_rng(0), name_prefix="rr")
    seen = set()
    for _ in range(20):
        candidate = generator.propose([])
        assert candidate is not None
        assert candidate["recipe_name"] in RECIPE_NAMES
        ok, reason = check_recipe_conformance(
            candidate["expression"], candidate["recipe_name"], spec)
        assert ok, f"{candidate['expression']} vs {candidate['recipe_name']}: {reason}"
        signature = compile_expression(candidate["expression"]).signature()
        assert signature not in seen
        seen.add(signature)


def test_recipe_baseline_requires_two_families():
    with pytest.raises(ValueError, match="two families"):
        RecipeBaselineGenerator(["a"], {"a": "f1"}, np.random.default_rng(0))


# --------------------------------------------------------------------------- #
# protocol + interval logic                                                   #
# --------------------------------------------------------------------------- #
def test_factorial_protocol_loads_and_plans_all_arms():
    from agent.factorial_ablation import _arm_plan, load_protocol, wilson_lower
    from pathlib import Path
    protocol = load_protocol(
        Path(__file__).resolve().parents[2] / "experiments"
        / "factorial_mechanism_2026-07-24.json")
    plan = _arm_plan(protocol)
    assert len(plan) == 12  # 4 baselines + 8 GLM cells
    assert {arm["arm_id"] for arm in plan} >= {"random_narrow", "recipe_random_wide",
                                               "glm_vn_r0_h0", "glm_vw_r1_h1"}
    assert wilson_lower(0, 40) == 0.0
    assert wilson_lower(40, 40) > 0.9
    upper_low_k = wilson_lower(10, 40)
    assert upper_low_k is not None and 0 < upper_low_k < 0.25


def test_enumerator_generator_covers_pairs_without_repeats():
    from agent.baseline_generator import EnumeratorBaselineGenerator
    spec = _spec()
    generator = EnumeratorBaselineGenerator(
        ["a", "b", "c"], {"a": "f1", "b": "f2", "c": "f3"},
        np.random.default_rng(7), name_prefix="enum")
    assert generator.n_pairs == 6  # 3 columns, all distinct families, ordered pairs
    seen_expressions = set()
    proposals = []
    while True:
        candidate = generator.propose([])
        if candidate is None:
            break
        proposals.append(candidate)
        assert candidate["expression"] not in seen_expressions
        seen_expressions.add(candidate["expression"])
        ok, reason = check_recipe_conformance(
            candidate["expression"], candidate["recipe_name"], spec)
        assert ok, reason
    assert len(proposals) == 6  # exhausts exactly the pair space, then returns None
    # deterministic per seed: same rng seed -> same order
    again = EnumeratorBaselineGenerator(
        ["a", "b", "c"], {"a": "f1", "b": "f2", "c": "f3"},
        np.random.default_rng(7), name_prefix="enum")
    assert [again.propose([])["expression"] for _ in range(6)] == \
        [p["expression"] for p in proposals]


def test_form_blind_enumerator_covers_pair_x_form_cells_without_repeats():
    from collections import Counter
    from agent.baseline_generator import FormBlindEnumeratorGenerator
    from agent.exploration import classify_gate_form

    spec = _spec()
    generator = FormBlindEnumeratorGenerator(
        ["a", "b", "c"], {"a": "f1", "b": "f2", "c": "f3"},
        np.random.default_rng(7), name_prefix="fbe")
    assert generator.n_cells == 24  # 6 ordered pairs x 4 gate forms
    proposals = []
    while True:
        candidate = generator.propose([])
        if candidate is None:
            break
        proposals.append(candidate)
        assert classify_gate_form(candidate["expression"]) == candidate["gate_form"]
        ok, reason = check_recipe_conformance(
            candidate["expression"], candidate["recipe_name"], spec)
        assert ok, reason
    assert len({candidate["expression"] for candidate in proposals}) == 24
    assert Counter(candidate["gate_form"] for candidate in proposals) == {
        "level_gate": 6, "delta_gate": 6, "clip_gate": 6, "spread": 6}


def test_hybrid_glm_requires_an_api_key_just_like_plain_glm():
    from agent.factorial_ablation import _glm_status, _requires_glm_key

    assert _requires_glm_key([{"kind": "glm"}])
    assert _requires_glm_key([{"kind": "hybrid_glm"}])
    assert not _requires_glm_key([{"kind": "hybrid_random"}])
    assert _glm_status([{"kind": "hybrid_random"}], []) == "NOT_APPLICABLE"
    assert _glm_status([{"kind": "hybrid_glm"}], []) == "PENDING_GLM"
    assert _glm_status(
        [{"kind": "hybrid_glm"}], [{"arm_kind": "hybrid_glm"}]) == "COMPLETE"


def test_budget_ladder_protocol_loads_with_overrides():
    from agent.factorial_ablation import _arm_plan, load_protocol
    from pathlib import Path
    protocol = load_protocol(
        Path(__file__).resolve().parents[2] / "experiments"
        / "factorial_mechanism_2026-07-27_F3.json")
    plan = _arm_plan(protocol)
    by_id = {arm["arm_id"]: arm for arm in plan}
    assert by_id["enumerator_narrow_b24"]["oracle_trials"] == 24
    assert by_id["enumerator_narrow_b12"]["scenarios"] == ["alternative"]
    assert by_id["enumerator_narrow_b36"]["scenarios"] == ["alternative", "null"]
    assert by_id["glm_vn_r1_h1_b36"]["generation_attempts"] == 108
    assert by_id["glm_vn_r1_h1_b36"]["cell"]["hard_constraint"] is True
    assert protocol["interpretation"]["mode"] == "budget_ladder"


def test_f4b_protocol_freezes_new_seeds_and_four_local_lines():
    from pathlib import Path
    from agent.factorial_ablation import _arm_plan, load_protocol

    protocol = load_protocol(
        Path(__file__).resolve().parents[2] / "experiments"
        / "hybrid_form_aware_2026-07-31_F4b.json")
    plan = _arm_plan(protocol)
    assert [arm["kind"] for arm in plan] == [
        "hybrid_random", "hybrid_form_aware", "form_blind", "enumerator"]
    assert protocol["dgp"]["alternative_seeds"] == list(range(2300, 2315))
    assert protocol["dgp"]["null_seeds"] == list(range(2400, 2405))
    assert protocol["interpretation"]["checkpoints"] == [12, 24, 36]
    assert all(arm["oracle_trials"] == 36 for arm in plan)


def test_f4b_exact_paired_tests_are_enumerated_not_asymptotic():
    from benchmark.f4b_analysis import exact_paired_binary, exact_sign_flip

    continuous = exact_sign_flip([0.1, 0.2, 0.3])
    binary = exact_paired_binary([1, 1, 1], [0, 0, 0])
    assert continuous["one_sided_exact_p"] == 0.125
    assert continuous["n_nonzero"] == 3
    assert binary["one_sided_exact_p"] == 0.125
    assert binary["discordant"] == 3


def test_truth_loading_keeps_values_below_the_cutoff(monkeypatch):
    import pandas as pd
    from agent.factorial_ablation import _truth_loading as mechanism_loading
    from agent.b2_compare import _truth_loading as family_loading
    from benchmark.f4b_analysis import _prefix_metrics, exact_sign_flip

    monkeypatch.setattr(pd.Series, "corr", lambda *a, **k: 0.4999996)
    panel = pd.DataFrame({"date": ["2000-01-31"], "asset": ["s"], "a": [1.0]})
    assert mechanism_loading("a", panel, pd.Series([1.0])) < 0.5
    assert family_loading("a", panel, ["a"]) < 0.5
    result = {"scenario": "alternative", "seed": 1, "arm_id": "test",
              "candidates": [{"truth_loading": 0.4999996, "score_decision": "REJECT",
                              "is_true_candidate": False}]}
    assert _prefix_metrics(result, 1)["max_truth_loading"] == 0.4999996
    assert exact_sign_flip([1e-10])["mean_difference"] > 0


def test_factorial_protocol_rejects_seed_overlap(tmp_path):
    from agent.factorial_ablation import load_protocol
    from pathlib import Path
    raw = json.loads((Path(__file__).resolve().parents[2] / "experiments"
                      / "factorial_mechanism_2026-07-24.json").read_text(encoding="utf-8"))
    raw["dgp"]["null_seeds"] = [2100]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="overlap"):
        load_protocol(bad)
