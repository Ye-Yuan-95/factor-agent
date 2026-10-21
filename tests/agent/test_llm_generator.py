"""Verify model-response validation, sanitized feedback and explicit provider failures."""
import json
from types import SimpleNamespace

import pytest

from agent.factor_dsl import compile_expression
from agent.generator_contract import (
    FeedbackLeakageError,
    GeneratorSpec,
    RequestValidationError,
    ResearchRequest,
    sanitize_generator_history,
)
from agent.llm_generator import (
    CandidateProposal,
    DuplicateProposalError,
    GLMClient,
    LLMCallResult,
    LLMGenerator,
    MissingCredentialError,
    ProposalConstraintError,
    ProposalFormatError,
    interpret_research_request,
)


def _spec():
    return GeneratorSpec(
        allowed_columns=("ep", "mom6m", "roa"),
        column_families={"ep": "value", "mom6m": "momentum", "roa": "profitability"},
    )


def _request(**overrides):
    """Create a frozen request with deterministic scope and budgets for mocked proposals."""
    raw = {
        "original_query": "寻找低冗余价值因子",
        "objective": "寻找低冗余的价值因子",
        "universe": "synthetic_gen1_seed7_ic0.06",
        "allowed_families": ["value"],
        "excluded_families": [],
        "preferences": {"low_redundancy": True},
        "existing_factors": ["size"],
        "generation_attempt_budget": 4,
        "oracle_trial_budget": 2,
    }
    raw.update(overrides)
    return ResearchRequest.from_dict(raw)


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


def test_research_request_is_strict_and_host_owns_budgets():
    request = _request()
    assert request.oracle_trial_budget == 2
    assert request.to_dict()["allowed_families"] == ["value"]
    with pytest.raises(RequestValidationError, match="unsupported"):
        _request(model_decision="ACCEPT")
    with pytest.raises(RequestValidationError, match="positive integer"):
        _request(oracle_trial_budget=True)
    with pytest.raises(RequestValidationError, match=">="):
        _request(generation_attempt_budget=1, oracle_trial_budget=2)


def test_candidate_schema_forbids_model_owned_name_family_and_decision():
    good = CandidateProposal.from_json(
        json.dumps({"expression": "rank(ep)", "rationale": "估值假设"}))
    assert good.expression == "rank(ep)"
    for extra in ("name", "family", "decision", "sharpe"):
        raw = {"expression": "rank(ep)", "rationale": "估值假设", extra: "x"}
        with pytest.raises(ProposalFormatError, match="exactly"):
            CandidateProposal.from_json(json.dumps(raw))


def test_history_projection_fails_closed_on_test_or_lockbox_fields():
    safe = sanitize_generator_history([{
        "name": "c1", "expression": "rank(ep)", "outcome": "REJECT",
        "reject_reason": "fdr", "fdr_q": 0.9,
    }])
    assert set(safe[0]) == {
        "name", "expression_signature", "outcome", "failed_gates", "reject_reason"}
    assert "fdr_q" not in safe[0]
    with pytest.raises(FeedbackLeakageError, match="test_sharpe"):
        sanitize_generator_history([{"test_sharpe_delta": 1.0}])
    with pytest.raises(FeedbackLeakageError, match="lockbox"):
        sanitize_generator_history([{"nested": {"lockbox_result": "pass"}}])


def test_llm_generator_validates_dsl_family_and_duplicates():
    client = FakeClient([
        json.dumps({"expression": "rank(ep)", "rationale": "低估值假设"}),
        json.dumps({"expression": "rank(ep)", "rationale": "重复"}),
    ])
    generator = LLMGenerator(client=client, spec=_spec())
    proposal = generator.propose(_request(), history=[])
    assert proposal["family"] == "value"
    assert proposal["family_exposures"] == ["value"]
    assert proposal["expression_signature"] == compile_expression("rank(ep)").signature()
    assert proposal["generation"]["model"] == "fake-model"
    prompt = json.loads(client.calls[0][1])
    signatures = prompt["generator_spec"]["function_signatures"]
    assert signatures["ts_zscore"] == "ts_zscore(series, window)"
    assert signatures["demean"] == "demean(series)"
    assert signatures["clip"] == "clip(series, lower, upper)"
    with pytest.raises(DuplicateProposalError) as duplicate_error:
        generator.propose(_request(), history=[])
    _assert_safe_failure_trace(duplicate_error.value)

    wrong_family = LLMGenerator(
        client=FakeClient([json.dumps({
            "expression": "rank(mom6m)", "rationale": "动量假设"})]),
        spec=_spec())
    with pytest.raises(ProposalConstraintError, match="outside allowed") as family_error:
        wrong_family.propose(_request(), history=[])
    _assert_safe_failure_trace(family_error.value)


@pytest.mark.parametrize("raw, error_type", [
    (json.dumps({
        "expression": "rank(ep)", "rationale": "extra field", "decision": "ACCEPT",
    }), ProposalFormatError),
    (json.dumps({
        "expression": "rank(unknown_column)", "rationale": "outside the spec",
    }), ProposalConstraintError),
    (json.dumps({
        "expression": "rank(ts_zscore(ep))", "rationale": "missing window",
    }), ProposalConstraintError),
])
def test_provider_success_local_failure_keeps_safe_provenance(raw, error_type):
    generator = LLMGenerator(client=FakeClient([raw]), spec=_spec())
    with pytest.raises(error_type) as caught:
        generator.propose(_request(), history=[])
    _assert_safe_failure_trace(caught.value)


def test_request_interpreter_cannot_override_host_scope_or_budgets():
    response = json.dumps({
        "objective": "寻找验证期稳健的价值信号",
        "allowed_families": ["profitability"],
        "excluded_families": ["momentum"],
        "preferences": {
            "low_redundancy": True,
            "low_turnover": False,
            "validation_stability": True,
        },
        "assumptions": ["将低冗余解释为相对现有因子的信息差异"],
        "unsupported_requirements": [],
    })
    request, call = interpret_research_request(
        query="找一个稳健的价值因子", client=FakeClient([response]), spec=_spec(),
        universe="fixed-panel", existing_factors=["size"],
        hard_allowed_families=["value"], generation_attempt_budget=6,
        oracle_trial_budget=3)
    assert request.universe == "fixed-panel"
    assert request.existing_factors == ("size",)
    assert request.allowed_families == ("value",)
    assert request.generation_attempt_budget == 6
    assert request.oracle_trial_budget == 3
    assert call.total_tokens == 15


def test_glm_adapter_uses_json_mode_without_exposing_key(monkeypatch):
    response = SimpleNamespace(
        id="glm-request-1",
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"expression":"rank(ep)",'
                                                         '"rationale":"估值假设"}'))],
        usage=SimpleNamespace(prompt_tokens=8, completion_tokens=4, total_tokens=12),
    )

    class Completions:
        def __init__(self):
            self.kwargs = None

        def create(self, **kwargs):
            self.kwargs = kwargs
            return response

    completions = Completions()
    fake_sdk = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    client = GLMClient(client=fake_sdk, model="glm-4.7-flash", max_retries=0)
    result = client.generate_json(system_prompt="system", user_prompt="user")
    assert completions.kwargs["response_format"] == {"type": "json_object"}
    assert completions.kwargs["thinking"] == {"type": "disabled"}
    assert result.total_tokens == 12
    assert "raw_text" not in result.trace_dict()

    monkeypatch.delenv("GLM_API_KEY", raising=False)
    with pytest.raises(MissingCredentialError, match="GLM_API_KEY"):
        GLMClient()


def _assert_safe_failure_trace(error):
    """Assert failed model calls preserve diagnostic evidence without admitting invalid candidates."""
    assert error.trace["provider"] == "fake"
    assert error.trace["model"] == "fake-model"
    assert error.trace["input_tokens"] == 10
    assert error.trace["output_tokens"] == 5
    assert error.trace["total_tokens"] == 15
    assert error.trace["latency_ms"] == 1
    assert error.trace["request_id"] == "req-1"
    assert error.trace["prompt_version"] == "glm-b1-v3-conditional"
    assert len(error.trace["prompt_hash"]) == 16
    assert len(error.trace["generator_spec_hash"]) == 16
    assert "raw_text" not in error.trace
    assert "api_key" not in json.dumps(error.trace).casefold()


def test_query_integrity_rejects_mojibake_before_any_provider_call():
    """Corrupted text must be rejected BEFORE the provider is called or the request archived."""
    import pytest
    from agent.generator_contract import (RequestValidationError, ResearchRequest,
                                          ensure_text_integrity)
    from agent.llm_generator import interpret_research_request

    garbled = "Ѱ���� 因子"          # the actual observed corruption shape
    with pytest.raises(RequestValidationError, match="corrupted"):
        ensure_text_integrity(garbled, "query")
    with pytest.raises(RequestValidationError, match="corrupted"):
        ensure_text_integrity("bad\x00query", "query")
    # clean Chinese/English must pass untouched
    assert ensure_text_integrity("\u5bfb\u627e\u4f4e\u5197\u4f59\u56e0\u5b50", "query")
    # from_dict freezes nothing garbled
    with pytest.raises(RequestValidationError, match="corrupted"):
        ResearchRequest.from_dict({"original_query": garbled, "universe": "u"})

    class MustNotBeCalled:
        provider = "test"
        model = "test"
        def generate_json(self, **kwargs):
            raise AssertionError("provider must not be called for a corrupted query")

    spec = _spec()
    with pytest.raises(RequestValidationError, match="corrupted"):
        interpret_research_request(query=garbled, client=MustNotBeCalled(), spec=spec,
                                   universe="u")


def test_spec_operator_restriction_and_ladder_enforcement():
    """Ladder arm A must not accept proposals using operators outside its frozen set."""
    import json, pytest
    from agent.generator_contract import (GeneratorSpec, ResearchRequest,
                                          expression_functions)
    from agent.llm_generator import LLMGenerator, ProposalConstraintError

    assert expression_functions("rank(ts_corr(a, b, 6))") == {"rank", "ts_corr"}
    cfg = {"factor_cols": ["a", "b"],
           "family_constraints": {"factor_families": {"a": "momentum", "b": "momentum"}}}
    legacy = GeneratorSpec.from_config(cfg, functions=["rank", "ts_mean", "zscore"])
    assert set(legacy.allowed_functions) == {"rank", "ts_mean", "zscore"}
    assert set(legacy.function_signatures) == {"rank", "ts_mean", "zscore"}
    with pytest.raises(ValueError, match="unknown DSL functions"):
        GeneratorSpec.from_config(cfg, functions=["rank", "nosuch"])

    class FakeClient:
        provider = "test"
        model = "test"
        def __init__(self, payload):
            self.payload = payload
        def generate_json(self, **kwargs):
            from agent.llm_generator import LLMCallResult
            return LLMCallResult(raw_text=json.dumps(self.payload), provider="test",
                                 model="test", input_tokens=1, output_tokens=1,
                                 total_tokens=2, latency_ms=1, request_id="r",
                                 parameters={})

    request = ResearchRequest.from_text("momentum test", universe="u",
                                        families=["momentum"])
    bad = {"expression": "rank(ts_corr(a, b, 6))", "rationale": "interaction hypothesis"}
    gen = LLMGenerator(client=FakeClient(bad), spec=legacy)
    with pytest.raises(ProposalConstraintError, match="outside the frozen GeneratorSpec"):
        gen.propose(request, history=[])
    full = GeneratorSpec.from_config(cfg)          # all operators incl. ts_corr
    ok = LLMGenerator(client=FakeClient(bad), spec=full).propose(request, history=[])
    assert ok["columns_used"] == ["a", "b"]
    assert ok["generation"]["prompt_version"] == "glm-b1-v3-conditional"


def test_mechanism_mode_requires_cross_family_columns():
    import json, pytest
    from agent.generator_contract import GeneratorSpec, ResearchRequest, MECHANISM_RECIPES
    from agent.llm_generator import LLMGenerator, LLMCallResult, ProposalConstraintError
    assert len(MECHANISM_RECIPES) == 3

    class FakeClient:
        provider = "test"; model = "test"
        def __init__(self, payload): self.payload = payload
        def generate_json(self, **kw):
            return LLMCallResult(raw_text=json.dumps(self.payload), provider="test",
                                 model="test", input_tokens=1, output_tokens=1,
                                 total_tokens=2, latency_ms=1, request_id="r", parameters={})

    cfg = {"factor_cols": ["m1", "t1"],
           "family_constraints": {"factor_families": {"m1": "momentum", "t1": "liquidity"}}}
    spec = GeneratorSpec.from_config(cfg)
    req = ResearchRequest.from_text("mechanism test", universe="u",
                                    families=["momentum", "liquidity"])
    single = {"expression": "rank(ts_mean(m1, 6))", "rationale": "single family"}
    with pytest.raises(ProposalConstraintError, match="two distinct families"):
        LLMGenerator(client=FakeClient(single), spec=spec,
                     mechanism_mode=True).propose(req, history=[])
    cross = {"expression": "zscore(m1) * sign(delta(t1, 3))", "rationale": "conditional"}
    out = LLMGenerator(client=FakeClient(cross), spec=spec,
                       mechanism_mode=True, prompt_version="glm-b1-v4-mechanism"
                       ).propose(req, history=[])
    assert sorted(out["family_exposures"]) == ["liquidity", "momentum"]
    # non-mechanism mode unaffected
    ok = LLMGenerator(client=FakeClient(single), spec=spec).propose(req, history=[])
    assert ok["family"] == "momentum"
