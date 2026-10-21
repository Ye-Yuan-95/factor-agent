"""Provider-neutral LLM proposal loop plus the GLM-4.7-Flash adapter.

The model proposes only ``expression`` and ``rationale``.  Names, family assignment, budgets,
DSL legality, duplicate detection and every statistical decision remain deterministic host code.
The GLM SDK import is lazy so the core package and test suite do not require cloud dependencies.

Use the validated candidate workflow through main.py run --generator glm.
This module supplies provider/request/proposal helpers rather than a separate CLI.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Protocol

from agent.factor_dsl import DSLError, compile_expression
from agent.generator_contract import (
    GeneratorSpec,
    ResearchRequest,
    assign_expression_family,
    expression_functions,
    sanitize_generator_history,
)


_PREFERENCE_FIELDS = {"low_redundancy", "low_turnover", "validation_stability"}


class GenerationError(RuntimeError):
    """Base class for bounded, traceable generation failures."""

    def __init__(self, message: str, *, trace: dict[str, Any] | None = None):
        super().__init__(message)
        self.trace = dict(trace or {})

    def attach_trace(self, trace: dict[str, Any]) -> "GenerationError":
        """Attach provider-safe provenance after a successful call and before re-raising.

        ``trace`` comes from :meth:`LLMCallResult.trace_dict`, which deliberately excludes the
        raw response. Existing fields win so a more specific adapter error cannot be overwritten.
        """
        self.trace = {**trace, **self.trace}
        return self


class MissingCredentialError(GenerationError):
    """The selected provider has no API credential in the process environment."""


class ProviderCallError(GenerationError):
    """The provider call failed or returned a response that could not be read."""


class ProposalFormatError(GenerationError):
    """The response does not match the required JSON schema."""


class ProposalConstraintError(GenerationError):
    """A schema-valid proposal violates the frozen request or GeneratorSpec."""


class DuplicateProposalError(GenerationError):
    """The expression has already been proposed in this run/history."""


@dataclass(frozen=True)
class CandidateProposal:
    expression: str
    rationale: str
    recipe_name: str | None = None

    @classmethod
    def from_json(cls, raw_text: str, *, require_recipe_name: bool = False
                  ) -> "CandidateProposal":
        """Parse only the allowed proposal schema; malformed or extra model-controlled fields are rejected."""
        try:
            raw = json.loads(raw_text)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ProposalFormatError(f"proposal is not valid JSON: {exc}") from exc
        if not isinstance(raw, dict):
            raise ProposalFormatError("proposal must be a JSON object")
        expected = {"expression", "rationale"}
        if require_recipe_name:
            expected = expected | {"recipe_name"}
        if set(raw) != expected:
            raise ProposalFormatError(
                f"proposal fields must be exactly {sorted(expected)}; got {sorted(raw)}")
        recipe_name = None
        if require_recipe_name:
            recipe_name = raw["recipe_name"]
            if not isinstance(recipe_name, str) or not recipe_name.strip():
                raise ProposalFormatError("recipe_name must be a non-empty string")
            recipe_name = recipe_name.strip()
        expression = raw["expression"]
        rationale = raw["rationale"]
        if not isinstance(expression, str) or not expression.strip():
            raise ProposalFormatError("expression must be a non-empty string")
        if len(expression) > 400:
            raise ProposalFormatError("expression exceeds 400 characters")
        if not isinstance(rationale, str) or not rationale.strip():
            raise ProposalFormatError("rationale must be a non-empty string")
        if len(rationale) > 800:
            raise ProposalFormatError("rationale exceeds 800 characters")
        return cls(expression=expression.strip(), rationale=rationale.strip(),
                   recipe_name=recipe_name)


@dataclass(frozen=True)
class ResearchIntent:
    objective: str
    allowed_families: tuple[str, ...]
    excluded_families: tuple[str, ...]
    preferences: dict[str, bool]
    assumptions: tuple[str, ...]
    unsupported_requirements: tuple[str, ...]

    @classmethod
    def from_json(cls, raw_text: str, *, known_families: set[str]) -> "ResearchIntent":
        """Validate structured research intent against known families before freezing the request."""
        try:
            raw = json.loads(raw_text)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ProposalFormatError(f"research intent is not valid JSON: {exc}") from exc
        if not isinstance(raw, dict):
            raise ProposalFormatError("research intent must be a JSON object")
        expected = {
            "objective", "allowed_families", "excluded_families", "preferences",
            "assumptions", "unsupported_requirements",
        }
        if set(raw) != expected:
            raise ProposalFormatError(
                f"research-intent fields must be exactly {sorted(expected)}; got {sorted(raw)}")
        objective = _non_empty_text(raw["objective"], "objective", max_length=800)
        allowed = _string_list(raw["allowed_families"], "allowed_families")
        excluded = _string_list(raw["excluded_families"], "excluded_families")
        unknown = sorted((set(allowed) | set(excluded)) - known_families)
        if unknown:
            raise ProposalFormatError(f"research intent contains unknown families: {unknown}")
        overlap = sorted(set(allowed) & set(excluded))
        if overlap:
            raise ProposalFormatError(f"families cannot be allowed and excluded: {overlap}")
        preferences = raw["preferences"]
        if not isinstance(preferences, dict):
            raise ProposalFormatError("preferences must be an object")
        extra_preferences = sorted(set(preferences) - _PREFERENCE_FIELDS)
        if extra_preferences:
            raise ProposalFormatError(f"unsupported preference fields: {extra_preferences}")
        if any(type(value) is not bool for value in preferences.values()):
            raise ProposalFormatError("preference values must be booleans")
        normalized_preferences = {key: bool(preferences.get(key, False))
                                  for key in sorted(_PREFERENCE_FIELDS)}
        return cls(
            objective=objective,
            allowed_families=allowed,
            excluded_families=excluded,
            preferences=normalized_preferences,
            assumptions=_string_list(raw["assumptions"], "assumptions"),
            unsupported_requirements=_string_list(
                raw["unsupported_requirements"], "unsupported_requirements"),
        )


@dataclass(frozen=True)
class LLMCallResult:
    raw_text: str
    provider: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    latency_ms: int
    request_id: str | None
    parameters: dict[str, Any]

    def trace_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("raw_text", None)
        return data


class JSONLLMClient(Protocol):
    provider: str
    model: str

    def generate_json(self, *, system_prompt: str, user_prompt: str) -> LLMCallResult:
        ...


class GLMClient:
    """Small GLM adapter.  It never reads/writes research data and never logs the API key."""

    provider = "zhipu"

    def __init__(self, *, api_key: str | None = None, model: str = "glm-4.7-flash",
                 max_tokens: int = 400, max_retries: int = 2,
                 temperature: float = 0.7, top_p: float = 0.95,
                 client: Any | None = None, sleep: Callable[[float], None] = time.sleep):
        """Validate call limits and initialize either an injected test client or the credentialed provider."""
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if max_retries < 0 or max_retries > 5:
            raise ValueError("max_retries must be in [0, 5]")
        self.model = model
        self.max_tokens = int(max_tokens)
        self.max_retries = int(max_retries)
        self.temperature = float(temperature)   # frozen + recorded
        self.top_p = float(top_p)
        self._sleep = sleep
        if client is not None:
            self._client = client
            return
        key = api_key or os.environ.get("GLM_API_KEY")
        if not key:
            raise MissingCredentialError(
                "GLM_API_KEY is not set; set it in the current shell before using GLM")
        try:
            from zai import ZhipuAiClient
        except ImportError as exc:
            raise MissingCredentialError(
                "zai-sdk is not installed; run: pip install -e \".[llm-glm]\"") from exc
        self._client = ZhipuAiClient(api_key=key)

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "response_format": "json_object",
            "thinking": "disabled",
            "max_tokens": self.max_tokens,
            "max_retries": self.max_retries,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }

    def generate_json(self, *, system_prompt: str, user_prompt: str) -> LLMCallResult:
        """Call the provider with bounded retries and record token/latency metadata.
        Errors remain explicit; an exhausted call is not converted into a fabricated proposal."""
        last_error: Exception | None = None
        for retry in range(self.max_retries + 1):
            started = time.perf_counter()
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    thinking={"type": "disabled"},
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                )
                latency_ms = round((time.perf_counter() - started) * 1000)
                raw_text = response.choices[0].message.content
                if not isinstance(raw_text, str) or not raw_text.strip():
                    raise ProviderCallError("GLM returned empty or non-text content")
                usage = getattr(response, "usage", None)
                return LLMCallResult(
                    raw_text=raw_text,
                    provider=self.provider,
                    model=self.model,
                    input_tokens=_usage_value(usage, "prompt_tokens", "input_tokens"),
                    output_tokens=_usage_value(usage, "completion_tokens", "output_tokens"),
                    total_tokens=_usage_value(usage, "total_tokens"),
                    latency_ms=latency_ms,
                    request_id=_optional_text(getattr(response, "id", None)),
                    parameters=self.parameters,
                )
            except Exception as exc:  # SDK exception types are optional/lazy provider details
                last_error = exc
                if retry >= self.max_retries or not _is_retryable_provider_error(exc):
                    break
                self._sleep(min(2 ** retry, 4))
        if isinstance(last_error, GenerationError):
            raise last_error
        status = getattr(last_error, "status_code", None)
        suffix = f" (status={status})" if status is not None else ""
        raise ProviderCallError(
            f"GLM call failed after bounded retries: {type(last_error).__name__}{suffix}") \
            from last_error


def interpret_research_request(*, query: str, client: JSONLLMClient, spec: GeneratorSpec,
                               universe: str, existing_factors: list[str] | tuple[str, ...] = (),
                               hard_allowed_families: list[str] | tuple[str, ...] = (),
                               generation_attempt_budget: int = 24,
                               oracle_trial_budget: int = 12
                               ) -> tuple[ResearchRequest, LLMCallResult]:
    """Parse only supported intent fields; data scope, existing set and budgets stay host-owned."""
    query = query.strip()
    if not query:
        raise ValueError("query must be non-empty")
    # fail BEFORE any provider call: a mojibake query must not be interpreted or archived
    from agent.generator_contract import ensure_text_integrity
    ensure_text_integrity(query, "query")
    known_families = set(spec.column_families.values())
    hard_families = tuple(dict.fromkeys(str(x).strip() for x in hard_allowed_families if str(x).strip()))
    unknown_hard = sorted(set(hard_families) - known_families)
    if unknown_hard:
        raise ValueError(f"--families contains unknown families: {unknown_hard}")
    system_prompt = (
        "你是量化因子研究请求解析器。只把用户意图映射到给定 schema，不生成因子，"
        "不增加用户没说过的约束。只输出一个 JSON 对象，字段必须恰好为："
        "objective, allowed_families, excluded_families, preferences, assumptions, "
        "unsupported_requirements。preferences 必须恰好使用 low_redundancy、low_turnover、"
        "validation_stability 三个布尔字段。因子家族只能来自给定列表。"
    )
    user_prompt = json.dumps({
        "user_query": query,
        "known_families": sorted(known_families),
        "hard_allowed_families": list(hard_families),
        "instructions": (
            "若 hard_allowed_families 非空，将其原样放入 allowed_families。"
            "无法映射的要求写入 unsupported_requirements；解释性假设写入 assumptions。"
        ),
        "output_example": {
            "objective": "寻找低冗余、验证期稳健的价值因子",
            "allowed_families": ["value"],
            "excluded_families": [],
            "preferences": {
                "low_redundancy": True,
                "low_turnover": False,
                "validation_stability": True,
            },
            "assumptions": [],
            "unsupported_requirements": [],
        },
    }, ensure_ascii=False, sort_keys=True)
    call = client.generate_json(system_prompt=system_prompt, user_prompt=user_prompt)
    intent = ResearchIntent.from_json(call.raw_text, known_families=known_families)
    allowed = hard_families or intent.allowed_families
    excluded = tuple(f for f in intent.excluded_families if f not in allowed)
    request = ResearchRequest.from_dict({
        "original_query": query,
        "objective": intent.objective,
        "universe": universe,
        "allowed_families": list(allowed),
        "excluded_families": list(excluded),
        "preferences": intent.preferences,
        "existing_factors": list(existing_factors),
        "generation_attempt_budget": generation_attempt_budget,
        "oracle_trial_budget": oracle_trial_budget,
        "assumptions": list(intent.assumptions),
        "unsupported_requirements": list(intent.unsupported_requirements),
    })
    return request, call


class LLMGenerator:
    """Propose one expression at a time; the runner owns attempts, trials and names.

recipe_prompt shows structural recipes and requires a recipe_name declaration;
the host records AST conformance. cross_family_required independently enforces
at least two factor families and announces that constraint in the prompt.
mechanism_mode enables recipe display without a recipe_name requirement and
enforces cross-family use for protocols configured with that combined switch."""

    def __init__(self, *, client: JSONLLMClient, spec: GeneratorSpec,
                 prompt_version: str = "glm-b1-v3-conditional",
                 mechanism_mode: bool = False,
                 recipe_prompt: bool | None = None,
                 cross_family_required: bool | None = None):
        """Bind the provider and frozen vocabulary, including optional mechanism recipe constraints."""
        self.client = client
        self.spec = spec
        self.prompt_version = prompt_version
        legacy = bool(mechanism_mode)
        self.recipe_prompt = bool(recipe_prompt) if recipe_prompt is not None else False
        self.cross_family_required = bool(
            legacy if cross_family_required is None else cross_family_required)
        self.show_recipes = self.recipe_prompt or legacy
        self._seen_signatures: set[str] = set()

    def propose(self, request: ResearchRequest, history: list[dict[str, Any]] | None = None,
                *, extra_payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """``extra_payload`` is merged into the user prompt JSON. It exists for host-side
        exploration policies (agent/hybrid_generator.py) that need to tell the model which
        region to work in and what has already been covered. It is prompt content only: it
        cannot relax any constraint checked below, and it goes through the same integrity
        check as the rest of the prompt."""
        safe_history = sanitize_generator_history(history)
        history_signatures = {
            str(item["expression_signature"])
            for item in safe_history if item.get("expression_signature")
        }
        system_prompt, user_prompt = self._build_prompts(request, safe_history,
                                                         extra_payload=extra_payload)
        call = self.client.generate_json(system_prompt=system_prompt, user_prompt=user_prompt)
        prompt_hash = hashlib.sha256(
            (system_prompt + "\n" + user_prompt).encode("utf-8")).hexdigest()[:16]
        generation_trace = {
            **call.trace_dict(),
            "prompt_version": self.prompt_version,
            "prompt_hash": prompt_hash,
            "generator_spec_hash": self.spec.fingerprint(),
        }
        recipe_conformant: bool | None = None
        recipe_violation: str | None = None
        try:
            proposal = CandidateProposal.from_json(
                call.raw_text, require_recipe_name=self.recipe_prompt)
            try:
                compiled = compile_expression(proposal.expression)
                assignment = assign_expression_family(proposal.expression, self.spec)
            except (DSLError, ValueError) as exc:
                raise ProposalConstraintError(str(exc)) from exc
            signature = compiled.signature()
            if signature in self._seen_signatures or signature in history_signatures:
                raise DuplicateProposalError("proposal duplicates an expression already tried")
            exposures = set(assignment.exposures)
            if request.allowed_families and not exposures.issubset(request.allowed_families):
                raise ProposalConstraintError(
                    f"proposal family exposures {sorted(exposures)} are outside allowed families "
                    f"{list(request.allowed_families)}")
            forbidden = exposures & set(request.excluded_families)
            if forbidden:
                raise ProposalConstraintError(
                    f"proposal uses excluded family exposures: {sorted(forbidden)}")
            if self.cross_family_required and len(set(assignment.exposures)) < 2:
                raise ProposalConstraintError(
                    "cross-family constraint requires columns from at least two distinct "
                    f"families (got exposures {sorted(set(assignment.exposures))})")
            illegal = sorted(expression_functions(proposal.expression)
                             - set(self.spec.allowed_functions))
            if illegal:
                raise ProposalConstraintError(
                    f"proposal uses functions outside the frozen GeneratorSpec: {illegal}")
            if self.recipe_prompt:
                from agent.generator_contract import RECIPE_NAMES, check_recipe_conformance
                if proposal.recipe_name not in RECIPE_NAMES:
                    raise ProposalConstraintError(
                        f"recipe_name {proposal.recipe_name!r} is not one of {list(RECIPE_NAMES)}")
                # conformance is a MEASURED outcome, not a filter: blocking here would
                # conflate discovery power with recipe adherence in the factorial readout
                recipe_conformant, recipe_violation = check_recipe_conformance(
                    proposal.expression, proposal.recipe_name, self.spec)
        except GenerationError as exc:
            raise exc.attach_trace(generation_trace)
        self._seen_signatures.add(signature)
        return {
            "expression": proposal.expression,
            "rationale": proposal.rationale,
            "columns_used": sorted(compiled.columns),
            "family": assignment.primary,
            "family_exposures": list(assignment.exposures),
            "expression_signature": signature,
            "recipe_name": proposal.recipe_name,
            "recipe_conformant": recipe_conformant,
            "recipe_violation": recipe_violation,
            "generation": generation_trace,
        }

    def _build_prompts(self, request: ResearchRequest,
                       safe_history: list[dict[str, Any]],
                       *, extra_payload: dict[str, Any] | None = None) -> tuple[str, str]:
        """Build system/user prompts from host-owned constraints and sanitized history.

        Recipe display, required recipe declarations and cross-family constraints
        are separate switches. Prompt text describes permitted hypotheses; actual
        expression legality is checked by the host after the provider responds.
        """
        fields_text = ("expression、rationale 和 recipe_name" if self.recipe_prompt
                       else "expression 和 rationale")
        system_prompt = (
            "你是量化因子假设生成器，不是验证器。只输出一个 JSON 对象，且字段必须恰好为 "
            f"{fields_text}。不要输出 name、family、decision、IC、Sharpe、阈值或任何"
            "测试期信息。expression 必须严格使用 GeneratorSpec 中的列、函数、窗口和运算符。"
            "每个函数必须严格遵守 GeneratorSpec.function_signatures 中的参数个数和位置。"
            "不得声称因子有效；rationale 只描述可证伪的经济或统计假设。"
        )
        if self.show_recipes:
            system_prompt += "必须从 mechanism_recipes 中选择一个机制模板，"
            system_prompt += ("并在 recipe_name 字段填入所选模板的 name。"
                              if self.recipe_prompt else "")
        if self.cross_family_required:
            system_prompt += "表达式必须引用至少两个不同家族的列。"
        system_prompt += (
            "硬性要求：假设必须相对表达式成分列"
            "具有非线性、条件性或交互性质的增量信息；"
            "禁止只对单一列做平滑、标准化或线性重组；"
            "只能使用 function_signatures 中列出的函数。"
        )
        from collections import Counter
        usage: Counter = Counter()
        near_misses = []
        for item in safe_history:
            for col in item.get("columns_used") or []:
                usage[col] += 1
            if item.get("failed_gates") == ["incremental"]:
                near_misses.append({
                    "expression_signature": item.get("expression_signature"),
                    "columns_used": item.get("columns_used") or [],
                })
        reserved = {"research_request", "generator_spec", "safe_history", "mechanism_recipes",
                    "recent_column_usage", "incremental_near_misses", "instruction",
                    "output_example"}
        clash = sorted(reserved & set(extra_payload or {}))
        if clash:
            raise ValueError(f"extra_payload may not override prompt fields: {clash}")
        cross_family_example = self.show_recipes or self.cross_family_required
        output_example: dict[str, str] = {
            "expression": ("zscore(mom1m) * sign(delta(turn, 3))"
                           if cross_family_example else "rank(ts_mean(roa, 6))"),
            "rationale": ("高换手环境下的短期反转可能比无条件动量携带独立信息"
                          if cross_family_example
                          else "持续盈利能力可能比单月盈利水平提供更稳定的横截面信号"),
        }
        if self.recipe_prompt:
            output_example["recipe_name"] = "conditional_signal"
        user_prompt = json.dumps({
            "research_request": request.to_dict(),
            "generator_spec": self.spec.to_dict(),
            "safe_history": safe_history,
            "mechanism_recipes": (list(__import__("agent.generator_contract", fromlist=["MECHANISM_RECIPES"]).MECHANISM_RECIPES)
                                  if self.show_recipes else []),
            "recent_column_usage": dict(usage.most_common(10)),
            "incremental_near_misses": near_misses[-5:],
            "instruction": "提出一个新的、未重复、满足研究请求的候选因子。优先使用 recent_column_usage 中出现次数少或未出现的列；若 incremental_near_misses 非空，针对其成分列提出非线性/条件性新假设。",
            "output_example": output_example,
            **(extra_payload or {}),
        }, ensure_ascii=False, sort_keys=True)
        from agent.generator_contract import ensure_text_integrity
        ensure_text_integrity(system_prompt, "system_prompt")
        ensure_text_integrity(user_prompt, "user_prompt")
        return system_prompt, user_prompt


def _non_empty_text(value: Any, name: str, *, max_length: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProposalFormatError(f"{name} must be a non-empty string")
    text = value.strip()
    if len(text) > max_length:
        raise ProposalFormatError(f"{name} exceeds {max_length} characters")
    return text


def _string_list(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ProposalFormatError(f"{name} must be a list of strings")
    cleaned = tuple(item.strip() for item in value)
    if any(not item for item in cleaned):
        raise ProposalFormatError(f"{name} cannot contain empty strings")
    return tuple(dict.fromkeys(cleaned))


def _usage_value(usage: Any, *names: str) -> int | None:
    for name in names:
        if isinstance(usage, dict) and usage.get(name) is not None:
            return int(usage[name])
        value = getattr(usage, name, None)
        if value is not None:
            return int(value)
    return None


def _optional_text(value: Any) -> str | None:
    return str(value) if value is not None else None


def _is_retryable_provider_error(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status == 429 or (isinstance(status, int) and status >= 500):
        return True
    name = type(exc).__name__.casefold()
    return "timeout" in name or "connection" in name
