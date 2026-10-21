"""Provider-independent contracts for factor generators.

This module deliberately has no LLM SDK dependency.  Random/adaptive baselines and cloud/local
LLM providers consume the same :class:`GeneratorSpec`, research request and feedback projection.
The projection is fail-closed: if a caller accidentally passes a test/lockbox field anywhere in
history, generation stops instead of silently dropping the leaked field.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

from agent.factor_dsl import (
    FUNCTIONS,
    MAX_AST_NODES,
    MAX_DEPTH,
    MAX_EXPRESSION_CHARS,
    MAX_WINDOW,
    column_is_allowed,
    compile_expression,
)


DEFAULT_WINDOWS = (3, 6, 12)
DEFAULT_BINARY_OPERATORS = ("+", "-", "*", "/")

# Extended vocabulary: point-in-time RAW panel columns the DSL/oracle
# already support but the frozen base vocabulary deliberately excluded. Extending the
# generator's leaves does NOT touch gate code -- the certificate stays bound -- but any run
# Freeze the selected vocabulary in the experiment protocol.
PIT_RAW_COLUMNS = ("close", "amount", "volume", "market_cap", "float_mktcap", "market_return")


# Mechanism recipe library: structural hypothesis templates
# that force information from MULTIPLE families into one candidate, so the result is not a
# transformation of a single ingredient column (the failure mode of every near-miss so far).
MECHANISM_RECIPES = (
    {"name": "conditional_signal",
     "form": "signal_transform(signal_col) * condition_transform(conditioner_col)",
     "rule": "signal_col and conditioner_col MUST come from different families",
     "example": "zscore(mom1m) * sign(delta(turn, 3))"},
    {"name": "regime_gated",
     "form": "signal only where conditioner is extreme",
     "rule": "use clip/sign/rank of a conditioner from another family as the gate",
     "example": "rank(mom3m) * clip(zscore(retvol), 0, 3)"},
    {"name": "cross_family_spread",
     "form": "normalized difference of two signals from different families",
     "rule": "both legs cross-sectionally normalized before differencing",
     "example": "zscore(mom6m) - zscore(ts_mean(turn, 6))"},
)
RECIPE_NAMES = tuple(recipe["name"] for recipe in MECHANISM_RECIPES)
_GATE_CALLS = {"clip", "sign"}
_CS_CALLS = {"rank", "zscore"}


def check_recipe_conformance(expression: str, recipe_name: str,
                             spec: "GeneratorSpec") -> tuple[bool, str | None]:
    """Deterministic AST check that an expression structurally matches its DECLARED recipe
    Non-conformance is RECORDED by callers, never silently accepted as recipe
    evidence. Rules are intentionally coarse but checkable:

      conditional_signal : top-level product A * B, each side referencing >=1 column,
                           family sets of the two sides disjoint;
      regime_gated       : top-level product where at least one side's outermost call is
                           clip/sign (the gate), families of the two sides disjoint;
      cross_family_spread: top-level difference with each leg's outermost call rank/zscore,
                           families of the two legs disjoint.
    """
    import ast as _ast
    if recipe_name not in RECIPE_NAMES:
        return False, f"unknown recipe {recipe_name!r}"
    try:
        tree = _ast.parse(expression.strip(), mode="eval")
    except SyntaxError:
        return False, "expression does not parse"

    def strip_wrappers(node: Any) -> Any:
        while True:
            if isinstance(node, _ast.UnaryOp):
                node = node.operand
                continue
            if isinstance(node, _ast.BinOp) and isinstance(node.op, _ast.Mult):
                if not _side_columns(node.left):
                    node = node.right
                    continue
                if not _side_columns(node.right):
                    node = node.left
                    continue
            return node

    def _side_columns(node: Any) -> set[str]:
        return {n.id for n in _ast.walk(node)
                if isinstance(n, _ast.Name) and n.id in spec.column_families}

    def _side_families(node: Any) -> set[str]:
        return {spec.column_families[c] for c in _side_columns(node)}

    def _outer_call(node: Any) -> str | None:
        while isinstance(node, _ast.UnaryOp):
            node = node.operand
        if isinstance(node, _ast.Call) and isinstance(node.func, _ast.Name):
            return node.func.id
        return None

    node = strip_wrappers(tree.body)
    if recipe_name in ("conditional_signal", "regime_gated"):
        if not (isinstance(node, _ast.BinOp) and isinstance(node.op, _ast.Mult)):
            return False, f"{recipe_name} requires a top-level product of two data terms"
        left_families, right_families = _side_families(node.left), _side_families(node.right)
        if not left_families or not right_families:
            return False, "both product terms must reference at least one data column"
        shared = sorted(left_families & right_families)
        if shared:
            return False, f"signal and conditioner share families: {shared}"
        if recipe_name == "regime_gated" and not (
                _outer_call(node.left) in _GATE_CALLS or _outer_call(node.right) in _GATE_CALLS):
            return False, "regime_gated requires a clip/sign gate as one product term"
        return True, None
    if not (isinstance(node, _ast.BinOp) and isinstance(node.op, _ast.Sub)):
        return False, "cross_family_spread requires a top-level difference of two legs"
    if _outer_call(node.left) not in _CS_CALLS or _outer_call(node.right) not in _CS_CALLS:
        return False, "both legs must be wrapped in rank/zscore before differencing"
    left_families, right_families = _side_families(node.left), _side_families(node.right)
    if not left_families or not right_families:
        return False, "both legs must reference at least one data column"
    shared = sorted(left_families & right_families)
    if shared:
        return False, f"the two legs share families: {shared}"
    return True, None


def build_vocabulary(config_raw: dict[str, Any], *, mode: str = "base",
                     panel_columns: list[str] | tuple[str, ...] | None = None) -> list[str]:
    """Leaf columns for a GeneratorSpec. 'base' = the frozen 44 factor columns (unchanged
    behaviour); 'extended' additionally offers the PIT raw columns, intersected with the
    ACTUAL panel columns so synthetic panels (which carry no raw columns) never advertise
    leaves the oracle cannot evaluate."""
    columns = [str(c) for c in config_raw.get("factor_cols", [])]
    if mode == "base":
        return columns
    if mode == "extended":
        available = {str(c) for c in (panel_columns or ())}
        extra = [c for c in PIT_RAW_COLUMNS if not available or c in available]
        return columns + [c for c in extra if c not in columns]
    raise ValueError(f"unknown vocabulary mode {mode!r} (base|extended)")
FUNCTION_SIGNATURES = {
    "rank": "rank(series)",
    "zscore": "zscore(series)",
    "demean": "demean(series)",
    "industry_demean": "industry_demean(series)",
    "ts_corr": "ts_corr(series, series, window)",
    "winsor": "winsor(series[, fraction])",
    "ts_mean": "ts_mean(series, window)",
    "ts_std": "ts_std(series, window)",
    "ts_sum": "ts_sum(series, window)",
    "ts_min": "ts_min(series, window)",
    "ts_max": "ts_max(series, window)",
    "ts_rank": "ts_rank(series, window)",
    "ts_zscore": "ts_zscore(series, window)",
    "delay": "delay(series, positive_lag)",
    "delta": "delta(series, positive_lag)",
    "log": "log(series)",
    "sqrt": "sqrt(series)",
    "abs": "abs(series)",
    "sign": "sign(series)",
    "clip": "clip(series, lower, upper)",
}
_REQUEST_FIELDS = {
    "original_query", "objective", "universe", "allowed_families", "excluded_families",
    "preferences", "existing_factors", "generation_attempt_budget", "oracle_trial_budget",
    "assumptions", "unsupported_requirements",
}
_FEEDBACK_FIELDS = {
    "name", "expression_signature", "outcome", "failed_gates", "reject_reason",
    "columns_used", "recipe_name",
}


class RequestValidationError(ValueError):
    """The structured research request is invalid or contains unsupported fields."""


class FeedbackLeakageError(ValueError):
    """History contained a test/lockbox field and therefore cannot be sent to a generator."""


def ensure_text_integrity(text: str, name: str) -> str:
    """Reject corrupted text before provider calls or frozen audit records.

    Corruption includes U+FFFD, control characters other than tabs/newlines,
    surrogates and unassigned/private-use codepoints. Fail before interpreting intent."""
    import unicodedata
    bad: list[str] = []
    for ch in text:
        if ch == "�":
            bad.append("U+FFFD replacement char")
        elif ch in ("\t", "\n", "\r"):
            continue
        elif unicodedata.category(ch) in ("Cc", "Cs", "Co", "Cn"):
            bad.append(f"U+{ord(ch):04X} ({unicodedata.category(ch)})")
    if bad:
        uniq = sorted(set(bad))[:5]
        raise RequestValidationError(
            f"{name} contains corrupted characters ({', '.join(uniq)}) -- this usually means the "
            "console encoding mangled the input. Run `chcp 65001` in PowerShell (or use an "
            "English query) and retry; refusing to freeze/interpret a garbled request.")
    return text


@dataclass(frozen=True)
class ResearchRequest:
    original_query: str
    objective: str
    universe: str
    allowed_families: tuple[str, ...] = ()
    excluded_families: tuple[str, ...] = ()
    preferences: dict[str, Any] = field(default_factory=dict)
    existing_factors: tuple[str, ...] = ()
    generation_attempt_budget: int = 24
    oracle_trial_budget: int = 12
    assumptions: tuple[str, ...] = ()
    unsupported_requirements: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ResearchRequest":
        """Validate user-supplied scope and budgets, rejecting unknown or unsupported request fields."""
        if not isinstance(raw, dict):
            raise RequestValidationError("research request must be a JSON object")
        extra = sorted(set(raw) - _REQUEST_FIELDS)
        if extra:
            raise RequestValidationError(f"unsupported research-request fields: {extra}")
        original = str(raw.get("original_query", "")).strip()
        if not original:
            raise RequestValidationError("original_query must be non-empty")
        ensure_text_integrity(original, "original_query")
        objective = str(raw.get("objective") or original).strip()
        ensure_text_integrity(objective, "objective")
        universe = str(raw.get("universe", "")).strip()
        if not universe:
            raise RequestValidationError("universe must be non-empty")
        generation_budget = _positive_int(raw.get("generation_attempt_budget", 24),
                                          "generation_attempt_budget")
        oracle_budget = _positive_int(raw.get("oracle_trial_budget", 12),
                                      "oracle_trial_budget")
        if generation_budget < oracle_budget:
            raise RequestValidationError(
                "generation_attempt_budget must be >= oracle_trial_budget")
        preferences = raw.get("preferences", {})
        if not isinstance(preferences, dict):
            raise RequestValidationError("preferences must be an object")
        allowed = _string_tuple(raw.get("allowed_families", ()), "allowed_families")
        excluded = _string_tuple(raw.get("excluded_families", ()), "excluded_families")
        overlap = sorted(set(allowed) & set(excluded))
        if overlap:
            raise RequestValidationError(f"families cannot be both allowed and excluded: {overlap}")
        return cls(
            original_query=original,
            objective=objective,
            universe=universe,
            allowed_families=allowed,
            excluded_families=excluded,
            preferences=dict(preferences),
            existing_factors=_string_tuple(raw.get("existing_factors", ()), "existing_factors"),
            generation_attempt_budget=generation_budget,
            oracle_trial_budget=oracle_budget,
            assumptions=_string_tuple(raw.get("assumptions", ()), "assumptions"),
            unsupported_requirements=_string_tuple(
                raw.get("unsupported_requirements", ()), "unsupported_requirements"),
        )

    @classmethod
    def from_text(cls, text: str, *, universe: str, families: list[str] | tuple[str, ...] = (),
                  existing_factors: list[str] | tuple[str, ...] = (),
                  generation_attempt_budget: int = 24,
                  oracle_trial_budget: int = 12) -> "ResearchRequest":
        """Preserve the research query and use explicitly supplied constraints.

        Family, universe, existing factors and budgets come from the caller's configuration;
        this local parser does not infer additional preferences from prose."""
        return cls.from_dict({
            "original_query": text,
            "objective": text,
            "universe": universe,
            "allowed_families": list(families),
            "existing_factors": list(existing_factors),
            "generation_attempt_budget": generation_attempt_budget,
            "oracle_trial_budget": oracle_trial_budget,
        })

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        for key in ("allowed_families", "excluded_families", "existing_factors",
                    "assumptions", "unsupported_requirements"):
            out[key] = list(out[key])
        return out


@dataclass(frozen=True)
class GeneratorSpec:
    allowed_columns: tuple[str, ...]
    column_families: dict[str, str]
    allowed_functions: tuple[str, ...] = tuple(sorted(FUNCTIONS))
    function_signatures: dict[str, str] = field(
        default_factory=lambda: dict(FUNCTION_SIGNATURES))
    windows: tuple[int, ...] = DEFAULT_WINDOWS
    binary_operators: tuple[str, ...] = DEFAULT_BINARY_OPERATORS
    max_expression_chars: int = MAX_EXPRESSION_CHARS
    max_ast_nodes: int = MAX_AST_NODES
    max_depth: int = MAX_DEPTH
    max_window: int = MAX_WINDOW
    clip_bounds: tuple[float, float] = (-3.0, 3.0)

    @classmethod
    def from_config(cls, config_raw: dict[str, Any], *, columns: list[str] | None = None,
                    windows: tuple[int, ...] = DEFAULT_WINDOWS,
                    functions: list[str] | tuple[str, ...] | None = None) -> "GeneratorSpec":
        """Build an immutable allowed vocabulary and function specification from configuration."""
        configured = columns if columns is not None else config_raw.get("factor_cols", [])
        allowed_columns = tuple(sorted({str(c) for c in configured if column_is_allowed(str(c))}))
        if not allowed_columns:
            raise ValueError("GeneratorSpec has no allowed columns")
        families = config_raw.get("family_constraints", {}).get("factor_families", {})
        clean_families = {c: str(families.get(c, "unclassified")) for c in allowed_columns}
        checked_windows = tuple(sorted({_window(w) for w in windows}))
        if functions is None:
            allowed_fns = tuple(sorted(FUNCTIONS))
        else:
            unknown = sorted(set(functions) - set(FUNCTIONS))
            if unknown:
                raise ValueError(f"unknown DSL functions in spec: {unknown}")
            allowed_fns = tuple(sorted(set(str(f) for f in functions)))
        signatures = {name: FUNCTION_SIGNATURES[name] for name in allowed_fns}
        return cls(allowed_columns=allowed_columns, column_families=clean_families,
                   allowed_functions=allowed_fns, function_signatures=signatures,
                   windows=checked_windows)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        for key in ("allowed_columns", "allowed_functions", "windows", "binary_operators",
                    "clip_bounds"):
            out[key] = list(out[key])
        return out

    def fingerprint(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False,
                               separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class FamilyAssignment:
    primary: str
    exposures: tuple[str, ...]


def assign_expression_family(expression: str, spec: GeneratorSpec) -> FamilyAssignment:
    """Assign the primary family used by the oracle's single-family cap.

    Count distinct referenced columns per family; choose the largest count with
    lexicographic ties. Preserve all exposures for reporting. Other exposure families
    are descriptive fields and do not each impose a separate cap."""
    compiled = compile_expression(expression)
    unknown = sorted(set(compiled.columns) - set(spec.allowed_columns))
    if unknown:
        raise ValueError(f"expression references columns outside GeneratorSpec: {unknown}")
    counts = Counter(spec.column_families.get(c, "unclassified") for c in compiled.columns)
    ranked = sorted(counts, key=lambda family: (-counts[family], family))
    primary = ranked[0] if ranked else "unclassified"
    return FamilyAssignment(primary=primary, exposures=tuple(sorted(counts)))


def expression_functions(expression: str) -> set[str]:
    """DSL function names called by an expression (ast-based; used to enforce a spec's
    allowed_functions subset without touching the fingerprinted compiler)."""
    import ast as _ast
    try:
        tree = _ast.parse(expression.strip(), mode="eval")
    except SyntaxError:
        return set()
    used: set[str] = set()
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Call) and isinstance(node.func, _ast.Name):
            if node.func.id in FUNCTIONS:
                used.add(node.func.id)
    return used


def sanitize_generator_history(history: list[dict[str, Any]] | None, *, limit: int = 10
                               ) -> list[dict[str, Any]]:
    """Fail closed on test/lockbox keys, then project history onto the permitted feedback fields."""
    history = history or []
    if not isinstance(history, list):
        raise TypeError("generator history must be a list")
    forbidden = _find_forbidden_keys(history)
    if forbidden:
        raise FeedbackLeakageError(
            "generator history contains forbidden test/lockbox fields: " + ", ".join(forbidden))
    safe: list[dict[str, Any]] = []
    for item in history[-limit:]:
        if not isinstance(item, dict):
            raise TypeError("each generator-history item must be an object")
        projected = {key: item[key] for key in _FEEDBACK_FIELDS if key in item}
        if "expression_signature" not in projected and item.get("expression"):
            projected["expression_signature"] = compile_expression(
                str(item["expression"])).signature()
        projected.setdefault("failed_gates", [])
        if not isinstance(projected["failed_gates"], list):
            raise TypeError("failed_gates must be a list")
        safe.append(projected)
    return safe


def _find_forbidden_keys(value: Any, path: str = "history") -> list[str]:
    """Recursively report forbidden evaluation/test fields before history can reach a generator."""
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            low = key_text.casefold()
            child_path = f"{path}.{key_text}"
            if low.startswith("test") or "lockbox" in low:
                found.append(child_path)
            found.extend(_find_forbidden_keys(child, child_path))
    elif isinstance(value, (list, tuple)):
        for i, child in enumerate(value):
            found.extend(_find_forbidden_keys(child, f"{path}[{i}]"))
    return sorted(set(found))


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int:
        raise RequestValidationError(f"{name} must be a positive integer")
    parsed = value
    if parsed <= 0:
        raise RequestValidationError(f"{name} must be a positive integer")
    return parsed


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise RequestValidationError(f"{name} must be a list")
    items = tuple(str(item).strip() for item in value)
    if any(not item for item in items):
        raise RequestValidationError(f"{name} cannot contain empty values")
    return tuple(dict.fromkeys(items))


def _window(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("generator windows must be integers")
    parsed = int(value)
    if parsed != value or not 2 <= parsed <= MAX_WINDOW:
        raise ValueError(f"generator window must be in [2, {MAX_WINDOW}]")
    return parsed
