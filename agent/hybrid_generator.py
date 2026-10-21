"""Hybrid generators: host-owned exploration, model-owned (or random) hypothesis filling.

Two variants sharing one exploration policy (see agent/exploration.py):

  HybridRandomGenerator -- scheduler picks the (recipe, family pair) region, columns are filled
      at random inside it. No LLM, no credentials, fully local. This is the degenerate case of
      the hybrid architecture and therefore its floor: any claim that the model adds value must
      be measured against THIS line, not against unstructured random search.

  HybridLLMGenerator -- same scheduler, but the chosen region plus the coverage table are
      injected into the prompt and the model writes the expression. Proposals that repeat a
      structure already tried are rejected locally (novelty pressure), consuming a generation
      attempt but not an oracle trial.

Both learn from gate feedback only. Rewards are recomputed from the sanitized history the
harness already passes to generators, so no new plumbing carries privileged information: the
history contains outcome and failed_gates, never truth loading.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from agent.exploration import (
    DEFAULT_MIN_NOVELTY,
    Arm,
    CoverageTable,
    ExplorationScheduler,
    classify_gate_form,
    gate_progress_reward,
    min_ast_distance,
)
from agent.factor_dsl import DSLError, column_is_allowed, compile_expression
from agent.generator_contract import RECIPE_NAMES, ResearchRequest, check_recipe_conformance

WINDOWS = (3, 6, 12)


class _ArmMixin:
    """Shared bookkeeping: which arm produced which candidate, and how history feeds back."""

    def _init_exploration(self, columns: list[str], families: dict[str, str],
                          rng: np.random.Generator, recipes: list[str] | None) -> None:
        """Build allowed family groups, coverage state and the seeded recipe scheduler."""
        self.families = {c: families.get(c, "unclassified")
                         for c in columns if column_is_allowed(c)}
        self.by_family: dict[str, list[str]] = {}
        for column, family in self.families.items():
            self.by_family.setdefault(family, []).append(column)
        self.by_family = {family: sorted(cols) for family, cols in self.by_family.items()}
        if len(self.by_family) < 2:
            raise ValueError("hybrid generator needs columns from at least two families")
        self.rng = rng
        self.scheduler = ExplorationScheduler(
            list(recipes or RECIPE_NAMES), sorted(self.by_family), rng)
        self.coverage = CoverageTable()
        self._pending: dict[str, dict[str, Any]] = {}
        self._processed: set[str] = set()
        self._tried_expressions: list[str] = []
        self._seen_signatures: set[str] = set()
        self._seq = 0

    def _absorb_history(self, history: list[dict[str, Any]] | None) -> None:
        """Turn newly returned gate feedback into bandit and coverage updates."""
        for item in history or []:
            name = str(item.get("name") or "")
            if not name or name in self._processed or name not in self._pending:
                continue
            record = self._pending[name]
            reward = gate_progress_reward(item.get("failed_gates"),
                                          str(item.get("outcome") or "REJECT"))
            self.scheduler.update(record["arm"], reward, expression=record["expression"],
                                  columns=record["columns"])
            self.coverage.record(record["columns"], record["expression"], reward)
            self._processed.add(name)

    def _arm_columns(self, arm: Arm) -> tuple[str, str] | None:
        signal_pool = self.by_family.get(arm.signal_family, [])
        conditioner_pool = self.by_family.get(arm.conditioner_family, [])
        if not signal_pool or not conditioner_pool:
            return None
        return (str(self.rng.choice(signal_pool)), str(self.rng.choice(conditioner_pool)))

    def scheduler_stats(self) -> list[dict[str, Any]]:
        return self.scheduler.stats()

    def finalize(self, history: list[dict[str, Any]]) -> None:
        """Absorb the final validated candidate even though no next proposal is requested."""
        self._absorb_history(history)

    def diagnostics(self) -> dict[str, Any]:
        return {
            "scheduler": self.scheduler_stats(),
            "coverage": self.coverage.rows(),
            "n_registered_candidates": len(self._pending),
            "n_pending_feedback": sum(
                name not in self._processed for name in self._pending),
            "n_processed_feedback": len(self._processed),
        }


class HybridRandomGenerator(_ArmMixin):
    """Stratified enumerator: bandit-scheduled region, random column fill, no LLM."""

    def __init__(self, columns: list[str], families: dict[str, str], rng: np.random.Generator,
                 *, recipes: list[str] | None = None, name_prefix: str = "hr",
                 min_novelty: float = DEFAULT_MIN_NOVELTY):
        self._init_exploration(columns, families, rng, recipes)
        self.name_prefix = name_prefix
        self.min_novelty = float(min_novelty)

    def _render(self, arm: Arm, signal_col: str, conditioner_col: str) -> str:
        """Render a sampled arm with locally chosen transforms and time windows."""
        window = int(self.rng.choice(WINDOWS))
        cs = str(self.rng.choice(("rank", "zscore")))
        body = (signal_col if self.rng.random() < 0.6
                else f"ts_mean({signal_col}, {window})")
        if arm.recipe == "conditional_signal":
            gate = (f"sign({conditioner_col})" if self.rng.random() < 0.5
                    else f"sign(delta({conditioner_col}, {window}))")
            return f"{cs}({body}) * {gate}"
        if arm.recipe == "regime_gated":
            core = (f"zscore({conditioner_col})" if self.rng.random() < 0.5
                    else f"sign({conditioner_col})")
            return f"{cs}({body}) * clip({core}, 0, {int(self.rng.choice((1, 3)))})"
        other = (conditioner_col if self.rng.random() < 0.6
                 else f"ts_mean({conditioner_col}, {window})")
        return f"zscore({body}) - zscore({other})"

    def propose(self, history: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
        """Update from allowed history and sample bounded attempts with coverage and novelty filters."""
        self._absorb_history(history)
        for _ in range(80):
            arm = self.scheduler.select_arm()
            columns = self._arm_columns(arm)
            if columns is None:
                continue
            expression = self._render(arm, *columns)
            try:
                compiled = compile_expression(expression)
            except DSLError:
                continue
            signature = compiled.signature()
            if signature in self._seen_signatures:
                continue
            if min_ast_distance(expression, self._tried_expressions) < self.min_novelty:
                continue
            self._seen_signatures.add(signature)
            self._tried_expressions.append(expression)
            self._seq += 1
            name = f"{self.name_prefix}{self._seq:03d}"
            self._pending[name] = {"arm": arm, "columns": columns, "expression": expression}
            fams = [self.families.get(c) for c in compiled.columns if self.families.get(c)]
            return {
                "name": name,
                "expression": expression,
                "rationale": (f"hybrid arm {arm.key}: {columns[0]} conditioned on "
                              f"{columns[1]}"),
                "family": max(set(fams), key=fams.count) if fams else "unclassified",
                "recipe_name": arm.recipe,
            }
        return None


class FormAwareHybridRandomGenerator(HybridRandomGenerator):
    """F4b intervention: complete untried gate forms on promising column pairs.

    The ordinary scheduler remains the fallback. Once a completed proposal reaches the frozen
    gate-progress threshold, the host prioritises one of that exact ordered column pair's
    untried forms. The priority proposal bypasses the generic AST-distance filter because the
    intentionally new form is the intervention; exact signatures remain forbidden.
    """

    FORM_RECIPES = {
        "level_gate": "conditional_signal",
        "delta_gate": "conditional_signal",
        "clip_gate": "regime_gated",
        "spread": "cross_family_spread",
    }

    def __init__(self, columns: list[str], families: dict[str, str],
                 rng: np.random.Generator, *, recipes: list[str] | None = None,
                 name_prefix: str = "hfa", min_novelty: float = DEFAULT_MIN_NOVELTY,
                 form_completion_reward_min: float = 0.4):
        super().__init__(columns, families, rng, recipes=recipes, name_prefix=name_prefix,
                         min_novelty=min_novelty)
        if not 0.0 <= form_completion_reward_min <= 1.0:
            raise ValueError("form completion reward threshold must be in [0, 1]")
        self.form_completion_reward_min = float(form_completion_reward_min)
        self.form_priority_picks = 0

    @staticmethod
    def _render_form(form: str, signal_col: str, conditioner_col: str) -> str:
        if form == "level_gate":
            return f"zscore({signal_col}) * sign({conditioner_col})"
        if form == "delta_gate":
            return f"zscore({signal_col}) * sign(delta({conditioner_col}, 6))"
        if form == "clip_gate":
            return f"zscore({signal_col}) * clip(sign({conditioner_col}), 0, 3)"
        if form == "spread":
            return f"zscore({signal_col}) - zscore({conditioner_col})"
        raise ValueError(f"unknown gate form: {form}")

    def _completion_choice(self) -> tuple[tuple[str, str], str] | None:
        """Choose an untried gating form for a pair whose recorded reward meets the completion threshold."""
        eligible: list[tuple[tuple[str, str], dict[str, Any], list[str]]] = []
        available_recipes = {arm.recipe for arm in self.scheduler.arms}
        for columns, cell in self.coverage.cells.items():
            if float(cell["best_reward"]) < self.form_completion_reward_min:
                continue
            if self.families.get(columns[0]) == self.families.get(columns[1]):
                continue
            forms = [form for form in self.coverage.untried_forms(columns)
                     if self.FORM_RECIPES[form] in available_recipes]
            if forms:
                eligible.append((columns, cell, forms))
        if not eligible:
            return None
        best_reward = max(float(item[1]["best_reward"]) for item in eligible)
        best = [item for item in eligible
                if float(item[1]["best_reward"]) == best_reward]
        fewest_forms = min(len(item[1]["forms"]) for item in best)
        best = sorted(item for item in best if len(item[1]["forms"]) == fewest_forms)
        columns, _cell, forms = best[int(self.rng.integers(0, len(best)))]
        return columns, str(self.rng.choice(sorted(forms)))

    def propose(self, history: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
        """Try form completion first, then fall back to the original hybrid exploration path."""
        self._absorb_history(history)
        for _ in range(16):
            choice = self._completion_choice()
            if choice is None:
                break
            columns, gate_form = choice
            expression = self._render_form(gate_form, *columns)
            try:
                compiled = compile_expression(expression)
            except DSLError:
                continue
            signature = compiled.signature()
            if signature in self._seen_signatures:
                continue
            if classify_gate_form(expression) != gate_form:
                raise RuntimeError(f"form renderer produced wrong form: {gate_form}")
            recipe = self.FORM_RECIPES[gate_form]
            arm = Arm(recipe, self.families[columns[0]], self.families[columns[1]])
            self._seen_signatures.add(signature)
            self._tried_expressions.append(expression)
            self._seq += 1
            self.form_priority_picks += 1
            name = f"{self.name_prefix}{self._seq:03d}"
            self._pending[name] = {"arm": arm, "columns": columns, "expression": expression}
            fams = [self.families.get(c) for c in compiled.columns if self.families.get(c)]
            return {
                "name": name,
                "expression": expression,
                "rationale": (f"coverage-priority completion: {columns[0]} conditioned on "
                              f"{columns[1]} with untried form {gate_form}"),
                "family": max(set(fams), key=fams.count) if fams else "unclassified",
                "recipe_name": recipe,
                "gate_form": gate_form,
                "form_priority": True,
            }
        return super().propose(history)

    def diagnostics(self) -> dict[str, Any]:
        return {
            **super().diagnostics(),
            "policy": "gate_reward_then_untried_form_priority",
            "form_completion_reward_min": self.form_completion_reward_min,
            "form_priority_picks": self.form_priority_picks,
        }


class HybridLLMGenerator(_ArmMixin):
    """Scheduler chooses the region and the coverage table is shown; the model writes the
    expression. Wraps an existing LLMGenerator rather than replacing it, so prompt-integrity
    checks, schema strictness and family enforcement stay in one place."""

    def __init__(self, *, llm_generator: Any, columns: list[str], families: dict[str, str],
                 rng: np.random.Generator, recipes: list[str] | None = None,
                 min_novelty: float = DEFAULT_MIN_NOVELTY,
                 use_scheduler: bool = True, use_coverage: bool = True,
                 use_novelty: bool = True):
        """Bind the model to host-controlled scheduling, coverage and novelty options for ablations."""
        self._init_exploration(columns, families, rng, recipes)
        self.llm = llm_generator
        self.min_novelty = float(min_novelty)
        self.use_scheduler = bool(use_scheduler)
        self.use_coverage = bool(use_coverage)
        self.use_novelty = bool(use_novelty)
        self.novelty_rejections = 0
        self.off_arm_proposals = 0

    def propose(self, request: ResearchRequest,
                history: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """Request a constrained model proposal using host-selected exploration context.
        The host enforces the selected vocabulary and records metadata; it does not let the model set gates."""
        self._absorb_history(history)
        payload: dict[str, Any] = {}
        arm = None
        if self.use_scheduler:
            arm = self.scheduler.select_arm()
            payload["assigned_region"] = {
                "recipe_name": arm.recipe,
                "signal_family": arm.signal_family,
                "conditioner_family": arm.conditioner_family,
                "signal_columns": self.by_family.get(arm.signal_family, []),
                "conditioner_columns": self.by_family.get(arm.conditioner_family, []),
                "instruction": ("本轮必须使用 assigned_region 指定的 recipe，"
                                "信号列取自 signal_columns，条件列取自 conditioner_columns。"),
            }
        if self.use_coverage:
            payload["coverage"] = self.coverage.prompt_payload()
        proposal = self.llm.propose(request, history=history, extra_payload=payload or None)

        expression = proposal["expression"]
        if self.use_novelty:
            distance = min_ast_distance(expression, self._tried_expressions)
            if distance < self.min_novelty:
                self.novelty_rejections += 1
                from agent.llm_generator import ProposalConstraintError
                raise ProposalConstraintError(
                    f"proposal is structurally too close to an earlier one "
                    f"(AST distance {distance:.3f} < {self.min_novelty})"
                ).attach_trace(dict(proposal.get("generation") or {}))

        columns = tuple(proposal.get("columns_used") or ())
        chosen = self._infer_columns(arm, columns)
        if arm is not None:
            # HARD reject, not a counter. An off-region proposal that is still validated would
            # be credited to the arm the scheduler chose, corrupting the bandit posterior with
            # a reward the arm never earned -- and the ablation would then be measuring a
            # scheduler that does not actually schedule.
            in_region = (chosen is not None and
                         self.families.get(chosen[0]) == arm.signal_family and
                         self.families.get(chosen[1]) == arm.conditioner_family)
            declared_recipe = str(proposal.get("recipe_name") or "")
            recipe_ok, recipe_violation = check_recipe_conformance(
                expression, arm.recipe, self.llm.spec)
            on_arm = in_region and declared_recipe == arm.recipe and recipe_ok
            if not on_arm:
                self.off_arm_proposals += 1
                from agent.llm_generator import ProposalConstraintError
                raise ProposalConstraintError(
                    f"proposal leaves the assigned arm {arm.key}: columns "
                    f"{sorted(columns)} map to families "
                    f"{sorted({self.families.get(c, '?') for c in columns})}; "
                    f"declared recipe={declared_recipe or None!r}; "
                    f"assigned-recipe conformance={recipe_ok} ({recipe_violation})"
                ).attach_trace(dict(proposal.get("generation") or {}))
        self._tried_expressions.append(expression)
        pending_key = expression
        if chosen is not None:
            effective_arm = arm or Arm(
                str(proposal.get("recipe_name") or RECIPE_NAMES[0]),
                self.families.get(chosen[0], "unclassified"),
                self.families.get(chosen[1], "unclassified"))
            self._pending[pending_key] = {
                "arm": effective_arm, "columns": chosen, "expression": expression}
        conformant = None
        if proposal.get("recipe_name"):
            conformant, _ = check_recipe_conformance(
                expression, str(proposal["recipe_name"]), self.llm.spec)
        return {**proposal,
                "assigned_arm": arm.key if arm else None,
                # the harness names candidates after generation; it must hand this key back
                # via register_name() so gate feedback can be matched to its arm
                "pending_key": pending_key,
                "recipe_conformant": proposal.get("recipe_conformant", conformant)}

    def _infer_columns(self, arm: Arm | None,
                       columns_used: tuple[str, ...]) -> tuple[str, str] | None:
        """Best-effort (signal, conditioner) attribution for coverage bookkeeping."""
        if len(columns_used) < 2:
            return None
        if arm is not None:
            signal = next((c for c in columns_used
                           if self.families.get(c) == arm.signal_family), None)
            conditioner = next((c for c in columns_used
                                if self.families.get(c) == arm.conditioner_family), None)
            if signal and conditioner:
                return (signal, conditioner)
        return (columns_used[0], columns_used[1])

    def register_name(self, placeholder: str, name: str) -> None:
        """The harness names candidates after the generator returns; re-key so history
        feedback can be matched back to its arm."""
        if placeholder in self._pending and name not in self._pending:
            self._pending[name] = self._pending.pop(placeholder)
