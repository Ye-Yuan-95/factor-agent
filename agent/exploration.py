"""Host-side exploration over recipe templates and factor-family pairs.

ExplorationScheduler uses Thompson sampling over search regions. CoverageTable
records attempted regions and gate progress. AST node-multiset distance rejects
cosmetic rewrites before they consume an oracle trial. These components expose
separate switches for controlled comparisons.

Bandit rewards use only train/validation gate feedback: the reward function takes
failed gates and the decision, never a truth series, test result or lockbox metric.
This separation prevents synthetic truth from directing candidate generation."""
from __future__ import annotations

import ast as _ast
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

# gate order matters: reward is "how far down the funnel did this candidate get"
GATE_ORDER = ("fdr", "redundancy", "survival", "incremental", "sparse_weight", "dsr")
GATE_FORMS = ("level_gate", "delta_gate", "clip_gate", "spread", "other")


def gate_progress_reward(failed_gates: Iterable[str] | None, decision: str) -> float:
    """Graded reward in [0, 1] from gate feedback alone (never the truth).

    A candidate that dies at the first gate teaches less than one that reaches the incremental
    test, so a binary accept/reject reward would throw away most of the signal available at
    realistic budgets, where accepts are rare.
    """
    if decision == "ACCEPT":
        return 1.0
    if failed_gates is None:
        return 0.0              # no gate feedback at all (sandbox/coverage failure): nothing learned
    failed = set(failed_gates)
    if not failed:
        return 0.9              # gates ran and none failed, yet not accepted (e.g. SHELVED)
    for i, gate in enumerate(GATE_ORDER):
        if gate in failed:
            return round(0.8 * i / len(GATE_ORDER), 6)
    return 0.0                  # sandbox/coverage failure: no gate was reached


def classify_gate_form(expression: str) -> str:
    """Classify the expression form so coverage records both column pairs and gating structures."""
    try:
        tree = _ast.parse(expression.strip(), mode="eval")
    except SyntaxError:
        return "other"
    calls = [node.func.id for node in _ast.walk(tree)
             if isinstance(node, _ast.Call) and isinstance(node.func, _ast.Name)]
    has_product = any(isinstance(node, _ast.BinOp) and isinstance(node.op, _ast.Mult)
                      for node in _ast.walk(tree))
    has_difference = any(isinstance(node, _ast.BinOp) and isinstance(node.op, _ast.Sub)
                         for node in _ast.walk(tree))
    if "clip" in calls and has_product:
        return "clip_gate"
    if "sign" in calls and has_product:
        return "delta_gate" if "delta" in calls else "level_gate"
    if has_difference and ("zscore" in calls or "rank" in calls):
        return "spread"
    return "other"


def ast_node_multiset(expression: str) -> Counter:
    """Bag of AST node signatures. Deliberately simple: cheap, order-insensitive, and enough
    to catch the rewrite-with-a-different-wrapper family. A full tree edit distance buys
    little here and costs a dependency."""
    try:
        tree = _ast.parse(expression.strip(), mode="eval")
    except SyntaxError:
        return Counter()
    bag: Counter = Counter()
    # a Call's func is itself a Name node; without this it would be counted as a data column,
    # inflating every distance and polluting the col: namespace with operator names
    function_names = {id(node.func) for node in _ast.walk(tree)
                      if isinstance(node, _ast.Call) and isinstance(node.func, _ast.Name)}
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Call) and isinstance(node.func, _ast.Name):
            bag[f"call:{node.func.id}"] += 1
        elif isinstance(node, _ast.Name):
            if id(node) not in function_names:
                bag[f"col:{node.id}"] += 1
        elif isinstance(node, _ast.BinOp):
            bag[f"op:{type(node.op).__name__}"] += 1
        elif isinstance(node, _ast.Constant):
            bag[f"const:{node.value}"] += 1
    return bag


def ast_distance(expression_a: str, expression_b: str) -> float:
    """Normalised multiset distance in [0, 1]; 0 means structurally identical."""
    bag_a, bag_b = ast_node_multiset(expression_a), ast_node_multiset(expression_b)
    if not bag_a and not bag_b:
        return 0.0
    difference = sum(((bag_a - bag_b) + (bag_b - bag_a)).values())
    return round(difference / max(sum((bag_a | bag_b).values()), 1), 6)


def min_ast_distance(expression: str, previous: Iterable[str]) -> float:
    distances = [ast_distance(expression, other) for other in previous]
    return min(distances) if distances else 1.0


# Minimum node-multiset distance for novelty. Keep the threshold weak enough
# to permit column exploration while refusing near-identical restatements.
DEFAULT_MIN_NOVELTY = 0.30


@dataclass(frozen=True)
class Arm:
    recipe: str
    signal_family: str
    conditioner_family: str

    @property
    def key(self) -> str:
        return f"{self.recipe}|{self.signal_family}->{self.conditioner_family}"


@dataclass
class ArmState:
    pulls: int = 0
    reward_sum: float = 0.0
    alpha: float = 1.0
    beta: float = 1.0
    best_reward: float = 0.0
    columns_tried: set[tuple[str, str]] = field(default_factory=set)
    forms_tried: Counter = field(default_factory=Counter)


class ExplorationScheduler:
    """Thompson sampling over (recipe, ordered family pair) arms.

    Coarse by design: the arm fixes a REGION (which structure, which two families), and the
    downstream generator -- LLM or random -- picks the actual columns inside it. That split is
    what allows a controlled comparison: random column choice supplies a measurable
    local-generator reference under the same search-region policy.
    """

    def __init__(self, recipes: list[str], families: list[str], rng: np.random.Generator,
                 *, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        ordered_pairs = [(a, b) for a in sorted(families) for b in sorted(families) if a != b]
        if not ordered_pairs:
            raise ValueError("exploration scheduler needs at least two families")
        if not recipes:
            raise ValueError("exploration scheduler needs at least one recipe")
        self.arms = [Arm(recipe, a, b) for recipe in sorted(recipes) for a, b in ordered_pairs]
        self.rng = rng
        self.state: dict[str, ArmState] = {
            arm.key: ArmState(alpha=prior_alpha, beta=prior_beta) for arm in self.arms}

    def select_arm(self) -> Arm:
        """Untried arms first (guarantees coverage before exploitation), then Thompson."""
        untried = [arm for arm in self.arms if self.state[arm.key].pulls == 0]
        if untried:
            return untried[int(self.rng.integers(0, len(untried)))]
        samples = [self.rng.beta(self.state[arm.key].alpha, self.state[arm.key].beta)
                   for arm in self.arms]
        return self.arms[int(np.argmax(samples))]

    def update(self, arm: Arm, reward: float, *, expression: str = "",
               columns: tuple[str, str] | None = None) -> None:
        """Apply a bounded reward to one arm and record expression/pair coverage after a completed attempt."""
        if not 0.0 <= reward <= 1.0:
            raise ValueError("reward must be in [0, 1]")
        state = self.state[arm.key]
        state.pulls += 1
        state.reward_sum += reward
        state.best_reward = max(state.best_reward, reward)
        # Beta update with a continuous reward: the standard Bernoulli-relaxation trick
        state.alpha += reward
        state.beta += 1.0 - reward
        if columns:
            state.columns_tried.add(tuple(columns))
        if expression:
            state.forms_tried[classify_gate_form(expression)] += 1

    def stats(self) -> list[dict[str, Any]]:
        """Export per-arm exploration counters and reward summaries without mutating scheduler state."""
        rows = []
        for arm in self.arms:
            state = self.state[arm.key]
            rows.append({
                "arm": arm.key, "recipe": arm.recipe,
                "signal_family": arm.signal_family,
                "conditioner_family": arm.conditioner_family,
                "pulls": state.pulls,
                "mean_reward": round(state.reward_sum / state.pulls, 6) if state.pulls else None,
                "best_reward": round(state.best_reward, 6),
                "n_column_pairs_tried": len(state.columns_tried),
                "forms_tried": dict(state.forms_tried),
            })
        return rows


class CoverageTable:
    """Record attempted column pairs and gating forms with allowed gate rewards.

    Render covered and untried cells into the generator prompt."""

    def __init__(self) -> None:
        self.cells: dict[tuple[str, str], dict[str, Any]] = {}

    def record(self, columns: tuple[str, str], expression: str, reward: float) -> None:
        cell = self.cells.setdefault(tuple(columns), {"tried": 0, "best_reward": 0.0,
                                                      "forms": Counter()})
        cell["tried"] += 1
        cell["best_reward"] = max(cell["best_reward"], reward)
        cell["forms"][classify_gate_form(expression)] += 1

    def untried_forms(self, columns: tuple[str, str]) -> list[str]:
        cell = self.cells.get(tuple(columns))
        tried = set(cell["forms"]) if cell else set()
        return [form for form in GATE_FORMS if form != "other" and form not in tried]

    def rows(self) -> list[dict[str, Any]]:
        """Stable, JSON-serialisable audit view used in run artifacts."""
        return [
            {
                "columns": list(columns),
                "tried": int(cell["tried"]),
                "best_gate_progress": round(float(cell["best_reward"]), 6),
                "forms_tried": dict(sorted(cell["forms"].items())),
                "forms_not_yet_tried": self.untried_forms(columns),
            }
            for columns, cell in sorted(self.cells.items())
        ]

    def prompt_payload(self, *, limit: int = 12) -> dict[str, Any]:
        """Compact, model-readable view. Sorted worst-first so the empty and weak cells are
        the ones that survive truncation."""
        rows = sorted(self.cells.items(), key=lambda item: (item[1]["best_reward"],
                                                           -item[1]["tried"]))
        return {
            "tried_column_pairs": [
                {"columns": list(columns),
                 "tried": cell["tried"],
                 "best_gate_progress": round(cell["best_reward"], 3),
                 "forms_tried": sorted(cell["forms"]),
                 "forms_not_yet_tried": self.untried_forms(columns)}
                for columns, cell in rows[:limit]],
            "n_distinct_column_pairs_tried": len(self.cells),
            "instruction": ("优先选择未出现在 tried_column_pairs 中的列对;若复用已试列对，"
                            "必须改用 forms_not_yet_tried 中的门控形式。"),
        }
