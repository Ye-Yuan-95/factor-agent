"""No-LLM baseline generator: random / genetic search over the DSL grammar.

Why it exists: without a dumb generator wired to the SAME oracle and the SAME trial budget, the
final system cannot attribute its results -- "did the LLM add anything beyond oracle + brute
force?" This module is that control arm, and it doubles as the load generator for the L2
realized-FDR acceptance harness (agent/realized_fdr.py).

Expressions are built as small recipe trees (not strings), so mutation is structural:
  * swap a column / a time-series operator / an element-wise operator,
  * resize a rolling window, toggle the cross-sectional wrap (rank <-> zscore),
  * negate, or replace a random subtree with a fresh sample.
Every rendered expression is verified through the real sandbox (compile_expression) before it is
proposed, and exact duplicates (same canonical AST signature) are never proposed twice.

Two modes:
  random    -- ignores feedback entirely (pure brute-force baseline).
  adaptive  -- mutates "near-miss" parents (ACCEPT, SHELVED, or REJECT that failed <= 1 gate),
               i.e. the cheapest possible feedback loop: reject_reason in, next candidate out.
               This exercises the same feedback plumbing an LLM generator will use.

CLI (writes a candidates JSON consumable by agent/l1_runner.py):
  python -m agent.baseline_generator --columns-from data_big_v2/a_share_factor_panel.csv       --config a_share_big_config.json --n 30 --mode random --seed 0 --out candidates_bg.json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from agent.factor_dsl import compile_expression, DSLError, column_is_allowed  # noqa: E402

TS_OPS = ("ts_mean", "ts_std", "ts_rank", "ts_zscore", "ts_min", "ts_max")
EW_OPS = ("log", "sqrt", "abs", "sign")
CS_OPS = ("rank", "zscore")
WINDOWS = (3, 6, 12)
BIN_OPS = ("+", "-", "*")


@dataclass
class Node:
    kind: str                    # 'col' | 'ts' | 'ew' | 'cs' | 'delta' | 'bin' | 'neg'
    op: str | None = None
    col: str | None = None
    window: int | None = None
    children: list["Node"] = field(default_factory=list)

    def render(self) -> str:
        """Serialize the expression tree into restricted DSL syntax without evaluating it."""
        if self.kind == "col":
            return self.col
        if self.kind == "ts":
            return f"{self.op}({self.children[0].render()}, {self.window})"
        if self.kind == "delta":
            return f"delta({self.children[0].render()}, {self.window})"
        if self.kind == "ew":
            return f"{self.op}({self.children[0].render()})"
        if self.kind == "cs":
            return f"{self.op}({self.children[0].render()})"
        if self.kind == "neg":
            return f"-1 * {self.children[0].render()}"
        if self.kind == "bin":
            return f"({self.children[0].render()} {self.op} {self.children[1].render()})"
        raise ValueError(self.kind)

    def all_nodes(self) -> list["Node"]:
        out = [self]
        for c in self.children:
            out.extend(c.all_nodes())
        return out

    def clone(self) -> "Node":
        return Node(self.kind, self.op, self.col, self.window, [c.clone() for c in self.children])


class BaselineGenerator:
    """propose(history) -> {name, expression, rationale, family}. The same interface an LLM
    generator will implement; `history` entries carry decision + reject_reason (train/val info
    only, per the leakage iron rule)."""

    def __init__(self, columns: list[str], families: dict[str, str], rng: np.random.Generator,
                 mode: str = "random", max_depth: int = 3, name_prefix: str = "bg"):
        """Initialize a seeded search over allowed columns; empty safe vocabularies are rejected."""
        self.columns = [c for c in columns if column_is_allowed(c)]
        if not self.columns:
            raise ValueError("no allowed columns to generate over")
        self.families = families
        self.rng = rng
        self.mode = mode
        self.max_depth = max_depth
        self.name_prefix = name_prefix
        self._recipes: dict[str, Node] = {}      # name -> recipe (for mutation)
        self._seen_signatures: set[str] = set()
        self._seq = 0

    # ------------------------------------------------------------------ #
    # sampling                                                            #
    # ------------------------------------------------------------------ #
    def _rand_col(self) -> Node:
        return Node("col", col=str(self.rng.choice(self.columns)))

    def _rand_term(self, depth: int) -> Node:
        """Sample a bounded-depth DSL subtree using the configured operator mixture."""
        r = self.rng.random()
        if depth >= self.max_depth or r < 0.30:
            return self._rand_col()
        if r < 0.60:
            return Node("ts", op=str(self.rng.choice(TS_OPS)),
                        window=int(self.rng.choice(WINDOWS)), children=[self._rand_term(depth + 1)])
        if r < 0.72:
            return Node("delta", window=int(self.rng.choice(WINDOWS)),
                        children=[self._rand_term(depth + 1)])
        if r < 0.86:
            return Node("ew", op=str(self.rng.choice(EW_OPS)), children=[self._rand_term(depth + 1)])
        return Node("bin", op=str(self.rng.choice(BIN_OPS)),
                    children=[self._rand_term(depth + 1), self._rand_term(depth + 1)])

    def _rand_expr(self) -> Node:
        body = self._rand_term(1)
        if self.rng.random() < 0.65:                      # usually wrap cross-sectionally
            body = Node("cs", op=str(self.rng.choice(CS_OPS)), children=[body])
        if self.rng.random() < 0.30:
            body = Node("neg", children=[body])
        return body

    # ------------------------------------------------------------------ #
    # mutation (genetic step)                                             #
    # ------------------------------------------------------------------ #
    def _mutate(self, recipe: Node) -> Node:
        """Mutate a cloned recipe with bounded retries, leaving the stored parent unchanged."""
        out = recipe.clone()
        nodes = out.all_nodes()
        for _ in range(8):                                # try mutations until one applies
            choice = self.rng.random()
            node = nodes[int(self.rng.integers(0, len(nodes)))]
            if choice < 0.25 and node.kind == "col":
                node.col = str(self.rng.choice(self.columns))
                return out
            if choice < 0.45 and node.kind in ("ts", "delta") and node.window is not None:
                node.window = int(self.rng.choice([w for w in WINDOWS if w != node.window]))
                return out
            if choice < 0.60 and node.kind == "ts":
                node.op = str(self.rng.choice([o for o in TS_OPS if o != node.op]))
                return out
            if choice < 0.70 and node.kind == "ew":
                node.op = str(self.rng.choice([o for o in EW_OPS if o != node.op]))
                return out
            if choice < 0.80 and node.kind == "cs":
                node.op = "zscore" if node.op == "rank" else "rank"
                return out
            if choice < 0.90:                             # replace a subtree with a fresh sample
                node_new = self._rand_term(2)
                node.kind, node.op, node.col = node_new.kind, node_new.op, node_new.col
                node.window, node.children = node_new.window, node_new.children
                return out
        return Node("neg", children=[out])                # fallback: flip the sign

    # ------------------------------------------------------------------ #
    # the generator interface                                             #
    # ------------------------------------------------------------------ #
    def _near_misses(self, history: list[dict]) -> list[dict]:
        """Parents worth mutating: ACCEPT/SHELVED, or REJECT that failed at most one gate."""
        out = []
        for h in history:
            if h.get("outcome") in ("ACCEPT", "SHELVED"):
                out.append(h)
            elif h.get("outcome") == "REJECT":
                reasons = [r for r in (h.get("reject_reason") or "").split(";") if r]
                if len(reasons) <= 1:
                    out.append(h)
        return [h for h in out if h.get("name") in self._recipes]

    def propose(self, history: list[dict] | None = None) -> dict | None:
        """Return the next candidate, or None if the sampler cannot find a novel valid one."""
        history = history or []
        for _ in range(60):
            parent = None
            if self.mode == "adaptive":
                pool = self._near_misses(history)
                if pool and self.rng.random() < 0.5:
                    parent = pool[int(self.rng.integers(0, len(pool)))]
            recipe = (self._mutate(self._recipes[parent["name"]])
                      if parent is not None else self._rand_expr())
            expression = recipe.render()
            try:
                compiled = compile_expression(expression)
            except DSLError:
                continue                                   # invalid sample: resample, never propose
            sig = compiled.signature()
            if sig in self._seen_signatures:
                continue
            self._seen_signatures.add(sig)
            self._seq += 1
            name = f"{self.name_prefix}{self._seq:03d}"
            self._recipes[name] = recipe
            cols = compiled.columns
            fams = [self.families.get(c) for c in cols if self.families.get(c)]
            family = max(set(fams), key=fams.count) if fams else "unclassified"
            rationale = (f"baseline mutation of {parent['name']}" if parent is not None
                         else "baseline random DSL sample")
            return {"name": name, "expression": expression, "rationale": rationale,
                    "family": family}
        return None


class RecipeBaselineGenerator:
    """Recipe-following random control: samples cross-family
    expressions INSIDE the mechanism recipe templates, with zero language understanding.

    This is the arm that separates 'the recipe templates carry the value' from 'the model
    carries the value': if GLM with recipe prompting only matches this generator's discovery
    rate, the templates explain the result. Emits ``recipe_name`` and is conformant to
    ``check_recipe_conformance`` by construction (the harness still verifies)."""

    def __init__(self, columns: list[str], families: dict[str, str], rng: np.random.Generator,
                 name_prefix: str = "rr"):
        """Index allowed columns by family for cross-family recipe sampling."""
        from agent.generator_contract import RECIPE_NAMES
        self._recipe_names = RECIPE_NAMES
        self.families = {c: families.get(c, "unclassified")
                         for c in columns if column_is_allowed(c)}
        by_family: dict[str, list[str]] = {}
        for column, family in self.families.items():
            by_family.setdefault(family, []).append(column)
        self.by_family = {family: sorted(cols) for family, cols in by_family.items()}
        if len(self.by_family) < 2:
            raise ValueError("recipe baseline needs columns from at least two families")
        self.rng = rng
        self.name_prefix = name_prefix
        self._seen_signatures: set[str] = set()
        self._seq = 0

    def _sample(self) -> tuple[str, str]:
        """Sample a recipe and two distinct families, returning its label and expression."""
        rng = self.rng
        recipe = str(rng.choice(self._recipe_names))
        family_names = sorted(self.by_family)
        picked = rng.choice(len(family_names), size=2, replace=False)
        col_a = str(rng.choice(self.by_family[family_names[int(picked[0])]]))
        col_b = str(rng.choice(self.by_family[family_names[int(picked[1])]]))
        window = int(rng.choice(WINDOWS))
        cs = str(rng.choice(CS_OPS))
        wrap_a = col_a if rng.random() < 0.6 else f"ts_mean({col_a}, {window})"
        if recipe == "conditional_signal":
            gate = (f"sign({col_b})" if rng.random() < 0.5
                    else f"sign(delta({col_b}, {window}))")
            return recipe, f"{cs}({wrap_a}) * {gate}"
        if recipe == "regime_gated":
            gate_core = f"zscore({col_b})" if rng.random() < 0.5 else f"sign({col_b})"
            upper = int(rng.choice((1, 3)))
            return recipe, f"{cs}({wrap_a}) * clip({gate_core}, 0, {upper})"
        wrap_b = col_b if rng.random() < 0.6 else f"ts_mean({col_b}, {window})"
        return recipe, f"zscore({wrap_a}) - zscore({wrap_b})"

    def propose(self, history: list[dict] | None = None) -> dict | None:
        """Try up to 60 draws for a legal unseen recipe; return None if no proposal is available."""
        for _ in range(60):
            recipe, expression = self._sample()
            try:
                compiled = compile_expression(expression)
            except DSLError:
                continue
            sig = compiled.signature()
            if sig in self._seen_signatures:
                continue
            self._seen_signatures.add(sig)
            self._seq += 1
            fams = [self.families.get(c) for c in compiled.columns if self.families.get(c)]
            family = max(set(fams), key=fams.count) if fams else "unclassified"
            return {"name": f"{self.name_prefix}{self._seq:03d}", "expression": expression,
                    "rationale": f"recipe-following random sample of {recipe}",
                    "family": family, "recipe_name": recipe}
        return None


class EnumeratorBaselineGenerator:
    """Canonical-form enumerator: a seed-shuffled,
    NO-REPEAT enumeration of all ordered cross-family (signal, conditioner) column pairs in
    the canonical conditional form ``zscore(A) * sign(B)``.

    It has no truth-pair knowledge, but it *does* privilege the truth's canonical gating form.
    F4b therefore reports it as a separate canonical-form reference, not as an exploration
    upper bound. ``FormBlindEnumeratorGenerator`` is the form-blind comparison."""

    def __init__(self, columns: list[str], families: dict[str, str], rng: np.random.Generator,
                 name_prefix: str = "enum"):
        """Seed the column-pair queue for a fixed canonical form.
        This baseline has form privilege when that canonical form matches injected truth."""
        self.families = {c: families.get(c, "unclassified")
                         for c in columns if column_is_allowed(c)}
        pairs = [(a, b) for a in sorted(self.families) for b in sorted(self.families)
                 if self.families[a] != self.families[b]]
        if not pairs:
            raise ValueError("enumerator needs columns from at least two families")
        order = rng.permutation(len(pairs))
        self._queue = [pairs[i] for i in order]
        self.n_pairs = len(pairs)
        self.name_prefix = name_prefix
        self._seen_signatures: set[str] = set()
        self._seq = 0

    def propose(self, history: list[dict] | None = None) -> dict | None:
        """Pop the next legal unseen canonical-form pair; return None when exhausted."""
        while self._queue:
            signal_col, conditioner_col = self._queue.pop(0)
            expression = f"zscore({signal_col}) * sign({conditioner_col})"
            try:
                compiled = compile_expression(expression)
            except DSLError:
                continue
            sig = compiled.signature()
            if sig in self._seen_signatures:
                continue
            self._seen_signatures.add(sig)
            self._seq += 1
            fams = [self.families.get(c) for c in compiled.columns if self.families.get(c)]
            family = max(set(fams), key=fams.count) if fams else "unclassified"
            return {"name": f"{self.name_prefix}{self._seq:03d}", "expression": expression,
                    "rationale": (f"systematic enumeration {self._seq}/{self.n_pairs}: "
                                  f"{signal_col} conditioned on sign({conditioner_col})"),
                    "family": family, "recipe_name": "conditional_signal"}
        return None


class FormBlindEnumeratorGenerator:
    """Seed-shuffled enumeration of ordered cross-family pairs and four gate forms.

    Each pair/form cell appears once. Unlike the canonical enumerator, this control
    does not restrict generation to the injected truth's privileged form."""

    GATE_FORMS = ("level_gate", "delta_gate", "clip_gate", "spread")

    def __init__(self, columns: list[str], families: dict[str, str],
                 rng: np.random.Generator, name_prefix: str = "fbe"):
        """Construct a seeded queue spanning cross-family pairs and declared gating forms."""
        self.families = {c: families.get(c, "unclassified")
                         for c in columns if column_is_allowed(c)}
        cells = [(a, b, form)
                 for a in sorted(self.families)
                 for b in sorted(self.families)
                 if self.families[a] != self.families[b]
                 for form in self.GATE_FORMS]
        if not cells:
            raise ValueError("form-blind enumerator needs columns from at least two families")
        order = rng.permutation(len(cells))
        self._queue = [cells[i] for i in order]
        self.n_cells = len(cells)
        self.name_prefix = name_prefix
        self._seen_signatures: set[str] = set()
        self._seq = 0

    @staticmethod
    def _render(signal_col: str, conditioner_col: str,
                gate_form: str) -> tuple[str, str]:
        """Render one declared pair/form cell and its recipe label, rejecting unknown forms."""
        if gate_form == "level_gate":
            return (f"zscore({signal_col}) * sign({conditioner_col})",
                    "conditional_signal")
        if gate_form == "delta_gate":
            return (f"zscore({signal_col}) * sign(delta({conditioner_col}, 6))",
                    "conditional_signal")
        if gate_form == "clip_gate":
            return (f"zscore({signal_col}) * clip(sign({conditioner_col}), 0, 3)",
                    "regime_gated")
        if gate_form == "spread":
            return (f"zscore({signal_col}) - zscore({conditioner_col})",
                    "cross_family_spread")
        raise ValueError(f"unknown gate form: {gate_form}")

    def propose(self, history: list[dict] | None = None) -> dict | None:
        """Consume the queue until a legal unseen pair/form candidate is found or exhausted."""
        while self._queue:
            signal_col, conditioner_col, gate_form = self._queue.pop(0)
            expression, recipe = self._render(signal_col, conditioner_col, gate_form)
            try:
                compiled = compile_expression(expression)
            except DSLError:
                continue
            signature = compiled.signature()
            if signature in self._seen_signatures:
                continue
            self._seen_signatures.add(signature)
            self._seq += 1
            fams = [self.families.get(c) for c in compiled.columns if self.families.get(c)]
            family = max(set(fams), key=fams.count) if fams else "unclassified"
            return {
                "name": f"{self.name_prefix}{self._seq:03d}",
                "expression": expression,
                "rationale": (f"form-blind enumeration {self._seq}/{self.n_cells}: "
                              f"{signal_col}, {conditioner_col}, {gate_form}"),
                "family": family,
                "recipe_name": recipe,
                "gate_form": gate_form,
            }
        return None

    def diagnostics(self) -> dict[str, object]:
        return {
            "policy": "uniform_no_repeat_column_pair_x_gate_form",
            "n_cells": self.n_cells,
            "gate_forms": list(self.GATE_FORMS),
        }


def main() -> None:
    """Generate local DSL proposals using a frozen vocabulary, mode and seed.

    Validate a positive proposal budget and save candidate JSON. Existing outputs
    require an explicit overwrite flag so a diagnostic run cannot erase proposals.
    """
    ap = argparse.ArgumentParser(description="Generate baseline candidate factors over the DSL.")
    ap.add_argument("--columns-from", default=str(_PROJECT_DIR / "data_big_v2" / "a_share_factor_panel.csv"),
                    help="panel CSV whose allowed columns become the leaf vocabulary")
    ap.add_argument("--config", default=str(_PROJECT_DIR / "a_share_big_config.json"))
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--mode", choices=["random", "adaptive"], default="random",
                    help="adaptive only matters inside a feedback loop; for a static file use random")
    ap.add_argument("--factor-cols-only", action=argparse.BooleanOptionalAction, default=True,
                    help="restrict leaves to the config's factor_cols "
                         "(--no-factor-cols-only uses all allowed panel columns)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(_PROJECT_DIR / "outputs" / "candidates" / "baseline.json"))
    ap.add_argument("--overwrite", action="store_true",
                    help="explicitly replace an existing candidate file")
    args = ap.parse_args()
    if args.n <= 0:
        ap.error("--n must be a positive integer")
    out = Path(args.out)
    if out.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite candidates: {out}; use --out or --overwrite")

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    families = cfg["family_constraints"]["factor_families"]
    if args.factor_cols_only:
        columns = list(cfg["factor_cols"])
    else:
        columns = [c for c in pd.read_csv(args.columns_from, nrows=0).columns]
    gen = BaselineGenerator(columns, families, np.random.default_rng(args.seed), mode=args.mode)
    candidates = []
    while len(candidates) < args.n:
        cand = gen.propose(history=[])
        if cand is None:
            break
        candidates.append(cand)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with out.open("w" if args.overwrite else "x", encoding="utf-8") as handle:
            handle.write(json.dumps(candidates, indent=2, ensure_ascii=False))
    except FileExistsError as exc:
        raise SystemExit(f"refusing to overwrite candidates: {out}") from exc
    print(f"wrote {len(candidates)} candidates to {args.out}")
    for c in candidates[:8]:
        print(f"  {c['name']}: {c['expression']}   [{c['family']}]")


if __name__ == "__main__":
    main()
