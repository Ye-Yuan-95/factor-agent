"""Minimal factor-expression DSL + sandboxed executor.

Compute candidate expressions against the panel built by
`factor_research.build_a_share_factor_panel`, using a restricted expression grammar:

    expr = "rank(ts_mean(illiq, 6)) * sign(mom6m)"
    compiled = compile_expression(expr)
    series   = compiled.evaluate(panel)        # aligned to panel.index

Execution constraints:
  * AST whitelist: only Call / Name / Constant / BinOp(+ - * /) / UnaryOp(-) nodes survive parsing.
    No attributes, subscripts, lambdas, comprehensions, keywords -> no sandbox escape surface.
  * Column whitelist: expressions may reference numeric point-in-time panel columns only.
    `forward_return` (and anything matching forward/future/target) is FORBIDDEN -> no target leakage.
  * Time direction: every time-series operator looks strictly backward (rolling/shift on past rows of
    the monthly panel, grouped per asset). Windows are bounded ints, so no unbounded lookback.
  * Complexity metrics (n_nodes/depth) are computed for every expression -> the AlphaAgent-style
    complexity guard, recorded on the FactorCard and cappable by the oracle.
  * `signature()` = sha1 of the canonical AST dump -> exact-duplicate detection for the trial-counting
    policy; near-duplicates are folded via return-series correlation in agent.memory.

Operator families:
  cross-sectional (per date): rank, zscore, demean, winsor
  time-series (per asset)   : ts_mean, ts_std, ts_sum, ts_min, ts_max, ts_rank, ts_zscore,
                              delay, delta
  elementwise               : log, sqrt, abs, sign, clip
  arithmetic                : + - * / and unary -
"""
from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


class DSLError(ValueError):
    """Any compile- or run-time violation of the DSL sandbox."""


MAX_EXPRESSION_CHARS = 400
MAX_AST_NODES = 80
MAX_DEPTH = 12
MAX_WINDOW = 60          # months of lookback on the monthly panel
MAX_ABS_CONST = 1e6

# Columns an expression may never reference (target / bookkeeping / non-numeric).
FORBIDDEN_SUBSTRINGS = ("forward", "future", "target", "fwd")
FORBIDDEN_COLUMNS = frozenset({
    "date", "asset", "industry", "report_date", "available_date", "ym",
})

_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div)
_ALLOWED_UNARY = (ast.USub, ast.UAdd)


# --------------------------------------------------------------------------- #
# Operator implementations. Each takes/returns pd.Series aligned to the panel  #
# index; `ctx` carries the panel and its date/asset group keys.                #
# --------------------------------------------------------------------------- #
def _by_date(ctx, s: pd.Series):
    return s.groupby(ctx["date_key"], group_keys=False)


def _by_asset(ctx, s: pd.Series):
    return s.groupby(ctx["asset_key"], group_keys=False)


def _op_rank(ctx, x):        # cross-sectional percentile rank in [0,1]
    return _by_date(ctx, x).rank(pct=True)


def _op_zscore(ctx, x):
    def z(g):
        sd = g.std(ddof=0)
        return (g - g.mean()) / sd if sd and np.isfinite(sd) and sd > 0 else g * 0.0
    return _by_date(ctx, x).transform(z)


def _op_demean(ctx, x):
    return _by_date(ctx, x).transform(lambda g: g - g.mean())


def _op_winsor(ctx, x, p=0.05):
    p = _check_fraction(p, "winsor(p)")
    def w(g):
        lo, hi = g.quantile(p), g.quantile(1.0 - p)
        return g.clip(lower=lo, upper=hi)
    return _by_date(ctx, x).transform(w)


def _rolling(ctx, x, window, fn):
    w = _check_window(window)
    return _by_asset(ctx, x).transform(
        lambda g: getattr(g.rolling(w, min_periods=max(2, w // 2)), fn)())


def _op_ts_mean(ctx, x, w):  return _rolling(ctx, x, w, "mean")
def _op_ts_std(ctx, x, w):   return _rolling(ctx, x, w, "std")
def _op_ts_sum(ctx, x, w):   return _rolling(ctx, x, w, "sum")
def _op_ts_min(ctx, x, w):   return _rolling(ctx, x, w, "min")
def _op_ts_max(ctx, x, w):   return _rolling(ctx, x, w, "max")


def _op_ts_rank(ctx, x, w):  # position of today's value within its own trailing window, in [0,1]
    w_ = _check_window(w)
    def tr(g):
        return g.rolling(w_, min_periods=max(2, w_ // 2)).apply(
            lambda a: (a[:-1] <= a[-1]).mean() if len(a) > 1 else np.nan, raw=True)
    return _by_asset(ctx, x).transform(tr)


def _op_ts_zscore(ctx, x, w):
    mean = _rolling(ctx, x, w, "mean")
    std = _rolling(ctx, x, w, "std")
    return (x - mean) / std.replace(0.0, np.nan)


def _op_delay(ctx, x, k):
    k = _check_lag(k)
    return _by_asset(ctx, x).shift(k)


def _op_delta(ctx, x, k):
    k = _check_lag(k)
    return x - _by_asset(ctx, x).shift(k)


def _op_log(ctx, x):   return np.log(x.where(x > 0))          # non-positive -> NaN
def _op_sqrt(ctx, x):  return np.sqrt(x.where(x >= 0))
def _op_abs(ctx, x):   return x.abs()
def _op_sign(ctx, x):  return np.sign(x)


def _op_clip(ctx, x, lo, hi):
    lo, hi = _check_const(lo, "clip(lo)"), _check_const(hi, "clip(hi)")
    if lo >= hi:
        raise DSLError(f"clip: lo must be < hi, got ({lo}, {hi})")
    return x.clip(lower=lo, upper=hi)


def _op_industry_demean(ctx, x):
    """Cross-sectional demean within each (date, industry) group -- the industry-neutral
    version of a signal. Requires an 'industry' panel column;
    panels without one (e.g. Gen-1 synthetic) fail at evaluate time, which costs a generation
    attempt but never a statistical trial."""
    industry = ctx.get("industry_key")
    if industry is None:
        raise DSLError("industry_demean requires an 'industry' column in the panel")
    return x.groupby([ctx["date_key"], industry], group_keys=False).transform(
        lambda g: g - g.mean())


def _op_ts_corr(ctx, x, y, w):
    """Rolling correlation of two series per asset over a trailing window (first two-series
    operator). Backward-looking only, like every ts_* operator."""
    w_ = _check_window(w)

    def corr(g):
        return g.rolling(w_, min_periods=max(3, w_ // 2)).corr(y.loc[g.index])

    return _by_asset(ctx, x).transform(corr)


FUNCTIONS = {
    # name: (fn, n_series_args, n_scalar_args_min, n_scalar_args_max)
    "rank":      (_op_rank, 1, 0, 0),
    "zscore":    (_op_zscore, 1, 0, 0),
    "demean":    (_op_demean, 1, 0, 0),
    "winsor":    (_op_winsor, 1, 0, 1),
    "ts_mean":   (_op_ts_mean, 1, 1, 1),
    "ts_std":    (_op_ts_std, 1, 1, 1),
    "ts_sum":    (_op_ts_sum, 1, 1, 1),
    "ts_min":    (_op_ts_min, 1, 1, 1),
    "ts_max":    (_op_ts_max, 1, 1, 1),
    "ts_rank":   (_op_ts_rank, 1, 1, 1),
    "ts_zscore": (_op_ts_zscore, 1, 1, 1),
    "delay":     (_op_delay, 1, 1, 1),
    "delta":     (_op_delta, 1, 1, 1),
    "log":       (_op_log, 1, 0, 0),
    "sqrt":      (_op_sqrt, 1, 0, 0),
    "abs":       (_op_abs, 1, 0, 0),
    "sign":      (_op_sign, 1, 0, 0),
    "clip":      (_op_clip, 1, 2, 2),
    "industry_demean": (_op_industry_demean, 1, 0, 0),
    "ts_corr":   (_op_ts_corr, 2, 1, 1),
}


def _check_window(w) -> int:
    if not isinstance(w, (int, float)) or float(w) != int(w):
        raise DSLError(f"window must be an integer, got {w!r}")
    w = int(w)
    if not 2 <= w <= MAX_WINDOW:
        raise DSLError(f"window must be in [2, {MAX_WINDOW}], got {w}")
    return w


def _check_lag(k) -> int:
    if not isinstance(k, (int, float)) or float(k) != int(k):
        raise DSLError(f"lag must be an integer, got {k!r}")
    k = int(k)
    if not 1 <= k <= MAX_WINDOW:
        raise DSLError(f"lag must be in [1, {MAX_WINDOW}] (backward only), got {k}")
    return k


def _check_fraction(p, what) -> float:
    if not isinstance(p, (int, float)) or not 0.0 < float(p) < 0.5:
        raise DSLError(f"{what} must be a fraction in (0, 0.5), got {p!r}")
    return float(p)


def _check_const(c, what) -> float:
    if not isinstance(c, (int, float)) or not np.isfinite(c) or abs(c) > MAX_ABS_CONST:
        raise DSLError(f"{what} must be a finite number with |c| <= {MAX_ABS_CONST}, got {c!r}")
    return float(c)


def column_is_allowed(name: str) -> bool:
    low = name.lower()
    if name in FORBIDDEN_COLUMNS:
        return False
    return not any(bad in low for bad in FORBIDDEN_SUBSTRINGS)


# --------------------------------------------------------------------------- #
# Compilation: parse -> whitelist walk -> CompiledExpression.                  #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CompiledExpression:
    source: str
    tree: ast.expression = field(repr=False)
    columns: tuple[str, ...]
    n_nodes: int
    depth: int

    def signature(self) -> str:
        canonical = ast.dump(self.tree, annotate_fields=False)
        return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]

    def complexity(self) -> dict:
        return {"n_nodes": self.n_nodes, "depth": self.depth, "n_columns": len(self.columns)}

    def evaluate(self, panel: pd.DataFrame, date_col: str = "date", asset_col: str = "asset") -> pd.Series:
        """Compute the factor on `panel` (long format). Rows must be sortable by (asset, date);
        the result is returned aligned to panel.index regardless of input order."""
        missing = [c for c in self.columns if c not in panel.columns]
        if missing:
            raise DSLError(f"expression references columns missing from the panel: {missing}")
        order = panel.sort_values([asset_col, date_col]).index
        sorted_panel = panel.loc[order]
        ctx = {
            "date_key": sorted_panel[date_col],
            "asset_key": sorted_panel[asset_col],
            "industry_key": sorted_panel["industry"] if "industry" in sorted_panel.columns else None,
        }
        cols = {
            c: pd.to_numeric(sorted_panel[c], errors="coerce") for c in self.columns
        }
        result = _eval_node(self.tree.body, ctx, cols, sorted_panel.index)
        if np.isscalar(result):
            raise DSLError("expression is a constant; it must reference at least one panel column")
        result = pd.Series(result, index=sorted_panel.index, dtype=float)
        result = result.replace([np.inf, -np.inf], np.nan)
        return result.reindex(panel.index)


def compile_expression(source: str) -> CompiledExpression:
    """Parse and validate an expression against the AST/operator whitelist and resource bounds.
    Return a compiled evaluator; arbitrary Python execution and forbidden columns are rejected."""
    if not isinstance(source, str) or not source.strip():
        raise DSLError("expression must be a non-empty string")
    if len(source) > MAX_EXPRESSION_CHARS:
        raise DSLError(f"expression too long ({len(source)} > {MAX_EXPRESSION_CHARS} chars)")
    try:
        tree = ast.parse(source.strip(), mode="eval")
    except SyntaxError as exc:
        raise DSLError(f"syntax error: {exc}") from exc

    columns: set[str] = set()
    n_nodes = 0

    def walk(node: ast.AST, depth: int, as_call_func: bool = False) -> None:
        nonlocal n_nodes
        n_nodes += 1
        if n_nodes > MAX_AST_NODES:
            raise DSLError(f"expression too complex (> {MAX_AST_NODES} AST nodes)")
        if depth > MAX_DEPTH:
            raise DSLError(f"expression too deeply nested (> {MAX_DEPTH})")
        if isinstance(node, ast.Expression):
            walk(node.body, depth + 1)
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise DSLError("only plain function calls like rank(x) are allowed")
            if node.keywords:
                raise DSLError("keyword arguments are not allowed; use positional arguments")
            fname = node.func.id
            if fname not in FUNCTIONS:
                raise DSLError(f"unknown function '{fname}'; allowed: {sorted(FUNCTIONS)}")
            _, n_series, scalar_min, scalar_max = FUNCTIONS[fname]
            n_args = len(node.args)
            if not n_series + scalar_min <= n_args <= n_series + scalar_max:
                raise DSLError(
                    f"{fname} expects {n_series} series + {scalar_min}..{scalar_max} "
                    f"scalar args, got {n_args}")
            walk(node.func, depth + 1, as_call_func=True)
            for arg in node.args:
                walk(arg, depth + 1)
        elif isinstance(node, ast.Name):
            if not as_call_func:
                if node.id in FUNCTIONS:
                    raise DSLError(f"'{node.id}' is a function; call it like {node.id}(...)")
                if not column_is_allowed(node.id):
                    raise DSLError(f"column '{node.id}' is forbidden (target/bookkeeping leakage guard)")
                columns.add(node.id)
        elif isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
                raise DSLError(f"only numeric constants are allowed, got {node.value!r}")
            _check_const(node.value, "constant")
        elif isinstance(node, ast.BinOp):
            if not isinstance(node.op, _ALLOWED_BINOPS):
                raise DSLError(f"operator {type(node.op).__name__} is not allowed (use + - * /)")
            walk(node.left, depth + 1)
            walk(node.right, depth + 1)
        elif isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, _ALLOWED_UNARY):
                raise DSLError(f"unary {type(node.op).__name__} is not allowed")
            walk(node.operand, depth + 1)
        else:
            raise DSLError(f"AST node {type(node).__name__} is not allowed")

    walk(tree, 0)
    if not columns:
        raise DSLError("expression must reference at least one panel column")
    return CompiledExpression(
        source=source.strip(), tree=tree, columns=tuple(sorted(columns)),
        n_nodes=n_nodes, depth=_tree_depth(tree),
    )


def _tree_depth(node: ast.AST) -> int:
    children = list(ast.iter_child_nodes(node))
    if not children:
        return 1
    return 1 + max(_tree_depth(c) for c in children)


def _eval_node(node: ast.AST, ctx: dict, cols: dict, index) -> pd.Series | float:
    """Recursively evaluate a previously validated AST using registered operators and aligned series."""
    if isinstance(node, ast.Call):
        fn, n_series, s_min, s_max = FUNCTIONS[node.func.id]
        args = [_eval_node(a, ctx, cols, index) for a in node.args]
        series_args = args[:n_series]
        scalar_args = args[n_series:]
        if len(scalar_args) < s_min or len(scalar_args) > s_max:
            raise DSLError(f"{node.func.id} expects {n_series} series + "
                           f"{s_min}..{s_max} scalar args, got {len(args)}")
        for i, a in enumerate(series_args):
            if np.isscalar(a):
                raise DSLError(f"{node.func.id}: argument {i + 1} must be a series, not a constant")
        for a in scalar_args:
            if not np.isscalar(a):
                raise DSLError(f"{node.func.id}: window/scalar arguments must be numeric constants")
        return fn(ctx, *series_args, *scalar_args)
    if isinstance(node, ast.Name):
        return cols[node.id]
    if isinstance(node, ast.Constant):
        return float(node.value)
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, ctx, cols, index)
        right = _eval_node(node.right, ctx, cols, index)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if np.isscalar(right):
                if right == 0:
                    raise DSLError("division by constant zero")
                return left / right
            return left / right.replace(0.0, np.nan)
    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand, ctx, cols, index)
        return -operand if isinstance(node.op, ast.USub) else operand
    raise DSLError(f"unexpected node {type(node).__name__}")  # unreachable after compile walk
