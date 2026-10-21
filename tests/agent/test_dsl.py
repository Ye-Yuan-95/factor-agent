"""DSL sandbox: leakage bans, escape bans, rolling direction (no lookahead)."""
import numpy as np
import pandas as pd
import pytest

from agent.factor_dsl import compile_expression, DSLError


def test_valid_expressions_evaluate(tiny_panel):
    for expr in ["-1 * ts_mean(a, 3)", "rank(b) * rank(c)", "zscore(a) + zscore(b)",
                 "winsor(log(abs(a) + 1), 0.05)", "delta(a, 3) / ts_std(a, 6)", "a"]:
        s = compile_expression(expr).evaluate(tiny_panel)
        assert s.index.equals(tiny_panel.index)
        assert s.notna().sum() > 0


@pytest.mark.parametrize("expr", [
    "rank(forward_return)",              # target leakage
    "fwd_ret + a",                       # forbidden substring
    "__import__('os').system('x')",      # escape
    "delay(a, -2)",                      # forward lag
    "ts_mean(a, 999)",                   # window bound
    "rank(a)[0]",                        # subscript
    "lambda x: x",                       # lambda
    "a ** 2",                            # pow
    "1 + 2",                             # constant only
])
def test_forbidden_expressions_blocked(expr, tiny_panel):
    with pytest.raises(DSLError):     # window/lag bounds raise at evaluate; the rest at compile
        compile_expression(expr).evaluate(tiny_panel)


def test_rolling_uses_no_future_values(tiny_panel):
    """Perturbing month t+1 must not change any ts_* output at months <= t."""
    panel = tiny_panel.sort_values(["asset", "date"]).reset_index(drop=True)
    expr = compile_expression("ts_mean(a, 3) + delta(a, 2)")
    base = expr.evaluate(panel)
    dates = sorted(panel["date"].unique())
    cut = dates[40]
    perturbed = panel.copy()
    perturbed.loc[perturbed["date"] > cut, "a"] += 100.0     # poison the FUTURE only
    after = expr.evaluate(perturbed)
    mask = panel["date"] <= cut
    pd.testing.assert_series_equal(base[mask.values], after[mask.values])


def test_delay_matches_groupby_shift(tiny_panel):
    panel = tiny_panel.sort_values(["asset", "date"]).reset_index(drop=True)
    got = compile_expression("delay(a, 1)").evaluate(panel)
    want = panel.groupby("asset")["a"].shift(1)
    assert np.allclose(got.dropna(), want.dropna())


def test_signature_stable_and_distinct():
    assert compile_expression("rank(a)").signature() == compile_expression("rank(a)").signature()
    assert compile_expression("rank(a)").signature() != compile_expression("rank(b)").signature()


@pytest.mark.parametrize("expr, message", [
    ("ts_zscore(a)", "ts_zscore expects 1 series + 1..1 scalar args, got 1"),
    ("demean(a, 12)", "demean expects 1 series + 0..0 scalar args, got 2"),
    ("clip(a, -1)", "clip expects 1 series + 2..2 scalar args, got 2"),
    ("rank(a, 3)", "rank expects 1 series + 0..0 scalar args, got 2"),
])
def test_function_arity_is_rejected_at_compile_time(expr, message):
    with pytest.raises(DSLError, match=message.replace("+", r"\+")):
        compile_expression(expr)


def test_function_signatures_accept_required_and_optional_scalars():
    for expr in ("ts_zscore(a, 12)", "demean(a)", "winsor(a)",
                 "winsor(a, 0.05)", "clip(a, -1, 1)"):
        assert compile_expression(expr).columns == ("a",)


def test_industry_demean_and_ts_corr_operators():
    import numpy as np, pandas as pd, pytest
    from agent.factor_dsl import compile_expression, DSLError
    rng = np.random.default_rng(3)
    dates = pd.date_range("2020-01-31", periods=30, freq="ME")
    rows = []
    for d in dates:
        for i in range(12):
            rows.append({"date": d, "asset": f"A{i}", "industry": "sec0" if i < 6 else "sec1",
                         "a": rng.normal(), "b": rng.normal(), "forward_return": rng.normal()})
    panel = pd.DataFrame(rows)
    out = compile_expression("industry_demean(a)").evaluate(panel)
    gm = panel.assign(v=out).groupby(["date", "industry"])["v"].mean().abs().max()
    assert gm < 1e-12                          # demeaned within every (date, industry) cell
    with pytest.raises(DSLError, match="industry"):
        compile_expression("industry_demean(a)").evaluate(panel.drop(columns=["industry"]))
    out = compile_expression("ts_corr(a, b, 6)").evaluate(panel)
    p2 = panel.sort_values(["asset", "date"]).reset_index()
    got = out.loc[p2["index"]].to_numpy()
    exp = pd.concat([g["a"].rolling(6, min_periods=3).corr(g["b"])
                     for _, g in p2.groupby("asset")]).to_numpy()
    mask = np.isfinite(exp)
    assert mask.sum() > 100 and np.allclose(got[mask], exp[mask])
    for bad in ("ts_corr(a, 6)", "industry_demean(a, 3)"):
        with pytest.raises(DSLError):
            compile_expression(bad)


def test_vocabulary_modes_and_new_signatures():
    import pytest
    from agent.generator_contract import (FUNCTION_SIGNATURES, GeneratorSpec,
                                          build_vocabulary)
    cfg = {"factor_cols": ["a", "b"],
           "family_constraints": {"factor_families": {"a": "f1", "b": "f2"}}}
    assert build_vocabulary(cfg, mode="base") == ["a", "b"]
    ext = build_vocabulary(cfg, mode="extended", panel_columns=["a", "b", "close", "volume"])
    assert ext == ["a", "b", "close", "volume"]   # raw cols only when present in the panel
    with pytest.raises(ValueError):
        build_vocabulary(cfg, mode="wild")
    spec = GeneratorSpec.from_config(cfg, columns=ext)
    assert "close" in spec.allowed_columns
    assert spec.column_families["close"] == "unclassified"
    assert FUNCTION_SIGNATURES["ts_corr"] == "ts_corr(series, series, window)"
    assert "industry_demean" in spec.function_signatures
