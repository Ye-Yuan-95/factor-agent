"""Input, leakage, family-cap, month-predicate and token contracts.

These tests run with the dev extra alone. Solver-dependent coverage lives in
test_portfolio_contracts.py and requires the optional cvxpy dependency."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from factor_research.portfolio_construction import (  # noqa: E402
    cap_long_only, compute_scores, month_is_tradable)
from agent.l1_runner import _is_true  # noqa: E402


# ------------------------------------------------------------------ point-in-time scores
def _panel(n_assets: int = 8, n_months: int = 12, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-31", periods=n_months, freq="ME")
    rows = [{"date": d, "asset": f"A{i}", "f1": rng.normal(), "f2": rng.normal()}
            for d in dates for i in range(n_assets)]
    return pd.DataFrame(rows)


WEIGHTS = pd.Series({"f1": 0.7, "f2": -0.3})


def test_scores_do_not_depend_on_future_data():
    """The leak in one assertion: corrupt the last month's factor values and every
    earlier month's score must be bit-identical. Whole-panel standardization fails this."""
    panel = _panel()
    base = compute_scores(panel, WEIGHTS, ["f1", "f2"])
    corrupted = panel.copy()
    last = corrupted["date"] == corrupted["date"].max()
    corrupted.loc[last, ["f1", "f2"]] = corrupted.loc[last, ["f1", "f2"]] * 100 + 7
    after = compute_scores(corrupted, WEIGHTS, ["f1", "f2"])
    early = panel["date"] < panel["date"].max()
    assert np.allclose(base[early], after[early], equal_nan=True)
    assert not np.allclose(base[~early], after[~early])  # the change itself was seen


def test_scores_match_expanding_moments_by_hand():
    panel = _panel(n_assets=4, n_months=3, seed=1)
    got = compute_scores(panel, pd.Series({"f1": 1.0}), ["f1"])
    second = panel["date"] == sorted(panel["date"].unique())[1]
    visible = panel[panel["date"] <= panel.loc[second, "date"].iloc[0]]["f1"]
    expected = (panel.loc[second, "f1"] - visible.mean()) / visible.std(ddof=0)
    assert np.allclose(got[second], expected)


def test_first_date_scores_are_finite():
    panel = _panel(n_assets=5, n_months=2)
    scores = compute_scores(panel, WEIGHTS, ["f1", "f2"])
    assert np.isfinite(scores).all()


# ------------------------------------------------------------------ cap discipline
@pytest.mark.parametrize("raw,w_max", [
    (np.array([0.9, 0.05, 0.05]), 0.5),
    (np.array([1.0, 0.0, 0.0, 0.0]), 0.3),
    (np.ones(20), 0.05),
    (np.array([np.nan, 1.0, 2.0]), 0.5),
    # When all tilted names hit the cap, redistribute the remaining budget to other names.
    (np.array([0.5, 0.3, 0.2, 0.0, 0.0, 0.0]), 0.2),
])
def test_cap_long_only_always_feasible(raw, w_max):
    w = cap_long_only(raw, w_max)
    assert abs(w.sum() - 1.0) < 1e-9
    assert w.max() <= w_max + 1e-9
    assert (w >= -1e-12).all()


def test_cap_long_only_rejects_an_infeasible_cap():
    """Ten names with a five-percent cap cannot sum to one; direct calls must refuse."""
    with pytest.raises(ValueError, match="infeasible"):
        cap_long_only(np.ones(10), 0.05)
    with pytest.raises(ValueError, match="infeasible"):
        cap_long_only(np.ones(0), 0.5)
    # Non-finite or non-positive caps are a parameter error, reported as such rather
    # than as constraint infeasibility. NaN comparisons are all False; without an
    # isfinite check the NaN case returned [nan]*5.
    for bad in (0.0, float("nan"), float("-inf"), float("inf")):
        with pytest.raises(ValueError, match="finite positive"):
            cap_long_only(np.ones(5), bad)


def test_month_is_tradable_thresholds():
    """The predicate every walk-forward caller shares: thin months and cap-infeasible
    months are the same skip, decided in one place."""
    assert month_is_tradable(50, 0.02)
    assert month_is_tradable(20, 0.05)
    assert not month_is_tradable(9, 0.5)        # too thin even though the cap is fine
    assert not month_is_tradable(49, 0.02)      # 49 * 0.02 < 1: the cap cannot sum to 1
    assert not month_is_tradable(10, 0.05)      # the infeasible ten-name, five-percent cap case
    assert not month_is_tradable(50, float("nan"))  # NaN comparisons are False -> skip
    # The tradability guard and solver both refuse a nonfinite cap.
    assert not month_is_tradable(50, float("inf"))


# ------------------------------------------------------------------ truth-summary tokens
def test_truth_match_is_token_level_not_substring():
    assert not _is_true({"expression": "rank(std_dolvol)"}, ["dolvol"])
    assert _is_true({"expression": "rank(dolvol) * sign(ep)"}, ["dolvol"])
    assert _is_true({"expression": "zscore(ep)*sign(illiq)"}, ["illiq"])
    assert not _is_true({"expression": "zscore(epq)"}, ["ep"])
