"""Portfolio solver contracts requiring the optional cvxpy dependency.

Input and leakage checks that need no solver live in test_input_contracts.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

cp = pytest.importorskip("cvxpy", reason="solver tests need the [opt] extra")

from factor_research.portfolio_construction import solve_long_only_mvo  # noqa: E402


def test_infeasible_cap_raises_instead_of_pretending():
    with pytest.raises(ValueError, match="infeasible"):
        solve_long_only_mvo(np.ones(10), np.eye(10), 10.0, 0.05)


def test_fallback_respects_the_cap(monkeypatch):
    """Solver failure must still produce feasible capped portfolio weights."""
    def boom(self, *args, **kwargs):
        raise RuntimeError("solver unavailable")
    monkeypatch.setattr(cp.Problem, "solve", boom)
    mu = np.linspace(1.0, 3.0, 30)
    w = solve_long_only_mvo(mu, np.eye(30), 10.0, 0.05)
    assert w.max() <= 0.05 + 1e-9
    assert abs(w.sum() - 1.0) < 1e-9
    assert (w >= -1e-12).all()


def test_solver_failure_returns_a_feasible_fallback(monkeypatch):
    """Solver exceptions must produce the feasible fallback without writing read-only solver status."""
    def boom(self, *args, **kwargs):
        raise RuntimeError("solver unavailable")
    monkeypatch.setattr(cp.Problem, "solve", boom)
    w = solve_long_only_mvo(np.ones(25), np.eye(25), 10.0, 0.05)
    assert abs(w.sum() - 1.0) < 1e-9


def test_solved_weights_also_projected_under_cap():
    rng = np.random.default_rng(3)
    mu = rng.normal(size=40)
    a = rng.normal(size=(60, 40))
    sigma = a.T @ a / 60 + np.eye(40) * 1e-4
    w = solve_long_only_mvo(mu, sigma, 5.0, 0.04)
    assert w.max() <= 0.04 + 1e-9
    assert abs(w.sum() - 1.0) < 1e-8


def test_resolve_solver_auto_prefers_available(monkeypatch):
    from factor_research.solve_unified_v2 import resolve_solver
    monkeypatch.setattr(cp, "installed_solvers", lambda: ["ECOS", "SCS", "SCIP", "ECOS_BB"])
    assert resolve_solver("auto") == "SCIP"
    monkeypatch.setattr(cp, "installed_solvers", lambda: ["MOSEK", "SCIP"])
    assert resolve_solver("auto") == "MOSEK"
    monkeypatch.setattr(cp, "installed_solvers", lambda: ["ECOS_BB"])
    assert resolve_solver("auto") == "ECOS_BB"
    assert resolve_solver("SCIP") == "SCIP"     # explicit choice passes through untouched
    monkeypatch.setattr(cp, "installed_solvers", lambda: ["ECOS"])
    with pytest.raises(SystemExit):
        resolve_solver("auto")


def test_solver_options_route_limits_to_each_solver():
    """MOSEK-only limits meant `auto` (SCIP in a clean [opt] env) ran with no ceiling.
    Every solver must receive its own limit/tolerance spelling, and the caller must be
    told when no wall-clock limit is actually enforced (ECOS_BB)."""
    from factor_research.solve_unified_v2 import solver_options
    kwargs, enforced = solver_options("MOSEK", 45.0, 1e-3)
    assert enforced is True
    assert kwargs["mosek_params"]["MSK_DPAR_MIO_MAX_TIME"] == 45.0
    assert kwargs["mosek_params"]["MSK_DPAR_MIO_TOL_REL_GAP"] == 1e-3
    kwargs, enforced = solver_options("SCIP", 45.0, 1e-3)
    assert enforced is True
    assert kwargs["scip_params"] == {"limits/time": 45.0, "limits/gap": 1e-3}
    kwargs, enforced = solver_options("ECOS_BB", 45.0, 1e-3)
    assert enforced is False          # honesty flag: iterations-capped, not time-capped
    assert kwargs["mi_rel_eps"] == 1e-3


def test_classify_outcome_gates_the_exit():
    """Only OPTIMAL / FEASIBLE_INACCURATE are usable; infeasible-with-zero-vector had
    been exiting 0 and miqp_recheck wrapped it as ok=true."""
    from factor_research.solve_unified_v2 import classify_outcome
    assert classify_outcome("optimal", True) == ("OPTIMAL", True)
    assert classify_outcome("optimal_inaccurate", True) == ("FEASIBLE_INACCURATE", True)
    assert classify_outcome("optimal_inaccurate", False) == ("NO_INCUMBENT", False)
    assert classify_outcome("infeasible", False) == ("INFEASIBLE", False)
    assert classify_outcome("unbounded", False) == ("UNBOUNDED", False)
    assert classify_outcome(None, False) == ("NO_INCUMBENT", False)
    # "optimal" is a claim about the model; without an incumbent there is nothing to
    # consume, so it is not a success
    assert classify_outcome("optimal", False) == ("NO_INCUMBENT", False)
    # SCIP's own termination reason wins: a solution found under a hit time limit is
    # usable but is not a certified optimum
    assert classify_outcome("optimal_inaccurate", True, "timelimit") == (
        "FEASIBLE_LIMIT", True)
    assert classify_outcome("optimal", True, "optimal") == ("OPTIMAL", True)


def test_solver_options_reject_degenerate_limits():
    """A zero solver time limit produces a controlled failure."""
    from factor_research.solve_unified_v2 import solver_options
    for bad in (0.0, -5.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="time_limit"):
            solver_options("SCIP", bad, 1e-2)
    with pytest.raises(ValueError, match="mip_gap"):
        solver_options("SCIP", 60.0, -1e-3)
    solver_options("SCIP", 60.0, 0.0)   # certified-optimal gap is legitimate


def test_walkforward_split_follows_label_settlement():
    """Segments follow WHEN THE COHORT SETTLES, not the signal month: a 2022-12 signal
    whose return is made of 2023 prices is test data. A cohort with no settlement
    information is test-side -- unknown availability takes the conservative side of a
    lockbox boundary."""
    from factor_research.hybrid_walkforward import split_of
    settle = {pd.Period("2019-12", "M"): pd.Timestamp("2019-12-31"),
              pd.Period("2020-01", "M"): pd.Timestamp("2020-02-28"),
              pd.Period("2022-12", "M"): pd.Timestamp("2023-01-31"),
              pd.Period("2022-11", "M"): pd.Timestamp("2022-12-31")}
    assert split_of(pd.Period("2019-12", "M"), settle) == "train"
    assert split_of(pd.Period("2020-01", "M"), settle) == "validation"
    assert split_of(pd.Period("2022-11", "M"), settle) == "validation"
    assert split_of(pd.Period("2022-12", "M"), settle) == "test"   # settles in 2023
    assert split_of(pd.Period("2021-06", "M"), {"other": None}) == "test"
