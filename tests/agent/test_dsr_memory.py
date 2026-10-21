"""DSR math + Memory counting policy and DATE-ALIGNED N_eff."""
import numpy as np

from agent.deflated_sharpe import (deflated_sharpe, effective_n_trials,
                                   expected_max_sharpe, probabilistic_sharpe)
from agent.memory import MemoryStore


def test_dsr_basics():
    assert expected_max_sharpe(1, 0.5) == 0.0
    assert 0 < expected_max_sharpe(10, 0.04) < expected_max_sharpe(100, 0.04)
    assert probabilistic_sharpe(0.5, 0.0, 120) > 0.99
    d = deflated_sharpe(0.3, [0.1, -0.05, 0.2], 3, 120, skew=-0.5, kurt=5)
    assert 0 <= d["deflated_sharpe"] <= 1 and d["sharpe_ref"] > 0
    corr = np.eye(4); corr[0, 1] = corr[1, 0] = 0.9; corr[2, 3] = corr[3, 2] = 0.95
    assert effective_n_trials(corr, 0.7) == 2


def _record(mem, name, sig, values, dates, decision="REJECT"):
    return mem.record_trial(name=name, expression=name, signature=sig, family="f",
                            universe="u", sharpe=0.1, mean_ic=0.01, monthly_ls=values,
                            monthly_ls_dates=dates, decision=decision, reject_reason=None)


def test_counting_policy(tmp_path):
    mem = MemoryStore(tmp_path)
    _record(mem, "x1", "sigA", [0.1] * 8, None)
    assert mem.n_trials() == 1
    mem.record_failure(name="bad", expression="rank(", error="syntax", universe="u")
    assert mem.n_trials() == 1                       # sandbox failure NOT counted
    _record(mem, "x1b", "sigA", [0.1] * 8, None)     # exact dup (same signature+universe)
    assert mem.n_trials() == 1
    _record(mem, "x2", "sigB", [0.2] * 8, None)
    assert mem.n_trials() == 2


def test_neff_date_alignment(tmp_path):
    """A middle-month gap must NOT break the correlation: with dates the two identical series
    fold into ONE cluster; naive tail alignment would call them independent."""
    mem = MemoryStore(tmp_path)
    values = [1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0, 5.0, -5.0]
    dates = [f"2020-{m:02d}-28" for m in range(1, 11)]
    _record(mem, "full", "s1", values, dates)
    gap_vals = values[:4] + values[5:]               # month 5 missing
    gap_dates = dates[:4] + dates[5:]
    _record(mem, "gap", "s2", gap_vals, gap_dates)
    assert mem.n_eff() == 1                           # identical on the date intersection
    rng = np.random.default_rng(0)
    _record(mem, "indep", "s3", list(rng.standard_normal(10)), dates)
    assert mem.n_eff() == 2
