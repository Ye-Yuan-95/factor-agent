"""Deflated Sharpe Ratio after Bailey and Lopez de Prado (2014).

Inputs are observed per-period Sharpe, effective independent trial count,
cross-trial Sharpe variance, sample length and return skew/kurtosis. Memory logs
all counted trials, including rejections, and clusters near-duplicate return
series for the effective trial count.

BH screening addresses a same-batch set of hypotheses; DSR adjusts a Sharpe
benchmark for search multiplicity. Combining them does not by itself establish
error control for the adaptive discovery loop."""
from __future__ import annotations

from statistics import NormalDist

import numpy as np

EULER_GAMMA = 0.5772156649015329
_NORMAL = NormalDist()


def expected_max_sharpe(n_eff: float, var_trial_sharpes: float) -> float:
    """E[max SR] over n_eff independent zero-true-SR trials with the given cross-trial variance
    (Bailey & Lopez de Prado 2014, Gumbel-limit EV; grows like sqrt(2 ln N)):
        E[max] ~= sqrt(V) * ((1-gamma) * z(1 - 1/N) + gamma * z(1 - 1/(N*e)))"""
    if n_eff is None or n_eff <= 1 or not np.isfinite(n_eff):
        return 0.0
    if var_trial_sharpes is None or not np.isfinite(var_trial_sharpes) or var_trial_sharpes <= 0:
        return 0.0
    n = float(n_eff)
    z1 = _NORMAL.inv_cdf(1.0 - 1.0 / n)
    z2 = _NORMAL.inv_cdf(1.0 - 1.0 / (n * np.e))
    return float(np.sqrt(var_trial_sharpes) * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def probabilistic_sharpe(sharpe: float, sharpe_ref: float, n_obs: int,
                         skew: float = 0.0, kurt: float = 3.0) -> float:
    """PSR = P[true SR > sharpe_ref], with the non-normality-adjusted standard error.
    `kurt` is NON-excess kurtosis (normal = 3). Returns NaN when undefined."""
    if n_obs is None or n_obs < 3 or not np.isfinite(sharpe):
        return float("nan")
    skew = float(skew) if np.isfinite(skew) else 0.0
    kurt = float(kurt) if np.isfinite(kurt) else 3.0
    denom = 1.0 - skew * sharpe + (kurt - 1.0) / 4.0 * sharpe * sharpe
    if denom <= 0:
        # variance estimate of the SR breaks down (extreme skew/kurt vs SR); be conservative
        return 0.0
    z = (sharpe - sharpe_ref) * np.sqrt(n_obs - 1.0) / np.sqrt(denom)
    return float(_NORMAL.cdf(z))


def deflated_sharpe(sharpe: float, trial_sharpes, n_eff: float, n_obs: int,
                    skew: float = 0.0, kurt: float = 3.0) -> dict:
    """DSR = PSR evaluated at sharpe_ref = E[max SR | n_eff trials, Var(trial SRs)].
    trial_sharpes: Memory's per-trial history (counted trials, INCLUDING rejected).
    Returns the DSR, the deflation benchmark, and the inputs (audit trail)."""
    trials = np.asarray([s for s in (trial_sharpes or []) if s is not None and np.isfinite(s)],
                        dtype=float)
    var_trials = float(np.var(trials, ddof=1)) if len(trials) >= 2 else 0.0
    n_eff_used = max(float(n_eff or 1.0), 1.0)
    sr_ref = expected_max_sharpe(n_eff_used, var_trials)
    dsr = probabilistic_sharpe(sharpe, sr_ref, n_obs, skew=skew, kurt=kurt)
    return {
        "deflated_sharpe": dsr,
        "sharpe_ref": sr_ref,
        "observed_sharpe": float(sharpe) if np.isfinite(sharpe) else None,
        "n_eff": n_eff_used,
        "n_trial_history": int(len(trials)),
        "var_trial_sharpes": var_trials,
        "n_obs": int(n_obs) if n_obs else 0,
        "skew": float(skew) if np.isfinite(skew) else None,
        "kurt": float(kurt) if np.isfinite(kurt) else None,
    }


def effective_n_trials(corr: np.ndarray, threshold: float = 0.7) -> int:
    """Estimate effective trial count as connected components of absolute correlations.

    Edges require |corr| >= threshold; unavailable correlations count as zero.
    The cluster count is a dependence approximation, not a validated universal
    correction for adaptive discovery."""
    corr = np.asarray(corr, dtype=float)
    n = corr.shape[0]
    if n == 0:
        return 0
    if corr.shape != (n, n):
        raise ValueError(f"corr must be square, got {corr.shape}")
    absd = np.abs(np.nan_to_num(corr, nan=0.0))
    cluster = [-1] * n
    n_clusters = 0
    for i in range(n):
        if cluster[i] >= 0:
            continue
        cluster[i] = n_clusters
        stack = [i]
        while stack:
            j = stack.pop()
            for k in range(n):
                if cluster[k] < 0 and absd[j, k] >= threshold:
                    cluster[k] = n_clusters
                    stack.append(k)
        n_clusters += 1
    return n_clusters
