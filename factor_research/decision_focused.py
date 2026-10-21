"""Decision-focused weighting for explicitly supplied training observations.

The caller must purge unavailable labels before fitting. These helpers optimize
a softmax portfolio's in-sample Sharpe with optional L1 regularization; they do
not choose evaluation splits or establish out-of-sample performance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

def softmax_weights(score, kappa):
    """Convert scores to stable softmax weights without rescaling factor coefficients."""
    # NOTE: do NOT z-score the score here -- that would make the policy invariant to the scale
    # of b and defeat the L1 sparsity. Use the raw factor combination so |b| genuinely matters.
    z = kappa * score
    e = np.exp(z - z.max())
    return e / e.sum()


def neg_sharpe(b, months, kappa, l1, ppy=12):
    """Negative annualized training Sharpe plus a smooth L1 coefficient penalty."""
    rp = np.empty(len(months))
    for k, (Z, r) in enumerate(months):
        rp[k] = softmax_weights(Z @ b, kappa) @ r
    sd = rp.std(ddof=1)
    sharpe = rp.mean() / sd * np.sqrt(ppy) if sd > 0 else 0.0
    return -sharpe + l1 * float(np.sum(np.sqrt(b * b + 1e-8)))


def fit_df(train, factors_subset, kappa=1.0, l1=0.0):
    """Decision-focused softmax-Sharpe fit over the given factor subset on train -> {factor: b}."""
    if not factors_subset:
        return {}
    Xraw = train[factors_subset].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    mean, std = np.nanmean(Xraw, 0), np.nanstd(Xraw, 0); std = np.where(std > 0, std, 1.0)
    months = []
    for _, g in train.groupby("ym"):
        Z = np.nan_to_num((g[factors_subset].apply(pd.to_numeric, errors="coerce").to_numpy(float) - mean) / std)
        r = pd.to_numeric(g["forward_return"], errors="coerce").to_numpy(float)
        ok = np.isfinite(r)
        if ok.sum() >= 10:
            months.append((Z[ok], r[ok]))
    yv = np.nan_to_num(pd.to_numeric(train["forward_return"], errors="coerce").to_numpy(float))
    c = np.nan_to_num((Xraw - mean) / std).T @ yv
    b0 = c / (np.linalg.norm(c) + 1e-12) * 5.0
    b = minimize(neg_sharpe, b0, args=(months, kappa, l1), method="L-BFGS-B", options={"maxiter": 400}).x
    return {factors_subset[j]: float(b[j]) for j in range(len(factors_subset))}
