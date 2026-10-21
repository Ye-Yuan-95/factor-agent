"""Synthetic panels with an injected cross-family mechanism.

The premium uses a normalized signal scaled by the requested IC and calibrated
return scale: conditional (F_x * sign(F_y)), regime (F_x * 1{F_y > 0}) or nonlinear
(F_x * F_y). The IC argument controls injection scale; it does not enforce the
realized rank IC. Panels share the calibration contract of the linear generator."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage_a_fpr_power import calibrate  # noqa: E402,F401  (re-exported for callers)

MECHANISM_KINDS = ("conditional", "regime", "nonlinear")


def _mech_signal(kind, fx, fy):
    if kind == "conditional":
        return fx * np.sign(fy)
    if kind == "regime":
        return fx * (fy > 0)
    if kind == "nonlinear":
        return fx * fy
    raise ValueError(f"unknown mechanism kind {kind!r}")


def generate_mechanism_panel(rng, calib, n_assets, n_months, *, kind, x_col, y_col,
                             ic, market_frac=0.3):
    """stage_a-style Student-t panel whose ONLY premium is the chosen mechanism term."""
    factors = calib["factors"]
    p = len(factors)
    s_r = calib["s_r"]
    nu = calib["t_dof"]
    chol = np.linalg.cholesky(calib["sigma_f"])
    ix, iy = factors.index(x_col), factors.index(y_col)
    s_market = np.sqrt(market_frac) * s_r
    s_eps = np.sqrt(max(1.0 - market_frac, 0.05)) * s_r
    beta = rng.normal(1.0, 0.3, size=n_assets)
    rows = []
    for t in range(n_months):
        F = rng.standard_normal((n_assets, p)) @ chol.T
        F = (F - F.mean(0)) / (F.std(0) + 1e-12)
        z = _mech_signal(kind, F[:, ix], F[:, iy])
        z = (z - z.mean()) / (z.std() + 1e-12)
        m_t = rng.normal(0.0, s_market)
        raw = rng.standard_t(nu, size=n_assets)
        eps = raw / np.sqrt(nu / (nu - 2.0)) * s_eps
        eps = eps - eps.mean()
        r_next = beta * m_t + ic * s_r * z + eps
        block = pd.DataFrame(F, columns=factors)
        block.insert(0, "forward_return", r_next)
        block.insert(0, "asset", [f"S{i:04d}" for i in range(n_assets)])
        block.insert(0, "date", pd.Timestamp("2000-01-31") + pd.offsets.MonthEnd(t))
        rows.append(block)
    return pd.concat(rows, ignore_index=True)
