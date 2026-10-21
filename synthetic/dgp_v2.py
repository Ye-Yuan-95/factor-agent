"""Gen-2 known-truth generator with heterogeneous and time-varying volatility.

The panel supplies correlation/return calibration, industry weights (when present)
and estimated exposure persistence. Fixed configuration defaults separately specify tail
heterogeneity, volatility dynamics and fallback coverage/sector structure. These
settings are not all estimated from every supplied panel, and matching a few
moments does not establish a realistic distribution in all respects.

Outputs have the same panel/truth schema as Gen-1. Injection strength scales a
premium coefficient; it does not fix the realized rank IC exactly.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage_a_fpr_power import calibrate


# Fallback settings for panel properties that cannot be estimated from the input.
DEFAULT_REALISM_SETTINGS = {
    "exposure_phi": 0.77,         # factor-exposure lag-1 autocorrelation (persistence)
    "coverage": 0.973,
}


def calibrate_v2(panel_csv, factors, fam_map=None, industry_col="industry"):
    """Estimate correlation, return scale, industry weights and exposure persistence.

    Persistence averages lag-one estimates from the first eight factors and up to
    150 assets with sufficient history. Missing estimators use the explicit
    fallback settings. fam_map is accepted by the shared calibrator interface;
    Gen-2 sector weights come from industry_col rather than factor families.
    """
    calib = calibrate(panel_csv, factors)
    df = pd.read_csv(panel_csv)
    # sector structure from the real industry column
    if industry_col in df.columns:
        vc = df[industry_col].astype(str).value_counts(normalize=True)
        calib["sector_weights"] = vc.values.tolist()
        calib["n_sectors"] = int(len(vc))
    else:
        calib["sector_weights"] = None
        calib["n_sectors"] = 10
    # exposure persistence: mean lag-1 autocorr over a few factors
    phis = []
    for f in factors[:8]:
        if f in df.columns and "asset" in df.columns and "date" in df.columns:
            try:
                piv = df.pivot_table(index="date", columns="asset", values=f)
                acs = [piv[c].astype(float).autocorr(1) for c in piv.columns[:150] if piv[c].notna().sum() > 10]
                if acs:
                    phis.append(np.nanmean(acs))
            except Exception:
                # An unusable factor pivot contributes no persistence estimate;
                # the remaining factors can still supply calibration information.
                pass
    calib["exposure_phi"] = float(np.clip(np.nanmean(phis), 0.0, 0.95)) if phis else DEFAULT_REALISM_SETTINGS["exposure_phi"]
    if "tradable" in df.columns:
        calib["coverage"] = float(pd.to_numeric(df["tradable"], errors="coerce").mean())
    else:
        calib["coverage"] = DEFAULT_REALISM_SETTINGS["coverage"]
    return calib


def _logar1_vol_path(rng, n_months, rho_h, sigma_h):
    """Generate a log-AR(1) volatility multiplier normalized to unit second moment.

    rho_h controls persistence and sigma_h controls volatility dispersion. The
    normalization preserves return variance while allowing time-varying tail scale."""
    h = np.empty(n_months)
    stat_sd = sigma_h / np.sqrt(max(1.0 - rho_h ** 2, 1e-6))
    h[0] = rng.normal(0.0, stat_sd)
    for t in range(1, n_months):
        h[t] = rho_h * h[t - 1] + sigma_h * rng.standard_normal()
    var_h = stat_sd ** 2
    return np.exp(h - var_h)                              # E[exp(2(h-var_h))] = 1


def generate_panel_v2(rng, calib, n_assets, n_months, true_idx, strengths_ic,
                      market_frac=0.3, hetero_delta=0.85, rho_h=0.35, sigma_h=0.55,
                      sector_frac=0.15, exposure_phi=None, coverage=1.0, tails="gaussian"):
    """Realistic mirror of stage_a.generate_panel. Returns (panel DataFrame, truth bool array).

    hetero_delta : per-stock log-vol dispersion (fat CROSS-SECTIONAL tails; E[lam^2]=1 preserves var).
    rho_h/sigma_h : log-AR(1) stochastic vol -- persistence (vol ACF) and vol-of-vol (pooled tails).
    sector_frac : share of return variance from the shared per-sector shock (block co-movement).
    exposure_phi : AR(1) coefficient for exposure persistence (defaults to calib['exposure_phi']).
    coverage : fraction of (asset,month) cells kept observed (rest -> NaN)."""
    factors = calib["factors"]; p = len(factors); s_r = calib["s_r"]; nu = calib.get("t_dof", 6.0)
    chol = np.linalg.cholesky(calib["sigma_f"])
    if exposure_phi is None:
        exposure_phi = calib.get("exposure_phi", 0.77)

    # split variance: market + sector + idiosyncratic (premium adds on top)
    s_market = np.sqrt(market_frac) * s_r
    s_sector = np.sqrt(sector_frac) * s_r
    s_idio = np.sqrt(max(1.0 - market_frac - sector_frac, 0.05)) * s_r

    beta = rng.normal(1.0, 0.3, size=n_assets)
    # persistent per-stock vol scale (fat cross-sectional tails); E[lam^2]=1 preserves total variance
    lam = np.exp(hetero_delta * rng.standard_normal(n_assets) - hetero_delta ** 2)
    # sector assignment
    n_sec = max(int(calib.get("n_sectors", 10)), 1)
    sw = calib.get("sector_weights")
    if sw is not None and len(sw) == n_sec:
        sec_of = rng.choice(n_sec, size=n_assets, p=np.array(sw) / np.sum(sw))
    else:
        sec_of = rng.integers(0, n_sec, size=n_assets)
    vol_path = _logar1_vol_path(rng, n_months, rho_h, sigma_h)

    a = np.zeros(p)
    for j, ic in zip(true_idx, strengths_ic):
        a[j] = ic * s_r

    F_prev = None
    rows = []
    for t in range(n_months):
        znew = rng.standard_normal((n_assets, p)) @ chol.T
        znew = (znew - znew.mean(0)) / (znew.std(0) + 1e-12)
        if F_prev is None or exposure_phi <= 0:
            F = znew
        else:                                            # AR(1) exposure persistence
            F = exposure_phi * F_prev + np.sqrt(1.0 - exposure_phi ** 2) * znew
            F = (F - F.mean(0)) / (F.std(0) + 1e-12)
        F_prev = F
        vt = vol_path[t]                                 # regime vol multiplier this month
        m_t = rng.normal(0.0, s_market * vt)
        sec_shock = rng.normal(0.0, s_sector * vt, size=n_sec)
        if tails == "t":
            innov = rng.standard_t(nu, size=n_assets) / np.sqrt(nu / (nu - 2.0))
        else:
            innov = rng.standard_normal(n_assets)
        eps = s_idio * vt * lam * innov                  # hetero-scale idiosyncratic (fat XS tails)
        eps = eps - eps.mean()
        premium = F @ a
        r_next = beta * m_t + sec_shock[sec_of] + premium + eps
        block = pd.DataFrame(F, columns=factors)
        block.insert(0, "forward_return", r_next)
        block.insert(0, "industry", [f"sec{sec_of[i]:02d}" for i in range(n_assets)])
        block.insert(0, "asset", [f"S{i:04d}" for i in range(n_assets)])
        block.insert(0, "date", pd.Timestamp("2000-01-31") + pd.offsets.MonthEnd(t))
        rows.append(block)
    panel = pd.concat(rows, ignore_index=True)

    if coverage < 1.0:                                   # missing / coverage mask
        keep = rng.random(len(panel)) < coverage
        panel.loc[~keep, ["forward_return"] + factors] = np.nan

    truth = np.zeros(p, dtype=bool); truth[list(true_idx)] = True
    return panel, truth


def report_moments(panel, factors):
    """Compute descriptive tail, volatility and scale statistics from the generated panel."""
    r = pd.to_numeric(panel["forward_return"], errors="coerce").dropna()
    xs_kurt = panel.groupby("date")["forward_return"].apply(
        lambda x: pd.to_numeric(x, errors="coerce").kurt()).mean()
    mv = panel.groupby("date")["forward_return"].apply(
        lambda x: pd.to_numeric(x, errors="coerce").std())
    return {"pooled_excess_kurt": float(r.kurt()), "xs_excess_kurt": float(xs_kurt),
            "vol_acf1": float(mv.autocorr(1)) if len(mv) > 2 else float("nan"),
            "ret_std": float(r.std()), "coverage": float(panel["forward_return"].notna().mean())}
