"""Variant A: long-only convex mean-variance portfolio from sparse factor weights.

Pipeline:
    sparse factor weights b (elastic-net)  ->  score_i = b' x_i  (predicted return)
    Ledoit-Wolf shrinkage covariance       ->  Sigma_t
    long-only convex QP (cvxpy)            ->  stock weights w_t
    realized forward returns               ->  cost-adjusted Sharpe

The score is the predicted cross-sectional expected return: factors are z-scored with
expanding point-in-time moments (only data at or before each date), then multiplied by b.
Because the budget constraint 1'w = 1 is fixed, an additive constant in the score does
not change the optimizer, so we use score = b' x_std directly as the expected-return mu.

References:
- Kozak, Nagel & Santosh (2020): b proportional to MVE weights; L1+L2 shrinkage.
- DeMiguel et al. (2009): norm-constrained MVE weights.
- Ledoit & Wolf (2004): shrink the sample covariance toward a scaled identity.
- Markowitz (1952): mean-variance objective.

NOT OOS-SAFE for the covariance window: `run_backtest` selects trailing months by
signal month, so a trailing return may still be unrealized at formation. The scores are
point-in-time and the cap logic is shared, but for a strict out-of-sample backtest use
`hybrid_walkforward.py`, which draws its covariance from settled history.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import cvxpy as cp
except ImportError:  # pragma: no cover - only the MVO solver needs cvxpy.
    cp = None


@dataclass
class PortfolioConfig:
    panel_csv: str = "data/a_share_factor_panel.csv"
    weights_csv: str = "outputs/a_share_baseline/sparse_weights.csv"
    output_dir: str = "outputs/portfolio_variant_a"
    date_col: str = "date"
    asset_col: str = "asset"
    target_col: str = "forward_return"
    mktcap_col: str = "float_mktcap"
    gamma: float = 10.0            # risk aversion in max mu'w - (gamma/2) w'Sigma w
    cov_window: int = 36           # trailing months for the covariance estimate
    min_history: int = 24          # minimum trailing months an asset needs to be eligible
    w_max: float = 0.05            # per-name weight cap (box constraint)
    cost_rate: float = 0.0015      # cost per unit of traded weight (one-way, ~15 bps blended)
    periods_per_year: int = 12
    factors: list[str] = field(default_factory=list)  # filled from weights_csv if empty


# --------------------------------------------------------------------------- #
# Covariance: Ledoit-Wolf linear shrinkage toward a scaled identity (LW 2004). #
# --------------------------------------------------------------------------- #
def ledoit_wolf_identity(returns: np.ndarray) -> tuple[np.ndarray, float]:
    """returns: (T, N) realized returns. Returns (Sigma, shrinkage_intensity)."""
    t_obs, n_assets = returns.shape
    if t_obs < 2 or n_assets == 0:
        return np.cov(returns, rowvar=False, ddof=0), 0.0
    x = returns - returns.mean(axis=0, keepdims=True)
    sample = (x.T @ x) / t_obs
    mu = np.trace(sample) / n_assets
    target = mu * np.eye(n_assets)
    d2 = float(np.sum((sample - target) ** 2)) / n_assets
    if d2 <= 0.0:
        return sample, 0.0
    # b2 = (1/N) * (1/T^2) * sum_t || x_t x_t' - S ||_F^2 , bounded by d2 (LW 2004).
    # Vectorized: sum_t ||x_t x_t' - S||^2 = sum_t (x_t'x_t)^2 - 2 sum_t x_t'Sx_t + T||S||^2.
    norms2 = np.einsum("ti,ti->t", x, x)
    b2_sum = float(np.sum(norms2 ** 2) - 2.0 * np.sum((x @ sample) * x) + t_obs * np.sum(sample ** 2))
    b2 = min(b2_sum / (n_assets * t_obs * t_obs), d2)
    shrink = float(np.clip(b2 / d2, 0.0, 1.0))
    sigma = shrink * target + (1.0 - shrink) * sample
    sigma = 0.5 * (sigma + sigma.T)
    return sigma, shrink


# --------------------------------------------------------------------------- #
# Signal: predicted cross-sectional expected return  score = b' x_std.         #
# --------------------------------------------------------------------------- #
def load_weights(weights_csv: Path) -> pd.Series:
    raw = pd.read_csv(weights_csv)
    raw = raw[raw["weight"].astype(float) != 0.0]
    return pd.Series(raw["weight"].to_numpy(dtype=float), index=raw["factor"].astype(str))


def compute_scores(panel: pd.DataFrame, weights: pd.Series, factors: list[str],
                   date_col: str = "date") -> pd.Series:
    """Point-in-time standardization: each date's z-scores use factor moments estimated from
    rows dated at or before it. Whole-panel moments would let the validation/test
    distribution set the scale of historical signals -- lookahead the backtest then reports
    as out-of-sample skill. Factor values at a date are observable at that date, so the
    expanding window ends at the scoring date inclusive."""
    dates = pd.to_datetime(panel[date_col])
    score = pd.Series(0.0, index=panel.index)
    for factor in factors:
        col = pd.to_numeric(panel[factor], errors="coerce")
        frame = pd.DataFrame({"date": dates.to_numpy(), "value": col.to_numpy(dtype=float)})
        grouped = frame.groupby("date")["value"]
        count = grouped.count().sort_index().cumsum()
        total = grouped.sum().sort_index().cumsum()
        total_sq = (frame.assign(sq=frame["value"] ** 2)
                    .groupby("date")["sq"].sum().sort_index().cumsum())
        mean = total / count
        std = np.sqrt((total_sq / count - mean ** 2).clip(lower=0.0))
        mean_at = dates.map(mean).to_numpy(dtype=float)
        std_at = dates.map(std).to_numpy(dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            z = (col.to_numpy(dtype=float) - mean_at) / np.where(std_at > 0, std_at, np.nan)
        score = score + float(weights[factor]) * pd.Series(z, index=panel.index).fillna(0.0)
    return score


# --------------------------------------------------------------------------- #
# Long-only convex mean-variance QP.                                           #
# --------------------------------------------------------------------------- #
def month_is_tradable(n_assets: int, w_max: float, min_names: int = 10) -> bool:
    """A month is skipped, not traded, when it is too thin to diversify (fewer than
    min_names eligible) or when the cap cannot even sum to 1 (n * w_max < 1) -- in the
    second case solve_long_only_mvo would rightly raise, so every caller must share this
    predicate rather than re-deriving half of it."""
    # np.isfinite mirrors _require_feasible_cap: an Inf cap passed this predicate and
    # then crashed in the solver layer, so guard and solver must share one semantics.
    return bool(np.isfinite(w_max)) and n_assets >= min_names and n_assets * w_max >= 1.0


def _require_feasible_cap(n: int, w_max: float) -> None:
    """Shared by every cap consumer. NaN comparisons are all False, so without the
    isfinite check a NaN w_max sailed past zero/negative/n*w_max tests alike and
    propagated NaN weights."""
    if not np.isfinite(w_max) or w_max <= 0.0:
        raise ValueError(f"w_max must be a finite positive number, got {w_max}")
    if n == 0 or n * w_max < 1.0 - 1e-9:
        raise ValueError(
            f"long-only cap infeasible: {n} assets with w_max={w_max} cannot sum to 1; "
            "raise w_max or widen the eligible universe")


def cap_long_only(weights: np.ndarray, w_max: float) -> np.ndarray:
    """Redistribute onto {sum w = 1, 0 <= w <= w_max}: normalize, truncate at the cap,
    then refill the truncated mass in one closed-form pass proportional to each name's
    remaining room. Validates its own inputs (finite positive w_max, n * w_max >= 1), so
    direct callers get the same guarantee as solve_long_only_mvo."""
    w = np.clip(np.asarray(weights, dtype=float), 0.0, None)
    _require_feasible_cap(int(w.size), w_max)
    if not np.isfinite(w).all() or w.sum() <= 0.0:
        w = np.ones_like(w)
    w = np.minimum(w / w.sum(), w_max)
    deficit = 1.0 - float(w.sum())
    if deficit > 1e-15:
        # Redistribute the truncated mass in proportion to each name's remaining room.
        # deficit <= room.sum() whenever n * w_max >= 1, so one pass lands exactly on the
        # simplex with every name still at or under the cap, and below-cap ordering is
        # preserved.
        room = w_max - w
        w = w + room * (deficit / float(room.sum()))
    return w


def solve_long_only_mvo(mu: np.ndarray, sigma: np.ndarray, gamma: float, w_max: float) -> np.ndarray:
    """Solve capped long-only mean-variance weights, checking cap feasibility first.
    A solver failure uses the constrained fallback; an infeasible cap is an error."""
    if cp is None:  # pragma: no cover - environment guard.
        raise SystemExit("cvxpy is required for solve_long_only_mvo: pip install '.[opt]'")
    n = len(mu)
    # sum(w)=1 with 0<=w<=w_max is infeasible outright when n*w_max < 1 (or w_max is not a
    # finite positive number). Returning a "closest" vector would dress a configuration
    # error up as a portfolio result.
    _require_feasible_cap(n, w_max)
    w = cp.Variable(n)
    objective = cp.Maximize(mu @ w - 0.5 * gamma * cp.quad_form(w, cp.psd_wrap(sigma)))
    constraints = [cp.sum(w) == 1, w >= 0, w <= w_max]
    problem = cp.Problem(objective, constraints)
    status = "failed"   # Problem.status is a read-only property; track the outcome locally
    try:
        problem.solve(solver=cp.OSQP, verbose=False)
        status = problem.status
    except Exception:  # noqa: BLE001 - fall through to the analytic fallback.
        pass
    if w.value is None or status not in ("optimal", "optimal_inaccurate"):
        # Fallback: long-only score tilt, redistributed under the cap (renormalizing after
        # a plain clip pushed capped names straight back above w_max).
        pos = np.clip(mu, 0.0, None)
        return cap_long_only(pos if pos.sum() > 0.0 else np.ones(n), w_max)
    return cap_long_only(np.asarray(w.value).flatten(), w_max)


# --------------------------------------------------------------------------- #
# Backtest driver.                                                             #
# --------------------------------------------------------------------------- #
def run_backtest(panel: pd.DataFrame, config: PortfolioConfig) -> dict:
    """Run the fixed-factor monthly portfolio diagnostic with turnover costs.

    Its covariance window follows signal months, so unsettled labels may enter;
    use hybrid_walkforward for settlement-aware research.
    """
    panel = panel.copy()
    panel["ym"] = pd.to_datetime(panel[config.date_col]).dt.to_period("M")
    weights = load_weights(Path(config.weights_csv))
    factors = config.factors or [f for f in weights.index if f in panel.columns]
    panel["score"] = compute_scores(panel, weights, factors, date_col=config.date_col)

    returns_wide = panel.pivot_table(index="ym", columns=config.asset_col, values=config.target_col)
    months = list(returns_wide.index)

    weight_records: list[dict] = []
    rows: list[dict] = []
    prev_w: dict[str, float] = {}

    for idx in range(config.cov_window, len(months)):
        month = months[idx]
        # "Strictly past" by SIGNAL month only. These returns may not have
        # settled by the formation date -- hybrid_walkforward draws its covariance from
        # settled history instead. Do not use this path for a strict OOS claim.
        window_months = months[idx - config.cov_window: idx]
        cross = panel[panel["ym"] == month]
        cross = cross.dropna(subset=["score", config.target_col])
        if cross.empty:
            continue

        hist = returns_wide.loc[window_months]
        eligible = [a for a in cross[config.asset_col] if a in hist.columns
                    and hist[a].notna().sum() >= config.min_history]
        cross = cross[cross[config.asset_col].isin(eligible)]
        if not month_is_tradable(len(cross), config.w_max):
            continue

        assets = cross[config.asset_col].tolist()
        hist_mat = hist[assets].fillna(0.0).to_numpy()
        sigma, shrink = ledoit_wolf_identity(hist_mat)
        mu = cross["score"].to_numpy(dtype=float)

        w = solve_long_only_mvo(mu, sigma, config.gamma, config.w_max)
        w_map = dict(zip(assets, w))

        realized = cross[config.target_col].to_numpy(dtype=float)
        gross = float(np.dot(w, realized))

        traded = sum(abs(w_map.get(a, 0.0) - prev_w.get(a, 0.0))
                     for a in set(w_map) | set(prev_w))
        cost = config.cost_rate * traded
        net = gross - cost

        # Benchmarks over the same eligible set.
        ew = float(np.mean(realized))
        caps = pd.to_numeric(cross[config.mktcap_col], errors="coerce").to_numpy()
        if np.isfinite(caps).any() and np.nansum(caps) > 0:
            cap_w = np.where(np.isfinite(caps), caps, 0.0)
            cap_w = cap_w / cap_w.sum()
            vw = float(np.dot(cap_w, realized))
        else:
            vw = np.nan

        rows.append({"ym": str(month), "n_assets": len(assets), "gross": gross,
                     "net": net, "turnover": traded, "cost": cost,
                     "ew": ew, "vw": vw, "lw_shrink": shrink})
        for a, wa in w_map.items():
            if wa > 1e-6:
                weight_records.append({"ym": str(month), "asset": a, "weight": wa})
        prev_w = w_map

    returns_df = pd.DataFrame(rows)
    weights_df = pd.DataFrame(weight_records)
    summary = summarize(returns_df, config)
    return {"returns": returns_df, "weights": weights_df, "summary": summary, "factors": factors}


def _sharpe(series: pd.Series, ppy: int) -> float:
    s = series.dropna()
    if len(s) < 2 or s.std(ddof=1) == 0:
        return float("nan")
    return float(s.mean() / s.std(ddof=1) * np.sqrt(ppy))


def _ann_return(series: pd.Series, ppy: int) -> float:
    s = series.dropna()
    return float(s.mean() * ppy) if len(s) else float("nan")


def summarize(returns_df: pd.DataFrame, config: PortfolioConfig) -> dict:
    """Aggregate return, risk and turnover diagnostics; undefined Sharpe values remain NaN."""
    ppy = config.periods_per_year
    out = {
        "n_months": int(len(returns_df)),
        "avg_n_assets": float(returns_df["n_assets"].mean()) if len(returns_df) else float("nan"),
        "avg_turnover": float(returns_df["turnover"].mean()) if len(returns_df) else float("nan"),
        "sharpe_gross": _sharpe(returns_df.get("gross", pd.Series(dtype=float)), ppy),
        "sharpe_net": _sharpe(returns_df.get("net", pd.Series(dtype=float)), ppy),
        "sharpe_equal_weight": _sharpe(returns_df.get("ew", pd.Series(dtype=float)), ppy),
        "sharpe_value_weight": _sharpe(returns_df.get("vw", pd.Series(dtype=float)), ppy),
        "ann_return_net": _ann_return(returns_df.get("net", pd.Series(dtype=float)), ppy),
        "ann_return_equal_weight": _ann_return(returns_df.get("ew", pd.Series(dtype=float)), ppy),
        "ann_return_value_weight": _ann_return(returns_df.get("vw", pd.Series(dtype=float)), ppy),
        "ann_vol_net": (float(returns_df["net"].std(ddof=1) * np.sqrt(ppy))
                        if len(returns_df) > 1 else float("nan")),
    }
    return out


def main() -> None:
    """Load fixed factor weights and save the portfolio diagnostic and provenance.

    This command uses signal-month covariance timing, not settlement-aware OOS
    evaluation. An empty tradable sample writes INSUFFICIENT_DATA and exits nonzero.
    """
    parser = argparse.ArgumentParser(description="Variant A long-only convex MVO from sparse factor weights.")
    defaults = PortfolioConfig()
    parser.add_argument("--panel-csv", default=defaults.panel_csv)
    parser.add_argument("--weights-csv", default=defaults.weights_csv)
    parser.add_argument("--output-dir", default=defaults.output_dir)
    parser.add_argument("--gamma", type=float, default=defaults.gamma)
    parser.add_argument("--cov-window", type=int, default=defaults.cov_window)
    parser.add_argument("--min-history", type=int, default=defaults.min_history)
    parser.add_argument("--w-max", type=float, default=defaults.w_max)
    parser.add_argument("--cost-rate", type=float, default=defaults.cost_rate)
    args = parser.parse_args()

    config = PortfolioConfig(
        panel_csv=args.panel_csv, weights_csv=args.weights_csv, output_dir=args.output_dir,
        gamma=args.gamma, cov_window=args.cov_window, min_history=args.min_history,
        w_max=args.w_max, cost_rate=args.cost_rate,
    )
    panel = pd.read_csv(config.panel_csv)
    result = run_backtest(panel, config)

    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result["returns"].to_csv(out / "portfolio_returns.csv", index=False, encoding="utf-8-sig")
    result["weights"].to_csv(out / "portfolio_weights.csv", index=False, encoding="utf-8-sig")
    from factor_research.experiment_manifest import build_manifest, write_strict_json
    status = "OK" if len(result["returns"]) else "INSUFFICIENT_DATA"
    write_strict_json(out / "portfolio_summary.json", {
        "status": status,
        "manifest": build_manifest(panel_csv=config.panel_csv,
                                   extra={"weights_csv": str(config.weights_csv),
                                          "weights_sha256": __import__(
                                              "factor_research.experiment_manifest", fromlist=["file_sha256"]).file_sha256(
                                                  config.weights_csv),
                                          "gamma": config.gamma, "w_max": config.w_max,
                                          "cost_rate": config.cost_rate,
                                          "cov_window": config.cov_window,
                                          "min_history": config.min_history}),
        "config": vars(config), "factors": result["factors"], "summary": result["summary"]})
    if status != "OK":
        print("no tradable months under the current filters -- wrote INSUFFICIENT_DATA "
              "summary and exiting non-zero (a zero-month backtest is not a success)")
        raise SystemExit(1)

    s = result["summary"]
    print("Variant A long-only convex MVO")
    print(f"  factors: {result['factors']}")
    print(f"  months: {s['n_months']}, avg assets: {s['avg_n_assets']:.0f}, avg turnover: {s['avg_turnover']:.3f}")
    print(f"  Sharpe  gross={s['sharpe_gross']:.3f}  net={s['sharpe_net']:.3f}")
    print(f"  Sharpe  equal-weight={s['sharpe_equal_weight']:.3f}  value-weight={s['sharpe_value_weight']:.3f}")
    print(f"  Ann return net={s['ann_return_net']:.3%}  vol={s['ann_vol_net']:.3%}")
    print(f"  Wrote outputs to: {out}")


if __name__ == "__main__":
    main()
