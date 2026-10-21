"""Training-only factor screening, redundancy, incremental testing and stability diagnostics.

The pipeline purges labels unavailable by train_end before any fit. A passing
factor is a candidate for further research, not a guarantee of trading returns.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PipelineConfig:
    input_csv: Path
    output_dir: Path
    date_col: str
    asset_col: str
    target_col: str
    factor_cols: tuple[str, ...]
    factor_families: dict[str, str]
    max_factors_per_family: dict[str, int]
    min_coverage: float
    q_value_max: float
    min_abs_ic: float
    min_long_short: float
    long_short_quantile: float
    hac_lags: int
    max_abs_corr: float
    l1_alpha: float
    l2_alpha: float
    max_iter: int
    tol: float
    min_abs_weight: float
    n_scenarios: int
    mean_block_length: float
    noise_std_fraction: float
    random_seed: int
    survival_metric: str
    survival_threshold: float
    directional_survival: bool
    min_degradation_ratio: float
    min_survival_rate: float
    min_incremental_tstat: float
    # train_end is REQUIRED for a run (enforced in run_pipeline): computing the screen,
    # survival and sparse weights on the full sample lets the evaluation window set the
    # selection. Optional here only so load_config can build the object before the CLI
    # attaches the date.
    train_end: str | None = None
    config_path: str | None = None
    label_embargo_days: int | None = None


def load_config(path: str | Path) -> PipelineConfig:
    """Parse the JSON configuration into typed paths, gate thresholds and family limits.
    Paths retain their supplied meaning; train_end is enforced by run_pipeline."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    family_constraints = raw.get("family_constraints", {})
    screening = raw.get("screening", {})
    redundancy = raw.get("redundancy", {})
    optimization = raw.get("optimization", {})
    synthetic = raw.get("synthetic", {})
    incremental = raw.get("incremental", {})
    return PipelineConfig(
        input_csv=Path(raw["input_csv"]),
        output_dir=Path(raw["output_dir"]),
        date_col=str(raw.get("date_col", "date")),
        asset_col=str(raw.get("asset_col", "asset")),
        target_col=str(raw.get("target_col", "forward_return")),
        factor_cols=tuple(str(col) for col in raw["factor_cols"]),
        factor_families={str(key): str(value) for key, value in family_constraints.get("factor_families", {}).items()},
        max_factors_per_family={
            str(key): int(value) for key, value in family_constraints.get("max_per_family", {}).items()
        },
        min_coverage=float(screening.get("min_coverage", 0.5)),
        q_value_max=float(screening.get("q_value_max", 0.2)),
        min_abs_ic=float(screening.get("min_abs_ic", 0.0)),
        min_long_short=float(screening.get("min_long_short", 0.0)),
        long_short_quantile=float(screening.get("long_short_quantile", 0.3)),
        hac_lags=int(screening.get("hac_lags", 0)),
        max_abs_corr=float(redundancy.get("max_abs_corr", 0.85)),
        l1_alpha=float(optimization.get("l1_alpha", 1e-4)),
        l2_alpha=float(optimization.get("l2_alpha", 1e-3)),
        max_iter=int(optimization.get("max_iter", 2000)),
        tol=float(optimization.get("tol", 1e-8)),
        min_abs_weight=float(optimization.get("min_abs_weight", 1e-8)),
        n_scenarios=int(synthetic.get("n_scenarios", 100)),
        mean_block_length=float(synthetic.get("mean_block_length", 20.0)),
        noise_std_fraction=float(synthetic.get("noise_std_fraction", 0.05)),
        random_seed=int(synthetic.get("random_seed", 42)),
        survival_metric=str(synthetic.get("survival_metric", "long_short")),
        survival_threshold=float(synthetic.get("survival_threshold", 0.0)),
        directional_survival=bool(synthetic.get("directional_survival", True)),
        min_degradation_ratio=float(synthetic.get("min_degradation_ratio", 0.5)),
        min_survival_rate=float(synthetic.get("min_survival_rate", 0.6)),
        min_incremental_tstat=float(incremental.get("min_abs_tstat", 2.0)),
        train_end=raw.get("train_end"),
        config_path=str(path),
        label_embargo_days=raw.get("label_embargo_days"),
    )


def load_panel(config: PipelineConfig) -> pd.DataFrame:
    """Load required columns, coerce numeric observations and sort date/asset rows.
    This does not enforce training cutoffs; callers must purge unsettled labels before fitting."""
    frame = pd.read_csv(config.input_csv)
    required = [config.date_col, config.asset_col, config.target_col, *config.factor_cols]
    if LABEL_END_COL in frame.columns:
        required = required + [LABEL_END_COL]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")
    frame = frame[required].copy()
    frame[config.date_col] = pd.to_datetime(frame[config.date_col])
    for col in [config.target_col, *config.factor_cols]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=[config.date_col, config.asset_col, config.target_col])
    return frame.sort_values([config.date_col, config.asset_col]).reset_index(drop=True)


def _dated_series(values: list[tuple[pd.Timestamp, float]], name: str) -> pd.Series:
    if not values:
        return pd.Series(dtype=float, name=name)
    return pd.Series([value for _, value in values], index=[date for date, _ in values], name=name)


def rank_ic_by_date(frame: pd.DataFrame, factor: str, config: PipelineConfig) -> pd.Series:
    """Compute cross-sectional rank correlation on observed pairs for each date."""
    values: list[tuple[pd.Timestamp, float]] = []
    for date, group in frame.groupby(config.date_col, sort=True):
        clean = group[[factor, config.target_col]].dropna()
        if len(clean) < 5 or clean[factor].nunique() < 2 or clean[config.target_col].nunique() < 2:
            continue
        corr = clean[factor].rank().corr(clean[config.target_col].rank())
        if pd.notna(corr):
            values.append((pd.Timestamp(date), float(corr)))
    return _dated_series(values, factor)


def long_short_by_date(frame: pd.DataFrame, factor: str, config: PipelineConfig) -> pd.Series:
    """Return equal-weight high-minus-low forward returns by signal date.
    Quantile ties may enlarge a leg; this diagnostic excludes transaction costs."""
    q = config.long_short_quantile
    if not 0.0 < q < 0.5:
        raise ValueError("long_short_quantile must be between 0 and 0.5.")
    values: list[tuple[pd.Timestamp, float]] = []
    for date, group in frame.groupby(config.date_col, sort=True):
        clean = group[[factor, config.target_col]].dropna()
        if len(clean) < 10 or clean[factor].nunique() < 3:
            continue
        low_cut = clean[factor].quantile(q)
        high_cut = clean[factor].quantile(1.0 - q)
        low = clean.loc[clean[factor] <= low_cut, config.target_col]
        high = clean.loc[clean[factor] >= high_cut, config.target_col]
        if len(low) == 0 or len(high) == 0:
            continue
        spread = float(high.mean() - low.mean())
        if np.isfinite(spread):
            values.append((pd.Timestamp(date), spread))
    return _dated_series(values, factor)


def t_stat(values: pd.Series, hac_lags: int = 0) -> float:
    """Estimate the mean t statistic using IID or Bartlett-weighted HAC variance.
    Return NaN for insufficient or degenerate observations; hac_lags controls serial dependence."""
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) < 3:
        return float("nan")
    mean = float(clean.mean())
    if hac_lags <= 0:
        std = float(clean.std(ddof=1))
        if std <= 0.0 or not np.isfinite(std):
            return float("nan")
        return float(mean / (std / np.sqrt(len(clean))))

    x = clean.to_numpy(dtype=float) - mean
    n = len(x)
    max_lag = min(int(hac_lags), n - 1)
    long_run_var = float(np.dot(x, x) / n)
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        gamma = float(np.dot(x[lag:], x[:-lag]) / n)
        long_run_var += 2.0 * weight * gamma
    var_mean = long_run_var / n
    if var_mean <= 0.0 or not np.isfinite(var_mean):
        return float("nan")
    return float(mean / np.sqrt(var_mean))


def normal_two_sided_pvalue(t_value: float) -> float:
    """Map a finite t statistic to a normal-approximation p-value; invalid values fail closed at 1."""
    if not np.isfinite(t_value):
        return 1.0
    cdf = NormalDist().cdf(abs(float(t_value)))
    return float(2.0 * (1.0 - cdf))


def benjamini_hochberg_qvalues(p_values: np.ndarray) -> np.ndarray:
    """Return monotone BH-adjusted p-values in original order, treating missing tests as p=1."""
    p = np.asarray(p_values, dtype=float)
    p = np.where(np.isfinite(p), p, 1.0)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.empty(n, dtype=float)
    running = 1.0
    for idx in range(n - 1, -1, -1):
        rank = idx + 1
        running = min(running, ranked[idx] * n / rank)
        adjusted[order[idx]] = running
    return np.clip(adjusted, 0.0, 1.0)


def single_factor_screen(frame: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Combine IC and long-short evidence using a two-test Bonferroni minimum, then BH.
    These are marginal tests; passing does not establish independent incremental contribution."""
    rows: list[dict[str, Any]] = []
    total_rows = len(frame)
    for factor in config.factor_cols:
        valid_factor_rows = frame[factor].notna() & frame[config.target_col].notna()
        coverage = float(valid_factor_rows.mean()) if total_rows else 0.0
        ic_series = rank_ic_by_date(frame, factor, config)
        ls_series = long_short_by_date(frame, factor, config)
        ic_t = t_stat(ic_series, config.hac_lags)
        ls_t = t_stat(ls_series, config.hac_lags)
        rows.append({
            "factor": factor,
            "coverage": coverage,
            "n_ic_dates": int(ic_series.count()),
            "mean_ic": float(ic_series.mean()) if len(ic_series) else np.nan,
            "ic_std": float(ic_series.std(ddof=1)) if len(ic_series) > 1 else np.nan,
            "ic_t_stat": ic_t,
            "ic_p_value": normal_two_sided_pvalue(ic_t),
            "n_long_short_dates": int(ls_series.count()),
            "mean_long_short": float(ls_series.mean()) if len(ls_series) else np.nan,
            "long_short_t_stat": ls_t,
            "long_short_p_value": normal_two_sided_pvalue(ls_t),
        })
    result = pd.DataFrame(rows)
    p_for_q = 2.0 * np.minimum(
        result["ic_p_value"].to_numpy(dtype=float),
        result["long_short_p_value"].to_numpy(dtype=float),
    )
    result["combined_p_value"] = np.clip(p_for_q, 0.0, 1.0)
    result["q_value"] = benjamini_hochberg_qvalues(result["combined_p_value"].to_numpy(dtype=float))
    result["passes_stat_screen"] = (
        (result["coverage"] >= config.min_coverage)
        & (result["q_value"] <= config.q_value_max)
        & (
            (result["mean_ic"].abs() > config.min_abs_ic)
            | (result["mean_long_short"] > config.min_long_short)
        )
    )
    result["screen_score"] = (
        result["ic_t_stat"].abs().fillna(0.0)
        + result["long_short_t_stat"].abs().fillna(0.0)
        + result["mean_long_short"].fillna(0.0).clip(lower=0.0)
    )
    return result.sort_values("screen_score", ascending=False).reset_index(drop=True)


def factor_correlation(frame: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Compute the candidate exposure Spearman correlation matrix on the supplied fitting rows."""
    return frame[list(config.factor_cols)].corr(method="spearman", min_periods=20)


def infer_factor_family(factor: str, config: PipelineConfig) -> str:
    if factor in config.factor_families:
        return config.factor_families[factor]
    if factor.startswith("mom"):
        return "momentum"
    if factor in {"volatility", "retvol", "idiovol", "beta"}:
        return "risk"
    if factor in {"dolvol", "volume_proxy", "turnover", "std_turn"}:
        return "liquidity"
    if factor in {"currat", "quick", "roeq", "agr"}:
        return "fundamental"
    return "unclassified"


def redundancy_filter(screen: pd.DataFrame, corr: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Greedily retain eligible factors subject to correlation and family limits.
    Ordering affects the result; this is not a global subset optimum."""
    selected: list[str] = []
    selected_by_family: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    ordered = screen.sort_values(["passes_stat_screen", "screen_score"], ascending=[False, False])
    for row in ordered.itertuples(index=False):
        factor = str(row.factor)
        family = infer_factor_family(factor, config)
        family_max = config.max_factors_per_family.get(family)
        family_selected_count = selected_by_family.get(family, 0)
        family_selected_rank = np.nan
        redundant_with = ""
        max_corr = 0.0
        is_redundant = False
        is_family_limited = False
        if bool(row.passes_stat_screen):
            for chosen in selected:
                value = corr.loc[factor, chosen] if factor in corr.index and chosen in corr.columns else np.nan
                if pd.notna(value) and abs(float(value)) > abs(max_corr):
                    max_corr = float(value)
                    redundant_with = chosen
            if redundant_with and abs(max_corr) >= config.max_abs_corr:
                is_redundant = True
            elif family_max is not None and family_selected_count >= family_max:
                is_family_limited = True
            else:
                selected.append(factor)
                selected_by_family[family] = family_selected_count + 1
                family_selected_rank = selected_by_family[family]
        rows.append({
            "factor": factor,
            "factor_family": family,
            "family_selected_rank": family_selected_rank,
            "max_factors_per_family": family_max if family_max is not None else np.nan,
            "passes_stat_screen": bool(row.passes_stat_screen),
            "is_redundant": bool(is_redundant),
            "redundant_with": redundant_with,
            "max_abs_corr_with_selected": abs(float(max_corr)) if redundant_with else np.nan,
            "is_family_limited": bool(is_family_limited),
            "passes_family_screen": bool(row.passes_stat_screen) and not bool(is_family_limited),
            "passes_redundancy_screen": bool(row.passes_stat_screen) and not bool(is_redundant) and not bool(is_family_limited),
        })
    return pd.DataFrame(rows)


def factor_family_audit(final: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Count selected and available factors by family and report configured caps."""
    selected = final.loc[final["selected_final"]].copy()
    rows: list[dict[str, Any]] = []
    families = sorted(set(final.get("factor_family", pd.Series(dtype=str)).dropna().astype(str)))
    for family in families:
        all_family = final.loc[final["factor_family"] == family, "factor"].astype(str).tolist()
        selected_family = selected.loc[selected["factor_family"] == family, "factor"].astype(str).tolist()
        family_max = config.max_factors_per_family.get(family)
        rows.append({
            "factor_family": family,
            "n_candidate_factors": len(all_family),
            "n_selected_factors": len(selected_family),
            "max_factors_per_family": family_max if family_max is not None else np.nan,
            "family_constraint_active": family_max is not None,
            "passes_family_limit": len(selected_family) <= family_max if family_max is not None else True,
            "candidate_factors": "|".join(all_family),
            "selected_factors": "|".join(selected_family),
        })
    return pd.DataFrame(rows)


def factor_correlation_pairs(corr: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Expand distinct factor pairs into signed and absolute Spearman diagnostics."""
    rows: list[dict[str, Any]] = []
    factors = [str(factor) for factor in corr.index]
    for left_idx, left in enumerate(factors):
        for right in factors[left_idx + 1 :]:
            value = corr.loc[left, right] if left in corr.index and right in corr.columns else np.nan
            if pd.isna(value):
                continue
            rows.append({
                "factor_left": left,
                "factor_right": right,
                "spearman_corr": float(value),
                "abs_corr": abs(float(value)),
                "exceeds_max_abs_corr": abs(float(value)) >= config.max_abs_corr,
            })
    return pd.DataFrame(rows).sort_values("abs_corr", ascending=False).reset_index(drop=True)


def selected_factor_correlation_audit(
    corr: pd.DataFrame,
    final: pd.DataFrame,
    config: PipelineConfig,
) -> pd.DataFrame:
    """Report each selected factor's largest exposure correlation with another selected factor."""
    selected = final.loc[final["selected_final"], "factor"].astype(str).tolist()
    rows: list[dict[str, Any]] = []
    for factor in selected:
        max_abs_corr = 0.0
        most_correlated = ""
        signed_corr = np.nan
        for other in selected:
            if other == factor:
                continue
            value = corr.loc[factor, other] if factor in corr.index and other in corr.columns else np.nan
            if pd.notna(value) and abs(float(value)) > max_abs_corr:
                max_abs_corr = abs(float(value))
                signed_corr = float(value)
                most_correlated = other
        rows.append({
            "factor": factor,
            "n_selected_factors": len(selected),
            "most_correlated_selected_factor": most_correlated,
            "corr_with_most_correlated": signed_corr,
            "max_abs_corr_with_selected_final": max_abs_corr if most_correlated else np.nan,
            "max_abs_corr_threshold": config.max_abs_corr,
            "passes_selected_correlation_check": (max_abs_corr < config.max_abs_corr) if most_correlated else True,
        })
    return pd.DataFrame(rows)


def _standardize_matrix(matrix: np.ndarray) -> np.ndarray:
    """Standardize each fitting column and replace unavailable standardized values with zero."""
    means = np.nanmean(matrix, axis=0)
    stds = np.nanstd(matrix, axis=0)
    stds = np.where((stds > 0.0) & np.isfinite(stds), stds, 1.0)
    standardized = (matrix - means) / stds
    return np.where(np.isfinite(standardized), standardized, 0.0)


def soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    """Apply the L1 proximal operator elementwise without changing coefficient signs."""
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


def fit_sparse_linear_weights(
    frame: pd.DataFrame,
    factors: list[str],
    config: PipelineConfig,
) -> pd.DataFrame:
    """Fit elastic-net-style prediction weights on complete fitting rows.
    The caller controls the time split; too few rows yield zero weights rather than evidence of validity."""
    if not factors:
        return pd.DataFrame(columns=["factor", "weight", "abs_weight", "selected_by_weight"])
    clean = frame[[*factors, config.target_col]].dropna()
    if len(clean) < max(20, len(factors) + 5):
        return pd.DataFrame({
            "factor": factors,
            "weight": np.zeros(len(factors), dtype=float),
            "abs_weight": np.zeros(len(factors), dtype=float),
            "selected_by_weight": [False] * len(factors),
        })
    x = _standardize_matrix(clean[factors].to_numpy(dtype=float))
    y = clean[config.target_col].to_numpy(dtype=float)
    y = y - float(np.mean(y))

    spectral_norm = np.linalg.norm(x, ord=2)
    lipschitz = (spectral_norm * spectral_norm / len(y)) + config.l2_alpha
    step = 1.0 / max(lipschitz, 1e-12)
    weights = np.zeros(len(factors), dtype=float)

    for _ in range(config.max_iter):
        residual = x @ weights - y
        grad = (x.T @ residual) / len(y) + config.l2_alpha * weights
        updated = soft_threshold(weights - step * grad, step * config.l1_alpha)
        if np.linalg.norm(updated - weights) <= config.tol * (1.0 + np.linalg.norm(weights)):
            weights = updated
            break
        weights = updated

    return pd.DataFrame({
        "factor": factors,
        "weight": weights,
        "abs_weight": np.abs(weights),
        "selected_by_weight": np.abs(weights) >= config.min_abs_weight,
    }).sort_values("abs_weight", ascending=False).reset_index(drop=True)


def stationary_bootstrap_dates(
    unique_dates: np.ndarray,
    mean_block_length: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw a circular stationary bootstrap with geometric block lengths.
    Adjacent dates stay together until a random restart preserves some temporal dependence."""
    if mean_block_length <= 1.0:
        raise ValueError("mean_block_length must be greater than 1.")
    n_dates = len(unique_dates)
    if n_dates == 0:
        return unique_dates
    restart_probability = 1.0 / mean_block_length
    sampled = np.empty(n_dates, dtype=unique_dates.dtype)
    cursor = int(rng.integers(0, n_dates))
    for idx in range(n_dates):
        if idx > 0 and rng.random() < restart_probability:
            cursor = int(rng.integers(0, n_dates))
        sampled[idx] = unique_dates[cursor]
        cursor = (cursor + 1) % n_dates
    return sampled


def build_bootstrap_frame(
    frame: pd.DataFrame,
    sampled_dates: np.ndarray,
    config: PipelineConfig,
) -> pd.DataFrame:
    """Copy sampled date groups and assign distinct artificial dates, including repeated blocks."""
    chunks = []
    grouped = {date: group for date, group in frame.groupby(config.date_col, sort=False)}
    for out_idx, date in enumerate(sampled_dates):
        group = grouped[pd.Timestamp(date)].copy()
        group[config.date_col] = pd.Timestamp("1900-01-01") + pd.Timedelta(days=int(out_idx))
        chunks.append(group)
    return pd.concat(chunks, ignore_index=True)


def add_factor_noise(
    frame: pd.DataFrame,
    factors: list[str],
    config: PipelineConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Perturb observed exposures relative to their scale while preserving the missing-value mask."""
    noisy = frame.copy()
    for factor in factors:
        std = float(pd.to_numeric(noisy[factor], errors="coerce").std(skipna=True))
        if std > 0.0 and np.isfinite(std):
            noise = rng.normal(0.0, config.noise_std_fraction * std, size=len(noisy))
            mask = noisy[factor].notna().to_numpy()
            values = noisy[factor].to_numpy(dtype=float).copy()
            values[mask] = values[mask] + noise[mask]
            noisy[factor] = values
    return noisy


def synthetic_survival(
    frame: pd.DataFrame,
    factors: list[str],
    config: PipelineConfig,
) -> pd.DataFrame:
    """Measure directional stability and degradation under date bootstrap and exposure noise.
    This robustness diagnostic cannot distinguish a stable proxy from an independent signal."""
    if not factors:
        return pd.DataFrame(columns=[
            "factor", "real_metric", "metric_orientation", "trade_direction", "median_metric",
            "survival_rate", "sign_stability", "degradation", "passes_degradation",
        ])
    metric_by_date = long_short_by_date if config.survival_metric == "long_short" else rank_ic_by_date
    rng = np.random.default_rng(config.random_seed)
    unique_dates = np.array(sorted(frame[config.date_col].dropna().unique()))
    real_metrics: dict[str, float] = {}
    for factor in factors:
        series = metric_by_date(frame, factor, config)
        real_metrics[factor] = float(series.mean()) if len(series) else np.nan

    scenario_values = {factor: [] for factor in factors}
    for _ in range(config.n_scenarios):
        sampled_dates = stationary_bootstrap_dates(unique_dates, config.mean_block_length, rng)
        scenario = build_bootstrap_frame(frame, sampled_dates, config)
        scenario = add_factor_noise(scenario, factors, config, rng)
        for factor in factors:
            series = metric_by_date(scenario, factor, config)
            scenario_values[factor].append(float(series.mean()) if len(series) else np.nan)

    rows: list[dict[str, Any]] = []
    for factor, values in scenario_values.items():
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        real = real_metrics.get(factor, np.nan)
        orientation = np.sign(real) if np.isfinite(real) else np.nan
        if len(arr) == 0 or not np.isfinite(real):
            survival_rate = sign_stability = median_metric = degradation = np.nan
            passes_degradation = False
        else:
            sign = np.sign(real)
            sign_stability = float(np.mean(np.sign(arr) == sign))
            if config.directional_survival and sign != 0.0:
                survival_rate = float(np.mean((np.sign(arr) == sign) & (np.abs(arr) >= abs(config.survival_threshold))))
            else:
                survival_rate = float(np.mean(arr > config.survival_threshold))
            median_metric = float(np.median(arr))
            degradation = float(median_metric / real) if abs(real) > 1e-12 else np.nan
            passes_degradation = bool(np.isfinite(degradation) and degradation >= config.min_degradation_ratio)
        rows.append({
            "factor": factor,
            "real_metric": real,
            "metric_orientation": orientation,
            "trade_direction": "unknown" if not np.isfinite(orientation) else ("high_minus_low" if orientation >= 0 else "low_minus_high"),
            "median_metric": median_metric,
            "survival_rate": survival_rate,
            "sign_stability": sign_stability,
            "degradation": degradation,
            "passes_degradation": passes_degradation,
            "n_scenarios": config.n_scenarios,
        })
    return pd.DataFrame(rows)


def incremental_value_filter(
    frame: pd.DataFrame, candidate_factors: list[str], config: PipelineConfig
) -> pd.DataFrame:
    """Fama-MacBeth incremental-value test: per-date cross-sectional OLS of the target on the
    standardized redundancy-passed factors; keep a factor only if |FM t| >= min_incremental_tstat,
    i.e. it stays significant after jointly controlling for the other selected factors."""
    factors = list(config.factor_cols)
    result = pd.DataFrame({"factor": factors})
    result["fm_tstat"] = np.nan
    result["passes_incremental"] = False
    # Date groups are not necessarily months or independent observations. Report
    # the complete-case fits actually used, rather than the input's date count.
    result["fm_n_date_groups"] = int(frame[config.date_col].nunique())
    result["fm_n_periods"] = 0
    result["fm_required_periods"] = 5
    result["fm_n_complete_rows"] = 0
    result["fm_min_cross_section_obs"] = 0
    result["fm_max_cross_section_obs"] = 0
    result["fm_skipped_insufficient_obs"] = 0
    result["fm_skipped_fit_failed"] = 0
    result["fm_status"] = "not_tested"
    sel = [f for f in candidate_factors if f in frame.columns]
    if not sel:
        return result
    coefs: list[np.ndarray] = []
    sample_sizes: list[int] = []
    skipped_small = skipped_fit = 0
    for _, g in frame.groupby(config.date_col, sort=True):
        Z = g[sel].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(g[config.target_col], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(y) & np.all(np.isfinite(Z), axis=1)
        if int(ok.sum()) < len(sel) + 2:
            skipped_small += 1
            continue
        Z, y = Z[ok], y[ok]
        Z = (Z - Z.mean(axis=0)) / (Z.std(axis=0) + 1e-9)
        try:
            beta = np.linalg.lstsq(np.column_stack([np.ones(len(y)), Z]), y, rcond=None)[0][1:]
        except np.linalg.LinAlgError:
            skipped_fit += 1
            continue
        if not np.isfinite(beta).all():
            skipped_fit += 1
            continue
        coefs.append(beta)
        sample_sizes.append(int(ok.sum()))
    tested = result["factor"].isin(sel)
    for column, value in {
        "fm_n_periods": len(coefs), "fm_n_complete_rows": sum(sample_sizes),
        "fm_min_cross_section_obs": min(sample_sizes, default=0),
        "fm_max_cross_section_obs": max(sample_sizes, default=0),
        "fm_skipped_insufficient_obs": skipped_small,
        "fm_skipped_fit_failed": skipped_fit,
    }.items():
        result.loc[tested, column] = value
    if len(coefs) < 5:
        # Unavailable evidence is a refusal, never a passing gate.
        result.loc[tested, "fm_status"] = "insufficient_cross_sections"
        return result
    coef_mat = np.array(coefs)
    # Newey-West, not iid: the monthly coefficient series is autocorrelated because
    # overlapping forward returns share price paths, and an iid standard error
    # understates it -- inflating |t| exactly where the gate decides. The lag count is
    # the config's hac_lags, which the screen already uses.
    tstat = np.array([t_stat(pd.Series(coef_mat[:, i]), hac_lags=config.hac_lags)
                      for i in range(coef_mat.shape[1])])
    for i, f in enumerate(sel):
        result.loc[result["factor"] == f, "fm_tstat"] = float(tstat[i])
        result.loc[result["factor"] == f, "passes_incremental"] = bool(
            np.isfinite(tstat[i]) and abs(tstat[i]) >= config.min_incremental_tstat)
        result.loc[result["factor"] == f, "fm_status"] = (
            "nonfinite_statistic" if not np.isfinite(tstat[i]) else
            "passed" if abs(tstat[i]) >= config.min_incremental_tstat else "below_threshold")
    return result


def final_selection(
    screen: pd.DataFrame,
    redundancy: pd.DataFrame,
    weights: pd.DataFrame,
    survival: pd.DataFrame,
    incremental: pd.DataFrame,
    config: PipelineConfig,
) -> pd.DataFrame:
    """Join gate evidence and require the configured pass conditions for final selection.
    Missing gate outcomes must not be interpreted as affirmative evidence."""
    merged = screen.merge(redundancy, on=["factor", "passes_stat_screen"], how="left")
    merged = merged.merge(weights, on="factor", how="left")
    merged = merged.merge(survival, on="factor", how="left")
    merged = merged.merge(incremental, on="factor", how="left")
    for col in ("passes_incremental", "selected_by_weight", "passes_degradation"):
        merged[col] = merged[col].eq(True)
    for col in ("weight", "abs_weight"):
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)
    merged["passes_survival"] = (merged["survival_rate"] >= config.min_survival_rate) & merged["passes_degradation"]
    merged["selected_final"] = (
        merged["passes_stat_screen"].eq(True)
        & merged["passes_redundancy_screen"].eq(True)
        & merged["passes_incremental"]
        & merged["selected_by_weight"]
        & merged["passes_survival"].eq(True)
    )
    reasons: list[str] = []
    for row in merged.itertuples(index=False):
        row_reasons: list[str] = []
        passes_stat = bool(row.passes_stat_screen)
        passes_redundancy = bool(getattr(row, "passes_redundancy_screen", False))
        selected_by_weight = bool(row.selected_by_weight)
        if not passes_stat:
            row_reasons.append("failed_stat_screen")
        if bool(getattr(row, "is_redundant", False)):
            row_reasons.append(f"redundant_with:{getattr(row, 'redundant_with', '')}")
        if bool(getattr(row, "is_family_limited", False)):
            row_reasons.append(f"family_limit:{getattr(row, 'factor_family', '')}")
        if passes_stat and passes_redundancy and not bool(getattr(row, "passes_incremental", False)):
            row_reasons.append(
                "incremental_insufficient_data"
                if getattr(row, "fm_status", None) == "insufficient_cross_sections"
                else "failed_incremental_value")
        if passes_stat and passes_redundancy and not selected_by_weight:
            row_reasons.append("small_sparse_weight")
        if passes_stat and passes_redundancy and selected_by_weight and not bool(getattr(row, "passes_degradation", False)):
            row_reasons.append("failed_degradation")
        if passes_stat and passes_redundancy and selected_by_weight and not bool(row.passes_survival):
            row_reasons.append("failed_synthetic_survival")
        reasons.append(";".join(row_reasons))
    merged["exclude_reason"] = reasons
    return merged.sort_values(
        ["selected_final", "survival_rate", "abs_weight", "screen_score"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def write_outputs(output_dir: Path, outputs: dict[str, pd.DataFrame]) -> None:
    """Write the named diagnostic tables as UTF-8 CSV files in the output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")


LABEL_END_COL = "forward_return_end_date"


def purge_label_overlap(frame: pd.DataFrame, cutoff, date_col: str = "date", *,
                        embargo_days: int | None = None,
                        context: str = "training slice") -> tuple[pd.DataFrame, dict]:
    """Drop labels unavailable at the declared training cutoff.

    Signal dates alone do not determine availability: a forward return uses subsequent
    prices. Purge by forward_return_end_date when present; otherwise require an
    explicit positive embargo. Refuse labels settling before their own signal."""
    cutoff = pd.to_datetime(cutoff)
    dates = pd.to_datetime(frame[date_col])
    # The cutoff is unconditional: whatever the embargo says, a row dated after the
    # cutoff is never training data. An explicit embargo must be positive.
    in_window = dates <= cutoff
    if LABEL_END_COL in frame.columns:
        end = pd.to_datetime(frame[LABEL_END_COL])
        corrupt = in_window & end.notna() & (end < dates)
        if bool(corrupt.any()):
            raise ValueError(
                f"{context}: {int(corrupt.sum())} rows have {LABEL_END_COL} earlier than "
                "their own date; a label cannot settle before its signal")
        overlap = in_window & (end.isna() | (end > cutoff))
        kept = frame.loc[in_window & ~overlap]
        kept_end = end.loc[in_window & ~overlap]
        return kept, {"label_end_col_present": True, "embargo_days": None,
                      "n_purged_label_overlap": int(overlap.sum()),
                      "max_train_label_end": (str(kept_end.max().date())
                                              if len(kept_end) else None)}
    if embargo_days is None:
        raise ValueError(
            f"{context}: the panel has no {LABEL_END_COL} column and no label embargo was "
            "given. date <= cutoff alone leaks post-cutoff prices through the last "
            "training labels; rebuild the panel with the current builder or set "
            "label_embargo_days explicitly.")
    if isinstance(embargo_days, bool) or not isinstance(embargo_days, (int,)) or embargo_days <= 0:
        raise ValueError(
            f"{context}: label_embargo_days must be a positive integer number of CALENDAR "
            f"days covering the full label horizon, got {embargo_days!r}")
    edge = cutoff - pd.Timedelta(days=int(embargo_days))
    kept = frame.loc[in_window & (dates <= edge)]
    return kept, {"label_end_col_present": False, "embargo_days": int(embargo_days),
                  "n_purged_label_overlap": int(in_window.sum() - len(kept)),
                  "max_train_label_end": None}


MAX_LABEL_SPAN_DAYS_DEFAULT = 45


def label_end_series(frame: pd.DataFrame, *, context: str,
                     date_col: str = "date", target_col: str = "forward_return",
                     allow_missing: bool = False) -> pd.Series | None:
    """The settlement dates, or None when the caller has declared there is no horizon.

    One place decides what "available" means, so the oracle, the manual lockbox, the
    walk-forward and the tuning scripts cannot drift apart again."""
    if LABEL_END_COL in frame.columns:
        end = pd.to_datetime(frame[LABEL_END_COL])
        target = frame.get(target_col)
        if date_col not in frame.columns:
            raise ValueError(f"{context}: no {date_col!r} column to validate "
                             f"{LABEL_END_COL} against")
        signal = pd.to_datetime(frame[date_col])
        bad_order = end.notna() & (end < signal)
        if bool(bad_order.any()):
            raise ValueError(
                f"{context}: {int(bad_order.sum())} rows settle before their own signal "
                "date; the panel's settlement dates are corrupt")
        if target is not None:
            orphan = pd.to_numeric(target, errors="coerce").notna() & end.isna()
            if bool(orphan.any()):
                raise ValueError(
                    f"{context}: {int(orphan.sum())} rows carry a forward_return with no "
                    "settlement date, so their availability cannot be established")
        return end
    if allow_missing:
        return None
    raise ValueError(
        f"{context}: the panel has no {LABEL_END_COL} column, so rows carrying "
        "post-cutoff prices in their labels cannot be identified. Migrate the panel with "
        "python -m factor_research.migrate_label_end_dates.")


def cohort_split(frame: pd.DataFrame, train_end, val_end, *, date_col: str = "date",
                 context: str = "split") -> pd.Series:
    """Assign each row to the segment where its label becomes available.

    Settlement dates keep post-boundary prices out of training and validation.
    A missing settlement date places the row on the conservative test side."""
    end = label_end_series(frame, context=context, date_col=date_col)
    train_end, val_end = pd.to_datetime(train_end), pd.to_datetime(val_end)
    # A row whose label never settles cannot be evaluated anywhere; it is test-side by
    # the same conservative rule as split_period_by_settlement, not signal-dated.
    settle = end.fillna(pd.Timestamp.max)
    return pd.Series(
        pd.cut(settle, bins=[pd.Timestamp.min, train_end, val_end, pd.Timestamp.max],
               labels=["train", "validation", "test"]).astype(str),
        index=frame.index)


def split_period_by_settlement(period, settle_max: dict, train_end, val_end) -> str:
    """Assign each signal cohort by the latest settlement date of its usable labels.

    A cohort belongs to the segment where its labels become available. Missing
    settlement information cannot qualify it for training or validation."""
    latest = settle_max.get(period)
    if latest is None or pd.isna(latest):
        return "test"
    stamp = pd.Timestamp(latest)
    return ("train" if stamp <= pd.Timestamp(train_end)
            else ("validation" if stamp <= pd.Timestamp(val_end) else "test"))


def settled_history(settle_max: dict, formation, window: int) -> list:
    """Return the last requested number of months fully settled before formation.

    Filter by settlement first, then take the trailing window. If eligible history
    is short, return fewer months so the caller can refuse insufficient evidence."""
    formation = pd.to_datetime(formation)
    available = sorted(m for m, latest in settle_max.items() if latest < formation)
    return available[-window:]


def run_pipeline(config: PipelineConfig) -> dict[str, pd.DataFrame]:
    """Fit all selection steps only on labels settled by the required training cutoff.
    Write the diagnostic tables and provenance needed by downstream optimizers."""
    frame = load_panel(config)
    if not config.train_end:
        raise ValueError(
            "train_end is required: computing the screen, survival and sparse weights on "
            "the full sample would let evaluation data influence selection. Pass "
            "--train-end, or set train_end in the config.")
    fit, label_purge = purge_label_overlap(
        frame, config.train_end, config.date_col,
        embargo_days=config.label_embargo_days, context="factor selection")
    if fit.empty:
        raise ValueError(f"train_end {config.train_end} leaves no rows to fit on")
    screen = single_factor_screen(fit, config)
    corr = factor_correlation(fit, config)
    corr_pairs = factor_correlation_pairs(corr, config)
    redundancy = redundancy_filter(screen, corr, config)
    redundancy_passed = redundancy.loc[redundancy["passes_redundancy_screen"], "factor"].astype(str).tolist()
    incremental = incremental_value_filter(fit, redundancy_passed, config)
    candidate_factors = incremental.loc[incremental["passes_incremental"], "factor"].astype(str).tolist()
    weights = fit_sparse_linear_weights(fit, candidate_factors, config)
    weighted_factors = weights.loc[weights["selected_by_weight"], "factor"].astype(str).tolist()
    survival_factors = weighted_factors if weighted_factors else candidate_factors
    survival = synthetic_survival(fit, survival_factors, config)
    final = final_selection(screen, redundancy, weights, survival, incremental, config)
    selected_corr_audit = selected_factor_correlation_audit(corr, final, config)
    family_audit = factor_family_audit(final, config)
    outputs = {
        "single_factor_screen": screen,
        "factor_correlation": corr.reset_index().rename(columns={"index": "factor"}),
        "factor_correlation_pairs": corr_pairs,
        "selected_factor_correlation_audit": selected_corr_audit,
        "factor_family_audit": family_audit,
        "redundancy_report": redundancy,
        "incremental_value": incremental,
        "sparse_weights": weights,
        "synthetic_survival": survival,
        "final_factor_selection": final,
        "excluded_factors_with_reasons": final.loc[~final["selected_final"]].copy(),
        "optimization_selected_factors": weights.loc[weights["selected_by_weight"]].copy(),
        "final_strong_valid_factors": final.loc[final["selected_final"]].copy(),
    }
    write_outputs(config.output_dir, outputs)
    # The manifest is what lets the solver refuse a selection whose provenance it cannot
    # check: which panel, which config, and what the selection was allowed to see.
    from factor_research.experiment_manifest import build_manifest, file_sha256, write_strict_json
    write_strict_json(
        Path(config.output_dir) / "selection_manifest.json",
        build_manifest(panel_csv=config.input_csv, config_path=config.config_path,
                       extra={"train_end": str(config.train_end),
                              "selection_csv_sha256": file_sha256(
                                  Path(config.output_dir) / "final_factor_selection.csv"),
                              "n_rows_total": int(len(frame)), "n_rows_fit": int(len(fit)),
                              "label_purge": label_purge,
                              "factor_cols": list(config.factor_cols)}))
    return outputs


def main() -> None:
    """Load the config, apply the training cutoff and write selection artifacts.

    A cutoff is required so evaluation labels cannot enter screening or fitting.
    The saved manifest lets downstream solvers verify the selected factor set.
    """
    parser = argparse.ArgumentParser(description="Run training-only factor validity selection.")
    parser.add_argument("--config", required=True, help="Path to JSON config.")
    parser.add_argument("--label-embargo-days", type=int, default=None,
                        help="fallback purge window for panels without "
                             "forward_return_end_date (fail-closed otherwise)")
    parser.add_argument("--train-end", required=True,
                        help="last date the selection may see (e.g. 2019-12-31); "
                             "everything later is evaluation-only")
    args = parser.parse_args()
    import dataclasses
    config = dataclasses.replace(load_config(args.config), train_end=args.train_end)
    if args.label_embargo_days is not None:
        config = dataclasses.replace(config, label_embargo_days=args.label_embargo_days)
    outputs = run_pipeline(config)
    final = outputs["final_factor_selection"]
    selected = final.loc[final["selected_final"], "factor"].astype(str).tolist()
    print(f"Wrote outputs to: {config.output_dir}")
    print(f"Selected factors ({len(selected)}): {', '.join(selected) if selected else '(none)'}")


if __name__ == "__main__":
    main()
