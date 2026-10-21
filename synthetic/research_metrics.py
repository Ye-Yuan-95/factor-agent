"""Explicit sample accounting for optional threshold experiments.

Counts are factor/panel pairs, not independent trials. Undefined conditional
rates have a missing value; early screening and unavailable FM estimates remain
in the whole-pipeline denominators and cannot silently disappear from power.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class ThresholdPopulation:
    """Retain total, screened, estimable and selected populations separately."""

    fake: list[float] = field(default_factory=list)
    true: list[float] = field(default_factory=list)
    n_fake_total: int = 0
    n_true_total: int = 0
    samples: list[dict] = field(default_factory=list)

    def add(self, truth, survivors, statistics) -> None:
        """Account for one panel including factors rejected before the FM gate."""
        truth = np.asarray(truth, dtype=bool)
        self.n_fake_total += int((~truth).sum())
        self.n_true_total += int(truth.sum())
        for index in survivors:
            (self.true if truth[index] else self.fake).append(
                float(statistics.get(index, np.nan))
            )
        diagnostics = getattr(statistics, "diagnostics", {})
        self.samples.append(dict(diagnostics))

    def __iter__(self):
        """Support two-array unpacking of the collected statistics."""
        yield np.array([x for x in self.fake if np.isfinite(x)])
        yield np.array([x for x in self.true if np.isfinite(x)])

    def at(self, threshold: float) -> dict:
        """Return counts and rates using unrounded statistics at one cutoff."""
        fake, true = tuple(self)
        false_selected = int((fake >= threshold).sum())
        true_selected = int((true >= threshold).sum())
        return {
            "n_fake_total": self.n_fake_total,
            "n_true_total": self.n_true_total,
            "n_fake_reaching": len(self.fake),
            "n_true_reaching": len(self.true),
            "n_fake_finite": len(fake),
            "n_true_finite": len(true),
            "n_fake_invalid": len(self.fake) - len(fake),
            "n_true_invalid": len(self.true) - len(true),
            "n_fake_screened_out": self.n_fake_total - len(self.fake),
            "n_true_screened_out": self.n_true_total - len(self.true),
            "n_false_selected": false_selected,
            "n_true_selected": true_selected,
            "conditional_size": false_selected / len(fake) if len(fake) else np.nan,
            "conditional_power": true_selected / len(true) if len(true) else np.nan,
            "full_fpr": false_selected / self.n_fake_total
            if self.n_fake_total
            else np.nan,
            "full_power": true_selected / self.n_true_total
            if self.n_true_total
            else np.nan,
        }


def write_sample_accounting(
    path: Path, populations: dict[str, ThresholdPopulation]
) -> None:
    """Save per-fit effective sample sizes without claiming independent observations."""
    from factor_research.experiment_manifest import write_strict_json

    write_strict_json(
        path,
        {
            "schema_version": 2,
            "unit": "factor-panel pairs; candidates within a panel are dependent",
            "conditional_denominator": "screen survivors with finite FM statistics",
            "full_denominator": "all injected true/false factors before screening",
            "minimum_fm_periods": 5,
            "minimum_is_not_a_power_guarantee": True,
            "populations": {
                name: {
                    "n_fake_total": p.n_fake_total,
                    "n_true_total": p.n_true_total,
                    "samples": p.samples,
                }
                for name, p in populations.items()
            },
        },
    )


def ic_sample_accounting(panel, factors, ic) -> dict:
    """Report actual IC counts per factor rather than treating every date as usable."""
    sizes = panel.groupby("date").size()
    dates = pd.to_datetime(panel["date"], errors="coerce")
    return {
        "raw_date_groups": len(sizes),
        "calendar_months": int(dates.dt.to_period("M").nunique()),
        "median_assets_per_date": float(sizes.median()),
        "mean_assets_per_date": float(sizes.mean()),
        "usable_ic_dates": {
            factor: int(np.isfinite(ic[:, i]).sum()) for i, factor in enumerate(factors)
        },
        "counts_are_not_independent_sample_sizes": True,
    }
