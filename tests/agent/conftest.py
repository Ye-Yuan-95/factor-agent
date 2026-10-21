"""Shared fixtures: a tiny known-truth panel (premium ONLY on column `a`), a matching pipeline
config, and oracle/certificate factories. Everything is sized for speed (< ~10 s per gate test)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PROJECT_DIR = Path(__file__).resolve().parents[2]
for p in (str(_PROJECT_DIR), str(_PROJECT_DIR / "synthetic")):
    if p not in sys.path:
        sys.path.insert(0, p)

from agent.certification import compute_binding, content_hash  # noqa: E402
from agent.validator_oracle import ValidatorOracle  # noqa: E402

N_ASSETS, N_MONTHS = 80, 72
DATES = pd.date_range("2018-01-31", periods=N_MONTHS, freq="ME")  # ends 2023-12 -> test rows exist
TRAIN_END = pd.Timestamp(DATES[41])


def _build_panel(seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for d in DATES:
        a = rng.standard_normal(N_ASSETS)
        b = 0.7 * a + 0.7 * rng.standard_normal(N_ASSETS)      # correlated proxy of a
        c = rng.standard_normal(N_ASSETS)                       # independent
        fwd = 0.02 * a + 0.05 * rng.standard_normal(N_ASSETS)   # premium ONLY on a
        for i in range(N_ASSETS):
            rows.append({"date": d, "asset": f"A{i}", "a": a[i], "b": b[i], "c": c[i],
                         "forward_return": fwd[i]})
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def tiny_panel() -> pd.DataFrame:
    return _build_panel()


@pytest.fixture(scope="session")
def tiny_config_path(tmp_path_factory) -> Path:
    cfg = {"input_csv": "x.csv", "output_dir": "y", "factor_cols": ["a", "b", "c"],
           "family_constraints": {"factor_families": {"a": "f1", "b": "f2", "c": "f3"},
                                  "max_per_family": {}},
           "screening": {"q_value_max": 0.25, "min_coverage": 0.3},
           "redundancy": {"max_abs_corr": 0.85}, "incremental": {"min_abs_tstat": 2.0},
           "optimization": {}, "synthetic": {"n_scenarios": 10}}
    path = tmp_path_factory.mktemp("cfg") / "tiny_config.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def tiny_panel_csv(tmp_path_factory, tiny_panel) -> Path:
    path = tmp_path_factory.mktemp("panel") / "tiny_panel.csv"
    frame = tiny_panel.copy()
    # Once written to disk the oracle cannot know these labels are drawn rather than
    # differenced from later prices, and fails closed. Synthetic labels settle on their
    # own signal date, so the column states exactly that.
    frame["forward_return_end_date"] = frame["date"]
    frame.to_csv(path, index=False)
    return path


@pytest.fixture()
def oracle(tiny_config_path, tiny_panel) -> ValidatorOracle:
    o = ValidatorOracle(tiny_config_path)
    o.train_end = TRAIN_END
    o.val_end = pd.Timestamp(DATES[-1])
    o.certificate_path = Path("nonexistent_certificate.json")   # uncertified by default
    o.set_panel(tiny_panel, "data_tiny", label_semantics="direct")
    return o


def make_bound_certificate(oracle: ValidatorOracle, config_path: Path, panel_csv: Path,
                           out_path: Path, accepted: bool = True) -> Path:
    """Issue a certificate whose binding matches the oracle's CURRENT state."""
    acceptance = {"target_fpr": 0.1, "target_power": 0.3, "conf": 0.95, "rule": "test"}
    binding = compute_binding(
        pipeline_config_raw=json.loads(Path(config_path).read_text(encoding="utf-8")),
        oracle_gate_params=oracle.oracle_gate_params(),
        panel_content_hash=content_hash(panel_csv), acceptance=acceptance)
    cert = {"binding": binding, "accepted": accepted, "gate_fpr": 0.01,
            "gate_power": 0.6, "gate_power_alpha": "IC=0.03",
            "calibration_panel": str(panel_csv)}
    out_path.write_text(json.dumps(cert), encoding="utf-8")
    return out_path
