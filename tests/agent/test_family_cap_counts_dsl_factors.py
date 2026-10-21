"""Family caps include all validated DSL factors already present in the library."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import dataclasses

from agent.validator_oracle import ValidatorOracle  # noqa: E402
from factor_research.factor_validity_pipeline import infer_factor_family, load_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def _bare_oracle() -> ValidatorOracle:
    oracle = ValidatorOracle.__new__(ValidatorOracle)
    oracle.base_cfg = dataclasses.replace(
        load_config(ROOT / "a_share_big_config.json"),
        factor_families={"mom6m": "momentum"})
    oracle.panel_csv = Path("unused.csv")
    oracle._candidate_families = {}
    return oracle


def test_make_cfg_carries_dsl_families_to_the_classifier():
    oracle = _bare_oracle()
    oracle._candidate_families["llm_0001"] = "momentum"
    oracle._candidate_families["baseline_001"] = "liquidity"
    cfg = oracle._make_cfg("llm_0002", "momentum", ["llm_0001", "baseline_001"])
    assert infer_factor_family("llm_0001", cfg) == "momentum"
    assert infer_factor_family("baseline_001", cfg) == "liquidity"
    assert infer_factor_family("llm_0002", cfg) == "momentum"


def test_family_count_reaches_the_cap_with_dsl_factors():
    """two accepted momentum DSL factors, cap 2 -- the next
    momentum candidate must see a full family, not an empty one."""
    oracle = _bare_oracle()
    oracle._candidate_families = {"llm_0001": "momentum", "llm_0002": "momentum"}
    cfg = oracle._make_cfg("llm_0003", "momentum", ["llm_0001", "llm_0002"])
    existing = ["llm_0001", "llm_0002"]
    n_same = sum(1 for f in existing if infer_factor_family(f, cfg) == "momentum")
    assert n_same == 2


def test_unknown_names_still_unclassified():
    oracle = _bare_oracle()
    cfg = oracle._make_cfg("cand", "momentum", [])
    assert infer_factor_family("mystery_zzz", cfg) == "unclassified"


def test_panel_switch_clears_the_derived_registry():
    """Reusing one oracle across panels must not leak the previous panel's DSL
    expressions or family counts into the next experiment."""
    import pandas as pd
    oracle = ValidatorOracle(ROOT / "a_share_big_config.json")
    oracle.derived["llm_0001"] = "rank(ep)"
    oracle._candidate_families["llm_0001"] = "momentum"
    frame = pd.DataFrame({
        oracle.base_cfg.date_col: pd.to_datetime(["2020-01-31", "2020-02-29"]),
        oracle.base_cfg.asset_col: ["A0", "A0"],
        "forward_return": [0.0, 0.1]})
    oracle.set_panel(frame, "synthetic_switch_check", label_semantics="direct")
    assert oracle.derived == {}
    assert oracle._candidate_families == {}
