"""Regression checks for B2 aggregation, minimum evidence and protocol decisions."""
import json
from copy import deepcopy

import pandas as pd
import pytest
import agent.b2_compare as b2

from agent.b2_compare import (
    DEFAULT_PROTOCOL,
    _generator_summary,
    _go_no_go,
    _normalise_decision,
    _panel_hash,
    _proposal_quality,
    load_protocol,
)


def _row(generator: str, scenario: str, *, seed: int, attempts: int = 4,
         trials: int = 4, accepts: int = 1, true_accepts: int = 1) -> dict:
    """Build a synthetic arm result with explicit counts for aggregation tests."""
    false_accepts = accepts - true_accepts
    candidates = [
        {"name": f"{generator}_{seed}_{index}", "is_true_candidate": index == 0}
        for index in range(trials)
    ]
    return {
        "generator": generator,
        "scenario": scenario,
        "seed": seed,
        "generation_attempts": attempts,
        "oracle_trials": trials,
        "n_eff": trials,
        "family_adherence": 1.0,
        "nonduplicate_rate": 1.0,
        "n_true_candidates": 1 if scenario == "alternative" else 0,
        "n_would_be_accept": accepts,
        "n_true_accept": true_accepts,
        "n_false_accept": false_accepts,
        "power_recovered": int(true_accepts > 0) if scenario == "alternative" else None,
        "input_tokens": 10 if generator == "glm" else 0,
        "output_tokens": 2 if generator == "glm" else 0,
        "total_tokens": 12 if generator == "glm" else 0,
        "latency_ms": 100 if generator == "glm" else 0,
        "candidates": candidates,
    }


def test_preregistered_protocol_is_frozen_to_paired_momentum() -> None:
    protocol = load_protocol(DEFAULT_PROTOCOL)
    assert protocol["family"] == "momentum"
    assert protocol["dgp"]["true_factor"] == "mom1m"
    assert protocol["generators"] == ["random", "adaptive", "glm"]
    assert protocol["budgets"]["generation_attempts"] >= protocol["budgets"]["oracle_trials"]
    assert set(protocol["dgp"]["alternative_seeds"]).isdisjoint(
        protocol["dgp"]["null_seeds"])
    assert "api_key" not in json.dumps(protocol).casefold()


def test_protocol_rejects_truth_or_budget_changes(tmp_path) -> None:
    protocol = load_protocol(DEFAULT_PROTOCOL)
    changed = deepcopy(protocol)
    changed["dgp"]["true_factor"] = "bm"
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen to mom1m"):
        load_protocol(path)

    changed = deepcopy(protocol)
    changed["budgets"] = {"generation_attempts": 2, "oracle_trials": 4}
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="generation budget"):
        load_protocol(path)


def test_b2_normalises_only_uncertified_accept_side_drafts() -> None:
    assert _normalise_decision({
        "decision": "DRAFT", "reject_reason": "uncertified_would_be:ACCEPT",
    }) == ("ACCEPT", None)
    assert _normalise_decision({
        "decision": "DRAFT",
        "reject_reason": "uncertified_would_be:SHELVED;true_but_no_val_sharpe_increment",
    }) == ("SHELVED", "true_but_no_val_sharpe_increment")
    assert _normalise_decision({
        "decision": "REJECT", "reject_reason": "failed_synthetic_survival",
    }) == ("REJECT", "failed_synthetic_survival")


def test_panel_hash_is_content_sensitive_and_deterministic() -> None:
    panel = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-31", "2020-02-29"]),
        "asset": ["a", "a"],
        "mom1m": [0.1, 0.2],
    })
    assert _panel_hash(panel) == _panel_hash(panel.copy())
    changed = panel.copy()
    changed.loc[1, "mom1m"] = 0.3
    assert _panel_hash(changed) != _panel_hash(panel)


def test_proposal_quality_uses_returned_proposals_not_factorcards() -> None:
    # Five returned proposals can produce only four FactorCards when one later fails coverage.
    # The quality rates must still use five as their denominator and can never exceed one.
    adherence, nonduplicate = _proposal_quality(
        generated_proposals=5, family_adherent=5, unique_signatures=5)
    assert adherence == 1.0
    assert nonduplicate == 1.0
    assert _proposal_quality(
        generated_proposals=0, family_adherent=0, unique_signatures=0) == (None, None)


def test_summary_and_preregistered_go_no_go_are_denominator_aware() -> None:
    protocol = load_protocol(DEFAULT_PROTOCOL)
    rows = []
    for generator in ("random", "adaptive", "glm"):
        for seed in protocol["dgp"]["alternative_seeds"]:
            rows.append(_row(generator, "alternative", seed=seed))
        for seed in protocol["dgp"]["null_seeds"]:
            rows.append(_row(
                generator, "null", seed=seed, accepts=0, true_accepts=0))
    first_glm = next(row for row in rows if row["generator"] == "glm")
    first_glm["response_local_failures"] = 1
    first_glm["runtime_coverage_failures"] = 2
    summaries = [_generator_summary(generator, rows)
                 for generator in ("random", "adaptive", "glm")]
    glm = next(summary for summary in summaries if summary["generator"] == "glm")
    assert glm["would_be_accepts"] == 5
    assert glm["pooled_empirical_fdp"] == 0.0
    assert glm["mean_power_recovered"] == 1.0
    assert glm["total_tokens"] == 84
    assert glm["response_local_failures"] == 1
    assert glm["runtime_coverage_failures"] == 2
    result = _go_no_go(protocol, summaries)
    assert result["status"] == "GO"
    assert all(value is True for value in result["checks"].values())

    reduced_rows = [row for row in rows
                    if not (row["generator"] == "glm" and
                            row["scenario"] == "alternative" and row["seed"] != 7)]
    reduced = [_generator_summary(generator, reduced_rows)
               for generator in ("random", "adaptive", "glm")]
    result = _go_no_go(protocol, reduced)
    assert result["status"] == "INCONCLUSIVE"
    assert result["checks"]["sufficient_accept_denominator"] is False


@pytest.mark.parametrize("change", ["config", "missing_config_hash", "source", "format_only"])
def test_resume_checks_config_and_source_before_calibration(tmp_path, monkeypatch, change):
    """Refuse mixed evidence before generating panels; permit equivalent JSON formatting."""
    protocol = load_protocol(DEFAULT_PROTOCOL)
    protocol["output_dir"] = str(tmp_path / "output")
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    config = {"factor_cols": ["mom1m"], "screening": {"q_value_max": 0.05}}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    out = tmp_path / "output"
    out.mkdir()
    manifest = {"protocol_hash": b2._canonical_hash(protocol),
                "config_hash": b2._canonical_hash(config),
                "source_fingerprint": b2._source_fingerprint()}
    if change == "config":
        config["screening"]["q_value_max"] = 0.1
    elif change == "missing_config_hash":
        manifest.pop("config_hash")
    elif change == "source":
        manifest["source_fingerprint"] = "old-source"
    config_path.write_text(json.dumps(config, sort_keys=True, indent=4), encoding="utf-8")
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    before = manifest_path.read_bytes()

    def calibration_boundary(*args, **kwargs):
        raise RuntimeError("calibration reached")

    monkeypatch.setattr(b2, "calibrate", calibration_boundary)
    monkeypatch.setattr(b2.sys, "argv", ["b2", "--protocol", str(protocol_path),
                                       "--config", str(config_path), "--generators", "random"])
    expected = ("calibration reached" if change == "format_only" else
                "source fingerprint changed" if change == "source" else "config hash changed")
    with pytest.raises(RuntimeError, match=expected):
        b2.main()
    assert manifest_path.read_bytes() == before


def test_source_binding_includes_dsr_certificate_and_gen2():
    """Every shared gate dependency must invalidate B2 resume after a source edit."""
    assert set(b2.GATE_SOURCE_FILES) <= set(b2._SOURCE_FILES)
    assert {p.name for p in b2._SOURCE_FILES} >= {
        "deflated_sharpe.py", "certification.py", "dgp_v2.py"}
