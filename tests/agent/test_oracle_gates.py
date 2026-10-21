"""Five gates + controls + certificate-gated decisions."""

import numpy as np
import pandas as pd
import pytest

from agent.memory import MemoryStore
from conftest import make_bound_certificate


def _validate(oracle, tmp_path, name, expr, existing=()):
    mem = MemoryStore(tmp_path / f"mem_{name}")
    return oracle.validate(name=name, expression=expr, family="f1",
                           existing=list(existing), memory=mem)


def test_clean_recovery_passes_gates_but_drafts_without_cert(oracle, tmp_path):
    card = _validate(oracle, tmp_path, "clean", "zscore(a)")
    assert card.fdr_pass and card.incremental_pass and card.selected_by_weight
    assert "a" not in card.incremental_controls          # no self-regression
    assert not card.certified
    assert card.decision == "DRAFT"
    assert str(card.reject_reason).startswith("uncertified_would_be:")


def test_borrowed_proxy_killed_by_neighbour_control(oracle, tmp_path):
    card = _validate(oracle, tmp_path, "proxy", "rank(b)")
    assert "a" in card.incremental_controls
    assert not card.incremental_pass
    assert card.decision == "REJECT"


def test_evasion_chain_caught_by_ingredient_control(oracle, tmp_path):
    card = _validate(oracle, tmp_path, "evade", "delta(ts_zscore(b, 6), 3)")
    assert "b" in card.incremental_controls              # mandatory AST ingredient
    assert not card.incremental_pass


def test_exact_span_rejected_not_spuriously_significant(oracle, tmp_path):
    card = _validate(oracle, tmp_path, "dilute", "a + c")
    assert card.control_span_r2 is not None and card.control_span_r2 >= 0.9
    assert not card.incremental_pass
    assert "spanned_by_controls" in card.reject_reason


def test_degenerate_factor_rejected(oracle, tmp_path):
    card = _validate(oracle, tmp_path, "degen", "sign(abs(a))")
    assert card.decision == "REJECT"
    assert "no_long_short_series" in card.reject_reason


def test_bound_certificate_enables_accept(oracle, tmp_path, tiny_config_path, tiny_panel_csv):
    oracle.panel_csv = tiny_panel_csv   # applicability: cert binds its calibration panel
    oracle.certificate_path = make_bound_certificate(
        oracle, tiny_config_path, tiny_panel_csv, tmp_path / "cert.json")
    card = _validate(oracle, tmp_path, "clean2", "zscore(a)")
    assert card.certified and card.confidence == "high"
    assert card.decision in ("ACCEPT", "SHELVED")        # certified: no DRAFT
    refs = oracle.last_certificate_refs
    assert refs["bound"] and refs["status"] == "bound"


def test_stale_certificate_is_ignored(oracle, tmp_path, tiny_config_path, tiny_panel_csv):
    oracle.panel_csv = tiny_panel_csv   # applicability: cert binds its calibration panel
    oracle.certificate_path = make_bound_certificate(
        oracle, tiny_config_path, tiny_panel_csv, tmp_path / "cert.json")
    oracle.incremental_control_k = 4                     # gate param changed AFTER issue
    card = _validate(oracle, tmp_path, "clean3", "zscore(a)")
    assert not card.certified
    assert card.decision in ("DRAFT", "REJECT")
    assert "stale" in oracle.last_certificate_refs["status"]


def test_unaccepted_certificate_does_not_certify(oracle, tmp_path, tiny_config_path, tiny_panel_csv):
    oracle.panel_csv = tiny_panel_csv   # applicability: cert binds its calibration panel
    oracle.certificate_path = make_bound_certificate(
        oracle, tiny_config_path, tiny_panel_csv, tmp_path / "cert.json", accepted=False)
    card = _validate(oracle, tmp_path, "clean4", "zscore(a)")
    assert not card.certified and card.confidence == "medium"
    assert card.decision != "ACCEPT"


def test_synthetic_and_real_memories_do_not_mix(oracle, tmp_path):
    """Universe is stamped per trial; a duplicate expression in ANOTHER universe still counts."""
    mem = MemoryStore(tmp_path / "mem_iso")
    oracle.validate(name="u1", expression="rank(c)", family="f3", existing=[], memory=mem)
    assert mem.counted_trials()[-1]["universe"] == "data_tiny"
    panel2 = oracle.load_panel().copy() if oracle._panel is None else oracle._panel.copy()
    oracle.set_panel(panel2, "synthetic_gen1_test", label_semantics="direct")
    oracle.validate(name="u2", expression="rank(c)", family="f3", existing=[], memory=mem)
    universes = {t["universe"] for t in mem.counted_trials()}
    assert universes == {"data_tiny", "synthetic_gen1_test"}
    assert mem.n_trials() == 2                            # same expr, different universe: counts


def test_certificate_requires_matching_active_panel(oracle, tmp_path, tiny_config_path, tiny_panel_csv):
    """a certificate only applies to its calibration panel's exact content."""
    oracle.certificate_path = make_bound_certificate(
        oracle, tiny_config_path, tiny_panel_csv, tmp_path / "cert.json")
    # oracle's active panel is the in-memory synthetic 'data_tiny' -> not certifiable
    card = _validate(oracle, tmp_path, "clean5", "zscore(a)")
    assert not card.certified and card.decision != "ACCEPT"
    assert "panel-not-certifiable" in oracle.last_certificate_refs["status"]


def test_insufficient_control_samples_reject_and_explain(oracle, tiny_panel, tmp_path, monkeypatch):
    """A candidate can have many dates while its joint controls permit only four fits."""
    panel = tiny_panel.copy()
    usable = sorted(panel.date.unique())[:4]
    panel.loc[~panel.date.isin(usable), "b"] = float("nan")
    oracle.set_panel(panel, "data_tiny", label_semantics="direct")
    monkeypatch.setattr(oracle, "_select_base_controls", lambda *a, **k: ["b"])
    card = oracle.validate(name="boundary_alpha", expression="a", rationale="boundary",
                           family="f1", existing=[], n_trials=1)
    assert card.incremental_n_periods == 4
    assert card.incremental_n_complete_rows == 320
    assert card.incremental_sample_status == "insufficient_cross_sections"
    assert card.incremental_t is None and not card.incremental_pass
    assert card.decision == "REJECT"
    assert "incremental_insufficient_data" in card.reject_reason


@pytest.mark.parametrize("dsr,expected", [(0.9499996, False), (0.95, True)])
def test_dsr_boundary_uses_full_precision(oracle, monkeypatch, dsr, expected):
    monkeypatch.setattr("agent.validator_oracle.deflated_sharpe",
                        lambda *a, **k: {"deflated_sharpe": dsr, "sharpe_ref": 0.0})
    card = oracle.validate(name="precise_dsr", expression="a", rationale="boundary test",
                           family="f1", existing=[], n_trials=1)
    assert card.deflated_sharpe == dsr
    assert card.dsr_pass is expected


def test_control_span_boundary_is_not_rounded_up(oracle):
    rng = np.random.default_rng(1729)
    x = rng.normal(size=20)
    x = (x - x.mean()) / x.std()
    noise = rng.normal(size=20)
    noise -= noise.mean()
    noise -= x * (noise @ x) / (x @ x)
    noise /= noise.std()
    target = 0.89996
    y = np.sqrt(target) * x + np.sqrt(1 - target) * noise
    frame = pd.DataFrame({"date": pd.Timestamp("2010-01-31"), "a": x, "candidate": y})
    r2 = oracle._control_span_r2(frame, "candidate", ["a"])
    assert r2 == pytest.approx(target)
    assert r2 < oracle.control_span_r2_max


def test_small_positive_investment_delta_retains_its_sign(oracle, tiny_panel, monkeypatch):
    readings = iter([{"sharpe": 0.00004, "turnover": 0.2},
                     {"sharpe": 0.0, "turnover": 0.2}])
    monkeypatch.setattr("agent.validator_oracle._composite_ls_stats", lambda *a: next(readings))
    weights = pd.DataFrame([{"factor": "a", "weight": 1.0}])
    delta = oracle._val_composite_delta(tiny_panel, tiny_panel, [], "a", weights, oracle.base_cfg)
    assert delta["sharpe_delta"] == 0.00004
