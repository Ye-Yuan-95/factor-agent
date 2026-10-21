"""Reject missing evidence and verify reporting denominators and source identity.

Boundary fixtures distinguish input dates from usable fits, and all injected
factors from estimable survivors. FDP checks use integer counts and empty runs.
"""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

from factor_research.factor_validity_pipeline import load_config, incremental_value_filter
from synthetic.four_gate_ablation import gate_incremental
from synthetic.incremental_mechanism import FMStats, fm_tstats
from synthetic.research_metrics import ThresholdPopulation, ic_sample_accounting
from agent.realized_fdr import summarize_fdp

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("r2,expected_true", [(0.2496, 0), (0.25, 1)])
def test_closed_loop_truth_uses_unrounded_r2(tmp_path, monkeypatch, r2, expected_true):
    """Exercise one accepted trial with a known truth-boundary value, not a fixed result."""
    import agent.realized_fdr as fdr

    panel = _panel(6)

    class AcceptedOracle:
        def __init__(self, *args):
            self._panel = None

        def set_panel(self, frame, universe, **kwargs):
            self._panel = frame
            self.universe = universe

        def validate(self, **kwargs):
            kwargs["memory"].record_trial(
                name="candidate", expression="a", signature="one", family="f1",
                universe=self.universe, sharpe=0.1, mean_ic=None, monthly_ls=[],
                decision="ACCEPT", reject_reason=None)
            return SimpleNamespace(decision="ACCEPT", reject_reason=None)

    class SingleCandidate:
        def __init__(self, *args, **kwargs):
            pass

        def propose(self, history):
            return {"name": "candidate", "expression": "a", "family": "f1", "rationale": "test"}

    monkeypatch.setattr(fdr, "ValidatorOracle", AcceptedOracle)
    monkeypatch.setattr(fdr, "BaselineGenerator", SingleCandidate)
    monkeypatch.setattr(fdr, "generate_panel", lambda *a, **k: (panel, None))
    monkeypatch.setattr(fdr, "_truth_r2", lambda *a: r2)
    result = fdr.run_one(run_id="boundary", gen_kind="gen1", calib={"factors": ["a"]},
                         config_path="unused.json", seed=1, inject_ic=0.06, budget=1,
                         generator_mode="random", fam_map={"a": "f1"}, truth_corr=0.5,
                         n_assets=20, n_months=6, out_dir=tmp_path)
    assert result["n_accept_true"] == expected_true
    assert result["n_accept_false"] == 1 - expected_true
    history = json.loads((tmp_path / "runs/boundary/history.json").read_text())
    assert history["history"][0]["truth_r2"] == r2


def test_fdp_refuses_existing_evidence_before_generating(tmp_path, monkeypatch):
    import agent.realized_fdr as fdr

    history = tmp_path / "runs/existing/history.json"
    history.parent.mkdir(parents=True)
    history.write_text('{"preserve": true}', encoding="utf-8")
    original = history.read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        fdr.run_one(run_id="existing", gen_kind="gen1", calib={}, config_path="unused.json",
                    seed=1, inject_ic=0.06, budget=1, generator_mode="random", fam_map={},
                    truth_corr=0.5, n_assets=20, n_months=6, out_dir=tmp_path)
    monkeypatch.setattr(fdr.sys, "argv", ["fdp", "--output-dir", str(tmp_path)])
    with pytest.raises(SystemExit) as error:
        fdr.main()
    assert error.value.code == 2 and history.read_bytes() == original


def _panel(n_dates):
    rng = np.random.default_rng(97)
    frame = pd.MultiIndex.from_product(
        [pd.date_range("2000-01-31", periods=n_dates, freq="ME"), range(20)],
        names=["date", "asset"],
    ).to_frame(index=False)
    frame["a"] = rng.normal(size=len(frame))
    frame["forward_return"] = 0.05 * frame["a"] + rng.normal(0, 0.001, len(frame))
    return frame


@pytest.mark.parametrize("n_dates", [0, 4, 5])
def test_incremental_boundary_refuses_missing_evidence(n_dates):
    frame = _panel(n_dates)
    cfg = replace(load_config(ROOT / "a_share_baseline_config.json"), factor_cols=["a"])
    row = incremental_value_filter(frame, ["a"], cfg).iloc[0]
    kept = gate_incremental(frame, [0], ["a"], 2)
    stats = fm_tstats(frame, ["a"], [0])
    assert row.fm_n_periods == n_dates
    assert row.fm_n_complete_rows == 20 * n_dates
    assert stats.diagnostics["n_periods_used"] == n_dates
    if n_dates < 5:
        assert not row.passes_incremental and np.isnan(row.fm_tstat)
        assert row.fm_status == "insufficient_cross_sections"
        assert kept == [] and np.isnan(stats[0])
    else:
        assert row.passes_incremental and row.fm_status == "passed"
        assert kept == [0] and np.isfinite(stats[0])


def test_raw_dates_do_not_replace_complete_case_fits():
    frame = _panel(6)
    frame.loc[frame.date.isin(frame.date.unique()[-2:]), "forward_return"] = np.nan
    cfg = replace(load_config(ROOT / "a_share_baseline_config.json"), factor_cols=["a"])
    row = incremental_value_filter(frame, ["a"], cfg).iloc[0]
    assert row.fm_n_date_groups == 6 and row.fm_n_periods == 4
    assert row.fm_skipped_insufficient_obs == 2
    assert row.fm_n_complete_rows == 80 and not row.passes_incremental


def test_nonfinite_core_statistic_never_passes(monkeypatch):
    monkeypatch.setattr("factor_research.factor_validity_pipeline.t_stat", lambda *a, **k: np.nan)
    cfg = replace(load_config(ROOT / "a_share_baseline_config.json"), factor_cols=["a"])
    row = incremental_value_filter(_panel(6), ["a"], cfg).iloc[0]
    assert not row.passes_incremental and row.fm_status == "nonfinite_statistic"


def test_threshold_populations_include_screened_out_and_invalid_factors():
    p = ThresholdPopulation()
    p.add([True, True, False, False], [0, 2, 3], FMStats({0: 3, 2: np.nan, 3: 3}, {}))
    r = p.at(2)
    assert r["full_power"] == 0.5 and r["conditional_power"] == 1
    assert r["full_fpr"] == 0.5 and r["conditional_size"] == 1
    assert r["n_true_screened_out"] == 1 and r["n_fake_invalid"] == 1
    assert r["n_true_total"] == 2 and r["n_fake_total"] == 2


def test_empty_gate_is_zero_full_power_but_missing_conditional_power():
    p = ThresholdPopulation()
    p.add([True, False], [], FMStats({}, {"status": "no_survivors"}))
    r = p.at(2)
    assert r["full_power"] == 0 and r["full_fpr"] == 0
    assert np.isnan(r["conditional_power"]) and np.isnan(r["conditional_size"])
    assert r["n_true_screened_out"] == 1 and r["n_fake_screened_out"] == 1


def test_fdp_keeps_zero_discovery_runs_and_names_both_means():
    rows = [
        {"inject_ic": 0.06, "n_accept": 2, "n_accept_false": 1},
        {"inject_ic": 0.06, "n_accept": 0, "n_accept_false": 0},
    ]
    r = summarize_fdp(rows, 0.25)
    assert r["pooled_empirical_fdp"] == 0.5
    assert r["mean_run_fdp"] == 0.25
    assert r["mean_run_fdp_given_acceptance"] == 0.5
    assert r["all_run_fdp_denominator"] == 2
    assert r["conditional_run_fdp_denominator"] == 1
    assert r["n_zero_accept_alt_runs"] == 1


def test_empty_discovery_pool_does_not_certify_error_control():
    r = summarize_fdp([{"inject_ic": 0.06, "n_accept": 0, "n_accept_false": 0}], 0.25)
    assert r["pooled_empirical_fdp"] is None and not r["point_rule_passed"]
    assert r["mean_run_fdp"] == 0 and r["mean_run_fdp_given_acceptance"] is None


def test_point_rule_uses_unrounded_ratio():
    r = summarize_fdp(
        [{"inject_ic": 0.06, "n_accept": 100000, "n_accept_false": 25001}], 0.25
    )
    assert round(r["pooled_empirical_fdp"], 4) == 0.25
    assert not r["point_rule_passed"]


def test_ic_sample_report_uses_finite_counts_per_factor():
    panel = _panel(5)
    ic = np.array([[1, np.nan], [2, 1], [np.nan, 2], [np.nan, 3], [np.nan, np.nan]])
    report = ic_sample_accounting(panel, ["a", "b"], ic)
    assert report["raw_date_groups"] == 5
    assert report["usable_ic_dates"] == {"a": 2, "b": 3}
    assert report["counts_are_not_independent_sample_sizes"]


@pytest.mark.parametrize("layout", ["checkout", "nested-copy", "worktree"])
def test_git_identity_belongs_to_the_source_root(layout, tmp_path, monkeypatch):
    """A nested archive must not inherit a parent's commit; real worktrees retain identity."""
    if not shutil.which("git"):
        pytest.skip("Git is needed to create versioned provenance fixtures")
    parent = tmp_path / "parent"
    parent.mkdir()
    subprocess.run(["git", "init", "-q", str(parent)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(parent), "-c", "user.name=Fixture",
                    "-c", "user.email=fixture@example.invalid", "-c", "commit.gpgsign=false",
                    "commit", "--allow-empty", "-qm", "Fixture"], check=True, capture_output=True)
    head = subprocess.check_output(["git", "-C", str(parent), "rev-parse", "HEAD"], text=True).strip()
    if layout == "checkout":
        source = parent
    elif layout == "nested-copy":
        source = parent / "release"
        source.mkdir()
    else:
        source = tmp_path / "worktree"
        subprocess.run(["git", "-C", str(parent), "worktree", "add", "--detach", str(source)],
                       check=True, capture_output=True)

    from factor_research import experiment_manifest
    from agent import b2_compare, factorial_ablation, validator_oracle

    for module in (b2_compare, factorial_ablation, validator_oracle):
        monkeypatch.setattr(module, "_PROJECT_DIR", source)
    if layout == "nested-copy":
        assert experiment_manifest.git_commit(source) is None
        assert validator_oracle._git_commit() is None
        for module in (b2_compare, factorial_ablation):
            state = module._git_state()
            assert state["commit"] == "unknown" and state["porcelain_sha256"] is None
    else:
        assert experiment_manifest.git_commit(source) == head
        assert head.startswith(validator_oracle._git_commit())
        for module in (b2_compare, factorial_ablation):
            assert head.startswith(module._git_state()["commit"])
