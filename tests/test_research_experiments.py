"""Check research semantics and runnable entry points without private market data.

The fixture has correlated exposures so calibration can observe false proxies.
Small runs validate wiring and artifacts, never historical FPR/Power claims.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from synthetic import research_runtime
from synthetic.incremental_mechanism import fm_tstats
from synthetic.semi_synthetic import make_semi_synthetic

ROOT = Path(__file__).resolve().parents[1]
MODULES = (
    "p1_validate",
    "semi_synthetic",
    "incremental_robustness",
    "dgp_sensitivity",
    "omitted_factor_experiment",
    "tau_size_sweep",
    "split_sample_incremental",
    "calibrated_tau",
    "tau_transfer",
    "incremental_mechanism",
    "semi_mechanism_diagnostic",
    "four_gate_ablation",
    "stage_a_fpr_power",
)


@pytest.fixture(scope="module")
def research_inputs(tmp_path_factory):
    """Provide twelve correlated exposures in four families with direct noise labels."""
    folder = tmp_path_factory.mktemp("research_inputs")
    rng = np.random.default_rng(731)
    factors = [f"f{i}" for i in range(12)]
    frame = pd.MultiIndex.from_product(
        [pd.date_range("2000-01-31", periods=30, freq="ME"), range(60)],
        names=["date", "asset"],
    ).to_frame(index=False)
    common = rng.normal(size=len(frame))
    for factor in factors:
        frame[factor] = 0.8 * common + 0.6 * rng.normal(size=len(frame))
    frame["forward_return"] = 0.05 * rng.standard_t(6, size=len(frame))
    panel = folder / "panel.csv"
    frame.to_csv(panel, index=False)
    families = {
        f: ("momentum" if i < 4 else f"family{(i - 4) // 3}")
        for i, f in enumerate(factors)
    }
    config = folder / "config.json"
    config.write_text(
        json.dumps(
            {
                "input_csv": str(panel),
                "output_dir": str(folder / "selection"),
                "factor_cols": factors,
                "family_constraints": {"factor_families": families},
                "redundancy": {"max_abs_corr": 0.85},
            }
        ),
        encoding="utf-8",
    )
    return panel, config, frame, factors


def test_semi_null_preserves_exposures_and_return_multisets(research_inputs):
    panel, config, frame, factors = research_inputs
    generated, truth = make_semi_synthetic(
        frame, factors, [], 0, np.random.default_rng(5)
    )
    pd.testing.assert_frame_equal(generated[factors], frame[factors])
    assert not truth.any()
    for date, group in frame.groupby("date"):
        np.testing.assert_array_equal(
            np.sort(group.forward_return),
            np.sort(generated.loc[generated.date == date, "forward_return"]),
        )


def test_fm_insufficient_cross_sections_is_not_finite_evidence(research_inputs):
    _, _, frame, factors = research_inputs
    short = frame[frame.date.isin(frame.date.unique()[:4])]
    assert np.isnan(fm_tstats(short, factors, [0])[0])


def test_calibration_seed_overlap_is_rejected_before_writing(tmp_path, research_inputs):
    panel, config, _, _ = research_inputs
    args = argparse.Namespace(
        panel_csv=str(panel),
        config=str(config),
        output_dir=str(tmp_path / "not_created"),
        seed_cal=7,
        reps_cal=10,
        seed_val=16,
        reps_val=10,
    )
    with pytest.raises(ValueError, match="disjoint"):
        research_runtime.prepare_run(args, __file__)
    assert not Path(args.output_dir).exists()


def test_run_refuses_existing_evidence(tmp_path, research_inputs):
    panel, config, _, _ = research_inputs
    marker = tmp_path / "result.txt"
    marker.write_text("keep", encoding="utf-8")
    args = argparse.Namespace(
        panel_csv=str(panel), config=str(config), output_dir=str(tmp_path)
    )
    with pytest.raises(ValueError, match="not empty"):
        research_runtime.prepare_run(args, __file__)
    assert marker.read_text() == "keep"


def test_provenance_detects_input_mutation(tmp_path, research_inputs):
    panel, config, _, _ = research_inputs
    local = tmp_path / "config.json"
    local.write_bytes(config.read_bytes())
    out = tmp_path / "run"
    args = argparse.Namespace(
        panel_csv=str(panel), config=str(local), output_dir=str(out)
    )
    research_runtime.prepare_run(args, __file__)
    local.write_text(local.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        research_runtime.finish_run("COMPLETED")
    assert json.loads((out / "manifest.json").read_text())["status"] == "FAILED"


def test_unversioned_source_does_not_read_parent_git_status(tmp_path, research_inputs, monkeypatch):
    """Missing source Git identity also makes worktree status unavailable."""
    panel, config, _, _ = research_inputs
    monkeypatch.setattr("factor_research.experiment_manifest.git_commit", lambda: None)

    def unexpected_status(*args, **kwargs):
        pytest.fail("An unversioned source copy must not inspect an enclosing repository")

    monkeypatch.setattr(research_runtime.subprocess, "run", unexpected_status)
    out = tmp_path / "run"
    args = argparse.Namespace(panel_csv=str(panel), config=str(config), output_dir=str(out))
    research_runtime.prepare_run(args, __file__)
    research_runtime.finish_run("COMPLETED")
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["git_commit"] is None and manifest["git_dirty"] is None
    assert manifest["status"] == "COMPLETED" and manifest["inputs_unchanged"]


@pytest.mark.parametrize("module", MODULES)
def test_research_cli_smoke(module, tmp_path, research_inputs):
    panel, config, _, _ = research_inputs
    out = tmp_path / "run"
    # Use direct-script mode from a foreign cwd to catch accidental reliance on
    # the private research checkout. Module-mode help is checked separately.
    command = [
        sys.executable,
        str(ROOT / "synthetic" / f"{module}.py"),
        "--panel-csv",
        str(panel),
        "--config",
        str(config),
        "--output-dir",
        str(out),
    ]
    if module != "semi_mechanism_diagnostic":
        command += ["--q-max", "0.99"]
        if module != "stage_a_fpr_power":
            command += ["--kappa", "0.01", "--survival-B", "2"]
        command += (
            ["--reps-cal", "2", "--reps-val", "2"]
            if module in {"calibrated_tau", "tau_transfer"}
            else ["--reps", "1"]
        )
    if module == "stage_a_fpr_power":
        command += ["--pipeline-reps", "1"]
    if module not in {"semi_synthetic", "dgp_sensitivity"}:
        command += ["--n-assets", "50", "--n-months", "30"]
    proc = subprocess.run(
        command,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "COMPLETED"
    assert manifest["inputs_unchanged"] is True
    assert manifest["factor_order"] == research_inputs[3]
    assert manifest["output_sha256"]
    assert "factor_research/factor_validity_pipeline.py" in manifest["source_sha256"]
    assert "factor_research/decision_focused.py" in manifest["source_sha256"]
    for path in out.glob("*.json"):
        # Python otherwise silently accepts the nonstandard NaN/Infinity tokens.
        json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: pytest.fail(f"non-JSON token: {token}"),
        )
    if module in {"tau_size_sweep", "split_sample_incremental", "calibrated_tau", "tau_transfer"}:
        accounting = json.loads((out / "sample_accounting.json").read_text(encoding="utf-8"))
        assert accounting["schema_version"] == 2
        for population in accounting["populations"].values():
            assert population["samples"]
            for sample in population["samples"]:
                assert sample["n_periods_used"] == len(sample["cross_section_sizes"])
                assert sample["n_complete_rows_used"] == sum(sample["cross_section_sizes"])
        output_csv = next(out.glob("*.csv"))
        result = pd.read_csv(output_csv)
        modes = ["_same150", "_same75", "_split"] if module == "split_sample_incremental" else [""]
        for suffix in modes:
            assert (result["n_true_total" + suffix] > 0).all()
            np.testing.assert_allclose(result["full_power" + suffix],
                                       result["n_true_selected" + suffix] / result["n_true_total" + suffix])
            assert (result["n_true_reaching" + suffix] == result["n_true_finite" + suffix] + result["n_true_invalid" + suffix]).all()
    help_run = subprocess.run(
        [sys.executable, "-m", f"synthetic.{module}", "--help"],
        cwd=ROOT,
        capture_output=True,
        timeout=30,
    )
    assert help_run.returncode == 0


def test_semi_diagnostic_copy_on_write(tmp_path, research_inputs, monkeypatch):
    """Exercise pandas 3's read-only views, also on pandas 2 with CoW enabled."""
    # The CLI runs in a child process, so use pandas' startup environment option.
    monkeypatch.setenv("PANDAS_COPY_ON_WRITE", "1")
    test_research_cli_smoke("semi_mechanism_diagnostic", tmp_path, research_inputs)
    report = json.loads(
        (tmp_path / "run" / "semi_mechanism_diagnostic.json").read_text(
            encoding="utf-8"
        )
    )
    _, _, frame, factors = research_inputs
    corr = frame[factors].corr().to_numpy()
    offdiag = ~np.eye(len(factors), dtype=bool)
    # Copying must preserve the reported off-diagonal statistic.
    assert report["mean_abs_corr_real"] == pytest.approx(
        float(np.nanmean(np.abs(corr[offdiag])))
    )


def test_split_no_survivors_is_reported_as_missing(
    tmp_path, research_inputs, monkeypatch
):
    from synthetic import split_sample_incremental as split

    panel, config, _, _ = research_inputs
    out = tmp_path / "empty_gate"
    # Force the boundary deterministically: even a tiny positive q can admit
    # a p-value rounded to zero, so extreme thresholds are not an empty-set fixture.
    monkeypatch.setattr(split, "screen", lambda *args: [])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "split_sample_incremental",
            "--panel-csv",
            str(panel),
            "--config",
            str(config),
            "--output-dir",
            str(out),
            "--reps",
            "1",
            "--n-assets",
            "30",
            "--n-months",
            "10",
            "--survival-B",
            "2",
        ],
    )
    research_runtime.run_experiment(split.main)
    result = pd.read_csv(out / "split_sample.csv")
    assert result["size_split"].isna().all()


def test_full_power_includes_true_factors_rejected_at_screen(
    tmp_path, research_inputs, monkeypatch
):
    from synthetic import tau_size_sweep as sweep

    panel, config, _, factors = research_inputs
    out = tmp_path / "early_rejection"
    # All marginal p-values fail. Full-pipeline recall must be zero, whereas
    # conditional recall has no denominator and must remain missing.
    monkeypatch.setattr(
        sweep,
        "per_month_ic",
        lambda *args: (
            np.zeros((10, len(factors))),
            np.zeros(len(factors)),
            np.ones(len(factors)),
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tau_size_sweep",
            "--panel-csv",
            str(panel),
            "--config",
            str(config),
            "--output-dir",
            str(out),
            "--reps",
            "1",
            "--n-assets",
            "30",
            "--n-months",
            "10",
            "--survival-B",
            "2",
        ],
    )
    research_runtime.run_experiment(sweep.main)
    result = pd.read_csv(out / "tau_size_sweep.csv")
    assert (result["power"] == 0).all()
    assert result["conditional_power"].isna().all()


def test_empty_calibration_records_failure_instead_of_a_cutoff(
    tmp_path, research_inputs, monkeypatch
):
    from synthetic import calibrated_tau as calibration

    panel, config, _, _ = research_inputs
    out = tmp_path / "no_calibration_evidence"
    monkeypatch.setattr(
        calibration, "collect_split_tstats", lambda *args: (np.array([]), np.array([]))
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "calibrated_tau",
            "--panel-csv",
            str(panel),
            "--config",
            str(config),
            "--output-dir",
            str(out),
        ],
    )
    with pytest.raises(SystemExit, match="cannot estimate tau"):
        research_runtime.run_experiment(calibration.main)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "FAILED"
    assert not (out / "calibrated_tau.json").exists()
