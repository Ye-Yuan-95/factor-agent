"""Qlib exporter round-trip and de-alignment scanner semantics."""
import json
import sys
import shutil
import subprocess
from types import ModuleType, SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmark.dealignment_check import extract_expression_literals, scan
from benchmark.qlib_export import (
    ALPHAGEN_SLOTS,
    export_alphagen_slots,
    export_qlib,
    panel_content_hash,
    read_qlib_bin,
)


def _panel(n_assets=4, n_months=6):
    rng = np.random.default_rng(0)
    dates = pd.date_range("2020-01-31", periods=n_months, freq="ME")
    rows = []
    for date in dates:
        for i in range(n_assets):
            rows.append({"date": date, "asset": f"S{i:04d}",
                         "forward_return": float(rng.standard_normal()),
                         "ep": float(rng.standard_normal()),
                         "illiq": float(rng.standard_normal())})
    return pd.DataFrame(rows)


def test_qlib_export_round_trip(tmp_path):
    panel = _panel()
    truth = pd.Series(np.arange(len(panel), dtype=float))
    manifest = export_qlib(panel, tmp_path, truth=truth, truth_definition="test truth")

    calendar = (tmp_path / "qlib_data" / "calendars" / "day.txt").read_text().split()
    instruments = (tmp_path / "qlib_data" / "instruments" / "all.txt").read_text().strip().split("\n")
    assert len(calendar) == 6 and len(instruments) == 4
    assert instruments[0].split("\t")[0] == "s0000"          # qlib lowercases symbols
    assert manifest["n_feature_files"] == 4 * 3               # 3 fields x 4 instruments

    start, values = read_qlib_bin(
        tmp_path / "qlib_data" / "features" / "s0000" / "ep.day.bin", len(calendar))
    expected = panel[panel["asset"] == "S0000"].sort_values("date")["ep"].to_numpy()
    assert start == 0
    assert np.allclose(values, expected, atol=1e-6)


def test_truth_never_enters_the_dataset(tmp_path):
    panel = _panel()
    truth = pd.Series(np.arange(len(panel), dtype=float))
    export_qlib(panel, tmp_path, truth=truth, truth_definition="t")
    feature_files = {p.name for p in (tmp_path / "qlib_data" / "features" / "s0000").iterdir()}
    assert not any("truth" in name for name in feature_files)
    assert (tmp_path / "truth" / "truth_signal.csv").exists()
    # and a panel that smuggles the truth in as a column is refused outright
    smuggled = panel.copy()
    smuggled["truth_signal"] = truth.to_numpy()
    with pytest.raises(ValueError, match="sidecar-only"):
        export_qlib(smuggled, tmp_path / "other", truth=truth)


def test_export_manifest_pins_content(tmp_path):
    panel = _panel()
    manifest = export_qlib(panel, tmp_path, truth=None)
    assert manifest["panel_content_hash"] == panel_content_hash(panel)
    stored = json.loads((tmp_path / "export_manifest.json").read_text(encoding="utf-8"))
    assert stored["truth_sidecar"] is None
    assert stored["label_field"] == "forward_return"


def test_exporter_refuses_gapped_instruments(tmp_path):
    panel = _panel()
    gapped = panel.drop(panel.index[5])          # punch a hole in one instrument's history
    with pytest.raises(ValueError, match="not contiguous"):
        export_qlib(gapped, tmp_path)


def test_truth_column_is_refused_even_without_a_truth_series(tmp_path):
    """Truth-column refusal applies even when no truth sidecar is supplied."""
    panel = _panel()
    panel["truth_signal"] = np.arange(len(panel), dtype=float)
    with pytest.raises(ValueError, match="sidecar-only"):
        export_qlib(panel, tmp_path)                       # no truth argument at all
    with pytest.raises(ValueError, match="sidecar-only"):
        export_qlib(panel, tmp_path, truth=None)
    panel2 = _panel()
    panel2["truth_signal_lag1"] = 0.0                      # prefixed variants too
    with pytest.raises(ValueError, match="sidecar-only"):
        export_qlib(panel2, tmp_path)


def test_duplicate_dates_are_refused(tmp_path):
    """A duplicate date plus a gap keeps the row count intact and would slip past a pure
    length comparison, misaligning every value against the calendar."""
    panel = _panel()
    rows = panel[panel["asset"] == "S0000"].sort_values("date")
    duplicated = pd.concat([panel.drop(rows.index[-1]), rows.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate dates"):
        export_qlib(duplicated, tmp_path)


def test_manifest_carries_a_file_inventory_with_digests(tmp_path):
    panel = _panel()
    truth = pd.Series(np.arange(len(panel), dtype=float))
    manifest = export_qlib(panel, tmp_path, truth=truth, truth_definition="t")
    inventory = manifest["file_inventory"]
    assert "qlib_data/calendars/day.txt" in inventory
    assert "qlib_data/features/s0000/ep.day.bin" in inventory
    assert all(len(digest) == 64 for digest in inventory.values())   # full sha256
    assert manifest["truth_sidecar_sha256"] and manifest["file_inventory_hash"]
    # the inventory must describe what is actually on disk
    for rel, digest in inventory.items():
        assert (tmp_path / rel).exists()
        import hashlib
        assert hashlib.sha256((tmp_path / rel).read_bytes()).hexdigest() == digest


def test_truth_length_mismatch_is_refused(tmp_path):
    panel = _panel()
    with pytest.raises(ValueError, match="does not match panel"):
        export_qlib(panel, tmp_path, truth=pd.Series([0.0, 1.0]))


def test_alphagen_slot_remap_and_truth_reachability(tmp_path):
    panel = _panel()
    panel["bm"] = 1.0
    panel["sp"] = 2.0
    panel["turn"] = 3.0
    panel["dolvol"] = 4.0
    manifest = export_alphagen_slots(
        panel, tmp_path, ["ep", "illiq", "bm", "sp", "turn", "dolvol"],
        require_columns=["ep", "illiq"])
    assert manifest["alphagen_slot_mapping"]["open"] == "ep"
    assert manifest["fields"] == ["forward_return", *ALPHAGEN_SLOTS]
    start, values = read_qlib_bin(tmp_path / "qlib_data" / "features" / "s0000" / "open.day.bin")
    expected = panel[panel["asset"] == "S0000"].sort_values("date")["ep"].to_numpy()
    assert np.allclose(values, expected, atol=1e-6)        # slot carries the mapped column
    # truth unreachable -> refused, otherwise the arm would measure nothing
    with pytest.raises(ValueError, match="unreachable"):
        export_alphagen_slots(panel, tmp_path / "bad",
                              ["bm", "sp", "turn", "dolvol", "forward_return", "ep"],
                              require_columns=["ep", "illiq"])
    with pytest.raises(ValueError, match="exactly"):
        export_alphagen_slots(panel, tmp_path / "bad2", ["ep", "illiq"])


def test_expression_literal_extraction():
    literals = extract_expression_literals("example: zscore(mom1m) * sign(delta(turn, 3))")
    assert literals and "mom1m" in literals[0]
    assert extract_expression_literals('"allowed_columns": ["ep", "illiq"]') == []


def test_dealignment_verdicts(tmp_path):
    # vocabulary mention only -> CLEAN (the search space may contain the answer)
    (tmp_path / "vocab.py").write_text('COLUMNS = ["ep", "illiq", "mom1m"]', encoding="utf-8")
    assert scan(tmp_path, ["ep", "illiq"])["verdict"] == "CLEAN"
    # one truth column inside an example expression -> PARTIAL
    (tmp_path / "half.py").write_text('EXAMPLE = "zscore(ep) * sign(turn)"', encoding="utf-8")
    assert scan(tmp_path, ["ep", "illiq"])["verdict"] == "PARTIAL"
    # Both truth columns in one expression produce a LEAK verdict.
    (tmp_path / "leak.py").write_text('EXAMPLE = "zscore(ep) * sign(delta(illiq, 3))"',
                                      encoding="utf-8")
    report = scan(tmp_path, ["ep", "illiq"])
    assert report["verdict"] == "LEAK"
    assert report["full_pair_expressions"][0]["file"] == "leak.py"
    assert report["report_hash"]


def test_dealignment_distinguishes_aligned_and_unrelated_truth():
    """Prompt expressions must flag aligned truth columns and clear unrelated column pairs."""
    agent_dir = Path(__file__).resolve().parents[2] / "agent"
    prompts = ["generator_contract.py", "llm_generator.py"]
    assert scan(agent_dir, ["mom1m", "turn"], include=prompts)["verdict"] == "LEAK"
    assert scan(agent_dir, ["ep", "illiq"], include=prompts)["verdict"] == "CLEAN"


def test_include_globs_scope_the_scan(tmp_path):
    (tmp_path / "prompt.py").write_text('E = "zscore(ep) * sign(illiq)"', encoding="utf-8")
    (tmp_path / "harness.py").write_text('E = "zscore(ep) * sign(illiq)"', encoding="utf-8")
    assert scan(tmp_path, ["ep", "illiq"], include=["prompt.py"])["n_text_files_scanned"] == 1
    assert scan(tmp_path, ["ep", "illiq"])["n_text_files_scanned"] == 2


def test_reexport_refuses_to_mix_or_overwrite_existing_files(tmp_path):
    export_qlib(_panel(), tmp_path)
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="not empty"):
        export_qlib(_panel(n_assets=2), tmp_path)
    assert before == {p.relative_to(tmp_path): p.read_bytes()
                      for p in tmp_path.rglob("*") if p.is_file()}


@pytest.mark.parametrize("kind,ic,truth_columns", [
    ("linear", 0.06, ["ep"]), ("mechanism", 0.0, []), ("linear", 0.0, [])])
def test_native_verifier_uses_recorded_generation_inputs(
        tmp_path, monkeypatch, kind, ic, truth_columns):
    """Check verifier plumbing with mocked Qlib reads, including null and custom truth."""
    import benchmark.qlib_smoke_report as smoke

    panel = _panel()
    export_qlib(panel, tmp_path, extra_meta={
        "dgp": kind, "seed": 11, "inject_ic": ic, "mech_kind": "conditional",
        "x_col": "mom1m", "y_col": "turn", "true_factor": "ep",
        "truth_columns": truth_columns})
    observed_calls = []

    def rebuild(*args, **kwargs):
        observed_calls.append((args, kwargs))
        return panel, None, "fixture"

    def read_features(instruments, fields, **kwargs):
        expected = panel[panel.asset == "S0000"].sort_values("date")
        data = {}
        for field in fields:
            _, values = read_qlib_bin(tmp_path / "qlib_data/features/s0000" / f"{field[1:]}.day.bin")
            data[field] = values
        return pd.DataFrame(data, index=expected.date)

    qlib = ModuleType("qlib")
    qlib.__version__ = "test-double"
    qlib.init = lambda **kwargs: None
    config = ModuleType("qlib.config")
    config.REG_CN = "fixture"
    data = ModuleType("qlib.data")
    data.D = SimpleNamespace(features=read_features)
    for name, module in [("qlib", qlib), ("qlib.config", config), ("qlib.data", data)]:
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(smoke, "build_panel", rebuild)
    report = smoke.verify_export(tmp_path, config_path=tmp_path / "config.json",
                                 calibration_csv=tmp_path / "calibration.csv")
    assert report["status"] == "PASS"
    args, kwargs = observed_calls[0]
    assert args == (kind, 11)
    assert kwargs["true_factor"] == "ep"
    assert kwargs["x_col"] == "mom1m" and kwargs["y_col"] == "turn"
    assert kwargs["ic"] == ic


def test_cli_retains_generation_inputs_for_null_exports(tmp_path, monkeypatch):
    import benchmark.qlib_export as exporter

    monkeypatch.setattr(exporter, "build_panel", lambda *args, **kwargs: (_panel(), None, "null"))
    monkeypatch.setattr(sys, "argv", ["export", "--dgp", "mechanism", "--seed", "11",
                                     "--out", str(tmp_path), "--ic", "0", "--x-col", "mom1m",
                                     "--y-col", "turn", "--true-factor", "ep"])
    exporter.main()
    manifest = json.loads((tmp_path / "export_manifest.json").read_text())
    assert manifest["truth_columns"] == []
    assert manifest["x_col"] == "mom1m" and manifest["y_col"] == "turn"
    assert manifest["true_factor"] == "ep"


@pytest.mark.skipif(shutil.which("git") is None, reason="Git provenance requires git")
def test_export_does_not_attribute_an_enclosing_repository(tmp_path):
    from benchmark.qlib_export import _git_commit

    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "commit", "--allow-empty", "--quiet", "-m", "fixture"],
                   cwd=tmp_path, check=True)
    unversioned = tmp_path / "release-copy"
    unversioned.mkdir()
    assert _git_commit(unversioned) == "unknown"
    assert len(_git_commit(tmp_path)) == 40
