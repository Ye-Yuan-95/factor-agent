"""F4b artifact contracts using fabricated measurements, without historical results."""
import json
import shutil
from pathlib import Path

import pytest

from benchmark.f4b_analysis import _canonical_hash, analyse
from benchmark.f4b_artifact_audit import audit


@pytest.fixture
def artifact_set(tmp_path):
    """Create a minimal paired artifact set with no launcher-specific worker logs."""
    root = Path(__file__).resolve().parents[2]
    protocol = json.loads((root / "experiments/hybrid_form_aware_2026-07-31_F4b.json")
                          .read_text(encoding="utf-8"))
    protocol["dgp"].update(alternative_seeds=[2300], null_seeds=[2400])
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    out = tmp_path / "output"
    out.mkdir()
    manifest = {"protocol_hash": _canonical_hash(protocol),
                "source_fingerprint": "c" * 16, "git": {"commit": "a" * 40},
                "frozen_inputs": {"commit": "a" * 40, "protocol_blob": "b" * 40,
                                  "relevant_paths_clean": True}}
    (out / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for scenario, seed in [("alternative", 2300), ("null", 2400)]:
        for arm in protocol["baseline_arms"]:
            candidates = [{"truth_loading": 0.3 if scenario == "alternative" else None,
                           "score_decision": "REJECT", "is_true_candidate": False}
                          for _ in range(36)]
            result = {"protocol_id": protocol["protocol_id"], "scenario": scenario,
                      "seed": seed, "arm_id": arm["arm_id"], "arm_kind": arm["generator"],
                      "oracle_trials": 36, "oracle_trial_budget": 36,
                      "generation_attempts": 36, "generation_attempt_budget": 72,
                      "response_local_failures": 0, "runtime_coverage_failures": 0,
                      "panel_content_hash": f"paired-{scenario}", "valid_candidate_rate": 1.0,
                      "n_would_be_accept": 0, "generator_diagnostics": {"n_pending_feedback": 0},
                      "candidates": candidates}
            path = out / "runs" / f"{scenario}_s{seed}" / arm["arm_id"] / "result.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(result), encoding="utf-8")
    return protocol_path, out


def test_sequential_artifacts_need_no_worker_logs(artifact_set):
    protocol, out = artifact_set
    assert audit(protocol, out)["worker_logs"] == {"available": 0, "nonempty": []}
    report = analyse(protocol, out)
    assert report["decision"]["status"] == "NOT_SUPPORTED"
    assert str(protocol.resolve()) in report["input_sha256"]


def test_optional_worker_warnings_are_diagnostics(artifact_set):
    protocol, out = artifact_set
    logs = out / "worker_logs"
    logs.mkdir()
    (logs / "one.err.log").write_text("A warning, not a missing result", encoding="utf-8")
    report = audit(protocol, out)
    assert report["status"] == "PASS"
    assert report["worker_logs"]["nonempty"] == ["worker_logs/one.err.log"]


def test_budgeted_proposal_failures_are_accounted_not_forbidden(artifact_set):
    protocol, out = artifact_set
    path = next((out / "runs").glob("*/*/result.json"))
    result = json.loads(path.read_text())
    result.update(generation_attempts=37, response_local_failures=1,
                  valid_candidate_rate=round(36 / 37, 6))
    path.write_text(json.dumps(result), encoding="utf-8")
    assert audit(protocol, out)["checks"]["generation_attempts_accounted"]


@pytest.mark.parametrize("corruption", ["protocol", "duplicate", "panel", "candidate_count"])
def test_analysis_refuses_inconsistent_artifacts(artifact_set, corruption):
    protocol, out = artifact_set
    result_path = next((out / "runs").glob("*/*/result.json"))
    if corruption == "protocol":
        data = json.loads(protocol.read_text())
        data["objective"] = "Changed protocol under the same ID"
        protocol.write_text(json.dumps(data), encoding="utf-8")
    elif corruption == "duplicate":
        duplicate = out / "runs" / "duplicate" / "arm" / "result.json"
        duplicate.parent.mkdir(parents=True)
        shutil.copy2(result_path, duplicate)
    else:
        data = json.loads(result_path.read_text())
        if corruption == "panel":
            data["panel_content_hash"] = "different-panel"
        else:
            data["candidates"].pop()
        result_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(RuntimeError, match="artifact audit failed"):
        analyse(protocol, out)
