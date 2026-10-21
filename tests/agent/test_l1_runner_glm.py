"""Exercise the runner with mocked GLM responses; no live API calls are made."""
import json
import sys

import pytest

from agent import l1_runner
from agent.llm_generator import LLMCallResult


class FakeGLM:
    provider = "zhipu"
    model = "glm-4.7-flash"
    parameters = {
        "response_format": "json_object",
        "thinking": "disabled",
        "max_tokens": 400,
        "max_retries": 0,
    }

    def __init__(self, responses):
        self.responses = list(responses)

    def generate_json(self, *, system_prompt, user_prompt):
        return LLMCallResult(
            raw_text=self.responses.pop(0), provider=self.provider, model=self.model,
            input_tokens=10, output_tokens=5, total_tokens=15, latency_ms=1,
            request_id="fake-request", parameters=self.parameters)


def test_glm_runner_separates_attempts_from_oracle_trials(
        tmp_path, tiny_panel_csv, tiny_config_path, monkeypatch):
    intent = json.dumps({
        "objective": "寻找 f1 家族中的候选",
        "allowed_families": ["f1"],
        "excluded_families": [],
        "preferences": {
            "low_redundancy": True,
            "low_turnover": False,
            "validation_stability": True,
        },
        "assumptions": [],
        "unsupported_requirements": [],
    })
    malformed = json.dumps({
        "expression": "rank(a)", "rationale": "wrong schema", "decision": "ACCEPT"})
    valid = json.dumps({"expression": "rank(a)", "rationale": "已知测试用收益信号"})
    fake = FakeGLM([intent, malformed, valid])
    monkeypatch.setattr(l1_runner, "GLMClient", lambda **kwargs: fake)
    monkeypatch.setattr(sys, "argv", [
        "l1_runner", "--query", "找一个 f1 因子", "--families", "f1",
        "--panel-csv", str(tiny_panel_csv), "--config", str(tiny_config_path),
        "--runs-root", str(tmp_path / "runs"), "--generation-attempt-budget", "2",
        "--budget", "1", "--llm-max-retries", "0",
    ])

    l1_runner.main()

    run_dirs = list((tmp_path / "runs").iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    summary = json.loads((run_dir / "run_log.json").read_text(encoding="utf-8"))
    assert summary["generation_attempts"] == 2
    assert summary["generation_failures"] == 1
    assert summary["n_trials_after"] == 1
    assert (run_dir / "cards" / "llm_0002.json").exists()
    assert (run_dir / "research_request.json").exists()
    assert (run_dir / "factor_list.json").exists()
    report = (run_dir / "user_report.md").read_text(encoding="utf-8")
    assert "## 4.1 第一层：研究请求摘要" in report
    assert "## 4.3 第三层：每个因子的名片" in report

    trace_text = (run_dir / "trace.jsonl").read_text(encoding="utf-8")
    assert '"event": "generation_failure"' in trace_text
    assert '"event": "candidate_generated"' in trace_text
    assert "GLM_API_KEY" not in trace_text
    trace = [json.loads(line) for line in trace_text.splitlines()]
    failure = next(event for event in trace if event["event"] == "generation_failure")
    assert failure["input_tokens"] == 10
    assert failure["output_tokens"] == 5
    assert failure["total_tokens"] == 15
    assert failure["latency_ms"] == 1
    assert failure["request_id"] == "fake-request"
    assert len(failure["prompt_hash"]) == 16
    assert len(failure["generator_spec_hash"]) == 16
    assert "raw_text" not in failure


def test_static_candidate_file_mode_remains_compatible(
        tmp_path, tiny_panel_csv, tiny_config_path, monkeypatch):
    candidate_path = tmp_path / "candidates.json"
    candidate_path.write_text(json.dumps([{
        "name": "manual_a", "expression": "rank(a)",
        "rationale": "manual baseline", "family": "wrong-self-reported-family",
    }]), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "l1_runner", "--candidates", str(candidate_path),
        "--panel-csv", str(tiny_panel_csv), "--config", str(tiny_config_path),
        "--runs-root", str(tmp_path / "runs-static"), "--budget", "1",
    ])

    l1_runner.main()

    run_dir = next((tmp_path / "runs-static").iterdir())
    summary = json.loads((run_dir / "run_log.json").read_text(encoding="utf-8"))
    assert summary["input_mode"] == "file"
    assert summary["n_trials_after"] == 1
    card = json.loads((run_dir / "cards" / "manual_a.json").read_text(encoding="utf-8"))
    assert card["family"] == "f1"
    assert (run_dir / "user_report.md").exists()
    factor_rows = json.loads((run_dir / "factor_list.json").read_text(encoding="utf-8"))
    card_pointer = run_dir / factor_rows[0]["card_file"]
    assert card_pointer.exists()
    assert json.loads(card_pointer.read_text(encoding="utf-8"))["name"] == "manual_a"


def test_static_duplicate_names_fail_before_run_creation(
        tmp_path, tiny_panel_csv, tiny_config_path, monkeypatch):
    candidate_path = tmp_path / "duplicate_candidates.json"
    candidate_path.write_text(json.dumps([
        {"name": "duplicate", "expression": "rank(a)"},
        # Windows card paths are case-insensitive; this must be the same identity.
        {"name": "DUPLICATE", "expression": "rank(c)"},
    ]), encoding="utf-8")
    runs_root = tmp_path / "runs-duplicates"
    monkeypatch.setattr(sys, "argv", [
        "l1_runner", "--candidates", str(candidate_path),
        "--panel-csv", str(tiny_panel_csv), "--config", str(tiny_config_path),
        "--runs-root", str(runs_root), "--budget", "2",
    ])

    with pytest.raises(SystemExit, match="duplicate candidate name"):
        l1_runner.main()

    assert not runs_root.exists()
