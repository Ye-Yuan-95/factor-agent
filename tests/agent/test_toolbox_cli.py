"""The toolbox CLI must expose what DATA_CONTRACT prescribes: a formal
benchmark-relative run goes through --require-full-test-benchmark, and the recommended
entry point is `python -m agent.tools walkforward_report`, not the underlying script."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_walkforward_report_help_exposes_the_formal_coverage_flag():
    result = subprocess.run(
        [sys.executable, "-m", "agent.tools", "walkforward_report", "--help"],
        capture_output=True, text=True, cwd=ROOT, timeout=120)
    assert result.returncode == 0, result.stderr[-500:]
    assert "--require-full-test-benchmark" in result.stdout, (
        "the toolbox CLI does not expose the formal coverage flag DATA_CONTRACT "
        "prescribes")


def test_walkforward_report_passes_the_flag_bare_not_with_a_value(monkeypatch, tmp_path):
    """store_true flags take no value on the child CLI: '--require-full-test-benchmark
    True' would be rejected as an unrecognized argument."""
    sys.path.insert(0, str(ROOT))
    import agent.tools as tools

    captured = []

    def fake_run(cmd):
        captured.append(cmd)
        return {"ok": True, "cmd": cmd}

    monkeypatch.setattr(tools, "_run_subprocess", fake_run)
    panel = tmp_path / "panel.csv"
    panel.write_text("date,asset,forward_return\n", encoding="utf-8")

    tools.walkforward_report(panel_csv=str(panel), confirm=tools.LOCKBOX_CONFIRM,
                             require_full_test_benchmark=True)
    assert captured and "--require-full-test-benchmark" in captured[0]
    flag_at = captured[0].index("--require-full-test-benchmark")
    assert "True" not in captured[0], "a bare store_true flag must not carry a value"
    assert flag_at == len(captured[0]) - 1 or captured[0][flag_at + 1].startswith("--")

    captured.clear()
    tools.walkforward_report(panel_csv=str(panel), confirm=tools.LOCKBOX_CONFIRM)
    assert captured and "--require-full-test-benchmark" not in captured[0], (
        "the formal flag must be opt-in, not always on")
