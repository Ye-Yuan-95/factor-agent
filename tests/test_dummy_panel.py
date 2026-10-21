"""The tutorial must not silently replace a user's research panel."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_dummy_panel_requires_explicit_overwrite(tmp_path):
    panel = tmp_path / "panel.csv"
    original = b"date,asset,forward_return\n2020-01-31,REAL,0.123\n"
    panel.write_bytes(original)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"input_csv": str(panel), "factor_cols": ["mom1m"]}),
                      encoding="utf-8")
    cmd = [sys.executable, "-m", "factor_research.make_dummy_panel", "--config", str(config),
           "--assets", "3", "--months", "4"]

    refused = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    assert refused.returncode != 0
    assert "refusing to overwrite" in refused.stderr
    assert "Traceback" not in refused.stderr
    assert panel.read_bytes() == original

    replaced = subprocess.run([*cmd, "--overwrite"], capture_output=True, text=True,
                              timeout=30)
    assert replaced.returncode == 0, replaced.stderr
    frame = pd.read_csv(panel)
    assert len(frame) == 12
    assert frame["asset"].nunique() == 3
    assert "REAL" not in set(frame["asset"])
    assert frame["forward_return_end_date"].equals(frame["date"])


@pytest.mark.parametrize("option,value", [
    ("--assets", "0"), ("--assets", "-1"),
    ("--months", "0"), ("--months", "-1"),
])
def test_dummy_panel_rejects_nonpositive_size(tmp_path, option, value):
    panel = tmp_path / "panel.csv"
    result = subprocess.run(
        [sys.executable, "-m", "factor_research.make_dummy_panel", option, value,
         "--out", str(panel)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 2
    assert "must be positive integers" in result.stderr
    assert "Traceback" not in result.stderr
    assert not panel.exists()
