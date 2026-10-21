"""Check checkout/package entry points, forwarding and controlled CLI refusals."""
from __future__ import annotations

import subprocess
import sys
import json
import os
import shlex
import shutil
import tempfile
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
ENTRY_POINTS = [(str(ROOT / "main.py"),), ("-m", "factor_research")]


def test_audit_reports_stay_outside_the_checkout():
    """Permit private reports while refusing paths that could become public docs."""
    with tempfile.TemporaryDirectory(prefix="factor-agent-private-audit-") as folder:
        private_report = Path(folder) / "comment_audit.md"
        command = [sys.executable, str(ROOT / "scripts" / "audit_comments.py"), "--report"]
        result = subprocess.run([*command, str(private_report)], cwd=ROOT,
                                capture_output=True, text=True, encoding="utf-8", timeout=30)
        assert result.returncode == 0, result.stdout + result.stderr
        assert private_report.read_text(encoding="utf-8").startswith("# Code comment audit")
        # A unique target tests refusal without touching any existing user document.
        public_report = ROOT / "docs" / (Path(folder).name + ".md")
        assert not public_report.exists()
        refused = subprocess.run([*command, str(public_report)], cwd=ROOT,
                                 capture_output=True, text=True, encoding="utf-8", timeout=30)
        assert refused.returncode == 2 and "must be outside the checkout" in refused.stderr
        assert not public_report.exists()


def test_readme_dummy_workflow(tmp_path):
    """Execute the README tutorial verbatim in a copy without private inputs or results.

    Read the commands from the Quick start block so documentation changes are
    exercised by CI. Check workflow contracts rather than fixing acceptance counts
    or statistical estimates to one dependency version.
    """
    checkout = tmp_path / "release"
    checkout.mkdir()
    for package in ("factor_research", "agent", "synthetic", "benchmark"):
        shutil.copytree(ROOT / package, checkout / package,
                        ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("main.py", "a_share_big_config.json", "a_share_baseline_config.json"):
        shutil.copy2(ROOT / name, checkout / name)
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    quickstart = readme.split("## Quick start", 1)[1].split("```sh", 1)[1].split("```", 1)[0]
    commands = [shlex.split(line)[1:] for line in quickstart.splitlines()
                if line.strip().startswith("python main.py ")]
    assert [command[:2] for command in commands] == [
        ["main.py", "--help"], ["main.py", "demo-data"], ["main.py", "run"]]
    env = dict(os.environ)
    env.pop("GLM_API_KEY", None)
    for command in commands:
        result = subprocess.run([sys.executable, *command], cwd=checkout, env=env,
                                capture_output=True, text=True, encoding="utf-8", timeout=600)
        assert result.returncode == 0, result.stdout + result.stderr

    config = json.loads((checkout / "a_share_big_config.json").read_text(encoding="utf-8"))
    panel_path = checkout / config["input_csv"]
    panel = pd.read_csv(panel_path)
    assert len(panel) == 60 * 96
    assert panel.asset.nunique() == 60 and panel.date.nunique() == 96
    assert set(config["factor_cols"]).issubset(panel.columns)
    assert not panel.duplicated(["date", "asset"]).any()
    assert (panel.forward_return_end_date == panel.date).all()
    run_dirs = list((checkout / "runs").iterdir())
    assert len(run_dirs) == 1
    run = run_dirs[0]
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["state"] == "COMPLETED" and manifest["n_trials"] == 5
    assert manifest["synthetic"] == "gen1" and manifest["seed"] == 7
    snapshot = json.loads((run / "config.snapshot.json").read_text(encoding="utf-8"))
    assert snapshot["generator"] == {"provider": "file"}
    assert snapshot["cli_args"]["synthetic_assets"] == 150
    assert snapshot["cli_args"]["synthetic_months"] == 150
    ledger = [json.loads(line) for line in
              (run / "memory" / "trials.jsonl").read_text(encoding="utf-8").splitlines()]
    cards = [json.loads(path.read_text(encoding="utf-8")) for path in (run / "cards").glob("*.json")]
    assert len(ledger) == len(cards) == 5
    assert {row["name"] for row in ledger} == {card["name"] for card in cards}
    assert all(card["test_sharpe_delta"] is None and not card["certified"] for card in cards)
    assert all(card["code_commit"] is None for card in cards)  # copy has no Git metadata
    factors = json.loads((run / "factor_list.json").read_text(encoding="utf-8"))
    assert len(factors) == len(cards)
    assert {factor["name"] for factor in factors} == {card["name"] for card in cards}
    report = (run / "user_report.md").read_text(encoding="utf-8")
    assert all(card["name"] in report and card["decision"] in report for card in cards)
    assert not list(checkout.glob("outputs/**/gate_certificate.json"))

    # Repeating the documented data command must preserve the generated input.
    original = panel_path.read_bytes()
    refused = subprocess.run([sys.executable, "main.py", "demo-data"], cwd=checkout,
                             capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert refused.returncode != 0 and "refusing to overwrite" in refused.stderr
    assert panel_path.read_bytes() == original


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_entry_help_and_research_inventory(entry):
    result = subprocess.run([sys.executable, *entry, "--help"], cwd=ROOT,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "demo-data" in result.stdout and "walkforward" in result.stdout
    result = subprocess.run([sys.executable, *entry, "research", "--help"], cwd=ROOT,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "tau_transfer" in result.stdout and "semi_synthetic" in result.stdout


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_entry_forwards_data_options_and_preserves_refusal(entry, tmp_path):
    panel = tmp_path / "panel.csv"
    command = [sys.executable, *entry, "demo-data", "--out", str(panel),
               "--assets", "3", "--months", "4", "--seed", "13"]
    generated = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert generated.returncode == 0, generated.stderr
    frame = pd.read_csv(panel)
    assert len(frame) == 12 and frame["asset"].nunique() == 3
    original = panel.read_bytes()
    refused = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert refused.returncode != 0 and "refusing to overwrite" in refused.stderr
    assert panel.read_bytes() == original


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_unknown_command_and_experiment_are_refused(entry):
    for args in (("unknown",), ("research", "unknown"), ("compare", "unknown"),
                 ("benchmark", "unknown")):
        result = subprocess.run([sys.executable, *entry, *args], cwd=ROOT,
                                capture_output=True, text=True, timeout=30)
        assert result.returncode == 2 and "invalid choice" in result.stderr
        assert "Traceback" not in result.stderr


def test_dispatch_restores_argv_and_preserves_child_exit(monkeypatch):
    from factor_research import cli

    original = sys.argv

    def failing_target(module, *, run_name):
        assert module == "synthetic.tau_transfer" and run_name == "__main__"
        assert sys.argv == [module, "--reps-cal", "3"]
        raise SystemExit(7)

    monkeypatch.setattr(cli.runpy, "run_module", failing_target)
    with pytest.raises(SystemExit) as error:
        cli.main(["research", "tau_transfer", "--reps-cal", "3"])
    assert error.value.code == 7
    assert sys.argv is original


@pytest.mark.parametrize("group,tool,module", [
    ("compare", "family", "agent.b2_compare"),
    ("compare", "mechanism", "agent.factorial_ablation"),
    ("benchmark", "export", "benchmark.qlib_export"),
    ("benchmark", "analyse", "benchmark.f4b_analysis"),
])
def test_grouped_tools_forward_all_child_arguments(monkeypatch, group, tool, module):
    from factor_research import cli

    observed = []

    def target(name, *, run_name):
        observed.append((name, run_name, sys.argv[:]))

    monkeypatch.setattr(cli.runpy, "run_module", target)
    assert cli.main([group, tool, "--output", "my path"]) == 0
    assert observed == [(module, "__main__", [module, "--output", "my path"])]


def test_baseline_cli_writes_candidates_and_refuses_invalid_budget_or_overwrite(tmp_path):
    out = tmp_path / "candidates" / "baseline.json"
    command = [sys.executable, str(ROOT / "main.py"), "baseline", "--n", "3",
               "--seed", "5", "--out", str(out)]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    candidates = json.loads(out.read_text(encoding="utf-8"))
    assert len(candidates) == 3 and len({c["expression"] for c in candidates}) == 3
    original = out.read_bytes()
    refused = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert refused.returncode != 0 and "refusing to overwrite" in refused.stderr
    assert out.read_bytes() == original
    invalid = subprocess.run([sys.executable, str(ROOT / "main.py"), "baseline",
                              "--n", "0", "--out", str(tmp_path / "invalid.json")],
                             cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert invalid.returncode == 2 and "positive integer" in invalid.stderr
    assert not (tmp_path / "invalid.json").exists()


def test_positive_control_cli_records_actual_grid_and_refuses_overwrite(tmp_path):
    config = json.loads((ROOT / "a_share_big_config.json").read_text(encoding="utf-8"))
    factors = ["mom1m", "mom6m", "ep", "illiq", "turn", "bm", "retvol", "size"]
    config["factor_cols"] = factors
    config["family_constraints"]["factor_families"] = {
        f: config["family_constraints"]["factor_families"][f] for f in factors}
    config["synthetic"]["n_scenarios"] = 2
    panel = tmp_path / "panel.csv"
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(config), encoding="utf-8")
    made = subprocess.run([sys.executable, str(ROOT / "main.py"), "demo-data",
                           "--config", str(cfg), "--out", str(panel),
                           "--assets", "50", "--months", "36"],
                          cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert made.returncode == 0, made.stderr
    out = tmp_path / "control"
    command = [sys.executable, str(ROOT / "main.py"), "research", "oracle_positive_control",
               "--config", str(cfg), "--panel-csv", str(panel), "--output-dir", str(out),
               "--n-assets", "50", "--n-months", "36", "--kinds", "conditional",
               "--ics", "0.06", "--seeds", "2000"]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    rows = pd.read_csv(out / "results.csv")
    assert len(rows) == 3 and set(rows["role"]) == {"true", "negative", "bare_ingredient"}
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["grid"]["seeds"] == [2000] and not meta["template_grid_matches"]
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "COMPLETED" and manifest["inputs_unchanged"]
    original = (out / "results.csv").read_bytes()
    refused = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert refused.returncode != 0 and "not empty" in refused.stderr
    assert (out / "results.csv").read_bytes() == original
