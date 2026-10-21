"""Fresh-clone CLI integration: dummy panel, selection, unified solver, portfolio
and walk-forward. Subprocesses use the documented inputs and selection artifacts
on small panels, including controlled refusals for mismatched contracts."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FACTORS = ["mom6m", "ep", "illiq", "std_turn", "bm", "dolvol"]

cvxpy = pytest.importorskip("cvxpy", reason="the optimization stages need the [opt] extra")


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    result = subprocess.run([sys.executable, *cmd], capture_output=True, text=True,
                            cwd=ROOT, timeout=600, **kwargs)
    assert result.returncode == 0, f"{cmd[0]} failed:\n{result.stderr[-1500:]}"
    return result


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    """Run the tutorial chain in isolation and share its artifacts across assertions."""
    tmp = tmp_path_factory.mktemp("chain")
    raw = json.loads((ROOT / "a_share_big_config.json").read_text(encoding="utf-8"))
    raw["input_csv"] = str(tmp / "panel.csv")
    raw["output_dir"] = str(tmp / "selection")
    raw["factor_cols"] = FACTORS
    raw["family_constraints"]["factor_families"] = {
        "mom6m": "momentum", "ep": "value", "bm": "value",
        "illiq": "liquidity", "std_turn": "liquidity", "dolvol": "liquidity"}
    raw["family_constraints"]["max_per_family"] = {"momentum": 2, "value": 2, "liquidity": 2}
    raw.setdefault("synthetic", {})["n_scenarios"] = 8
    config = tmp / "config.json"
    config.write_text(json.dumps(raw), encoding="utf-8")

    _run(["-m", "factor_research.make_dummy_panel", "--config", str(config), "--assets", "40",
          "--months", "72", "--out", str(tmp / "panel.csv")])
    _run(["-m", "factor_research.factor_validity_pipeline", "--config", str(config),
          "--train-end", "2021-12-31"])
    _run(["-m", "factor_research.solve_unified_v2", "--panel-csv", str(tmp / "panel.csv"),
          "--selection-csv", str(tmp / "selection" / "final_factor_selection.csv"),
          "--config", str(config), "--output-dir", str(tmp / "v2"),
          "--train-end", "2021-12", "--no-survival", "--max-cardinality", "2",
          "--solver", "auto"])
    return tmp, config


def test_v2_solved_with_a_verified_selection(chain):
    tmp, _ = chain
    solution = json.loads((tmp / "v2" / "unified_v2_solution.json").read_text(encoding="utf-8"))
    assert solution["status"] in ("optimal", "optimal_inaccurate")
    assert solution["solver_requested"] == "auto"
    assert solution["solver"] in ("MOSEK", "SCIP", "ECOS_BB")  # the resolved one
    assert solution["manifest"]["panel_sha256"]
    assert solution["manifest"]["selection_manifest"]["train_end"] == "2021-12-31"


def test_portfolio_backtest_runs_on_the_dummy_panel(chain):
    tmp, _ = chain
    _run(["-m", "factor_research.portfolio_construction", "--panel-csv", str(tmp / "panel.csv"),
          "--weights-csv", str(tmp / "selection" / "sparse_weights.csv"),
          "--output-dir", str(tmp / "portfolio"), "--w-max", "0.05"])
    summary = json.loads((tmp / "portfolio" / "portfolio_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "OK"
    assert summary["manifest"]["panel_sha256"]
    # strict JSON: this must parse with a strict reader (no NaN tokens)
    json.loads((tmp / "portfolio" / "portfolio_summary.json").read_text(encoding="utf-8"),
               parse_constant=lambda name: pytest.fail(f"non-strict JSON constant {name}"))


def test_walkforward_reports_honest_insufficient_data_on_noise(chain):
    """Noise with no validated factors produces a controlled INSUFFICIENT_DATA report with failed refits and explicit skip reasons."""
    tmp, config = chain
    result = subprocess.run(
        [sys.executable, "-m", "factor_research.hybrid_walkforward", "--panel-csv", str(tmp / "panel.csv"),
         "--config", str(config), "--output-dir", str(tmp / "wf"),
         "--w-max", "0.05", "--cov-window", "12", "--solver", "auto",
         "--index-csv", str(tmp / "no_such_benchmark.csv")],
        capture_output=True, text=True, cwd=ROOT, timeout=600)
    assert result.returncode == 1
    report = json.loads((tmp / "wf" / "hybrid_walkforward.json").read_text(encoding="utf-8"),
                        parse_constant=lambda name: pytest.fail(f"non-strict JSON {name}"))
    assert report["status"] == "INSUFFICIENT_DATA"
    refits = report["refits"]
    assert refits and all(r["success"] is False for r in refits)   # gates reject noise
    assert all("weights" in r and "selector_info" in r for r in refits)
    assert report["skipped_months"]
    assert {s["reason"] for s in report["skipped_months"]} == {"no_validated_factors"}
    assert report["benchmark_index"] is None if "benchmark_index" in report else True
    assert report["manifest"]["benchmark_available"] is False
    assert report["manifest"]["solver_requested"] == "auto"
    refit_file = json.loads((tmp / "wf" / "walkforward_refits.json").read_text(encoding="utf-8"))
    assert refit_file["refits"]


def test_v2_refuses_an_unverifiable_selection(chain, tmp_path):
    """Missing or mismatched selection provenance must be refused."""
    tmp, config = chain
    import shutil
    orphan = tmp_path / "sel"
    shutil.copytree(tmp / "selection", orphan)
    (orphan / "selection_manifest.json").unlink()
    result = subprocess.run(
        [sys.executable, "-m", "factor_research.solve_unified_v2", "--panel-csv", str(tmp / "panel.csv"),
         "--selection-csv", str(orphan / "final_factor_selection.csv"),
         "--config", str(config), "--output-dir", str(tmp_path / "v2"),
         "--train-end", "2021-12", "--no-survival", "--max-cardinality", "2",
         "--solver", "auto"],
        capture_output=True, text=True, cwd=ROOT, timeout=300)
    assert result.returncode != 0
    assert "selection manifest missing" in (result.stderr + result.stdout)

    tampered = tmp_path / "sel2"
    shutil.copytree(tmp / "selection", tampered)
    manifest = json.loads((tampered / "selection_manifest.json").read_text(encoding="utf-8"))
    manifest["panel_sha256"] = "0" * 64
    (tampered / "selection_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "factor_research.solve_unified_v2", "--panel-csv", str(tmp / "panel.csv"),
         "--selection-csv", str(tampered / "final_factor_selection.csv"),
         "--config", str(config), "--output-dir", str(tmp_path / "v2b"),
         "--train-end", "2021-12", "--no-survival", "--max-cardinality", "2",
         "--solver", "auto"],
        capture_output=True, text=True, cwd=ROOT, timeout=300)
    assert result.returncode != 0
    assert "different panel" in (result.stderr + result.stdout)

    # config swap: same selection, different config -> refuse
    other_cfg = tmp_path / "other_config.json"
    other_cfg.write_text((ROOT / "a_share_baseline_config.json").read_text(encoding="utf-8"),
                         encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "factor_research.solve_unified_v2", "--panel-csv", str(tmp / "panel.csv"),
         "--selection-csv", str(tmp / "selection" / "final_factor_selection.csv"),
         "--config", str(other_cfg), "--output-dir", str(tmp_path / "v2c"),
         "--train-end", "2021-12", "--no-survival", "--max-cardinality", "2",
         "--solver", "auto"],
        capture_output=True, text=True, cwd=ROOT, timeout=300)
    assert result.returncode != 0
    assert "different config" in (result.stderr + result.stdout)

    # selection CSV edited after the pipeline wrote it -> refuse
    edited = tmp_path / "sel3"
    shutil.copytree(tmp / "selection", edited)
    csv_path = edited / "final_factor_selection.csv"
    csv_path.write_text(csv_path.read_text(encoding="utf-8") + "extra_row,0,False",
                        encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "factor_research.solve_unified_v2", "--panel-csv", str(tmp / "panel.csv"),
         "--selection-csv", str(csv_path),
         "--config", str(config), "--output-dir", str(tmp_path / "v2d"),
         "--train-end", "2021-12", "--no-survival", "--max-cardinality", "2",
         "--solver", "auto"],
        capture_output=True, text=True, cwd=ROOT, timeout=300)
    assert result.returncode != 0
    assert "does not match the hash" in (result.stderr + result.stdout)
