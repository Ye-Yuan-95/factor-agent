"""Real CLI regressions for config/source mixing in resumed experiments."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "experiments/resume_test.json"
OUTPUT = "outputs/resume_test"
pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="runner requires git")


def _git(repo, *args):
    return subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid", *args],
        cwd=repo, capture_output=True, text=True, check=True)


def _run(repo, *args, arms="random_narrow,random_wide"):
    return subprocess.run(
        [sys.executable, "-m", "agent.factorial_ablation", "--protocol", PROTOCOL,
         "--arms", arms, *args],
        cwd=repo, capture_output=True, text=True, timeout=120)


def _artifacts(repo):
    out = repo / OUTPUT
    return {p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in out.rglob("*") if p.is_file()}


@pytest.fixture(scope="module")
def completed_experiment(tmp_path_factory):
    """Create a small completed experiment whose hashes and artifacts can be deliberately corrupted."""
    repo = tmp_path_factory.mktemp("factorial_repo")
    # Copy current source (including local source files), then give the isolated
    # experiment its own committed protocol and source for real provenance checks.
    for package in ("agent", "synthetic", "factor_research"):
        shutil.copytree(ROOT / package, repo / package,
                        ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(ROOT / "a_share_big_config.json", repo / "a_share_big_config.json")
    protocol = json.loads((ROOT / "experiments/factorial_mechanism_2026-07-25_F2.json")
                          .read_text(encoding="utf-8"))
    protocol["protocol_id"] = "resume_test"
    protocol["output_dir"] = OUTPUT
    protocol["dgp"].update(n_assets=40, n_months=72,
                           alternative_seeds=[9901], null_seeds=[9902])
    protocol["budgets"] = {"generation_attempts": 4, "oracle_trials": 1}
    protocol["cells"] = []
    protocol["baseline_arms"] = protocol["baseline_arms"][:2]
    (repo / "experiments").mkdir()
    (repo / PROTOCOL).write_text(json.dumps(protocol), encoding="utf-8")
    (repo / ".gitignore").write_text("data*/\noutputs/\n__pycache__/\n", encoding="utf-8")
    _git(repo, "init", "--quiet")
    _git(repo, "config", "core.autocrlf", "false")
    _git(repo, "add", ".")
    _git(repo, "commit", "--quiet", "-m", "Test fixture")
    subprocess.run([sys.executable, "-m", "factor_research.make_dummy_panel", "--assets", "40",
                    "--months", "72"], cwd=repo, capture_output=True, text=True,
                   check=True, timeout=30)
    result = _run(repo, "--defer-summary", arms="random_narrow")
    assert result.returncode == 0, result.stderr
    assert len(list((repo / OUTPUT / "runs").glob("*/*/result.json"))) == 2
    return repo


@pytest.fixture()
def experiment(completed_experiment, tmp_path):
    return shutil.copytree(completed_experiment, tmp_path / "repo",
                           ignore=shutil.ignore_patterns("__pycache__"))


@pytest.mark.parametrize("mode", ["--defer-summary", "--summarize-only"])
def test_changed_config_refused_before_resume_or_summary(experiment, mode):
    path = experiment / "a_share_big_config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["screening"]["q_value_max"] = 0.0
    path.write_text(json.dumps(config), encoding="utf-8")
    before = _artifacts(experiment)
    result = _run(experiment, mode)
    assert result.returncode != 0
    assert "config hash changed" in result.stderr
    assert "[resume]" not in result.stdout
    assert _artifacts(experiment) == before


def test_equivalent_config_can_resume_and_add_an_arm(experiment):
    path = experiment / "a_share_big_config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    # Formatting and key order are not changes to the experiment.
    path.write_text(json.dumps(config, sort_keys=True, indent=4), encoding="utf-8")
    before = _artifacts(experiment)
    result = _run(experiment, "--defer-summary")
    assert result.returncode == 0, result.stderr
    assert "[resume]" in result.stdout
    after = _artifacts(experiment)
    assert all(after[name] == digest for name, digest in before.items())
    assert len(list((experiment / OUTPUT / "runs").glob("*/*/result.json"))) == 4


@pytest.mark.parametrize("source", ["agent/deflated_sharpe.py", "agent/certification.py"])
def test_changed_decision_dependency_refused_before_and_after_commit(experiment, source):
    path = experiment / source
    text = path.read_text(encoding="utf-8")
    if source.endswith("deflated_sharpe.py"):
        original = "dsr = probabilistic_sharpe(sharpe, sr_ref, n_obs, skew=skew, kurt=kurt)"
        assert original in text
        text = text.replace(original, "dsr = 1.0  # regression mutation")
    else:
        text += "\n# regression mutation of the certificate dependency\n"
    path.write_text(text, encoding="utf-8")
    before = _artifacts(experiment)
    result = _run(experiment, "--defer-summary")
    assert result.returncode != 0
    assert "frozen input differs from HEAD" in result.stderr
    assert source in result.stderr
    assert _artifacts(experiment) == before

    # Committing the change satisfies Git provenance, but must still invalidate
    # the source fingerprint associated with the already completed experiment.
    _git(experiment, "add", source)
    _git(experiment, "commit", "--quiet", "-m", "Change decision dependency")
    result = _run(experiment, "--defer-summary")
    assert result.returncode != 0
    assert "source fingerprint changed" in result.stderr
    assert "[resume]" not in result.stdout
    assert _artifacts(experiment) == before
