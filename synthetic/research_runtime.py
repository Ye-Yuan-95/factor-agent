"""Run bookkeeping for the optional research experiments, not the agent oracle.

Input hashes, effective arguments and source hashes describe a *new* experiment;
they do not certify it or establish agreement with historical research numbers.
The output directory must be empty, so a rerun cannot erase prior evidence.
"""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from factor_research.experiment_manifest import build_manifest, file_sha256, write_strict_json

ROOT = Path(__file__).resolve().parents[1]
_active: tuple[Path, dict] | None = None


def source_hashes() -> dict[str, str]:
    """Hash the experiment dependency surface, including uncommitted edits.

    Hash factor_research, synthetic, agent and benchmark Python files rather than trying
    to infer imports dynamically; shared gate changes must remain visible.
    """
    paths = list(ROOT.glob("*.py"))
    for folder in ("factor_research", "synthetic", "agent", "benchmark"):
        paths.extend((ROOT / folder).glob("*.py"))
    return {p.relative_to(ROOT).as_posix(): file_sha256(p) for p in sorted(paths)}


def validate_args(args) -> list[str]:
    """Reject invalid budgets, overlapping calibration seeds and incomplete inputs.

    These are research panels with direct synthetic labels, not tradable-panel
    certification. Missing exposures are allowed; missing factor *columns* are
    rejected to prevent the declared truth set changing silently.
    """
    for name in ("reps", "reps_cal", "reps_val", "survival_B", "n_assets", "n_months"):
        if hasattr(args, name) and getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    for name in ("q_max", "kappa", "target"):
        if hasattr(args, name) and not 0 < getattr(args, name) < 1:
            raise ValueError(f"{name} must lie strictly between 0 and 1")
    for name in ("seed", "seed_cal", "seed_val"):
        if hasattr(args, name) and getattr(args, name) < 0:
            raise ValueError(f"{name} must be nonnegative")
    if hasattr(args, "t_thresh") and (
        not np.isfinite(args.t_thresh) or args.t_thresh <= 0
    ):
        raise ValueError("t_thresh must be finite and positive")
    if hasattr(args, "strength") and (
        not np.isfinite(args.strength) or args.strength <= 0
    ):
        raise ValueError("strength must be finite and positive")
    for name in ("targets", "taus"):
        if hasattr(args, name):
            values = [float(x) for x in getattr(args, name).split(",")]
            if not values or any(not np.isfinite(v) or v <= 0 for v in values):
                raise ValueError(f"{name} must contain finite positive values")
            if name == "targets" and any(v >= 1 for v in values):
                raise ValueError("targets must lie strictly between 0 and 1")
    if hasattr(args, "seed_cal"):
        cal = range(args.seed_cal, args.seed_cal + args.reps_cal)
        val = range(args.seed_val, args.seed_val + args.reps_val)
        if max(cal.start, val.start) < min(cal.stop, val.stop):
            raise ValueError("calibration and validation seed ranges must be disjoint")
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    factors = cfg["factor_cols"]
    if len(factors) < 5 or len(factors) != len(set(factors)):
        raise ValueError("provide at least five distinct factor columns")
    frame = pd.read_csv(args.panel_csv)
    required = ["date", "asset", "forward_return", *factors]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"panel is missing columns: {missing}")
    if frame.empty or pd.to_datetime(frame["date"], errors="coerce").isna().any():
        raise ValueError("panel must contain observations with valid dates")
    if frame["asset"].isna().any() or frame.duplicated(["date", "asset"]).any():
        raise ValueError("panel requires unique, nonmissing date/asset keys")
    for col in ["forward_return", *factors]:
        values = pd.to_numeric(frame[col], errors="coerce")
        if np.isinf(values).any() or values.notna().sum() < 3 or values.std() <= 0:
            raise ValueError(f"{col} needs finite, nonconstant observed values")
    families = cfg["family_constraints"]["factor_families"]
    if any(not families.get(f) for f in factors):
        raise ValueError("each factor needs an explicit family")
    if len({families[f] for f in factors}) < 4:
        raise ValueError("the cross-family scenarios require at least four families")
    if getattr(args, "n_months", 10) < 10:
        raise ValueError("n_months must be at least 10 for split FM estimation")
    return factors


def prepare_run(args, script: str) -> None:
    """Validate experiment inputs and create RUNNING evidence before simulation.

    Refuse nonempty output directories and incompatible factor/family or seed
    contracts before consuming RNG draws. Snapshot input/source hashes, resolved
    arguments and environment; one active run is tracked per CLI process.
    """
    global _active
    factors = validate_args(args)
    if Path(script).stem in {"incremental_robustness", "tau_transfer"}:
        cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
        families = cfg["family_constraints"]["factor_families"]
        # These arms explicitly claim four momentum truth factors; an empty or
        # shortened slice would silently change the hypothesis being tested.
        if sum(families[f] == "momentum" for f in factors) < 4:
            raise ValueError(
                "same-family scenarios require at least four momentum factors"
            )
    out = Path(args.output_dir).resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError(f"output directory is not empty; choose a new path: {out}")
    out.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(
        panel_csv=args.panel_csv,
        config_path=args.config,
        extra={
            "status": "RUNNING",
            "experiment": Path(script).stem,
            "parameters": vars(args).copy(),
            "factor_order": factors,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                n: importlib.metadata.version(n) for n in ("numpy", "pandas", "scipy")
            },
            "source_sha256": source_hashes(),
            "evidence_scope": "new research run; historical numerical agreement not checked",
            "metric_notes": "FPR uses all false factors; conditional_power uses only finite gate-reaching true statistics. Neither is FDR.",
        },
    )
    try:
        # Without this source root's commit, a parent repository's status is
        # unrelated metadata and must remain unavailable too.
        state = None
        if manifest["git_commit"] is not None:
            state = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=10,
            )
        manifest["git_dirty"] = (
            bool(state.stdout.strip()) if state is not None and state.returncode == 0 else None
        )
    except (OSError, subprocess.TimeoutExpired):
        manifest["git_dirty"] = None
    write_strict_json(out / "manifest.json", manifest)
    _active = (out, manifest)


def finish_run(status: str, error: str | None = None) -> None:
    """Finalize hashes and strict JSON, refusing COMPLETED after input/source edits.

    Record output identities and preserve FAILED evidence on simulation errors.
    Undefined JSON metrics become null. Clear the active run after writing its
    terminal manifest; an integrity failure is an error, not a scientific result.
    """
    global _active
    if _active is None:
        return
    out, manifest = _active
    manifest["status"] = status
    if error:
        manifest["error"] = error
    stable = (
        file_sha256(manifest["panel_csv"]) == manifest["panel_sha256"]
        and file_sha256(manifest["config_path"]) == manifest["config_sha256"]
        and source_hashes() == manifest["source_sha256"]
    )
    manifest["inputs_unchanged"] = stable
    if not stable:
        manifest["status"] = "FAILED"
        manifest["error"] = "input or source changed during execution"
    # Strict JSON represents undefined rates as null; CSV missing values retain
    # their tabular representation.
    for path in out.glob("*.json"):
        if path.name != "manifest.json":
            write_strict_json(path, json.loads(path.read_text(encoding="utf-8")))
    manifest["output_sha256"] = {
        p.name: file_sha256(p)
        for p in sorted(out.iterdir())
        if p.is_file() and p.name != "manifest.json"
    }
    write_strict_json(out / "manifest.json", manifest)
    _active = None
    if not stable and status == "COMPLETED":
        raise ValueError(manifest["error"])


def run_experiment(main) -> None:
    """CLI wrapper: preserve failed-run evidence and return nonzero on failure."""
    try:
        main()
        finish_run("COMPLETED")
    except Exception as exc:
        finish_run("FAILED", f"{type(exc).__name__}: {exc}")
        raise SystemExit(f"Research experiment failed: {exc}") from exc
