"""Shared provenance manifests and strict-JSON output for research tools.

Every experiment output should say what it ran on (panel/config content hashes), at what
code state (git commit, best-effort), with what resolved parameters -- and it should be
parseable by a strict JSON reader. json.dumps happily emits NaN, which is not JSON; a
report that needs a lenient parser is a report that hides its missing data.
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str | None:
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def git_commit(root: str | Path = _ROOT) -> str | None:
    """Return this source root's commit, or null for an unversioned release copy.

    A tarball or installed wheel can sit inside another repository. Git's parent
    search must not attribute that unrelated repository's commit to this code.
    Both checkout directories and worktree .git files are supported.
    """
    try:
        if not (Path(root) / ".git").exists():
            return None
        proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(root),
                              capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    head = proc.stdout.strip()
    return head if proc.returncode == 0 and len(head) == 40 else None


def build_manifest(*, panel_csv: str | Path | None = None,
                   config_path: str | Path | None = None,
                   extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Record input file hashes and best-effort Git identity with caller-provided run metadata."""
    manifest: dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": git_commit(),
    }
    if panel_csv is not None:
        manifest["panel_csv"] = str(panel_csv)
        manifest["panel_sha256"] = file_sha256(panel_csv)
    if config_path is not None:
        manifest["config_path"] = str(config_path)
        manifest["config_sha256"] = file_sha256(config_path)
    if extra:
        manifest.update(extra)
    return manifest


def _sanitize(value: Any) -> Any:
    """NaN/Inf become null: strict JSON has no spelling for them, and a metric that is
    NaN is missing data, which null states honestly."""
    if isinstance(value, dict):
        return {key: _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_strict_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(_sanitize(payload), indent=2, ensure_ascii=False, allow_nan=False,
                   default=str),
        encoding="utf-8")
