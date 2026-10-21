"""Run-scoped outputs + sealed-run state machine.

Each run owns `runs/<run_id>/`: manifest.json (identity + state), config.snapshot.json,
cards/ (this run's FactorCards), memory/ (trial ledger), trace.jsonl (append-only event log),
certificate_refs.json (the binding the oracle checked).

State machine (crash/concurrency-safe one-shot): ACTIVE (cards/memory may grow) -> COMPLETED
(frozen lockbox input) -> TEST_OPENING (atomic claim BEFORE any test row is read) ->
TEST_OPENED | TEST_OPEN_FAILED (both terminal -- a failed opening never returns to COMPLETED).
"""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from pathlib import Path

_PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_ROOT = _PROJECT_DIR / "runs"

_ALLOWED_TRANSITIONS = {
    "ACTIVE": {"COMPLETED"},
    "COMPLETED": {"TEST_OPENING"},
    "TEST_OPENING": {"TEST_OPENED", "TEST_OPEN_FAILED"},
    "TEST_OPENED": set(),
    "TEST_OPEN_FAILED": set(),
}
STATES = tuple(_ALLOWED_TRANSITIONS)
_SAFE_CARD_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _json1(obj) -> str:
    return json.dumps(obj, indent=1, ensure_ascii=False, default=str)


class RunContext:
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.cards_dir = self.run_dir / "cards"
        self.memory_dir = self.run_dir / "memory"
        self._trace_path = self.run_dir / "trace.jsonl"
        self._manifest_path = self.run_dir / "manifest.json"

    # ------------------------------------------------------------------ #
    @classmethod
    def create(cls, *, purpose: str, universe: str, config_snapshot: dict,
               manifest_extra: dict | None = None, runs_root: str | Path | None = None) -> "RunContext":
        """Create a unique run directory and snapshot request, configuration and source identity."""
        root = Path(runs_root) if runs_root else DEFAULT_RUNS_ROOT
        run_id = time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
        ctx = cls(root / run_id)
        ctx.cards_dir.mkdir(parents=True, exist_ok=True)
        ctx.memory_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "run_id": run_id,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "purpose": purpose,
            "universe": universe,
            "state": "ACTIVE",
            **(manifest_extra or {}),
        }
        ctx._write_manifest(manifest)
        (ctx.run_dir / "config.snapshot.json").write_text(_json1(config_snapshot), encoding="utf-8")
        ctx.trace("run_created", purpose=purpose, universe=universe)
        return ctx

    @classmethod
    def load(cls, run_dir: str | Path) -> "RunContext":
        ctx = cls(Path(run_dir))
        if not ctx._manifest_path.exists():
            raise FileNotFoundError(f"{run_dir} has no manifest.json -- not a run directory")
        return ctx

    # ------------------------------------------------------------------ #
    def manifest(self) -> dict:
        return json.loads(self._manifest_path.read_text(encoding="utf-8"))

    def _write_manifest(self, manifest: dict) -> None:
        self._manifest_path.write_text(_json1(manifest), encoding="utf-8")

    @property
    def state(self) -> str:
        return self.manifest().get("state", "ACTIVE")

    def set_state(self, new_state: str, **extra) -> None:
        if new_state not in STATES:
            raise ValueError(f"unknown state {new_state}")
        manifest = self.manifest()
        current = manifest.get("state", "ACTIVE")
        if new_state not in _ALLOWED_TRANSITIONS.get(current, set()):
            raise PermissionError(f"illegal state transition: {current} -> {new_state}")
        manifest["state"] = new_state
        manifest.update(extra)
        self._write_manifest(manifest)
        self.trace("state_change", state=new_state, **extra)

    @property
    def lock_path(self) -> Path:
        return self.run_dir / "lockbox.lock"

    def claim_lockbox(self) -> None:
        """Atomically claim the one-shot lockbox opening via O_EXCL file creation: exactly one
        process can succeed, and the never-removed lock file keeps a crashed claim unopenable
        (conservative one-shot). Call BEFORE reading any test row."""
        if self.state != "COMPLETED":
            raise PermissionError(f"run is {self.state}; only COMPLETED runs can claim the lockbox")
        try:
            fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise PermissionError(
                "a lockbox opening was already claimed for this run (possibly by a concurrent "
                "or crashed process); strict one-shot forbids another attempt") from None
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"claimed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                 "pid": os.getpid()}))
        self.set_state("TEST_OPENING")

    def assert_writable(self) -> None:
        """Generation/validation is allowed only while the run is ACTIVE."""
        state = self.state
        if state != "ACTIVE":
            raise PermissionError(
                f"run {self.run_dir.name} is {state}: no further generation, validation or "
                "config change is allowed (sealed-run rule)")

    # ------------------------------------------------------------------ #
    def trace(self, event: str, **fields) -> None:
        record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": event, **fields}
        with self._trace_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def write_card(self, name: str, card_dict: dict) -> Path:
        self.assert_writable()
        if not isinstance(name, str) or not _SAFE_CARD_NAME.fullmatch(name):
            raise ValueError(
                "card name must match [A-Za-z0-9][A-Za-z0-9._-]{0,127}; "
                "model-generated or path-like names are forbidden")
        path = self.cards_dir / f"{name}.json"
        if path.exists():  # no silent overwrite of history
            path = self.cards_dir / f"{name}.{uuid.uuid4().hex[:4]}.json"
        path.write_text(json.dumps(card_dict, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def write_certificate_refs(self, refs: dict) -> None:
        (self.run_dir / "certificate_refs.json").write_text(_json1(refs), encoding="utf-8")
