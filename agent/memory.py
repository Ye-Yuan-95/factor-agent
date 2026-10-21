"""Trial memory + counting policy: the anti-data-snooping ledger.

Answers n_trials (candidates that reached the gates), trial_sharpes (per-trial history
INCLUDING rejects -- DSR input #3 needs the cross-trial variance) and n_eff (near-duplicates
folded by clustering monthly LS series at |corr| >= threshold).

Counting policy: a candidate counts once validate() ran; sandbox failures do NOT count
(logged to failures.jsonl); an exact duplicate (same signature+universe) is recorded with
counted=False; near-duplicates DO count in n_trials and fold into n_eff.

Storage: append-only JSONL (trials.jsonl / failures.jsonl); each trial keeps its monthly LS
series (train+val only) so n_eff and the DSR variance are recomputable from disk.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .deflated_sharpe import effective_n_trials


class MemoryStore:
    def __init__(self, memory_dir: str | Path):
        self.dir = Path(memory_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.trials_path = self.dir / "trials.jsonl"
        self.failures_path = self.dir / "failures.jsonl"
        self._trials: list[dict] = []
        if self.trials_path.exists():
            with self.trials_path.open(encoding="utf-8") as fh:
                self._trials = [json.loads(line) for line in fh if line.strip()]

    # -- recording ----------------------------------------------------- #
    def record_trial(self, *, name: str, expression: str, signature: str, family: str,
                     universe: str, sharpe: float | None, mean_ic: float | None,
                     monthly_ls: list[float] | None, decision: str,
                     reject_reason: str | None, monthly_ls_dates: list[str] | None = None,
                     extra: dict | None = None) -> dict:
        """Record a candidate that reached the statistical gates. Returns the stored record.
        Exact duplicates (signature+universe already counted) are stored with counted=False.
        monthly_ls_dates: stored WITH the values so cross-trial correlation aligns
        by month, not by array tail."""
        values = list(monthly_ls or [])
        dates = list(monthly_ls_dates or [])
        if dates and len(dates) != len(values):
            raise ValueError("monthly_ls_dates must align 1:1 with monthly_ls")
        vals, kept_dates = _finite_series(values, dates)
        record = {
            "trial_id": len(self._trials) + 1,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "name": name,
            "expression": expression,
            "signature": signature,
            "family": family,
            "universe": universe,
            "sharpe": _finite_or_none(sharpe),
            "mean_ic": _finite_or_none(mean_ic),
            "monthly_ls": [round(v, 8) for v in vals],
            "monthly_ls_dates": kept_dates,
            "decision": decision,
            "reject_reason": reject_reason,
            "counted": not self.is_duplicate(signature, universe),
        }
        if extra:
            record.update(extra)
        _append_jsonl(self.trials_path, record)
        self._trials.append(record)
        return record

    def record_failure(self, *, name: str, expression: str, error: str, universe: str) -> None:
        """Sandbox syntax/execution failure: logged for the audit but NOT a trial (trial counting policy)."""
        _append_jsonl(self.failures_path, {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "name": name, "expression": expression, "error": error, "universe": universe,
        })

    # -- queries (the validate() inputs) -------------------------------- #
    def counted_trials(self) -> list[dict]:
        return [t for t in self._trials if t.get("counted")]

    def n_trials(self) -> int:
        return len(self.counted_trials())

    def trial_sharpes(self) -> list[float]:
        return [t["sharpe"] for t in self.counted_trials() if t.get("sharpe") is not None]

    def is_duplicate(self, signature: str, universe: str) -> bool:
        return any(t["signature"] == signature and t["universe"] == universe
                   for t in self.counted_trials())

    def n_eff(self, corr_threshold: float = 0.7, include_candidate_ls: list[float] | None = None,
              include_candidate_dates: list[str] | None = None) -> int:
        """Effective independent trials: cluster counted trials (optionally + the current candidate)
        on the correlation of their monthly LS return series. Series WITH stored dates correlate on
        the date intersection; date-less series fall back to tail alignment.
        Trials with no usable series each count as their own cluster (conservative)."""
        entries: list[tuple[list[float], list[str] | None]] = [
            (t["monthly_ls"], t.get("monthly_ls_dates")) for t in self.counted_trials()]
        if include_candidate_ls is not None:
            entries.append(_finite_series(list(include_candidate_ls),
                                          list(include_candidate_dates or [])))
        with_data = [(v, d) for v, d in entries if len(v) >= 6]
        n_without = len(entries) - len(with_data)  # unusable series: one cluster each
        n = len(with_data)
        corr = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                corr[i, j] = corr[j, i] = _pair_corr(with_data[i], with_data[j])
        return effective_n_trials(corr, threshold=corr_threshold) + n_without

    def stats(self) -> dict:
        """Summarize the persistent trial ledger, retaining unsuccessful searches in trial counts."""
        counted = self.counted_trials()
        sharpes = self.trial_sharpes()
        decisions = dict(Counter(t["decision"] for t in counted))
        n_failures = 0
        if self.failures_path.exists():
            with self.failures_path.open(encoding="utf-8") as fh:
                n_failures = sum(1 for line in fh if line.strip())
        return {
            "n_trials": len(counted),
            "n_recorded": len(self._trials),
            "n_eff": self.n_eff() if counted else 0,
            "n_sandbox_failures_not_counted": n_failures,
            "decisions": decisions,
            "sharpe_mean": _finite_or_none(np.mean(sharpes)) if sharpes else None,
            "sharpe_var": _finite_or_none(np.var(sharpes, ddof=1)) if len(sharpes) >= 2 else None,
            "memory_dir": str(self.dir),
        }


def _append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _finite_series(values: list, dates: list[str]) -> tuple[list[float], list[str] | None]:
    """Drop non-finite values, keeping any dates aligned 1:1."""
    vals = [float(v) for v in values]
    keep = [i for i, v in enumerate(vals) if np.isfinite(v)]
    return [vals[i] for i in keep], ([dates[i] for i in keep] if dates else None)


def _pair_corr(a: tuple[list[float], list[str] | None],
               b: tuple[list[float], list[str] | None]) -> float:
    """Correlation of two monthly series: date-aligned when both carry dates, tail-aligned
    otherwise (date-less records). Returns 0.0 when overlap is too short to be meaningful."""
    va, da = a
    vb, db = b
    if da and db:
        common = [d for d in da if d in set(db)]
        if len(common) < 6:
            return 0.0
        map_a, map_b = dict(zip(da, va)), dict(zip(db, vb))
        xa = np.array([map_a[d] for d in common], dtype=float)
        xb = np.array([map_b[d] for d in common], dtype=float)
    else:
        m = min(len(va), len(vb))
        if m < 6:
            return 0.0
        xa = np.array(va[-m:], dtype=float)
        xb = np.array(vb[-m:], dtype=float)
    if xa.std() == 0 or xb.std() == 0:
        return 0.0
    c = float(np.corrcoef(xa, xb)[0, 1])
    return c if np.isfinite(c) else 0.0


def _finite_or_none(value) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None
