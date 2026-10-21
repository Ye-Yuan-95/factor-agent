"""Certificate binding + statistical acceptance helpers.

The binding hash covers: the pipeline config content, the oracle's gate parameters, a content
fingerprint of GATE_SOURCE_FILES (edits to THOSE files invalidate the certificate; docs/report
commits do not; decision logic outside the listed files is NOT fingerprinted), the DGP module
fingerprints, the calibration panel's CONTENT hash, and the acceptance rule + targets.

Acceptance is CI-based, not a point estimate: over R Monte-Carlo reps,
    one-sided upper CI of FPR  <= target_fpr   AND   one-sided lower CI of Power >= target_power.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import stats

_PROJECT_DIR = Path(__file__).resolve().parents[1]

# source files whose behaviour the certificate certifies
GATE_SOURCE_FILES = (
    _PROJECT_DIR / "factor_research" / "factor_validity_pipeline.py",
    _PROJECT_DIR / "agent" / "validator_oracle.py",
    _PROJECT_DIR / "agent" / "gate_certificate.py",
    _PROJECT_DIR / "agent" / "factor_dsl.py",
    _PROJECT_DIR / "agent" / "deflated_sharpe.py",
    _PROJECT_DIR / "agent" / "memory.py",          # date alignment / N_eff feeds the DSR gate
    _PROJECT_DIR / "agent" / "certification.py",   # the binding/CI algorithm itself
    _PROJECT_DIR / "synthetic" / "stage_a_fpr_power.py",
    _PROJECT_DIR / "synthetic" / "dgp_v2.py",
    _PROJECT_DIR / "synthetic" / "four_gate_ablation.py",
)

_CONTENT_HASH_CACHE: dict[tuple, str] = {}


def normalized_source_bytes(path: str | Path) -> bytes:
    """Normalize source line endings so LF and CRLF checkouts share a binding.
    Data inputs retain their exact byte identity through content_hash."""
    return Path(path).read_bytes().replace(b"\r\n", b"\n")


def content_hash(path: str | Path, chunk: int = 1 << 20) -> str:
    """Chunked SHA-256 of file content, cached by path, size and modification time."""
    p = Path(path)
    st = p.stat()
    key = (str(p.resolve()), st.st_size, st.st_mtime_ns)   # ns: same-second rewrites invalidate
    if key in _CONTENT_HASH_CACHE:
        return _CONTENT_HASH_CACHE[key]
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    digest = h.hexdigest()[:16]
    _CONTENT_HASH_CACHE[key] = digest
    return digest


def gate_code_fingerprint() -> str:
    """sha256 over the concatenated contents of the gate-relevant source files."""
    h = hashlib.sha256()
    for path in GATE_SOURCE_FILES:
        h.update(path.name.encode())
        h.update(normalized_source_bytes(path))
    return h.hexdigest()[:16]


def compute_binding(*, pipeline_config_raw: dict, oracle_gate_params: dict,
                    panel_content_hash: str, acceptance: dict,
                    dgp_versions: dict | None = None) -> dict:
    """Deterministic binding record + hash. Both the certificate issuer and the oracle call this
    with their OWN current state; equality of `binding_hash` is what 'bound' means."""
    elements = {
        "pipeline_config": pipeline_config_raw,
        "oracle_gate_params": oracle_gate_params,
        "gate_code_fingerprint": gate_code_fingerprint(),
        "panel_content_hash": panel_content_hash,
        "acceptance": acceptance,
        "dgp_versions": dgp_versions or {"gen1": "stage_a_fpr_power", "gen2": "dgp_v2"},
    }
    canonical = json.dumps(elements, sort_keys=True, ensure_ascii=False, default=str)
    return {
        "binding_hash": hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16],
        "gate_code_fingerprint": elements["gate_code_fingerprint"],
        "panel_content_hash": panel_content_hash,
        "oracle_gate_params": oracle_gate_params,
        "acceptance": acceptance,
    }


# --------------------------------------------------------------------------- #
# statistical acceptance                                                       #
# --------------------------------------------------------------------------- #
def _t_mean_margin(values, conf: float) -> tuple[float, float] | None:
    """(mean, t-interval margin) over iid Monte-Carlo reps; None if fewer than 2 finite values."""
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if len(arr) < 2:
        return None
    se = arr.std(ddof=1) / np.sqrt(len(arr))
    return arr.mean(), stats.t.ppf(conf, len(arr) - 1) * se


def one_sided_upper(values, conf: float = 0.95) -> float | None:
    """One-sided upper confidence bound of the mean over iid Monte-Carlo reps (t-interval)."""
    mm = _t_mean_margin(values, conf)
    return None if mm is None else float(mm[0] + mm[1])


def one_sided_lower(values, conf: float = 0.95) -> float | None:
    mm = _t_mean_margin(values, conf)
    return None if mm is None else float(mm[0] - mm[1])


def wilson_upper(k: int, n: int, conf: float = 0.95) -> float | None:
    """One-sided Wilson upper bound for a pooled proportion k/n. Candidate dependence must be assessed separately."""
    if n <= 0:
        return None
    z = stats.norm.ppf(conf)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(min(1.0, centre + half))
