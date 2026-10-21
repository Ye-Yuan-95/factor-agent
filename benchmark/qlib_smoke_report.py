"""Verify a Qlib export from disk and emit a compact, hash-bound smoke artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_PROJECT_DIR = Path(__file__).resolve().parents[1]
for _path in (str(_PROJECT_DIR), str(_PROJECT_DIR / "synthetic")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from benchmark.qlib_export import build_panel  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_export(export_dir: Path, *, config_path: Path,
                  calibration_csv: Path) -> dict[str, Any]:
    """Verify inventory hashes and native Qlib readback for sampled export fields.

    With pyqlib available, regenerate the seeded panel from the supplied inputs
    and compare ep, illiq and forward_return for instrument s0000 at float32
    tolerance. Check inventory paths for truth leakage; this sampled smoke check
    does not run a miner or prove all possible leakage channels are absent.
    """
    manifest_path = export_dir / "export_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    inventory = dict(manifest["file_inventory"])
    mismatches: list[str] = []
    for relative, expected in inventory.items():
        path = export_dir / relative
        if not path.is_file() or _sha256(path) != expected:
            mismatches.append(relative)
    inventory_hash = hashlib.sha256(
        json.dumps(inventory, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    if inventory_hash != manifest["file_inventory_hash"]:
        raise RuntimeError("manifest inventory aggregate hash does not match")
    if mismatches:
        raise RuntimeError(f"export inventory mismatch: {mismatches[:5]}")

    try:
        import qlib
    except ImportError as exc:
        raise SystemExit(
            "Native Qlib readback requires the optional pyqlib package in a compatible "
            "environment. Exporting data and read_qlib_bin do not require pyqlib."
        ) from exc
    from qlib.config import REG_CN
    from qlib.data import D

    qlib.init(provider_uri=str((export_dir / "qlib_data").resolve()), region=REG_CN)
    fields = [field for field in manifest["fields"] if field != "forward_return"][:2]
    fields.append("forward_return")
    observed = D.features(
        ["s0000"], [f"${field}" for field in fields],
        start_time=manifest["calendar_start"], end_time=manifest["calendar_end"],
        freq="day")
    truth_columns = manifest.get("truth_columns", [])
    x_col = manifest.get("x_col", truth_columns[0] if truth_columns else "ep")
    y_col = manifest.get("y_col", truth_columns[1] if len(truth_columns) > 1 else "illiq")
    true_factor = manifest.get("true_factor", truth_columns[0] if truth_columns else "mom1m")
    panel, _truth, _definition = build_panel(
        str(manifest["dgp"]), int(manifest["seed"]),
        n_assets=int(manifest["n_instruments"]), n_months=int(manifest["n_dates"]),
        config_path=config_path, calibration_csv=calibration_csv,
        ic=float(manifest["inject_ic"]), mech_kind=str(manifest["mech_kind"]),
        x_col=str(x_col), y_col=str(y_col), true_factor=str(true_factor))
    expected = panel[
        panel["asset"].astype(str).str.casefold() == "s0000"
    ].sort_values("date")
    if len(observed) != len(expected):
        raise RuntimeError(
            f"Qlib returned {len(observed)} rows, expected {len(expected)}")
    comparisons: dict[str, Any] = {}
    for field in fields:
        actual_values = observed[f"${field}"].to_numpy(dtype=float)
        expected_values = expected[field].to_numpy(dtype=float)
        error = np.abs(actual_values - expected_values)
        comparisons[field] = {
            "max_abs_error": round(float(np.nanmax(error)), 12),
            "allclose_float32": bool(np.allclose(
                actual_values, expected_values, rtol=1e-6, atol=1e-6,
                equal_nan=True)),
        }
    if not all(row["allclose_float32"] for row in comparisons.values()):
        raise RuntimeError(f"Qlib native readback differs from source: {comparisons}")
    truth_leaked = any(
        "truth_signal" in relative and relative.startswith("qlib_data/")
        for relative in inventory)
    if truth_leaked:
        raise RuntimeError("truth sidecar leaked into qlib_data")
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "status": "PASS",
        "export_manifest_sha256": _sha256(manifest_path),
        "export_git_commit": manifest["git_commit"],
        "external_repo_commits": manifest["external_repo_commits"],
        "file_inventory_hash": manifest["file_inventory_hash"],
        "n_inventory_files_verified": len(inventory),
        "pyqlib_version": qlib.__version__,
        "native_readback": {
            "instrument": "s0000",
            "n_rows": len(observed),
            "fields": comparisons,
        },
        "truth_absent_from_qlib_data": True,
        "verification_script": str(Path(__file__).resolve().relative_to(_PROJECT_DIR)),
        "verification_script_sha256": _sha256(Path(__file__)),
    }


def main() -> None:
    """Verify export files and input identities, then write a smoke report.

    Calibration/config paths identify the data being checked; this command
    validates an export and does not perform external-system discovery research.
    """
    parser = argparse.ArgumentParser(description="Verify Qlib export and write smoke artifact")
    parser.add_argument(
        "--export-dir",
        default=str(_PROJECT_DIR / "outputs" / "benchmark" / "mech_s2120"))
    parser.add_argument("--config", default=str(_PROJECT_DIR / "a_share_big_config.json"))
    parser.add_argument(
        "--calibration-csv",
        default=str(_PROJECT_DIR / "data_big_v2" / "a_share_factor_panel.csv"))
    parser.add_argument("--report", default="")
    args = parser.parse_args()
    export_dir = Path(args.export_dir).resolve()
    report = verify_export(
        export_dir, config_path=Path(args.config).resolve(),
        calibration_csv=Path(args.calibration_csv).resolve())
    report_path = (
        Path(args.report).resolve() if args.report
        else export_dir / "qlib_smoke_report.json")
    with report_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(report, indent=2, ensure_ascii=False))
        handle.write("\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
