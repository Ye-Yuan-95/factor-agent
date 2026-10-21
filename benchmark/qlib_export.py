"""Export a known-truth synthetic panel to Qlib's on-disk binary format.

Why this exists: external factor-mining systems (RD-Agent(Q), AlphaGen, AlphaForge) all read
Qlib datasets. To put them on OUR ground-truth panels we need the panels in their format --
this module is that adapter.

Two hard rules, both enforced here rather than left to run-time discipline:

  1. THE TRUTH NEVER ENTERS THE DATASET. The injected mechanism/linear signal is written to a
     separate sidecar file outside the Qlib directory. A competitor reading the Qlib dataset
     must be restricted to qlib_data/; only the evaluator should read the sidecar.
  2. THE EXPORT IS CONTENT-HASHED. Every export writes an export_manifest.json carrying the
     panel content hash, the truth definition, and the file inventory, so a benchmark run can
     be pinned to exactly the bytes the competitor read.

Qlib layout produced (freq="day" is Qlib's default bucket name; our panels are monthly and the
calendar simply holds month-end dates):

    <out>/qlib_data/calendars/day.txt         one date per line (YYYY-MM-DD)
    <out>/qlib_data/instruments/all.txt       <symbol>\\t<start>\\t<end>
    <out>/qlib_data/features/<symbol>/<field>.day.bin
    <out>/truth/truth_signal.csv              SIDECAR: date,asset,truth_signal
    <out>/export_manifest.json

Each .bin is a little-endian float32 array whose FIRST element is the instrument's start index
into the calendar, matching Qlib's own dump_bin.py convention.

CLI::

    python -m benchmark.qlib_export --dgp mechanism --seed 2120 --out outputs/benchmark/mech_s2120
    python -m benchmark.qlib_export --dgp linear --seed 7 --n-assets 300 --n-months 240 --out ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_PROJECT_DIR = Path(__file__).resolve().parents[1]
for _path in (str(_PROJECT_DIR), str(_PROJECT_DIR / "synthetic")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from factor_research.experiment_manifest import git_commit  # noqa: E402

LABEL_FIELD = "forward_return"
TRUTH_COLUMN = "truth_signal"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(repo_dir: Path | None = None) -> str:
    """Record a checkout's full commit; an unversioned copy has no parent-repo identity."""
    return git_commit(repo_dir or _PROJECT_DIR) or "unknown"


def external_repo_commits(paths: list[str] | tuple[str, ...]) -> dict[str, str]:
    """Pinned commits of the competitor checkouts a benchmark run will use.

    Recorded in the export manifest because "which build of RD-Agent read this dataset" is
    part of the evidence: external repositories change, and a result
    that cannot name the build it came from cannot be reproduced six months later.
    """
    out: dict[str, str] = {}
    for raw in paths:
        path = Path(raw)
        if path.exists():
            out[path.name] = _git_commit(path)
    return out


def panel_content_hash(panel: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(list(panel.columns), ensure_ascii=False).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(panel, index=True, categorize=True).values.tobytes())
    return digest.hexdigest()[:16]


def _write_bin(path: Path, start_index: int, values: np.ndarray) -> None:
    payload = np.hstack([np.array([start_index], dtype="float32"),
                         np.asarray(values, dtype="float32")])
    payload.astype("<f4").tofile(path)


def export_qlib(panel: pd.DataFrame, out_dir: Path, *, fields: list[str] | None = None,
                truth: pd.Series | None = None, truth_definition: str = "",
                external_commits: dict[str, str] | None = None,
                extra_meta: dict[str, Any] | None = None) -> dict[str, Any]:
    """Write a Qlib calendar, instrument list, feature binaries and file manifest.

    Require one contiguous, duplicate-free calendar span per instrument; gaps are
    refused rather than padded. Truth columns cannot enter the feature panel, and
    optional truth values are saved only in an evaluator sidecar outside qlib_data.
    Require an empty destination to keep older feature files out of the export.
    Return hashes and layout metadata without running an external miner.
    """
    if not {"date", "asset", LABEL_FIELD} <= set(panel.columns):
        raise ValueError(f"panel needs date/asset/{LABEL_FIELD} columns")
    # Refuse truth columns regardless of whether a truth sidecar was supplied.
    leaking = sorted(c for c in panel.columns
                     if c == TRUTH_COLUMN or str(c).startswith(f"{TRUTH_COLUMN}_"))
    if leaking:
        raise ValueError(
            f"panel carries truth column(s) {leaking}; the truth is sidecar-only and must "
            "never be exportable as a feature")
    fields = list(fields or [c for c in panel.columns if c not in ("date", "asset")])
    missing = sorted(set(fields) - set(panel.columns))
    if missing:
        raise ValueError(f"fields absent from panel: {missing}")
    if truth is not None and len(truth) != len(panel):
        raise ValueError(f"truth series length {len(truth)} does not match panel {len(panel)}")

    out_dir = Path(out_dir)
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise ValueError(f"output directory is not empty; choose a new path: {out_dir}")
    qlib_dir = out_dir / "qlib_data"
    features_dir = qlib_dir / "features"
    (qlib_dir / "calendars").mkdir(parents=True, exist_ok=True)
    (qlib_dir / "instruments").mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)

    frame = panel.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    calendar = pd.DatetimeIndex(sorted(frame["date"].unique()))
    calendar_pos = {stamp: i for i, stamp in enumerate(calendar)}
    (qlib_dir / "calendars" / "day.txt").write_text(
        "\n".join(stamp.strftime("%Y-%m-%d") for stamp in calendar) + "\n", encoding="utf-8")

    instrument_rows: list[str] = []
    written_files: list[Path] = []
    for asset, block in frame.groupby("asset", sort=True):
        block = block.sort_values("date")
        symbol = str(asset).lower()
        dates = block["date"]
        # duplicate dates would make the row count match a longer span and slip past a pure
        # length comparison, silently misaligning every value against the calendar
        if dates.duplicated().any():
            duplicates = sorted(dates[dates.duplicated()].dt.strftime("%Y-%m-%d").unique())
            raise ValueError(f"instrument {asset} has duplicate dates: {duplicates[:5]}")
        start, end = dates.iloc[0], dates.iloc[-1]
        span = calendar[(calendar >= start) & (calendar <= end)]
        if len(span) != len(block) or not np.array_equal(dates.to_numpy(), span.to_numpy()):
            raise ValueError(f"instrument {asset} is not contiguous on the calendar "
                             f"({len(block)} rows vs {len(span)} calendar dates); "
                             "the exporter does not pad gaps")
        symbol_dir = features_dir / symbol
        symbol_dir.mkdir(exist_ok=True)
        for field in fields:
            bin_path = symbol_dir / f"{field}.day.bin"
            _write_bin(bin_path, calendar_pos[start], block[field].to_numpy(dtype="float64"))
            written_files.append(bin_path)
        instrument_rows.append(
            f"{symbol}\t{start.strftime('%Y-%m-%d')}\t{end.strftime('%Y-%m-%d')}")
    instruments_path = qlib_dir / "instruments" / "all.txt"
    instruments_path.write_text("\n".join(instrument_rows) + "\n", encoding="utf-8")
    written_files.extend([qlib_dir / "calendars" / "day.txt", instruments_path])

    truth_path = None
    if truth is not None:
        truth_dir = out_dir / "truth"
        truth_dir.mkdir(parents=True, exist_ok=True)
        truth_path = truth_dir / "truth_signal.csv"
        pd.DataFrame({"date": frame["date"].to_numpy(), "asset": frame["asset"].to_numpy(),
                      TRUTH_COLUMN: np.asarray(truth)}).to_csv(
            truth_path, index=False, encoding="utf-8")

    # a file inventory with per-file digests: the panel hash alone pins what we INTENDED to
    # write, not what a competitor actually read off disk
    file_inventory = {path.relative_to(out_dir).as_posix(): _file_sha256(path)
                      for path in sorted(written_files)}
    if truth_path is not None:
        file_inventory[truth_path.relative_to(out_dir).as_posix()] = _file_sha256(truth_path)
    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "panel_content_hash": panel_content_hash(panel),
        "git_commit": _git_commit(),
        "external_repo_commits": dict(external_commits or {}),
        "n_instruments": len(instrument_rows),
        "n_dates": len(calendar),
        "n_feature_files": len(written_files) - 2,
        "file_inventory": file_inventory,
        "file_inventory_hash": hashlib.sha256(
            json.dumps(file_inventory, sort_keys=True).encode("utf-8")).hexdigest()[:16],
        "truth_sidecar_sha256": (_file_sha256(truth_path)[:16] if truth_path else None),
        "fields": fields,
        "label_field": LABEL_FIELD,
        "calendar_start": calendar[0].strftime("%Y-%m-%d"),
        "calendar_end": calendar[-1].strftime("%Y-%m-%d"),
        "truth_sidecar": str(truth_path.relative_to(out_dir)) if truth_path else None,
        "truth_definition": truth_definition,
        "leak_policy": ("The truth signal is NOT a field in qlib_data; competitors read only "
                        "qlib_data/, the evaluator reads truth/ separately."),
        **(extra_meta or {}),
    }
    with (out_dir / "export_manifest.json").open(
            "w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(manifest, indent=2, ensure_ascii=False))
        handle.write("\n")
    return manifest


ALPHAGEN_SLOTS = ("open", "close", "high", "low", "volume", "vwap")


def export_alphagen_slots(panel: pd.DataFrame, out_dir: Path, slot_columns: list[str],
                          **kwargs: Any) -> dict[str, Any]:
    """Map six factor columns to the fixed AlphaGen/AlphaForge feature-slot layout.

    Slot labels are open/close/high/low/volume/vwap, but the stored values remain
    factor exposures rather than market OHLCV. Persist the column-to-slot mapping
    in the export manifest. Pass require_columns to require particular truth
    ingredients; the exporter does not infer those columns automatically.

    For comparisons, callers must restrict competing generators to the same six
    inputs. Feature remapping alone does not enforce matched search spaces.
    """
    require = list(kwargs.pop("require_columns", []) or [])
    if len(slot_columns) != len(ALPHAGEN_SLOTS):
        raise ValueError(f"need exactly {len(ALPHAGEN_SLOTS)} slot columns, "
                         f"got {len(slot_columns)}")
    missing_truth = sorted(set(require) - set(slot_columns))
    if missing_truth:
        raise ValueError(
            f"truth columns {missing_truth} are not among the six exported slots; the truth "
            "would be unreachable and the benchmark arm would measure nothing")
    mapping = dict(zip(ALPHAGEN_SLOTS, slot_columns))
    renamed = panel[["date", "asset", LABEL_FIELD]].copy()
    for slot, column in mapping.items():
        renamed[slot] = panel[column].to_numpy()
    extra = kwargs.pop("extra_meta", {}) or {}
    extra = {**extra, "alphagen_slot_mapping": mapping,
             "slot_note": ("Field names are AlphaGen's six fixed leaves; the VALUES are our "
                           "factor columns per alphagen_slot_mapping.")}
    return export_qlib(renamed, out_dir, fields=[LABEL_FIELD, *ALPHAGEN_SLOTS],
                       extra_meta=extra, **kwargs)


def read_qlib_bin(path: Path, calendar_len: int | None = None) -> tuple[int, np.ndarray]:
    """Decode a nonempty Qlib float32 binary into (start_index, values).

    An optional calendar length rejects a series extending beyond its bounds.
    Export consumers and round-trip tests can use this helper without pyqlib.
    """
    raw = np.fromfile(path, dtype="<f4")
    start = int(raw[0])
    values = raw[1:]
    if calendar_len is not None and start + len(values) > calendar_len:
        raise ValueError("bin extends past the calendar")
    return start, values


def build_panel(kind: str, seed: int, *, n_assets: int, n_months: int, config_path: Path,
                calibration_csv: Path, ic: float, mech_kind: str, x_col: str, y_col: str,
                true_factor: str) -> tuple[pd.DataFrame, pd.Series | None, str]:
    """Build the chosen known-truth DGP panel from the provided calibration and seeded mechanism."""
    config_raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
    factor_cols = config_raw["factor_cols"]
    rng = np.random.default_rng(seed)
    if kind == "mechanism":
        from dgp_mechanism import calibrate, generate_mechanism_panel
        calib = calibrate(calibration_csv, factor_cols)
        panel = generate_mechanism_panel(rng, calib, n_assets, n_months, kind=mech_kind,
                                         x_col=x_col, y_col=y_col, ic=ic)
        from agent.factorial_ablation import _truth_series
        truth = (_truth_series(panel, mech_kind=mech_kind, x_col=x_col, y_col=y_col)
                 if ic > 0 else None)
        definition = (f"per-month standardized {mech_kind}({x_col}, {y_col}) mechanism, "
                      f"ic={ic}") if ic > 0 else "null panel (no injected premium)"
        return panel, truth, definition
    if kind == "linear":
        from dgp_v2 import calibrate_v2, generate_panel_v2
        calib = calibrate_v2(calibration_csv, factor_cols,
                             config_raw["family_constraints"]["factor_families"])
        idx = [list(calib["factors"]).index(true_factor)] if ic > 0 else []
        panel, _ = generate_panel_v2(rng, calib, n_assets, n_months, idx, [ic] * len(idx),
                                     coverage=calib.get("coverage", 0.973))
        truth = (pd.to_numeric(panel[true_factor], errors="coerce") if ic > 0 else None)
        definition = (f"linear premium on {true_factor}, ic={ic}" if ic > 0
                      else "null panel (no injected premium)")
        return panel, truth, definition
    raise ValueError(f"unknown dgp kind {kind!r} (mechanism|linear)")


def main() -> None:
    """Generate a calibrated synthetic panel and export a Qlib-readable dataset.

    Keep truth/evaluation sidecars separate from miner inputs and record supplied
    external repository identities. Exporting data does not run those systems.
    """
    parser = argparse.ArgumentParser(
        description="Export a known-truth synthetic panel to Qlib binary format")
    parser.add_argument("--dgp", choices=["mechanism", "linear"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-assets", type=int, default=150)
    parser.add_argument("--n-months", type=int, default=150)
    parser.add_argument("--ic", type=float, default=0.06, help="0 exports a null panel")
    parser.add_argument("--mech-kind", default="conditional")
    parser.add_argument("--x-col", default="ep")
    parser.add_argument("--y-col", default="illiq")
    parser.add_argument("--true-factor", default="mom1m", help="linear DGP only")
    parser.add_argument("--config", default=str(_PROJECT_DIR / "a_share_big_config.json"))
    parser.add_argument("--calibration-csv", default=str(
        _PROJECT_DIR / "data_big_v2" / "a_share_factor_panel.csv"))
    parser.add_argument("--external-repos", default="",
                        help="comma-separated competitor checkouts whose commits to pin")
    args = parser.parse_args()

    panel, truth, definition = build_panel(
        args.dgp, args.seed, n_assets=args.n_assets, n_months=args.n_months,
        config_path=Path(args.config), calibration_csv=Path(args.calibration_csv),
        ic=args.ic, mech_kind=args.mech_kind, x_col=args.x_col, y_col=args.y_col,
        true_factor=args.true_factor)
    external = external_repo_commits(
        [p.strip() for p in args.external_repos.split(",") if p.strip()])
    manifest = export_qlib(
        panel, Path(args.out), truth=truth, truth_definition=definition,
        external_commits=external,
        extra_meta={"dgp": args.dgp, "seed": args.seed, "inject_ic": args.ic,
                    "mech_kind": args.mech_kind if args.dgp == "mechanism" else None,
                    # Keep generation inputs even for null panels, whose truth
                    # list is empty, so readback can reconstruct the same market.
                    "x_col": args.x_col, "y_col": args.y_col,
                    "true_factor": args.true_factor,
                    "truth_columns": ([args.x_col, args.y_col] if args.dgp == "mechanism"
                                      else [args.true_factor]) if args.ic > 0 else []})
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
