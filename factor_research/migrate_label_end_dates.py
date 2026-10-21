"""Derive missing forward_return_end_date values from the panel's raw prices.

For each asset and signal date, the label settles --forward-days trading days
later, following the builder's return convention. Verify recomputed returns
against stored forward_return values before writing. A mismatch refuses migration
because a plausible settlement date cannot substitute for the correct horizon.

Run: python -m factor_research.migrate_label_end_dates --panel-csv <panel.csv>
     --history-dir <raw-prices> --forward-days 21"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from factor_research.build_a_share_factor_panel import PRICE_COLUMNS, _read_selected_csv


def derive(panel: pd.DataFrame, history_dir: Path, forward_days: int,
           tolerance: float) -> tuple[pd.Series, dict]:
    """Derive label settlement dates from subsequent observed prices.
    Long gaps and missing price histories must remain visible instead of assuming a fixed calendar horizon."""
    ends = pd.Series(pd.NaT, index=panel.index, dtype="datetime64[ns]")
    checked = mismatched = missing_files = 0
    for asset, rows in panel.groupby("asset", sort=False):
        path = history_dir / f"{asset}.daily_prices.csv"
        if not path.is_file():
            missing_files += 1
            continue
        prices = _read_selected_csv(path, PRICE_COLUMNS)
        prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
        prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
        prices = (prices.dropna(subset=["date", "close"])
                  .loc[lambda f: f["close"] > 0.0]
                  .sort_values("date").reset_index(drop=True))
        if prices.empty:
            continue
        end_date = prices["date"].shift(-forward_days)
        fwd = prices["close"].shift(-forward_days) / prices["close"] - 1.0
        lookup_end = dict(zip(prices["date"], end_date))
        lookup_fwd = dict(zip(prices["date"], fwd))
        row_dates = pd.to_datetime(rows["date"])
        ends.loc[rows.index] = row_dates.map(lookup_end)
        recomputed = row_dates.map(lookup_fwd)
        stored = pd.to_numeric(rows["forward_return"], errors="coerce")
        both = recomputed.notna() & stored.notna()
        checked += int(both.sum())
        mismatched += int((np.abs(recomputed[both] - stored[both]) > tolerance).sum())
    return ends, {"rows_checked": checked, "rows_mismatched": mismatched,
                  "assets_without_price_file": missing_files}


def main() -> None:
    """Verify price-derived returns, attach settlement dates and write a manifest.

    Refuse labels inconsistent with the supplied raw prices. Labels spanning more
    than the configured calendar-day limit are voided while their factor rows remain.
    """
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel-csv", required=True)
    ap.add_argument("--history-dir", required=True)
    ap.add_argument("--forward-days", type=int, default=21)
    ap.add_argument("--tolerance", type=float, default=1e-6)
    ap.add_argument("--max-mismatch-rate", type=float, default=0.0,
                    help="fraction of rows whose recomputed forward_return may differ "
                         "before the migration refuses (default: none may)")
    ap.add_argument("--max-label-span-days", type=int, default=45,
                    help="void labels settling more than this many calendar days out; "
                         "must match the builder's rule (0 disables)")
    ap.add_argument("--out", default=None, help="default: overwrite the panel in place")
    args = ap.parse_args()
    if args.forward_days < 1:
        ap.error(f"--forward-days must be >= 1, got {args.forward_days}")
    if args.max_label_span_days < 0:
        # a negative cap voids essentially every label instead of widening anything
        ap.error(f"--max-label-span-days must be >= 0, got {args.max_label_span_days}")
    # NaN/Inf must be rejected explicitly: every comparison against NaN is False, so a
    # NaN tolerance reports mismatched=0 for ANY panel and the horizon check -- the whole
    # reason this script is trustworthy -- silently disappears.
    if not math.isfinite(args.tolerance) or args.tolerance < 0:
        ap.error(f"--tolerance must be a finite number >= 0, got {args.tolerance}")
    if not math.isfinite(args.max_mismatch_rate) or not 0.0 <= args.max_mismatch_rate <= 1.0:
        ap.error(f"--max-mismatch-rate must be a finite number in [0, 1], got "
                 f"{args.max_mismatch_rate}")

    panel_path = Path(args.panel_csv)
    panel = pd.read_csv(panel_path)
    if "forward_return_end_date" in panel.columns:
        raise SystemExit(f"{panel_path} already has forward_return_end_date")
    for col in ("date", "asset", "forward_return"):
        if col not in panel.columns:
            raise SystemExit(f"{panel_path} has no {col} column")

    ends, stats = derive(panel, Path(args.history_dir), args.forward_days, args.tolerance)
    resolved = int(ends.notna().sum())
    rate = (stats["rows_mismatched"] / stats["rows_checked"]) if stats["rows_checked"] else 1.0
    print(f"rows={len(panel)} resolved_end_dates={resolved} "
          f"checked={stats['rows_checked']} mismatched={stats['rows_mismatched']} "
          f"({rate:.4%}) assets_without_price_file={stats['assets_without_price_file']}")
    if stats["rows_checked"] == 0:
        raise SystemExit("no row could be verified against the raw prices; refusing")
    if rate > args.max_mismatch_rate:
        raise SystemExit(
            f"recomputed forward_return differs on {rate:.4%} of checked rows "
            f"(limit {args.max_mismatch_rate:.4%}); --forward-days {args.forward_days} "
            "does not explain this panel's labels -- refusing to stamp settlement dates "
            "onto labels the horizon cannot reproduce")

    panel["forward_return_end_date"] = ends
    panel["forward_return_span_days"] = (
        ends - pd.to_datetime(panel["date"])).dt.days
    # Same target definition as a fresh build: void over-long labels (keeping the row, so
    # a migrated panel and a fresh build agree) and clear their settlement date with them.
    cap = args.max_label_span_days or None
    if cap is not None:
        over = panel["forward_return_span_days"] > int(cap)
        panel.loc[over, ["forward_return", "forward_return_end_date"]] = (np.nan, pd.NaT)
        print(f"voided {int(over.sum())} labels spanning more than {cap} calendar days")
    panel["forward_return_end_date"] = pd.to_datetime(
        panel["forward_return_end_date"]).dt.strftime("%Y-%m-%d")
    out = Path(args.out) if args.out else panel_path
    panel.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"wrote {out}")

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from factor_research.build_a_share_factor_panel import write_panel_manifest
    write_panel_manifest(out, panel, {"source": "factor_research.migrate_label_end_dates",
                                      "forward_days": args.forward_days,
                                      "max_label_span_days": cap,
                                      "history_dir": str(args.history_dir),
                                      "tolerance": args.tolerance})


if __name__ == "__main__":
    main()
