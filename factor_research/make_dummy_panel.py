"""Generate a small random factor panel so the agent loop can be tried with zero real data.

The panel matches a_share_big_config.json's schema (all factor columns + date/asset/
forward_return). In a checkout, its default path is data_big_v2/ (gitignored),
matching the agent tutorial's calibration input. An installed wheel needs explicit
--config and --out paths because repository configs are separate inputs.
Its factors and returns are independent random draws. A synthetic runner can use
this CSV for calibration and then inject signals into a separate generated panel.
The tutorial exercises the workflow; acceptance counts are not guaranteed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Generate schema-compatible noise data, refusing overwrite unless requested.

    Dummy labels settle on their signal dates because they use no future prices;
    this panel exercises interfaces and is not historical investment evidence.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "a_share_big_config.json"))
    ap.add_argument("--assets", type=int, default=60)
    ap.add_argument("--months", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None,
                    help="override the config's input path")
    ap.add_argument("--overwrite", action="store_true",
                    help="explicitly replace an existing CSV with dummy data")
    args = ap.parse_args()
    if args.assets <= 0 or args.months <= 0:
        ap.error("--assets and --months must be positive integers")

    config_file = Path(args.config)
    if not config_file.is_file():
        raise SystemExit(
            f"config not found: {config_file}\n"
            "The default config ships with the repository checkout, not the installed "
            "wheel. Run from a checkout or pass --config and --out explicitly.")
    cfg = json.loads(config_file.read_text(encoding="utf-8"))
    out = Path(args.out) if args.out else ROOT / cfg["input_csv"]
    if out.exists() and not args.overwrite:
        raise SystemExit(
            f"refusing to overwrite existing panel: {out}\n"
            "Use --out for a new path, or --overwrite to replace it with dummy data.")
    rng = np.random.default_rng(args.seed)

    dates = pd.date_range("2018-01-31", periods=args.months, freq="ME")
    assets = [f"S{i:04d}" for i in range(args.assets)]
    idx = pd.MultiIndex.from_product([dates, assets], names=["date", "asset"])
    df = pd.DataFrame(index=idx).reset_index()
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    for f in cfg["factor_cols"]:
        df[f] = rng.normal(size=len(df))
    # heavy-tailed returns so the Student-t dof calibration sees realistic excess kurtosis
    df["forward_return"] = 0.05 * rng.standard_t(df=6, size=len(df))
    # Portfolio columns: without these the chain stops at factor selection --
    # portfolio_construction needs float_mktcap and hybrid_walkforward filters on tradable.
    df["float_mktcap"] = np.exp(rng.normal(22.0, 1.0, size=len(df)))
    df["market_cap"] = df["float_mktcap"] * rng.uniform(1.0, 1.4, size=len(df))
    df["tradable"] = (rng.uniform(size=len(df)) > 0.03).astype(int)
    # Noise labels are not built from future prices, so they settle on the signal date;
    # this keeps the purge machinery a no-op on the tutorial panel while letting every
    # consumer enforce it uniformly.
    df["forward_return_end_date"] = df["date"]

    out.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also protects a file created after the existence check.
    with out.open("w" if args.overwrite else "x", encoding="utf-8", newline="") as handle:
        df.to_csv(handle, index=False)
    print(f"wrote {out}  rows={len(df)} cols={len(df.columns)}")


if __name__ == "__main__":
    main()
