"""Build A-share exposures and forward labels from dated raw source files.

Accounting and share data join by availability date; labels carry settlement
dates for downstream purging. docs/DATA_CONTRACT.md defines units and data limitations.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PRICE_COLUMNS = {
    "date": "date",
    "prices.open": "open",
    "prices.close": "close",
    "prices.high": "high",
    "prices.low": "low",
    "prices.amount": "amount",
    "prices.volume": "volume",
}

BALANCE_COLUMNS = {
    "date": "report_date",
    "notice_date": "notice_date_bs",
    "balance_sheet.assets": "assets",
    "balance_sheet.assets.current": "current_assets",
    "balance_sheet.assets.current.inventories": "inventories",
    "balance_sheet.assets.current.cash": "cash",
    "balance_sheet.liab": "total_liabilities",
    "balance_sheet.liab.current": "current_liabilities",
    "balance_sheet.equity": "equity",
    "balance_sheet.equity.parent_interests": "parent_equity",
}

INCOME_COLUMNS = {
    "date": "report_date",
    "notice_date": "notice_date_inc",
    "income_statement.profit": "net_profit",
    "income_statement.parent_interests": "parent_net_profit",
    "income_statement.profit.operating": "operating_profit",
    "income_statement.profit.operating.income.revenue": "revenue",
    "income_statement.profit.operating.expenses.costs": "costs",
}

CASHFLOW_COLUMNS = {
    "date": "report_date",
    "notice_date": "notice_date_cf",
    "cash_flow_statement.change.operating": "ocf",
}

EQUITY_COLUMNS = {
    "date": "report_date_eq",
    "notice_date": "notice_date_eq",
    "shares.total": "shares_total",
    "shares.circulating": "shares_float",
}

FACTOR_COLUMNS = (
    "mom1m",
    "mom6m",
    "mom12m",
    "mom_12_1",
    "mom36m",
    "volatility",
    "retvol",
    "beta",
    "idiovol",
    "dolvol",
    "volume_proxy",
    "turn",
    "size",
    "currat",
    "quick",
    "roeq",
    "agr",
    "mom3m",
    "mom9m",
    "maxret",
    "minret",
    "retvol6m",
    "retskew",
    "retkurt",
    "downvol",
    "high52w",
    "illiq",
    "mom24m",
    "std_dolvol",
    "std_turn",
    "roa",
    "gma",
    "grossmargin",
    "operprof",
    "assetturn",
    "lev",
    "cashdebt",
    "acc",
    "chinv",
    "ocf2assets",
    "bm",
    "ep",
    "sp",
    "cfp",
)

# fundamental columns (ratios + carried levels) produced by read_symbol_fundamentals
FUND_COLS = ["currat", "quick", "roeq", "agr", "roa", "gma", "grossmargin", "operprof",
             "assetturn", "lev", "cashdebt", "acc", "chinv", "ocf2assets",
             "book_equity", "earnings", "revenue_lvl", "ocf_lvl"]


@dataclass
class BuildResult:
    panel: pd.DataFrame
    audit: pd.DataFrame
    column_inventory: pd.DataFrame


def _read_selected_csv(path: Path, mapping: dict[str, str]) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    existing = [col for col in mapping if col in header.columns]
    frame = pd.read_csv(path, usecols=existing)
    return frame.rename(columns={col: mapping[col] for col in existing})


def _as_numeric(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for col in columns:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    den = denominator.replace(0.0, np.nan)
    out = numerator / den
    return out.replace([np.inf, -np.inf], np.nan)


def _symbol_from_price_path(path: Path) -> str:
    return path.name.removesuffix(".daily_prices.csv")


def load_symbols(history_dir: Path, max_symbols: int | None = None) -> list[str]:
    symbols = [_symbol_from_price_path(path) for path in sorted(history_dir.glob("*.daily_prices.csv"))]
    if max_symbols is not None:
        return symbols[:max_symbols]
    return symbols


def read_symbol_prices(history_dir: Path, symbol: str) -> pd.DataFrame:
    """Load one symbol's daily prices, coerce fields and discard invalid dates or nonpositive closes."""
    path = history_dir / f"{symbol}.daily_prices.csv"
    frame = _read_selected_csv(path, PRICE_COLUMNS)
    required = {"date", "close"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = _as_numeric(frame, ["open", "close", "high", "low", "amount", "volume"])
    frame = frame.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    frame = frame.loc[frame["close"] > 0.0].copy()
    frame["asset"] = symbol
    return frame


def void_overlong_labels(frame: pd.DataFrame, max_span_days: int | None) -> pd.DataFrame:
    """Void labels that span more than `max_span_days` calendar days.

    Dropping the value (rather than the row) keeps the factor observation available as a
    control while removing a target the cross-section cannot compare. The settlement date
    goes with it: a NaT end date is treated as "never settles" by the purge, so such a row
    can never enter a training window either."""
    if max_span_days is None:
        return frame
    over = frame["forward_return_span_days"] > int(max_span_days)
    frame.loc[over, ["forward_return", "forward_return_end_date"]] = (np.nan, pd.NaT)
    return frame


def add_price_factors(frame: pd.DataFrame, forward_days: int) -> pd.DataFrame:
    """Construct trailing price features and forward labels with their actual settlement dates.
    Forward labels are targets, never signal inputs; long gaps are handled by the panel builder."""
    frame = frame.copy()
    close = frame["close"]
    returns = close.pct_change(fill_method=None)
    frame["ret_1d"] = returns
    frame["forward_return"] = close.shift(-forward_days) / close - 1.0
    # The date whose close settles the label. Training cutoffs must compare against THIS,
    # not the signal date: a row dated 2021-12-31 with forward_days=21 is a label made of
    # January prices, and "date <= train_end" alone feeds it to the selection statistics.
    frame["forward_return_end_date"] = frame["date"].shift(-forward_days)
    # shift(-N) means "the Nth NEXT OBSERVATION", not "N trading days later": through a
    # halt it can span months. Keep calendar spans explicit so long-horizon labels
    # cannot share a cross-section with the intended short-horizon labels.
    frame["forward_return_span_days"] = (
        frame["forward_return_end_date"] - frame["date"]).dt.days
    frame["mom1m"] = close.pct_change(21, fill_method=None)
    frame["mom6m"] = close.pct_change(126, fill_method=None)
    frame["mom12m"] = close.pct_change(252, fill_method=None)
    frame["mom_12_1"] = close.shift(21) / close.shift(252) - 1.0
    frame["mom36m"] = close.pct_change(756, fill_method=None)
    frame["volatility"] = returns.rolling(60, min_periods=40).std() * np.sqrt(252.0)
    frame["retvol"] = returns.rolling(21, min_periods=15).std() * np.sqrt(252.0)
    if "amount" in frame.columns:
        frame["dolvol"] = np.log1p(frame["amount"].rolling(21, min_periods=10).mean())
    else:
        frame["dolvol"] = np.nan
    if "volume" in frame.columns:
        frame["volume_proxy"] = np.log1p(frame["volume"].rolling(21, min_periods=10).mean())
    else:
        frame["volume_proxy"] = np.nan
    # --- extended real price/volume factors (cheap, daily-only) ---
    frame["mom3m"] = close.pct_change(63, fill_method=None)
    frame["mom9m"] = close.pct_change(189, fill_method=None)
    frame["maxret"] = returns.rolling(21, min_periods=15).max()             # MAX / lottery effect
    frame["minret"] = returns.rolling(21, min_periods=15).min()
    frame["retvol6m"] = returns.rolling(126, min_periods=80).std() * np.sqrt(252.0)
    frame["retskew"] = returns.rolling(60, min_periods=40).skew()           # idiosyncratic skewness
    frame["retkurt"] = returns.rolling(60, min_periods=40).kurt()
    neg = returns.where(returns < 0, 0.0)
    frame["downvol"] = neg.rolling(60, min_periods=40).std() * np.sqrt(252.0)  # downside risk
    frame["high52w"] = close / close.rolling(252, min_periods=120).max()    # 52-week-high ratio
    if "amount" in frame.columns:
        illiq = (returns.abs() / frame["amount"].replace(0.0, np.nan)).rolling(21, min_periods=10).mean()
        frame["illiq"] = np.log1p(illiq * 1e8)                              # Amihud illiquidity (scaled)
        frame["std_dolvol"] = np.log1p(frame["amount"]).rolling(21, min_periods=10).std()
    else:
        frame["illiq"] = np.nan
        frame["std_dolvol"] = np.nan
    frame["mom24m"] = close.pct_change(504, fill_method=None)
    return frame


def read_symbol_equity_structures(history_dir: Path, symbol: str, lag_days: int) -> pd.DataFrame:
    """Load share counts and determine availability from disclosures or the configured reporting lag."""
    path = history_dir / f"{symbol}.equity_structures.csv"
    if not path.exists():
        return pd.DataFrame()
    eq = _read_selected_csv(path, EQUITY_COLUMNS)
    if "report_date_eq" not in eq.columns:
        return pd.DataFrame()
    eq["report_date_eq"] = pd.to_datetime(eq.get("report_date_eq"), errors="coerce")
    eq["notice_date_eq"] = pd.to_datetime(eq.get("notice_date_eq"), errors="coerce")
    eq = _as_numeric(eq, ["shares_total", "shares_float"])
    eq = eq.dropna(subset=["report_date_eq"]).sort_values("report_date_eq").reset_index(drop=True)
    if eq.empty:
        return eq
    # Point-in-time availability: prefer the announcement date, else lag the report date.
    available = eq["notice_date_eq"].fillna(eq["report_date_eq"] + pd.Timedelta(days=lag_days))
    eq["available_date"] = available
    eq = eq.dropna(subset=["available_date"]).sort_values("available_date").reset_index(drop=True)
    return eq[["available_date", "shares_total", "shares_float"]]


def attach_equity_to_daily(prices: pd.DataFrame, equity: pd.DataFrame) -> pd.DataFrame:
    """As-of join available share counts to prices and derive size and turnover features."""
    prices = prices.sort_values("date").reset_index(drop=True).copy()
    derived = ["shares_total", "shares_float", "market_cap", "float_mktcap", "turn", "size", "std_turn"]
    if equity is None or equity.empty:
        for col in derived:
            prices[col] = np.nan
        return prices
    merged = pd.merge_asof(
        prices,
        equity.sort_values("available_date"),
        left_on="date",
        right_on="available_date",
        direction="backward",
    )
    merged["market_cap"] = merged["close"] * merged["shares_total"]
    merged["float_mktcap"] = merged["close"] * merged["shares_float"]
    volume = merged["volume"] if "volume" in merged.columns else pd.Series(np.nan, index=merged.index)
    turnover_daily = _safe_divide(volume, merged["shares_float"])
    merged["turn"] = turnover_daily.rolling(21, min_periods=10).mean()
    merged["std_turn"] = turnover_daily.rolling(21, min_periods=10).std()
    merged["size"] = np.log(merged["float_mktcap"].replace(0.0, np.nan))
    return merged.drop(columns=["available_date"])


def add_market_factors(daily: pd.DataFrame) -> pd.DataFrame:
    """Estimate rolling beta and residual risk against the panel's equal-weight market proxy.
    The proxy depends on the supplied universe and is not an external benchmark index."""
    daily = daily.copy()
    market = daily.groupby("date", sort=True)["ret_1d"].mean().rename("market_return")
    daily = daily.merge(market, on="date", how="left")

    def _per_asset(group: pd.DataFrame) -> pd.DataFrame:
        asset = group.name
        group = group.sort_values("date").copy()
        if "asset" not in group.columns:
            group["asset"] = asset
        cov = group["ret_1d"].rolling(60, min_periods=40).cov(group["market_return"])
        var = group["market_return"].rolling(60, min_periods=40).var()
        group["beta"] = _safe_divide(cov, var)
        residual = group["ret_1d"] - group["beta"] * group["market_return"]
        group["idiovol"] = residual.rolling(60, min_periods=40).std() * np.sqrt(252.0)
        return group

    return daily.groupby("asset", group_keys=False).apply(_per_asset).reset_index(drop=True)


def read_symbol_fundamentals(history_dir: Path, symbol: str, lag_days: int) -> pd.DataFrame:
    """Merge dated statements and derive ratios under the raw export conventions.

    Availability uses notice dates or the configured reporting lag. Missing source
    components leave their derived fields unavailable; a symbol without balance
    or income files returns an empty frame. Quarterly values remain as reported
    rather than being converted from year-to-date into standalone quarters.
    """
    balance_path = history_dir / f"{symbol}.balance_sheets.csv"
    income_path = history_dir / f"{symbol}.income_statements.csv"
    if not balance_path.exists() and not income_path.exists():
        return pd.DataFrame()

    if balance_path.exists():
        bs = _read_selected_csv(balance_path, BALANCE_COLUMNS)
        bs["report_date"] = pd.to_datetime(bs.get("report_date"), errors="coerce")
        bs["notice_date_bs"] = pd.to_datetime(bs.get("notice_date_bs"), errors="coerce")
        bs = _as_numeric(bs, ["assets", "current_assets", "inventories", "current_liabilities", "equity", "parent_equity"])
    else:
        bs = pd.DataFrame(columns=["report_date"])

    if income_path.exists():
        inc = _read_selected_csv(income_path, INCOME_COLUMNS)
        inc["report_date"] = pd.to_datetime(inc.get("report_date"), errors="coerce")
        inc["notice_date_inc"] = pd.to_datetime(inc.get("notice_date_inc"), errors="coerce")
        inc = _as_numeric(inc, ["net_profit", "parent_net_profit", "operating_profit", "revenue", "costs"])
    else:
        inc = pd.DataFrame(columns=["report_date"])

    cashflow_path = history_dir / f"{symbol}.cash_flow_statements.csv"
    if cashflow_path.exists():
        cf = _read_selected_csv(cashflow_path, CASHFLOW_COLUMNS)
        cf["report_date"] = pd.to_datetime(cf.get("report_date"), errors="coerce")
        cf["notice_date_cf"] = pd.to_datetime(cf.get("notice_date_cf"), errors="coerce")
        cf = _as_numeric(cf, ["ocf"])
    else:
        cf = pd.DataFrame(columns=["report_date"])

    fund = bs.merge(inc, on="report_date", how="outer").merge(cf, on="report_date", how="outer")
    fund = fund.dropna(subset=["report_date"]).sort_values("report_date").reset_index(drop=True)
    if fund.empty:
        return fund

    notice_dates = fund[[col for col in ["notice_date_bs", "notice_date_inc", "notice_date_cf"] if col in fund.columns]]
    if notice_dates.empty:
        available_date = pd.Series(pd.NaT, index=fund.index)
    else:
        available_date = notice_dates.max(axis=1)
    fund["available_date"] = available_date.fillna(fund["report_date"] + pd.Timedelta(days=lag_days))

    g = lambda k, d=np.nan: fund.get(k, pd.Series(d, index=fund.index))
    assets = g("assets").replace(0.0, np.nan)
    current_assets = g("current_assets")
    current_liabilities = g("current_liabilities")
    inventories = g("inventories", 0.0).fillna(0.0)
    total_liab = g("total_liabilities").replace(0.0, np.nan)
    cash = g("cash")
    equity = fund.get("parent_equity", fund.get("equity", pd.Series(np.nan, index=fund.index))).replace(0.0, np.nan)
    net_profit = fund.get("parent_net_profit", fund.get("net_profit", pd.Series(np.nan, index=fund.index)))
    revenue = g("revenue").replace(0.0, np.nan)
    costs = g("costs")
    operating_profit = g("operating_profit")
    ocf = g("ocf")

    # liquidity / leverage / profitability / investment / accruals (no price needed)
    fund["currat"] = _safe_divide(current_assets, current_liabilities)
    fund["quick"] = _safe_divide(current_assets - inventories, current_liabilities)
    fund["roeq"] = _safe_divide(net_profit, equity)
    fund["agr"] = g("assets").pct_change(fill_method=None)
    fund["roa"] = _safe_divide(net_profit, assets)
    fund["gma"] = _safe_divide(revenue - costs, assets)            # gross profitability (Novy-Marx)
    fund["grossmargin"] = _safe_divide(revenue - costs, revenue)
    fund["operprof"] = _safe_divide(operating_profit, assets)
    fund["assetturn"] = _safe_divide(revenue, assets)
    fund["lev"] = _safe_divide(total_liab, assets)
    fund["cashdebt"] = _safe_divide(cash, total_liab)
    fund["acc"] = _safe_divide(net_profit - ocf, assets)           # accruals
    fund["chinv"] = _safe_divide(inventories - inventories.shift(), assets)
    fund["ocf2assets"] = _safe_divide(ocf, assets)
    # levels carried for value factors (need market cap, computed later in the monthly panel)
    fund["book_equity"] = equity
    fund["earnings"] = net_profit
    fund["revenue_lvl"] = revenue
    fund["ocf_lvl"] = ocf
    fund["asset"] = symbol
    cols = ["asset", "report_date", "available_date",
            "currat", "quick", "roeq", "agr", "roa", "gma", "grossmargin", "operprof",
            "assetturn", "lev", "cashdebt", "acc", "chinv", "ocf2assets",
            "book_equity", "earnings", "revenue_lvl", "ocf_lvl"]
    return fund[cols]


def month_end_rows(daily: pd.DataFrame, freq: str = "M") -> pd.DataFrame:
    """Keep the last trading day of each period (freq='M' monthly, 'W' weekly)."""
    daily = daily.sort_values(["asset", "date"]).copy()
    daily["period"] = daily["date"].dt.to_period(freq)
    sampled = daily.groupby(["asset", "period"], as_index=False).tail(1).copy()
    return sampled.drop(columns=["period"])


def attach_fundamentals(monthly: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Backward as-of join each asset's available accounting features to signal-date rows."""
    if fundamentals.empty:
        for col in FUND_COLS:
            monthly[col] = np.nan
        return monthly
    chunks: list[pd.DataFrame] = []
    for symbol, left in monthly.groupby("asset", sort=True):
        right = fundamentals.loc[fundamentals["asset"] == symbol].sort_values("available_date")
        left = left.sort_values("date")
        if right.empty:
            merged = left.copy()
            for col in FUND_COLS:
                merged[col] = np.nan
        else:
            merged = pd.merge_asof(
                left,
                right.drop(columns=["asset"]).sort_values("available_date"),
                left_on="date",
                right_on="available_date",
                direction="backward",
            )
        chunks.append(merged)
    return pd.concat(chunks, ignore_index=True)


def write_panel_manifest(panel_csv: Path, panel: pd.DataFrame, params: dict) -> Path:
    """Record reproducible build parameters and input identities beside the panel."""
    import hashlib
    import json
    import time
    digest = hashlib.sha256()
    with Path(panel_csv).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    span = pd.to_numeric(panel.get("forward_return_span_days"), errors="coerce")
    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "panel_csv": Path(panel_csv).name,
        "panel_sha256": digest.hexdigest(),
        "build_parameters": {k: (str(v) if isinstance(v, Path) else v)
                             for k, v in params.items()},
        "n_rows": int(len(panel)),
        "n_assets": int(panel["asset"].nunique()),
        "date_min": str(pd.to_datetime(panel["date"]).min().date()),
        "date_max": str(pd.to_datetime(panel["date"]).max().date()),
        "n_labels_valid": int(panel["forward_return"].notna().sum()),
        "n_labels_voided": int(panel["forward_return"].isna().sum()),
        "max_label_span_days_observed": (int(span.max()) if span is not None
                                         and span.notna().any() else None),
    }
    out = Path(panel_csv).with_name("panel_manifest.json")
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")
    return out


def preflight(history_dir: Path, max_symbols: int | None = None) -> int:
    """Schema preflight against docs/DATA_CONTRACT.md: report absent mapped columns per file
    (the selective reader would silently NaN them), fail only on hard requirements."""
    specs = [("daily_prices", PRICE_COLUMNS, {"date", "prices.close"}),
             ("equity_structures", EQUITY_COLUMNS, set()),
             ("balance_sheets", BALANCE_COLUMNS, set()),
             ("income_statements", INCOME_COLUMNS, set()),
             ("cash_flow_statements", CASHFLOW_COLUMNS, set())]
    symbols = load_symbols(history_dir, max_symbols=max_symbols)
    if not symbols:
        print(f"PREFLIGHT FAIL: no *.daily_prices.csv under {history_dir}")
        return 1
    hard_failures = 0
    for symbol in symbols:
        for kind, mapping, hard in specs:
            path = history_dir / f"{symbol}.{kind}.csv"
            if not path.exists():
                if kind == "daily_prices":
                    print(f"[FAIL] {path.name}: file missing")
                    hard_failures += 1
                else:
                    print(f"[warn] {path.name}: file missing (factors from it will be NaN)")
                continue
            header = set(pd.read_csv(path, nrows=0).columns)
            missing_hard = sorted(hard - header)
            missing_soft = sorted(set(mapping) - header - hard)
            if missing_hard:
                print(f"[FAIL] {path.name}: missing required columns {missing_hard}")
                hard_failures += 1
            elif missing_soft:
                print(f"[warn] {path.name}: absent optional columns {missing_soft}")
    print(f"PREFLIGHT {'FAIL' if hard_failures else 'PASS'}: {len(symbols)} symbols, "
          f"{hard_failures} hard failure(s)")
    return 1 if hard_failures else 0


def read_symbol_list(symbol_list: Path | None) -> pd.DataFrame:
    """Read optional industry metadata with UTF-8/GBK compatibility and normalize column names."""
    if symbol_list is None or not symbol_list.exists():
        return pd.DataFrame(columns=["asset", "industry"])
    try:
        frame = pd.read_csv(symbol_list, encoding="utf-8")
    except UnicodeDecodeError:
        frame = pd.read_csv(symbol_list, encoding="gbk")
    rename = {}
    for col in frame.columns:
        if col == "symbol":
            rename[col] = "asset"
        elif col.endswith(".industry"):
            rename[col] = "industry"
    frame = frame.rename(columns=rename)
    keep = [col for col in ["asset", "industry"] if col in frame.columns]
    return frame[keep].drop_duplicates("asset") if keep else pd.DataFrame(columns=["asset", "industry"])


def build_column_inventory(history_dir: Path, symbols: list[str]) -> pd.DataFrame:
    """Record available source files and columns without treating missing sources as valid data."""
    rows: list[dict[str, object]] = []
    suffixes = {
        "daily_prices": ".daily_prices.csv",
        "balance_sheets": ".balance_sheets.csv",
        "income_statements": ".income_statements.csv",
        "equity_structures": ".equity_structures.csv",
    }
    for symbol in symbols:
        for source, suffix in suffixes.items():
            path = history_dir / f"{symbol}{suffix}"
            if not path.exists():
                rows.append({"asset": symbol, "source": source, "exists": False, "n_columns": 0, "columns": ""})
                continue
            cols = list(pd.read_csv(path, nrows=0).columns)
            rows.append({"asset": symbol, "source": source, "exists": True, "n_columns": len(cols), "columns": "|".join(cols)})
    return pd.DataFrame(rows)


def build_panel(
    history_dir: Path,
    symbol_list: Path | None,
    output_dir: Path,
    max_symbols: int | None,
    start: str | None,
    end: str | None,
    forward_days: int,
    fundamental_lag_days: int,
    min_monthly_rows: int,
    freq: str = "M",
    max_label_span_days: int | None = 45,
) -> BuildResult:
    """Build the factor panel and per-symbol audits from raw A-share files.
    Apply date, coverage and label-span restrictions; output availability depends on the supplied history."""
    symbols = load_symbols(history_dir, max_symbols=max_symbols)
    daily_frames: list[pd.DataFrame] = []
    fundamental_frames: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []

    for symbol in symbols:
        try:
            prices = read_symbol_prices(history_dir, symbol)
            if start:
                prices = prices.loc[prices["date"] >= pd.Timestamp(start)]
            if end:
                prices = prices.loc[prices["date"] <= pd.Timestamp(end)]
            prices = add_price_factors(prices, forward_days=forward_days)
            if len(prices) < max(forward_days + 30, 80):
                audit_rows.append({"asset": symbol, "status": "skipped_too_few_price_rows", "daily_rows": len(prices), "monthly_rows": 0})
                continue
            equity = read_symbol_equity_structures(history_dir, symbol, lag_days=fundamental_lag_days)
            prices = attach_equity_to_daily(prices, equity)
            daily_frames.append(prices)
            fundamentals = read_symbol_fundamentals(history_dir, symbol, lag_days=fundamental_lag_days)
            if not fundamentals.empty:
                fundamental_frames.append(fundamentals)
            audit_rows.append({"asset": symbol, "status": "loaded", "daily_rows": len(prices), "monthly_rows": 0})
        except Exception as exc:  # noqa: BLE001 - audit should keep going across symbols.
            audit_rows.append({"asset": symbol, "status": f"failed:{type(exc).__name__}:{exc}", "daily_rows": 0, "monthly_rows": 0})

    if not daily_frames:
        raise RuntimeError("No usable daily price files were loaded.")

    daily = pd.concat(daily_frames, ignore_index=True)
    daily = add_market_factors(daily)
    daily = void_overlong_labels(daily, max_label_span_days)
    monthly = month_end_rows(daily, freq=freq)
    fundamentals_all = pd.concat(fundamental_frames, ignore_index=True) if fundamental_frames else pd.DataFrame()
    panel = attach_fundamentals(monthly, fundamentals_all)
    # value factors need market cap (in the monthly panel) -> compute after fundamentals are attached
    mcap = panel.get("market_cap", pd.Series(np.nan, index=panel.index)).replace(0.0, np.nan)
    panel["bm"] = _safe_divide(panel.get("book_equity", pd.Series(np.nan, index=panel.index)), mcap)
    panel["ep"] = _safe_divide(panel.get("earnings", pd.Series(np.nan, index=panel.index)), mcap)
    panel["sp"] = _safe_divide(panel.get("revenue_lvl", pd.Series(np.nan, index=panel.index)), mcap)
    panel["cfp"] = _safe_divide(panel.get("ocf_lvl", pd.Series(np.nan, index=panel.index)), mcap)
    # tradability (point-in-time, from daily data): a name is tradable at the rebalance date if it
    # traded (volume>0, i.e. not halted) and the rebalance-day move is not at the price limit (|ret|<9.7%).
    vol = pd.to_numeric(panel.get("volume", pd.Series(np.nan, index=panel.index)), errors="coerce").fillna(0.0)
    day_ret = pd.to_numeric(panel.get("ret_1d", pd.Series(np.nan, index=panel.index)), errors="coerce")
    panel["tradable"] = ((vol > 0) & (day_ret.abs() < 0.097)).astype(int)
    industries = read_symbol_list(symbol_list)
    if not industries.empty:
        panel = panel.merge(industries, on="asset", how="left")

    for col in FACTOR_COLUMNS:
        if col not in panel.columns:
            panel[col] = np.nan

    output_columns = [
        "date",
        "asset",
        "forward_return",
        "forward_return_end_date",
        "forward_return_span_days",
        "close",
        "amount",
        "volume",
        "market_cap",
        "float_mktcap",
        "market_return",
        "tradable",
        "industry",
        "report_date",
        "available_date",
        *FACTOR_COLUMNS,
    ]
    existing_columns = [col for col in output_columns if col in panel.columns]
    panel = panel[existing_columns].copy()
    # Rows with a voided (over-long) label are KEPT: the factor observation is still a
    # valid control, and dropping it here would make a fresh build disagree with a
    # migrated panel, which voids in place. Only rows with no identity are dropped.
    panel = panel.dropna(subset=["date", "asset"])

    counts = panel.groupby("asset").size()
    valid_assets = counts[counts >= min_monthly_rows].index
    panel = panel.loc[panel["asset"].isin(valid_assets)].sort_values(["date", "asset"]).reset_index(drop=True)

    audit = pd.DataFrame(audit_rows)
    monthly_counts = panel.groupby("asset").size().rename("monthly_rows_final")
    audit = audit.merge(monthly_counts, left_on="asset", right_index=True, how="left")
    audit["monthly_rows_final"] = audit["monthly_rows_final"].fillna(0).astype(int)

    inventory = build_column_inventory(history_dir, symbols)
    output_dir.mkdir(parents=True, exist_ok=True)
    return BuildResult(panel=panel, audit=audit, column_inventory=inventory)


def write_audit_report(result: BuildResult, output_dir: Path) -> None:
    """Write coverage, row counts and symbol statuses; this is a data audit, not a performance report."""
    panel = result.panel
    factor_coverage = panel[list(FACTOR_COLUMNS)].notna().mean().sort_values(ascending=False)
    status_counts = result.audit["status"].value_counts()
    lines = [
        "# A-share Factor Panel Data Audit",
        "",
        "This report is generated by `build_a_share_factor_panel.py`.",
        "",
        "## Panel Summary",
        "",
        f"- Rows: {len(panel):,}",
        f"- Assets: {panel['asset'].nunique():,}",
        f"- Dates: {panel['date'].nunique():,}",
        f"- Date range: {panel['date'].min().date()} to {panel['date'].max().date()}",
        "",
        "## Symbol Load Status",
        "",
        *[f"- `{idx}`: {value}" for idx, value in status_counts.items()],
        "",
        "## Factor Coverage",
        "",
        *[f"- `{factor}`: {coverage:.2%}" for factor, coverage in factor_coverage.items()],
        "",
        "## Caveats",
        "",
        "- Fundamental factors use `notice_date` when available; otherwise they use report date plus a conservative lag.",
        "- `beta` and `idiovol` use an equal-weight market proxy from the loaded stock universe, not an official benchmark.",
        "- `market_cap`/`float_mktcap` come from `equity_structures` (total/circulating shares) carried forward point-in-time by announcement date and multiplied by close.",
        "- `turn` is trailing-21d average daily turnover (volume / circulating shares); `size` is log(float market cap).",
        "- `dolvol` and `volume_proxy` are kept as raw liquidity proxies alongside the share-based turnover factor.",
    ]
    (output_dir / "data_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Preflight raw exports or build a dated A-share panel and its audit files.

    Validate input conventions before building factors, tradability flags and
    settlement-aware labels. Save the panel manifest beside the generated CSV.
    """
    parser = argparse.ArgumentParser(description="Build a monthly A-share factor panel from local raw CSV files.")
    parser.add_argument("--history-dir", default="data/a_shares/a_shares_history")
    parser.add_argument("--symbol-list", default="data/a_shares/symbol_list.csv")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--max-symbols", type=int, default=None,
                        help="SMOKE ONLY: truncate the filename-sorted symbol list; formal "
                             "runs use every symbol (see docs/DATA_CONTRACT.md)")
    parser.add_argument("--max-label-span-days", type=int, default=45,
                        help="void forward_return when its settlement is more than this "
                             "many CALENDAR days out (halts stretch a 21-observation "
                             "horizon into months); pass 0 to disable")
    parser.add_argument("--check-only", action="store_true",
                        help="validate every symbol's raw files against the data contract "
                             "and exit without building")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--forward-days", type=int, default=21)
    parser.add_argument("--fundamental-lag-days", type=int, default=90)
    parser.add_argument("--min-monthly-rows", type=int, default=24)
    parser.add_argument("--freq", default="M", choices=["M", "W"], help="M=monthly, W=weekly period-end sampling")
    args = parser.parse_args()

    if args.max_label_span_days < 0:
        parser.error("--max-label-span-days must be >= 0 (0 disables the cap)")
    if args.forward_days < 1:
        parser.error(f"--forward-days must be >= 1, got {args.forward_days}")
    if args.fundamental_lag_days < 0:
        # a negative lag makes statements visible BEFORE their announcement -- it does not
        # shorten the horizon, it silently breaks point-in-time semantics
        parser.error(f"--fundamental-lag-days must be >= 0, got {args.fundamental_lag_days}")
    if args.min_monthly_rows < 1:
        parser.error(f"--min-monthly-rows must be >= 1, got {args.min_monthly_rows}")
    if args.check_only:
        raise SystemExit(preflight(Path(args.history_dir), args.max_symbols))
    output_dir = Path(args.output_dir)
    result = build_panel(
        max_label_span_days=(args.max_label_span_days or None),
        history_dir=Path(args.history_dir),
        symbol_list=Path(args.symbol_list) if args.symbol_list else None,
        output_dir=output_dir,
        max_symbols=args.max_symbols,
        start=args.start,
        end=args.end,
        forward_days=args.forward_days,
        fundamental_lag_days=args.fundamental_lag_days,
        min_monthly_rows=args.min_monthly_rows,
        freq=args.freq,
    )
    result.panel.to_csv(output_dir / "a_share_factor_panel.csv", index=False, encoding="utf-8-sig")
    write_panel_manifest(output_dir / "a_share_factor_panel.csv", result.panel, vars(args))
    result.audit.to_csv(output_dir / "symbol_build_audit.csv", index=False, encoding="utf-8-sig")
    result.column_inventory.to_csv(output_dir / "available_columns_inventory.csv", index=False, encoding="utf-8-sig")
    write_audit_report(result, output_dir)
    print(f"Wrote panel to: {output_dir / 'a_share_factor_panel.csv'}")
    print(f"Rows: {len(result.panel):,}; assets: {result.panel['asset'].nunique():,}; dates: {result.panel['date'].nunique():,}")


if __name__ == "__main__":
    main()
