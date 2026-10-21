"""The panel builder against docs/DATA_CONTRACT.md: a tiny raw fixture builds end-to-end,
and the preflight reports contract violations before any build work happens."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _write_symbol(history: Path, symbol: str, n_days: int = 420, seed: int = 0) -> None:
    """Write one small raw-data symbol fixture matching the builder input contract."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n_days)
    close = 10.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n_days)))
    pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "prices.open": close * (1 + rng.normal(0, 0.003, n_days)),
        "prices.close": close,
        "prices.high": close * 1.01, "prices.low": close * 0.99,
        "prices.amount": rng.uniform(1e7, 5e7, n_days),
        "prices.volume": rng.uniform(1e6, 5e6, n_days),
    }).to_csv(history / f"{symbol}.daily_prices.csv", index=False)
    quarters = pd.date_range("2019-12-31", periods=8, freq="QE")
    pd.DataFrame({
        "date": quarters.strftime("%Y-%m-%d"),
        "notice_date": (quarters + pd.Timedelta(days=30)).strftime("%Y-%m-%d"),
        "shares.total": 1e9, "shares.circulating": 7e8,
    }).to_csv(history / f"{symbol}.equity_structures.csv", index=False)
    pd.DataFrame({
        "date": quarters.strftime("%Y-%m-%d"),
        "notice_date": (quarters + pd.Timedelta(days=45)).strftime("%Y-%m-%d"),
        "balance_sheet.assets": 5e9, "balance_sheet.assets.current": 2e9,
        "balance_sheet.assets.current.inventories": 5e8,
        "balance_sheet.assets.current.cash": 5e8,
        "balance_sheet.liab": 2e9, "balance_sheet.liab.current": 1e9,
        "balance_sheet.equity": 3e9, "balance_sheet.equity.parent_interests": 2.8e9,
    }).to_csv(history / f"{symbol}.balance_sheets.csv", index=False)
    pd.DataFrame({
        "date": quarters.strftime("%Y-%m-%d"),
        "notice_date": (quarters + pd.Timedelta(days=45)).strftime("%Y-%m-%d"),
        "income_statement.profit": 2e8, "income_statement.parent_interests": 1.8e8,
        "income_statement.profit.operating": 2.4e8,
        "income_statement.profit.operating.income.revenue": 2e9,
        "income_statement.profit.operating.expenses.costs": 1.5e9,
    }).to_csv(history / f"{symbol}.income_statements.csv", index=False)
    pd.DataFrame({
        "date": quarters.strftime("%Y-%m-%d"),
        "notice_date": (quarters + pd.Timedelta(days=45)).strftime("%Y-%m-%d"),
        "cash_flow_statement.change.operating": 3e8,
    }).to_csv(history / f"{symbol}.cash_flow_statements.csv", index=False)


@pytest.fixture()
def fixture_dir(tmp_path):
    """Create a temporary raw-data directory with explicit availability and units cases."""
    history = tmp_path / "history"
    history.mkdir()
    # Twelve assets give the screen enough observations for finite cross-sectional
    # ICs, so leakage checks exercise label purging rather than empty statistics.
    symbols = [f"6000{i:02d}.XSHG" for i in range(1, 13)]
    for i, symbol in enumerate(symbols):
        _write_symbol(history, symbol, seed=i)
    pd.DataFrame({"asset": symbols,
                  "industry": ["sector%d" % (i % 4) for i in range(len(symbols))]}).to_csv(
        tmp_path / "symbol_list.csv", index=False)
    return tmp_path


def _builder(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "factor_research.build_a_share_factor_panel", *args],
        capture_output=True, text=True, cwd=ROOT, timeout=300)


def test_preflight_passes_on_a_contract_conforming_fixture(fixture_dir):
    result = _builder(["--history-dir", str(fixture_dir / "history"), "--check-only"])
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PREFLIGHT PASS: 12 symbols" in result.stdout


def test_preflight_fails_when_a_hard_column_is_missing(fixture_dir):
    prices = fixture_dir / "history" / "600001.XSHG.daily_prices.csv"
    frame = pd.read_csv(prices).drop(columns=["prices.close"])
    frame.to_csv(prices, index=False)
    result = _builder(["--history-dir", str(fixture_dir / "history"), "--check-only"])
    assert result.returncode == 1
    assert "missing required columns" in result.stdout
    assert "prices.close" in result.stdout


def test_builder_builds_the_fixture_panel(fixture_dir, tmp_path):
    out = tmp_path / "out"
    result = _builder(["--history-dir", str(fixture_dir / "history"),
                       "--symbol-list", str(fixture_dir / "symbol_list.csv"),
                       "--output-dir", str(out), "--start", "2020-01-01",
                       "--min-monthly-rows", "3", "--forward-days", "5"])
    assert result.returncode == 0, result.stdout[-800:] + result.stderr[-800:]
    panel = pd.read_csv(out / "a_share_factor_panel.csv")
    assert len(panel) > 0
    for col in ("date", "asset", "forward_return", "float_mktcap", "tradable",
                "mom6m", "ep", "bm", "illiq"):
        assert col in panel.columns, col
    assert len(set(panel["asset"])) == 12


def test_builder_emits_label_end_dates(fixture_dir, tmp_path):
    out = tmp_path / "out"
    result = _builder(["--history-dir", str(fixture_dir / "history"),
                       "--output-dir", str(out), "--start", "2020-01-01",
                       "--min-monthly-rows", "3", "--forward-days", "5"])
    assert result.returncode == 0
    panel = pd.read_csv(out / "a_share_factor_panel.csv")
    assert "forward_return_end_date" in panel.columns
    dates = pd.to_datetime(panel["date"])
    ends = pd.to_datetime(panel["forward_return_end_date"])
    settled = ends.notna()
    assert (ends[settled] > dates[settled]).all()       # labels settle strictly later


def test_builder_rejects_degenerate_parameters(fixture_dir):
    for args, needle in ((["--forward-days", "0"], "forward-days"),
                         (["--fundamental-lag-days", "-1"], "fundamental-lag-days"),
                         (["--min-monthly-rows", "0"], "min-monthly-rows")):
        result = _builder(["--history-dir", str(fixture_dir / "history"),
                           "--check-only", *args])
        assert result.returncode != 0
        assert needle in (result.stderr + result.stdout)


def test_selection_ignores_tampered_post_cutoff_raw_prices(fixture_dir, tmp_path):
    """Post-cutoff raw prices cannot change purged factor selection.

    Twelve assets supply valid IC dates. The positive control confirms that tampering
    changes overlapping labels and naive date-only statistics; the settlement-purged
    selection must then remain identical."""
    import dataclasses as _dc
    import json as _json
    import sys as _sys
    _sys.path.insert(0, str(ROOT))
    from factor_research.factor_validity_pipeline import (load_config, purge_label_overlap, run_pipeline,
                                          single_factor_screen)

    cutoff = pd.Timestamp("2020-12-31")
    factors = ["mom1m", "mom6m", "volatility", "size"]

    def build(tag):
        out = tmp_path / f"panel_{tag}"
        result = _builder(["--history-dir", str(fixture_dir / "history"),
                           "--output-dir", str(out), "--start", "2020-01-01",
                           "--min-monthly-rows", "3", "--forward-days", "5"])
        assert result.returncode == 0, result.stderr[-500:]
        return out / "a_share_factor_panel.csv"

    def select(panel_csv, tag):
        raw = _json.loads((ROOT / "a_share_big_config.json").read_text(encoding="utf-8"))
        raw["input_csv"] = str(panel_csv)
        raw["output_dir"] = str(tmp_path / f"sel_{tag}")
        raw["factor_cols"] = factors
        raw["family_constraints"]["factor_families"] = {
            "mom1m": "momentum", "mom6m": "momentum",
            "volatility": "risk", "size": "liquidity"}
        raw["family_constraints"]["max_per_family"] = {"momentum": 2, "risk": 2,
                                                       "liquidity": 2}
        raw.setdefault("synthetic", {})["n_scenarios"] = 6
        cfg_path = tmp_path / f"cfg_{tag}.json"
        cfg_path.write_text(_json.dumps(raw), encoding="utf-8")
        cfg = _dc.replace(load_config(cfg_path), train_end=str(cutoff.date()))
        run_pipeline(cfg)
        return tmp_path / f"sel_{tag}", cfg

    base_panel = build("base")
    base_dir, base_cfg = select(base_panel, "base")

    for prices in sorted((fixture_dir / "history").glob("*.daily_prices.csv")):
        frame = pd.read_csv(prices)
        late = pd.to_datetime(frame["date"]) > cutoff
        assert late.any(), "fixture must span the cutoff for this test to mean anything"
        for col in ("prices.open", "prices.close", "prices.high", "prices.low"):
            frame.loc[late, col] = frame.loc[late, col] * 1.7 + 3.0
        frame.to_csv(prices, index=False)
    tampered_panel = build("tampered")

    # --- positive control 1: the leak channel is live. Rows dated at or before the
    # cutoff whose labels settle after it must have MOVED when post-cutoff prices moved.
    before = pd.read_csv(base_panel, parse_dates=["date", "forward_return_end_date"])
    after = pd.read_csv(tampered_panel, parse_dates=["date", "forward_return_end_date"])
    key = ["date", "asset"]
    merged = before.merge(after, on=key, suffixes=("_base", "_tampered"))
    crossing = ((merged["date"] <= cutoff)
                & (merged["forward_return_end_date_base"] > cutoff))
    assert crossing.any(), "no pre-cutoff row has a label settling after it"
    moved = (merged.loc[crossing, "forward_return_base"]
             != merged.loc[crossing, "forward_return_tampered"])
    assert moved.any(), "tampering did not change any crossing label -- test is inert"

    # --- positive control 2: without the purge, the statistics DO change. This is what
    # the degenerate version never checked.
    def naive_screen(panel_csv):
        frame = pd.read_csv(panel_csv, parse_dates=["date"])
        naive = frame.loc[frame["date"] <= cutoff]
        return single_factor_screen(naive, base_cfg)

    naive_base, naive_tampered = naive_screen(base_panel), naive_screen(tampered_panel)
    assert not naive_base.equals(naive_tampered), (
        "date-only truncation produced identical statistics -- the fixture cannot "
        "demonstrate the leak, so the purge assertion below would prove nothing")

    # --- and the purge removes exactly those rows
    frame = pd.read_csv(tampered_panel, parse_dates=["date", "forward_return_end_date"])
    _, info = purge_label_overlap(frame, cutoff)
    assert info["n_purged_label_overlap"] > 0
    assert pd.Timestamp(info["max_train_label_end"]) <= cutoff

    # --- the actual requirement: the purged selection does not move
    tampered_dir, _ = select(tampered_panel, "tampered")
    for name in ("single_factor_screen", "sparse_weights", "final_factor_selection"):
        pd.testing.assert_frame_equal(pd.read_csv(base_dir / f"{name}.csv"),
                                      pd.read_csv(tampered_dir / f"{name}.csv"))

def test_overlong_labels_are_voided_not_silently_mixed_in():
    """The Nth next price observation may span a long trading halt.

    Labels exceeding the declared calendar span must be unavailable rather than mixed
    with the intended short-horizon cross-section."""
    import numpy as np
    import pandas as pd
    from factor_research.build_a_share_factor_panel import void_overlong_labels
    frame = pd.DataFrame({
        "forward_return": [0.01, 0.02, 0.03],
        "forward_return_end_date": pd.to_datetime(["2020-02-03", "2021-06-01", "2020-02-05"]),
        "forward_return_span_days": [31, 400, 33]})
    out = void_overlong_labels(frame.copy(), 45)
    assert np.isnan(out.loc[1, "forward_return"])           # 400-day label voided
    assert pd.isna(out.loc[1, "forward_return_end_date"])   # and can never enter training
    assert out.loc[[0, 2], "forward_return"].tolist() == [0.01, 0.03]
    unchanged = void_overlong_labels(frame.copy(), None)
    assert unchanged["forward_return"].tolist() == [0.01, 0.02, 0.03]

def test_migration_rejects_a_non_finite_tolerance(fixture_dir, tmp_path):
    """Every comparison against NaN is False, so a NaN tolerance reported mismatched=0
    for any panel at all -- the horizon check that makes this script trustworthy simply
    vanished. +Inf did the same."""
    import subprocess as _sp
    import sys as _sys
    panel = tmp_path / "panel.csv"
    pd.DataFrame({"date": ["2020-06-30"], "asset": ["600001.XSHG"],
                  "forward_return": [9.99]}).to_csv(panel, index=False)
    for bad in ("nan", "inf", "-1"):
        result = _sp.run(
            [_sys.executable, "-m", "factor_research.migrate_label_end_dates",
             "--panel-csv", str(panel), "--history-dir", str(fixture_dir / "history"),
             "--tolerance", bad],
            capture_output=True, text=True, cwd=ROOT, timeout=300)
        assert result.returncode != 0, f"--tolerance {bad} was accepted"
        assert "tolerance" in (result.stderr + result.stdout)
    for bad in ("nan", "2.0"):
        result = _sp.run(
            [_sys.executable, "-m", "factor_research.migrate_label_end_dates",
             "--panel-csv", str(panel), "--history-dir", str(fixture_dir / "history"),
             "--max-mismatch-rate", bad],
            capture_output=True, text=True, cwd=ROOT, timeout=300)
        assert result.returncode != 0, f"--max-mismatch-rate {bad} was accepted"
