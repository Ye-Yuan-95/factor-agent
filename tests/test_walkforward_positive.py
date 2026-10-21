"""A walk-forward that actually forms portfolios.

The pure-noise chain test proves the gates reject noise, but it never reaches the
covariance, the solver or the turnover accounting -- which is how a covariance window
that could never fill (22 of 24 settled months on the real panel, every single month)
survived a green suite. This builds a panel with a real signal so factors pass, then
asserts the parts only a trading run exercises: a full settled window, portfolios that
solve, cash months that pay to liquidate, and one refit per year.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
pytest.importorskip("cvxpy", reason="the walk-forward solves a QP")

FACTORS = ["mom6m", "ep", "illiq", "std_turn", "bm", "dolvol"]


def _signal_panel(path: Path, n_assets: int = 60, n_months: int = 96,
                  seed: int = 7, halt_month: str | None = None,
                  signal_from: int | None = None) -> None:
    """Monthly panel whose forward return genuinely loads on two factors, with labels
    settling one month after their signal (a 21-trading-day horizon).

    `halt_month` marks one mid-stream month untradable so the run must liquidate a book
    it already holds -- without it the only cash month is the pre-inception one, whose
    turnover is zero and which therefore proves nothing about liquidation cost.
    `signal_from` leaves the earlier years pure noise. Refits train on an EXPANDING
    window, so it is the years BEFORE a refit that decide whether it succeeds: the early
    refits then genuinely fail, which is what distinguishes a yearly refit from a monthly
    retry after failure."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2016-01-31", periods=n_months, freq="ME")
    rows = []
    for date in dates:
        settle = (date + pd.offsets.MonthEnd(1)).strftime("%Y-%m-%d")
        ym = date.strftime("%Y-%m")
        alive = signal_from is None or date.year >= signal_from
        for i in range(n_assets):
            factors = {f: rng.normal() for f in FACTORS}
            signal = 0.9 * factors["mom6m"] - 0.6 * factors["ep"]
            rows.append({
                "date": date.strftime("%Y-%m-%d"), "asset": f"S{i:03d}", **factors,
                "forward_return": (0.02 * signal if alive else 0.0)
                                  + 0.01 * rng.standard_t(df=8),
                "forward_return_end_date": settle,
                "float_mktcap": float(np.exp(rng.normal(22.0, 0.6))),
                "tradable": 0 if ym == halt_month else 1})
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_config(tmp: Path, panel: Path, tag: str) -> Path:
    """Write a minimal walk-forward configuration with test-controlled paths."""
    raw = json.loads((ROOT / "a_share_big_config.json").read_text(encoding="utf-8"))
    raw["input_csv"] = str(panel)
    raw["output_dir"] = str(tmp / ("selection_" + tag))
    raw["factor_cols"] = FACTORS
    raw["family_constraints"]["factor_families"] = {
        "mom6m": "momentum", "ep": "value", "bm": "value",
        "illiq": "liquidity", "std_turn": "liquidity", "dolvol": "liquidity"}
    raw["family_constraints"]["max_per_family"] = {"momentum": 2, "value": 2,
                                                   "liquidity": 2}
    raw.setdefault("synthetic", {})["n_scenarios"] = 8
    config = tmp / ("config_" + tag + ".json")
    config.write_text(json.dumps(raw), encoding="utf-8")
    return config


def _run_walkforward(tmp: Path, panel: Path, tag: str = "out", **overrides):
    config = _write_config(tmp, panel, tag)
    argv = {"--cov-window": "24", "--w-max": "0.05", "--solver": "auto",
            "--min-test-months": "1"}
    argv.update(overrides)
    argv.setdefault("--index-csv", str(tmp / "no_benchmark.csv"))
    cmd = [sys.executable, "-m", "factor_research.hybrid_walkforward", "--panel-csv", str(panel),
           "--config", str(config), "--output-dir", str(tmp / tag)]
    for key, value in argv.items():
        cmd += [key] if value is None else [key, value]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, timeout=900)


@pytest.fixture(scope="module")
def wf(tmp_path_factory):
    """Run the synthetic positive-path walk-forward fixture once per test module."""
    tmp = tmp_path_factory.mktemp("wf_positive")
    panel = tmp / "panel.csv"
    _signal_panel(panel)
    raw = json.loads((ROOT / "a_share_big_config.json").read_text(encoding="utf-8"))
    raw["input_csv"] = str(panel)
    raw["output_dir"] = str(tmp / "selection")
    raw["factor_cols"] = FACTORS
    raw["family_constraints"]["factor_families"] = {
        "mom6m": "momentum", "ep": "value", "bm": "value",
        "illiq": "liquidity", "std_turn": "liquidity", "dolvol": "liquidity"}
    raw["family_constraints"]["max_per_family"] = {"momentum": 2, "value": 2,
                                                   "liquidity": 2}
    raw.setdefault("synthetic", {})["n_scenarios"] = 8
    config = tmp / "config.json"
    config.write_text(json.dumps(raw), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "-m", "factor_research.hybrid_walkforward", "--panel-csv", str(panel),
         "--config", str(config), "--output-dir", str(tmp / "out"),
         "--cov-window", "24", "--w-max", "0.05", "--solver", "auto",
         "--index-csv", str(tmp / "no_benchmark.csv"), "--min-test-months", "1"],
        capture_output=True, text=True, cwd=ROOT, timeout=900)
    return tmp, result


def test_the_walkforward_actually_trades(wf):
    tmp, result = wf
    assert result.returncode == 0, (result.stdout[-1200:] + result.stderr[-1200:])
    report = json.loads((tmp / "out" / "hybrid_walkforward.json").read_text(encoding="utf-8"))
    assert report["status"] == "OK"
    returns = pd.read_csv(tmp / "out" / "walkforward_returns.csv")
    traded = returns[~returns["cash_month"]]
    assert len(traded) > 0, "no month formed a portfolio -- the covariance path is dead"


def test_the_covariance_window_is_actually_full(wf):
    """Settled-history selection must supply the full requested covariance window when enough eligible months exist."""
    tmp, _ = wf
    returns = pd.read_csv(tmp / "out" / "walkforward_returns.csv")
    traded = returns[~returns["cash_month"]]
    assert (traded["cov_n_months"] == 24).all(), (
        f"observed window sizes {sorted(set(traded['cov_n_months']))}, expected 24")


def test_a_failed_refit_is_not_retried_every_month(tmp_path):
    """A failed yearly refit must remain failed until the next scheduled refit; the fixture supplies a failing year."""
    panel = tmp_path / "panel.csv"
    _signal_panel(panel, signal_from=2020)
    result = _run_walkforward(tmp_path, panel, "dead")
    assert result.returncode == 0, result.stdout[-1000:] + result.stderr[-1000:]
    refits = json.loads(
        (tmp_path / "dead" / "walkforward_refits.json").read_text(encoding="utf-8"))["refits"]

    by_year = {}
    for entry in refits:
        by_year.setdefault(entry["year"], []).append(entry)
    repeated = {y: len(v) for y, v in by_year.items() if len(v) > 1}
    assert not repeated, f"a year was refitted more than once: {repeated}"

    failures = [e for e in refits if not e["success"]]
    assert failures, "the fixture produced no failing refit -- the test would be inert"
    for entry in failures:
        assert entry["n_validated"] == 0 and entry["weights"] == {}

    returns = pd.read_csv(tmp_path / "dead" / "walkforward_returns.csv")
    years = sorted({pd.Period(ym, "M").year for ym in returns["ym"]})
    assert sorted(by_year) == years, (sorted(by_year), years)


def test_a_mid_stream_halt_liquidates_the_book_and_pays_for_it(tmp_path):
    """An untradable month after investment must liquidate the existing book and account for its turnover cost."""
    panel = tmp_path / "panel.csv"
    _signal_panel(panel, halt_month="2021-06")
    result = _run_walkforward(tmp_path, panel, "halt")
    assert result.returncode == 0, result.stdout[-1000:] + result.stderr[-1000:]
    returns = pd.read_csv(tmp_path / "halt" / "walkforward_returns.csv")

    halted = returns[returns["ym"] == "2021-06"]
    assert len(halted) == 1, "the halted month vanished from the return series"
    row = halted.iloc[0]
    assert bool(row["cash_month"]) and row["cash_reason"] == "month_not_tradable"

    prior = returns[returns["ym"] == "2021-05"].iloc[0]
    assert not bool(prior["cash_month"]), "the fixture must be invested before the halt"
    weights = pd.read_csv(tmp_path / "halt" / "walkforward_weights.csv")
    held = weights[weights["ym"] == "2021-05"]["weight"]
    assert np.isclose(held.sum(), 1.0, atol=1e-9), "the book before the halt is not full"

    assert np.isclose(row["turnover"], 1.0, atol=1e-9), row["turnover"]
    assert np.isclose(row["net"], -0.0015 * row["turnover"], atol=1e-12)
    assert weights[weights["ym"] == "2021-06"].empty, "a cash month holds nothing"

    after = returns[returns["ym"] == "2021-07"].iloc[0]
    assert not bool(after["cash_month"])
    assert np.isclose(after["turnover"], 1.0, atol=1e-9), after["turnover"]


def test_the_calendar_stays_contiguous(wf):
    tmp, _ = wf
    returns = pd.read_csv(tmp / "out" / "walkforward_returns.csv")
    months = pd.PeriodIndex(returns["ym"], freq="M")
    gaps = np.diff(months.asi8)          # PeriodIndex subtraction yields offsets, not ints
    assert (gaps == 1).all(), f"the return calendar has holes: {sorted(set(gaps))}"


def test_weights_reconstruct_the_reported_turnover(wf):
    """Full weights are written precisely so turnover can be recomputed; dropping the
    tiny ones left the monthly sums at 0.999986 and made this impossible."""
    tmp, _ = wf
    returns = pd.read_csv(tmp / "out" / "walkforward_returns.csv")
    weights = pd.read_csv(tmp / "out" / "walkforward_weights.csv")
    by_month = {ym: dict(zip(g["asset"], g["weight"]))
                for ym, g in weights.groupby("ym")}
    prev: dict[str, float] = {}
    for _, row in returns.iterrows():
        current = by_month.get(row["ym"], {})
        expected = sum(abs(current.get(a, 0.0) - prev.get(a, 0.0))
                       for a in set(current) | set(prev))
        assert np.isclose(expected, row["turnover"], atol=1e-9), row["ym"]
        prev = current


def test_segments_follow_label_settlement(wf):
    tmp, _ = wf
    returns = pd.read_csv(tmp / "out" / "walkforward_returns.csv")
    for _, row in returns.iterrows():
        if not row.get("cohort_label_end_max"):
            continue
        settle = pd.Timestamp(row["cohort_label_end_max"])
        expected = ("train" if settle <= pd.Timestamp("2019-12-31")
                    else ("validation" if settle <= pd.Timestamp("2022-12-31") else "test"))
        assert row["split"] == expected, (row["ym"], row["split"], settle)


def test_corrupt_settlement_dates_are_refused_at_the_cli(tmp_path):
    """The walk-forward parsed settlement dates directly, so a label 'settling' a decade
    before its own signal ran to OK/exit 0. It must go through the shared validator and
    refuse."""
    panel = tmp_path / "panel.csv"
    _signal_panel(panel)
    frame = pd.read_csv(panel)
    frame.loc[5000, "forward_return_end_date"] = "2005-01-31"     # long before its signal
    frame.to_csv(panel, index=False)
    result = _run_walkforward(tmp_path, panel, "corrupt")
    assert result.returncode != 0
    assert "settle before" in (result.stderr + result.stdout)


def test_a_non_positive_embargo_is_refused_at_the_cli(tmp_path):
    """A non-positive embargo cannot provide label isolation and is refused."""
    panel = tmp_path / "panel.csv"
    _signal_panel(panel)
    frame = pd.read_csv(panel).drop(columns=["forward_return_end_date"])
    frame.to_csv(panel, index=False)
    for bad in ("-30", "0"):
        result = _run_walkforward(tmp_path, panel, f"embargo_{bad}",
                                  **{"--label-embargo-days": bad})
        assert result.returncode != 0, f"embargo {bad} was accepted"
        assert "positive integer" in (result.stderr + result.stdout)


def test_a_legal_embargo_actually_produces_portfolios(tmp_path):
    """A legal embargo supplies proxy settlement dates shared by refit, splits and
    covariance estimation, so eligible history can produce portfolios."""
    panel = tmp_path / "panel.csv"
    _signal_panel(panel)
    frame = pd.read_csv(panel).drop(columns=["forward_return_end_date"])
    frame.to_csv(panel, index=False)
    result = _run_walkforward(tmp_path, panel, "embargo_ok",
                              **{"--label-embargo-days": "31"})
    assert result.returncode == 0, result.stdout[-1000:] + result.stderr[-1000:]
    report = json.loads(
        (tmp_path / "embargo_ok" / "hybrid_walkforward.json").read_text(encoding="utf-8"))
    assert report["status"] == "OK"
    assert report["manifest"]["label_availability_mode"] == "embargo_proxy"
    assert report["manifest"]["label_embargo_days"] == 31
    returns = pd.read_csv(tmp_path / "embargo_ok" / "walkforward_returns.csv")
    assert (~returns["cash_month"]).any(), "the embargo proxy still never traded"


def test_settlement_column_mode_is_recorded(wf):
    tmp, _ = wf
    report = json.loads((tmp / "out" / "hybrid_walkforward.json").read_text(encoding="utf-8"))
    assert report["manifest"]["label_availability_mode"] == "settlement_column"
    assert report["manifest"]["label_embargo_days"] is None


def test_corrupt_panel_refusal_is_clean_not_a_traceback(tmp_path):
    """Exit 1 with a full Python traceback is a crash that happens to have the right
    exit code; a refusal explains itself in one line."""
    panel = tmp_path / "panel.csv"
    _signal_panel(panel)
    frame = pd.read_csv(panel)
    frame.loc[5000, "forward_return_end_date"] = "2005-01-31"
    frame.to_csv(panel, index=False)
    result = _run_walkforward(tmp_path, panel, "corrupt2")
    assert result.returncode != 0
    assert "settle before" in (result.stderr + result.stdout)
    assert "Traceback" not in (result.stderr + result.stdout)



def _benchmark_csv(path, months=24, base=4000.0, extra_col=False, crlf=False,
                   float_fmt="{:.2f}", shuffle=False, name="CSI-300",
                   start="2020-01-31"):
    """Write a synthetic benchmark series supporting identity, coverage and corruption checks."""
    dates = pd.date_range(start, periods=months, freq="ME")
    rows = [{"date": d.strftime("%Y-%m-%d"), "benchmark": name, "secid": 1.0003,
             "close": float_fmt.format(base + 10 * i)}
            for i, d in enumerate(dates)]
    if extra_col:
        for r in rows:
            r["volume"] = 1e9
    frame = pd.DataFrame(rows)
    if shuffle:
        frame = frame.sample(frac=1.0, random_state=0)
    text = frame.to_csv(index=False)
    if crlf:
        text = text.replace("\n", "\r\n")
    Path(path).write_text(text, encoding="utf-8", newline="")


def test_canonical_benchmark_hash_ignores_the_container(tmp_path):
    """Same prices, different container -- extra columns, shuffled rows, CRLF, float
    text format -- must hash identically; a changed price must not."""
    import sys as _sys
    _sys.path.insert(0, str(ROOT))
    from factor_research.hybrid_walkforward import benchmark_series_sha256
    a = tmp_path / "a.csv"; _benchmark_csv(a)
    b = tmp_path / "b.csv"; _benchmark_csv(b, extra_col=True, crlf=True,
                                           float_fmt="{:.6f}", shuffle=True)
    assert benchmark_series_sha256(a) == benchmark_series_sha256(b)
    c = tmp_path / "c.csv"; _benchmark_csv(c, base=4001.0)
    assert benchmark_series_sha256(a) != benchmark_series_sha256(c)


def test_a_benchmark_file_without_csi300_is_refused(tmp_path):
    """A benchmark file must contain identifiable CSI-300 rows as well as columns."""
    import sys as _sys
    _sys.path.insert(0, str(ROOT))
    from factor_research.hybrid_walkforward import BenchmarkError, benchmark_series_sha256, load_index_fwd
    bad = tmp_path / "bad.csv"
    _benchmark_csv(bad, name="SSE-50")
    bad_frame = pd.read_csv(bad)
    bad_frame["secid"] = 2.0                     # neither name nor secid matches
    bad_frame.to_csv(bad, index=False)
    with pytest.raises(BenchmarkError, match="no CSI-300 rows"):
        load_index_fwd(bad)
    with pytest.raises(BenchmarkError, match="no CSI-300 rows"):
        benchmark_series_sha256(bad)


def test_a_wrong_benchmark_file_is_a_controlled_cli_refusal(tmp_path):
    panel = tmp_path / "panel.csv"
    _signal_panel(panel)
    bad = tmp_path / "bad_benchmark.csv"
    _benchmark_csv(bad, name="SSE-50")
    frame = pd.read_csv(bad); frame["secid"] = 2.0; frame.to_csv(bad, index=False)
    result = _run_walkforward(tmp_path, panel, "badbench", **{"--index-csv": str(bad)})
    assert result.returncode != 0
    assert "no CSI-300 rows" in (result.stderr + result.stdout)
    assert "Traceback" not in (result.stderr + result.stdout)


def test_a_lookalike_index_name_is_not_csi300(tmp_path):
    """Index names containing 300 cannot substitute for exact numeric CSI-300 identification."""
    import sys as _sys
    _sys.path.insert(0, str(ROOT))
    from factor_research.hybrid_walkforward import BenchmarkError, load_index_fwd
    lookalike = tmp_path / "sp300.csv"
    _benchmark_csv(lookalike, name="S&P 300")
    frame = pd.read_csv(lookalike)
    frame["secid"] = 999.0
    frame.to_csv(lookalike, index=False)
    with pytest.raises(BenchmarkError, match="no CSI-300 rows"):
        load_index_fwd(lookalike)


def test_a_name_secid_conflict_is_refused_not_resolved(tmp_path):
    """Rows NAMED CSI-300 whose secid says another index mean the file is mislabeled;
    guessing which signal to trust would silently benchmark the wrong series."""
    import sys as _sys
    _sys.path.insert(0, str(ROOT))
    from factor_research.hybrid_walkforward import (BenchmarkError, benchmark_series_sha256,
                                    load_index_fwd)
    conflicted = tmp_path / "conflict.csv"
    _benchmark_csv(conflicted)                                # named CSI-300 ...
    frame = pd.read_csv(conflicted)
    frame["secid"] = 2.0                                      # ... but secid says no
    frame.to_csv(conflicted, index=False)
    for fn in (load_index_fwd, benchmark_series_sha256):
        with pytest.raises(BenchmarkError, match="conflict"):
            fn(conflicted)


def test_secid_identifies_csi300_whatever_the_vendor_calls_it(tmp_path):
    import sys as _sys
    _sys.path.insert(0, str(ROOT))
    from factor_research.hybrid_walkforward import load_index_fwd
    vendor = tmp_path / "vendor.csv"
    _benchmark_csv(vendor, name="CSI 300 Index (vendor feed)")   # not in the whitelist
    fwd = load_index_fwd(vendor)
    assert int(fwd.notna().sum()) == 23           # 24 closes -> 23 forward returns


def test_the_series_hash_does_not_round_prices_away(tmp_path):
    """The canonical hash preserves distinct IEEE prices through shortest round-trip serialization."""
    import sys as _sys
    _sys.path.insert(0, str(ROOT))
    from factor_research.hybrid_walkforward import benchmark_series_sha256
    dates = pd.date_range("2020-01-31", periods=15, freq="ME")

    def write(path, bump):
        closes = [4000.0 + 10 * i for i in range(15)]
        closes[5] = bump
        pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "benchmark": "CSI300",
                      "secid": 1.0003, "close": [repr(c) for c in closes]}
                     ).to_csv(path, index=False)

    a = tmp_path / "a.csv"
    write(a, 4005.0)
    b = tmp_path / "b.csv"
    write(b, 4005.0000001)
    assert benchmark_series_sha256(a) != benchmark_series_sha256(b)


def test_incomplete_benchmark_coverage_is_flagged(tmp_path):
    """Twelve months IN TOTAL is not coverage: a benchmark ending mid-panel is a legal
    CSI-300 file, yet the uncovered months' idx is NaN and the benchmark-relative
    metrics quietly skip them. The manifest must say so."""
    panel = tmp_path / "panel.csv"
    _signal_panel(panel)
    short = tmp_path / "short_benchmark.csv"
    _benchmark_csv(short, months=60)          # 2020-01..2024-12: report starts 2018-01
    result = _run_walkforward(tmp_path, panel, "cov_short",
                              **{"--index-csv": str(short),
                                 "--require-full-test-benchmark": None})
    assert result.returncode == 0, result.stdout[-1000:] + result.stderr[-1000:]
    report = json.loads(
        (tmp_path / "cov_short" / "hybrid_walkforward.json").read_text(encoding="utf-8"))
    man = report["manifest"]
    assert man["benchmark_coverage"] == "INCOMPLETE_BENCHMARK"
    assert man["benchmark_missing_months"], "the gap was not recorded"
    assert "2018-01" in man["benchmark_missing_months"]
    returns = pd.read_csv(tmp_path / "cov_short" / "walkforward_returns.csv")
    n_covered = sum(man["benchmark_n_months"].values())
    assert n_covered + len(man["benchmark_missing_months"]) == len(returns)
    assert "WARNING: benchmark covers" in result.stdout
    # per-split audit: every gap sits before the test segment, so the formal flag
    # (passed above) must NOT have fired -- train/validation gaps only warn
    assert man["benchmark_coverage_by_split"]["test"] == "FULL"
    assert man["benchmark_missing_months_by_split"]["test"] == []
    assert set(man["benchmark_missing_months_by_split"]["train"]
               + man["benchmark_missing_months_by_split"]["validation"]) == set(
        man["benchmark_missing_months"])


def test_manifest_benchmark_hashes_are_the_startup_snapshot(tmp_path, monkeypatch):
    """Metrics and both benchmark hashes derive from one startup byte snapshot.

    Replacing the file after that read cannot change the recorded source identity."""
    import sys as _sys
    _sys.path.insert(0, str(ROOT))
    import factor_research.hybrid_walkforward as hw
    from factor_research.experiment_manifest import file_sha256

    panel = tmp_path / "panel.csv"
    _signal_panel(panel)
    bench = tmp_path / "benchmark.csv"
    _benchmark_csv(bench, months=98, start="2016-01-31")      # covers every report month
    orig_raw = file_sha256(bench)
    orig_series = hw.benchmark_series_sha256(bench)

    real = hw.load_benchmark_snapshot

    def snapshot_then_swap(path):
        snap = real(path)
        _benchmark_csv(bench, months=98, start="2016-01-31", base=9999.0)
        return snap

    monkeypatch.setattr(hw, "load_benchmark_snapshot", snapshot_then_swap)
    config = _write_config(tmp_path, panel, "swap")
    out = tmp_path / "swap_out"
    monkeypatch.setattr(sys, "argv", [
        "factor_research.hybrid_walkforward", "--panel-csv", str(panel), "--config", str(config),
        "--output-dir", str(out), "--cov-window", "24", "--w-max", "0.05",
        "--solver", "auto", "--index-csv", str(bench), "--min-test-months", "1"])
    try:
        hw.main()
    finally:
        hw._SETTLE_MAX.clear()

    report = json.loads((out / "hybrid_walkforward.json").read_text(encoding="utf-8"))
    assert report["status"] == "OK"
    man = report["manifest"]
    assert file_sha256(bench) != orig_raw, "the fixture failed to change the file"
    assert man["benchmark_csv_sha256"] == orig_raw
    assert man["benchmark_series_sha256"] == orig_series
    assert man["benchmark_coverage"] == "FULL"
    assert man["benchmark_coverage_by_split"] == {
        "train": "FULL", "validation": "FULL", "test": "FULL"}
    assert man["benchmark_missing_months"] == []
    returns = pd.read_csv(out / "walkforward_returns.csv")
    assert sum(man["benchmark_n_months"].values()) == len(returns)


def test_an_unreadable_benchmark_file_is_a_controlled_refusal(tmp_path):
    """Invalid UTF-8, empty files and directory inputs produce controlled benchmark
    refusals rather than decoder, parser or filesystem tracebacks."""
    import sys as _sys
    _sys.path.insert(0, str(ROOT))
    from factor_research.hybrid_walkforward import BenchmarkError, load_benchmark_snapshot
    notutf8 = tmp_path / "notutf8.csv"
    notutf8.write_bytes(b"\xff\xfe\x00\x01date,benchmark\n\x80\x81")
    empty = tmp_path / "empty.csv"
    empty.write_bytes(b"")
    adir = tmp_path / "a_directory"
    adir.mkdir()
    for path in (notutf8, empty, adir):
        with pytest.raises(BenchmarkError, match="cannot be read"):
            load_benchmark_snapshot(path)


def test_an_unreadable_benchmark_file_refuses_at_the_cli_without_traceback(tmp_path):
    panel = tmp_path / "panel.csv"
    _signal_panel(panel)
    bad = tmp_path / "empty_benchmark.csv"
    bad.write_bytes(b"")
    result = _run_walkforward(tmp_path, panel, "unreadable", **{"--index-csv": str(bad)})
    assert result.returncode != 0
    assert "cannot be read" in (result.stderr + result.stdout)
    assert "Traceback" not in (result.stderr + result.stdout)


def test_the_secid_match_is_exact_not_a_tolerance_band(tmp_path):
    """A nearby secid must be refused; CSI-300 identification requires exact equality."""
    import sys as _sys
    _sys.path.insert(0, str(ROOT))
    from factor_research.hybrid_walkforward import BenchmarkError, load_index_fwd
    near = tmp_path / "near.csv"
    _benchmark_csv(near, name="almost")
    frame = pd.read_csv(near)
    frame["secid"] = 1.0003005
    frame.to_csv(near, index=False)
    with pytest.raises(BenchmarkError, match="secid"):
        load_index_fwd(near)


def test_a_csi300_name_alone_is_not_identification(tmp_path):
    """The identity contract: the exact secid identifies, the name documents. A file
    whose secid column is empty is refused even when every row is named CSI-300."""
    import sys as _sys
    _sys.path.insert(0, str(ROOT))
    from factor_research.hybrid_walkforward import BenchmarkError, load_index_fwd
    named = tmp_path / "named.csv"
    _benchmark_csv(named)                          # canonical name ...
    frame = pd.read_csv(named)
    frame["secid"] = np.nan                        # ... but nothing to identify by
    frame.to_csv(named, index=False)
    with pytest.raises(BenchmarkError, match="secid"):
        load_index_fwd(named)


def test_conflicting_closes_on_one_date_are_refused(tmp_path):
    """Two different closes for the same date silently kept whichever row the vendor
    export ordered last. Identical duplicates are a legitimate export artifact and
    dedupe cleanly; conflicting ones are refused."""
    import sys as _sys
    _sys.path.insert(0, str(ROOT))
    from factor_research.hybrid_walkforward import BenchmarkError, benchmark_series_sha256
    clean = tmp_path / "clean.csv"
    _benchmark_csv(clean)
    frame = pd.read_csv(clean)
    frame = pd.concat([frame, frame.iloc[[3]]])    # same date, same close
    dup_same = tmp_path / "dup_same.csv"
    frame.to_csv(dup_same, index=False)
    assert benchmark_series_sha256(dup_same) == benchmark_series_sha256(clean)
    frame.iloc[-1, frame.columns.get_loc("close")] = 9999.0
    dup_conflict = tmp_path / "dup_conflict.csv"
    frame.to_csv(dup_conflict, index=False)
    with pytest.raises(BenchmarkError, match="conflicting closes"):
        benchmark_series_sha256(dup_conflict)


def test_require_full_test_benchmark_refuses_a_gappy_test_window(tmp_path):
    """--require-full-test-benchmark is the formal mode. Without it the same gappy run
    writes status OK / exit 0, which is right for exploration and wrong for a formal
    benchmark-relative claim; with it, an uncovered TEST month is a controlled
    non-zero exit with status INCOMPLETE_TEST_BENCHMARK."""
    panel = tmp_path / "panel.csv"
    _signal_panel(panel)
    ends_2022 = tmp_path / "ends_2022.csv"
    _benchmark_csv(ends_2022, months=78, start="2016-01-31")   # closes stop 2022-06
    plain = _run_walkforward(tmp_path, panel, "plain", **{"--index-csv": str(ends_2022)})
    assert plain.returncode == 0, plain.stdout[-800:]           # exploratory: warn only
    result = _run_walkforward(tmp_path, panel, "formal",
                              **{"--index-csv": str(ends_2022),
                                 "--require-full-test-benchmark": None})
    assert result.returncode != 0
    out = result.stdout + result.stderr
    assert "INCOMPLETE_TEST_BENCHMARK" in out
    assert "Traceback" not in out
    report = json.loads(
        (tmp_path / "formal" / "hybrid_walkforward.json").read_text(encoding="utf-8"))
    assert report["status"] == "INCOMPLETE_TEST_BENCHMARK"
    assert report["manifest"]["benchmark_coverage_by_split"]["test"] == "INCOMPLETE_BENCHMARK"
    assert report["manifest"]["require_full_test_benchmark"] is True


def test_require_full_test_benchmark_needs_a_benchmark_file(tmp_path):
    """An ABSENT file is a declared no-benchmark run -- unless the caller declared a
    formal benchmark-relative run, in which case it is a fail-fast refusal."""
    panel = tmp_path / "panel.csv"
    _signal_panel(panel)
    result = _run_walkforward(tmp_path, panel, "nofile",
                              **{"--require-full-test-benchmark": None})
    assert result.returncode != 0
    assert "no benchmark file" in (result.stdout + result.stderr)
    assert "Traceback" not in (result.stdout + result.stderr)
