# Data contract for the real-panel builder (A-share)

> **Scope**: the real-data chain in this repository is **A-share specific** -- the builder
> assumes A-share raw exports, marks a name untradable when the rebalance-day move sits at
> the ±9.7% price limit, and the walk-forward report benchmarks against CSI-300. Running
> US-equity experiments requires a separate data adapter (corporate-action adjustment,
> delisting returns, CRSP/Compustat-style point-in-time rules, a US benchmark); this is additional
> implementation work, not a configuration change. The synthetic/dummy flows in the
> README need none of this.

`factor_research/build_a_share_factor_panel.py` reads one directory of per-symbol raw CSVs plus an
optional symbol list and emits `a_share_factor_panel.csv`. Column names below are the
**raw export names** (the builder renames them internally). Columns missing from a file
are silently skipped by the selective reader, so factors that need them come out NaN --
run `--check-only` first: it validates every symbol's files against this contract and
reports which mapped columns are absent, failing only on the hard requirements.

## Per-symbol input files (in `--history-dir`)

| File | Raw columns consumed | Hard-required |
|---|---|---|
| `S.daily_prices.csv` | `date`, `prices.open`, `prices.close`, `prices.high`, `prices.low`, `prices.amount`, `prices.volume` | `date`, `prices.close` |
| `S.equity_structures.csv` | `date`, `notice_date`, `shares.total`, `shares.circulating` | file optional; without it every size/turnover factor is NaN |
| `S.balance_sheets.csv` | `date`, `notice_date`, `balance_sheet.assets`, `.assets.current`, `.assets.current.inventories`, `.assets.current.cash`, `.liab`, `.liab.current`, `.equity`, `.equity.parent_interests` | file optional |
| `S.income_statements.csv` | `date`, `notice_date`, `income_statement.profit`, `.parent_interests`, `.profit.operating`, `.profit.operating.income.revenue`, `.profit.operating.expenses.costs` | file optional |
| `S.cash_flow_statements.csv` | `date`, `notice_date`, `cash_flow_statement.change.operating` | file optional |

Symbol list (`--symbol-list`, optional): CSV with `asset` and `industry` columns;
missing file just leaves the industry tag empty.

Benchmark CSV (`--index-csv` of `factor_research/hybrid_walkforward.py`): columns
`date, benchmark, secid, close`. CSI-300 rows are identified by the EXACT parsed
`secid == 1.0003` -- no tolerance band, and the `benchmark` name is documentation, not
identification: a file whose only claim to CSI-300 is a name (`沪深300` included, let
alone one merely containing "300") is refused, and rows NAMED CSI-300 whose secid says
another index are refused as a name/secid conflict. A date carrying two different
closes is refused rather than one being picked (identical duplicates dedupe cleanly);
an unreadable file (empty, wrong encoding, not CSV, a directory) is the same one-line
controlled refusal as any other present-but-unusable file. The file is read **once at
startup** as a byte snapshot: the forward monthly returns the run trades against, the
raw-bytes hash and the canonical series hash in the manifest all describe those same
bytes. An absent file is a declared no-benchmark run (`benchmark_index: null` /
`benchmark_available: false`), never CSI-300 with NaN metrics.

Obtain daily CSI-300 (沪深300, secid 1.0003) closes from a licensed A-share data
vendor or public terminal and save them with the schema above. Benchmark data is
not distributed with the repository. Record the hashes emitted by each run:
`benchmark_csv_sha256` identifies the input bytes; `benchmark_series_sha256`
identifies the canonical CSI-300 series. The latter filters the CSI-300 rows,
sorts `date,close` and hashes round-trip (`repr`) text of each close, allowing
comparison of parsed prices across different columns, formatting and line endings.
The manifest also
records `benchmark_coverage`, per-split `benchmark_coverage_by_split` /
`benchmark_n_months` / `benchmark_missing_months_by_split` and the flat
`benchmark_missing_months`: twelve months of index in total does not mean the months
the report trades are covered, and benchmark-relative metrics (IR, `sharpe_index`)
skip missing months. Exploratory runs only warn; a FORMAL benchmark-relative claim
must run with `--require-full-test-benchmark`, which exits non-zero (status
`INCOMPLETE_TEST_BENCHMARK`) when any test month lacks a benchmark return, and
fail-fast when no benchmark file was provided at all.

## Units and conventions the builder assumes

- **Prices**: raw quotes in CNY as exported. The builder does not apply corporate-action
  adjustment; momentum/volatility factors inherit whatever adjustment the export has.
  The tradability flag additionally requires `prices.volume > 0` (not halted) and a
  rebalance-day absolute move below 9.7% (off the limit board).
- **Share counts** (`shares.total`, `shares.circulating`): in shares.
  `shares.circulating × close` is the float market cap used for value weights.
- **Statement values**: used **as reported**. Chinese quarterly exports are typically
  year-to-date; the builder does not difference them into single quarters, so ratio
  factors are YTD-convention ratios.

## Point-in-time rules enforced

- A statement (or share-structure change) becomes visible on its `notice_date`; when
  that is missing, on `report date + --fundamental-lag-days` (default 90).
- `forward_return` is `close[t + --forward-days] / close[t] - 1` (default 21 trading
  days), strictly after the sampling date.
- A symbol needs `--min-monthly-rows` (default 24) sampled periods to enter the panel.

## Label settlement (`forward_return_end_date`)

Every built panel carries `forward_return_end_date`: the date whose close settles that
row's `forward_return`. Training cutoffs compare against **this** column, not the signal
date. Panels without this column must be migrated with
`factor_research/migrate_label_end_dates.py` (which verifies the horizon against the
stored returns) or run with an explicit `--label-embargo-days` covering the full horizon.

## Label span (`forward_return_span_days`)

`forward_return` is the **N-th next observation**, not N trading days later: through a
halt it can span months. A year-long "monthly" return is a different random variable and must
not share a cross-section with one-month returns, nor be compared against a one-month
index return.

The builder records `forward_return_span_days` and voids `forward_return` (and its
settlement date, so the row can never enter a training window) beyond
`--max-label-span-days`, default 45 calendar days. **The row is kept**: the factor
observation is still a valid control. `factor_research/migrate_label_end_dates.py` applies the identical
rule, so a migrated panel and a fresh build agree.

Both writers emit `panel_manifest.json` beside the panel recording the build parameters,
date range, panel SHA-256, and the valid/voided label counts -- without it a panel's
horizon and universe cannot be recovered later.

## Rules NOT enforced (caller's responsibility)

- **Survivorship**: the panel contains exactly the symbols present in the raw directory.
  If the export omits delisted names, the panel is survivorship-biased; there is no
  delisting-return imputation.
- **Historical index membership**: the universe is the export-time listing, not
  point-in-time constituents.

## Universe truncation

`--max-symbols` truncates the **filename-sorted** symbol list and is for smoke tests
only; the default is *all symbols*. Formal experiments must not pass it: taking the
first N of a sorted listing selects by listing code, which is not random in A-shares.
