# Result fields and interpretation

## Incremental evidence

Both the core pipeline and the sequential experiment gate require at least five
usable complete-case cross-sectional fits and finite statistics. Missing evidence
is not a pass. Five fits is the minimum estimability rule, not a power guarantee.

| Field | Definition |
|---|---|
| `fm_n_date_groups` | Nonempty input date groups |
| `fm_n_periods` | Successful finite-coefficient fits |
| `fm_n_complete_rows` | Complete observations used by those fits |
| `fm_min_cross_section_obs`, `fm_max_cross_section_obs` | Usable fit-size range |
| `fm_skipped_insufficient_obs`, `fm_skipped_fit_failed` | Excluded fit counts |
| `fm_status` | Not tested, insufficient, nonfinite, below threshold or passed |

FactorCards expose `incremental_n_periods`, `incremental_n_complete_rows` and
`incremental_sample_status`. The user report displays sample adequacy and refusal
reasons. Insufficient evidence is not proof that a candidate lacks economic value.

## Threshold experiments

Scan, split-sample, calibration and transfer reporters save total, screen-survivor,
finite-statistic, invalid-estimate and selected populations. Counts refer to
factor/panel pairs, not independent trials.

| Metric | Numerator | Denominator |
|---|---|---|
| `conditional_size` | Selected false factors | False survivors with finite FM statistics |
| `conditional_power` | Selected true factors | True survivors with finite FM statistics |
| `full_fpr` | Selected false factors | All injected false factors before screening |
| `full_power` | Selected true factors | All injected true factors before screening |

`n_fake_reaching` / `n_true_reaching` include all survivors; `*_finite` fields count
estimable statistics. Screening rejections and invalid estimates remain in full
pipeline denominators. With no finite survivors, conditional rates are undefined;
if all injected true factors are rejected, full power is zero.

`sample_accounting.json` uses schema version 2 and records per-fit date counts,
complete rows, control count and estimator status. Split-sample fields carry mode
suffixes. Calibration/transfer fields identify their cutoffs. Semi-synthetic
reports also include calendar months and finite IC counts per factor. Raw dates,
calendar months, fitted dates and independent samples are different quantities.

## Closed-loop empirical FDP

| Field | Definition |
|---|---|
| `run_fdp` | False accepts / accepts in a run; zero for empty discoveries |
| `pooled_empirical_fdp` | Total false accepts / total accepts; null if every run is empty |
| `mean_run_fdp` | Mean FDP across all alternative runs, including empty ones |
| `mean_run_fdp_given_acceptance` | Mean across nonempty alternative runs |
| `all_run_fdp_denominator` | Number of all alternative runs |
| `conditional_run_fdp_denominator` | Number of nonempty alternative runs |
| `n_zero_accept_alt_runs` | Number of empty alternative runs |

`realized_fdr` is a display alias for per-run FDP; `mean_run_realized_fdr` aliases
the all-run mean. Point acceptance uses the unrounded pooled ratio. An empty
pooled discovery set cannot establish pooled-FDP acceptance.

FDR means the expectation of per-run FDP. A pooled ratio, one set of sample means
or candidate-level Wilson bounds do not establish a population FDR guarantee.
Wilson bounds require independent-candidate assumptions and are not seed-cluster
confidence bounds. Match schema, truth definition and unit of aggregation when
comparing runs.

## Truth, identity and limits

Linear experiments define truth by nonzero injected coefficients. Mechanism
experiments use the protocol's candidate loading threshold; DSL closed-loop
recovery uses the stated joint-truth rank-space criterion. Injection `IC` is a
coefficient-scale parameter, not an enforced realized rank correlation.

Research experiment manifests bind actual inputs, configs, source hashes, factor
order and parameters. The agent's L1 manifest records run state and budgets;
`config.snapshot.json` records the configuration and CLI arguments, and cards
record available data/commit identities. The L1 synthetic tutorial does not
archive its calibration CSV or generated market inside the run. Keep the exact
calibration input, its hash and the source/environment record for numerical
reproduction; a run folder or seed alone is insufficient.

A completed run means execution succeeded, not that the hypothesis was confirmed.
DSR, control-span and investment-increment decisions, and synthetic truth
classifications, use unrounded statistics. Human-readable formatting may round
those values; use saved numeric fields and pass flags when checking a boundary.
A certificate applies only to its specified selection process and calibration.
Independent investment evidence requires appropriate held-out data and trading
assumptions; engineering tests do not supply that evidence.
