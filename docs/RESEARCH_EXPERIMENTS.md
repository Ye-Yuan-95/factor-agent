# Research experiments

Use the eleven optional modules below to run incremental-gate experiments on
your calibration inputs. Record the inputs and configuration with each run.
No LLM, API key or commercial
solver is needed. Install with `python -m pip install -e "."` from a checkout.

## Inputs and commands

Create a workflow-only noise panel without overwriting research data:

```sh
python -m factor_research.make_dummy_panel --out data_research_demo/panel.csv
python -m synthetic.p1_validate --panel-csv data_research_demo/panel.csv --config a_share_baseline_config.json --output-dir outputs/research/demo_p1 --reps 2 --n-assets 80 --n-months 30 --survival-B 5
```

For substantive research, supply a documented calibration dataset.
Inputs require literal `date`, `asset`, `forward_return` and every
configured factor column. Missing exposure values are allowed; missing columns,
duplicate keys, constant columns and invalid budgets are rejected. Every factor
needs an explicit family; cross-family scenarios require at least four families.
Use the supplied 44-factor configuration to run the default scenarios.

Replace the module below and choose a fresh `--output-dir`. Use `--help` for its
parameters. Direct-script invocation from a foreign working directory is supported
with explicit absolute paths. Small repetition counts check execution only.

| Module (`python -m synthetic.NAME`) | Research question | Main output | Budget controls |
|---|---|---|---|
| `p1_validate` | Sequential gates under Gen-2 structure | `p1_four_gate_v2.csv`, `p1_validate.md` | `--reps`, `--n-assets`, `--n-months` |
| `semi_synthetic` | Supplied exposures with permuted/injected returns | `semi_synthetic.csv`, `.md` | `--reps`; input determines dimensions |
| `incremental_robustness` | Signal strength, truth structure and tails | `incremental_robustness.csv` | `--reps`, dimensions |
| `dgp_sensitivity` | Tails, clustering and sample size | `dgp_sensitivity.csv`, `.md` | `--reps`; fixed 12-cell grid |
| `omitted_factor_experiment` | Omitted true factor and proxy selection | `omitted_factor.csv`, `.json` | `--reps`, dimensions, `--strength` |
| `tau_size_sweep` | Conditional size and whole-pipeline FPR/power | `tau_size_sweep.csv` | `--reps`, dimensions, `--taus` |
| `split_sample_incremental` | Same-sample versus split-sample testing | `split_sample.csv` | `--reps`, dimensions, `--taus` |
| `calibrated_tau` | Cutoff calibration and held-out evaluation | `calibrated_tau.csv`, `.json` | `--reps-cal`, `--reps-val`, `--targets` |
| `tau_transfer` | Fixed-cutoff transfer across settings | `tau_transfer.csv`, `.json` | `--reps-cal`, `--reps-val`, `--target` |
| `incremental_mechanism` | Marginal versus multivariate statistics | `incremental_mechanism.csv`, `.json` | `--reps`, dimensions |
| `semi_mechanism_diagnostic` | Panel geometry, dependence and tails | `semi_mechanism_diagnostic.json` | dimensions, `--seed` |

Examples of the other CLI families:

```sh
python -m synthetic.semi_synthetic --panel-csv data_research_demo/panel.csv --output-dir outputs/research/demo_semi --reps 2
python -m synthetic.dgp_sensitivity --panel-csv data_research_demo/panel.csv --output-dir outputs/research/demo_grid --reps 2
python -m synthetic.calibrated_tau --panel-csv data_research_demo/panel.csv --output-dir outputs/research/demo_tau --reps-cal 30 --reps-val 20 --seed-cal 7 --seed-val 1007
python -m synthetic.tau_transfer --panel-csv data_research_demo/panel.csv --output-dir outputs/research/demo_transfer --reps-cal 30 --reps-val 15 --seed-cal 7 --seed-val 1007
python -m synthetic.semi_mechanism_diagnostic --panel-csv data_research_demo/panel.csv --output-dir outputs/research/demo_diagnostics
```

Independent dummy features may leave no false factors at the calibration gate.
That is an insufficient-evidence failure, not a reason to silently choose a
cutoff. The tests use correlated synthetic fixtures to check these entry points.
Default budgets do not guarantee statistical power.

The four-gate, F2/F3/F4b and paired-analysis commands are documented
in [REPRODUCIBILITY.md](REPRODUCIBILITY.md). The shared research runtime applies to the eleven modules listed above.

## Metric definitions and assumptions

Three supplementary controls have different research roles: `four_gate_ablation`
compares sequential gates on Gen-1 panels; `stage_a_fpr_power` compares BH-only
and screening/redundancy selection; `oracle_positive_control` tests known expressions
separately from generator discovery. They are available through
`python main.py research <module> --help`.

The oracle control requires `mom1m`, `turn`, `bm` and `retvol` in the supplied
calibration configuration. Its default grid is three mechanism kinds, strengths
0.03/0.06/0.10, five development seeds and 150 assets by 150 months. `--kinds`,
`--ics`, `--seeds`, `--n-assets` and `--n-months` allow smaller diagnostic runs;
`meta.json` records the actual grid and whether the template grid matches.
Sealed seeds 3000+ are refused. A reduced grid is an execution check, not the
template's inferential evidence. Output directories must be empty, and manifests
record input/source identities and completion status.

- **Truth** means a nonzero injected coefficient. A false factor can have marginal
  predictability through correlation with truth; marginal BH and independent
  recovery test different hypotheses.
- **FPR** divides false selections by all false factors. **Whole-pipeline Power**
  divides true selections by all true factors.
- **Conditional size / power** uses only false / true factors reaching the FM
  test with finite statistics. Earlier rejections and failed estimates are not
  in that denominator. Split/calibration/transfer fields use `conditional_power*`.
- **FDP** divides false selections by selections. The `FDR_emp` display field is a
  realized per-run FDP (zero for empty selection), not the expectation defining FDR.
- Factors within a seed are dependent. Descriptive tables are not confidence
  bounds and do not establish causal explanations or universal error control.

The FM prototype conditions on surviving candidates and uses IID standard errors
across dates; it does not ensure every true factor remains in the control set.
Both the core pipeline and shared
`four_gate_ablation.gate_incremental` refuse candidates when fewer than five
cross-sections fit or the required statistic is nonfinite. Diagnostic `fm_tstats`
returns NaN for unavailable evidence and records actual fit counts. Five fits is
a minimum, not a power guarantee.

Gen-2 combines panel-estimated quantities with explicit fixed defaults,
including fallback coverage and the sensitivity-grid center. Not every parameter
is re-estimated from each input. Injection IC controls coefficient scale, not an
enforced realized rank IC. Semi-synthetic permutation preserves the observed
return multiset before injection and removes systematic alignment in expectation;
finite draws can retain accidental correlation. Dummy exposures are not real data.

## Result schema and provenance

[RESULT_SCHEMA.md](RESULT_SCHEMA.md) defines total, survivor, finite-statistic
and selected counts, conditional versus full-pipeline power, per-fit usable
dates/rows and closed-loop FDP denominators. `sample_accounting.json` uses schema
version 2. `n_fake_reaching` counts every survivor; `n_fake_finite` separately
counts estimable statistics. Undefined metrics are null in JSON.

Each research CLI writes `manifest.json`: arguments, factor order, input/config/source
hashes, environment, output hashes and RUNNING/COMPLETED/FAILED state. Inputs and
source are rechecked after execution. Nonempty output directories and overlapping
calibration/validation seed ranges are refused. JSON uses null for undefined
metrics. COMPLETED means execution succeeded, not that the scientific hypothesis
was confirmed or the procedure certified.

Reports contain measurements from the supplied inputs. Cutoff calibration
requires finite false-factor statistics and disjoint calibration/evaluation seeds.
Removal counts include unavailable statistics, injection records identify the
actual factors, and correlation diagnostics exclude diagonal entries.

For portfolio research use the settlement-aware walk-forward entry point.
Treat inspected test periods as development OOS. Source/input identities
must match the results being interpreted; the package contains no private
calibration inputs or precomputed numerical evidence.

## Checks

```sh
python -m pip install -e ".[dev,opt]"
python -m pytest tests/test_research_experiments.py -q
python -m pytest tests -q
python scripts/audit_comments.py
```

Use small-input tests to check CLI execution, strict JSON, foreign working
directories and failure boundaries. Run full simulations separately with the
chosen calibration inputs and budgets. Keep audit reports and research records
outside the checkout; see [TESTING.md](TESTING.md) and the commenting policy.
