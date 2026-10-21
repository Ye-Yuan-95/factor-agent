# Reproducibility and experiments

Run commands below from the repository root after the README installation and
panel-generation steps. Long simulations are separate from the quick start.

## What this release includes

The repository ships source code, configs, experiment protocols and tests.
Supply calibration panels, benchmark prices and applicable process certificates
separately. Generate dummy inputs locally for workflow checks. CSVs and runtime
outputs are excluded by `.gitignore`.

Before a simulation, preserve the calibration panel, factor order, configuration,
protocol, source version, environment and seeds. Gen-1 and mechanism calibration
use the panel's factor-correlation matrix and return distribution; Gen-2 estimates
additional panel properties. A seed identifies RNG draws only within that full
setting. Record the emitted input/source hashes with the output artifacts.
Hashes identify an available dataset; keep the dataset itself or a sufficient
calibration snapshot with a loader to repeat numerical results.

Git identity is available only when the code's source root is itself a checkout
or worktree. Unversioned copies and installed wheels record null/unknown even
when stored inside another repository; that parent's commit does not identify
the release source. Source content hashes remain necessary for modified code.

Install the dependency ranges declared in `pyproject.toml`. For each experiment,
save the Python version, platform and `python -m pip freeze` output alongside
its private artifacts. Those records describe the environment actually used;
the package dependency ranges alone do not pin a numerical reproduction.

The two root configs use identical numerical settings and factor definitions.
`a_share_baseline_config.json` points to `data/` and `outputs/a_share_baseline/`;
`a_share_big_config.json` points to `data_big_v2/` and `outputs/a_share_big/`.
They are input/output path presets, not stored experiment outcomes. Choose the
one matching your panel path or supply a separate config.

## Certificates and local generators

The [research suite](RESEARCH_EXPERIMENTS.md) provides eleven optional
semi-synthetic, sensitivity, omission, threshold and mechanism experiments with
portable entry points and manifests. [RESULT_SCHEMA.md](RESULT_SCHEMA.md)
defines the metrics and effective sample counts used to interpret new runs.

These commands default to `data_big_v2/a_share_factor_panel.csv`. Supply
`--panel-csv` to the certificate / realized-FDR commands when using another panel.
Certificate generation and multi-seed acceptance can take substantially longer
than the tutorial, and generating a certificate does not guarantee it passes.

```sh
python -m agent.gate_certificate --panel-csv data_big_v2/a_share_factor_panel.csv
python main.py baseline --columns-from data_big_v2/a_share_factor_panel.csv --n 30 --seed 0 --out outputs/candidates/baseline.json
python main.py fdp --gen1-seeds 10 --gen1-null-seeds 4 --gen2-seeds 3 --budget 12 --generator adaptive
```

The FDP harness requires an empty output directory and refuses existing run IDs.
Choose a new `--output-dir` for another evaluation to preserve previous evidence.

`scripts/run_acceptance.py` is a research acceptance command: in addition to package tests,
it requires an accepted certificate bound to the supplied calibration panel and
current gate configuration. It is not the zero-data CI entry point. Use `pytest`
for the fresh-clone checks; do not interpret a dummy-panel certificate as evidence
that a real-data panel is certified.

## Comparison and control protocols

The frozen protocols specify known cross-family mechanisms and compare generators
on paired synthetic panels. Choose and record a calibration dataset before
comparing generators. Dummy calibration is suitable for workflow checks.

Files in `experiments/` are runnable input specifications, not result snapshots.
Dates and protocol IDs distinguish fixed seeds, budgets and search constraints;
keep them when recording a run. The retained protocols serve these comparisons:

| Protocol file | Experimental purpose |
|---|---|
| `b2_momentum_preregistered_2026-07-20_v2.json` | Gen-1 family comparison |
| `b2_momentum_ladder_2026-07-23_A.json` | Gen-2 with restricted operators |
| `b2_momentum_ladder_2026-07-23_B.json` | Paired Gen-2 comparison with the full supported operator set |
| `b2_momentum_ladder_2026-07-24_D2.json` | Gen-2 with mechanism recipe prompting |
| `factorial_mechanism_2026-07-24.json` | Vocabulary, recipe and family-constraint factorial design on `mom1m` / `turn` truth |
| `factorial_mechanism_2026-07-25_F2.json` | Factorial design on `ep` / `illiq` truth, absent from literal prompt examples |
| `factorial_mechanism_2026-07-27_F3.json` | Generation-budget ladder with GLM, recipe-random and enumeration arms |
| `hybrid_ladder_2026-07-28_F4.json` | Local hybrid exploration at paired development seeds |
| `hybrid_form_aware_2026-07-31_F4b.json` | Local form-completion and matched reference arms |
| `oracle_positive_control_2026-07-24.json` | Known-expression positive, negative and bare-ingredient controls |

Use `compare family` for B2 protocols, `compare mechanism` for factorial/hybrid
protocols, and `research oracle_positive_control` for the control grid. Read each
command's `--help` for input paths and subset options.

```sh
python main.py compare mechanism --protocol experiments/factorial_mechanism_2026-07-25_F2.json --arms random_narrow,random_wide,recipe_random_narrow,recipe_random_wide
python main.py compare mechanism --protocol experiments/factorial_mechanism_2026-07-27_F3.json --arms enumerator_narrow_b12,enumerator_narrow_b24,enumerator_narrow_b36
python main.py compare mechanism --protocol experiments/hybrid_form_aware_2026-07-31_F4b.json
python main.py benchmark audit
python main.py benchmark analyse
```

F4b's listed arms are local and need no credentials. GLM arms need `GLM_API_KEY`.
Before starting any mechanism comparison, commit the protocol and the runner's
run-affecting source files in the checkout. The runner refuses inputs that are
untracked or differ from `HEAD`; the research suite's other commands have their
own provenance requirements.
The factorial runner checks the protocol, config and source fingerprints before
resuming or summarizing, and checks generated panel hashes when reusing completed
arms. Source checks include the shared gate dependencies (DSR and certificate/CI
logic). Use a **new protocol/output directory** when changing inputs or moving
from an older source fingerprint; keep frozen research protocols and results unchanged.

The B2 family runner also binds the configuration and shared decision/DGP sources
before resuming. Use a new output directory after changing them. F4b audit and
analysis require the exact planned result set, matching protocol manifest, paired
panel hashes and complete candidate budgets. Worker log files are optional
diagnostics and are not required by sequential runs.

## External-system benchmarks

The Qlib exporter prepares a known-truth synthetic dataset; it does not install
or run RD-Agent, AlphaGen or AlphaForge. External repositories must be obtained
separately. Adjust the paths below to their local checkouts.

Choose an empty export directory; exporting again into existing results is refused.
Native `benchmark verify-export` requires `pyqlib` installed separately in a
compatible Python environment. Exporting and the `read_qlib_bin` helper require
only the core dependencies. Verification samples fields from instrument `s0000`;
it does not replace full external-system integration tests.

```sh
python main.py benchmark export --dgp mechanism --seed 2120 --out outputs/benchmark/mech_s2120 --external-repos ../external/RD-Agent,../external/alphagen,../external/AlphaForge
python main.py benchmark verify-export
python main.py benchmark dealign --root ../external/RD-Agent --truth-cols ep,illiq
```

Keep truth sidecars and manifests outside the external miner's readable inputs.
Run the prompt de-alignment check before experiments to detect truth columns in
worked expression examples. Exporting a dataset alone does not establish a fair
or completed external-system comparison.

Library integrations can call `benchmark.qlib_export.export_alphagen_slots`
to map six chosen factors onto the fixed feature-slot layout; pass
`require_columns` to enforce inclusion of required ingredients. The manifest
records the mapping. `read_qlib_bin` decodes exported binaries for readback
without pyqlib. These helpers export/read data and do not launch mining systems.

## Portfolio / optimization smoke run

Some portfolio modules default to `data/a_share_factor_panel.csv`.
Pass paths explicitly. The solver also needs a factor-selection CSV and its
manifest, both produced by the selection step; neither is included in Git.
After `python -m factor_research.make_dummy_panel` and installation of `.[opt]`:

```sh
python -m factor_research.factor_validity_pipeline --config a_share_big_config.json --train-end 2021-12-31
python -m factor_research.solve_unified_v2 --panel-csv data_big_v2/a_share_factor_panel.csv --selection-csv outputs/a_share_big/final_factor_selection.csv --config a_share_big_config.json --output-dir outputs/unified_v2_dummy --train-end 2021-12 --no-survival --max-cardinality 2 --solver auto
```

This is an optimization smoke run with survival disabled, not a certified strategy.
A walk-forward run may return controlled `INSUFFICIENT_DATA` when no validated
factors remain. Formal benchmark-relative test reports must supply CSI-300 data
and `--require-full-test-benchmark`; see [DATA_CONTRACT.md](DATA_CONTRACT.md).

For out-of-sample research, use label-settlement cutoffs enforced by
`factor_research/factor_validity_pipeline.py`, `factor_research/solve_unified_v2.py`, `factor_research/hybrid_walkforward.py` and the
oracle. An older panel can be migrated with `factor_research/migrate_label_end_dates.py` using its
raw prices. `factor_research.decision_focused` fits weights only on the observations
supplied by its caller; the walk-forward caller purges unavailable labels first.
The fixed-factor diagnostic in `factor_research.portfolio_construction` uses a
signal-month covariance window and is **not OOS-safe**. Use
`factor_research.hybrid_walkforward` for settlement-aware portfolio evaluation.
