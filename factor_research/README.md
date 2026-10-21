# Factor research package

This package provides data preparation, statistical selection, portfolio tools
and shared result provenance. Run commands from the checkout after installation,
or pass explicit input/config/output paths when using an installed wheel.

The shared interface is `python main.py <command>` in a checkout,
`python -m factor_research <command>`, or the installed `factor-agent <command>`.
Use `--help` to list commands. `cli.py` dispatches to existing module entry points;
`__main__.py` exposes that interface through package execution.

| Module | Purpose |
|---|---|
| `build_a_share_factor_panel` | Build and preflight point-in-time A-share panels |
| `make_dummy_panel` | Generate workflow-only random data without overwriting inputs |
| `migrate_label_end_dates` | Derive and verify missing label settlement dates |
| `factor_validity_pipeline` | Train-only screening, incremental testing and sparse weights |
| `portfolio_construction` | Covariance, cap and portfolio helpers; fixed-factor diagnostics |
| `solve_unified_v2` | Joint constrained factor selection with solver/provenance checks |
| `hybrid_walkforward` | Settlement-aware refits and portfolio evaluation |
| `decision_focused` | Shared softmax weights and training-only Sharpe fitting |
| `experiment_manifest` | Input identity and strict-JSON result manifests |

The first seven modules have command-line interfaces:

```sh
python -m factor_research.make_dummy_panel --out data_research_demo/panel.csv
python -m factor_research.build_a_share_factor_panel --history-dir /path/to/raw --check-only
python -m factor_research.factor_validity_pipeline --config a_share_big_config.json --train-end 2021-12-31
python -m factor_research.hybrid_walkforward --help
```

`decision_focused` and `experiment_manifest` are library helpers. The fitting
caller must supply only labels available at its training cutoff. Portfolio
optimization needs the `opt` extra. The fixed-factor diagnostic's covariance
window uses signal-month timing; use `hybrid_walkforward` for settlement-aware
out-of-sample evaluation.

For library integrations, `factor_validity_pipeline.cohort_split` assigns rows
by label settlement. `hybrid_walkforward.load_index_fwd` and
`benchmark_series_sha256` expose benchmark returns and canonical identity;
use `load_benchmark_snapshot` when both must come from one file read.

Config files and calibration data are separate inputs; the wheel contains code
and agent candidate examples. See [README](../README.md),
[DATA_CONTRACT](../docs/DATA_CONTRACT.md) and [REPRODUCIBILITY](../docs/REPRODUCIBILITY.md).
Research acceptance is a checkout tool: `python scripts/run_acceptance.py --help`.
