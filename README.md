# Factor Agent

An A-share factor research tool for candidate formulas, statistical validation
and portfolio evaluation. Candidates may come from files, local generators or
an optional LLM. The host controls budgets and decisions. This is a research
workflow and does not place trades.

## Quick start

Use Python 3.10 or newer. Run from the repository root:

```sh
git clone https://github.com/Ye-Yuan-95/factor-agent.git
cd factor-agent
python -m pip install -e "."
python main.py --help
python main.py demo-data
python main.py run --candidates agent/candidates_example.json --synthetic gen1 --inject-ic 0.06 --budget 5 --seed 7
```

The tutorial needs no market data, API key or optimization extra. It creates a
random calibration panel (60 assets by 96 months), generates a separate market
with injected signals (150 assets by 150 months) and
writes FactorCards, a trial ledger and `user_report.md` under `runs/<run_id>/`.
Candidates may be rejected; running successfully does not guarantee acceptance.
The example budget permits five trials; the final sandbox-rejection example is
beyond that budget. The test suite runs these exact tutorial commands in an
isolated copy; see [dummy workflow checks](docs/TESTING.md#dummy-workflow).
`demo-data` refuses to overwrite existing inputs. Choose another `--out` path
and pass it to `run --panel-csv` when keeping an existing research panel.

## Unified interface

`main.py` is the checkout entry. The same interface is available through
`python -m factor_research` or the installed `factor-agent` command.

| Command | Purpose |
|---|---|
| `run` | Validate supplied or generated candidates |
| `demo-data`, `build-panel`, `migrate-labels` | Prepare and validate inputs |
| `select`, `solve` | Train-only screening and joint factor selection |
| `portfolio`, `walkforward` | Portfolio diagnostics and settlement-aware evaluation |
| `research <module>` | Run an optional experiment |
| `baseline`, `fdp` | Local proposals and discovery-loop evaluation |
| `compare <profile>`, `benchmark <tool>` | Protocol comparisons and benchmark artifacts |
| `certificate`, `toolbox` | Process certification and structured tools |

Use `python main.py <command> --help` for options. Business implementations live
in packages; their existing `python -m <package>.<module>` commands also work.
Optimization needs `.[opt]`; GLM generation needs `.[llm-glm]` and `GLM_API_KEY`.

## Research experiment modules

These **11 optional experiment/diagnostic modules** are separate from routine
agent operation. Run `python main.py research <module> --help`.

| Module | Research question |
|---|---|
| `p1_validate` | Sequential gates under Gen-2 simulation |
| `semi_synthetic` | Supplied exposures with permuted/injected returns |
| `incremental_robustness` | Signal strength, truth structure and tails |
| `dgp_sensitivity` | Tails, clustering and sample size |
| `omitted_factor_experiment` | Omitted truth and proxy selection |
| `tau_size_sweep` | Threshold size and full-pipeline FPR/power |
| `split_sample_incremental` | Same-sample versus split-sample testing |
| `calibrated_tau` | Calibration with held-out evaluation |
| `tau_transfer` | Cutoff transfer across settings |
| `incremental_mechanism` | Marginal versus multivariate evidence |
| `semi_mechanism_diagnostic` | Panel geometry, dependence and tail diagnostics |

See [experiment inputs and metric definitions](docs/RESEARCH_EXPERIMENTS.md).
Supplementary controls are `four_gate_ablation`, `stage_a_fpr_power` and
`oracle_positive_control`; the same `research` command runs them.
Raw panels, results and accepted certificates are not distributed. For repeatable
runs, record calibration inputs, source, configuration, environment and seeds.
Passing statistical gates is not proof of profitability or universal FDR control.

## Documentation and validation

[Documentation](docs/README.md) links the architecture, real-data
contract, LLM guide, reproducibility requirements, result schema and validation.
The real-data adapter is A-share specific; US equities need a separate adapter.

```sh
python -m pip install -e ".[dev,opt]"
python -m pytest tests -q
python scripts/audit_comments.py
```

`factor_research/` holds data, selection and portfolio tools; `agent/` holds the
discovery loop; `synthetic/` holds simulations; `benchmark/` holds comparisons;
`experiments/` holds protocols; `scripts/` holds maintenance and acceptance.
See [LICENSE](LICENSE) and [CITATION.cff](CITATION.cff) for licensing and citation.
