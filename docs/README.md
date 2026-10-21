# Documentation

This project provides A-share data preparation, candidate generation, statistical
validation, portfolio evaluation and synthetic experiments. Start with the
[project README](../README.md) for installation and the shared command interface.

Run checkout examples from the repository root. Installed-wheel runs need explicit
input/config paths. Panels, API credentials, generated results and accepted
certificates are not distributed. Numerical reproduction requires matching
calibration inputs, source, protocols, environment and seeds.

| Guide | Purpose |
|---|---|
| [AGENT_DESIGN.md](AGENT_DESIGN.md) | Responsibilities, validation, decisions, budgets and lockbox |
| [DATA_CONTRACT.md](DATA_CONTRACT.md) | Raw A-share data, units, availability and label settlement |
| [LLM_GENERATOR_GUIDE.md](LLM_GENERATOR_GUIDE.md) | GLM installation, requests, feedback and proposal outputs |
| [REPRODUCIBILITY.md](REPRODUCIBILITY.md) | Evidence requirements, certification, comparisons and portfolio commands |
| [RESEARCH_EXPERIMENTS.md](RESEARCH_EXPERIMENTS.md) | Experiment inputs, controls, budgets and scientific limits |
| [RESULT_SCHEMA.md](RESULT_SCHEMA.md) | Usable samples, conditional/full power and FDP denominators |
| [TESTING.md](TESTING.md) | Dummy workflow, test suite, source checks and wheel installation |
| [COMMENTING_POLICY.md](COMMENTING_POLICY.md) | Source documentation and scientific interpretation standards |

## Command groups

Use `python main.py <command> --help`, `python -m factor_research`, or `factor-agent`.
Data, selection and portfolio commands are in the project README. `baseline`
writes local candidate proposals; `fdp` evaluates closed-loop discovery behavior.
`compare` selects a family or mechanism protocol runner. `benchmark` selects export,
export verification, de-alignment, derived metrics, artifact audit or paired analysis.
`research` exposes eleven primary experiments and three supplementary controls.

Library modules such as the DSL, provider adapters, memory, DGPs, metrics and
manifests are imported by these workflows. They do not each need a standalone CLI.
Maintenance and checkout-specific acceptance commands remain in `scripts/`.

See the [agent module guide](../agent/README.md) and
[factor research package guide](../factor_research/README.md). Frozen experiment
definitions live in `experiments/`; source hashes and run manifests bind actual inputs.

Missing incremental evidence cannot pass. Usable observations are not necessarily
independent samples, and process certificates do not guarantee universal FDR
control or investment returns. Consult the result schema and reproducibility guide
before interpreting a run.
