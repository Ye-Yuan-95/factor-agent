# Agent modules and runtime contracts

Start with the [project README](../README.md). The
[architecture guide](../docs/AGENT_DESIGN.md) describes responsibilities and data boundaries.

| Module | Responsibility |
|---|---|
| `factor_dsl.py` | Expression whitelist, signatures, complexity and past-only operators |
| `generator_contract.py` | Requests, search spaces, failure types and permitted feedback |
| `baseline_generator.py`, `hybrid_generator.py`, `llm_generator.py` | Local generators and optional provider adapters |
| `validator_oracle.py` | Five validation stages, DSR and FactorCard decisions |
| `memory.py`, `deflated_sharpe.py` | Trial ledger, date alignment, effective trials and DSR |
| `gate_certificate.py`, `certification.py` | Source-bound certificates for the specified base selection process |
| `run_context.py`, `l1_runner.py` | Run ownership, budgets and one-shot test lockbox |
| `tools.py` | Structured validation, joint recheck and evaluation tools |
| `user_report.py` | Deterministic reports from saved cards and requests |
| `realized_fdr.py` | Empirical closed-loop FDP and all-run/nonempty-run denominators |
| `b2_compare.py`, `factorial_ablation.py` | Frozen family and mechanism comparison protocols |

## Supported commands

Run from the checkout after installation:

```sh
python main.py demo-data
python main.py run --candidates agent/candidates_example.json --synthetic gen1 --inject-ic 0.06 --budget 5 --seed 7
python main.py baseline --n 10 --out outputs/candidates/local.json
python main.py compare --help
python main.py fdp --help
python main.py toolbox --help
```

Use `run --generator glm` for LLM candidates; `llm_generator.py` is a provider
library used by that workflow. Comparison profiles are `family` and `mechanism`.
The dummy panel checks execution only. Real inputs follow the
[data contract](../docs/DATA_CONTRACT.md); provider setup is in the
[LLM guide](../docs/LLM_GENERATOR_GUIDE.md).

## Results and boundaries

Runs store a request, config, manifest, trial ledger, trace, FactorCards and
`user_report.md`. Failed statistical checks produce REJECT; insufficient
validation-period portfolio value produces SHELVED; missing applicable
certification leaves otherwise acceptable candidates as DRAFT.

The incremental test controls accepted factors, expression ingredients and selected
correlated base columns. Fewer than five usable cross-sectional fits or nonfinite
statistics cannot pass. Cards report fitted dates and complete observations.

Certificates apply to their calibration panel and specified selection process;
they do not certify every adaptive discovery loop. See
[result definitions](../docs/RESULT_SCHEMA.md) for empirical FDP and denominators.

The test lockbox opens once at a human checkpoint after a run completes. Test
results and synthetic truth cannot enter generation feedback. Expression legality,
statistical acceptance and investment value are separate outcomes. Data frequency,
universe, licensing and point-in-time assumptions belong to the research record.
