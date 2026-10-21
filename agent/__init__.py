"""Factor expression execution, validation, trial memory and research tools.

The package exposes the DSL, DSR helpers, MemoryStore, ValidatorOracle and
FactorCard. Command workflows are documented in agent/README.md and reached
through the shared interface in main.py or factor-agent."""
from .factor_dsl import compile_expression, DSLError, CompiledExpression
from .deflated_sharpe import deflated_sharpe, expected_max_sharpe, probabilistic_sharpe, effective_n_trials
from .memory import MemoryStore
from .validator_oracle import ValidatorOracle, FactorCard

__all__ = [
    "compile_expression", "DSLError", "CompiledExpression",
    "deflated_sharpe", "expected_max_sharpe", "probabilistic_sharpe", "effective_n_trials",
    "MemoryStore", "ValidatorOracle", "FactorCard",
]
