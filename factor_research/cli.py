"""Dispatch common commands to their existing module entry points.

Imports are lazy so interface help does not load optional solvers or LLM SDKs.
Each target retains its own argument validation, exit status and research-run
bookkeeping. Dispatch never changes defaults, budgets or statistical decisions.
"""
from __future__ import annotations

import argparse
import runpy
import sys

COMMANDS = {
    "run": ("agent.l1_runner", "Validate supplied or generated factor candidates"),
    "demo-data": ("factor_research.make_dummy_panel", "Generate workflow-only random data"),
    "build-panel": ("factor_research.build_a_share_factor_panel", "Build or preflight A-share data"),
    "migrate-labels": ("factor_research.migrate_label_end_dates", "Verify and add settlement dates"),
    "select": ("factor_research.factor_validity_pipeline", "Run training-only factor selection"),
    "solve": ("factor_research.solve_unified_v2", "Run constrained joint selection"),
    "portfolio": ("factor_research.portfolio_construction", "Run fixed-factor portfolio diagnostics"),
    "walkforward": ("factor_research.hybrid_walkforward", "Evaluate settlement-aware refits"),
    "certificate": ("agent.gate_certificate", "Evaluate a process certificate"),
    "toolbox": ("agent.tools", "Use structured tools and their existing checkpoints"),
    "baseline": ("agent.baseline_generator", "Write local candidate proposals"),
    "fdp": ("agent.realized_fdr", "Evaluate empirical discovery-loop FDP"),
}
RESEARCH_MODULES = (
    "p1_validate", "semi_synthetic", "incremental_robustness", "dgp_sensitivity",
    "omitted_factor_experiment", "tau_size_sweep", "split_sample_incremental",
    "calibrated_tau", "tau_transfer", "incremental_mechanism",
    "semi_mechanism_diagnostic",
    "four_gate_ablation", "stage_a_fpr_power", "oracle_positive_control",
)
GROUPED_COMMANDS = {
    "compare": {
        "family": "agent.b2_compare",
        "mechanism": "agent.factorial_ablation",
    },
    "benchmark": {
        "export": "benchmark.qlib_export",
        "verify-export": "benchmark.qlib_smoke_report",
        "dealign": "benchmark.dealignment_check",
        "derived": "benchmark.derived_metrics",
        "audit": "benchmark.f4b_artifact_audit",
        "analyse": "benchmark.f4b_analysis",
    },
    "research": {name: "synthetic." + name for name in RESEARCH_MODULES},
}


def main(argv: list[str] | None = None) -> int:
    """Select a declared command, forward arguments unchanged and preserve child exits.

    Parse only the command name here: target --help and flags belong to the child
    parser. Restore sys.argv even when a target refuses input or exits with error.
    With no command, print help without running an experiment or calling an API.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    descriptions = "\n".join(f"  {name:15s} {entry[1]}" for name, entry in COMMANDS.items())
    parser = argparse.ArgumentParser(
        prog="factor-agent", description="Factor Agent: data, candidate validation and research tools.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Commands:\n" + descriptions + "\n  research        Run an optional experiment or control\n"
               "  compare         Compare generators under a frozen protocol\n"
               "  benchmark       Export, verify or analyse benchmark artifacts\n\n"
               "Use <command> --help for its inputs and options.",
    )
    parser.add_argument("command", choices=(*COMMANDS, *GROUPED_COMMANDS), metavar="<command>")
    if not args or args[0] in ("-h", "--help"):
        parser.print_help()
        return 0
    command = parser.parse_args(args[:1]).command
    forwarded = args[1:]
    if command in GROUPED_COMMANDS:
        group = GROUPED_COMMANDS[command]
        group_parser = argparse.ArgumentParser(
            prog=f"factor-agent {command}", description=f"{command.title()} tools.")
        group_parser.add_argument("tool", choices=tuple(group))
        if not forwarded or forwarded[0] in ("-h", "--help"):
            group_parser.print_help()
            return 0
        module = group[group_parser.parse_args(forwarded[:1]).tool]
        forwarded = forwarded[1:]
    else:
        module = COMMANDS[command][0]
    previous_argv = sys.argv
    try:
        sys.argv = [module, *forwarded]
        runpy.run_module(module, run_name="__main__")
    finally:
        sys.argv = previous_argv
    return 0
