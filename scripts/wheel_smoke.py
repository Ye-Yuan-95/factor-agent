"""Check installed code, including lazy provenance dependencies, outside the checkout.

Run with python -I scripts/wheel_smoke.py after installing the wheel. The optional
--installed-root supports scripts/run_acceptance.py's temporary pip --target installation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.resources
import json
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path


def main() -> None:
    """Check installed imports, packaged examples, provenance output and CLI help.

    Run outside the source checkout with optimization dependencies available.
    Reject imports resolved from another tree instead of accepting a masked
    packaging omission as a successful wheel installation.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installed-root", type=Path,
                        default=Path(sysconfig.get_paths()["purelib"]))
    root = parser.parse_args().installed_root.resolve()
    sys.path.insert(0, str(root))
    modules = (
        "agent", "agent.validator_oracle", "agent.tools", "agent.realized_fdr",
        "agent.l1_runner", "synthetic.dgp_v2", "benchmark.qlib_export",
        "factor_research.factor_validity_pipeline", "factor_research.portfolio_construction",
        "factor_research.experiment_manifest", "factor_research.migrate_label_end_dates",
        "factor_research.build_a_share_factor_panel", "factor_research.make_dummy_panel",
        "factor_research.hybrid_walkforward", "factor_research.solve_unified_v2",
        "factor_research.decision_focused",
        "factor_research.cli",
    )
    for name in modules:
        module = importlib.import_module(name)
        if not Path(module.__file__).resolve().is_relative_to(root):
            raise RuntimeError(f"{name} loaded outside the installed package: {module.__file__}")

    candidates = importlib.resources.files("agent").joinpath("candidates_example.json")
    if not json.loads(candidates.read_text(encoding="utf-8")):
        raise RuntimeError("candidate package data is empty")

    entry_files = list(root.glob("factor_validity_agent-*.dist-info/entry_points.txt"))
    if len(entry_files) != 1 or "factor-agent = factor_research.cli:main" not in entry_files[0].read_text(encoding="utf-8"):
        raise RuntimeError("installed wheel does not expose the factor-agent console entry")

    # These helpers are imported lazily by the pipeline/portfolio report writers.
    from factor_research.experiment_manifest import build_manifest, write_strict_json
    with tempfile.TemporaryDirectory() as td:
        panel = Path(td) / "panel.csv"
        panel.write_bytes(b"date,asset,forward_return\n2020-01-31,S0000,0.01\n")
        report = Path(td) / "manifest.json"
        write_strict_json(report, build_manifest(panel_csv=panel))
        manifest = json.loads(report.read_text(encoding="utf-8"))
        if manifest["panel_sha256"] != hashlib.sha256(panel.read_bytes()).hexdigest():
            raise RuntimeError("installed provenance writer produced an incorrect input hash")

    cli_modules = ("factor_research", "agent.l1_runner", "factor_research.make_dummy_panel",
                   "factor_research.build_a_share_factor_panel",
                   "factor_research.migrate_label_end_dates",
                   "factor_research.factor_validity_pipeline",
                   "factor_research.solve_unified_v2",
                   "factor_research.portfolio_construction",
                   "factor_research.hybrid_walkforward")
    for module in cli_modules:
        code = (f"import sys, runpy; sys.path.insert(0, {str(root)!r}); "
                f"sys.argv = [{module!r}, '--help']; "
                f"runpy.run_module({module!r}, run_name='__main__')")
        subprocess.run([sys.executable, "-I", "-c", code], check=True,
                       capture_output=True, text=True, timeout=60)
    print("wheel-ok: installed imports, console entry, package data, provenance JSON, 9 module CLIs")


if __name__ == "__main__":
    main()
