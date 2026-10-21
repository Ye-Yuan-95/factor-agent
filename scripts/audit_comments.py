"""Inventory documentation across the upload source tree without importing code.

This checks module descriptions and nontrivial top-level/class function docstrings.
It is a maintainability guard, not a semantic proof that comments are correct.
Test cases can rely on descriptive names; shared test helpers still need prose.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ("factor_research", "agent", "benchmark", "synthetic", "scripts", "tests")


def source_files(root: Path = ROOT) -> list[Path]:
    """List publishable Python sources; exclude generated build/runtime directories."""
    paths = list(root.glob("*.py"))
    for name in SOURCE_DIRS:
        paths.extend(
            p for p in (root / name).rglob("*.py") if "__pycache__" not in p.parts
        )
    return sorted(set(paths))


def substantive_functions(nodes, prefix: str = ""):
    """Yield nontrivial module/class functions; nested closures use enclosing docs.

    Twelve physical lines is a review trigger, not a quality score. CLI main
    functions rely on module purpose, argparse help and the experiment guide.
    """
    for node in nodes:
        if isinstance(node, ast.ClassDef):
            yield from substantive_functions(node.body, prefix + node.name + ".")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if (
                node.name != "main"
                and not node.name.startswith("test_")
                and node.end_lineno - node.lineno >= 12
            ):
                yield prefix + node.name, node


def inspect_sources(root: Path = ROOT) -> tuple[list[dict], list[str]]:
    """Parse every source and collect module/function documentation gaps and hashes."""
    rows, issues = [], []
    for path in source_files(root):
        relative = path.relative_to(root).as_posix()
        source = path.read_text(encoding="utf-8-sig")
        tree = ast.parse(source, filename=relative)
        module_doc = ast.get_docstring(tree)
        functions = list(substantive_functions(tree.body))
        missing = [name for name, node in functions if not ast.get_docstring(node)]
        if not module_doc:
            issues.append(f"{relative}: missing module description")
        issues.extend(
            f"{relative}: missing function description: {name}" for name in missing
        )
        rows.append(
            {
                "path": relative,
                "module": bool(module_doc),
                "functions": len(functions),
                "missing": missing,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return rows, issues


def write_report(path: Path, rows: list[dict], issues: list[str]) -> None:
    """Write a reproducible per-file inventory without claiming automated semantic review."""
    text = [
        "# Code comment audit",
        "",
        "Scope: main.py, factor_research/, agent/, benchmark/, synthetic/, scripts/, tests/.",
        "Generated data, environments and build trees are excluded.",
        "",
        f"Files checked: **{len(rows)}**. Structural documentation gaps: **{len(issues)}**.",
        "",
        "This is an AST-based coverage check, not proof of comment accuracy.",
        "For documentation requirements and interpretation limits, see",
        "docs/COMMENTING_POLICY.md and docs/RESULT_SCHEMA.md in the source checkout.",
        "Self-describing test functions, CLI main",
        "functions and nested closures are not required to duplicate their enclosing docs.",
        "",
        "| File | Module doc | Substantive functions | Missing docs | SHA-256 |",
        "|---|---|---:|---|---|",
    ]
    for row in rows:
        text.append(
            f"| {row['path']} | {row['module']} | {row['functions']} | "
            f"{', '.join(row['missing']) or '—'} | {row['sha256']} |"
        )
    if issues:
        text += ["", "## Required follow-up", ""] + [f"- {issue}" for issue in issues]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def main() -> None:
    """Check the checkout and optionally save a Markdown inventory for review."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path,
                        help="save a private inventory outside the repository")
    args = parser.parse_args()
    if args.report and args.report.resolve().is_relative_to(ROOT.resolve()):
        parser.error("--report must be outside the checkout; generated audit reports are private records")
    rows, issues = inspect_sources()
    if args.report:
        write_report(args.report, rows, issues)
    print(f"Comment structure: {len(rows)} files, {len(issues)} gaps")
    if issues:
        raise SystemExit("\n".join(issues))


if __name__ == "__main__":
    main()
