"""Scan competitor source trees for prompt literals aligned with injected truth.

The scanner writes a hashable CLEAN/LEAK report for literal truth-column mentions.
It cannot exclude semantic alignment: examples on other columns can still teach
the injected mechanism's shape. Structural alignment is reported separately and
must qualify any claim about independent model discovery.

Run: python -m benchmark.dealignment_check --root ../external/RD-Agent
     --truth-cols ep,illiq --out outputs/benchmark/dealign_rdagent.json"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".yaml", ".yml", ".j2", ".jinja",
                 ".jinja2", ".toml", ".cfg", ".ipynb", ".rst"}
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", "build", "dist",
             ".pytest_cache", ".mypy_cache", "site-packages"}
# structural shapes our mechanism truths use; a literal-clean repo that still teaches these
# shapes is reported as a structural warning, not a hard leak
STRUCTURAL_PATTERNS = {
    "conditional_product": r"(zscore|rank)\s*\([^)]*\)\s*\*\s*sign\s*\(",
    "clip_gate": r"\*\s*clip\s*\(",
    "cross_sectional_spread": r"(zscore|rank)\s*\([^)]*\)\s*-\s*(zscore|rank)\s*\(",
}


DSL_FUNCTIONS = ("rank", "zscore", "demean", "industry_demean", "ts_corr", "winsor",
                 "ts_mean", "ts_std", "ts_sum", "ts_min", "ts_max", "ts_rank", "ts_zscore",
                 "delay", "delta", "log", "sqrt", "abs", "sign", "clip",
                 # common competitor operator names (Alpha101 / Qlib expression dialects)
                 "Ref", "Mean", "Std", "Corr", "Rank", "Sum", "Max", "Min", "Abs", "Log",
                 "Greater", "Less", "If", "EMA", "WMA", "Quantile", "Delta", "Sign")
_EXPRESSION_RE = re.compile(
    r"\b(?:" + "|".join(DSL_FUNCTIONS) + r")\s*\([^\n]{0,200}")


def extract_expression_literals(text: str) -> list[str]:
    """Extract expression-like prompt literals rather than vocabulary listings.

    Vocabulary columns define a permitted search space. Example expressions containing
    truth columns can expose the target and invalidate independent discovery claims."""
    return [match.group(0) for match in _EXPRESSION_RE.finditer(text)]


def _iter_text_files(root: Path, max_bytes: int = 2_000_000):
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        try:
            if path.stat().st_size > max_bytes:
                continue
            yield path, path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue


def scan(root: Path, truth_columns: list[str], *, structural: bool = True,
         include: list[str] | None = None) -> dict[str, Any]:
    """Scan ``root`` for truth columns inside expression literals (the verdict driver) and,
    separately, for bare mentions anywhere (context only).

    ``include`` is a list of glob patterns; when given, only matching paths are scanned. Use it
    to point the scan at the files a model actually sees (prompt templates) rather than the
    whole repository, which will always mention every column somewhere.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"scan root does not exist: {root}")
    # word-boundary match so 'ep' does not fire on 'step' / 'deploy'
    column_patterns = {col: re.compile(rf"(?<![A-Za-z0-9_]){re.escape(col)}(?![A-Za-z0-9_])")
                       for col in truth_columns}
    structural_patterns = ({name: re.compile(pattern)
                            for name, pattern in STRUCTURAL_PATTERNS.items()}
                           if structural else {})
    expression_hits: list[dict[str, Any]] = []
    mention_hits: list[dict[str, Any]] = []
    structural_hits: list[dict[str, Any]] = []
    n_files = 0
    for path, text in _iter_text_files(root):
        rel = path.relative_to(root).as_posix()
        if include and not any(Path(rel).match(pattern) for pattern in include):
            continue
        n_files += 1
        for line_no, line in enumerate(text.splitlines(), start=1):
            if len(line) > 2000:
                line = line[:2000]
            hit_columns = [col for col, pattern in column_patterns.items()
                           if pattern.search(line)]
            if hit_columns:
                expressions = extract_expression_literals(line)
                for col in hit_columns:
                    inside = [expr for expr in expressions if column_patterns[col].search(expr)]
                    record = {"file": rel, "line": line_no, "column": col,
                              "text": line.strip()[:200]}
                    if inside:
                        expression_hits.append({**record, "expression": inside[0][:200]})
                    else:
                        mention_hits.append(record)
            for name, pattern in structural_patterns.items():
                if pattern.search(line):
                    structural_hits.append({"file": rel, "line": line_no, "pattern": name,
                                            "text": line.strip()[:200]})
    # A literal containing every truth column fails de-alignment.
    full_pair_expressions = []
    for hit in expression_hits:
        expression = hit["expression"]
        if all(pattern.search(expression) for pattern in column_patterns.values()):
            full_pair_expressions.append({"file": hit["file"], "line": hit["line"],
                                          "expression": expression})
    if full_pair_expressions:
        verdict = "LEAK"
    elif expression_hits:
        verdict = "PARTIAL"
    else:
        verdict = "CLEAN"
    report = {
        "scanned_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "root": str(root),
        "include_globs": list(include or []),
        "n_text_files_scanned": n_files,
        "truth_columns": list(truth_columns),
        "verdict": verdict,
        "verdict_rule": ("LEAK if one expression literal contains every truth column; "
                         "PARTIAL if a truth column appears inside any expression literal; "
                         "CLEAN if truth columns appear only as bare mentions (vocabulary "
                         "lists, config keys), which is legitimate -- the search space is "
                         "allowed to contain the answer, the examples are not."),
        "n_expression_hits": len(expression_hits),
        "n_bare_mentions": len(mention_hits),
        "full_pair_expressions": full_pair_expressions[:50],
        "expression_hits": expression_hits[:100],
        "bare_mention_files": sorted({hit["file"] for hit in mention_hits})[:50],
        "n_structural_warnings": len(structural_hits),
        "structural_warning_files": sorted({hit["file"] for hit in structural_hits})[:50],
        "structural_note": ("Structural warnings are NOT leaks. They mean the corpus teaches "
                            "the same expression SHAPE as the injected mechanism; report "
                            "results stratified by this flag rather than discarding the run."),
        "limitation": ("Literal scan only. Semantic paraphrase (an example that teaches the "
                       "mechanism without naming the columns) is out of scope."),
    }
    report["report_hash"] = hashlib.sha256(
        json.dumps({k: v for k, v in report.items() if k != "scanned_at"},
                   sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
    return report


def main() -> None:
    """Scan specified source patterns for truth-aligned examples and save a report.

    This is a source-literal diagnostic; it neither runs an external miner nor
    establishes that all possible prompt or data leakage has been excluded.
    """
    parser = argparse.ArgumentParser(
        description="Scan a source tree for truth-column literals before a benchmark run")
    parser.add_argument("--root", required=True, help="competitor repo or prompt directory")
    parser.add_argument("--truth-cols", required=True,
                        help="comma-separated injected truth columns, e.g. ep,illiq")
    parser.add_argument("--out", default="", help="write the JSON report here")
    parser.add_argument("--include", default="",
                        help="comma-separated globs limiting the scan to prompt-visible files")
    parser.add_argument("--no-structural", action="store_true")
    args = parser.parse_args()

    columns = [c.strip() for c in args.truth_cols.split(",") if c.strip()]
    includes = [g.strip() for g in args.include.split(",") if g.strip()]
    report = scan(Path(args.root), columns, structural=not args.no_structural,
                  include=includes or None)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"verdict={report['verdict']} hash={report['report_hash']} "
          f"files={report['n_text_files_scanned']} "
          f"expression_hits={report['n_expression_hits']} "
          f"bare_mentions={report['n_bare_mentions']} "
          f"structural_warnings={report['n_structural_warnings']}")
    for hit in report["full_pair_expressions"][:10]:
        print(f"  LEAK {hit['file']}:{hit['line']}  {hit['expression'][:120]}")
    for hit in report["expression_hits"][:10]:
        print(f"  expr {hit['file']}:{hit['line']} [{hit['column']}] {hit['expression'][:110]}")


if __name__ == "__main__":
    main()
