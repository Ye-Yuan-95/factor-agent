"""Agent toolbox: existing scripts wrapped as structured-I/O tools.

Every tool takes plain arguments and returns a JSON-serializable dict, so any driver (LLM loop,
baseline generator, or human CLI) calls the same interface. Two guard-rails are POLICY:

  * check_expression returns NO performance number (no IC preview) -- a "quick peek" IC would be
    an untracked statistical trial, exactly the snooping the Memory ledger exists to count.
  * open_test_lockbox is the ONLY code path that touches test rows: explicit human confirmation
    string, exactly one frozen (COMPLETED) run on its own frozen panel/config (content-hash
    checked), results stay inside the run -- never in Memory, so no feedback into generation.

CLI:  python -m agent.tools <tool> [args...]     (run with --help for each tool)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from agent.factor_dsl import compile_expression, DSLError, column_is_allowed  # noqa: E402
from agent.memory import MemoryStore  # noqa: E402
from agent.validator_oracle import ValidatorOracle, _composite_ls_stats, _resolve_path  # noqa: E402
from agent.certification import content_hash  # noqa: E402
from factor_research.factor_validity_pipeline import load_config, fit_sparse_linear_weights  # noqa: E402

LOCKBOX_CONFIRM = "I-AM-A-HUMAN-CHECKPOINT"
DEFAULT_OUT = _PROJECT_DIR / "outputs" / "agent"


# ---- tools ----
def check_expression(expression: str, panel_csv: str | None = None) -> dict:
    """Sandbox-compile an expression. NO performance information is returned (anti-snooping)."""
    try:
        compiled = compile_expression(expression)
    except DSLError as exc:
        return {"ok": False, "error": str(exc)}
    out = {"ok": True, "signature": compiled.signature(), "columns_used": list(compiled.columns),
           "complexity": compiled.complexity()}
    if panel_csv:
        header = pd.read_csv(panel_csv, nrows=0).columns
        missing = [c for c in compiled.columns if c not in header]
        if missing:
            return {"ok": False, "error": f"columns not in panel: {missing}",
                    "signature": compiled.signature()}
        out["panel_columns_ok"] = True
    return out


def list_columns(panel_csv: str) -> dict:
    """Columns of a panel, split into DSL-allowed inputs and forbidden bookkeeping columns."""
    header = list(pd.read_csv(panel_csv, nrows=0).columns)
    allowed = [c for c in header if column_is_allowed(c)]
    return {"panel_csv": panel_csv, "allowed_columns": allowed,
            "forbidden_columns": [c for c in header if c not in allowed]}


def validate_factor(*, name: str, expression: str, rationale: str = "", family: str = "unclassified",
                    panel_csv: str | None = None, config: str | None = None,
                    existing: list[str] | None = None, memory_dir: str | None = None,
                    cards_dir: str | None = None, dsr_min: float = 0.95, seed: int = 42,
                    runs_root: str | None = None,
                    certificate_path: str | None = None) -> dict:
    """Run the validator oracle for one candidate as a single-candidate RUN:
    card/memory/trace/certificate_refs land in runs/<run_id>/."""
    from agent.run_context import RunContext
    config = config or str(_PROJECT_DIR / "a_share_big_config.json")
    oracle = ValidatorOracle(config, panel_csv=panel_csv, dsr_min=dsr_min,
                             certificate_path=certificate_path)
    ctx = RunContext.create(
        purpose="validate_factor", universe=oracle.panel_csv.parent.name,
        config_snapshot={"pipeline_config": json.loads(Path(config).read_text(encoding="utf-8")),
                         "oracle_gate_params": oracle.oracle_gate_params()},
        manifest_extra={"panel_csv": str(oracle.panel_csv), "seed": seed},
        runs_root=runs_root)
    memory = MemoryStore(memory_dir or ctx.memory_dir)
    card = oracle.validate(name=name, expression=expression, rationale=rationale, family=family,
                           existing=existing or [], memory=memory, seed=seed)
    card_dict = card.to_dict()
    card_path = ctx.write_card(name, card_dict)
    ctx.trace("validated", name=name, decision=card.decision, certified=card.certified)
    if getattr(oracle, "last_certificate_refs", None):
        ctx.write_certificate_refs(oracle.last_certificate_refs)
    ctx.set_state("COMPLETED")
    if cards_dir:  # Optional caller-requested card copy
        Path(cards_dir).mkdir(parents=True, exist_ok=True)
        (Path(cards_dir) / card_path.name).write_text(
            json.dumps(card_dict, indent=2, ensure_ascii=False), encoding="utf-8")
    card_dict["run_dir"] = str(ctx.run_dir)
    return card_dict


def gate_certificate(**kwargs) -> dict:
    """Re-issue the process-level FPR/Power certificate (Gen-1 + Gen-2). See agent/gate_certificate.py."""
    from agent.gate_certificate import build_certificate
    defaults = dict(panel_csv=str(_PROJECT_DIR / "data_big_v2" / "a_share_factor_panel.csv"),
                    config_path=str(_PROJECT_DIR / "a_share_big_config.json"),
                    gen1_reps=60, gen2_reps=30, n_assets=150, n_months=150,
                    alpha_ic=0.03, extra_ic=0.06, seed=2026,
                    output_path=DEFAULT_OUT / "gate_certificate.json")
    defaults.update(kwargs)
    return build_certificate(**defaults)


def memory_stats(memory_dir: str | None = None) -> dict:
    """The anti-snooping ledger: n_trials, N_eff, per-decision counts, trial-Sharpe variance."""
    return MemoryStore(memory_dir or DEFAULT_OUT / "memory").stats()


def existing_factors(selection_csv: str) -> dict:
    """Read an already-validated factor set from a pipeline final_factor_selection.csv."""
    sel = pd.read_csv(selection_csv)
    col = "selected_final" if "selected_final" in sel.columns else "passes_incremental"
    chosen = sel.loc[sel[col] == True, "factor"].astype(str).tolist()  # noqa: E712
    return {"selection_csv": selection_csv, "criterion": col, "factors": chosen}


def open_test_lockbox(*, run_dir: str, panel_csv: str | None = None, config: str | None = None,
                      confirm: str = "") -> dict:
    """HUMAN CHECKPOINT ONLY -- sealed-run lockbox. Accepts exactly ONE frozen run:
    the run must be COMPLETED (not ACTIVE, not already TEST_OPENED), must be a REAL universe
    (synthetic runs are refused), and only that run's ACCEPTed cards matching the manifest
    universe are evaluated. Opening flips the run to TEST_OPENED permanently; results go to the
    run's cards and lockbox report -- NEVER to Memory."""
    from agent.run_context import RunContext
    if confirm != LOCKBOX_CONFIRM:
        return {"ok": False,
                "error": f"refused: pass confirm='{LOCKBOX_CONFIRM}' -- this is a one-shot human "
                         "approval checkpoint; test results must never flow back to the agent loop."}
    try:
        ctx = RunContext.load(run_dir)
    except FileNotFoundError as exc:
        return {"ok": False, "error": str(exc)}
    manifest = ctx.manifest()
    if manifest.get("state") == "TEST_OPENED":
        return {"ok": False, "error": f"refused: run {manifest['run_id']} already TEST_OPENED at "
                                      f"{manifest.get('lockbox_opened_at')} -- the lockbox is one-shot"}
    if manifest.get("state") != "COMPLETED":
        return {"ok": False, "error": f"refused: run state is {manifest.get('state')}; only a "
                                      "frozen (COMPLETED) run may open the lockbox"}
    universe = str(manifest.get("universe", ""))
    if universe.startswith("synthetic"):
        return {"ok": False, "error": f"refused: run universe '{universe}' is synthetic -- the test "
                                      "lockbox exists for real panels only (isolation rule)"}
    cards = [(p, json.loads(p.read_text(encoding="utf-8")))
             for p in sorted(ctx.cards_dir.glob("*.json"))]
    accepted = [(p, c) for p, c in cards
                if c.get("decision") == "ACCEPT" and c.get("universe") == universe]
    if not accepted:
        return {"ok": False, "error": f"no ACCEPTed cards for universe '{universe}' in this run"}

    # the lockbox evaluates the run's OWN frozen data and config.
    # The panel must be the manifest's panel AND its CONTENT hash must equal the cards'
    # data_version; the config/splits come from config.snapshot.json -- no caller overrides.
    if config is not None:
        return {"ok": False, "error": "refused: config override not allowed -- the run's "
                                      "config.snapshot.json is the frozen configuration"}
    manifest_panel = manifest.get("panel_csv")
    if not manifest_panel:
        return {"ok": False, "error": "refused: manifest has no panel_csv (pre-sealed-format run)"}
    panel_path = _resolve_path(manifest_panel)
    if panel_csv and Path(panel_csv).resolve() != panel_path.resolve():
        return {"ok": False, "error": f"refused: panel '{panel_csv}' is not the run's frozen "
                                      f"panel '{manifest_panel}'"}
    if not panel_path.exists():
        return {"ok": False, "error": f"frozen panel not found: {panel_path}"}
    panel_hash = content_hash(panel_path)
    mismatched = [c["name"] for _, c in accepted if c.get("data_version") != panel_hash]
    if mismatched:
        return {"ok": False, "error": f"refused: panel content hash {panel_hash} does not match "
                                      f"data_version on cards {mismatched} -- the lockbox opens "
                                      "only on the exact data the cards were validated on"}
    snapshot = json.loads((ctx.run_dir / "config.snapshot.json").read_text(encoding="utf-8"))
    oparams = snapshot.get("oracle_gate_params", {})
    if "train_end" not in oparams or "val_end" not in oparams:
        # splits must come from the frozen snapshot -- no default-date fallback
        return {"ok": False, "error": "refused: config.snapshot.json lacks "
                                      "oracle_gate_params.train_end/val_end (no fallback splits)"}
    cfg_path = ctx.run_dir / "_lockbox_config.json"
    cfg_path.write_text(json.dumps(snapshot["pipeline_config"], ensure_ascii=False),
                        encoding="utf-8")
    cfg = load_config(cfg_path)
    train_end = pd.Timestamp(oparams["train_end"])
    val_end = pd.Timestamp(oparams["val_end"])

    # ---- point of no return: atomic one-shot claim BEFORE any test row is read ----
    try:
        ctx.claim_lockbox()
    except PermissionError as exc:
        return {"ok": False, "error": f"refused: {exc}"}
    try:
        frame = pd.read_csv(panel_path)
        frame[cfg.date_col] = pd.to_datetime(frame[cfg.date_col])
        if frame.loc[frame[cfg.date_col] > val_end].empty:
            raise ValueError(f"panel has no test rows (date > {val_end.date()})")
        # compute every accepted factor column on the full panel
        for _, card in accepted:
            compiled = compile_expression(card["expression"])
            frame[card["name"]] = compiled.evaluate(frame, date_col=cfg.date_col,
                                                    asset_col=cfg.asset_col)
        # Same availability rule as the oracle: a training row whose label settles
        # after train_end carries post-cutoff prices, and a "test" row must be one whose
        # label actually settles in the test window.
        from factor_research.factor_validity_pipeline import label_end_series
        label_end = label_end_series(frame, context="open_test_lockbox",
                                     date_col=cfg.date_col, target_col=cfg.target_col)
        train = frame.loc[(frame[cfg.date_col] <= train_end) & (label_end <= train_end)]
        test = frame.loc[label_end > val_end]
        names = [c["name"] for _, c in accepted]

        def composite_sharpe(subset: list[str]) -> float | None:
            if not subset:
                return 0.0
            w_frame = fit_sparse_linear_weights(train.dropna(subset=[cfg.target_col]), subset, cfg)
            w = {r["factor"]: float(r["weight"]) for _, r in w_frame.iterrows() if float(r["weight"]) != 0.0}
            if not w:
                w = {f: 1.0 for f in subset}
            return _composite_ls_stats(test, w, cfg, cfg.long_short_quantile)["sharpe"]

        full_sharpe = composite_sharpe(names)
        results = []
        for path, card in accepted:
            loo = composite_sharpe([n for n in names if n != card["name"]])
            delta = None
            if full_sharpe is not None and loo is not None:
                delta = round(full_sharpe - loo, 4)
            card["test_sharpe_delta"] = delta
            path.write_text(json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")
            results.append({"name": card["name"], "test_sharpe_delta": delta})
        opened_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        report = {"ok": True, "opened_at": opened_at, "run_id": manifest["run_id"],
                  "panel_csv": str(panel_path), "panel_content_hash": panel_hash,
                  "test_composite_sharpe": full_sharpe, "cards": results,
                  "warning": "lockbox results are for the human report ONLY; never feed them back"}
        # Keep reports and evidence inside the owning run directory.
        (ctx.run_dir / "lockbox_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        ctx.set_state("TEST_OPENED", lockbox_opened_at=opened_at)   # one-shot, irreversible
        return report
    except Exception as exc:
        # strict one-shot: a failure AFTER the claim seals the run permanently
        ctx.set_state("TEST_OPEN_FAILED", error=str(exc))
        return {"ok": False, "error": f"lockbox opening failed after claim (run sealed as "
                                      f"TEST_OPEN_FAILED): {exc}"}


def _project_path(value: str) -> str:
    """README examples use project-relative paths; subprocesses may run with a different
    cwd, so anchor every relative path argument to the project directory."""
    path = Path(value)
    return str(path if path.is_absolute() else _PROJECT_DIR / path)


def build_panel(**cli_args) -> dict:
    """Wrap build_a_share_factor_panel.py (point-in-time panel builder) as a structured tool."""
    cmd = [sys.executable, "-m", "factor_research.build_a_share_factor_panel"]
    path_args = {"history_dir", "symbol_list", "output_dir"}
    for key, value in cli_args.items():
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                cmd += [flag]
        else:
            cmd += [flag, _project_path(str(value)) if key in path_args else str(value)]
    return _run_subprocess(cmd)


def miqp_recheck(*, panel_csv: str, train_end: str, run_dir: str | None = None,
                 cards_dir: str | None = None, config: str | None = None,
                 max_cardinality: int = 6, solver: str = "auto") -> dict:
    """SHELVED joint re-check: after K ACCEPTs, run solve_unified_v2.py over
    ACCEPT + SHELVED cards jointly to correct greedy path-dependence. DSL card factors are
    materialized onto an augmented panel first. Train-only (no lockbox risk)."""
    config = config or str(_PROJECT_DIR / "a_share_big_config.json")
    cfg = load_config(config)
    # Canonical cards belong to the selected run, unless the caller supplies cards_dir.
    if cards_dir:
        cards_path = Path(_project_path(cards_dir))
    elif run_dir:
        cards_path = Path(_project_path(run_dir)) / "cards"
    else:
        return {"ok": False, "error": "pass --run-dir runs/<run_id> (or --cards-dir)"}
    all_cards = (json.loads(p.read_text(encoding="utf-8"))
                 for p in sorted(cards_path.glob("*.json")))
    pool = [c for c in all_cards if c.get("decision") in ("ACCEPT", "SHELVED")]
    if len(pool) < 2:
        return {"ok": False, "error": f"need >= 2 ACCEPT/SHELVED cards, found {len(pool)}"}

    frame = pd.read_csv(panel_csv)
    frame[cfg.date_col] = pd.to_datetime(frame[cfg.date_col])
    names = []
    for card in pool:
        compiled = compile_expression(card["expression"])
        frame[card["name"]] = compiled.evaluate(frame, date_col=cfg.date_col,
                                                asset_col=cfg.asset_col)
        names.append(card["name"])

    work = DEFAULT_OUT / "miqp_recheck"
    work.mkdir(parents=True, exist_ok=True)
    aug_panel = work / "panel_augmented.csv"
    frame.to_csv(aug_panel, index=False, encoding="utf-8-sig")
    cfg_raw = json.loads(Path(config).read_text(encoding="utf-8"))
    cfg_raw["factor_cols"] = names
    cfg_raw["input_csv"] = str(aug_panel)
    cfg_raw["output_dir"] = str(work / "selection")
    for card in pool:
        cfg_raw["family_constraints"]["factor_families"][card["name"]] = card.get("family", "unclassified")
    aug_config = work / "config_augmented.json"
    aug_config.write_text(json.dumps(cfg_raw, indent=1, ensure_ascii=False), encoding="utf-8")

    # A hand-written selection CSV cannot satisfy the solver's provenance rules (no
    # manifest, no p-values) -- and should not: the recheck is a joint decision under the
    # SAME statistics. Run the real train-only pipeline on the augmented panel, then hand
    # its verified selection to the solver.
    pipeline = _run_subprocess([sys.executable, "-m", "factor_research.factor_validity_pipeline",
                                "--config", str(aug_config), "--train-end", train_end])
    if not pipeline["ok"]:
        pipeline["error"] = "train-only pipeline on the augmented panel failed"
        return pipeline

    cmd = [sys.executable, "-m", "factor_research.solve_unified_v2",
           "--panel-csv", str(aug_panel),
           "--selection-csv", str(work / "selection" / "final_factor_selection.csv"),
           "--config", str(aug_config), "--output-dir", str(work / "solve"),
           "--train-end", pd.Period(pd.Timestamp(train_end), "M").strftime("%Y-%m"),
           "--max-cardinality", str(max_cardinality), "--solver", solver]
    result = _run_subprocess(cmd)
    result["pool"] = [{"name": c["name"], "decision": c["decision"]} for c in pool]
    return result


def walkforward_report(*, panel_csv: str, config: str | None = None, confirm: str = "",
                       require_full_test_benchmark: bool = False, **cli_args) -> dict:
    """HUMAN CHECKPOINT: full walk-forward OOS report (hybrid_walkforward.py). Prints TEST Sharpe,
    so it is confirm-gated like the lockbox; results are for the final report, never for feedback."""
    if confirm != LOCKBOX_CONFIRM:
        return {"ok": False,
                "error": f"refused: walk-forward prints test-period results -- pass "
                         f"confirm='{LOCKBOX_CONFIRM}' at a human approval checkpoint only."}
    cmd = [sys.executable, "-m", "factor_research.hybrid_walkforward",
           "--panel-csv", _project_path(panel_csv),
           "--config", _project_path(config) if config else str(_PROJECT_DIR / "a_share_big_config.json")]
    if require_full_test_benchmark:
        # DATA_CONTRACT formal mode: a named parameter rather than **cli_args, so the
        # toolbox interface itself shows that a formal benchmark-relative run must ask
        # for full test coverage
        cmd += ["--require-full-test-benchmark"]
    path_keys = {"output_dir", "index_csv"}
    for key, value in cli_args.items():
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            # Boolean store_true options are bare flags and take no value.
            if value:
                cmd += [flag]
        else:
            cmd += [flag, _project_path(str(value)) if key in path_keys else str(value)]
    return _run_subprocess(cmd)


def _run_subprocess(cmd: list[str]) -> dict:
    import subprocess
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=_PROJECT_DIR,
                              timeout=3600)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ok": False, "cmd": cmd, "error": str(exc)}
    tail = "\n".join((proc.stdout or "").splitlines()[-25:])
    return {"ok": proc.returncode == 0, "cmd": cmd, "returncode": proc.returncode,
            "stdout_tail": tail, "stderr_tail": "\n".join((proc.stderr or "").splitlines()[-10:])}


TOOLS = {
    "check_expression": check_expression,
    "list_columns": list_columns,
    "validate_factor": validate_factor,
    "gate_certificate": gate_certificate,
    "memory_stats": memory_stats,
    "existing_factors": existing_factors,
    "open_test_lockbox": open_test_lockbox,
    "build_panel": build_panel,
    "miqp_recheck": miqp_recheck,
    "walkforward_report": walkforward_report,
}


def main() -> None:
    """Dispatch a declared structured tool and emit its JSON result and exit status.

    Convert CLI inputs to the tool public arguments. Test/lockbox operations keep
    their own explicit checkpoint checks rather than gaining access from dispatch.
    """
    ap = argparse.ArgumentParser(description="Agent toolbox: structured wrappers over the pipeline.")
    sub = ap.add_subparsers(dest="tool", required=True)

    p = sub.add_parser("check_expression", help="sandbox-compile an expression (no IC preview)")
    p.add_argument("expression")
    p.add_argument("--panel-csv", default=None)

    p = sub.add_parser("list_columns", help="allowed/forbidden panel columns for the DSL")
    p.add_argument("panel_csv")

    p = sub.add_parser("validate_factor", help="run the five-gate + DSR oracle on one candidate")
    p.add_argument("--name", required=True)
    p.add_argument("--expression", required=True)
    p.add_argument("--rationale", default="")
    p.add_argument("--family", default="unclassified")
    p.add_argument("--panel-csv", default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--existing", default="", help="comma-separated already-accepted factors")
    p.add_argument("--memory-dir", default=None)
    p.add_argument("--dsr-min", type=float, default=0.95)
    p.add_argument("--certificate", dest="certificate_path", default=None,
                   help="explicit gate-certificate path (default: the oracle's own)")

    p = sub.add_parser("gate_certificate", help="re-issue the process-level FPR/Power certificate")
    p.add_argument("--panel-csv", default=None)
    p.add_argument("--gen1-reps", type=int, default=60)
    p.add_argument("--gen2-reps", type=int, default=30)

    p = sub.add_parser("memory_stats", help="trial ledger stats (n_trials, N_eff, decisions)")
    p.add_argument("--memory-dir", default=None)

    p = sub.add_parser("existing_factors", help="read the validated set from a selection CSV")
    p.add_argument("selection_csv")

    p = sub.add_parser("open_test_lockbox", help="HUMAN ONLY: one-shot sealed-run test evaluation")
    p.add_argument("--run-dir", required=True, help="runs/<run_id> directory (must be COMPLETED)")
    p.add_argument("--panel-csv", default=None,
                   help="optional; must equal the run manifest panel (content-hash checked)")
    p.add_argument("--confirm", default="")

    p = sub.add_parser("build_panel", help="point-in-time A-share panel builder (see docs/DATA_CONTRACT.md)")
    p.add_argument("--history-dir", default=None)
    p.add_argument("--symbol-list", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--max-symbols", type=int, default=None)
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--freq", default=None, choices=["M", "W"])
    p.add_argument("--forward-days", type=int, default=None)
    p.add_argument("--fundamental-lag-days", type=int, default=None)
    p.add_argument("--min-monthly-rows", type=int, default=None)
    p.add_argument("--check-only", action="store_true", default=None)

    p = sub.add_parser("miqp_recheck", help="joint MIQP re-selection over ACCEPT+SHELVED cards")
    p.add_argument("--panel-csv", required=True)
    p.add_argument("--train-end", required=True,
                   help="selection fits only on dates <= this (same rule as the pipeline)")
    p.add_argument("--run-dir", default=None, help="runs/<run_id>; cards read from its cards/")
    p.add_argument("--config", default=None)
    p.add_argument("--cards-dir", default=None)
    p.add_argument("--max-cardinality", type=int, default=6)
    p.add_argument("--solver", default="auto",
                   choices=["auto", "MOSEK", "SCIP", "ECOS_BB"])

    p = sub.add_parser("walkforward_report", help="HUMAN ONLY: walk-forward OOS report (test Sharpe)")
    p.add_argument("--panel-csv", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--confirm", default="")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--index-csv", default=None)
    p.add_argument("--min-test-months", type=int, default=None)
    p.add_argument("--label-embargo-days", type=int, default=None)
    p.add_argument("--solver", default=None, choices=["auto", "MOSEK", "SCIP", "ECOS_BB"])
    p.add_argument("--overwrite-output", action="store_true", default=None)
    p.add_argument("--require-full-test-benchmark", action="store_true", default=None,
                   help="formal-report mode (DATA_CONTRACT): controlled non-zero exit "
                        "when any test month lacks a benchmark return")

    args = vars(ap.parse_args())
    tool = args.pop("tool")
    if tool == "validate_factor":
        args["existing"] = [f for f in args["existing"].split(",") if f.strip()]
    kwargs = {k: v for k, v in args.items() if v is not None}
    result = TOOLS[tool](**kwargs)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=_json_default))
    # A failed tool must fail the process: shells, CI and batch harnesses read the exit
    # code; a structured failure also requires a nonzero process exit.
    if isinstance(result, dict) and result.get("ok") is False:
        sys.exit(1)


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


if __name__ == "__main__":
    main()
