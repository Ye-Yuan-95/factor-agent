"""L1 factor-research runner: static candidates or a natural-language GLM request.

The LLM is only a hypothesis generator.  Host code owns the research scope, candidate names,
budgets, DSL sandbox, family assignment, Oracle gates, certificates and user-facing conclusions.
Every run remains train/validation only; the test lockbox is opened only by the separate one-shot
lockbox command.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_PROJECT_DIR = Path(__file__).resolve().parents[1]
for p in (str(_PROJECT_DIR), str(_PROJECT_DIR / "synthetic")):
    if p not in sys.path:
        sys.path.insert(0, p)

from agent.factor_dsl import DSLError, compile_expression  # noqa: E402
from agent.generator_contract import (  # noqa: E402
    GeneratorSpec,
    ResearchRequest,
    assign_expression_family,
)
from agent.llm_generator import (  # noqa: E402
    GLMClient,
    GenerationError,
    LLMGenerator,
    interpret_research_request,
)
from agent.memory import MemoryStore  # noqa: E402
from agent.run_context import RunContext  # noqa: E402
from agent.user_report import write_user_report  # noqa: E402
from agent.validator_oracle import ValidatorOracle  # noqa: E402


_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def make_synthetic_panel(kind: str, panel_csv: str, config_path: str, *, n_assets: int,
                         n_months: int, inject_ic: float, seed: int):
    """Generate a Gen-1 or Gen-2 panel calibrated to the real panel."""
    from stage_a_fpr_power import calibrate, generate_panel
    cfg_raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
    factors = cfg_raw["factor_cols"]
    fam_map = cfg_raw["family_constraints"]["factor_families"]
    rng = np.random.default_rng(seed)
    true_idx: list[int] = []
    if inject_ic > 0:
        seen: set[str] = set()
        for j, factor in enumerate(factors):
            family = fam_map.get(factor)
            if family not in seen:
                seen.add(family)
                true_idx.append(j)
            if len(true_idx) >= 4:
                break
    strengths = [inject_ic] * len(true_idx)
    if kind == "gen1":
        calib = calibrate(Path(panel_csv), factors)
        panel, truth = generate_panel(
            rng, calib, n_assets, n_months, true_idx, strengths, tails="t")
    elif kind == "gen2":
        from dgp_v2 import calibrate_v2, generate_panel_v2
        calib = calibrate_v2(panel_csv, factors, fam_map)
        panel, truth = generate_panel_v2(
            rng, calib, n_assets, n_months, true_idx, strengths,
            coverage=calib.get("coverage", 0.973))
    else:
        raise ValueError(
            f"unknown synthetic kind {kind!r} (gen1|gen2; gen3 is not an agent-trial DGP)")
    names = calib["factors"]
    truth_names = [names[j] for j in range(len(names)) if truth[j]]
    return panel, truth_names


def main() -> None:
    """Freeze the request, run file or LLM proposals and persist candidate decisions.

    Resolve the active panel and vocabulary before generation. The host enforces
    attempt/trial budgets and writes run-scoped cards, memory and a user report;
    this entry point never opens the test lockbox.
    """
    ap = argparse.ArgumentParser(
        description="L1: candidates JSON or natural-language request -> FactorCards + user report")
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--candidates", help="JSON list of pre-built candidate factors")
    source.add_argument("--query", help="natural-language factor-screening request")
    source.add_argument("--request-json", help="frozen ResearchRequest JSON; GLM proposes factors")
    ap.add_argument("--generator", choices=["file", "glm"], default=None,
                    help="normally inferred from the selected input source")
    ap.add_argument("--panel-csv", default=str(
        _PROJECT_DIR / "data_big_v2" / "a_share_factor_panel.csv"))
    ap.add_argument("--config", default=str(_PROJECT_DIR / "a_share_big_config.json"))
    ap.add_argument("--existing", default="",
                    help="comma-separated factors, or a final_factor_selection.csv path")
    ap.add_argument("--memory-dir", default=None,
                    help="override; default = this run's runs/<run_id>/memory")
    ap.add_argument("--cards-dir", default=None,
                    help="optional duplicate card output; canonical cards remain run-scoped")
    ap.add_argument("--runs-root", default=None,
                    help="root for runs/<run_id> (default: project runs/)")
    ap.add_argument("--budget", type=int, default=None,
                    help="Oracle trial budget (default: 100 file mode, 12 GLM mode)")
    ap.add_argument("--generation-attempt-budget", type=int, default=None,
                    help="GLM attempts including malformed/duplicate proposals (default: 24)")
    ap.add_argument("--families", default="",
                    help="hard family allowlist for --query, comma-separated")
    ap.add_argument("--model", default="glm-4.7-flash")
    ap.add_argument("--prompt-version", default="glm-b1-v3-conditional")
    ap.add_argument("--llm-max-tokens", type=int, default=400)
    ap.add_argument("--llm-max-retries", type=int, default=2)
    ap.add_argument("--dsr-min", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--vocabulary", choices=["base", "extended"], default="base",
                    help="generator leaf columns: base = frozen 44 factor cols; extended adds "
                         "PIT raw columns present in the supplied panel")
    ap.add_argument("--synthetic", choices=["gen1", "gen2"], default=None,
                    help="run the same loop on a synthetic panel with known truth")
    ap.add_argument("--synthetic-assets", type=int, default=150)
    ap.add_argument("--synthetic-months", type=int, default=150)
    ap.add_argument("--inject-ic", type=float, default=0.0,
                    help="synthetic mode: inject 4 cross-family true factors at this IC")
    args = ap.parse_args()

    mode = "file" if args.candidates else "glm"
    if args.generator is not None and args.generator != mode:
        ap.error(f"--generator {args.generator} conflicts with the selected input source ({mode})")
    if args.budget is not None and args.budget <= 0:
        ap.error("--budget must be positive")
    if args.generation_attempt_budget is not None and args.generation_attempt_budget <= 0:
        ap.error("--generation-attempt-budget must be positive")

    config_file = Path(args.config)
    if not config_file.is_file():
        raise SystemExit(
            f"config not found: {config_file}\n"
            "The default config ships with the repository checkout, not the installed wheel; "
            "run from a checkout or pass --config explicitly.")
    config_raw = json.loads(config_file.read_text(encoding="utf-8"))
    spec = None  # built after the panel is known, so 'extended' can intersect real columns
    existing = _parse_existing(args.existing)
    candidates: list[dict[str, Any]] = []
    if args.candidates:
        candidates = json.loads(Path(args.candidates).read_text(encoding="utf-8"))
        if not isinstance(candidates, list):
            raise SystemExit("candidates file must be a JSON list")
        _validate_static_candidate_identities(candidates)

    oracle = ValidatorOracle(args.config, panel_csv=args.panel_csv, dsr_min=args.dsr_min)
    truth_names: list[str] = []
    universe_note = Path(args.panel_csv).parent.name
    if args.synthetic:
        panel, truth_names = make_synthetic_panel(
            args.synthetic, args.panel_csv, args.config, n_assets=args.synthetic_assets,
            n_months=args.synthetic_months, inject_ic=args.inject_ic, seed=args.seed)
        universe_note = f"synthetic_{args.synthetic}_seed{args.seed}_ic{args.inject_ic:g}"
        dates = pd.to_datetime(sorted(pd.unique(panel["date"])))
        oracle.val_end = pd.Timestamp(dates[-1])
        oracle.train_end = pd.Timestamp(dates[int(len(dates) * 0.7) - 1])
        oracle.set_panel(panel, universe_note, label_semantics="direct")
        print(f"[synthetic {args.synthetic}] {args.synthetic_assets} assets x "
              f"{args.synthetic_months} months, train<= {oracle.train_end.date()}, "
              f"true={truth_names or 'none (null market)'}")

    # vocabulary is resolved against the ACTIVE panel's columns
    if args.synthetic:
        panel_columns = list(panel.columns)
    else:
        panel_columns = list(pd.read_csv(args.panel_csv, nrows=0).columns)
    from agent.generator_contract import build_vocabulary
    spec = GeneratorSpec.from_config(
        config_raw, columns=build_vocabulary(config_raw, mode=args.vocabulary,
                                             panel_columns=panel_columns))

    client: GLMClient | None = None
    request_call: dict[str, Any] | None = None
    if mode == "glm":
        try:
            client = GLMClient(
                model=args.model, max_tokens=args.llm_max_tokens,
                max_retries=args.llm_max_retries)
        except GenerationError as exc:
            raise SystemExit(str(exc)) from exc
        if args.query:
            generation_budget = args.generation_attempt_budget or 24
            oracle_budget = args.budget or 12
            if generation_budget < oracle_budget:
                ap.error("generation-attempt budget must be >= Oracle trial budget")
            try:
                request, call = interpret_research_request(
                    query=args.query, client=client, spec=spec, universe=universe_note,
                    existing_factors=existing, hard_allowed_families=_csv_list(args.families),
                    generation_attempt_budget=generation_budget,
                    oracle_trial_budget=oracle_budget)
            except (GenerationError, ValueError) as exc:
                raise SystemExit(f"research-request interpretation failed: {exc}") from exc
            request_call = call.trace_dict()
        else:
            request = ResearchRequest.from_dict(
                json.loads(Path(args.request_json).read_text(encoding="utf-8")))
            if request.universe != universe_note:
                raise SystemExit(
                    "request universe does not match the active panel: "
                    f"{request.universe!r} != {universe_note!r}")
            if args.budget is not None and args.budget != request.oracle_trial_budget:
                raise SystemExit("--budget conflicts with oracle_trial_budget in --request-json")
            if (args.generation_attempt_budget is not None and
                    args.generation_attempt_budget != request.generation_attempt_budget):
                raise SystemExit(
                    "--generation-attempt-budget conflicts with --request-json")
            if _csv_list(args.families):
                raise SystemExit("--families cannot override a frozen --request-json")
            oracle_budget = request.oracle_trial_budget
            generation_budget = request.generation_attempt_budget
    else:
        oracle_budget = args.budget or 100
        generation_budget = max(args.generation_attempt_budget or len(candidates) or 1,
                                oracle_budget)
        request = ResearchRequest.from_text(
            f"Validate the candidate file {Path(args.candidates).name}",
            universe=universe_note, existing_factors=existing,
            generation_attempt_budget=generation_budget, oracle_trial_budget=oracle_budget)

    provider_snapshot = None if client is None else {
        "provider": client.provider,
        "model": client.model,
        "parameters": client.parameters,
        "prompt_version": args.prompt_version,
    }
    ctx = RunContext.create(
        purpose="l1_runner", universe=universe_note,
        config_snapshot={
            "pipeline_config": config_raw,
            "oracle_gate_params": oracle.oracle_gate_params(),
            "research_request": request.to_dict(),
            "generator_spec": spec.to_dict(),
            "generator": provider_snapshot or {"provider": "file"},
            "cli_args": vars(args),
        },
        manifest_extra={
            "panel_csv": str(oracle.panel_csv), "budget": oracle_budget,
            "generation_attempt_budget": generation_budget,
            "seed": args.seed, "synthetic": args.synthetic,
        },
        runs_root=args.runs_root)
    print(f"[run] {ctx.run_dir}")
    if request_call is not None:
        ctx.trace("research_request_interpreted", **request_call,
                  prompt_version=args.prompt_version,
                  generator_spec_hash=spec.fingerprint())

    memory = MemoryStore(args.memory_dir or ctx.memory_dir)
    extra_cards_dir = Path(args.cards_dir) if args.cards_dir else None
    if extra_cards_dir:
        extra_cards_dir.mkdir(parents=True, exist_ok=True)

    log: list[dict[str, Any]] = []
    cards: list[dict[str, Any]] = []
    generator_metadata: dict[str, dict[str, Any]] = {}
    history: list[dict[str, Any]] = []
    n_before = memory.n_trials()
    generation_attempts = 0
    generation_failures = 0

    if mode == "glm":
        assert client is not None
        generator = LLMGenerator(
            client=client, spec=spec, prompt_version=args.prompt_version)
        while (generation_attempts < generation_budget and
               memory.n_trials() < oracle_budget):
            generation_attempts += 1
            try:
                proposal = generator.propose(request, history=history)
            except GenerationError as exc:
                generation_failures += 1
                failure = {
                    **exc.trace,
                    "attempt": generation_attempts,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "provider": client.provider,
                    "model": client.model,
                    "prompt_version": args.prompt_version,
                }
                ctx.trace("generation_failure", **failure)
                log.append({"name": None, "outcome": "generation_failure", **failure})
                print(f"[generation-fail {generation_attempts}/{generation_budget}] "
                      f"{type(exc).__name__}: {exc}")
                continue
            name = f"llm_{generation_attempts:04d}"
            candidate = {
                "name": name,
                "expression": proposal["expression"],
                "rationale": proposal["rationale"],
                "family": proposal["family"],
            }
            generator_metadata[name] = proposal["generation"]
            ctx.trace(
                "candidate_generated", name=name,
                expression_signature=proposal["expression_signature"],
                family=proposal["family"],
                family_exposures=proposal["family_exposures"],
                **proposal["generation"])
            entry, card_dict, existing = _validate_candidate(
                candidate=candidate, oracle=oracle, memory=memory, existing=existing,
                ctx=ctx, universe_note=universe_note, truth_names=truth_names,
                extra_cards_dir=extra_cards_dir, seed=args.seed, spec=spec)
            log.append(entry)
            if card_dict is not None:
                cards.append(card_dict)
            history.append(_feedback(entry, proposal["expression_signature"]))
    else:
        for candidate in candidates:
            if memory.n_trials() >= oracle_budget:
                print(f"[budget] Oracle trial budget {oracle_budget} reached -- stopping "
                      f"(counted trials: {memory.n_trials()})")
                break
            generation_attempts += 1
            entry, card_dict, existing = _validate_candidate(
                candidate=candidate, oracle=oracle, memory=memory, existing=existing,
                ctx=ctx, universe_note=universe_note, truth_names=truth_names,
                extra_cards_dir=extra_cards_dir, seed=args.seed, spec=spec)
            log.append(entry)
            if card_dict is not None:
                cards.append(card_dict)

    stats = memory.stats()
    summary: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "universe": universe_note,
        "input_mode": mode,
        "oracle_trial_budget": oracle_budget,
        "generation_attempt_budget": generation_budget,
        "generation_attempts": generation_attempts,
        "generation_failures": generation_failures,
        "n_trials_before": n_before,
        "n_trials_after": stats["n_trials"],
        "n_eff": stats["n_eff"],
        "decisions_this_run": _count(log, "outcome"),
        "existing_after": existing,
        "candidates_log": log,
        "research_request": request.to_dict(),
    }
    if truth_names:
        accepted = [e["name"] for e in log if e["outcome"] == "ACCEPT"]
        true_hits = [e["name"] for e in log
                     if e["outcome"] == "ACCEPT" and e.get("is_true_factor")]
        summary["synthetic_truth"] = {
            "true_factors": truth_names,
            "accepted": accepted,
            "accepted_loading_on_truth": true_hits,
            "realized_false_share": (
                round(1.0 - len(true_hits) / len(accepted), 3) if accepted else None),
        }
        print(f"[truth] accepted={len(accepted)}, on-truth={len(true_hits)}, "
              f"realized false share={summary['synthetic_truth']['realized_false_share']}")

    (ctx.run_dir / "run_log.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    report_paths = write_user_report(
        run_dir=ctx.run_dir, run_id=ctx.run_dir.name, request=request,
        cards=cards, spec=spec, generator_metadata=generator_metadata,
        data_version=_first_data_version(cards))
    if getattr(oracle, "last_certificate_refs", None):
        ctx.write_certificate_refs(oracle.last_certificate_refs)
    ctx.set_state("COMPLETED", n_trials=stats["n_trials"],
                  generation_attempts=generation_attempts)
    print(f"\ntrials: {n_before} -> {stats['n_trials']} (N_eff={stats['n_eff']}); "
          f"run dir: {ctx.run_dir}")
    print(f"user report: {report_paths['user_report']}")


def _validate_candidate(*, candidate: dict[str, Any], oracle: ValidatorOracle,
                        memory: MemoryStore, existing: list[str], ctx: RunContext,
                        universe_note: str, truth_names: list[str],
                        extra_cards_dir: Path | None, seed: int, spec: GeneratorSpec
                        ) -> tuple[dict[str, Any], dict[str, Any] | None, list[str]]:
    """Assign host identity/family, validate one proposal and persist its artifacts.

    Sandbox or coverage failures produce a failure entry without a card/trial.
    A validated candidate produces a run-scoped card and ledger entry; only ACCEPT
    extends the control set returned to later candidates. Extra card output is a
    duplicate of the canonical run card, not a separate validation decision.
    """
    if not isinstance(candidate, dict):
        raise SystemExit("each candidate must be a JSON object")
    if "name" not in candidate or "expression" not in candidate:
        raise SystemExit("each candidate requires name and expression")
    name = str(candidate["name"])
    if not _SAFE_NAME.fullmatch(name):
        raise SystemExit(
            f"unsafe candidate name {name!r}; use letters, digits, dot, underscore or hyphen")
    expression = str(candidate["expression"])
    try:
        assignment = assign_expression_family(expression, spec)
    except (DSLError, ValueError) as exc:
        memory.record_failure(name=name, expression=expression, error=str(exc),
                              universe=universe_note)
        ctx.trace("sandbox_failure", name=name, error=str(exc))
        print(f"[sandbox-fail] {name:24s} NOT counted as a trial: {exc}")
        return ({"name": name, "outcome": "sandbox_failure", "error": str(exc)},
                None, existing)
    supplied_family = str(candidate.get("family", "unclassified"))
    if supplied_family != assignment.primary:
        ctx.trace("family_normalized", name=name, supplied=supplied_family,
                  assigned=assignment.primary, exposures=list(assignment.exposures))
    try:
        card = oracle.validate(
            name=name, expression=expression, rationale=str(candidate.get("rationale", "")),
            family=assignment.primary, existing=existing,
            memory=memory, seed=seed)
    except DSLError as exc:
        memory.record_failure(name=name, expression=expression, error=str(exc),
                              universe=universe_note)
        ctx.trace("sandbox_failure", name=name, error=str(exc))
        print(f"[sandbox-fail] {name:24s} NOT counted as a trial: {exc}")
        return ({"name": name, "outcome": "sandbox_failure", "error": str(exc)},
                None, existing)

    card_dict = card.to_dict()
    ctx.write_card(name, card_dict)
    if extra_cards_dir:
        (extra_cards_dir / f"{name}.json").write_text(
            json.dumps(card_dict, indent=2, ensure_ascii=False), encoding="utf-8")
    ctx.trace("validated", name=name, decision=card.decision, certified=card.certified,
              reject_reason=card.reject_reason)
    if card.decision == "ACCEPT":
        existing = existing + [name]
    val_delta = "n/a" if card.val_sharpe_delta is None else f"{card.val_sharpe_delta:+.3f}"
    dsr = "n/a" if card.deflated_sharpe is None else f"{card.deflated_sharpe:.3f}"
    incremental_t = "nan" if card.incremental_t is None else f"{card.incremental_t:.2f}"
    print(f"[{card.decision:>7}] {name:24s} q={card.fdr_q:.3f} inc_t={incremental_t} "
          f"DSR={dsr} val_dSR={val_delta}"
          + (f"  reason={card.reject_reason}" if card.reject_reason else ""))
    entry = {
        "name": name,
        "columns_used": sorted(compile_expression(expression).columns),
        "outcome": card.decision,
        "reject_reason": card.reject_reason,
        "failed_gates": _failed_gates(card_dict),
        "is_true_factor": _is_true(card_dict, truth_names) if truth_names else None,
    }
    return entry, card_dict, existing


def _feedback(entry: dict[str, Any], expression_signature: str) -> dict[str, Any]:
    """Allowlisted train/validation feedback; no metric or test field reaches the model."""
    return {
        "name": entry.get("name"),
        "columns_used": entry.get("columns_used", []),
        "expression_signature": expression_signature,
        "outcome": entry.get("outcome"),
        "failed_gates": entry.get("failed_gates", []),
        "reject_reason": entry.get("reject_reason") or entry.get("error"),
    }


def _failed_gates(card: dict[str, Any]) -> list[str]:
    checks = (
        ("fdr", card.get("fdr_pass")),
        ("redundancy", card.get("redundancy_pass")),
        ("survival", card.get("survival_pass")),
        ("incremental", card.get("incremental_pass")),
        ("sparse_weight", card.get("selected_by_weight")),
        ("dsr", card.get("dsr_pass")),
    )
    return [name for name, passed in checks if passed is not True]


def _parse_existing(spec: str) -> list[str]:
    spec = spec.strip()
    if not spec:
        return []
    if spec.lower().endswith(".csv"):
        selection = pd.read_csv(spec)
        column = "selected_final" if "selected_final" in selection.columns else "passes_incremental"
        return selection.loc[selection[column] == True, "factor"].astype(str).tolist()  # noqa: E712
    return _csv_list(spec)


def _validate_static_candidate_identities(candidates: list[Any]) -> None:
    """Fail before run creation when file-mode identities are missing, unsafe or ambiguous.

    A candidate name is the factor identity used by Memory, the Oracle, card filenames and the
    user report.  Collision suffixes are suitable as a last-resort write guard, but they cannot
    repair an ambiguous research identity.  Windows paths are case-insensitive, so names that
    differ only by case are duplicates here as well.
    """
    seen: dict[str, str] = {}
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise SystemExit(f"candidate at index {index} must be a JSON object")
        if "name" not in candidate or "expression" not in candidate:
            raise SystemExit(f"candidate at index {index} requires name and expression")
        name = str(candidate["name"])
        if not _SAFE_NAME.fullmatch(name):
            raise SystemExit(
                f"unsafe candidate name {name!r} at index {index}; use letters, digits, "
                "dot, underscore or hyphen")
        identity_key = name.casefold()
        if identity_key in seen:
            raise SystemExit(
                f"duplicate candidate name {name!r} at index {index}; it conflicts with "
                f"{seen[identity_key]!r}. Static candidate names must be unique within a run")
        seen[identity_key] = name


def _csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _is_true(card: dict[str, Any], truth_names: list[str]) -> bool:
    # Token-level match: substring matching counted rank(std_dolvol) as a hit for "dolvol",
    # inflating the truth summary. Identifiers must match as whole names.
    tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", card.get("expression", "")))
    return any(name in tokens for name in truth_names)


def _count(log: list[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for entry in log:
        value = str(entry.get(key, "unknown"))
        out[value] = out.get(value, 0) + 1
    return out


def _first_data_version(cards: list[dict[str, Any]]) -> str | None:
    for card in cards:
        if card.get("data_version"):
            return str(card["data_version"])
    return None


if __name__ == "__main__":
    main()
