"""Independent integrity audit for completed F4b run artifacts.

This does not calculate scientific endpoints. It verifies that the artifact set consumed by
``f4b_analysis.py`` is complete, paired, budget-conformant, and closed over final feedback.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_PROJECT_DIR = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(protocol_path: Path, output_dir: Path) -> dict[str, Any]:
    """Check F4b completeness, budget invariants and recorded identity consistency.

    Refuse an incomplete or inconsistent run set and return checks plus hashes of
    the artifacts read. Recorded manifest identities are checked as metadata;
    this audit does not reconstruct an archived source tree or rerun its gates.
    """
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result_paths = sorted((output_dir / "runs").glob("*/*/result.json"))
    results = [json.loads(path.read_text(encoding="utf-8")) for path in result_paths]
    arms = [entry["arm_id"] for entry in protocol["baseline_arms"]]
    expected = {
        (scenario, int(seed), arm)
        for scenario, seeds in (
            ("alternative", protocol["dgp"]["alternative_seeds"]),
            ("null", protocol["dgp"]["null_seeds"]),
        )
        for seed in seeds for arm in arms
    }
    actual = {
        (result["scenario"], int(result["seed"]), result["arm_id"])
        for result in results
    }
    checks: dict[str, bool] = {
        "exact_expected_run_set": actual == expected and len(results) == len(expected),
        "protocol_id_consistent": {
            result["protocol_id"] for result in results
        } == {protocol["protocol_id"]},
        "protocol_hash_matches_manifest": manifest.get("protocol_hash") == hashlib.sha256(
            json.dumps(protocol, sort_keys=True, ensure_ascii=False,
                       separators=(",", ":"), default=str).encode("utf-8")
        ).hexdigest()[:16],
        "oracle_budget_exact": all(
            result["oracle_trials"] == result["oracle_trial_budget"] == 36
            for result in results),
        "generation_budget_respected": all(
            result["generation_attempts"] <= result["generation_attempt_budget"] == 72
            for result in results),
        "candidate_count_matches_trials": all(
            len(result["candidates"]) == result["oracle_trials"] for result in results),
        # Failed proposals are permitted by the generation budget; they must be
        # counted rather than silently requiring the historical zero-failure case.
        "generation_attempts_accounted": all(
            result["response_local_failures"] >= 0
            and result["runtime_coverage_failures"] >= 0
            and result["oracle_trials"] + result["response_local_failures"]
            + result["runtime_coverage_failures"] == result["generation_attempts"]
            for result in results),
    }
    paired: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        paired[(result["scenario"], int(result["seed"]))].append(result)
    checks["four_arms_per_panel"] = (
        len(paired) == len(protocol["dgp"]["alternative_seeds"])
        + len(protocol["dgp"]["null_seeds"])
        and all(len(group) == len(arms) for group in paired.values())
    )
    checks["paired_panel_hashes_identical"] = all(
        len({result["panel_content_hash"] for result in group}) == 1
        for group in paired.values())
    hybrid_results = [
        result for result in results
        if result["arm_kind"] in {"hybrid_random", "hybrid_form_aware"}
    ]
    checks["hybrid_final_feedback_absorbed"] = all(
        result["generator_diagnostics"]["n_pending_feedback"] == 0
        for result in hybrid_results)
    checks["manifest_commit_is_full_hash"] = (
        len(manifest["frozen_inputs"]["commit"]) == 40
        and manifest["git"]["commit"] == manifest["frozen_inputs"]["commit"])
    checks["manifest_relevant_paths_clean"] = bool(
        manifest["frozen_inputs"]["relevant_paths_clean"])
    error_logs = sorted((output_dir / "worker_logs").glob("*.err.log"))
    # Shell redirection/parallel-worker logs are optional diagnostics. The public
    # sequential runner does not create them; result integrity cannot depend on
    # an old launcher's log naming convention or on warnings being absent.
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"F4b artifact audit failed: {failed}")
    priority_picks = sum(
        result["generator_diagnostics"].get("form_priority_picks", 0)
        for result in results if result["arm_kind"] == "hybrid_form_aware")
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "status": "PASS",
        "protocol_id": protocol["protocol_id"],
        "frozen_git_commit": manifest["frozen_inputs"]["commit"],
        "protocol_blob": manifest["frozen_inputs"]["protocol_blob"],
        "source_fingerprint": manifest["source_fingerprint"],
        "checks": checks,
        "worker_logs": {
            "available": len(error_logs),
            "nonempty": [path.relative_to(output_dir).as_posix()
                         for path in error_logs if path.stat().st_size],
        },
        "n_results": len(results),
        "arm_counts": dict(sorted(Counter(result["arm_id"] for result in results).items())),
        "scenario_counts": dict(sorted(
            Counter(result["scenario"] for result in results).items())),
        "n_paired_panels": len(paired),
        "total_oracle_trials": sum(result["oracle_trials"] for result in results),
        "total_generation_attempts": sum(
            result["generation_attempts"] for result in results),
        "form_priority_picks": priority_picks,
        "result_sha256": {
            path.relative_to(output_dir).as_posix(): _sha256(path)
            for path in result_paths
        },
        "manifest_sha256": _sha256(manifest_path),
        "audit_script": str(Path(__file__).resolve().relative_to(_PROJECT_DIR)),
        "audit_script_sha256": _sha256(Path(__file__)),
    }


def main() -> None:
    """Verify F4b run integrity and write the current hash-bound audit report.

    Require the declared paired runs, source/protocol bindings and budget checks
    before writing artifact_audit.json. A failed check raises instead of recording
    a successful audit status.
    """
    parser = argparse.ArgumentParser(description="Audit completed F4b artifacts")
    parser.add_argument(
        "--protocol",
        default=str(_PROJECT_DIR / "experiments"
                    / "hybrid_form_aware_2026-07-31_F4b.json"))
    parser.add_argument(
        "--output-dir",
        default=str(_PROJECT_DIR / "outputs" / "agent"
                    / "hybrid_form_aware_20260731_F4b"))
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    report = audit(Path(args.protocol).resolve(), output_dir)
    report_path = output_dir / "artifact_audit.json"
    with report_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(report, indent=2, ensure_ascii=False))
        handle.write("\n")
    print(f"wrote {report_path}")
    print(json.dumps({"status": report["status"], "checks": report["checks"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
