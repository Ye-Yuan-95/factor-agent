"""Artifact-based paired analysis for the frozen F4b prefix experiment.

The script reads only completed result artifacts. B12/B24 are prefixes of each B36 candidate
stream, so no additional run or random path is introduced during analysis.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

_PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from agent.certification import wilson_upper  # noqa: E402
from benchmark.f4b_artifact_audit import audit  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value: Any, length: int = 16) -> str:
    payload = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


def exact_sign_flip(differences: list[float]) -> dict[str, Any]:
    """One-sided exact paired randomisation test, statistic = mean difference."""
    values = [float(value) for value in differences]
    nonzero = [value for value in values if value != 0.0]
    observed = sum(nonzero) / len(nonzero) if nonzero else 0.0
    if not nonzero:
        p_value = 1.0
    else:
        extreme = 0
        total = 2 ** len(nonzero)
        tolerance = 1e-15
        for signs in itertools.product((-1.0, 1.0), repeat=len(nonzero)):
            statistic = sum(sign * value for sign, value in zip(signs, nonzero)) / len(nonzero)
            extreme += int(statistic >= observed - tolerance)
        p_value = extreme / total
    ordered = sorted(values)
    middle = len(ordered) // 2
    median = (
        ordered[middle] if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2
    ) if ordered else None
    return {
        "n_pairs": len(values),
        "n_nonzero": len(nonzero),
        "mean_difference": sum(values) / len(values) if values else None,
        "median_difference": float(median) if median is not None else None,
        "one_sided_exact_p": p_value,
        "statistic": "mean paired difference",
    }


def exact_paired_binary(left: list[int], right: list[int]) -> dict[str, Any]:
    """One-sided exact McNemar/binomial test for P(left discovery > right)."""
    wins = sum(int(a == 1 and b == 0) for a, b in zip(left, right))
    losses = sum(int(a == 0 and b == 1) for a, b in zip(left, right))
    discordant = wins + losses
    if discordant:
        p_value = sum(math.comb(discordant, k) for k in range(wins, discordant + 1))
        p_value /= 2 ** discordant
    else:
        p_value = 1.0
    return {
        "n_pairs": len(left),
        "left_only": wins,
        "right_only": losses,
        "discordant": discordant,
        "one_sided_exact_p": round(p_value, 9),
    }


def _holm_adjust(named_p: dict[str, float]) -> dict[str, float]:
    ordered = sorted(named_p.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (name, p_value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * p_value))
        adjusted[name] = round(running, 9)
    return adjusted


def _prefix_metrics(result: dict[str, Any], checkpoint: int) -> dict[str, Any]:
    """Compute metrics at an exact candidate budget prefix; refuse runs shorter than the checkpoint."""
    candidates = list(result["candidates"])[:checkpoint]
    if len(candidates) != checkpoint:
        raise RuntimeError(
            f"{result['scenario']} seed={result['seed']} arm={result['arm_id']} "
            f"has only {len(candidates)} candidates before checkpoint {checkpoint}")
    loadings = [float(candidate["truth_loading"]) for candidate in candidates
                if candidate.get("truth_loading") is not None]
    true_accepts = [
        candidate for candidate in candidates
        if candidate["score_decision"] == "ACCEPT" and candidate["is_true_candidate"]
    ]
    return {
        "scenario": result["scenario"],
        "seed": int(result["seed"]),
        "arm_id": result["arm_id"],
        "checkpoint": checkpoint,
        "max_truth_loading": max(loadings) if loadings else None,
        "discovery": int(bool(true_accepts)) if result["scenario"] == "alternative" else None,
        "n_true_accept": len(true_accepts) if result["scenario"] == "alternative" else None,
        "n_would_be_accept": sum(
            candidate["score_decision"] == "ACCEPT" for candidate in candidates),
    }


def _paired(rows: list[dict[str, Any]], *, left: str, right: str,
            checkpoint: int) -> dict[str, Any]:
    """Align alternative runs by seed for an arm comparison at one checkpoint.
    The seed, not each dependent candidate, is the paired unit."""
    selected = {
        (row["arm_id"], row["seed"]): row for row in rows
        if row["scenario"] == "alternative" and row["checkpoint"] == checkpoint
        and row["arm_id"] in {left, right}
    }
    seeds = sorted(seed for arm, seed in selected if arm == left)
    if any((right, seed) not in selected for seed in seeds):
        raise RuntimeError(f"unpaired comparison {left} vs {right} at B{checkpoint}")
    loading_differences = [
        selected[(left, seed)]["max_truth_loading"]
        - selected[(right, seed)]["max_truth_loading"]
        for seed in seeds
    ]
    left_discovery = [selected[(left, seed)]["discovery"] for seed in seeds]
    right_discovery = [selected[(right, seed)]["discovery"] for seed in seeds]
    return {
        "left": left,
        "right": right,
        "checkpoint": checkpoint,
        "seeds": seeds,
        "max_loading": exact_sign_flip(loading_differences),
        "discovery": exact_paired_binary(left_discovery, right_discovery),
        "left_discoveries": sum(left_discovery),
        "right_discoveries": sum(right_discovery),
    }


def _budget_trend(rows: list[dict[str, Any]], arm_id: str,
                  low: int, high: int) -> dict[str, Any]:
    """Compare low and high prefixes within each seed of one arm."""
    selected = {
        (row["checkpoint"], row["seed"]): row for row in rows
        if row["scenario"] == "alternative" and row["arm_id"] == arm_id
        and row["checkpoint"] in {low, high}
    }
    seeds = sorted(seed for checkpoint, seed in selected if checkpoint == high)
    differences = [
        selected[(high, seed)]["max_truth_loading"]
        - selected[(low, seed)]["max_truth_loading"]
        for seed in seeds
    ]
    high_discovery = [selected[(high, seed)]["discovery"] for seed in seeds]
    low_discovery = [selected[(low, seed)]["discovery"] for seed in seeds]
    return {
        "arm_id": arm_id,
        "contrast": f"B{high}-B{low}",
        "max_loading": exact_sign_flip(differences),
        "discovery": exact_paired_binary(high_discovery, low_discovery),
        "discoveries": {
            f"B{low}": sum(low_discovery),
            f"B{high}": sum(high_discovery),
        },
    }


def analyse(protocol_path: Path, output_dir: Path) -> dict[str, Any]:
    """Require planned run keys and compute prefix, paired-seed and safety metrics.

    Audit protocol identity, exact run keys and paired panels first. Use the primary
    paired comparison as declared and Holm-adjust the secondary family. Input
    hashes identify read artifacts; this analysis does not rerun the candidate gates.
    """
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    audit(protocol_path, output_dir)
    checkpoints = [int(value) for value in protocol["interpretation"]["checkpoints"]]
    arms = [entry["arm_id"] for entry in protocol["baseline_arms"]]
    expected = {
        (scenario, int(seed), arm_id)
        for scenario, seeds in (
            ("alternative", protocol["dgp"]["alternative_seeds"]),
            ("null", protocol["dgp"]["null_seeds"]),
        )
        for seed in seeds for arm_id in arms
    }
    results: dict[tuple[str, int, str], dict[str, Any]] = {}
    input_hashes: dict[str, str] = {}
    for path in sorted((output_dir / "runs").glob("*/*/result.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        key = (result["scenario"], int(result["seed"]), result["arm_id"])
        if key in expected:
            results[key] = result
            input_hashes[path.relative_to(output_dir).as_posix()] = _sha256(path)
    missing = sorted(expected - set(results))
    if missing:
        raise RuntimeError(f"F4b is incomplete; missing {len(missing)} runs, first={missing[:3]}")
    prefix_rows = [
        _prefix_metrics(result, checkpoint)
        for result in results.values()
        for checkpoint in checkpoints
    ]

    form_aware = "hybrid_form_aware_b36"
    unchanged = "hybrid_random_b36"
    form_blind = "form_blind_b36"
    primary = _paired(
        prefix_rows, left=form_aware, right=unchanged, checkpoint=36)
    secondary = _paired(
        prefix_rows, left=form_aware, right=form_blind, checkpoint=36)
    trends = [_budget_trend(prefix_rows, arm_id, 12, 36) for arm_id in arms]
    secondary_p = {
        "form_aware_vs_form_blind_b36":
            secondary["max_loading"]["one_sided_exact_p"],
        **{
            f"{trend['arm_id']}_b36_vs_b12":
                trend["max_loading"]["one_sided_exact_p"]
            for trend in trends
        },
    }
    holm = _holm_adjust(secondary_p)
    secondary["max_loading"]["holm_adjusted_p"] = holm[
        "form_aware_vs_form_blind_b36"]
    for trend in trends:
        trend["max_loading"]["holm_adjusted_p"] = holm[
            f"{trend['arm_id']}_b36_vs_b12"]

    alt_seeds = [int(seed) for seed in protocol["dgp"]["alternative_seeds"]]
    valid_differences = [
        float(results[("alternative", seed, form_aware)]["valid_candidate_rate"])
        - float(results[("alternative", seed, unchanged)]["valid_candidate_rate"])
        for seed in alt_seeds
    ]
    null_seeds = [int(seed) for seed in protocol["dgp"]["null_seeds"]]
    null_rate_differences = [
        (
            results[("null", seed, form_aware)]["n_would_be_accept"]
            / results[("null", seed, form_aware)]["oracle_trials"]
        ) - (
            results[("null", seed, unchanged)]["n_would_be_accept"]
            / results[("null", seed, unchanged)]["oracle_trials"]
        )
        for seed in null_seeds
    ]
    f4b_null_accepts = sum(
        results[("null", seed, form_aware)]["n_would_be_accept"] for seed in null_seeds)
    f4b_null_trials = sum(
        results[("null", seed, form_aware)]["oracle_trials"] for seed in null_seeds)
    valid_mean = sum(valid_differences) / len(valid_differences)
    null_mean = sum(null_rate_differences) / len(null_rate_differences)
    null_upper = wilson_upper(f4b_null_accepts, f4b_null_trials, 0.95)
    safety = {
        "valid_candidate_rate": {
            "mean_paired_difference": round(valid_mean, 9),
            "margin": -0.05,
            "passes": valid_mean >= -0.05,
        },
        "relative_null_accept_rate": {
            "mean_paired_difference": round(null_mean, 9),
            "margin": 0.02,
            "passes": null_mean <= 0.02,
        },
        "absolute_null_accept_rate": {
            "accepts": f4b_null_accepts,
            "trials": f4b_null_trials,
            "rate": round(f4b_null_accepts / f4b_null_trials, 9),
            "one_sided_wilson_upper_95": round(null_upper, 9),
            "target": 0.10,
            "passes": null_upper <= 0.10,
        },
    }
    safety_passes = all(item["passes"] for item in safety.values())
    primary_passes = (
        primary["max_loading"]["one_sided_exact_p"] <= 0.05
        and primary["max_loading"]["mean_difference"] > 0
    )
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        input_hashes["manifest.json"] = _sha256(manifest_path)
    try:
        protocol_key = protocol_path.resolve().relative_to(_PROJECT_DIR).as_posix()
    except ValueError:
        protocol_key = str(protocol_path.resolve())
    input_hashes[protocol_key] = _sha256(protocol_path)
    report = {
        "protocol_id": protocol["protocol_id"],
        "protocol_hash": _canonical_hash(protocol),
        "analysis_script": str(Path(__file__).resolve().relative_to(_PROJECT_DIR)),
        "analysis_script_sha256": _sha256(Path(__file__)),
        "input_sha256": dict(sorted(input_hashes.items())),
        "checkpoints": checkpoints,
        "prefix_metrics": sorted(
            prefix_rows,
            key=lambda row: (row["scenario"], row["seed"], row["arm_id"],
                             row["checkpoint"])),
        "primary_comparison": primary,
        "secondary_comparison": secondary,
        "budget_trends": trends,
        "safety_constraints": safety,
        "decision": {
            "primary_passes": primary_passes,
            "all_safety_constraints_pass": safety_passes,
            "status": "SUPPORTED" if primary_passes and safety_passes else "NOT_SUPPORTED",
        },
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    return report


def main() -> None:
    """Analyze the complete F4b artifact set and save the paired-test report.

    The analysis requires the protocol expected arms and seeds; missing runs
    are refused rather than silently changing the comparison population.
    """
    parser = argparse.ArgumentParser(description="Analyse frozen F4b paired prefix artifacts")
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
    report = analyse(Path(args.protocol).resolve(), output_dir)
    report_path = output_dir / "f4b_paired_analysis.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {report_path}")
    print(json.dumps(report["decision"], ensure_ascii=False))


if __name__ == "__main__":
    main()
