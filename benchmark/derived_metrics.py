"""Read completed protocol artifacts and report descriptive continuous metrics.

Result and summary files are immutable evidence; outputs go to derived/.
Generation endpoints include maximum truth loading and prefix-best curves.
Validation endpoints include discoveries and true/false/null accepts. A high
loading candidate that fails validation is not an accepted discovery.

Continuous metrics are descriptive for protocols with a binary primary endpoint.
An inferential primary endpoint must be declared in the protocol before evaluation.
Paired comparisons use seeds, not individual trials, as independent units."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

_PROJECT_DIR = Path(__file__).resolve().parents[1]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_arm_results(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Read every result.json under run_dir, plus a hash of each input file."""
    paths = sorted((Path(run_dir) / "runs").glob("*/*/result.json"))
    if not paths:
        raise FileNotFoundError(f"no completed arms under {run_dir}")
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    hashes = {path.relative_to(run_dir).as_posix(): _file_sha256(path)[:16] for path in paths}
    return rows, hashes


def prefix_best(candidates: list[dict[str, Any]]) -> list[float | None]:
    """Best truth loading achieved after each successive trial (cumulative max).

    This is the shape that answers "does more budget help", which a single end-of-run number
    cannot: a generator that finds its best candidate in the first three trials and then
    plateaus looks identical to one that improves steadily, unless the curve is reported.
    """
    best = 0.0
    curve: list[float | None] = []
    for candidate in candidates:
        loading = candidate.get("truth_loading")
        if loading is not None:
            best = max(best, float(loading))
        curve.append(round(best, 6))
    return curve


def summarise_run(run_dir: Path) -> dict[str, Any]:
    """Derive pair coverage and exploration metrics from saved arms, preserving input hashes."""
    rows, input_hashes = load_arm_results(Path(run_dir))
    by_arm: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        if row["scenario"] != "alternative":
            continue
        key = (row["arm_id"], int(row["oracle_trial_budget"]))
        entry = by_arm.setdefault(key, {
            "arm_id": row["arm_id"], "arm_kind": row["arm_kind"],
            "budget": int(row["oracle_trial_budget"]), "seeds": {},
        })
        loadings = [c["truth_loading"] for c in row["candidates"]
                    if c.get("truth_loading") is not None]
        entry["seeds"][str(row["seed"])] = {
            # generation endpoint
            "max_truth_loading": round(max(loadings), 6) if loadings else None,
            "prefix_best": prefix_best(row["candidates"]),
            "n_candidates": len(row["candidates"]),
            # validation endpoint -- reported alongside, never replaced
            "discovery": row.get("discovery"),
            "n_true_accept": row.get("n_true_accept"),
            "n_false_accept": row.get("n_false_accept"),
            "n_would_be_accept": row.get("n_would_be_accept"),
            "oracle_trials": row.get("oracle_trials"),
        }
    null_rows = [row for row in rows if row["scenario"] == "null"]
    arms = []
    for entry in by_arm.values():
        values = [seed["max_truth_loading"] for seed in entry["seeds"].values()
                  if seed["max_truth_loading"] is not None]
        arms.append({
            **{k: v for k, v in entry.items() if k != "seeds"},
            "n_seeds": len(entry["seeds"]),
            "median_max_loading": round(float(np.median(values)), 6) if values else None,
            "min_max_loading": round(float(np.min(values)), 6) if values else None,
            "max_max_loading": round(float(np.max(values)), 6) if values else None,
            "discoveries": sum(int(bool(s["discovery"])) for s in entry["seeds"].values()),
            "true_accepts": sum(int(s["n_true_accept"] or 0) for s in entry["seeds"].values()),
            "false_accepts": sum(int(s["n_false_accept"] or 0) for s in entry["seeds"].values()),
            "per_seed": entry["seeds"],
        })
    arms.sort(key=lambda a: (a["arm_kind"], a["budget"]))
    return {
        "run_dir": str(run_dir),
        "arms": arms,
        "null_accepts_pooled": sum(int(row.get("n_would_be_accept") or 0) for row in null_rows),
        "null_trials_pooled": sum(int(row.get("oracle_trials") or 0) for row in null_rows),
        "input_file_hashes": input_hashes,
    }


def paired_differences(run_a: dict[str, Any], run_b: dict[str, Any], *,
                       budget: int | None = None) -> list[dict[str, Any]]:
    """Per-seed paired differences in max truth loading for arms sharing a budget.

    Paired because both runs use the same seeds and therefore the same panels; the seed, not
    the trial, is the independent unit.
    """
    out = []
    for arm_a in run_a["arms"]:
        for arm_b in run_b["arms"]:
            if arm_a["budget"] != arm_b["budget"]:
                continue
            if budget is not None and arm_a["budget"] != budget:
                continue
            shared = sorted(set(arm_a["per_seed"]) & set(arm_b["per_seed"]))
            pairs = []
            for seed in shared:
                left = arm_a["per_seed"][seed]["max_truth_loading"]
                right = arm_b["per_seed"][seed]["max_truth_loading"]
                if left is None or right is None:
                    continue
                pairs.append({"seed": seed, "a": left, "b": right,
                              "difference": round(left - right, 6)})
            if not pairs:
                continue
            differences = [p["difference"] for p in pairs]
            positive = sum(1 for d in differences if d > 0)
            negative = sum(1 for d in differences if d < 0)
            out.append({
                "budget": arm_a["budget"],
                "arm_a": arm_a["arm_id"], "arm_b": arm_b["arm_id"],
                "n_paired_seeds": len(pairs),
                "median_difference": round(float(np.median(differences)), 6),
                "n_a_better": positive, "n_b_better": negative,
                "pairs": pairs,
                "test_note": ("Sign counts and the median paired difference only. A formal "
                              "verdict requires the test preregistered for the protocol "
                              "(exact sign-flip permutation or exact Wilcoxon for this "
                              "continuous endpoint); it is deliberately not computed here so "
                              "that a descriptive view cannot be mistaken for a decision."),
            })
    return out


def main() -> None:
    """Read completed run artifacts and save descriptive continuous/paired metrics.

    Record input hashes with the report so its measurements can be associated
    with the exact artifacts rather than an earlier run in the same directory.
    """
    parser = argparse.ArgumentParser(
        description="Derive continuous-metric views from completed protocol runs")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--compare-run-dir", default="")
    parser.add_argument("--out-name", default="")
    parser.add_argument("--budget", type=int, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    primary = summarise_run(run_dir)
    payload: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "script": "benchmark/derived_metrics.py",
        "script_sha256": _file_sha256(Path(__file__))[:16],
        "status": ("DESCRIPTIVE for protocols preregistered on the binary discovery rule "
                   "(F3, F4). The continuous endpoint may serve as a primary endpoint only "
                   "where a protocol preregisters it prospectively."),
        "endpoints": {
            "generation": "per-seed max truth loading and prefix-best curve",
            "validation": "discovery / true accepts / false accepts / null accepts",
            "note": ("Both are reported. The continuous endpoint does not replace the gate "
                     "endpoint: a high-loading candidate that fails the gates is not a "
                     "success for a validator-first system."),
        },
        "primary": primary,
    }
    if args.compare_run_dir:
        comparison = summarise_run(Path(args.compare_run_dir))
        payload["comparison"] = comparison
        payload["paired_differences"] = paired_differences(primary, comparison,
                                                           budget=args.budget)
    out_dir = run_dir / "derived"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.out_name or "continuous_metrics"
    out_path = out_dir / f"{name}.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    rows = []
    for arm in primary["arms"]:
        rows.append(f"{arm['arm_id']:26s} B={arm['budget']:2d} "
                    f"median_max_loading={arm['median_max_loading']} "
                    f"range=[{arm['min_max_loading']}, {arm['max_max_loading']}] "
                    f"discoveries={arm['discoveries']}/{arm['n_seeds']} "
                    f"true/false accepts={arm['true_accepts']}/{arm['false_accepts']}")
    print("\n".join(rows))
    print(f"null pooled: {primary['null_accepts_pooled']}/{primary['null_trials_pooled']}")
    for block in payload.get("paired_differences", []):
        print(f"paired B={block['budget']} {block['arm_a']} vs {block['arm_b']}: "
              f"median diff {block['median_difference']} "
              f"({block['n_a_better']} better / {block['n_b_better']} worse "
              f"of {block['n_paired_seeds']})")
    print(f"written: {out_path}")


if __name__ == "__main__":
    main()
