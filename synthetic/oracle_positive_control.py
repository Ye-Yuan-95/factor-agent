"""Isolated oracle diagnostics on mechanism-truth synthetic panels.

The grid uses three mechanism kinds, IC values 0.03/0.06/0.10 and development seeds
2000-2004. Evaluation seeds 3000+ are reserved. Each panel tests the matching
expression, a noise interaction and the bare ingredient. These are diagnostic
controls, not evidence of guaranteed power or independence from correlated
ingredients. Each candidate uses fresh trial memory. Manifests record the actual grid and
input/source identities."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_DIR = Path(__file__).resolve().parents[1]
for p in (str(_PROJECT_DIR), str(_PROJECT_DIR / "synthetic")):
    if p not in sys.path:
        sys.path.insert(0, p)

from agent.memory import MemoryStore  # noqa: E402
from agent.validator_oracle import ValidatorOracle  # noqa: E402
from dgp_mechanism import calibrate, generate_mechanism_panel  # noqa: E402
from synthetic.research_runtime import prepare_run, run_experiment  # noqa: E402

X_COL, Y_COL = "mom1m", "turn"
TRUE_EXPR = {
    "conditional": "zscore(mom1m) * sign(turn)",
    "regime": "zscore(mom1m) * clip(sign(turn), 0, 1)",
    "nonlinear": "zscore(mom1m) * zscore(turn)",
}
NEGATIVE = "zscore(bm) * sign(retvol)"
SEEDS = [2000, 2001, 2002, 2003, 2004]
ICS = [0.03, 0.06, 0.10]


def main() -> None:
    """Evaluate known mechanism expressions separately from generator discovery.

    Use isolated candidate memory for each true, negative and ingredient control;
    record the actual grid and whether it matches the protocol template.
    Run bookkeeping rejects nonempty output directories and sealed seeds.
    """
    import argparse
    import tempfile
    ap = argparse.ArgumentParser(
        description="Isolated oracle controls on mechanism-truth DGPs")
    ap.add_argument("--config", default=str(_PROJECT_DIR / "a_share_big_config.json"))
    ap.add_argument("--panel-csv",
                    default=str(_PROJECT_DIR / "data_big_v2" / "a_share_factor_panel.csv"),
                    help="real panel used only to calibrate the DGP's moments")
    ap.add_argument("--output-dir", default=str(_PROJECT_DIR / "outputs" / "research" / "oracle_positive_control"))
    ap.add_argument("--n-assets", type=int, default=150)
    ap.add_argument("--n-months", type=int, default=150)
    ap.add_argument("--kinds", nargs="+", choices=tuple(TRUE_EXPR), default=list(TRUE_EXPR))
    ap.add_argument("--ics", nargs="+", type=float, default=ICS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = ap.parse_args()
    if any(not np.isfinite(ic) or ic <= 0 for ic in args.ics):
        ap.error("--ics must contain finite positive strengths")
    if any(seed < 0 or seed >= 3000 for seed in args.seeds):
        ap.error("--seeds must be development seeds in [0, 3000); sealed seeds remain reserved")
    if any(len(values) != len(set(values)) for values in (args.kinds, args.ics, args.seeds)):
        ap.error("grid values must be distinct")
    prepare_run(args, __file__)
    config = args.config
    panel_csv = Path(args.panel_csv)
    if not panel_csv.is_file():
        raise SystemExit(f"calibration panel not found: {panel_csv}; pass --panel-csv")
    calib = calibrate(panel_csv, json.loads(Path(config).read_text(encoding="utf-8"))["factor_cols"])
    rows = []
    for kind in args.kinds:
        true_expr = TRUE_EXPR[kind]
        for ic in args.ics:
            for seed in args.seeds:
                panel = generate_mechanism_panel(
                    np.random.default_rng(seed), calib, args.n_assets, args.n_months,
                    kind=kind, x_col=X_COL, y_col=Y_COL, ic=ic)
                oracle = ValidatorOracle(config)
                dates = pd.to_datetime(sorted(pd.unique(panel["date"])))
                oracle.train_end = pd.Timestamp(dates[int(len(dates) * 0.7) - 1])
                oracle.val_end = pd.Timestamp(dates[-1])
                oracle.set_panel(panel, f"poscontrol_{kind}_ic{ic:g}_s{seed}", label_semantics="direct")
                for label, expr in (("true", true_expr), ("negative", NEGATIVE),
                                    ("bare_ingredient", X_COL)):
                    with tempfile.TemporaryDirectory() as memory_dir:
                        mem = MemoryStore(memory_dir)
                        card = oracle.validate(name=f"{label}_{kind}_{ic:g}_{seed}",
                                               expression=expr, family="mechanism",
                                               existing=[], memory=mem, seed=seed)
                    rows.append({
                        "kind": kind, "ic": ic, "seed": seed, "role": label,
                        "expression": expr,
                        "fdr_pass": card.fdr_pass, "survival_pass": card.survival_pass,
                        "incremental_pass": card.incremental_pass,
                        "incremental_t": card.incremental_t,
                        "weight_pass": card.selected_by_weight, "dsr_pass": card.dsr_pass,
                        "would_be_accept": card.reject_reason is not None
                                           and card.reject_reason.startswith("uncertified_would_be:ACCEPT"),
                        "controls": card.incremental_controls,
                        "val_sharpe_delta": card.val_sharpe_delta,
                    })
                print(f"[{kind} ic={ic:g} s{seed}] " + " | ".join(
                    f"{r['role']}: inc_t={r['incremental_t']} wbA={r['would_be_accept']}"
                    for r in rows[-3:]), flush=True)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "results.csv", index=False, encoding="utf-8-sig")
    df = pd.DataFrame(rows)
    summary = df.groupby(["role", "kind", "ic"]).agg(
        wbA_rate=("would_be_accept", "mean"),
        inc_pass_rate=("incremental_pass", "mean"),
        mean_abs_inc_t=("incremental_t", lambda s: float(np.nanmean(np.abs(pd.to_numeric(s))))),
    ).reset_index()
    summary.to_csv(out / "power_curve.csv", index=False, encoding="utf-8-sig")
    (out / "meta.json").write_text(json.dumps({
        "protocol_template": "experiments/oracle_positive_control_2026-07-24.json",
        "grid": {"kinds": args.kinds, "ics": args.ics, "seeds": args.seeds,
                 "n_assets": args.n_assets, "n_months": args.n_months},
        "template_grid_matches": (args.kinds == list(TRUE_EXPR) and args.ics == ICS
                                  and args.seeds == SEEDS and args.n_assets == 150
                                  and args.n_months == 150),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    run_experiment(main)
