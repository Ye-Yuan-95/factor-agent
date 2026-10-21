"""Synthetic null and alternative markets with FPR, FDP and power diagnostics.

Pipeline:
  1. Calibrate to the real A-share panel: factor correlation Sigma_f, return scale s_r, tail dof.
  2. Generate synthetic panels with the SAME schema as the real panel:
       null        : every factor is fake (return independent of all factors);
       alternative : a few KNOWN-true factors get an injected premium (a = IC * s_r).
  3. Run a selection procedure (FDR screen, or the real pipeline) as a black box -> S_hat.
  4. Score against ground truth: FPR = sel fake / all fake, Power = sel true / all true,
     FDR_emp = sel fake / max(#selected, 1).
  5. Repeat R reps and aggregate (Monte-Carlo estimate over P_synth).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

# allow importing the real pipeline (one dir up) to use its ACTUAL screen + redundancy code
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_REPO_ROOT = Path(__file__).resolve().parents[1]
from synthetic.research_runtime import prepare_run, run_experiment  # noqa: E402


# -- calibration: estimate Sigma_f (factor corr) and s_r from the real panel --
def nearest_psd_corr(corr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Clip eigenvalues to make a symmetric matrix PSD, then renormalize to unit diagonal."""
    corr = 0.5 * (corr + corr.T)
    vals, vecs = np.linalg.eigh(corr)
    vals = np.clip(vals, eps, None)
    psd = vecs @ np.diag(vals) @ vecs.T
    d = np.sqrt(np.clip(np.diag(psd), eps, None))
    psd = psd / np.outer(d, d)
    np.fill_diagonal(psd, 1.0)
    return psd


def calibrate(panel_csv: Path, factors: list[str]) -> dict:
    """Estimate factor correlation, return scale and clipped Student-t degrees of freedom.
    The factor order and input panel determine generated results even when the RNG seed is fixed."""
    df = pd.read_csv(panel_csv)
    factors = [f for f in factors if f in df.columns]
    fmat = df[factors].apply(pd.to_numeric, errors="coerce")
    corr = fmat.corr().to_numpy()
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    sigma_f = nearest_psd_corr(corr)
    r = pd.to_numeric(df["forward_return"], errors="coerce").dropna()
    excess_kurt = float(r.kurt())                       # pandas .kurt() is EXCESS (Fisher)
    # Student-t dof matching real excess kurtosis: excess = 6/(nu-4) -> nu = 4 + 6/excess.
    # Real A-share kurtosis is extreme (limit moves/illiquidity); clip nu to keep variance finite.
    nu = 4.0 + 6.0 / max(excess_kurt, 1e-6)
    nu = float(np.clip(nu, 4.5, 40.0))
    return {
        "factors": factors,
        "sigma_f": sigma_f,
        "s_r": float(r.std()),
        "ret_skew": float(r.skew()),
        "ret_kurt": excess_kurt,
        "t_dof": nu,
        "real_corr": corr,
    }


# -- generator --
def generate_panel(rng, calib, n_assets, n_months, true_idx, strengths_ic,
                   market_frac=0.3, tails="t"):
    """Return (panel DataFrame, truth boolean array). truth[j]=True => factor j is a real factor.

    tails='t'  -> idiosyncratic noise is Student-t (fat tails, calibrated to real kurtosis);
    tails='gaussian' -> normal noise (thin tails)."""
    factors = calib["factors"]
    p = len(factors)
    s_r = calib["s_r"]
    nu = calib["t_dof"]
    chol = np.linalg.cholesky(calib["sigma_f"])

    # split return variance into market + idiosyncratic (premium adds on top, it is small)
    s_market = np.sqrt(market_frac) * s_r
    s_eps = np.sqrt(max(1.0 - market_frac, 0.05)) * s_r
    beta = rng.normal(1.0, 0.3, size=n_assets)

    # injected-true strengths: a_j chosen so realized IC ~= target IC (a = IC * s_r)
    a = np.zeros(p)
    for j, ic in zip(true_idx, strengths_ic):
        a[j] = ic * s_r

    rows = []
    for t in range(n_months):
        z = rng.standard_normal((n_assets, p))
        F = z @ chol.T                                   # MVN(0, Sigma_f) per asset
        F = (F - F.mean(0)) / (F.std(0) + 1e-12)         # standardize cross-sectionally
        m_t = rng.normal(0.0, s_market)
        if tails == "t":                                 # standardized Student-t, scaled to s_eps
            raw = rng.standard_t(nu, size=n_assets)
            eps = raw / np.sqrt(nu / (nu - 2.0)) * s_eps
        else:
            eps = rng.normal(0.0, s_eps, size=n_assets)
        eps = eps - eps.mean()                           # INDEPENDENT of F  -> null
        premium = F @ a                                  # zero if no true factors
        r_next = beta * m_t + premium + eps
        block = pd.DataFrame(F, columns=factors)
        block.insert(0, "forward_return", r_next)
        block.insert(0, "asset", [f"S{i:04d}" for i in range(n_assets)])
        block.insert(0, "date", pd.Timestamp("2000-01-31") + pd.offsets.MonthEnd(t))
        rows.append(block)
    panel = pd.concat(rows, ignore_index=True)
    truth = np.zeros(p, dtype=bool)
    truth[list(true_idx)] = True
    return panel, truth


# -- selection procedures (black box): compact FDR screen / real pipeline --
def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Return BH-adjusted p-values in input order for the supplied marginal tests."""
    m = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    q = ranked * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(m)
    out[order] = q
    return np.clip(out, 0.0, 1.0)


def rank_ic_screen(panel: pd.DataFrame, factors: list[str], q_max: float):
    """Per-factor rank IC by date -> t-stat -> normal p-value -> BH-FDR q -> select if q<=q_max."""
    y = panel.groupby("date")["forward_return"].rank()
    pvals, mean_ics = [], []
    for f in factors:
        fr = panel.groupby("date")[f].rank()
        tmp = pd.DataFrame({"date": panel["date"], "fr": fr, "y": y})
        ics = tmp.groupby("date").apply(
            lambda g: np.corrcoef(g["fr"], g["y"])[0, 1] if g["fr"].std() > 0 else np.nan,
            include_groups=False,
        ).dropna()
        if len(ics) < 3 or ics.std(ddof=1) == 0:
            pvals.append(1.0); mean_ics.append(0.0); continue
        t = ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics)))
        pvals.append(float(2 * (1 - norm.cdf(abs(t)))))
        mean_ics.append(float(ics.mean()))
    q = bh_fdr(np.array(pvals))
    selected = q <= q_max
    return selected, np.array(mean_ics), q


def select_pipeline(panel: pd.DataFrame, calib, pipe_config):
    """Use the REAL pipeline's FDR screen + redundancy filter as the black-box selector."""
    from factor_research.factor_validity_pipeline import single_factor_screen, factor_correlation, redundancy_filter
    screen = single_factor_screen(panel, pipe_config)
    corr = factor_correlation(panel, pipe_config)
    red = redundancy_filter(screen, corr, pipe_config)
    sel = set(red.loc[red["passes_redundancy_screen"], "factor"].astype(str))
    return np.array([f in sel for f in calib["factors"]]), None


def metrics(selected: np.ndarray, truth: np.ndarray) -> dict:
    """Score a selection against injected truth.
    FPR divides by all false factors; Power divides by all true factors. The
    FDR_emp display field is a realized FDP (zero for empty selection), not an expected FDR."""
    fake = ~truth
    sel_fake = int((selected & fake).sum())
    sel_true = int((selected & truth).sum())
    n_sel = int(selected.sum())
    return {
        "n_selected": n_sel,
        "FPR": sel_fake / max(int(fake.sum()), 1),
        "Power": (sel_true / int(truth.sum())) if truth.any() else float("nan"),
        "FDR_emp": sel_fake / max(n_sel, 1),
    }


# -- experiment driver --
def run_reps(calib, n_assets, n_months, q_max, reps, true_idx, strengths_ic, seed0,
             tails="t", selector="fdr", pipe_config=None):
    """Generate seeded panels and return per-replicate metrics plus their mean/standard deviation."""
    accum = []
    for r in range(reps):
        rng = np.random.default_rng(seed0 + r)
        panel, truth = generate_panel(rng, calib, n_assets, n_months, true_idx, strengths_ic, tails=tails)
        if selector == "pipeline":
            selected, mean_ics = select_pipeline(panel, calib, pipe_config)
        else:
            selected, mean_ics, _ = rank_ic_screen(panel, calib["factors"], q_max)
        m = metrics(selected, truth)
        if truth.any() and mean_ics is not None:
            m["true_mean_ic"] = float(np.mean(np.abs(mean_ics[truth])))
        accum.append(m)
    df = pd.DataFrame(accum)
    agg = {c: (float(df[c].mean()), float(df[c].std())) for c in df.columns}
    return agg, df


def calibration_report(calib, n_assets, n_months, seed=0) -> dict:
    """Compare generated null-panel moments with the supplied calibration, using one draw per tail family."""
    out = {}
    for tails in ("gaussian", "t"):
        rng = np.random.default_rng(seed)
        panel, _ = generate_panel(rng, calib, n_assets, n_months, [], [], tails=tails)
        r = panel["forward_return"]
        out[f"synth_ret_kurt_{tails}"] = float(r.kurt())
        if tails == "t":
            synth_corr = panel[calib["factors"]].corr().to_numpy()
            out["corr_frobenius_dist"] = float(np.linalg.norm(synth_corr - calib["real_corr"]))
            out["synth_ret_std"] = float(r.std())
    out["real_ret_std"] = calib["s_r"]
    out["real_ret_kurt"] = calib["ret_kurt"]
    out["t_dof"] = calib["t_dof"]
    return out


def main() -> None:
    """Compare BH-only and screening/redundancy selection on known-truth panels.

    Calibrate exposure correlation and returns, repeat null/alternative markets,
    and write FPR, recovery power and per-run FDP. These procedures do not run
    the full oracle discovery loop.
    """
    ap = argparse.ArgumentParser(description="Synthetic null/alternative FPR and power diagnostics.")
    ap.add_argument("--panel-csv", default=str(_REPO_ROOT / "data" / "a_share_factor_panel.csv"))
    ap.add_argument("--config", default=str(_REPO_ROOT / "a_share_baseline_config.json"))
    ap.add_argument("--output-dir", default=str(_REPO_ROOT / "outputs" / "synthetic" / "stage_a"))
    ap.add_argument("--n-assets", type=int, default=120)
    ap.add_argument("--n-months", type=int, default=120)
    ap.add_argument("--q-max", type=float, default=0.25)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--pipeline-reps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    if args.pipeline_reps <= 0:
        ap.error("--pipeline-reps must be positive")
    prepare_run(args, __file__)

    def result_row(experiment, true_mode, ic, agg, has_power):
        return {"experiment": experiment, "true_mode": true_mode, "strength_IC": ic,
                "FPR": agg["FPR"][0], "FPR_sd": agg["FPR"][1],
                "Power": agg["Power"][0] if has_power else np.nan,
                "FDR_emp": agg["FDR_emp"][0], "n_selected": agg["n_selected"][0]}

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    factors = cfg["factor_cols"]
    fam_map = cfg["family_constraints"]["factor_families"]
    calib = calibrate(Path(args.panel_csv), factors)
    flist = calib["factors"]
    p = len(flist)
    print(f"calibrated: {p} factors, s_r={calib['s_r']:.4f}, real ret_kurt={calib['ret_kurt']:.2f}, "
          f"t_dof={calib['t_dof']:.2f}")

    cal = calibration_report(calib, args.n_assets, args.n_months)
    print(f"calibration: corr Frobenius={cal['corr_frobenius_dist']:.3f}; "
          f"ret std real={cal['real_ret_std']:.4f} vs synth={cal['synth_ret_std']:.4f}; "
          f"ret kurt real={cal['real_ret_kurt']:.1f} vs synth gaussian={cal['synth_ret_kurt_gaussian']:.1f} "
          f"vs synth-t={cal['synth_ret_kurt_t']:.1f}")

    # true-factor index sets: same family (4 momentum) vs cross family (one per distinct family)
    same_idx = [j for j in range(p) if fam_map.get(flist[j]) == "momentum"][:4]
    cross_idx, seen = [], set()
    for j in range(p):
        fam = fam_map.get(flist[j])
        if fam not in seen:
            seen.add(fam); cross_idx.append(j)
        if len(cross_idx) >= 4:
            break
    modes = {"same_family": same_idx, "cross_family": cross_idx}
    print(f"true factors  same_family={[flist[j] for j in same_idx]}")
    print(f"true factors cross_family={[flist[j] for j in cross_idx]}")

    rows = []
    # NULL (Student-t tails)
    agg_null, _ = run_reps(calib, args.n_assets, args.n_months, args.q_max, args.reps,
                           true_idx=[], strengths_ic=[], seed0=args.seed, tails="t")
    rows.append(result_row("null", "-", 0.0, agg_null, has_power=False))
    print(f"\n[null]  FPR={agg_null['FPR'][0]:.3f}(sd {agg_null['FPR'][1]:.3f})  "
          f"P(any FP)={agg_null['FDR_emp'][0]:.3f}  avg#sel={agg_null['n_selected'][0]:.2f}")

    # ALTERNATIVE: same_family vs cross_family, each at weak/med/strong
    for mode, idx in modes.items():
        print(f"\n-- alternative [{mode}] true={[flist[j] for j in idx]} --")
        for label, ic in [("weak", 0.03), ("med", 0.06), ("strong", 0.10)]:
            agg, _ = run_reps(calib, args.n_assets, args.n_months, args.q_max, args.reps,
                              true_idx=idx, strengths_ic=[ic] * len(idx), seed0=args.seed, tails="t")
            rows.append(result_row(f"alt_{label}", mode, ic, agg, has_power=True))
            print(f"  {label:>6} IC={ic:.2f}  Power={agg['Power'][0]:.3f}  FPR={agg['FPR'][0]:.3f}  "
                  f"realized|IC|={agg.get('true_mean_ic',(float('nan'),))[0]:.3f}")

    # Compare BH-only selection with the shared pipeline screening/redundancy selector.
    from factor_research.factor_validity_pipeline import load_config
    pipe_config = load_config(args.config)
    print(f"\n=== screening/redundancy selector (actual pipeline code; "
          f"{args.pipeline_reps} reps) ===")
    agg_pn, _ = run_reps(calib, args.n_assets, args.n_months, args.q_max, args.pipeline_reps,
                         true_idx=[], strengths_ic=[], seed0=args.seed, tails="t",
                         selector="pipeline", pipe_config=pipe_config)
    rows.append(result_row("null", "pipeline", 0.0, agg_pn, has_power=False))
    print(f"[null  pipeline] FPR={agg_pn['FPR'][0]:.3f}  P(any FP)={agg_pn['FDR_emp'][0]:.3f}  "
          f"avg#sel={agg_pn['n_selected'][0]:.2f}")
    print(f"-- alternative [pipeline, cross_family true={[flist[j] for j in cross_idx]}] --")
    for label, ic in [("weak", 0.03), ("med", 0.06), ("strong", 0.10)]:
        agg, _ = run_reps(calib, args.n_assets, args.n_months, args.q_max, args.pipeline_reps,
                          true_idx=cross_idx, strengths_ic=[ic] * len(cross_idx), seed0=args.seed,
                          tails="t", selector="pipeline", pipe_config=pipe_config)
        rows.append(result_row(f"alt_{label}", "pipeline_cross", ic, agg, has_power=True))
        print(f"  {label:>6} IC={ic:.2f}  Power={agg['Power'][0]:.3f}  FPR={agg['FPR'][0]:.3f}  "
              f"avg#sel={agg['n_selected'][0]:.2f}")

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "fpr_power_report.csv", index=False, encoding="utf-8-sig")
    (out / "calibration.json").write_text(json.dumps(
        {"same_family_true": [flist[j] for j in same_idx],
         "cross_family_true": [flist[j] for j in cross_idx],
         "q_max": args.q_max, "n_assets": args.n_assets, "n_months": args.n_months,
         "reps": args.reps, "tails": "t", **cal}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote outputs to: {out}")


if __name__ == "__main__":
    run_experiment(main)
