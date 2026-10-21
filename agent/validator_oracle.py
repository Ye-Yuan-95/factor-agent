"""Validator oracle: validate(candidate) -> FactorCard.

Packages the pipeline's five gates + DSR behind one generator-agnostic interface:
FDR screen (BH q <= q_max), redundancy (|corr| vs existing + family cap), synthetic
survival, incremental multivariate FM t, sparse L1/L2 weight, plus deflated Sharpe
(deflation benchmark from Memory's trial history).

Iron rules implemented here:
  * LOCKBOX: the panel is truncated to date <= val_end AT LOAD TIME -- the oracle never
    holds test rows, so no gate, delta, or reject_reason can leak test information.
    Only tools.open_test_lockbox (human checkpoint, separate code path) opens the lockbox.
  * Gates run on TRAIN only; val is used only for the investment-value fields
    (val_ic / val_sharpe_delta / turnover_delta).
  * gate_fpr / gate_power are PROCESS-level certificate fields read from the JSON produced
    by agent/gate_certificate.py -- never computed per factor (category-error guard).
  * decision: any hard gate fails -> REJECT (+reason); statistically true but
    val_sharpe_delta <= 0 -> SHELVED (periodic MIQP joint re-selection pool); else ACCEPT.

FDR scope: the BH q-value covers the {existing + candidate} cross-section (horizontal
multiplicity); the agent's sequential-search multiplicity is DSR's axis.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass, asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from factor_research.factor_validity_pipeline import (  # noqa: E402
    PipelineConfig, load_config, single_factor_screen, factor_correlation,
    synthetic_survival, incremental_value_filter, fit_sparse_linear_weights,
    long_short_by_date, rank_ic_by_date, infer_factor_family,
)
from agent.factor_dsl import compile_expression, DSLError  # noqa: E402
from agent.deflated_sharpe import deflated_sharpe  # noqa: E402
from agent.certification import compute_binding, content_hash  # noqa: E402

TRAIN_END = pd.Timestamp("2019-12-31")
VAL_END = pd.Timestamp("2022-12-31")
DEFAULT_CERTIFICATE = _PROJECT_DIR / "outputs" / "agent" / "gate_certificate.json"


@dataclass
class FactorCard:
    # identity
    name: str
    expression: str
    rationale: str
    family: str
    universe: str
    # falsification evidence -- the five hard gates (+ DSR)
    fdr_q: float | None = None
    fdr_pass: bool = False
    redundant_with: str | None = None
    max_abs_corr_with_existing: float | None = None
    family_limited: bool = False
    redundancy_pass: bool = False
    survival_rate: float | None = None
    degradation: float | None = None
    survival_pass: bool = False
    incremental_t: float | None = None
    incremental_pass: bool = False
    incremental_controls: list | None = None   # base-library neighbours controlled in the FM gate
    incremental_n_periods: int | None = None
    incremental_n_complete_rows: int | None = None
    incremental_sample_status: str | None = None
    control_span_r2: float | None = None       # R2 of candidate on controls; ~1 = unidentifiable
    selected_by_weight: bool = False
    sparse_weight: float | None = None
    deflated_sharpe: float | None = None
    dsr_pass: bool = False
    dsr_benchmark: float | None = None
    n_trials_used: int = 0
    n_eff_used: int = 0
    # process-level certificate (shared by every card under one gate config; NOT per-factor)
    gate_fpr: float | None = None
    gate_power: float | None = None
    gate_power_alpha: str | None = None
    # investment value (train/val ONLY; test is a lockbox)
    train_mean_ic: float | None = None
    trial_sharpe: float | None = None          # directional monthly LS Sharpe on train+val
    val_ic: float | None = None
    val_sharpe_delta: float | None = None      # annualized composite LS Sharpe delta on val
    turnover_delta: float | None = None
    test_sharpe_delta: float | None = None     # filled ONLY by tools.open_test_lockbox
    # verdict
    decision: str = "REJECT"                   # ACCEPT / SHELVED / DRAFT / REJECT
    certified: bool = False                    # true only under a BOUND, accepted certificate
    reject_reason: str | None = None
    confidence: str = "low"
    # reproducibility
    complexity: dict | None = None
    data_version: str | None = None
    seed: int | None = None
    code_commit: str | None = None
    timestamp: str | None = None

    def to_dict(self) -> dict:
        return _jsonable(asdict(self))


class ValidatorOracle:
    def __init__(self, config_path: str | Path, panel_csv: str | Path | None = None,
                 train_end: pd.Timestamp = TRAIN_END, val_end: pd.Timestamp = VAL_END,
                 dsr_min: float = 0.95, certificate_path: str | Path | None = None,
                 long_short_quantile: float | None = None,
                 incremental_control_k: int = 5, control_corr_ceiling: float = 0.95,
                 control_corr_floor: float = 0.15, control_span_r2_max: float = 0.9):
        """Initialize gate thresholds, configuration, certificate binding and persistent trial memory.
        Panel loading is a separate operation; validation must not assume a certificate matches an unseen panel."""
        self.base_cfg = load_config(config_path)
        self.config_path = str(config_path)
        self.panel_csv = _resolve_path(panel_csv if panel_csv is not None else self.base_cfg.input_csv)
        self.train_end = pd.Timestamp(train_end)
        self.val_end = pd.Timestamp(val_end)
        self.dsr_min = float(dsr_min)
        self.certificate_path = Path(certificate_path) if certificate_path else DEFAULT_CERTIFICATE
        self.ls_quantile = long_short_quantile or self.base_cfg.long_short_quantile
        # the incremental FM gate must control (1) the candidate's
        # nearest neighbours in the BASE library, not just `existing` -- otherwise derived
        # expressions borrow truth-correlation untested (rFDR 0.55-0.69); and (2) the
        # expression's own AST ingredient columns as MANDATORY controls -- adaptive mutation
        # can decorrelate from the library but cannot hide its ingredient list.
        self.incremental_control_k = int(incremental_control_k)      # 0 disables
        self.control_corr_ceiling = float(control_corr_ceiling)      # >= ceiling = "is" the column
        self.control_corr_floor = float(control_corr_floor)          # < floor = not a neighbour
        # a candidate (nearly) inside its controls' linear span has no identifiable incremental
        # content; collinear lstsq splits the premium deterministically each month -> stable
        # SPURIOUS FM t. Hard-fail instead.
        self.control_span_r2_max = float(control_span_r2_max)
        self._panel: pd.DataFrame | None = None
        self._panel_source: str | None = None
        # registry of validated DSL factors (name -> expression), so `existing` can reference
        # accepted candidates that are not physical panel columns
        self.derived: dict[str, str] = {}
        self._candidate_families: dict[str, str] = {}
        # panels loaded from disk must prove their label horizon; set_panel overrides this
        self._label_semantics: str = "from_file"
        self._derived_cache: dict[str, pd.Series] = {}

    # ---- panel handling (lockbox enforced at load) ----
    def _lockbox_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """LOCKBOX: physically drop test rows (date > val_end) before anything can touch them."""
        frame = frame.copy()
        frame[self.base_cfg.date_col] = pd.to_datetime(frame[self.base_cfg.date_col])
        frame = frame.loc[frame[self.base_cfg.date_col] <= self.val_end]
        frame = self._lockbox_settle(frame)
        return frame.sort_values(
            [self.base_cfg.date_col, self.base_cfg.asset_col]).reset_index(drop=True)


    def _lockbox_settle(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Exclude labels settling after val_end from all discovery gates.

        Pre-boundary signal dates cannot make post-boundary return prices observable.
        Explicit direct-label synthetic panels use their declared exemption."""
        end = self._label_end_series(frame)
        if end is None:
            return frame
        return frame.loc[end.notna() & (end <= self.val_end)]

    def load_panel(self) -> pd.DataFrame:
        key = str(self.panel_csv)
        if self._panel is None or self._panel_source != key:
            self._label_semantics = "from_file"   # a file panel must prove its horizon
            self._panel = self._lockbox_frame(pd.read_csv(self.panel_csv))
            self._panel_source = key
            self.derived.clear()
            self._derived_cache.clear()
            self._candidate_families.clear()
        return self._panel

    def set_panel(self, frame: pd.DataFrame, source: str, label_semantics: str) -> None:
        """Inject an in-memory panel (synthetic trials). The same lockbox truncation is applied."""
        self._label_semantics = label_semantics   # set BEFORE the lockbox reads it
        self._panel = self._lockbox_frame(frame)
        self._panel_source = source
        self.panel_csv = Path(source)
        # Derived expressions and their families are panel-scoped: an expression validated
        # on the previous panel must not survive as a control or a family count here.
        self.derived.clear()
        self._derived_cache.clear()
        self._candidate_families.clear()

    def _label_end_series(self, panel: pd.DataFrame) -> pd.Series | None:
        """Settlement dates for this panel, or None when there is no horizon to purge.

        Delegates to the shared rule so the oracle, the manual lockbox and the
        walk-forward cannot answer "is this row available" differently."""
        from factor_research.factor_validity_pipeline import label_end_series
        return label_end_series(
            panel, context=f"panel {self.panel_csv}",
            date_col=self.base_cfg.date_col, target_col=self.base_cfg.target_col,
            allow_missing=(self._label_semantics == "direct"))

    def _existing_column(self, panel: pd.DataFrame, factor: str) -> pd.Series:
        """An `existing` factor: physical panel column, or registered DSL factor materialized on demand."""
        if factor in panel.columns:
            return pd.to_numeric(panel[factor], errors="coerce")
        if factor in self._derived_cache:
            return self._derived_cache[factor]
        if factor in self.derived:
            values = compile_expression(self.derived[factor]).evaluate(
                panel, date_col=self.base_cfg.date_col, asset_col=self.base_cfg.asset_col)
            self._derived_cache[factor] = values
            return values
        raise ValueError(f"existing factor '{factor}' is neither a panel column nor a "
                         f"registered validated expression")

    def _select_base_controls(self, panel: pd.DataFrame, values: pd.Series,
                              existing: list[str], name: str,
                              ingredient_cols: tuple[str, ...] = ()) -> list[str]:
        """Incremental-gate controls = MANDATORY ingredient columns (base columns the expression
        references) + top-k base-library columns with train |Spearman| in [floor, ceiling).
        Columns above the ceiling are (near-)monotone recoveries of the candidate itself --
        controlling for them would be self-regression and auto-reject clean discoveries.
        Columns in `existing` are already controls. k=0 disables the mechanism."""
        if self.incremental_control_k <= 0:
            return []
        mask = panel[self.base_cfg.date_col] <= self.train_end
        cand_rank = values[mask].rank()

        def spearman(col: str) -> float:
            x = pd.to_numeric(panel.loc[mask, col], errors="coerce")
            rho = cand_rank.corr(x.rank())            # pooled Spearman, as in the redundancy gate
            return abs(float(rho)) if pd.notna(rho) else float("nan")

        controls: list[str] = []
        for col in ingredient_cols:                   # mandatory unless self-like or unavailable
            if (col in self.base_cfg.factor_cols and col not in existing
                    and col != name and col in panel.columns):
                rho = spearman(col)
                if np.isfinite(rho) and rho < self.control_corr_ceiling:
                    controls.append(col)
        scored: list[tuple[float, str]] = []
        for col in self.base_cfg.factor_cols:
            if col == name or col in existing or col in controls or col not in panel.columns:
                continue
            rho = spearman(col)
            if np.isfinite(rho) and self.control_corr_floor <= rho < self.control_corr_ceiling:
                scored.append((rho, col))
        scored.sort(reverse=True)
        return controls + [col for _, col in scored[: self.incremental_control_k]]

    # ---- the oracle ----
    def validate(self, *, name: str, expression: str, rationale: str = "", family: str = "unclassified",
                 existing: list[str] | None = None, memory=None,
                 n_trials: int | None = None, trial_sharpes: list[float] | None = None,
                 seed: int = 42) -> FactorCard:
        """Run five gates and DSR using existing factors as the accepted control set.

        Purge labels unsettled at train_end before statistical fits. Optional
        memory supplies search history and records the outcome; explicit trial
        arguments allow validation without writing a ledger. The oracle still
        updates its derived-factor registry and candidate-family cache. Sandbox
        errors propagate and do not become counted statistical trials.
        """
        existing = [f for f in (existing or []) if f != name]
        card = FactorCard(name=name, expression=expression, rationale=rationale,
                          family=family, universe=self.panel_csv.parent.name or str(self.panel_csv),
                          seed=seed, timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                          code_commit=_git_commit(), data_version=_data_version(self.panel_csv))

        compiled = compile_expression(expression)   # DSLError propagates: sandbox failures are
        card.complexity = compiled.complexity()     # NOT trials (l1_runner catches + logs them)

        panel = self.load_panel()
        cfg = self._make_cfg(name, family, existing)
        date_col, asset_col, target = cfg.date_col, cfg.asset_col, cfg.target_col

        values = compiled.evaluate(panel, date_col=date_col, asset_col=asset_col)
        if float(values.notna().mean()) < cfg.min_coverage:
            raise DSLError(f"expression coverage {values.notna().mean():.2%} "
                           f"< min_coverage {cfg.min_coverage:.0%} on this panel")
        self.derived[name] = compiled.source   # register so later candidates can control for it
        self._candidate_families[name] = family

        frame = panel[[date_col, asset_col, target]].copy()
        for f in existing:
            frame[f] = self._existing_column(panel, f)
        frame[name] = values
        controls = self._select_base_controls(panel, values, existing, name,
                                              ingredient_cols=compiled.columns)
        for c in controls:
            frame[c] = pd.to_numeric(panel[c], errors="coerce")
        frame = frame.dropna(subset=[target])
        train_mask = frame[date_col] <= self.train_end
        label_end = self._label_end_series(panel)
        if label_end is not None:
            # A label that settles after train_end is post-cutoff information regardless
            # of the row's own date.
            train_mask &= label_end.reindex(frame.index) <= self.train_end
        train = frame.loc[train_mask]

        # ---- gate 1: FDR over the {existing + candidate} cross-section (train) ----
        screen = single_factor_screen(train, cfg)
        row = screen.loc[screen["factor"] == name].iloc[0]
        card.fdr_q = float(row["q_value"])
        card.train_mean_ic = _f(row["mean_ic"])
        card.fdr_pass = bool(row["passes_stat_screen"])

        # ---- gate 2: redundancy vs existing + family cap (train) ----
        corr = factor_correlation(train, cfg)
        max_corr, partner = 0.0, None
        for other in existing:
            if name in corr.index and other in corr.columns:
                v = corr.loc[name, other]
                if pd.notna(v) and abs(float(v)) > abs(max_corr):
                    max_corr, partner = float(v), other
        is_redundant = partner is not None and abs(max_corr) >= cfg.max_abs_corr
        family_cap = cfg.max_factors_per_family.get(family)
        n_same_family = sum(1 for f in existing if infer_factor_family(f, cfg) == family)
        card.family_limited = family_cap is not None and n_same_family >= family_cap
        card.redundant_with = partner if is_redundant else None
        card.max_abs_corr_with_existing = abs(max_corr) if partner is not None else None
        card.redundancy_pass = not is_redundant and not card.family_limited

        # ---- gate 3: synthetic survival (train) ----
        surv = synthetic_survival(train, [name], cfg)
        if len(surv):
            srow = surv.iloc[0]
            card.survival_rate = _f(srow["survival_rate"])
            card.degradation = _f(srow["degradation"])
            card.survival_pass = bool(
                pd.notna(srow["survival_rate"]) and srow["survival_rate"] >= cfg.min_survival_rate
                and bool(srow["passes_degradation"]))

        # ---- gate 4: incremental value, multivariate Fama-MacBeth (train) ----
        # jointly controls `existing` + base-library ingredients/neighbours; spanning pre-check
        # rejects candidates whose FM coefficient is unidentified (collinear -> spurious stable t)
        card.incremental_controls = controls
        card.control_span_r2 = self._control_span_r2(train, name, existing + controls)
        spanned = (card.control_span_r2 is not None
                   and card.control_span_r2 >= self.control_span_r2_max)
        inc = incremental_value_filter(train, existing + controls + [name], cfg)
        irow = inc.loc[inc["factor"] == name].iloc[0]
        card.incremental_t = _f(irow["fm_tstat"]) if not spanned else None
        card.incremental_n_periods = int(irow.get("fm_n_periods", 0))
        card.incremental_n_complete_rows = int(irow.get("fm_n_complete_rows", 0))
        card.incremental_sample_status = irow.get("fm_status", "unknown")
        card.incremental_pass = (bool(irow["passes_incremental"]) and not spanned
                                 and card.incremental_t is not None
                                 and np.isfinite(card.incremental_t))

        # ---- gate 5: sparse-weight selection (train) ----
        weights = fit_sparse_linear_weights(train, existing + [name], cfg)
        wrow = weights.loc[weights["factor"] == name]
        if len(wrow):
            card.sparse_weight = _f(wrow.iloc[0]["weight"])
            card.selected_by_weight = bool(wrow.iloc[0]["selected_by_weight"])

        # ---- DSR: deflate the trial Sharpe against the Memory's search history ----
        ls = long_short_by_date(frame, name, cfg)                       # train+val months
        monthly = pd.Series(dtype=float)
        if len(ls):
            # degenerate factors (< 3 distinct values per month) yield an empty/short series
            train_mean = ls.loc[ls.index <= self.train_end].mean()
            direction = float(np.sign(train_mean)) if np.isfinite(train_mean) and train_mean != 0 else 1.0
            monthly = (direction * ls).dropna()
        trial_sharpe = monthly.mean() / monthly.std(ddof=1) if len(monthly) > 2 else float("nan")
        card.trial_sharpe = float(trial_sharpe) if np.isfinite(trial_sharpe) else None
        if memory is not None:
            history = memory.trial_sharpes()
            n_eff = memory.n_eff(include_candidate_ls=monthly.tolist(),
                                 include_candidate_dates=[str(pd.Timestamp(d).date())
                                                          for d in monthly.index])
            n_raw = memory.n_trials() + 1
        else:
            history = list(trial_sharpes or [])
            n_eff = max(int(n_trials or 1), 1)
            n_raw = max(int(n_trials or 0), len(history)) + 1
        dsr = deflated_sharpe(card.trial_sharpe if card.trial_sharpe is not None else float("nan"),
                              history, n_eff, len(monthly),
                              skew=float(monthly.skew()), kurt=float(monthly.kurt()) + 3.0)
        # Keep decision inputs at full precision; display rounding must not move
        # a below-threshold statistic onto the acceptance boundary.
        card.deflated_sharpe = (float(dsr["deflated_sharpe"])
                               if np.isfinite(dsr["deflated_sharpe"]) else None)
        card.dsr_benchmark = _f(dsr["sharpe_ref"])
        card.dsr_pass = card.deflated_sharpe is not None and card.deflated_sharpe >= self.dsr_min
        card.n_trials_used = n_raw
        card.n_eff_used = int(n_eff)

        # ---- investment value on val (train/val only; test locked away) ----
        val = frame.loc[frame[date_col] > self.train_end]
        val_ic_series = rank_ic_by_date(val, name, cfg)
        card.val_ic = _f(val_ic_series.mean()) if len(val_ic_series) else None
        delta = self._val_composite_delta(train, val, existing, name, weights, cfg)
        card.val_sharpe_delta = delta["sharpe_delta"]
        card.turnover_delta = delta["turnover_delta"]

        # ---- process-level certificate (must be BOUND to the current gate config) ----
        cert, cert_refs = self._load_certificate()
        self.last_certificate_refs = cert_refs
        if cert:
            card.gate_fpr = _f(cert.get("gate_fpr"))
            card.gate_power = _f(cert.get("gate_power"))
            card.gate_power_alpha = cert.get("gate_power_alpha")
            card.certified = bool(cert.get("accepted"))
            card.confidence = "high" if card.certified else "medium"
        # else: defaults stand (certified=False, confidence="low") -- re-run agent/gate_certificate.py

        # ---- decision (reasons contain train/val info only) ----
        failed: list[str] = []
        if not card.fdr_pass:
            failed.append(f"fdr_q={card.fdr_q:.3f}>{cfg.q_value_max}")
        if card.redundant_with:
            failed.append(f"redundant_with:{card.redundant_with}"
                          f"(|corr|={card.max_abs_corr_with_existing:.2f})")
        if card.family_limited:
            failed.append(f"family_limit:{family}")
        if not card.survival_pass:
            failed.append("failed_synthetic_survival")
        if not card.incremental_pass:
            if spanned:
                failed.append(f"spanned_by_controls(R2={card.control_span_r2:.2f})")
            elif card.incremental_sample_status == "insufficient_cross_sections":
                failed.append(f"incremental_insufficient_data(periods={card.incremental_n_periods},required=5)")
            else:
                t_txt = "nan" if card.incremental_t is None else f"{card.incremental_t:.2f}"
                failed.append(f"incremental_t={t_txt}<{cfg.min_incremental_tstat}")
        if not card.selected_by_weight:
            failed.append("zero_sparse_weight")
        if card.trial_sharpe is None:
            failed.append("no_long_short_series(degenerate_factor)")
        if card.deflated_sharpe is not None and not card.dsr_pass:
            failed.append(f"deflated_sharpe={card.deflated_sharpe:.3f}<{self.dsr_min}")
        if failed:
            card.decision = "REJECT"
            card.reject_reason = ";".join(failed)
        elif card.val_sharpe_delta is None:
            # an unevaluable investment increment must not be silently ACCEPTed
            card.decision = "SHELVED"
            card.reject_reason = "val_sharpe_delta_unavailable(conservative shelve)"
        elif card.val_sharpe_delta <= 0:
            card.decision = "SHELVED"
            card.reject_reason = "true_but_no_val_sharpe_increment(joint MIQP re-check pool)"
        else:
            card.decision = "ACCEPT"
        # without a BOUND + accepted certificate, accept-side decisions are DRAFT only
        if card.decision in ("ACCEPT", "SHELVED") and not card.certified:
            card.reject_reason = (f"uncertified_would_be:{card.decision}"
                                  + (f";{card.reject_reason}" if card.reject_reason else ""))
            card.decision = "DRAFT"

        if memory is not None:
            memory.record_trial(
                name=name, expression=compiled.source, signature=compiled.signature(),
                family=family, universe=card.universe, sharpe=card.trial_sharpe,
                mean_ic=card.train_mean_ic, monthly_ls=monthly.tolist(),
                monthly_ls_dates=[str(pd.Timestamp(d).date()) for d in monthly.index],
                decision=card.decision, reject_reason=card.reject_reason,
                extra={"fdr_q": card.fdr_q, "incremental_t": card.incremental_t,
                       "deflated_sharpe": card.deflated_sharpe})
        return card

    # ---- helpers ----
    def _make_cfg(self, name: str, family: str, existing: list[str]) -> PipelineConfig:
        # Merge the families of every DSL candidate seen so far: without this, an accepted
        # llm_0001 classifies as "unclassified" and the family cap never counts it.
        families = {**self.base_cfg.factor_families, **self._candidate_families}
        families[name] = family
        return replace(self.base_cfg, input_csv=self.panel_csv,
                       factor_cols=tuple(existing + [name]), factor_families=families)

    def _val_composite_delta(self, train: pd.DataFrame, val: pd.DataFrame, existing: list[str],
                             name: str, weights: pd.DataFrame, cfg: PipelineConfig) -> dict:
        """Annualized long-short Sharpe of the sparse-weight composite on val, with vs without the
        candidate. Weights are fit on TRAIN only; val is scored out-of-fit. Also the turnover delta."""
        def nonzero(frame: pd.DataFrame) -> dict:
            return {r["factor"]: float(r["weight"]) for _, r in frame.iterrows()
                    if float(r["weight"]) != 0.0}

        w_with = nonzero(weights) or self._fallback_weights(train, existing + [name], cfg)
        w_without = {}
        if existing:
            w_without = (nonzero(fit_sparse_linear_weights(train, existing, cfg))
                         or self._fallback_weights(train, existing, cfg))
        with_stats = _composite_ls_stats(val, w_with, cfg, self.ls_quantile)
        wo_stats = _composite_ls_stats(val, w_without, cfg, self.ls_quantile)

        def delta(key: str) -> float | None:
            if with_stats[key] is None:
                return None
            base = wo_stats[key] if wo_stats[key] is not None else 0.0
            return float(with_stats[key] - base)

        return {"sharpe_delta": delta("sharpe"), "turnover_delta": delta("turnover")}

    def _control_span_r2(self, train: pd.DataFrame, name: str, controls: list[str]) -> float | None:
        """Pooled R2 of the (per-month z-scored) candidate on its (per-month z-scored) controls,
        mirroring the FM gate's standardization. None when there are no controls."""
        controls = [c for c in controls if c in train.columns]
        if not controls:
            return None
        date_col = self.base_cfg.date_col
        cols = [name, *controls]
        z = train[cols].apply(pd.to_numeric, errors="coerce")
        z = z.groupby(train[date_col]).transform(
            lambda g: (g - g.mean()) / (g.std(ddof=0) + 1e-12))
        z = z.dropna()
        if len(z) < len(controls) + 10:
            return None
        X = np.column_stack([np.ones(len(z)), z[controls].to_numpy(float)])
        y = z[name].to_numpy(float)
        var_y = y.var()
        if var_y <= 0:
            return None
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        return float(1.0 - resid.var() / var_y)

    @staticmethod
    def _fallback_weights(train: pd.DataFrame, factors: list[str], cfg: PipelineConfig) -> dict:
        out = {}
        for f in factors:
            ic = rank_ic_by_date(train, f, cfg)
            if len(ic):
                s = np.sign(ic.mean())
                if s != 0:
                    out[f] = float(s)
        return out

    def oracle_gate_params(self) -> dict:
        """The oracle-side parameters a certificate must be bound to."""
        return {
            "incremental_control_k": self.incremental_control_k,
            "control_corr_ceiling": self.control_corr_ceiling,
            "control_corr_floor": self.control_corr_floor,
            "control_span_r2_max": self.control_span_r2_max,
            "dsr_min": self.dsr_min,
            "long_short_quantile": self.ls_quantile,
            "train_end": str(self.train_end.date()),
            "val_end": str(self.val_end.date()),
        }

    def _load_certificate(self) -> tuple[dict | None, dict]:
        """Load the certificate AND verify its binding against the oracle's CURRENT state.
        Returns (cert_or_None, refs); refs records what was checked (-> certificate_refs.json).
        An unbound certificate is treated as absent -- stale certs must not lend confidence to a
        different gate configuration."""
        refs: dict = {"certificate_path": str(self.certificate_path), "bound": False}
        if not self.certificate_path.exists():
            refs["status"] = "missing"
            return None, refs
        try:
            cert = json.loads(self.certificate_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
            refs["status"] = f"unreadable: {exc}"
            return None, refs
        if not isinstance(cert, dict):
            refs["status"] = ("invalid certificate schema: top level must be a JSON "
                              f"object, got {type(cert).__name__}")
            return None, refs
        cert_binding = cert.get("binding", {})
        if not isinstance(cert_binding, dict):
            refs["status"] = ("invalid certificate schema: binding must be a JSON "
                              f"object, got {type(cert_binding).__name__}")
            return None, refs
        # Applicability: the certificate certifies the gate stack ON ITS
        # CALIBRATION PANEL; the active panel must be that exact content. Synthetic/in-memory
        # panels are never certifiable (their acceptance lives in the realized-FDP harness).
        if not self.panel_csv.exists():
            refs["status"] = "panel-not-certifiable (synthetic/in-memory panel)"
            return None, refs
        try:
            active_panel_hash = content_hash(self.panel_csv)
        except OSError as exc:
            refs["status"] = f"panel-unreadable: {exc}"
            return None, refs
        refs["active_panel_hash"] = active_panel_hash
        if active_panel_hash != cert_binding.get("panel_content_hash"):
            refs["status"] = ("panel-mismatch: certificate was calibrated on "
                              f"{cert_binding.get('panel_content_hash')}, active panel is "
                              f"{active_panel_hash} -- re-issue on this panel")
            return None, refs
        try:
            current = compute_binding(
                pipeline_config_raw=json.loads(Path(self.config_path).read_text(encoding="utf-8")),
                oracle_gate_params=self.oracle_gate_params(),
                panel_content_hash=active_panel_hash,
                acceptance=cert_binding.get("acceptance", {}),
            )
        except (OSError, json.JSONDecodeError) as exc:
            refs["status"] = f"binding-recompute-failed: {exc}"
            return None, refs
        refs["certificate_binding_hash"] = cert_binding.get("binding_hash")
        refs["current_binding_hash"] = current["binding_hash"]
        if cert_binding.get("binding_hash") != current["binding_hash"]:
            refs["status"] = "stale: binding hash mismatch (gates/config/data changed since issue)"
            return None, refs
        refs["bound"] = True
        refs["status"] = "bound"
        refs["accepted"] = bool(cert.get("accepted"))
        return cert, refs


def _composite_ls_stats(frame: pd.DataFrame, weights: dict, cfg: PipelineConfig,
                        quantile: float) -> dict:
    """Monthly long-short returns of the z-scored weighted composite; annualized Sharpe + turnover
    (mean fraction of the long+short book replaced per month)."""
    if not weights or frame.empty:
        return {"sharpe": None, "turnover": None}
    date_col, asset_col, target = cfg.date_col, cfg.asset_col, cfg.target_col
    rets: list[float] = []
    prev_book: set | None = None
    turns: list[float] = []
    for _, g in frame.groupby(date_col, sort=True):
        score = pd.Series(0.0, index=g.index)
        any_factor = pd.Series(False, index=g.index)
        for f, w in weights.items():
            if f not in g.columns:
                continue
            x = pd.to_numeric(g[f], errors="coerce")
            sd = x.std(ddof=0)
            if not sd or not np.isfinite(sd):
                continue
            z = ((x - x.mean()) / sd).fillna(0.0)
            score = score + w * z
            any_factor = any_factor | x.notna()
        g2 = g.assign(_score=score)[any_factor.values]
        g2 = g2.dropna(subset=[target])
        if len(g2) < 10:
            continue
        lo = g2["_score"].quantile(quantile)
        hi = g2["_score"].quantile(1.0 - quantile)
        long_leg = g2.loc[g2["_score"] >= hi]
        short_leg = g2.loc[g2["_score"] <= lo]
        if long_leg.empty or short_leg.empty:
            continue
        rets.append(float(long_leg[target].mean() - short_leg[target].mean()))
        book = set(long_leg[asset_col]) | {f"-{a}" for a in short_leg[asset_col]}
        if prev_book is not None and (book or prev_book):
            turns.append(1.0 - len(book & prev_book) / max(len(book | prev_book), 1))
        prev_book = book
    if len(rets) < 3:
        return {"sharpe": None, "turnover": None}
    arr = np.asarray(rets, dtype=float)
    sd = arr.std(ddof=1)
    sharpe = float(arr.mean() / sd * np.sqrt(12.0)) if sd > 0 else None
    return {"sharpe": sharpe, "turnover": float(np.mean(turns)) if turns else None}


def _f(value) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return round(v, 6) if np.isfinite(v) else None


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _git_commit() -> str | None:
    """Record only this checkout's commit, never an enclosing repository's HEAD."""
    try:
        if not (_PROJECT_DIR / ".git").exists():
            return None
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=_PROJECT_DIR,
                             capture_output=True, text=True, timeout=10)
        return (out.stdout.strip() or None) if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _data_version(path: Path) -> str | None:
    """CONTENT hash of the panel."""
    try:
        return content_hash(path)
    except OSError:
        return None


def _resolve_path(path_like) -> Path:
    """Resolve config-relative paths against the repo root, then the project dir."""
    p = Path(path_like)
    if p.is_absolute() or p.exists():
        return p
    for base in (_PROJECT_DIR.parent, _PROJECT_DIR):
        candidate = base / p
        if candidate.exists():
            return candidate
    return p
