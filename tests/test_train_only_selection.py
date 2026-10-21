"""The selection chain uses only labels settled before the training cutoff.

Tampering excluded future data must leave screening and sparse weights identical.
Positive controls establish that the tampered data could affect an unpurged fit."""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from factor_research.factor_validity_pipeline import load_config, run_pipeline  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FACTORS = ["mom6m", "ep", "illiq", "std_turn"]
TRAIN_END = "2020-12-31"


def _write_panel(path: Path, seed: int = 0, tamper_after_train: bool = False) -> None:
    """Write a seeded panel whose future rows can be perturbed to test cutoff isolation."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2019-01-31", periods=36, freq="ME")
    rows = []
    for date in dates:
        for i in range(25):
            rows.append({"date": date.strftime("%Y-%m-%d"), "asset": f"S{i:03d}",
                         **{f: rng.normal() for f in FACTORS},
                         "forward_return": 0.03 * rng.standard_t(df=6),
                         # synthetic noise labels settle on the signal date
                         "forward_return_end_date": date.strftime("%Y-%m-%d")})
    frame = pd.DataFrame(rows)
    if tamper_after_train:
        late = pd.to_datetime(frame["date"]) > pd.to_datetime(TRAIN_END)
        frame.loc[late, "forward_return"] = frame.loc[late, "forward_return"] * 7 + 0.05
        frame.loc[late, FACTORS] = frame.loc[late, FACTORS] * 3 - 1
    frame.to_csv(path, index=False)


def _config_for(tmp_path: Path, panel: Path, out: Path) -> Path:
    raw = json.loads((ROOT / "a_share_big_config.json").read_text(encoding="utf-8"))
    raw["input_csv"] = str(panel)
    raw["output_dir"] = str(out)
    raw["factor_cols"] = FACTORS
    raw["family_constraints"]["factor_families"] = {
        "mom6m": "momentum", "ep": "value", "illiq": "liquidity", "std_turn": "liquidity"}
    raw["family_constraints"]["max_per_family"] = {"momentum": 2, "value": 2, "liquidity": 2}
    raw.setdefault("synthetic", {})["n_scenarios"] = 8
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    return config_path


def _run(tmp_path: Path, tag: str, tamper: bool) -> Path:
    panel = tmp_path / f"panel_{tag}.csv"
    out = tmp_path / f"out_{tag}"
    _write_panel(panel, tamper_after_train=tamper)
    config = dataclasses.replace(load_config(_config_for(tmp_path, panel, out)),
                                 train_end=TRAIN_END)
    run_pipeline(config)
    return out


def test_selection_ignores_post_train_data(tmp_path):
    """Tamper everything after train_end; the screen p-values, sparse weights and final
    selection must be bit-identical. Excluded data cannot influence any fitted selection output."""
    base = _run(tmp_path, "base", tamper=False)
    tampered = _run(tmp_path, "tampered", tamper=True)
    for name in ("single_factor_screen", "sparse_weights", "final_factor_selection"):
        left = pd.read_csv(base / f"{name}.csv")
        right = pd.read_csv(tampered / f"{name}.csv")
        pd.testing.assert_frame_equal(left, right)


def test_selection_manifest_records_provenance(tmp_path):
    out = _run(tmp_path, "manifest", tamper=False)
    manifest = json.loads((out / "selection_manifest.json").read_text(encoding="utf-8"))
    assert manifest["train_end"] == TRAIN_END
    assert manifest["panel_sha256"]
    assert manifest["config_sha256"]
    assert manifest["n_rows_fit"] < manifest["n_rows_total"]


def test_panel_without_label_end_fails_closed(tmp_path):
    """A panel with no forward_return_end_date and no embargo must be refused: date <=
    cutoff alone keeps rows whose labels are built from post-cutoff prices."""
    import dataclasses as _dc
    panel = tmp_path / "panel.csv"
    _write_panel(panel)
    frame = pd.read_csv(panel).drop(columns=["forward_return_end_date"])
    frame.to_csv(panel, index=False)
    config = _dc.replace(load_config(_config_for(tmp_path, panel, tmp_path / "out")),
                         train_end=TRAIN_END)
    with pytest.raises(ValueError, match="no label embargo"):
        run_pipeline(config)
    # an explicit embargo unlocks it
    config = _dc.replace(config, label_embargo_days=31)
    run_pipeline(config)


def test_purge_drops_rows_whose_labels_settle_late(tmp_path):
    from factor_research.factor_validity_pipeline import purge_label_overlap
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2021-11-30", "2021-12-31", "2021-12-31", "2022-01-31"]),
        "forward_return_end_date": pd.to_datetime(
            ["2021-12-21", "2021-12-31", "2022-01-21", "2022-02-21"]),
        "x": [1, 2, 3, 4]})
    kept, info = purge_label_overlap(frame, "2021-12-31")
    # the 2021-12-31 row whose label settles 2022-01-21 is the leak; the same-dated row
    # settling on the cutoff stays
    assert kept["x"].tolist() == [1, 2]
    assert info["n_purged_label_overlap"] == 1
    assert info["max_train_label_end"] == "2021-12-31"


def test_missing_train_end_is_refused(tmp_path):
    panel = tmp_path / "panel.csv"
    _write_panel(panel)
    config = load_config(_config_for(tmp_path, panel, tmp_path / "out"))
    assert config.train_end is None
    with pytest.raises(ValueError, match="train_end is required"):
        run_pipeline(config)


def test_purge_rejects_a_non_positive_or_fractional_embargo():
    """A negative embargo widened the window past the cutoff and reported a NEGATIVE
    purge count -- it re-admitted the very rows it exists to remove."""
    from factor_research.factor_validity_pipeline import purge_label_overlap
    frame = pd.DataFrame({"date": pd.to_datetime(["2020-12-31", "2021-01-31"]),
                          "x": [1, 2]})
    for bad in (-30, 0, 1.5, True, "30"):
        with pytest.raises(ValueError, match="positive integer"):
            purge_label_overlap(frame, "2020-12-31", embargo_days=bad)
    kept, info = purge_label_overlap(frame, "2020-12-31", embargo_days=1)
    assert len(kept) == 0                     # 2020-12-31 is inside the embargo edge
    assert info["n_purged_label_overlap"] == 1


def test_purge_never_returns_rows_after_the_cutoff():
    from factor_research.factor_validity_pipeline import purge_label_overlap
    frame = pd.DataFrame({"date": pd.to_datetime(["2020-11-30", "2021-06-30"]),
                          "x": [1, 2]})
    kept, _ = purge_label_overlap(frame, "2020-12-31", embargo_days=1)
    assert kept["x"].tolist() == [1]


def test_purge_rejects_labels_settling_before_their_signal():
    from factor_research.factor_validity_pipeline import purge_label_overlap
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2020-06-30"]),
        "forward_return_end_date": pd.to_datetime(["2020-05-31"]),
        "x": [1]})
    with pytest.raises(ValueError, match="settle before"):
        purge_label_overlap(frame, "2020-12-31")

# ------------------------------------------------------------------ label-settlement boundaries
def test_cohort_split_assigns_by_settlement_not_signal():
    """Assign cohorts by when their labels settle, excluding test-price information from parameter selection."""
    from factor_research.factor_validity_pipeline import cohort_split
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2019-06-30", "2022-12-31", "2022-12-31", "2023-03-31"]),
        "forward_return_end_date": pd.to_datetime(
            ["2019-07-31", "2022-12-31", "2023-01-31", "2023-04-30"])})
    got = cohort_split(frame, "2019-12-31", "2022-12-31").tolist()
    assert got == ["train", "validation", "test", "test"]


def test_cohort_split_refuses_a_panel_without_settlement_dates():
    from factor_research.factor_validity_pipeline import cohort_split
    frame = pd.DataFrame({"date": pd.to_datetime(["2020-01-31"])})
    with pytest.raises(ValueError, match="forward_return_end_date"):
        cohort_split(frame, "2019-12-31", "2022-12-31")


def test_settled_history_returns_a_full_window():
    """Taking the last N signal months and dropping unsettled ones silently shrank the
    estimator: a 24-month request used 22 months in 108 of 110 evaluable months."""
    from factor_research.factor_validity_pipeline import settled_history
    months = pd.period_range("2020-01", periods=40, freq="M")
    # every month settles two months after its signal, so 33 have settled by formation
    settle_max = {m: m.to_timestamp(how="end") + pd.DateOffset(months=2) for m in months}
    got = settled_history(settle_max, pd.Timestamp("2023-01-01"), 24)
    assert len(got) == 24
    assert all(settle_max[m] < pd.Timestamp("2023-01-01") for m in got)
    assert got == sorted(got)


def test_settled_history_returns_short_when_history_is_short():
    from factor_research.factor_validity_pipeline import settled_history
    months = pd.period_range("2020-01", periods=5, freq="M")
    settle_max = {m: m.to_timestamp(how="end") for m in months}
    assert len(settled_history(settle_max, pd.Timestamp("2021-01-01"), 24)) == 5



def test_incremental_gate_uses_hac_not_iid(monkeypatch, tmp_path):
    """Verify HAC statistics and the incremental gate's configured estimator wiring.

    An autocorrelated coefficient series distinguishes HAC from IID errors. A recorder
    then checks that the gate passes the configured hac_lags to t_stat."""
    import numpy as np
    import pandas as pd
    import factor_research.factor_validity_pipeline as fvp

    # (a) semantics
    rng = np.random.default_rng(0)
    autocorr = pd.Series(np.convolve(rng.normal(size=240), np.ones(4) / 4, mode="same")
                         + 0.05)
    t_iid = fvp.t_stat(autocorr, 0)
    t_hac = fvp.t_stat(autocorr, 3)
    assert abs(t_hac) < 0.75 * abs(t_iid), (t_iid, t_hac)

    # (b) wiring
    calls = []
    real_t_stat = fvp.t_stat

    def recording_t_stat(values, hac_lags=0):
        calls.append(hac_lags)
        return real_t_stat(values, hac_lags)

    monkeypatch.setattr(fvp, "t_stat", recording_t_stat)
    rng = np.random.default_rng(1)
    dates = pd.date_range("2019-01-31", periods=30, freq="ME")
    rows = []
    for date in dates:
        for i in range(20):
            f1, f2 = rng.normal(), rng.normal()
            rows.append({"date": date, "asset": f"A{i}", "f1": f1, "f2": f2,
                         "forward_return": 0.03 * f1 + 0.02 * rng.normal()})
    frame = pd.DataFrame(rows)
    import dataclasses as _dc2
    config = _dc2.replace(
        load_config(ROOT / "a_share_big_config.json"),
        factor_cols=("f1", "f2"), factor_families={"f1": "a", "f2": "b"})
    fvp.incremental_value_filter(frame, ["f1", "f2"], config)
    assert calls, "incremental_value_filter no longer routes through t_stat"
    assert set(calls) == {config.hac_lags}, (
        f"t_stat called with lags {set(calls)}, expected {{{config.hac_lags}}}")

def test_label_helpers_honor_non_default_column_names():
    """cohort_split advertised a date_col parameter but passed the default through to
    label_end_series, which then failed with "no \'date\' column" on any panel using a
    different name."""
    from factor_research.factor_validity_pipeline import cohort_split, label_end_series
    frame = pd.DataFrame({
        "signal_date": pd.to_datetime(["2019-06-30", "2023-03-31"]),
        "target": [0.01, 0.02],
        "forward_return_end_date": pd.to_datetime(["2019-07-31", "2023-04-30"])})
    got = cohort_split(frame, "2019-12-31", "2022-12-31",
                       date_col="signal_date", context="col-test").tolist()
    assert got == ["train", "test"]
    end = label_end_series(frame, context="col-test", date_col="signal_date",
                           target_col="target")
    assert end.notna().all()
