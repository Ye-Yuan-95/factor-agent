"""A malformed certificate must be a controlled verdict, not a traceback.

The schema check lives in ValidatorOracle._load_certificate, so acceptance, the L1
harness and the tools CLI are all protected by the same guard.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest  # noqa: E402

from agent.validator_oracle import ValidatorOracle  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def _oracle(certificate: Path) -> ValidatorOracle:
    return ValidatorOracle(ROOT / "a_share_big_config.json", certificate_path=certificate)


def test_list_certificate_is_a_controlled_verdict(tmp_path):
    bad = tmp_path / "cert.json"
    bad.write_text("[1, 2, 3]", encoding="utf-8")
    cert, refs = _oracle(bad)._load_certificate()
    assert cert is None
    assert "invalid certificate schema" in refs["status"]
    assert refs["bound"] is False


def test_non_dict_binding_is_a_controlled_verdict(tmp_path):
    bad = tmp_path / "cert.json"
    bad.write_text('{"binding": [1, 2]}', encoding="utf-8")
    cert, refs = _oracle(bad)._load_certificate()
    assert cert is None
    assert "binding must be a JSON object" in refs["status"]


def test_non_utf8_certificate_is_a_controlled_verdict(tmp_path):
    """Invalid UTF-8 certificates must yield a controlled refusal."""
    bad = tmp_path / "cert.json"
    bad.write_bytes(b"\xff\xfe\x00garbage")
    cert, refs = _oracle(bad)._load_certificate()
    assert cert is None
    assert refs["status"].startswith("unreadable:")


def test_scalar_certificate_is_a_controlled_verdict(tmp_path):
    bad = tmp_path / "cert.json"
    bad.write_text("42", encoding="utf-8")
    cert, refs = _oracle(bad)._load_certificate()
    assert cert is None
    assert "invalid certificate schema" in refs["status"]


def test_gate_fingerprint_is_newline_invariant(tmp_path):
    """A CRLF checkout and an LF checkout of the same source must produce the same
    binding."""
    from agent.certification import normalized_source_bytes
    crlf = tmp_path / "a.py"
    lf = tmp_path / "b.py"
    crlf.write_bytes(b"def f():\r\n    return 1\r\n")
    lf.write_bytes(b"def f():\n    return 1\n")
    assert normalized_source_bytes(crlf) == normalized_source_bytes(lf)
    assert b"\r" not in normalized_source_bytes(crlf)


def test_oracle_refuses_a_file_panel_without_label_end_dates(tmp_path):
    """File-backed panels require settlement dates and cannot use the synthetic exemption."""
    import pandas as pd
    panel = tmp_path / "panel.csv"
    pd.DataFrame({"date": ["2020-01-31"], "asset": ["A0"],
                  "forward_return": [0.0]}).to_csv(panel, index=False)
    oracle = ValidatorOracle(ROOT / "a_share_big_config.json", panel_csv=panel)
    with pytest.raises(ValueError, match="forward_return_end_date"):
        oracle._label_end_series(pd.read_csv(panel))


def test_in_memory_panels_may_declare_direct_label_semantics(tmp_path):
    """In-memory synthetic panels may explicitly declare direct labels without a settlement horizon."""
    import pandas as pd
    frame = pd.DataFrame({"date": pd.to_datetime(["2020-01-31"]), "asset": ["A0"],
                          "forward_return": [0.0]})
    oracle = ValidatorOracle(ROOT / "a_share_big_config.json")
    oracle.set_panel(frame, "synthetic_check", label_semantics="direct")
    assert oracle._label_end_series(frame) is None

def test_lockbox_drops_rows_whose_labels_settle_after_val_end(tmp_path):
    """A pre-boundary signal with a post-boundary label is excluded from discovery data."""
    import pandas as pd
    panel = tmp_path / "panel.csv"
    pd.DataFrame({
        "date": ["2022-11-30", "2022-12-31", "2022-12-31"],
        "asset": ["A0", "A0", "A1"],
        "forward_return": [0.01, 0.02, 0.03],
        "forward_return_end_date": ["2022-12-21", "2023-01-31", "2022-12-31"],
    }).to_csv(panel, index=False)
    oracle = ValidatorOracle(ROOT / "a_share_big_config.json", panel_csv=panel)
    oracle.val_end = pd.Timestamp("2022-12-31")
    kept = oracle.load_panel()
    assert len(kept) == 2                                    # the 2023-01-31 label is gone
    assert pd.to_datetime(kept["forward_return_end_date"]).max() <= oracle.val_end


def test_set_panel_requires_explicit_label_semantics(tmp_path):
    """In-memory panels must explicitly declare their label availability semantics."""
    import pandas as pd
    frame = pd.DataFrame({"date": pd.to_datetime(["2020-01-31"]), "asset": ["A0"],
                          "forward_return": [0.0]})
    oracle = ValidatorOracle(ROOT / "a_share_big_config.json")
    with pytest.raises(TypeError):
        oracle.set_panel(frame, "no_semantics_given")


def test_load_panel_resets_semantics_after_a_direct_panel(tmp_path):
    """Switching back to a file panel must not inherit the synthetic exemption."""
    import pandas as pd
    frame = pd.DataFrame({"date": pd.to_datetime(["2020-01-31"]), "asset": ["A0"],
                          "forward_return": [0.0]})
    panel = tmp_path / "panel.csv"
    pd.DataFrame({"date": ["2020-01-31"], "asset": ["A0"],
                  "forward_return": [0.0]}).to_csv(panel, index=False)
    oracle = ValidatorOracle(ROOT / "a_share_big_config.json")
    oracle.set_panel(frame, "synthetic", label_semantics="direct")
    oracle.panel_csv = panel
    oracle._panel_source = None
    with pytest.raises(ValueError, match="forward_return_end_date"):
        oracle.load_panel()

def test_oracle_threads_configured_columns_to_the_shared_validator(tmp_path):
    """Call-level: a real oracle whose config uses non-default column names must load a
    panel keyed by those names -- the threading exists precisely so this cannot regress
    to the default "date"/"forward_return" lookup."""
    import dataclasses
    import pandas as pd

    panel = tmp_path / "panel.csv"
    pd.DataFrame({
        "trade_dt": ["2020-01-31", "2020-02-29"],
        "asset": ["A0", "A0"],
        "fwd_ret": [0.01, 0.02],
        "forward_return_end_date": ["2020-02-28", "2020-03-31"],
    }).to_csv(panel, index=False)
    oracle = ValidatorOracle(ROOT / "a_share_big_config.json", panel_csv=panel)
    oracle.base_cfg = dataclasses.replace(oracle.base_cfg, date_col="trade_dt",
                                          target_col="fwd_ret")
    oracle.val_end = pd.Timestamp("2020-12-31")
    kept = oracle.load_panel()                 # raises "no 'date' column" if not threaded
    assert len(kept) == 2


def test_lockbox_forwards_configured_columns_through_a_real_call(
        monkeypatch, tmp_path):
    """Dynamic, not source-grep: monkeypatch label_end_series with a recorder and drive
    a REAL open_test_lockbox call. The recorder must receive the run config's column
    names -- if the call site stops forwarding them, this fails even though the source
    string a static check would grep for might survive in a comment."""
    import json

    import pandas as pd

    import agent.tools as tools
    from agent.certification import content_hash
    from agent.run_context import RunContext

    dates = pd.date_range("2019-01-31", periods=70, freq="ME")
    # Non-default names verify that callers forward the frozen configuration fields.
    frame = pd.DataFrame(
        [{"trade_dt": d.strftime("%Y-%m-%d"), "asset": f"A{i}",
          "a": float(i) * 0.1, "fwd_ret": 0.01,
          "forward_return_end_date": d.strftime("%Y-%m-%d")}
         for d in dates for i in range(6)])
    panel = tmp_path / "panel.csv"
    frame.to_csv(panel, index=False)

    raw = json.loads((ROOT / "a_share_big_config.json").read_text(encoding="utf-8"))
    raw["factor_cols"] = ["a"]
    raw["family_constraints"]["factor_families"] = {"a": "f1"}
    raw["date_col"] = "trade_dt"
    raw["target_col"] = "fwd_ret"
    config = tmp_path / "config.json"
    config.write_text(json.dumps(raw), encoding="utf-8")

    snapshot = {"pipeline_config": raw,
                "oracle_gate_params": {"train_end": "2019-12-31",
                                       "val_end": "2022-12-31"}}
    ctx = RunContext.create(purpose="test", universe="data_tiny",
                            config_snapshot=snapshot,
                            manifest_extra={"panel_csv": str(panel)},
                            runs_root=tmp_path / "runs")
    ctx.write_card("clean", {"name": "clean", "expression": "zscore(a)",
                             "decision": "ACCEPT", "universe": "data_tiny",
                             "family": "f1", "data_version": content_hash(panel)})
    ctx.set_state("COMPLETED")

    import factor_research.factor_validity_pipeline as fvp

    calls = []
    real = fvp.label_end_series

    def recorder(frame, *, context, date_col="date", target_col="forward_return",
                 allow_missing=False):
        calls.append({"date_col": date_col, "target_col": target_col})
        return real(frame, context=context, date_col=date_col,
                    target_col=target_col, allow_missing=allow_missing)

    # the lockbox imports label_end_series inside the function, so the patch must land
    # on the source module it resolves from at call time
    monkeypatch.setattr(fvp, "label_end_series", recorder)
    # no config= override: the lockbox rightly refuses one -- the run's frozen
    # config.snapshot.json IS the configuration
    result = tools.open_test_lockbox(run_dir=str(ctx.run_dir), panel_csv=str(panel),
                                     confirm=tools.LOCKBOX_CONFIRM)
    assert calls, f"open_test_lockbox never consulted the shared label validator: {result}"
    assert calls[0] == {"date_col": "trade_dt", "target_col": "fwd_ret"}
