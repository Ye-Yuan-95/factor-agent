"""Run-scoping + sealed-run lockbox state machine and certification helpers.
the lockbox must evaluate the run's OWN frozen panel/config -- these tests
pin the data binding (manifest panel + content hash vs card data_version, no config override)."""
import json

import pandas as pd
import pytest

from agent.certification import content_hash, one_sided_lower, one_sided_upper, wilson_upper
from agent.run_context import RunContext
from agent.tools import open_test_lockbox, LOCKBOX_CONFIRM

from conftest import DATES, TRAIN_END


def _make_run(tmp_path, tiny_panel_csv, tiny_config_path, universe="data_tiny",
              state="COMPLETED", card=None, panel_override=None):
    """Create the run artifacts required by lockbox tests without accessing real market data."""
    snapshot = {
        "pipeline_config": json.loads(tiny_config_path.read_text(encoding="utf-8")),
        "oracle_gate_params": {"train_end": str(TRAIN_END.date()),
                               "val_end": str(pd.Timestamp(DATES[59]).date())},
    }
    ctx = RunContext.create(purpose="test", universe=universe, config_snapshot=snapshot,
                            manifest_extra={"panel_csv": str(panel_override or tiny_panel_csv)},
                            runs_root=tmp_path / "runs")
    if card:
        ctx.write_card(card["name"], card)
    if state != "ACTIVE":
        ctx.set_state(state)
    return ctx


def _accept_card(tiny_panel_csv, universe="data_tiny"):
    return {"name": "clean", "expression": "zscore(a)", "decision": "ACCEPT",
            "universe": universe, "family": "f1",
            "data_version": content_hash(tiny_panel_csv)}


def test_run_context_layout_and_trace(tmp_path, tiny_panel_csv, tiny_config_path):
    ctx = _make_run(tmp_path, tiny_panel_csv, tiny_config_path, state="ACTIVE",
                    card=_accept_card(tiny_panel_csv))
    m = ctx.manifest()
    assert m["state"] == "ACTIVE" and m["universe"] == "data_tiny"
    assert (ctx.run_dir / "config.snapshot.json").exists()
    events = [json.loads(l)["event"] for l in (ctx.run_dir / "trace.jsonl").open(encoding="utf-8")]
    assert "run_created" in events
    p1 = ctx.write_card("dup", {"name": "dup"})
    p2 = ctx.write_card("dup", {"name": "dup"})          # same name: no silent overwrite
    assert p1 != p2 and p1.exists() and p2.exists()


def test_run_context_rejects_path_like_card_names(tmp_path, tiny_panel_csv, tiny_config_path):
    ctx = _make_run(tmp_path, tiny_panel_csv, tiny_config_path, state="ACTIVE")
    for unsafe in ("../escape", "folder/card", "folder\\card", " model name "):
        with pytest.raises(ValueError, match="card name"):
            ctx.write_card(unsafe, {"name": unsafe})
    assert not (ctx.run_dir.parent / "escape.json").exists()


def test_sealed_run_blocks_writes_after_completion(tmp_path, tiny_panel_csv, tiny_config_path):
    ctx = _make_run(tmp_path, tiny_panel_csv, tiny_config_path, state="COMPLETED")
    with pytest.raises(PermissionError):
        ctx.write_card("late", {"name": "late"})
    with pytest.raises(PermissionError):                  # no backwards transition
        ctx.set_state("ACTIVE")


def test_lockbox_refusals(tmp_path, tiny_panel_csv, tiny_config_path):
    ctx = _make_run(tmp_path, tiny_panel_csv, tiny_config_path, state="COMPLETED",
                    card=_accept_card(tiny_panel_csv))
    r = open_test_lockbox(run_dir=str(ctx.run_dir), confirm="nope")
    assert not r["ok"] and "refused" in r["error"]
    active = _make_run(tmp_path, tiny_panel_csv, tiny_config_path, state="ACTIVE",
                       card=_accept_card(tiny_panel_csv))
    r = open_test_lockbox(run_dir=str(active.run_dir), confirm=LOCKBOX_CONFIRM)
    assert not r["ok"] and "COMPLETED" in r["error"]
    synth = _make_run(tmp_path, tiny_panel_csv, tiny_config_path, universe="synthetic_gen1_s7",
                      state="COMPLETED", card=_accept_card(tiny_panel_csv, "synthetic_gen1_s7"))
    r = open_test_lockbox(run_dir=str(synth.run_dir), confirm=LOCKBOX_CONFIRM)
    assert not r["ok"] and "synthetic" in r["error"]


def test_lockbox_binds_frozen_data_and_config(tmp_path, tiny_panel_csv, tiny_config_path):
    """caller cannot substitute panel or config; card data_version must match."""
    ctx = _make_run(tmp_path, tiny_panel_csv, tiny_config_path, state="COMPLETED",
                    card=_accept_card(tiny_panel_csv))
    # config override refused outright
    r = open_test_lockbox(run_dir=str(ctx.run_dir), confirm=LOCKBOX_CONFIRM,
                          config=str(tiny_config_path))
    assert not r["ok"] and "config override" in r["error"]
    # a different panel than the manifest's is refused
    other = tmp_path / "other_panel.csv"
    other.write_text(tiny_panel_csv.read_text(encoding="utf-8")[:2000], encoding="utf-8")
    r = open_test_lockbox(run_dir=str(ctx.run_dir), confirm=LOCKBOX_CONFIRM,
                          panel_csv=str(other))
    assert not r["ok"] and "frozen panel" in r["error"]
    # manifest panel whose CONTENT no longer matches the cards' data_version is refused
    tampered = tmp_path / "tampered.csv"
    tampered.write_text(tiny_panel_csv.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    ctx2 = _make_run(tmp_path, tiny_panel_csv, tiny_config_path, state="COMPLETED",
                     card=_accept_card(tiny_panel_csv), panel_override=tampered)
    r = open_test_lockbox(run_dir=str(ctx2.run_dir), confirm=LOCKBOX_CONFIRM)
    assert not r["ok"] and "content hash" in r["error"]


def test_lockbox_opens_once_then_seals(tmp_path, tiny_panel_csv, tiny_config_path):
    ctx = _make_run(tmp_path, tiny_panel_csv, tiny_config_path, state="COMPLETED",
                    card=_accept_card(tiny_panel_csv))
    r1 = open_test_lockbox(run_dir=str(ctx.run_dir), confirm=LOCKBOX_CONFIRM)
    assert r1["ok"], r1
    assert (ctx.run_dir / "lockbox_report.json").exists()
    assert r1["panel_content_hash"] == content_hash(tiny_panel_csv)
    assert ctx.state == "TEST_OPENED"
    r2 = open_test_lockbox(run_dir=str(ctx.run_dir), confirm=LOCKBOX_CONFIRM)
    assert not r2["ok"] and "one-shot" in r2["error"]     # one-shot


def test_certification_helpers(tiny_panel_csv):
    h1 = content_hash(tiny_panel_csv)
    assert h1 == content_hash(tiny_panel_csv) and len(h1) == 16
    vals = [0.02, 0.05, 0.0, 0.03, 0.04, 0.01]
    assert one_sided_upper(vals) > sum(vals) / len(vals) > one_sided_lower(vals)
    assert abs(wilson_upper(3, 15, conf=0.975) - 0.4516) < 0.01   # expected interval [0.07, 0.45]


def test_lockbox_atomic_claim_blocks_concurrent_and_crashed_opens(tmp_path, tiny_panel_csv,
                                                                  tiny_config_path):
    """the O_EXCL claim makes opening one-shot even under crash/concurrency."""
    # simulated concurrent claim: first process moved the run to TEST_OPENING
    ctx = _make_run(tmp_path, tiny_panel_csv, tiny_config_path, state="COMPLETED",
                    card=_accept_card(tiny_panel_csv))
    ctx.claim_lockbox()
    assert ctx.state == "TEST_OPENING" and ctx.lock_path.exists()
    r = open_test_lockbox(run_dir=str(ctx.run_dir), confirm=LOCKBOX_CONFIRM)
    assert not r["ok"]
    # simulated crash BETWEEN lock creation and manifest write: state COMPLETED but lock exists
    ctx2 = _make_run(tmp_path, tiny_panel_csv, tiny_config_path, state="COMPLETED",
                     card=_accept_card(tiny_panel_csv))
    ctx2.lock_path.write_text("{}", encoding="utf-8")
    r = open_test_lockbox(run_dir=str(ctx2.run_dir), confirm=LOCKBOX_CONFIRM)
    assert not r["ok"] and "claimed" in r["error"]
    assert ctx2.state == "COMPLETED"                      # refused BEFORE any test read
    # terminal states accept no transitions
    ctx.set_state("TEST_OPEN_FAILED", error="simulated")
    with pytest.raises(PermissionError):
        ctx.set_state("TEST_OPENED")


def test_lockbox_refuses_missing_snapshot_splits(tmp_path, tiny_panel_csv, tiny_config_path):
    """no default-date fallback -- absent train/val splits refuse BEFORE the claim."""
    snapshot = {"pipeline_config": json.loads(tiny_config_path.read_text(encoding="utf-8")),
                "oracle_gate_params": {}}                 # splits missing
    ctx = RunContext.create(purpose="test", universe="data_tiny", config_snapshot=snapshot,
                            manifest_extra={"panel_csv": str(tiny_panel_csv)},
                            runs_root=tmp_path / "runs")
    ctx.write_card("clean", _accept_card(tiny_panel_csv))
    ctx.set_state("COMPLETED")
    r = open_test_lockbox(run_dir=str(ctx.run_dir), confirm=LOCKBOX_CONFIRM)
    assert not r["ok"] and "train_end" in r["error"]
    assert ctx.state == "COMPLETED" and not ctx.lock_path.exists()   # claim never happened
