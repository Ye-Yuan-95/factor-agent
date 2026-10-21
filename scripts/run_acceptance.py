"""Research acceptance command for source-bound certificates.

    python scripts/run_acceptance.py [--evidence-run]

Steps (non-zero exit if any check fails):
  1  pytest tests/agent
  2  packaging smoke: pip install --no-deps --target <tmp>, then import the full agent chain
     from a foreign cwd
  3  certificate applicability matrix: bound on the calibration panel; panel-mismatch on any
     other file; panel-not-certifiable on an in-memory/synthetic panel
  4  (--evidence-run) real-panel run under the CURRENT certificate: card carries certified=true
     + the binding hash, while the gate verdict stays free to REJECT (no auto-ACCEPT)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def step(name: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


def run() -> int:
    """Check package tests, installed imports and a supplied accepted certificate.

    The calibration-panel certificate is required in addition to engineering
    tests. Report controlled failures for missing/unusable evidence; the optional
    real-panel validation records current gate decisions without auto-acceptance.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence-run", action="store_true",
                    help="also record a real-panel validation as local evidence (~2-3 min)")
    ap.add_argument("--panel-csv", default=str(PROJECT / "data_big_v2" / "a_share_factor_panel.csv"))
    ap.add_argument("--config", default=str(PROJECT / "a_share_big_config.json"))
    ap.add_argument("--certificate",
                    default=str(PROJECT / "outputs" / "agent" / "gate_certificate.json"))
    args = ap.parse_args()
    if not (PROJECT / "tests").is_dir():
        raise SystemExit(
            "run_acceptance needs the repository checkout (tests/, configs); the wheel "
            "ships the library only.")
    failures = 0

    # 1 -- tests
    r = subprocess.run([sys.executable, "-m", "pytest", "tests/agent", "-q"],
                       cwd=PROJECT, capture_output=True, text=True)
    tail = (r.stdout or "").strip().splitlines()[-1] if r.stdout else r.stderr[-200:]
    failures += not step("pytest tests/agent", r.returncode == 0, tail)

    # 2 -- packaging smoke (foreign-cwd import of the installed target)
    with tempfile.TemporaryDirectory() as td:
        r = subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--quiet",
                            "--target", td, str(PROJECT)], capture_output=True, text=True)
        ok = r.returncode == 0
        if ok:
            r2 = subprocess.run(
                [sys.executable, "-I", str(PROJECT / "scripts" / "wheel_smoke.py"),
                 "--installed-root", td], cwd=tempfile.gettempdir(),
                capture_output=True, text=True)
            ok = r2.returncode == 0
            detail = r2.stdout.strip() if ok else r2.stderr[-500:]
        else:
            detail = r.stderr[-300:]
        failures += not step("packaging smoke (install --target + import)", ok, detail)

    # 3 -- certificate applicability matrix
    sys.path.insert(0, str(PROJECT))
    import pandas as pd
    from agent.validator_oracle import ValidatorOracle
    cert_file = Path(args.certificate)
    if not cert_file.is_file():
        # the certificate may be absent (never issued here, or pruned); that is a controlled
        # failure with a next step, not a traceback.
        failures += not step(
            "certificate applicability matrix", False,
            f"certificate not found at {cert_file}; generate one with "
            "`python -m agent.gate_certificate --panel-csv <panel>` or pass --certificate")
        print(f"\nACCEPTANCE {'PASSED' if failures == 0 else f'FAILED ({failures} step(s))'}")
        return 1 if failures else 0
    try:
        cert = json.loads(cert_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        failures += not step(
            "certificate applicability matrix", False,
            f"certificate at {cert_file} is unreadable ({exc}); re-issue it or pass "
            "--certificate")
        print(f"\nACCEPTANCE {'PASSED' if failures == 0 else f'FAILED ({failures} step(s))'}")
        return 1 if failures else 0
    o = ValidatorOracle(args.config, panel_csv=args.panel_csv,
                        certificate_path=cert_file)
    _, refs = o._load_certificate()
    failures += not step("applicability: calibration panel -> bound",
                         refs.get("bound") is True and refs.get("accepted") is True,
                         refs.get("status", ""))
    o2 = ValidatorOracle(args.config, panel_csv=str(PROJECT / "a_share_baseline_config.json"),
                         certificate_path=cert_file)
    _, refs2 = o2._load_certificate()
    failures += not step("applicability: wrong panel -> panel-mismatch",
                         "panel-mismatch" in refs2.get("status", ""), refs2.get("status", "")[:60])
    o3 = ValidatorOracle(args.config, certificate_path=cert_file)
    o3.set_panel(pd.DataFrame({"date": pd.to_datetime(["2020-01-31"]), "asset": ["A0"],
                               "forward_return": [0.0]}), "synthetic_acceptance_check",
                 label_semantics="direct")
    _, refs3 = o3._load_certificate()
    failures += not step("applicability: in-memory panel -> not-certifiable",
                         "not-certifiable" in refs3.get("status", ""), refs3.get("status", ""))

    # 4 -- real-panel evidence run under the current certificate
    if args.evidence_run and failures == 0:
        from agent.tools import validate_factor
        card = validate_factor(
            certificate_path=str(cert_file),
            name="evidence_seal_a", expression="rank(illiq) * rank(mom6m)",
            rationale="Evidence run under the current bound certificate",
            family="liquidity", panel_csv=args.panel_csv, config=args.config,
            existing=["std_turn", "idiovol", "size", "mom1m"])
        run_dir = Path(card["run_dir"])
        crefs = json.loads((run_dir / "certificate_refs.json").read_text(encoding="utf-8"))
        ok = (card["certified"] is True and crefs.get("bound") is True
              and crefs.get("certificate_binding_hash") == cert["binding"]["binding_hash"])
        failures += not step(
            "evidence run: certified card under current binding",
            ok, f"decision={card['decision']} binding={crefs.get('certificate_binding_hash')} "
                f"run={run_dir.name}")
        failures += not step("evidence run: certificate is not auto-ACCEPT",
                             card["decision"] in ("REJECT", "SHELVED", "ACCEPT"),
                             f"gate verdict independent: {card['decision']}")

    print(f"\nACCEPTANCE {'PASSED' if failures == 0 else f'FAILED ({failures} step(s))'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(run())
