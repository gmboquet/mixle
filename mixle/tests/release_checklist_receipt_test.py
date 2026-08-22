"""Release checklist completion requires exact candidate receipts."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _module():
    path = ROOT / "scripts" / "verify_release_checklist.py"
    spec = importlib.util.spec_from_file_location("_verify_release_checklist", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _receipt(**overrides: object) -> dict:
    receipt = {
        "artifact": "mixle.release_gate_receipt/v1",
        "gate": "Build",
        "candidate_commit": "a" * 40,
        "evidence_digest": "d" * 64,
        "command": "python -m build",
        "result": "2 artifacts verified",
        "date": "2026-07-27",
    }
    receipt.update(overrides)
    return receipt


def _checklist(tmp_path: Path, status: str) -> Path:
    checklist = tmp_path / "checklist.md"
    checklist.write_text(f"| Gate | Status | Evidence |\n|---|---|---|\n| Build | `{status}` | e |\n")
    return checklist


def test_implemented_status_is_not_final_evidence(tmp_path: Path) -> None:
    errors = _module().validate(_checklist(tmp_path, "IMPLEMENTED"), tmp_path, "a" * 40, require_final=True)
    assert errors == ["Build: pre-publication status remains IMPLEMENTED"]


def test_done_receipt_must_measure_a_result_on_this_candidate_content(tmp_path: Path) -> None:
    checklist = _checklist(tmp_path, "DONE")
    validate = _module().validate
    (tmp_path / "build.json").write_text(json.dumps(_receipt()), encoding="utf-8")
    assert validate(checklist, tmp_path, "b" * 40, require_final=True, digest="d" * 64) == []
    # The receipt is bound to the candidate's content, not to the SHA of the commit that carries it:
    # requiring the latter is unsatisfiable for an in-repo file (D-0195).
    errors = validate(checklist, tmp_path, "b" * 40, require_final=True, digest="e" * 64)
    assert len(errors) == 1 and "measured on different bytes" in errors[0]


def test_done_receipt_must_still_name_the_commit_it_measured(tmp_path: Path) -> None:
    checklist = _checklist(tmp_path, "DONE")
    (tmp_path / "build.json").write_text(json.dumps(_receipt(candidate_commit="HEAD")), encoding="utf-8")
    errors = _module().validate(checklist, tmp_path, "a" * 40, require_final=True, digest="d" * 64)
    assert errors == ["Build: receipt names no measured commit"]


def test_done_receipt_must_report_a_measured_number(tmp_path: Path) -> None:
    checklist = _checklist(tmp_path, "DONE")
    (tmp_path / "build.json").write_text(json.dumps(_receipt(result="passed")), encoding="utf-8")
    errors = _module().validate(checklist, tmp_path, "a" * 40, require_final=True, digest="d" * 64)
    assert errors == ["Build: receipt result has no measured value"]


def test_evidence_digest_covers_the_tree_but_not_the_receipts(tmp_path: Path) -> None:
    """Changing any tracked file invalidates every receipt; writing a receipt does not."""
    run = lambda *a: subprocess.run(["git", "-C", str(tmp_path), *a], check=True, capture_output=True)
    run("init", "-q")
    run("config", "user.email", "t@example.invalid")
    run("config", "user.name", "t")
    (tmp_path / "mixle").mkdir()
    (tmp_path / "mixle" / "core.py").write_text("x = 1\n")
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    run("add", "-A")
    digest = _module().evidence_digest

    before = digest(tmp_path, receipts)
    (receipts / "build.json").write_text("{}\n")
    run("add", "-A")
    assert digest(tmp_path, receipts) == before, "a receipt must not change the digest it records"

    (tmp_path / "mixle" / "core.py").write_text("x = 2\n")
    run("add", "-A")
    assert digest(tmp_path, receipts) != before, "a source change must invalidate every receipt"
