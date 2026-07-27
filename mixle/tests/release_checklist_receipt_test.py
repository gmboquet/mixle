"""Release checklist completion requires exact candidate receipts."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _module():
    path = ROOT / "scripts" / "verify_release_checklist.py"
    spec = importlib.util.spec_from_file_location("_verify_release_checklist", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_implemented_status_is_not_final_evidence(tmp_path: Path) -> None:
    checklist = tmp_path / "checklist.md"
    checklist.write_text("| Gate | Status | Evidence |\n|---|---|---|\n| Build | `IMPLEMENTED` | local |\n")
    errors = _module().validate(checklist, tmp_path, "a" * 40, require_final=True)
    assert errors == ["Build: pre-publication status remains IMPLEMENTED"]


def test_done_receipt_must_match_candidate_and_measure_result(tmp_path: Path) -> None:
    checklist = tmp_path / "checklist.md"
    checklist.write_text("| Gate | Status | Evidence |\n|---|---|---|\n| Build | `DONE` | receipt |\n")
    receipt = {
        "artifact": "mixle.release_gate_receipt/v1",
        "gate": "Build",
        "candidate_commit": "a" * 40,
        "command": "python -m build",
        "result": "2 artifacts verified",
        "date": "2026-07-27",
    }
    (tmp_path / "build.json").write_text(json.dumps(receipt), encoding="utf-8")
    assert _module().validate(checklist, tmp_path, "a" * 40, require_final=True) == []
    assert _module().validate(checklist, tmp_path, "b" * 40, require_final=True) == [
        "Build: receipt commit differs"
    ]
