"""Required copyright notices are preserved and publication fails pending approval."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RECORD = ROOT / "release-checklists" / "0.8.0-license-provenance.json"


def _load():
    path = ROOT / "scripts" / "verify_license_provenance.py"
    spec = importlib.util.spec_from_file_location("_verify_license_provenance", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_notices_are_preserved_but_pending_record_blocks_publication() -> None:
    module = _load()
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    assert module.validate(record) == []
    errors = module.validate(record, require_approved=True)
    assert any("approval is required" in error for error in errors)
    notice = record["original_notice"]
    assert notice in (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert notice in (ROOT / "NOTICE").read_text(encoding="utf-8")


def test_complete_independent_approval_record_passes() -> None:
    module = _load()
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    record.update(
        {
            "status": "approved",
            "reviewer": "named independent counsel",
            "review_date": "2026-07-27",
            "evidence": "retained authorization record URL or digest",
        }
    )
    assert module.validate(record, require_approved=True) == []


def test_publish_workflow_requires_approval_before_release_checks() -> None:
    workflow = (ROOT / ".github" / "workflows" / "publish.yml").read_text(encoding="utf-8")
    approval = workflow.index("verify_license_provenance.py --require-approved")
    checks = workflow.index("verify_required_checks.py")
    assert approval < checks
