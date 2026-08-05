"""Required copyright notices are preserved and the approved provenance record is honest."""

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


def test_notices_are_preserved_and_the_approved_record_is_complete_and_honest() -> None:
    """The record was approved 2026-08-04 by the release owner; this test pins WHAT was approved.

    Approval here is not the independent third-party review the field names could suggest: the
    reviewer is the release owner, who is also the sole author. The record says so in its own
    reviewer string, and the evidence cites only facts checkable inside this repository -- the
    LLNL/pysparkplug MIT origin (LLNL-CODE-844837) in the root commit and the author's own rights
    in every later commit. Dropping either the disclosure or the citations would upgrade the
    evidence by edit, which is what this test exists to prevent.
    """
    module = _load()
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    assert module.validate(record) == []
    assert module.validate(record, require_approved=True) == []
    assert record["status"] == "approved"
    assert "not an independent third party" in record["reviewer"]
    assert "LLNL-CODE-844837" in record["evidence"]
    assert "0e0905a68623e4321721183c3d7aff6f0ea1c6a1" in record["evidence"]
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
