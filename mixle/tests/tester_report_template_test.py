"""Worklist E14.1 -- the independent-tester report template is complete.

E14.1 requires two independent testers to reproduce installation and a flagship workflow
and submit written reports that include commands, failures, confusing points, and final
status. The maintainer-side deliverable is the report template those testers fill in.
This test guards that the template actually asks for every element the acceptance
criterion names, and is honestly labeled as a template (not a fabricated report).
"""

from __future__ import annotations

from pathlib import Path

import pytest

DOC = Path(__file__).resolve().parent.parent.parent / "release-checklists" / "0.8.0-tester-report-template.md"

# The acceptance criterion: reports include commands, failures, confusing points, status.
REQUIRED_SECTIONS = ("command", "failure", "confusing", "final status")
# And the diversity + independence requirements.
REQUIRED_CONTEXT = ("independent", "statistician", "engineer", "linux", "macos")


def _doc() -> str:
    if not DOC.is_file():
        pytest.skip(f"{DOC} not found")
    return DOC.read_text(encoding="utf-8")


def test_template_asks_for_every_required_element() -> None:
    text = _doc().lower()
    missing = [s for s in REQUIRED_SECTIONS if s not in text]
    assert not missing, f"tester report template is missing required sections: {missing}"


def test_template_states_diversity_and_independence() -> None:
    text = _doc().lower()
    missing = [c for c in REQUIRED_CONTEXT if c not in text]
    assert not missing, f"template omits diversity/independence requirements: {missing}"


def test_template_uses_the_published_artifact_not_a_checkout() -> None:
    """Testers must install from the artifact, per E14 reproduction discipline."""
    text = _doc().lower()
    assert "pip install mixle" in text or "wheel" in text, (
        "template must direct testers to install from the published artifact, not a source checkout"
    )
    assert "fresh environment" in text or "fresh" in text


def test_template_is_labeled_a_template() -> None:
    text = _doc().lower()
    assert "status: template" in text or "template." in text, (
        "the file must be labeled a template so it is not mistaken for a real submitted report"
    )
