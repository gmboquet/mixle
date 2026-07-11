"""Worklist E14.4 -- the external use-case intake asks for everything, honestly.

E14.4 requires one real external use case, with evidence covering the problem statement,
data shape, chosen model, failure points, runtime, and whether mixle reduced or increased
work -- and the acceptance that findings feed docs/bug fixes and marketing does not
cherry-pick only successes. This test guards the intake template so it keeps asking for
every element, including the honest "did mixle increase work?" question.
"""

from __future__ import annotations

from pathlib import Path

import pytest

DOC = Path(__file__).resolve().parent.parent.parent / "release-checklists" / "0.8.0-external-use-case-template.md"

REQUIRED = (
    "problem statement",
    "data shape",
    "chosen model",
    "failure points",
    "runtime",
    "reduce or increase",  # the honest both-outcomes question
)


def _doc() -> str:
    if not DOC.is_file():
        pytest.skip(f"{DOC} not found")
    return DOC.read_text(encoding="utf-8")


def test_intake_asks_for_every_required_element() -> None:
    text = _doc().lower()
    missing = [r for r in REQUIRED if r not in text]
    assert not missing, f"use-case intake is missing required elements: {missing}"


def test_intake_forbids_cherry_picking() -> None:
    text = _doc().lower()
    assert "cherry-pick" in text or "cherry pick" in text, (
        "the template must state marketing does not cherry-pick only successes (E14.4)"
    )
    assert "increase" in text and "reduce" in text, (
        "the template must record both increased-work and reduced-work outcomes"
    )


def test_intake_feeds_docs_and_bugs() -> None:
    text = _doc().lower()
    assert "docs to fix" in text or "docs" in text
    assert "bug" in text, "findings must feed bug fixes"


def test_is_labeled_a_template() -> None:
    text = _doc().lower()
    assert "status: template" in text or "template." in text
    assert "external" in text and "not design" in text, (
        "the use case must be on data the maintainer did not design (external)"
    )
