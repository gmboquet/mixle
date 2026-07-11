"""Worklist E14.2 -- the adversarial statistical-review protocol is complete and honest.

E14.2 requires an adversarial review of automatic selection, calibration, uncertainty,
and latent-model claims, structured around five specific questions, executed by someone
other than the implementer. This test guards the *instrument* for that review: the
protocol must cover all five required questions, carry a findings table for the reviewer
to fill in, and stay honestly labeled as a protocol -- not misrepresented as a completed
external review.
"""

from __future__ import annotations

from pathlib import Path

import pytest

DOC = Path(__file__).resolve().parent.parent.parent / "release-checklists" / "0.8.0-statistical-review.md"

# The five E14.2 questions, by a distinctive phrase each.
REQUIRED_QUESTIONS = (
    "estimand",  # Q1
    "leakage",  # Q2
    "calibration guarantees",  # Q3
    "approximate routes",  # Q4
    "scientific validity",  # Q5
)


def _doc() -> str:
    if not DOC.is_file():
        pytest.skip(f"{DOC} not found")
    return DOC.read_text(encoding="utf-8")


def test_all_five_questions_are_covered() -> None:
    text = _doc().lower()
    missing = [q for q in REQUIRED_QUESTIONS if q not in text]
    assert not missing, f"statistical-review protocol is missing required questions: {missing}"


def test_protocol_has_a_findings_table() -> None:
    text = _doc()
    assert "## Findings" in text, "protocol must include a Findings table for the reviewer"
    assert "Severity" in text, "findings table must track severity (acceptance keys on high-severity)"
    assert "Resolution" in text, "findings table must link the resolving PR"


def test_protocol_is_honestly_labeled_not_a_completed_review() -> None:
    """It must not misrepresent itself as an already-done independent review."""
    text = _doc().lower()
    assert "protocol, not a completed review" in text or "protocol, not a" in text, (
        "the doc must state it is a protocol to be executed, not a finished external review"
    )
    assert "other than the implementer" in text, (
        "the protocol must require execution by someone other than the implementer (E14.2)"
    )


def test_protocol_ties_acceptance_to_rc1() -> None:
    text = _doc().lower()
    assert "rc1" in text and ("resolved" in text or "reduced" in text), (
        "protocol must tie high-severity resolution to the RC1 independent-review gate"
    )
