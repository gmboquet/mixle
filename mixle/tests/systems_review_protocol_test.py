"""Worklist E14.3 -- the adversarial systems-review protocol is complete and honest.

E14.3 requires an independent review of packaging, distributed execution, checkpointing,
and benchmark methodology, with the acceptance that the reviewer reproduces at least one
backend and one benchmark from a clean clone. This test guards the protocol instrument:
it must cover all four areas, name the clean-clone reproduction of a backend and a
benchmark, carry a findings table, and stay honestly labeled as a protocol.
"""

from __future__ import annotations

from pathlib import Path

import pytest

DOC = Path(__file__).resolve().parent.parent.parent / "release-checklists" / "0.8.0-systems-review.md"

REQUIRED_AREAS = ("packaging", "distributed execution", "checkpointing", "benchmark methodology")


def _doc() -> str:
    if not DOC.is_file():
        pytest.skip(f"{DOC} not found")
    return DOC.read_text(encoding="utf-8")


def test_all_four_areas_covered() -> None:
    text = _doc().lower()
    missing = [a for a in REQUIRED_AREAS if a not in text]
    assert not missing, f"systems-review protocol is missing areas: {missing}"


def test_requires_clean_clone_reproduction_of_backend_and_benchmark() -> None:
    text = _doc().lower()
    assert "clean clone" in text, "protocol must require reproduction from a clean clone"
    assert "one backend" in text or "≥1 backend" in text or "one distributed backend" in text
    assert "one benchmark" in text or "≥1 benchmark" in text


def test_has_findings_table_and_rc1_tie() -> None:
    text = _doc()
    assert "## Findings" in text and "Severity" in text
    assert "rc1" in text.lower() or "RC1" in text


def test_is_labeled_a_protocol() -> None:
    text = _doc().lower()
    assert "protocol, not a completed review" in text or "protocol, not a" in text
    assert "other than the implementer" in text
