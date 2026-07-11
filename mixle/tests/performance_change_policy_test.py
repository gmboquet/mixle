"""Worklist B7.5 -- the performance-change policy states the full required workflow.

B7.5's acceptance: no performance PR without a before/after result and a correctness
gate. This test guards the policy document so the required workflow cannot be quietly
weakened: it must name every step (profile, hypothesis, patch, parity, benchmark, memory,
receipt), require both a before/after result and a correctness/parity gate, and keep the
"only fix measured bottlenecks / no abstraction just to cut lines" discipline.
"""

from __future__ import annotations

from pathlib import Path

import pytest

DOC = Path(__file__).resolve().parent.parent.parent / "release-checklists" / "0.8.0-performance-change-policy.md"

REQUIRED_STEPS = ("profile", "hypothesis", "patch", "parity", "benchmark", "memory", "receipt")


def _doc() -> str:
    if not DOC.is_file():
        pytest.skip(f"{DOC} not found")
    return DOC.read_text(encoding="utf-8")


def test_all_workflow_steps_present() -> None:
    text = _doc().lower()
    missing = [s for s in REQUIRED_STEPS if s not in text]
    assert not missing, f"performance policy is missing workflow steps: {missing}"


def test_requires_before_after_and_correctness_gate() -> None:
    text = _doc().lower()
    assert "before/after" in text or ("before" in text and "after" in text), "policy must require a before/after result"
    assert "correctness gate" in text or "parity" in text, (
        "policy must require a correctness/parity gate (B7.5 acceptance)"
    )


def test_keeps_measured_bottleneck_discipline() -> None:
    text = _doc().lower()
    assert "measured bottleneck" in text or "only fix measured" in text or "not optimize on intuition" in text
    assert "line count" in text, "policy must keep the 'no abstraction just to cut lines' rule"


def test_has_a_receipt_template() -> None:
    text = _doc().lower()
    assert "receipt template" in text or "performance change receipt" in text, (
        "policy must include a retained-receipt template so claims are auditable"
    )
