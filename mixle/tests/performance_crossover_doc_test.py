"""Worklist B7.6 -- performance documentation must contain losses as well as wins.

B7.6's acceptance is explicit: the performance docs must state where specialized
packages (scikit-learn, hmmlearn, pomegranate) are faster than mixle, distinguish
generality overhead from kernel inefficiency, and say whether GPU numbers are
throughput, latency, or capability demonstrations. This test pins that the honest
crossover page keeps saying those things -- a later edit cannot quietly turn it into a
wins-only page.
"""

from __future__ import annotations

from pathlib import Path

import pytest

DOC = Path(__file__).resolve().parent.parent.parent / "docs" / "performance-crossover.rst"
INDEX = Path(__file__).resolve().parent.parent.parent / "docs" / "index.rst"


def _doc() -> str:
    if not DOC.is_file():
        pytest.skip(f"{DOC} not found")
    return DOC.read_text(encoding="utf-8")


def test_doc_names_specialized_packages_that_win() -> None:
    text = _doc().lower()
    for pkg in ("scikit-learn", "hmmlearn"):
        assert pkg in text, f"crossover doc must name {pkg!r} as a package that can be faster"


def test_doc_states_a_loss_not_only_wins() -> None:
    """It must concede that a specialized package wins, in plain language."""
    text = _doc().lower()
    assert "faster" in text, "doc must acknowledge a competitor being faster"
    # A concrete concession phrase -- mixle does not win this comparison.
    assert "scikit-learn wins" in text or "does not claim" in text or "does not overtake" in text, (
        "crossover doc must explicitly concede a loss, not just describe methodology"
    )


def test_doc_distinguishes_generality_overhead() -> None:
    text = _doc().lower()
    assert "generality overhead" in text, (
        "doc must distinguish generality overhead from a worse/different algorithm (B7.6)"
    )
    # And anchor that both reach the same optima (parity), so it is overhead not a worse fit.
    assert "same optima" in text or "parity" in text, (
        "doc should note the fits reach the same optima, so the gap is overhead not inaccuracy"
    )


def test_doc_classifies_gpu_claims() -> None:
    text = _doc().lower()
    assert "throughput" in text and "latency" in text and "capability" in text, (
        "doc must state whether GPU/backend numbers are throughput, latency, or capability"
    )


def test_doc_is_registered_in_toctree() -> None:
    if not INDEX.is_file():
        pytest.skip("docs/index.rst not found")
    assert "performance-crossover" in INDEX.read_text(encoding="utf-8"), (
        "performance-crossover must be linked from the docs toctree so it is reachable"
    )
