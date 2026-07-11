"""Worklist X12.1 -- keep the README's claims mapped to what mixle actually is.

The 0.8.0 credibility pass rewrote the README opening around the stable thesis and
removed four specific overclaims. This test pins that outcome so the removed phrases do
not creep back in a later edit, and so the required qualifications stay present:

  * "safe to put in front of users"  -- a safety guarantee the library cannot make;
  * "Lab-grade AI, without the lab"  -- implies frontier-scale training;
  * universal engine/backend language ("on any engine", "across any backend",
    "runs unchanged", "not a rewrite") -- portability is real but bounded to the
    supported engines/backends and maturity limits;
  * the implication that one-line fitting removes the modeling judgment (data,
    objective, validation).

It also asserts the two positive requirements from the X12.1 acceptance criteria: a
visible maturity link, and an explicit disclaimer that mixle does not train frontier
models.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

README = Path(__file__).resolve().parent.parent.parent / "README.md"

# Phrases that must not reappear. Each maps a banned substring (case-insensitive) to the
# reason it was removed, so a future failure explains itself.
FORBIDDEN: dict[str, str] = {
    "safe to put in front of users": "a safety guarantee the library cannot make",
    "lab-grade ai": "implies frontier-scale training capability",
    "on any engine": "universal-engine overclaim; portability is bounded",
    "across any backend": "universal-backend overclaim; only supported backends",
    "runs unchanged": "overstates cross-backend portability",
    "not a rewrite": "absolute claim; use 'rather than a rewrite'",
    "does the heavy lifting": "obscures that the user still owns modeling judgment",
}

# Substrings that must be present (case-insensitive) with the reason each is required.
REQUIRED: dict[str, str] = {
    "maturity.html": "the opening must carry a visible maturity link (X12.1)",
    "does not train frontier models": "must explicitly disclaim frontier training",
    "not the modeling judgment": "must qualify that one-line fitting keeps modeling judgment",
}


@pytest.fixture(scope="module")
def readme_text() -> str:
    if not README.is_file():
        pytest.skip(f"README not found at {README}")
    return README.read_text(encoding="utf-8")


@pytest.mark.parametrize("phrase, reason", sorted(FORBIDDEN.items()))
def test_forbidden_overclaim_absent(readme_text: str, phrase: str, reason: str) -> None:
    idx = readme_text.lower().find(phrase)
    assert idx == -1, (
        f"README reintroduces the removed overclaim {phrase!r} (at offset {idx}); it was removed because: {reason}."
    )


@pytest.mark.parametrize("phrase, reason", sorted(REQUIRED.items()))
def test_required_qualifier_present(readme_text: str, phrase: str, reason: str) -> None:
    assert phrase.lower() in readme_text.lower(), f"README is missing required text {phrase!r}: {reason}."


def test_maturity_link_is_a_real_link(readme_text: str) -> None:
    """The maturity reference must be a markdown link, not bare prose."""
    assert re.search(r"\[[^\]]*maturity[^\]]*\]\([^)]*maturity\.html[^)]*\)", readme_text, re.IGNORECASE), (
        "the maturity reference must be a clickable markdown link to maturity.html"
    )


def test_negative_control_detects_a_planted_overclaim() -> None:
    """Guard the guard: the forbidden scan must fire on a planted phrase."""
    planted = "This model is Lab-grade AI and safe to put in front of users.\n"
    low = planted.lower()
    hits = [p for p in FORBIDDEN if p in low]
    assert "lab-grade ai" in hits and "safe to put in front of users" in hits
