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

REPO = Path(__file__).resolve().parent.parent.parent
README = REPO / "README.md"

# Overclaims that are false wherever they appear: a safety guarantee mixle cannot make,
# frontier-training capability it does not have, and universal engine/backend portability
# that is bounded to the supported engines. A reader meets these as statements about what
# mixle *is*, so the README is not the only surface that has to stay honest about them.
UNIVERSAL_FORBIDDEN: dict[str, str] = {
    "safe to put in front of users": "a safety guarantee the library cannot make",
    "lab-grade ai": "implies frontier-scale training capability",
    "on any engine": "universal-engine overclaim; portability is bounded",
    "across any backend": "universal-backend overclaim; only supported backends",
    "runs unchanged": "overstates cross-backend portability",
}

# Phrases the credibility pass ruled on for the README's *opening positioning*, where they
# read as unbounded. They stay legal elsewhere because a bounded, checkable local use is a
# different claim: "scale as a flag, not a rewrite" names one keyword argument, and it is
# true. Banning them project-wide would reject accurate prose, so this tier is not widened.
README_FORBIDDEN: dict[str, str] = {
    "not a rewrite": "absolute claim in the opening; use 'rather than a rewrite'",
    "does the heavy lifting": "obscures that the user still owns modeling judgment",
}

FORBIDDEN: dict[str, str] = {**UNIVERSAL_FORBIDDEN, **README_FORBIDDEN}


def _claim_surfaces() -> list[Path]:
    """Every shipped surface whose prose a reader takes as a claim about mixle.

    ``docs/audits/`` is excluded deliberately: an audit quotes an overclaim in order to
    rule on it, so scanning it would make this gate fire on its own evidence.
    """
    surfaces = [README, REPO / "CHANGELOG.md"]
    surfaces += sorted((REPO / "docs").rglob("*.rst"))
    surfaces += sorted((REPO / "examples").rglob("*.py"))
    return [path for path in surfaces if path.is_file() and "audits" not in path.parts]


CLAIM_SURFACES = _claim_surfaces()

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


def test_claim_surfaces_were_actually_discovered() -> None:
    """A scan over an empty file list passes vacuously; require the surfaces to exist."""
    assert README in CLAIM_SURFACES, "the README must be among the scanned claim surfaces"
    assert len(CLAIM_SURFACES) > 20, (
        f"expected the docs/ and examples/ claim surfaces to be discovered, found {len(CLAIM_SURFACES)}"
    )
    assert not [path for path in CLAIM_SURFACES if "audits" in path.parts]


@pytest.mark.parametrize("phrase, reason", sorted(UNIVERSAL_FORBIDDEN.items()))
def test_universal_overclaim_absent_from_every_claim_surface(phrase: str, reason: str) -> None:
    """These five are false wherever they appear, so no shipped surface may carry them."""
    offenders = []
    for path in CLAIM_SURFACES:
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        if phrase in text:
            line = text[: text.find(phrase)].count("\n") + 1
            offenders.append(f"{path.relative_to(REPO)}:{line}")
    assert not offenders, f"overclaim {phrase!r} appears in {offenders}; it is not allowed because: {reason}."


def test_negative_control_detects_a_planted_overclaim() -> None:
    """Guard the guard: the forbidden scan must fire on a planted phrase."""
    planted = "This model is Lab-grade AI and safe to put in front of users.\n"
    low = planted.lower()
    hits = [p for p in FORBIDDEN if p in low]
    assert "lab-grade ai" in hits and "safe to put in front of users" in hits


def test_negative_control_covers_a_non_readme_surface(tmp_path: Path) -> None:
    """Guard the widened scan: planting an overclaim in a non-README surface must be caught.

    This is the property that distinguishes the widened gate from the README-only one it
    replaced, so it is asserted against the same read-and-search step the scan uses.
    """
    planted = tmp_path / "some_example.py"
    planted.write_text('"""Fits on any engine you have."""\n', encoding="utf-8")
    text = planted.read_text(encoding="utf-8", errors="replace").lower()
    assert "on any engine" in text
    assert [phrase for phrase in UNIVERSAL_FORBIDDEN if phrase in text] == ["on any engine"]
