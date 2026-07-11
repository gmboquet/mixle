"""Worklist F10.5 -- classify showcase examples so a reader is not misled.

Several examples carry aspirational names ("frontier", "flagship", "showcase", "win
demo"). Every one of them runs on small synthetic / stand-in or in-file constructed
data -- none are measured results on a real frontier-scale dataset. That is a fine thing
for an illustration to be, but the name invites the opposite reading, so the distinction
must be made *prominent* rather than left implicit (F10.5: "rename 'frontier' examples
that use tiny stand-ins unless the distinction is unavoidable and prominent").

Rather than rename the files (which would break every doc cross-reference and the
execution manifest), we require each aspirational example to open with a recognizable
``Classification:`` tag naming its evidentiary status. This test enforces that the tag
is present, near the top of the module docstring, and names a known category. The
negative control proves the check fails when the tag is absent.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent.parent / "examples"

# Names that promise more than a small illustration delivers. Any example whose file
# name contains one of these tokens must classify itself.
ASPIRATIONAL_TOKENS = ("frontier", "flagship", "showcase", "win_demo")

# Recognized evidentiary categories a Classification tag may name.
#   illustrative -- synthetic / stand-in / constructed inputs; workflow shape, not a result
#   evidence     -- measured on a real, named public dataset
#   tutorial     -- teaches an API on toy inputs, makes no real-data claim
KNOWN_CATEGORIES = ("illustrative", "evidence", "tutorial")

# The tag must appear within the first few lines of the docstring to count as prominent.
PROMINENCE_LINE_BUDGET = 8

_CLASSIFICATION_RE = re.compile(r"^\s*Classification:\s*(.+)$", re.IGNORECASE)


def _aspirational_examples() -> list[Path]:
    if not EXAMPLES_DIR.is_dir():
        pytest.skip(f"examples directory not found at {EXAMPLES_DIR}")
    hits = [p for p in sorted(EXAMPLES_DIR.glob("*.py")) if any(tok in p.name.lower() for tok in ASPIRATIONAL_TOKENS)]
    if not hits:
        pytest.skip("no aspirationally-named examples found to classify")
    return hits


def _module_docstring(path: Path) -> str | None:
    return ast.get_docstring(ast.parse(path.read_text(encoding="utf-8")))


def _classification_line(docstring: str) -> tuple[str, int] | None:
    """Return (payload, line_index) of the first Classification: line, or None."""
    for idx, line in enumerate(docstring.splitlines()):
        m = _CLASSIFICATION_RE.match(line)
        if m:
            return m.group(1).strip(), idx
    return None


@pytest.mark.parametrize("path", _aspirational_examples(), ids=lambda p: p.name)
def test_aspirational_example_is_classified(path: Path) -> None:
    doc = _module_docstring(path)
    assert doc, f"{path.name}: aspirational example has no module docstring"

    found = _classification_line(doc)
    assert found is not None, (
        f"{path.name}: an aspirationally-named example must open with a "
        f"'Classification:' tag disclosing its evidentiary status "
        f"(one of {KNOWN_CATEGORIES}); none found in the module docstring."
    )

    payload, line_idx = found
    assert line_idx < PROMINENCE_LINE_BUDGET, (
        f"{path.name}: Classification tag appears on docstring line {line_idx}; "
        f"it must be within the first {PROMINENCE_LINE_BUDGET} lines to be prominent."
    )

    category = payload.split()[0].strip(" -:.").lower() if payload.split() else ""
    assert category in KNOWN_CATEGORIES, (
        f"{path.name}: Classification names '{category or payload!r}', which is not a "
        f"known category {KNOWN_CATEGORIES}."
    )

    # Every aspirational example we ship today is illustrative (synthetic / constructed
    # inputs). If one is later reclassified as 'evidence', it must justify that with a
    # real-data marker so the stronger claim is not made casually.
    if category == "evidence":
        low = doc.lower()
        assert any(marker in low for marker in ("real data", "real-data", "public dataset", "dataset:")), (
            f"{path.name}: classified as 'evidence' but the docstring names no real "
            f"dataset. An evidence claim must identify its measured source."
        )


def test_illustrative_examples_disclose_the_stand_in() -> None:
    """An 'illustrative' tag must say *why* -- synthetic, stand-in, or constructed."""
    offenders = []
    for path in _aspirational_examples():
        doc = _module_docstring(path) or ""
        found = _classification_line(doc)
        if not found:
            continue  # covered by the parametrized test above
        payload, _ = found
        category = payload.split()[0].strip(" -:.").lower() if payload.split() else ""
        if category != "illustrative":
            continue
        low = payload.lower()
        if not any(w in low for w in ("synthetic", "stand-in", "stand in", "constructed", "in-file", "toy")):
            offenders.append(path.name)
    assert not offenders, (
        "illustrative examples whose Classification line does not say the data is "
        f"synthetic / stand-in / constructed: {offenders}"
    )


def test_negative_control_missing_tag_is_detected() -> None:
    """Guard the guard: a docstring with no Classification tag must be caught."""
    doc = "Flashy frontier demo, end to end.\n\nDoes lots of impressive things.\n"
    assert _classification_line(doc) is None


def test_negative_control_unknown_category_is_detected() -> None:
    """A tag naming an unrecognized category must not pass the category check."""
    doc = "Frontier demo.\n\nClassification: groundbreaking -- trust me.\n"
    found = _classification_line(doc)
    assert found is not None
    payload, _ = found
    category = payload.split()[0].strip(" -:.").lower()
    assert category not in KNOWN_CATEGORIES


def test_negative_control_buried_tag_is_not_prominent() -> None:
    """A tag pushed far down the docstring must fail the prominence budget."""
    body = "\n".join(f"line {i}" for i in range(PROMINENCE_LINE_BUDGET + 3))
    doc = f"Frontier demo.\n\n{body}\nClassification: illustrative -- synthetic.\n"
    found = _classification_line(doc)
    assert found is not None
    _, line_idx = found
    assert line_idx >= PROMINENCE_LINE_BUDGET
