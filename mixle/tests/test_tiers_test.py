"""Worklist T3.1 -- the test-tier taxonomy is declared, documented, and consistent.

T3.1 redesigns the suite around named tiers with time budgets instead of one broad
``fast`` marker. This test pins the taxonomy so the three sources agree:

  * the tier markers are declared in ``pyproject.toml`` (``--strict-markers`` then makes
    them usable and typo-proof);
  * every tier is documented in ``docs/test-tiers.rst`` with a budget;
  * a real ``smoke`` tier exists and is small.

It does not re-mark the whole suite (that migration is incremental); it guarantees the
vocabulary and the smoke tier are real and stay in sync.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
PYPROJECT = ROOT / "pyproject.toml"
TIERS_DOC = ROOT / "docs" / "test-tiers.rst"
SMOKE = Path(__file__).resolve().parent / "smoke_test.py"

# The tier vocabulary T3.1 introduces (plus the pre-existing optional/benchmark).
TIER_MARKERS = ("smoke", "core", "full", "optional", "numerical", "benchmark", "hardware")


def _declared_markers() -> dict[str, str]:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    markers = data["tool"]["pytest"]["ini_options"]["markers"]
    out = {}
    for entry in markers:
        name, _, desc = entry.partition(":")
        out[name.strip()] = desc.strip()
    return out


def test_all_tier_markers_are_declared() -> None:
    declared = _declared_markers()
    missing = [m for m in TIER_MARKERS if m not in declared]
    assert not missing, f"tier markers not declared in pyproject markers: {missing}"


def test_new_tier_markers_state_a_budget() -> None:
    """The tiers T3.1 adds must name their time budget in the marker description."""
    declared = _declared_markers()
    for tier in ("smoke", "core", "full"):
        desc = declared.get(tier, "").lower()
        assert "budget" in desc or "min" in desc or "s " in desc or "second" in desc, (
            f"tier marker {tier!r} does not state a time budget: {desc!r}"
        )


def test_every_tier_is_documented_with_a_budget() -> None:
    if not TIERS_DOC.is_file():
        pytest.skip("docs/test-tiers.rst not found")
    doc = TIERS_DOC.read_text(encoding="utf-8").lower()
    for tier in TIER_MARKERS:
        assert tier in doc, f"tier {tier!r} is not documented in docs/test-tiers.rst"
    # The doc must talk about budgets/time, not just list names.
    assert "budget" in doc and ("min" in doc or "s" in doc), "tiers doc must state budgets"


def test_smoke_tier_exists_and_is_small() -> None:
    """A real smoke suite must exist, be marked, and stay small (it is the fast gate)."""
    assert SMOKE.is_file(), "expected a dedicated smoke_test.py for the smoke tier"
    src = SMOKE.read_text(encoding="utf-8")
    assert "pytest.mark.smoke" in src, "smoke_test.py must mark its tests as the smoke tier"
    n_tests = src.count("def test_")
    assert 1 <= n_tests <= 12, (
        f"smoke suite has {n_tests} tests; keep it small and fast (the smoke tier is the "
        f"<=30 s critical-path gate, not a correctness dumping ground)"
    )
    # Smoke must stay dependency-light: no torch / backend imports at module scope.
    for heavy in ("import torch", "pyspark", "mpi4py"):
        assert heavy not in src, f"smoke tier must stay dependency-light; found {heavy!r}"
