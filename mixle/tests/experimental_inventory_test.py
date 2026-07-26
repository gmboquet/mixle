"""Completeness and evidence-boundary tests for the experimental inventory."""

from __future__ import annotations

from pathlib import Path

import pytest

from mixle.experimental import EXPERIMENTAL_INVENTORY

pytestmark = pytest.mark.experimental

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_EXPERIMENTAL_ROOT = _REPOSITORY_ROOT / "mixle" / "experimental"
_MATURITIES = {"prototype", "locally_receipted", "unvalidated", "bookkeeping_only"}


def test_every_top_level_experimental_surface_has_one_inventory_record() -> None:
    source_modules = {path.stem for path in _EXPERIMENTAL_ROOT.glob("*.py") if path.name != "__init__.py"} | {
        "typed_runtime"
    }
    recorded_modules = [surface.module for surface in EXPERIMENTAL_INVENTORY]
    assert len(recorded_modules) == len(set(recorded_modules))
    assert set(recorded_modules) == source_modules


def test_inventory_receipts_and_maturity_boundaries_are_well_formed() -> None:
    for surface in EXPERIMENTAL_INVENTORY:
        assert surface.maturity in _MATURITIES
        assert surface.purpose
        for test_path in surface.acceptance_tests:
            assert test_path.startswith("mixle/tests/")
            assert (_REPOSITORY_ROOT / test_path).is_file(), (surface.module, test_path)
        if surface.maturity == "locally_receipted":
            assert surface.acceptance_tests
            assert surface.evidence_id is not None
            assert surface.evidence_id.startswith("EVID-")
        else:
            assert surface.evidence_id is None


def test_disputed_claims_are_narrow_or_explicitly_unvalidated() -> None:
    inventory = {surface.module: surface for surface in EXPERIMENTAL_INVENTORY}
    assert inventory["typed_runtime"].maturity == "unvalidated"
    assert inventory["summary_tree"].maturity == "locally_receipted"
    assert "Test-scale" in inventory["summary_tree"].purpose
    assert "fixed-hypothesis" in inventory["pac_bayes"].purpose
    assert "Descriptive" in inventory["spectral_health"].purpose
    assert "diagnosis" in inventory["spectral_health"].limitations[0]
    assert "jointly supported" in inventory["wake_sleep"].purpose
