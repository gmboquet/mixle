"""E0 scaffold: mixle.experimental imports cleanly and its graduation bookkeeping round-trips."""

import pytest

import mixle.experimental as experimental
from mixle.experimental.graduation import REGISTRY, ExperimentalMechanism


def test_experimental_package_imports():
    assert experimental.__doc__


def test_ungraduated_mechanism_with_no_receipts_is_ineligible():
    mechanism = ExperimentalMechanism(name="scaffold_test_no_receipts")

    assert mechanism.graduated is False
    assert mechanism.baseline_receipt is None
    assert mechanism.misfit_receipt is None
    assert mechanism.is_eligible() is False


def test_mechanism_with_only_one_receipt_is_ineligible():
    baseline_only = ExperimentalMechanism(
        name="scaffold_test_baseline_only",
        baseline_receipt={"metric": "bpb", "mechanism": 1.02, "baseline": 1.05, "flops": 3.1e20},
    )
    misfit_only = ExperimentalMechanism(
        name="scaffold_test_misfit_only",
        misfit_receipt={"truncation_error": 0.01},
    )

    assert baseline_only.is_eligible() is False
    assert misfit_only.is_eligible() is False


def test_mechanism_with_both_receipts_is_eligible():
    mechanism = ExperimentalMechanism(
        name="scaffold_test_both_receipts",
        baseline_receipt={"metric": "bpb", "mechanism": 1.02, "baseline": 1.05, "flops": 3.1e20},
        misfit_receipt={"truncation_error": 0.01},
    )

    assert mechanism.is_eligible() is True
    # Eligibility is separate from having actually graduated.
    assert mechanism.graduated is False


def test_registry_round_trips_registered_mechanisms():
    mechanism = ExperimentalMechanism(name="scaffold_test_registry_roundtrip")

    registered = REGISTRY.register(mechanism)

    assert registered is mechanism
    assert REGISTRY.get("scaffold_test_registry_roundtrip") is mechanism
    assert mechanism in list(REGISTRY)
    assert len(REGISTRY) >= 1


@pytest.mark.experimental
def test_experimental_marker_is_registered_and_collectible():
    """A trivial test tagged `experimental` -- confirms the marker is registered, not just documented."""
    assert True
