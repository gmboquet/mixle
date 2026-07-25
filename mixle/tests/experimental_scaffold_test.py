"""E0 scaffold: mixle.experimental imports cleanly and its graduation bookkeeping round-trips."""

from dataclasses import replace

import pytest

import mixle.experimental as experimental
from mixle.experimental.graduation import (
    REGISTRY,
    BaselineComparisonReceipt,
    DuplicateMechanismError,
    ExperimentalMechanism,
    MisfitReceipt,
)

_OBSERVED_AT = "2026-07-25T00:00:00Z"


def _baseline(name: str, *, mechanism_value: float = 1.02) -> BaselineComparisonReceipt:
    return BaselineComparisonReceipt.create(
        mechanism_name=name,
        metric="bpb",
        mechanism_value=mechanism_value,
        baseline_value=1.05,
        matched_flops=3.1e20,
        lower_is_better=True,
        observed_at=_OBSERVED_AT,
        producer="experimental-scaffold-test",
        artifact_ref="memory://baseline-comparison",
    )


def _misfit(name: str, *, value: float = 0.01) -> MisfitReceipt:
    return MisfitReceipt.create(
        mechanism_name=name,
        metric="truncation_error",
        value=value,
        threshold=0.02,
        observed_at=_OBSERVED_AT,
        producer="experimental-scaffold-test",
        artifact_ref="memory://misfit-measurement",
    )


def test_experimental_package_imports():
    assert experimental.__doc__


def test_ungraduated_mechanism_with_no_receipts_is_ineligible():
    mechanism = ExperimentalMechanism(name="scaffold_test_no_receipts")

    assert mechanism.graduated is False
    assert mechanism.baseline_receipt is None
    assert mechanism.misfit_receipt is None
    assert mechanism.is_eligible() is False


def test_mechanism_with_only_one_receipt_is_ineligible():
    baseline_name = "scaffold_test_baseline_only"
    baseline_only = ExperimentalMechanism(
        name=baseline_name,
        baseline_receipt=_baseline(baseline_name),
    )
    misfit_name = "scaffold_test_misfit_only"
    misfit_only = ExperimentalMechanism(
        name=misfit_name,
        misfit_receipt=_misfit(misfit_name),
    )

    assert baseline_only.is_eligible() is False
    assert misfit_only.is_eligible() is False


def test_mechanism_with_both_receipts_is_eligible():
    name = "scaffold_test_both_receipts"
    mechanism = ExperimentalMechanism(
        name=name,
        baseline_receipt=_baseline(name),
        misfit_receipt=_misfit(name),
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
    assert REGISTRY.register(mechanism) is mechanism
    with pytest.raises(DuplicateMechanismError):
        REGISTRY.register(ExperimentalMechanism(name="scaffold_test_registry_roundtrip"))


def test_empty_or_failing_evidence_cannot_graduate():
    with pytest.raises(TypeError):
        ExperimentalMechanism(name="empty-evidence", baseline_receipt={})  # type: ignore[arg-type]

    name = "scaffold_test_failed_evidence"
    mechanism = ExperimentalMechanism(
        name=name,
        baseline_receipt=_baseline(name, mechanism_value=1.10),
        misfit_receipt=_misfit(name, value=0.03),
    )
    assert mechanism.is_eligible() is False


def test_receipt_integrity_and_mechanism_identity_are_verified():
    name = "scaffold_test_integrity"
    baseline = _baseline(name)
    assert baseline.verify()
    with pytest.raises(ValueError, match="digest"):
        replace(baseline, mechanism_value=0.5)

    wrong_identity = ExperimentalMechanism(
        name=name,
        baseline_receipt=_baseline("different-mechanism"),
        misfit_receipt=_misfit(name),
    )
    assert wrong_identity.is_eligible() is False


@pytest.mark.experimental
def test_experimental_marker_is_registered_and_collectible():
    """A trivial test tagged `experimental` -- confirms the marker is registered, not just documented."""
    assert True
