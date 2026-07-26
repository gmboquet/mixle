"""Finite planning-cost domains and complete public contract schema."""

import pytest

import mixle.experimental.typed_runtime as typed_runtime
from mixle.experimental.typed_runtime import (
    ComputeBand,
    ConvergenceCertificate,
    CostEstimate,
    weakest_band,
    weakest_certificate,
)

pytestmark = [pytest.mark.experimental, pytest.mark.fast]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), -1.0])
def test_cost_estimate_rejects_non_finite_or_negative_compute(value):
    with pytest.raises(ValueError, match="compute_units"):
        CostEstimate(compute_units=value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), -1.0])
def test_cost_estimate_rejects_non_finite_or_negative_wall_time(value):
    with pytest.raises(ValueError, match="wall_time_seconds"):
        CostEstimate(wall_time_seconds=value)


@pytest.mark.parametrize("field", ["communication_bytes", "peak_memory_bytes", "sample_count"])
def test_cost_estimate_requires_integer_counts(field):
    with pytest.raises(TypeError, match=field):
        CostEstimate(**{field: 1.5})
    with pytest.raises(TypeError, match=field):
        CostEstimate(**{field: True})


def test_cost_source_is_non_empty():
    with pytest.raises(ValueError, match="source"):
        CostEstimate(source="")


def test_complete_contract_schema_is_public_and_composable():
    expected = {
        "ComputeBand",
        "ConvergenceCertificate",
        "weakest_band",
        "weakest_certificate",
    }
    assert expected <= set(typed_runtime.__all__)
    assert all(hasattr(typed_runtime, name) for name in expected)
    assert weakest_band((ComputeBand.FLOAT32_ELIGIBLE, ComputeBand.FLOAT64)) is ComputeBand.FLOAT64
    assert (
        weakest_certificate(
            (
                ConvergenceCertificate.MONOTONE_CERTIFIED,
                ConvergenceCertificate.UNKNOWN,
            )
        )
        is ConvergenceCertificate.UNKNOWN
    )
