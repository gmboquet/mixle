"""Finite planning-cost domains and complete public contract schema."""

import importlib
import pkgutil

import pytest

import mixle.experimental.typed_runtime as typed_runtime
from mixle.experimental.typed_runtime import (
    ComputeBand,
    ContractEvidenceKind,
    ConvergenceCertificate,
    CostEstimate,
    MergeLaw,
    ObjectiveKind,
    RuntimeCapabilityStatus,
    UpdateContract,
    UpdateGraph,
    UpdateGraphValidationError,
    UpdateKind,
    UpdateNode,
    runtime_capabilities,
    validate_update_graph,
    weakest_band,
    weakest_certificate,
)

pytestmark = [pytest.mark.experimental]


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


def test_unverified_contract_cannot_supply_positive_execution_assurance():
    contract = UpdateContract(
        objective_kind=ObjectiveKind.MLE,
        update_kind=UpdateKind.EXACT_CLOSED_FORM,
        merge_law=MergeLaw.ADDITIVE,
    )
    graph = UpdateGraph(
        (
            UpdateNode(
                node_id="n0",
                path="root",
                model_type="Fixture",
                estimator_type="FixtureEstimator",
                contract=contract,
                cost=CostEstimate(),
                parameter_count=1,
            ),
        ),
        (),
        "n0",
    )
    with pytest.raises(UpdateGraphValidationError) as captured:
        validate_update_graph(graph, strict=True)
    assert {issue.code for issue in captured.value.issues} == {
        "unsupported-assurance",
        "unverified-contract",
    }


def test_runtime_capability_inventory_states_boundaries_explicitly():
    capabilities = {capability.name: capability for capability in runtime_capabilities()}
    assert capabilities["semantic_update_compiler"].status is RuntimeCapabilityStatus.IMPLEMENTED
    assert capabilities["local_mixture_execution"].status is RuntimeCapabilityStatus.NARROW
    assert capabilities["multi_host_transport"].status is RuntimeCapabilityStatus.UNAVAILABLE
    assert capabilities["general_estimator_executor"].status is RuntimeCapabilityStatus.UNAVAILABLE
    assert all(capability.boundary and capability.evidence for capability in capabilities.values())


def test_package_reexports_every_declared_module_api_without_duplicates():
    declared: set[str] = set()
    for module_info in pkgutil.iter_modules(typed_runtime.__path__, typed_runtime.__name__ + "."):
        module = importlib.import_module(module_info.name)
        declared.update(getattr(module, "__all__", ()))
    assert len(typed_runtime.__all__) == len(set(typed_runtime.__all__))
    assert declared <= set(typed_runtime.__all__)
    assert all(hasattr(typed_runtime, name) for name in declared)
    assert ContractEvidenceKind.UNVERIFIED.value == "unverified"
