"""Parameter routing, measured fallback, curvature, and batch-semantics tests."""

import json

import numpy as np
import pytest

from mixle.experimental.typed_runtime import (
    BatchSemanticsReceipt,
    CurvatureCache,
    CurvatureKind,
    CurvatureSketch,
    GeometryRouterConfig,
    MergeLaw,
    ObjectiveKind,
    OptimizerEvidence,
    OptimizerFamily,
    ParameterRole,
    UpdateContract,
    UpdateKind,
    apply_optimizer_evidence,
    compile_update_graph,
    describe_parameters,
    kronecker_precondition,
    natural_gradient_direction,
    orthogonalized_matrix_direction,
    route_optimizer_geometry,
)
from mixle.stats import GaussianDistribution, GaussianEstimator

pytestmark = [pytest.mark.experimental, pytest.mark.fast]


class _Parameter:
    def __init__(self, shape, *, itemsize=4, requires_grad=True):
        self.shape = shape
        self.itemsize = itemsize
        self.requires_grad = requires_grad

    def numel(self):
        return int(np.prod(self.shape)) if self.shape else 1

    def element_size(self):
        return self.itemsize


class _Module:
    def __init__(self):
        self.rows = (
            ("blocks.0.mlp.weight", _Parameter((128, 128))),
            ("token_embedding.weight", _Parameter((50_000, 128))),
            ("blocks.0.norm.weight", _Parameter((128,))),
            ("blocks.0.mlp.bias", _Parameter((128,))),
            ("router.weight", _Parameter((4, 16))),
            ("adapter.lora_A", _Parameter((8, 128))),
            ("experts.0.weight", _Parameter((128, 128))),
            ("temperature", _Parameter(())),
        )

    def named_parameters(self):
        return iter(self.rows)


def _contract(*, curvature=CurvatureKind.UNAVAILABLE, update=UpdateKind.FIRST_ORDER):
    return UpdateContract(
        objective_kind=ObjectiveKind.MLE,
        update_kind=update,
        merge_law=MergeLaw.LOW_RANK,
        curvature_kind=curvature,
        exact=False,
    )


def test_parameter_roles_and_conservative_neural_routes():
    descriptors = describe_parameters(_Module())
    roles = {row.name: row.role for row in descriptors}
    assert roles["blocks.0.mlp.weight"] is ParameterRole.MATRIX
    assert roles["token_embedding.weight"] is ParameterRole.EMBEDDING
    assert roles["router.weight"] is ParameterRole.ROUTER
    assert roles["adapter.lora_A"] is ParameterRole.LOW_RANK_ADAPTER
    assert roles["experts.0.weight"] is ParameterRole.SPARSE_EXPERT
    assert roles["temperature"] is ParameterRole.SCALAR

    plan = route_optimizer_geometry(descriptors, _contract())
    assert plan.route("blocks.0.mlp.weight").family is OptimizerFamily.MUON
    assert plan.route("token_embedding.weight").family is OptimizerFamily.ADAMW
    assert plan.route("blocks.0.norm.weight").family is OptimizerFamily.ADAMW
    assert plan.route("blocks.0.mlp.bias").family is OptimizerFamily.ADAMW
    assert plan.route("adapter.lora_A").family is OptimizerFamily.LOW_RANK_ADAPTIVE
    assert plan.route("experts.0.weight").family is OptimizerFamily.MUON
    assert plan.route("experts.0.weight").separate_clock
    json.dumps(plan.as_dict(), allow_nan=False)


def test_exact_statistical_parameters_never_enter_neural_optimizer():
    exact = compile_update_graph(GaussianDistribution(0.0, 1.0), GaussianEstimator()).node("n0000").contract
    plan = route_optimizer_geometry(describe_parameters(_Module()), exact)
    assert {route.family for route in plan.routes} == {OptimizerFamily.EXACT}
    assert plan.optimizer_state_bytes == 0


def test_fisher_router_uses_natural_gradient_and_kronecker_memory_guard_falls_back():
    descriptors = describe_parameters(_Module())
    fisher = route_optimizer_geometry(descriptors, _contract(curvature=CurvatureKind.FISHER))
    assert fisher.route("router.weight").family is OptimizerFamily.NATURAL_GRADIENT

    tight = GeometryRouterConfig(max_state_to_parameter_ratio=1.5)
    kronecker = route_optimizer_geometry(descriptors, _contract(curvature=CurvatureKind.KRONECKER), tight)
    assert kronecker.route("blocks.0.mlp.weight").family is OptimizerFamily.ADAMW
    assert "memory ratio" in kronecker.route("blocks.0.mlp.weight").reason


def test_measured_time_to_target_forces_and_avoids_fallback():
    descriptor = describe_parameters(_Module())[0]
    plan = route_optimizer_geometry((descriptor,), _contract())
    baseline = OptimizerEvidence(
        descriptor.name,
        OptimizerFamily.ADAMW,
        "loss<=1",
        True,
        9.0,
        10_000,
        100,
        1_000,
    )
    slow = OptimizerEvidence(
        descriptor.name,
        OptimizerFamily.MUON,
        "loss<=1",
        True,
        10.0,
        10_000,
        100,
        500,
    )
    fast = OptimizerEvidence(
        descriptor.name,
        OptimizerFamily.MUON,
        "loss<=1",
        True,
        7.0,
        10_000,
        100,
        500,
    )
    assert apply_optimizer_evidence(plan, (baseline, slow)).routes[0].family is OptimizerFamily.ADAMW
    assert apply_optimizer_evidence(plan, (baseline, fast)).routes[0].family is OptimizerFamily.MUON


def test_batch_receipt_makes_effective_batch_and_update_clock_explicit():
    receipt = BatchSemanticsReceipt(8, 4_096, 6.5, 4, 16, 100, "mean", 1_024.0, 100)
    assert receipt.effective_global_examples == 512
    assert receipt.effective_global_tokens == 262_144
    assert receipt.effective_responsibility_mass == pytest.approx(416.0)
    assert receipt.as_dict()["optimizer_updates"] == 100


def test_reference_geometry_transforms_are_finite_and_shape_checked():
    rng = np.random.default_rng(4)
    gradient = rng.normal(size=(8, 4))
    direction = orthogonalized_matrix_direction(gradient)
    assert direction.shape == gradient.shape
    np.testing.assert_allclose(direction.T @ direction, 2.0 * np.eye(4), rtol=1.0e-10, atol=1.0e-10)

    preconditioned = kronecker_precondition(gradient, np.eye(8), np.eye(4), damping=1.0e-6)
    np.testing.assert_allclose(preconditioned, gradient / np.sqrt(1.0 + 1.0e-6), rtol=1.0e-10)

    natural = natural_gradient_direction(np.array([2.0, 6.0]), np.diag([2.0, 3.0]), damping=1.0e-12)
    np.testing.assert_allclose(natural, [1.0, 2.0], rtol=1.0e-10)


def test_curvature_cache_enforces_version_lag():
    cache = CurvatureCache(max_version_lag=1)
    sketch = CurvatureSketch("experts:shared", CurvatureKind.KRONECKER, (np.eye(2),), 3, 100.0)
    cache.put(sketch)
    assert cache.get("experts:shared", model_version=4) is sketch
    assert cache.get("experts:shared", model_version=5) is None
    with pytest.raises(ValueError, match="future"):
        cache.get("experts:shared", model_version=2)
