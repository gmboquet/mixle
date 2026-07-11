"""Executable PyTorch tests for routed AdamW, Muon, Kronecker, natural, and proximal updates."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mixle.experimental.typed_runtime import (  # noqa: E402
    CurvatureKind,
    GeometryRouterConfig,
    MergeLaw,
    ObjectiveKind,
    OptimizerFamily,
    UpdateContract,
    UpdateKind,
    build_routed_torch_optimizer,
    compile_update_graph,
    describe_parameters,
    route_optimizer_geometry,
)
from mixle.stats import GaussianDistribution, GaussianEstimator  # noqa: E402

pytestmark = [pytest.mark.experimental, pytest.mark.fast, pytest.mark.torch]


def _contract(*, curvature=CurvatureKind.UNAVAILABLE, update=UpdateKind.FIRST_ORDER):
    return UpdateContract(
        objective_kind=ObjectiveKind.MLE,
        update_kind=update,
        merge_law=MergeLaw.LOW_RANK,
        curvature_kind=curvature,
        exact=False,
    )


def test_routed_muon_matrix_and_adamw_bias_train_a_real_module():
    torch.manual_seed(4)
    module = torch.nn.Linear(4, 8)
    descriptors = describe_parameters(module)
    plan = route_optimizer_geometry(
        descriptors,
        _contract(),
        GeometryRouterConfig(matrix_min_elements=16, matrix_min_dimension=4),
    )
    assert plan.route("weight").family is OptimizerFamily.MUON
    assert plan.route("bias").family is OptimizerFamily.ADAMW
    optimizer = build_routed_torch_optimizer(module, plan, lr=0.03)
    inputs = torch.randn(32, 4)
    targets = torch.randn(32, 8)

    losses = []
    for _ in range(20):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(module(inputs), targets)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0]
    assert "momentum" in optimizer.state[module.weight]
    assert "exp_avg_sq" in optimizer.state[module.bias]


def test_newton_schulz_muon_tracks_exact_polar_reference_without_svd(monkeypatch):
    torch.manual_seed(12)
    module = torch.nn.Linear(32, 64, bias=False)
    plan = route_optimizer_geometry(
        describe_parameters(module),
        _contract(),
        GeometryRouterConfig(matrix_min_elements=16, matrix_min_dimension=4),
    )
    exact_optimizer = build_routed_torch_optimizer(module, plan, muon_backend="svd")
    momentum = torch.randn_like(module.weight)
    reference = exact_optimizer._polar_direction(momentum)

    monkeypatch.setattr(torch.linalg, "svd", lambda *args, **kwargs: pytest.fail("default Muon called SVD"))
    optimizer = build_routed_torch_optimizer(module, plan)
    approximate = optimizer._polar_direction(momentum)

    cosine = torch.nn.functional.cosine_similarity(approximate.flatten(), reference.flatten(), dim=0)
    assert float(cosine) > 0.99
    assert torch.isfinite(approximate).all()


def test_muon_kernel_configuration_is_validated():
    module = torch.nn.Linear(4, 4, bias=False)
    plan = route_optimizer_geometry(
        describe_parameters(module),
        _contract(),
        GeometryRouterConfig(matrix_min_elements=16, matrix_min_dimension=4),
    )
    with pytest.raises(ValueError, match="muon_backend"):
        build_routed_torch_optimizer(module, plan, muon_backend="unknown")
    with pytest.raises(ValueError, match="muon_steps"):
        build_routed_torch_optimizer(module, plan, muon_steps=0)


def test_kronecker_route_builds_and_uses_axis_factors():
    torch.manual_seed(2)
    module = torch.nn.Linear(4, 4, bias=False)
    plan = route_optimizer_geometry(
        describe_parameters(module),
        _contract(curvature=CurvatureKind.KRONECKER),
        GeometryRouterConfig(
            matrix_min_elements=16,
            matrix_min_dimension=4,
            max_state_to_parameter_ratio=10.0,
        ),
    )
    assert plan.route("weight").family is OptimizerFamily.KRONECKER
    optimizer = build_routed_torch_optimizer(module, plan, lr=0.01, precondition_frequency=1)
    loss = module(torch.randn(8, 4)).square().mean()
    loss.backward()
    optimizer.step()
    state = optimizer.state[module.weight]
    assert state["row_factor"].shape == (4, 4)
    assert state["column_factor"].shape == (4, 4)
    assert torch.isfinite(module.weight).all()


def test_natural_gradient_requires_and_uses_fisher_provider():
    class Router(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.router = torch.nn.Linear(2, 1, bias=False)

        def forward(self, value):
            return self.router(value)

    module = Router()
    plan = route_optimizer_geometry(
        describe_parameters(module),
        _contract(curvature=CurvatureKind.FISHER),
    )
    assert plan.route("router.weight").family is OptimizerFamily.NATURAL_GRADIENT
    with pytest.raises(ValueError, match="fisher_provider"):
        build_routed_torch_optimizer(module, plan)

    optimizer = build_routed_torch_optimizer(
        module,
        plan,
        lr=0.1,
        fisher_provider=lambda name, parameter: np.eye(parameter.numel()),
    )
    before = module.router.weight.detach().clone()
    module(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    assert not torch.equal(module.router.weight, before)


def test_proximal_route_applies_projection_after_gradient_step():
    module = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        module.weight.fill_(2.0)
    plan = route_optimizer_geometry(describe_parameters(module), _contract(update=UpdateKind.PROXIMAL))
    optimizer = build_routed_torch_optimizer(
        module,
        plan,
        lr=0.1,
        projection_provider=lambda name, parameter: parameter.clamp_(-1.0, 1.0),
    )
    module(torch.ones(1, 1)).sum().backward()
    optimizer.step()
    assert -1.0 <= float(module.weight.detach()) <= 1.0


def test_exact_plan_has_no_executable_neural_parameters():
    module = torch.nn.Linear(2, 2)
    exact = compile_update_graph(GaussianDistribution(0.0, 1.0), GaussianEstimator()).node("n0000").contract
    plan = route_optimizer_geometry(describe_parameters(module), exact)
    with pytest.raises(ValueError, match="no executable"):
        build_routed_torch_optimizer(module, plan)
