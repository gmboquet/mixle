"""Automatic neural updates are inspectable, non-Adam-first, and analytically bypassable."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mixle.inference.estimation import optimize
from mixle.models import GradLeaf
from mixle.models.optimizer_routing import (
    NeuralOptimizerPlan,
    RoutedNeuralOptimizer,
    build_auto_optimizer,
    build_routed_optimizer,
    plan_neural_optimizer,
)


class _RoutedModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb = torch.nn.Embedding(32, 16)
        self.body = torch.nn.Linear(64, 64)
        self.norm = torch.nn.LayerNorm(64)


class _MeanDensity(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mean = torch.nn.Parameter(torch.zeros(1))

    def log_density(self, x):
        return torch.distributions.Normal(self.mean, 1.0).log_prob(x).sum(-1)


class _AnalyticMeanDensity(_MeanDensity):
    def mixle_analytic_m_step(self, x, *, weights, batch_size):
        del batch_size
        with torch.no_grad():
            self.mean.copy_((weights[:, None] * x).sum(dim=0).to(self.mean.device))
        return {"solver": "weighted_mean"}


def test_auto_plan_routes_by_role_shape_and_sign_stability_without_adam():
    module = _RoutedModule()
    stable = {route.name: route.family for route in plan_neural_optimizer(module).routes}
    unstable = {route.name: route.family for route in plan_neural_optimizer(module, sign_stable=False).routes}

    assert stable["emb.weight"] == "adagrad"
    assert stable["body.weight"] == "muon"
    assert stable["body.bias"] == "rprop"
    assert stable["norm.weight"] == "rprop"
    assert unstable["body.bias"] == "adagrad"
    assert unstable["norm.weight"] == "adagrad"
    assert not any("adam" in family for family in (*stable.values(), *unstable.values()))


def test_homogeneous_full_batch_plan_uses_native_rprop():
    module = _MeanDensity()
    optimizer = build_auto_optimizer(module, lr=0.05)
    assert isinstance(optimizer, torch.optim.Rprop)
    assert optimizer.optimizer_plan.families == ("rprop",)


def test_grad_leaf_default_receipts_non_adam_auto_route():
    rng = np.random.default_rng(4)
    data = [float(value) for value in rng.normal(2.5, 1.0, 256)]
    fitted = optimize(data, GradLeaf(_MeanDensity(), m_steps=60, lr=0.05), max_its=1, out=None)

    assert fitted.fit_receipt["optimizer"] == "auto"
    assert fitted.fit_receipt["update_method"] == "autograd"
    assert fitted.fit_receipt["optimizer_plan"]["families"] == ["rprop"]
    assert float(fitted.module.mean.detach()) == pytest.approx(2.5, abs=0.25)


def test_registered_analytic_m_step_bypasses_autograd_and_optimizer():
    data = [-3.0, 1.0, 2.0, 4.0]
    fitted = optimize(data, GradLeaf(_AnalyticMeanDensity(), m_steps=100), max_its=1, out=None)

    assert float(fitted.module.mean.detach()) == pytest.approx(1.0)
    assert fitted.fit_receipt["update_method"] == "analytic_m_step"
    assert fitted.fit_receipt["optimizer"] == "none"
    assert fitted.fit_receipt["optimizer_steps"] == 0
    assert fitted.fit_receipt["analytic_receipt"] == {"solver": "weighted_mean"}


def test_adam_remains_an_explicit_last_resort():
    data = [1.0] * 32
    fitted = optimize(
        data,
        GradLeaf(_MeanDensity(), m_steps=2, optimizer="adam"),
        max_its=1,
        out=None,
    )
    assert fitted.fit_receipt["optimizer"] == "adam"
    assert fitted.fit_receipt["optimizer_plan"] is None


def test_routed_plan_requires_one_route_per_trainable_parameter():
    module = _RoutedModule()
    plan = plan_neural_optimizer(module)
    with pytest.raises(ValueError, match="more than once"):
        build_routed_optimizer(module, NeuralOptimizerPlan(plan.routes + (plan.routes[0],)))
    with pytest.raises(ValueError, match="every trainable parameter"):
        build_routed_optimizer(module, NeuralOptimizerPlan(plan.routes[:-1]))


@pytest.mark.parametrize(
    "controls",
    [
        {"lr": np.nan},
        {"lr": 0.0},
        {"eps": np.inf},
        {"weight_decay": np.nan},
        {"weight_decay": -1.0},
        {"betas": (np.nan, 0.9)},
        {"betas": (0.9, 1.0)},
        {"precondition_frequency": 1.5},
        {"precondition_frequency": 0},
        {"muon_steps": True},
    ],
)
def test_routed_optimizer_controls_must_be_exact_and_finite(controls):
    module = _RoutedModule()
    with pytest.raises((TypeError, ValueError)):
        build_routed_optimizer(module, plan_neural_optimizer(module), **controls)


def test_mixed_routed_optimizer_full_object_and_state_checkpoints_round_trip():
    module = _RoutedModule()
    optimizer = build_auto_optimizer(module, lr=0.01)
    assert isinstance(optimizer, RoutedNeuralOptimizer)

    for parameter in module.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    state = optimizer.state_dict()

    restored_object = pickle.loads(pickle.dumps(optimizer))
    assert isinstance(restored_object, RoutedNeuralOptimizer)
    assert restored_object.state_dict()["param_groups"] == state["param_groups"]

    clone = _RoutedModule()
    restored_state = build_auto_optimizer(clone, lr=0.01)
    restored_state.load_state_dict(state)
    assert restored_state.state_dict()["param_groups"] == state["param_groups"]
