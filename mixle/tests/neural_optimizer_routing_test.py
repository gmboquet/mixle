"""Automatic neural updates are inspectable, non-Adam-first, and analytically bypassable."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mixle.inference.estimation import optimize
from mixle.models import GradLeaf
from mixle.models.optimizer_routing import build_auto_optimizer, plan_neural_optimizer


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
