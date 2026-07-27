"""Focused release-contract probes for real-options and value-of-information evidence."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from mixle.analysis.real_options import (
    GaussianObservationModel,
    real_option_value,
    voi_estimate,
)


class _NPV:
    def __init__(self, mean: float) -> None:
        self.mean = mean


class _GaussianPosterior:
    def __init__(self, mean=(0.0,), cov=None, *, probe_skip: int = 0) -> None:
        self._mean = np.asarray(mean)
        self._cov = np.eye(len(mean)) if cov is None else np.asarray(cov)
        self._probe_skip = probe_skip

    @property
    def mean(self):
        return self._mean

    @property
    def cov(self):
        return self._cov

    def samples(self, n, rng):
        if n == 512 and self._probe_skip:
            rng.standard_normal(self._probe_skip)
        return rng.multivariate_normal(np.asarray(self._mean), np.eye(np.asarray(self._mean).size), size=n)


def _decision(samples):
    return max(float(np.mean(samples[:, 0])), 0.0)


def _heuristic(**values):
    return {"method": "variance_rescaling_heuristic", **values}


def test_real_option_terminal_boundaries_follow_each_exercise_contract():
    expansion = real_option_value(
        _NPV(100.0),
        volatility=0.2,
        horizon=2,
        n_steps=2,
        kind="expand",
        rate=0.05,
        expand_fraction=0.5,
        expansion_cost=40.0,
    )
    abandonment = real_option_value(
        _NPV(100.0),
        volatility=0.2,
        horizon=2,
        n_steps=2,
        kind="abandon",
        rate=0.05,
        salvage_value=50.0,
    )
    assert expansion.exercise_boundary[-1] == 80.0
    assert abandonment.exercise_boundary[-1] == 50.0


def test_running_base_project_is_not_reported_as_exercising_expansion():
    result = real_option_value(
        _NPV(100.0),
        volatility=0.1,
        horizon=1,
        n_steps=1,
        kind="expand",
        rate=0.5,
        expand_fraction=0.5,
        expansion_cost=1_000.0,
    )
    assert np.isnan(result.exercise_boundary[0])


@pytest.mark.parametrize("name,value", [("horizon", True), ("horizon", 2.0), ("n_steps", True), ("n_steps", 2.0)])
def test_real_option_counts_require_integral_non_boolean_scalars(name, value):
    kwargs = {"horizon": 2, "n_steps": 2}
    kwargs[name] = value
    with pytest.raises(ValueError, match=name):
        real_option_value(_NPV(10.0), volatility=0.2, kind="defer", rate=0.05, **kwargs)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"expand_fraction": -0.1}, "expand_fraction"),
        ({"expansion_cost": -1.0}, "expansion_cost"),
    ],
)
def test_expansion_controls_cannot_reverse_the_investment_contract(kwargs, match):
    with pytest.raises(ValueError, match=match):
        real_option_value(_NPV(10.0), volatility=0.2, horizon=2, kind="expand", rate=0.05, **kwargs)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"rate": -1e308}, "discount factor"),
        ({"volatility": 1e308}, "lattice step size"),
    ],
)
def test_real_option_rejects_unrepresentable_finite_arithmetic(kwargs, match):
    with pytest.raises(ValueError, match=match):
        real_option_value(_NPV(1e308), horizon=2, kind="defer", **{"volatility": 0.2, "rate": 0.05, **kwargs})


def test_real_option_rejects_unrepresentable_expansion_boundary():
    with pytest.raises(ValueError, match="terminal expansion boundary"):
        real_option_value(
            _NPV(10.0),
            volatility=0.2,
            horizon=2,
            kind="expand",
            rate=0.05,
            expand_fraction=1e-308,
            expansion_cost=1e308,
        )


@pytest.mark.parametrize("reduction", [-1.0, 1.0, 2.0, np.nan])
def test_heuristic_rejects_impossible_variance_reduction(reduction):
    with pytest.raises(ValueError, match="variance_reduction"):
        voi_estimate(
            _GaussianPosterior(),
            _decision,
            _heuristic(variance_reduction=reduction),
            rng=np.random.default_rng(0),
            n_outer=2,
            n_inner=2,
        )


def test_pde_hook_reduction_uses_the_same_validation(monkeypatch):
    package = types.ModuleType("mixle_pde")
    package.__path__ = []
    hook = types.ModuleType("mixle_pde.voi")
    hook.expected_variance_reduction = lambda *args, **kwargs: 2.0
    monkeypatch.setitem(sys.modules, "mixle_pde", package)
    monkeypatch.setitem(sys.modules, "mixle_pde.voi", hook)
    with pytest.raises(ValueError, match="variance_reduction"):
        voi_estimate(
            _GaussianPosterior(),
            _decision,
            _heuristic(candidate_geometry=object(), forward_op=object()),
            rng=np.random.default_rng(0),
            n_outer=2,
            n_inner=2,
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"n_outer_samples": True},
        {"n_inner_samples": True},
        {"n_outer_samples": 2.9},
        {"n_inner_samples": 2.9},
        {"n_outer_samples": 1},
    ],
)
def test_voi_counts_are_exact_and_outer_count_supports_uncertainty(overrides):
    with pytest.raises(ValueError, match="samples"):
        voi_estimate(
            _GaussianPosterior(),
            _decision,
            _heuristic(variance_reduction=0.5, **overrides),
            rng=np.random.default_rng(0),
            n_outer=2,
            n_inner=2,
        )


@pytest.mark.parametrize(
    "covariance,name",
    [
        (np.array([[2.0, 1.0], [0.0, 2.0]]), "posterior.cov"),
        (np.array([[1.0, 0.0], [0.0, -1.0]]), "posterior.cov"),
    ],
)
def test_gaussian_evsi_rejects_invalid_posterior_covariance(covariance, name):
    posterior = _GaussianPosterior(mean=(0.0, 0.0), cov=covariance)
    observation = GaussianObservationModel(np.eye(2), np.eye(2))
    with pytest.raises(ValueError, match=name):
        voi_estimate(
            posterior,
            _decision,
            {},
            rng=np.random.default_rng(0),
            n_outer=2,
            n_inner=2,
            observation_model=observation,
        )


@pytest.mark.parametrize(
    "obs_cov",
    [
        np.array([[2.0, 1.0], [0.0, 2.0]]),
        np.array([[1.0, 0.0], [0.0, 0.0]]),
    ],
)
def test_gaussian_evsi_rejects_invalid_observation_covariance(obs_cov):
    observation = GaussianObservationModel(np.eye(2), obs_cov)
    with pytest.raises(ValueError, match="observation_model.obs_cov"):
        voi_estimate(
            _GaussianPosterior(mean=(0.0, 0.0)),
            _decision,
            {},
            rng=np.random.default_rng(0),
            n_outer=2,
            n_inner=2,
            observation_model=observation,
        )


@pytest.mark.parametrize("mean", [np.array(np.nan), np.array([np.nan]), np.array([[0.0]])])
def test_gaussian_evsi_rejects_invalid_posterior_mean(mean):
    posterior = _GaussianPosterior()
    posterior._mean = mean
    with pytest.raises(ValueError, match="posterior.mean"):
        voi_estimate(
            posterior,
            _decision,
            {},
            rng=np.random.default_rng(0),
            n_outer=2,
            n_inner=2,
            observation_model=GaussianObservationModel(np.ones((1, 1)), np.ones((1, 1))),
        )


@pytest.mark.parametrize("method", ["heuristic", "gaussian"])
def test_voi_rejects_nonfinite_decision_values_with_side_and_replicate(method):
    kwargs = {}
    drill_info = _heuristic(variance_reduction=0.5)
    if method == "gaussian":
        kwargs["observation_model"] = GaussianObservationModel(np.ones((1, 1)), np.ones((1, 1)))
        drill_info = {}
    with pytest.raises(ValueError, match=r"replicate 0 \(with information\)|replicate 0 \(without information\)"):
        voi_estimate(
            _GaussianPosterior(),
            lambda _samples: np.nan,
            drill_info,
            rng=np.random.default_rng(0),
            n_outer=2,
            n_inner=2,
            **kwargs,
        )


def test_gaussian_diagnostic_does_not_advance_estimator_random_stream():
    observation = GaussianObservationModel(np.ones((1, 1)), np.ones((1, 1)))
    ordinary = voi_estimate(
        _GaussianPosterior(probe_skip=0),
        _decision,
        {},
        rng=np.random.default_rng(7),
        n_outer=4,
        n_inner=5,
        observation_model=observation,
    )
    different_probe = voi_estimate(
        _GaussianPosterior(probe_skip=37),
        _decision,
        {},
        rng=np.random.default_rng(7),
        n_outer=4,
        n_inner=5,
        observation_model=observation,
    )
    assert ordinary == different_probe
