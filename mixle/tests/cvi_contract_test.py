"""Support and damping contracts for conjugate-computation VI."""

from __future__ import annotations

import numpy as np
import pytest

from mixle.experimental.cvi import conjugate_posterior, cvi_step, damped_to_convergence


@pytest.mark.parametrize(
    "family, data",
    [
        ("normal_normal", [0.0, np.nan]),
        ("beta_bernoulli", [0.0, 0.5, 1.0]),
        ("beta_bernoulli", [-1.0, 1.0]),
        ("gamma_poisson", [-1.0, 2.0]),
        ("gamma_poisson", [1.5, 2.0]),
    ],
)
def test_cvi_rejects_observations_outside_the_family_support(family: str, data: list[float]) -> None:
    prior = (0.0, 1.0) if family == "normal_normal" else (1.0, 1.0)
    with pytest.raises(ValueError):
        cvi_step(family, prior, data)
    with pytest.raises(ValueError):
        conjugate_posterior(family, prior, data)


@pytest.mark.parametrize(
    "family, prior",
    [
        ("normal_normal", (0.0, 0.0)),
        ("normal_normal", (np.inf, 1.0)),
        ("beta_bernoulli", (0.0, 1.0)),
        ("beta_bernoulli", (1.0, -1.0)),
        ("gamma_poisson", (-1.0, 1.0)),
        ("gamma_poisson", (1.0, np.inf)),
    ],
)
def test_cvi_rejects_invalid_prior_parameters(family: str, prior: tuple[float, float]) -> None:
    with pytest.raises(ValueError):
        cvi_step(family, prior, [0.0])


@pytest.mark.parametrize("rho", [0.0, -0.1, 1.1, np.nan, np.inf])
def test_cvi_requires_a_finite_convex_damping_weight(rho: float) -> None:
    with pytest.raises(ValueError):
        cvi_step("normal_normal", (0.0, 1.0), [1.0], rho=rho)
    with pytest.raises(ValueError):
        damped_to_convergence("normal_normal", (0.0, 1.0), [1.0], rho=rho)


@pytest.mark.parametrize("obs_var", [0.0, -1.0, np.nan, np.inf])
def test_normal_updates_require_positive_finite_observation_variance(obs_var: float) -> None:
    with pytest.raises(ValueError):
        cvi_step("normal_normal", (0.0, 1.0), [1.0], obs_var=obs_var)
    with pytest.raises(ValueError):
        conjugate_posterior("normal_normal", (0.0, 1.0), [1.0], obs_var=obs_var)


def test_damped_updates_require_a_positive_iteration_count() -> None:
    with pytest.raises(ValueError):
        damped_to_convergence("beta_bernoulli", (1.0, 1.0), [0.0, 1.0], iters=0)
