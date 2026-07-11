"""P3 (experimental) -- conjugate-computation VI reproduces the exact conjugate update.

The card's first, self-contained claim: a single natural-gradient CVI step with unit step size is
exactly the conjugate (EM M-step) update for an exponential-family leaf. These tests verify that
for Normal-Normal, Beta-Bernoulli, and Gamma-Poisson, that streaming the data in any chunking is
order-independent (natural parameters are additive), that a damped step converges to the same
posterior, and that the flat-prior limit recovers the MLE M-step.
"""

from __future__ import annotations

import numpy as np
import pytest

from mixle.experimental.cvi import (
    conjugate_posterior,
    cvi_step,
    damped_to_convergence,
)


def _data(family, rng):
    if family == "normal_normal":
        return rng.normal(2.0, 1.0, 50), (0.0, 1.0)
    if family == "beta_bernoulli":
        return (rng.random(40) < 0.7).astype(float), (1.0, 1.0)
    return rng.poisson(3.0, 30).astype(float), (2.0, 1.0)


@pytest.mark.parametrize("family", ["normal_normal", "beta_bernoulli", "gamma_poisson"])
def test_cvi_step_reproduces_conjugate_update(family) -> None:
    rng = np.random.default_rng(0)
    data, prior = _data(family, rng)
    cvi = cvi_step(family, prior, data, rho=1.0)
    closed = conjugate_posterior(family, prior, data)
    assert np.allclose(cvi, closed), f"{family}: CVI {cvi} != conjugate {closed}"


@pytest.mark.parametrize("family", ["normal_normal", "beta_bernoulli", "gamma_poisson"])
def test_streaming_is_order_independent(family) -> None:
    """Folding CVI over any chunking of the data gives the identical posterior (additive naturals)."""
    rng = np.random.default_rng(1)
    data, prior = _data(family, rng)
    batch = cvi_step(family, prior, data, rho=1.0)

    # process in three uneven chunks, each updating the running posterior
    running = prior
    for chunk in np.array_split(data, 3):
        running = cvi_step(family, running, chunk, rho=1.0)
    assert np.allclose(running, batch), f"{family}: streamed {running} != batch {batch}"

    # a different order must also match
    shuffled = prior
    for chunk in np.array_split(rng.permutation(data), 5):
        shuffled = cvi_step(family, shuffled, chunk, rho=1.0)
    assert np.allclose(shuffled, batch), f"{family}: order dependence detected"


@pytest.mark.parametrize("family", ["normal_normal", "beta_bernoulli", "gamma_poisson"])
def test_damped_step_converges_to_posterior(family) -> None:
    rng = np.random.default_rng(2)
    data, prior = _data(family, rng)
    damped = damped_to_convergence(family, prior, data, rho=0.3, iters=300)
    closed = conjugate_posterior(family, prior, data)
    assert np.allclose(damped, closed, atol=1e-6), f"{family}: damped {damped} != {closed}"


def test_flat_prior_limit_recovers_the_mle_m_step() -> None:
    """With a vanishing prior precision, the Normal-Normal CVI posterior mean is the sample mean."""
    rng = np.random.default_rng(3)
    data = rng.normal(-1.5, 2.0, 200)
    mean, _ = cvi_step("normal_normal", (0.0, 1e-9), data, rho=1.0, obs_var=1.0)
    assert np.isclose(mean, data.mean(), atol=1e-3), f"CVI mean {mean} != MLE {data.mean()}"


def test_unknown_family_raises() -> None:
    with pytest.raises(ValueError, match="unknown conjugate family"):
        cvi_step("not_a_family", (1.0, 1.0), [1.0, 2.0])


def test_determinism() -> None:
    rng = np.random.default_rng(4)
    data, prior = _data("normal_normal", rng)
    assert cvi_step("normal_normal", prior, data) == cvi_step("normal_normal", prior, data)
