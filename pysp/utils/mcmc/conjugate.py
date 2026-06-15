"""Exact conjugate posterior sampling for closed-form ``pysp.stats`` leaves."""

from __future__ import annotations

from typing import Any

import numpy as np

from .parameter_bridge import _encode_data
from .samplers import MCMCResult


def sample_conjugate_posterior(
    dist: Any, data: Any, draws: int = 1000, seed: int | None = None, return_distributions: bool = False
) -> MCMCResult:
    """Draw exact posterior parameter samples for a conjugate ``pysp.stats`` leaf.

    For ``pysp.stats`` distributions carrying a closed-form conjugate prior, the
    posterior over parameters is available analytically.  This runs the
    distribution's own conjugate estimator over ``data`` to obtain the posterior
    hyperparameters (read back via the fitted model's ``get_prior()``), then
    draws iid parameter samples from that posterior.  This is an exact
    alternative to :func:`sample_parameter_posterior`.

    Supported leaves: Gaussian (NormalGamma posterior, samples ``(mu, sigma2)``),
    Poisson (Gamma posterior, samples ``lam``), and Bernoulli (Beta posterior,
    samples ``p``).

    Args:
        dist: A ``pysp.stats`` distribution; if it carries no conjugate prior a
            non-informative default for the family is attached automatically.
        data: Observations for the family.
        draws: Number of iid posterior samples.
        seed: Seed for the RandomState.
        return_distributions: Return rebuilt distributions instead of parameters.

    Returns:
        MCMCResult with iid samples (all accepted, no autocorrelation).
    """
    if draws < 0:
        raise ValueError("draws must be non-negative.")
    rng = np.random.RandomState(seed)
    cls_name = type(dist).__name__

    if cls_name not in ("GaussianDistribution", "PoissonDistribution", "BernoulliDistribution"):
        raise NotImplementedError(
            "sample_conjugate_posterior supports Gaussian, Poisson, and Bernoulli leaves; got %s." % cls_name
        )

    # the stats leaves default to prior=None; supply a non-informative conjugate
    # prior so the closed-form update has something to update against.
    if dist.get_prior() is None:
        default_prior = _default_conjugate_prior(cls_name)
        if default_prior is not None:
            dist.set_prior(default_prior)

    # run the family's conjugate posterior update via accumulate + estimate; the
    # fitted model carries the conjugate posterior as its prior.
    posterior_dist = _conjugate_posterior(dist, data)
    posterior = posterior_dist.get_prior()

    samples: list[Any] = []
    if cls_name == "GaussianDistribution":
        from pysp.stats.bayes.normgamma import NormalGammaDistribution

        if not isinstance(posterior, NormalGammaDistribution):
            raise NotImplementedError("sample_conjugate_posterior(Gaussian) requires a NormalGamma posterior.")
        mu0, lam, a, b = posterior.get_parameters()
        for mu, tau in posterior.sampler(seed=rng.randint(0, 2**31 - 1)).sample(size=draws):
            tau = max(float(tau), 1.0e-300)
            sigma2 = 1.0 / tau
            samples.append(type(dist)(float(mu), sigma2) if return_distributions else (float(mu), float(sigma2)))
    elif cls_name == "PoissonDistribution":
        from pysp.stats.leaf.gamma import GammaDistribution

        if not isinstance(posterior, GammaDistribution):
            raise NotImplementedError("sample_conjugate_posterior(Poisson) requires a Gamma posterior.")
        k, theta = posterior.get_parameters()
        for _ in range(draws):
            lam = rng.gamma(shape=k, scale=theta)
            samples.append(type(dist)(lam) if return_distributions else float(lam))
    else:  # BernoulliDistribution
        from pysp.stats.leaf.beta import BetaDistribution

        if not isinstance(posterior, BetaDistribution):
            raise NotImplementedError("sample_conjugate_posterior(Bernoulli) requires a Beta posterior.")
        a, b = posterior.get_parameters()
        for _ in range(draws):
            p = rng.beta(a, b)
            samples.append(type(dist)(p) if return_distributions else float(p))

    return MCMCResult(
        samples=samples, log_probs=np.zeros(len(samples), dtype=float), accepted=np.ones(len(samples), dtype=bool)
    )


def _default_conjugate_prior(cls_name: str) -> Any:
    """Non-informative conjugate prior for a ``pysp.stats`` leaf family.

    The ``pysp.stats`` leaves default to ``prior=None``; the closed-form
    posterior update needs an explicit prior, so attach a near-improper member of
    the conjugate family.
    """
    if cls_name == "GaussianDistribution":
        from pysp.stats.bayes.normgamma import NormalGammaDistribution

        return NormalGammaDistribution(0.0, 1.0e-8, 0.500001, 1.0)
    if cls_name == "PoissonDistribution":
        from pysp.stats.leaf.gamma import GammaDistribution

        return GammaDistribution(1.0001, 1.0e6)
    if cls_name == "BernoulliDistribution":
        from pysp.stats.leaf.beta import BetaDistribution

        return BetaDistribution(1.000001, 1.000001)
    return None


def _conjugate_posterior(dist: Any, data: Any) -> Any:
    """Run a family's conjugate update over ``data`` and return the estimated
    distribution (which carries the conjugate posterior as its prior)."""
    estimator = dist.estimator()
    factory = estimator.accumulator_factory()
    acc = factory.make()
    encoded, _ = _encode_data(dist, data)
    weights = np.ones(len(data), dtype=float)
    acc.seq_update(encoded, weights, None)
    return estimator.estimate(float(len(data)), acc.value())
