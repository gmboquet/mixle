"""Exact posterior sampling for conjugate ``mixle.stats`` leaves.

The sampler uses each supported distribution's closed-form conjugate update to
draw independent parameter samples, providing an analytic alternative to generic
MCMC for Gaussian, count, and Bernoulli-family leaves.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np

from mixle.inference.priors import ImproperPriorReceipt
from mixle.utils.immutable import detach_receipt_container

from .parameter_bridge import _encode_data
from .samplers import MCMCResult


@dataclass(frozen=True)
class ConjugateUpdateReceipt:
    """Prior provenance and propriety status for an analytic posterior update."""

    prior_source: str
    prior_status: str
    posterior_status: str
    improper_prior: dict[str, Any] | None = None
    input_mutated: bool = False

    def __post_init__(self) -> None:
        # detach, not freeze: this receipt travels through paths that copy and pickle it
        # (MXR-080-1876).
        object.__setattr__(self, "improper_prior", detach_receipt_container(self.improper_prior))

    def as_dict(self) -> dict[str, Any]:
        """Return a plain durable receipt."""
        return {
            "prior_source": self.prior_source,
            "prior_status": self.prior_status,
            "posterior_status": self.posterior_status,
            "improper_prior": self.improper_prior,
            "input_mutated": self.input_mutated,
        }


class ImproperPosteriorError(ValueError):
    """Raised when an explicitly improper prior does not yield a proper posterior."""

    def __init__(self, receipt: ConjugateUpdateReceipt) -> None:
        super().__init__("the conjugate update did not produce a proper posterior.")
        self.receipt = receipt


def sample_conjugate_posterior(
    dist: Any,
    data: Any,
    draws: int = 1000,
    seed: int | None = None,
    return_distributions: bool = False,
    improper_receipt: ImproperPriorReceipt | None = None,
) -> MCMCResult:
    """Draw exact posterior parameter samples for a conjugate ``mixle.stats`` leaf.

    For ``mixle.stats`` distributions carrying a closed-form conjugate prior, the
    posterior over parameters is available analytically.  This runs the
    distribution's own conjugate estimator over ``data`` to obtain the posterior
    hyperparameters (read back via the fitted model's ``get_prior()``), then
    draws iid parameter samples from that posterior.  This is an exact
    alternative to :func:`sample_parameter_posterior`.

    Supported leaves: Gaussian (NormalGamma posterior, samples ``(mu, sigma2)``),
    Poisson (Gamma posterior, samples ``lam``), Exponential (Gamma posterior over
    the rate, samples the scale ``beta``), and Bernoulli, Binomial, and
    Geometric (Beta posterior, samples ``p``).  Binomial draws keep the prototype
    trial count and support shift fixed.

    Args:
        dist: A ``mixle.stats`` distribution; if it carries no conjugate prior, a weak proper
            family default is attached to an internal copy.
        data: Observations for the family.
        draws: Number of iid posterior samples.
        seed: Seed for the RandomState.
        return_distributions: Return rebuilt distributions instead of parameters.
        improper_receipt: Explicit acknowledgement for a supported zero-hyperparameter limiting
            prior. Without this receipt, improper priors are rejected. A still-improper posterior
            cannot be sampled and raises :class:`ImproperPosteriorError` carrying its status receipt.

    Returns:
        MCMCResult with iid samples (all accepted, no autocorrelation).
    """
    if isinstance(draws, (bool, np.bool_)) or not isinstance(draws, (int, np.integer)) or int(draws) < 0:
        raise ValueError("draws must be a non-negative integer.")
    if improper_receipt is not None and not isinstance(improper_receipt, ImproperPriorReceipt):
        raise TypeError("improper_receipt must be an ImproperPriorReceipt.")
    draws = int(draws)
    rng = np.random.RandomState(seed)
    cls_name = type(dist).__name__

    supported = (
        "GaussianDistribution",
        "PoissonDistribution",
        "ExponentialDistribution",
        "BernoulliDistribution",
        "BinomialDistribution",
        "GeometricDistribution",
    )
    if cls_name not in supported:
        raise NotImplementedError(
            "sample_conjugate_posterior supports Gaussian, Poisson, Exponential, Bernoulli, "
            "Binomial, and Geometric leaves; got %s." % cls_name
        )

    # Never mutate the caller's distribution while attaching a default or running an estimator.
    working_dist = deepcopy(dist)
    prior = working_dist.get_prior()
    prior_source = "provided"
    if prior is None:
        default_prior = _default_conjugate_prior(cls_name)
        if default_prior is not None:
            working_dist.set_prior(default_prior)
            prior = default_prior
            prior_source = "weak_proper_default"

    prior_status = _conjugate_prior_status(cls_name, prior)
    if prior_status == "improper" and improper_receipt is None:
        raise ValueError("an improper conjugate prior requires an explicit ImproperPriorReceipt.")
    if prior_status == "proper" and improper_receipt is not None:
        raise ValueError("improper_receipt was supplied, but the conjugate prior is proper.")
    if prior_status == "improper" and len(data) == 0:
        raise ImproperPosteriorError(
            ConjugateUpdateReceipt(
                prior_source=prior_source,
                prior_status=prior_status,
                posterior_status="invalid_or_improper",
                improper_prior=improper_receipt.as_dict(),
            )
        )

    # run the family's conjugate posterior update via accumulate + estimate; the
    # fitted model carries the conjugate posterior as its prior.
    try:
        posterior_dist = _conjugate_posterior(working_dist, data)
    except ValueError as exc:
        if prior_status != "improper":
            raise
        receipt = ConjugateUpdateReceipt(
            prior_source=prior_source,
            prior_status=prior_status,
            posterior_status="invalid_or_improper",
            improper_prior=improper_receipt.as_dict(),
        )
        raise ImproperPosteriorError(receipt) from exc
    posterior = posterior_dist.get_prior()
    posterior_status = _conjugate_prior_status(cls_name, posterior)
    receipt = ConjugateUpdateReceipt(
        prior_source=prior_source,
        prior_status=prior_status,
        posterior_status=posterior_status,
        improper_prior=None if improper_receipt is None else improper_receipt.as_dict(),
    )
    if posterior_status != "proper":
        raise ImproperPosteriorError(receipt)

    samples: list[Any] = []
    if cls_name == "GaussianDistribution":
        from mixle.stats.bayes.normal_gamma import NormalGammaDistribution

        if not isinstance(posterior, NormalGammaDistribution):
            raise NotImplementedError("sample_conjugate_posterior(Gaussian) requires a NormalGamma posterior.")
        mu0, lam, a, b = posterior.get_parameters()
        for mu, tau in posterior.sampler(seed=rng.randint(0, 2**31 - 1)).sample(size=draws):
            tau = max(float(tau), 1.0e-300)
            sigma2 = 1.0 / tau
            samples.append(
                type(dist)(float(mu), sigma2, name=dist.name, keys=dist.keys)
                if return_distributions
                else (float(mu), float(sigma2))
            )
    elif cls_name == "PoissonDistribution":
        from mixle.stats.univariate.continuous.gamma import GammaDistribution

        if not isinstance(posterior, GammaDistribution):
            raise NotImplementedError("sample_conjugate_posterior(Poisson) requires a Gamma posterior.")
        k, theta = posterior.get_parameters()
        for _ in range(draws):
            lam = rng.gamma(shape=k, scale=theta)
            samples.append(type(dist)(lam, name=dist.name, keys=dist.keys) if return_distributions else float(lam))
    elif cls_name == "ExponentialDistribution":
        from mixle.stats.univariate.continuous.gamma import GammaDistribution

        if not isinstance(posterior, GammaDistribution):
            raise NotImplementedError("sample_conjugate_posterior(Exponential) requires a Gamma posterior.")
        k, theta = posterior.get_parameters()
        for _ in range(draws):
            rate = max(float(rng.gamma(shape=k, scale=theta)), 1.0e-300)
            beta = 1.0 / rate
            samples.append(type(dist)(beta, name=dist.name, keys=dist.keys) if return_distributions else float(beta))
    else:
        from mixle.stats.univariate.continuous.beta import BetaDistribution

        if not isinstance(posterior, BetaDistribution):
            raise NotImplementedError("sample_conjugate_posterior(%s) requires a Beta posterior." % cls_name)
        a, b = posterior.get_parameters()
        for _ in range(draws):
            p = rng.beta(a, b)
            if return_distributions and cls_name == "BinomialDistribution":
                samples.append(type(dist)(p, dist.n, min_val=dist.min_val, name=dist.name, keys=dist.keys))
            elif return_distributions:
                samples.append(type(dist)(p, name=dist.name, keys=dist.keys))
            else:
                samples.append(float(p))

    return MCMCResult(
        samples=samples,
        log_probs=np.zeros(len(samples), dtype=float),
        accepted=np.ones(len(samples), dtype=bool),
        receipt=receipt,
    )


def _default_conjugate_prior(cls_name: str) -> Any:
    """Weak proper conjugate prior for a ``mixle.stats`` leaf family.

    The ``mixle.stats`` leaves default to ``prior=None``; the closed-form
    posterior update needs an explicit prior. Defaults remain proper so the
    result is an ordinary probability distribution, not an undisclosed limit.
    """
    if cls_name == "GaussianDistribution":
        from mixle.stats.bayes.normal_gamma import NormalGammaDistribution

        return NormalGammaDistribution(0.0, 1.0e-8, 0.500001, 1.0)
    if cls_name == "PoissonDistribution":
        from mixle.stats.univariate.continuous.gamma import GammaDistribution

        return GammaDistribution(1.0001, 1.0e6)
    if cls_name == "ExponentialDistribution":
        from mixle.stats.univariate.continuous.gamma import GammaDistribution

        return GammaDistribution(1.0001, 1.0e6)
    if cls_name in ("BernoulliDistribution", "BinomialDistribution", "GeometricDistribution"):
        from mixle.stats.univariate.continuous.beta import BetaDistribution

        return BetaDistribution(1.000001, 1.000001)
    return None


def _conjugate_prior_status(cls_name: str, prior: Any) -> str:
    """Validate a supported prior and return ``'proper'`` or ``'improper'``.

    Zero hyperparameters are the only supported improper limits. Negative or non-finite values are
    invalid under every policy.
    """
    if cls_name == "GaussianDistribution":
        from mixle.stats.bayes.normal_gamma import NormalGammaDistribution

        if not isinstance(prior, NormalGammaDistribution):
            raise NotImplementedError("Gaussian conjugate sampling requires a NormalGamma prior.")
        mu, *hyper = prior.get_parameters()
        if not np.isfinite(mu):
            raise ValueError("NormalGamma prior mean must be finite.")
    elif cls_name in ("PoissonDistribution", "ExponentialDistribution"):
        from mixle.stats.univariate.continuous.gamma import GammaDistribution

        if not isinstance(prior, GammaDistribution):
            raise NotImplementedError(f"{cls_name} conjugate sampling requires a Gamma prior.")
        hyper = list(prior.get_parameters())
    else:
        from mixle.stats.univariate.continuous.beta import BetaDistribution

        if not isinstance(prior, BetaDistribution):
            raise NotImplementedError(f"{cls_name} conjugate sampling requires a Beta prior.")
        hyper = list(prior.get_parameters())
    values = np.asarray(hyper, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("conjugate prior hyperparameters must be finite and non-negative.")
    return "proper" if np.all(values > 0.0) else "improper"


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
