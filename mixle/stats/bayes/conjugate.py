"""Conjugate-prior posterior inference, derived from the exponential-family map.

Every exponential-family likelihood ``p(x | eta) = h(x) exp(<eta, T(x)> - A(eta))`` has a
conjugate prior ``p(eta | chi, nu) propto exp(<eta, chi> - nu A(eta))``; after observing data the
posterior is simply

    chi' = chi + sum_i T(x_i),    nu' = nu + n.

This module realises that parameter posterior in closed form for every supported exponential-family
leaf: exact parameter samples, marginal likelihood (evidence), and posterior mean / point estimate.
Posterior predictives are exact when Mixle has the compound distribution; otherwise the returned
plug-in distribution is explicitly marked with machine-readable approximation metadata:

* fully (all parameters): Gaussian, multivariate & diagonal Gaussian, LogGaussian, Poisson,
  Exponential, Bernoulli/Binomial/Geometric, Categorical, Rayleigh, half-normal;
* conditional on the distribution's known nuisance parameter (shape / location / scale /
  number-of-trials / concentration, taken from the instance exactly as a Binomial's ``n`` is):
  Gamma, InverseGamma, InverseGaussian, Pareto, NegativeBinomial, von Mises.

Likelihoods with no tractable closed-form conjugate (full Beta, full Gamma, LogSeries, …) and
structured distributions (mixtures, HMMs, …) raise rather than return a partial answer.
:func:`mixture_conjugate_posterior` extends this to priors that are themselves mixtures of
conjugates (Diaconis-Ylvisaker). The public entry point is :func:`conjugate_posterior`.
"""

from __future__ import annotations

import math
import operator
from collections import Counter
from typing import Any

import numpy as np
from scipy.special import betaln, gammaln, logsumexp, multigammaln

__all__ = [
    "ConjugatePosterior",
    "ConjugatePosteriorSampler",
    "MixtureConjugatePosterior",
    "conjugate_posterior",
    "mixture_conjugate_posterior",
    "is_conjugate_family",
]


def is_conjugate_family(dist: Any) -> bool:
    """Return whether ``dist`` (instance or type) has a closed-form conjugate posterior.

    Single source of truth: membership in the ``conjugate_posterior`` builder registry. This is the
    family-level capability — "can this distribution be updated in closed form?" — distinct from the
    instance-level ``has_conj_prior`` flag (whether a conjugate prior is currently attached for the
    MAP path). Backs :class:`mixle.capability.ConjugateUpdatable` and
    :meth:`ProbabilityDistribution.has_conjugate_prior`.
    """
    cls = dist if isinstance(dist, type) else type(dist)
    return cls in _registry()


def _validate_sample_size(size: Any) -> int:
    if isinstance(size, (bool, np.bool_)):
        raise TypeError("sample size must be a non-negative integer, not bool.")
    try:
        result = operator.index(size)
    except TypeError as exc:
        raise TypeError("sample size must be a non-negative integer.") from exc
    if result < 0:
        raise ValueError("sample size must be non-negative.")
    return int(result)


def _as_weight_vector(weights: Any, count: int) -> np.ndarray:
    if weights is None:
        return np.ones(count, dtype=np.float64)
    try:
        result = np.asarray(weights, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("conjugate_posterior weights must be numeric.") from exc
    if result.shape != (count,):
        raise ValueError(
            "conjugate_posterior weights must have exact shape (%d,), got %r." % (count, result.shape)
        )
    if not np.all(np.isfinite(result)):
        raise ValueError("conjugate_posterior requires finite weights (no NaN/Inf).")
    if np.any(result < 0.0):
        raise ValueError("conjugate_posterior requires non-negative weights.")
    return result


def _as_weighted_array(data: Any, weights: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    try:
        x = np.asarray(list(data), dtype=np.float64) if not isinstance(data, np.ndarray) else data.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("conjugate_posterior observations must be numeric.") from exc
    if x.ndim != 1:
        raise ValueError(
            "univariate conjugate_posterior requires a one-dimensional observation vector, got shape %r."
            % (x.shape,)
        )
    if not np.all(np.isfinite(x)):
        raise ValueError("conjugate_posterior requires finite observations (no NaN/Inf).")
    return x, _as_weight_vector(weights, len(x))


def _finite_scalar(value: Any, name: str) -> float:
    if np.ndim(value) != 0:
        raise ValueError("%s must be a finite scalar." % name)
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("%s must be a finite scalar." % name) from exc
    if not math.isfinite(result):
        raise ValueError("%s must be a finite scalar." % name)
    return result


def _positive_scalar(value: Any, name: str) -> float:
    result = _finite_scalar(value, name)
    if result <= 0.0:
        raise ValueError("%s must be positive." % name)
    return result


def _nonnegative_scalar(value: Any, name: str) -> float:
    result = _finite_scalar(value, name)
    if result < 0.0:
        raise ValueError("%s must be non-negative." % name)
    return result


def _positive_vector(value: Any, name: str, *, length: int | None = None) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("%s must be a numeric vector." % name) from exc
    if result.ndim != 1 or (length is not None and result.shape != (length,)):
        expected = " of length %d" % length if length is not None else ""
        raise ValueError("%s must be a one-dimensional vector%s." % (name, expected))
    if result.size == 0 or np.any(~np.isfinite(result)) or np.any(result <= 0.0):
        raise ValueError("%s must contain finite positive values." % name)
    return result


def _probability_vector(value: Any, length: int, name: str, *, normalize: bool) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("%s must be a numeric vector." % name) from exc
    if result.shape != (length,):
        raise ValueError("%s must have exact shape (%d,), got %r." % (name, length, result.shape))
    if np.any(~np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError("%s must contain finite non-negative values." % name)
    total = float(result.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("%s must have positive finite total mass." % name)
    if normalize:
        return result / total
    if not np.isclose(total, 1.0, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError("%s must sum to 1.0." % name)
    return result


def _spd_matrix(value: Any, dimension: int, name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("%s must be a numeric matrix." % name) from exc
    if result.shape != (dimension, dimension):
        raise ValueError("%s must have exact shape (%d, %d)." % (name, dimension, dimension))
    if np.any(~np.isfinite(result)):
        raise ValueError("%s must contain only finite values." % name)
    if not np.allclose(result, result.T, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError("%s must be symmetric." % name)
    result = 0.5 * (result + result.T)
    try:
        np.linalg.cholesky(result)
    except np.linalg.LinAlgError as exc:
        raise ValueError("%s must be positive definite." % name) from exc
    return result


def _mark_approximate_predictive(distribution: Any, posterior_family: str, reason: str) -> Any:
    distribution.is_exact_posterior_predictive = False
    distribution.posterior_predictive_kind = "posterior_mean_plugin"
    distribution.approximation_receipt = {
        "method": "posterior_mean_plugin",
        "posterior_family": posterior_family,
        "reason": reason,
    }
    return distribution


def _mark_exact_predictive(distribution: Any, posterior_family: str, method: str) -> Any:
    distribution.is_exact_posterior_predictive = True
    distribution.posterior_predictive_kind = "exact"
    distribution.approximation_receipt = {
        "method": method,
        "posterior_family": posterior_family,
        "exact": True,
    }
    return distribution


class ConjugatePosterior:
    """Base class for a closed-form conjugate posterior over a likelihood's parameters.

    Subclasses provide the family-specific hyperparameters and the four capabilities: ``mean`` (the
    posterior mean of the parameters), ``sample`` (exact draws of the parameters), ``point_estimate``
    (a fitted distribution at the posterior mean), ``log_marginal_likelihood`` (the evidence of the
    observed data under the prior), and ``posterior_predictive`` (the distribution of a new draw).
    """

    family: str = "conjugate"
    log_base: float = 0.0  # sum_i log h(x_i): the base-measure term of the absolute marginal likelihood

    # -- to be provided by subclasses -------------------------------------
    def mean(self) -> dict[str, Any]:
        """Return posterior mean parameters as a family-specific dictionary."""
        raise NotImplementedError

    def sample(self, n: int = 1, rng: np.random.RandomState | None = None) -> dict[str, np.ndarray]:
        """Draw ``n`` exact parameter samples from the posterior."""
        raise NotImplementedError

    def sampler(self, seed: int | None = None) -> ConjugatePosteriorSampler:
        """Return a sampler exposing the standard ``obj.sampler(seed).sample(size)`` API.

        Mirrors the distribution sampling convention so conjugate posteriors read the same way as every
        other mixle object; here each draw is a *parameter set* from the posterior. ``size=None`` returns
        one parameter set (scalars), ``size=n`` a dict of length-``n`` arrays. The explicit-rng form
        ``sample(n, rng)`` remains available.
        """
        return ConjugatePosteriorSampler(self, seed)

    def point_estimate(self):
        """Return a likelihood distribution built from a representative posterior parameter value."""
        raise NotImplementedError

    def log_marginal_likelihood(self) -> float:
        """Return the closed-form log evidence of the observed data."""
        raise NotImplementedError

    def posterior_predictive(self):
        """Return the posterior predictive distribution for a new observation."""
        raise NotImplementedError

    def summary(self) -> dict[str, Any]:
        """Return a compact summary of the posterior family, mean, and hyperparameters."""
        return {"family": self.family, "mean": self.mean(), "hyper": self.hyper()}

    def hyper(self) -> dict[str, Any]:
        """Return family-specific posterior hyperparameters."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return "%s(%s)" % (type(self).__name__, self.hyper())


class ConjugatePosteriorSampler:
    """Standard ``.sample(size)`` adapter over a :class:`ConjugatePosterior` (draws parameter sets)."""

    def __init__(self, posterior: ConjugatePosterior, seed: int | None = None) -> None:
        self.posterior = posterior
        self.rng = np.random.RandomState(seed)

    def sample(self, size: int | None = None, *, batched: bool = True) -> dict[str, Any]:
        """Draw one parameter set when ``size`` is ``None`` or a batch otherwise."""
        draws = self.posterior.sample(n=1 if size is None else _validate_sample_size(size), rng=self.rng)
        if size is None:
            return {
                k: (v[0] if isinstance(v, np.ndarray) and v.shape and v.shape[0] == 1 else v) for k, v in draws.items()
            }
        return draws


# ---------------------------------------------------------------------------
# Beta posterior (Bernoulli / Binomial / Geometric likelihoods)
# ---------------------------------------------------------------------------
class BetaPosterior(ConjugatePosterior):
    """Beta posterior for Bernoulli, binomial, geometric, or negative-binomial probabilities."""

    family = "Beta"

    def __init__(self, a: float, b: float, kind: str = "bernoulli", n_trials: int = 1):
        self.a = _positive_scalar(a, "Beta posterior a")
        self.b = _positive_scalar(b, "Beta posterior b")
        self.kind = kind
        self.n_trials = int(n_trials)
        self._a0 = None  # prior hyperparameters, set by the builder for the evidence term
        self._b0 = None

    def mean(self) -> dict[str, Any]:
        """Return the posterior mean success probability."""
        return {"p": self.a / (self.a + self.b)}

    def variance(self) -> float:
        """Return the posterior variance of the success probability."""
        a, b = self.a, self.b
        return a * b / ((a + b) ** 2 * (a + b + 1.0))

    def sample(self, n: int = 1, rng: np.random.RandomState | None = None) -> dict[str, np.ndarray]:
        """Draw success-probability samples from the Beta posterior."""
        n = _validate_sample_size(n)
        rng = rng or np.random.RandomState()
        return {"p": rng.beta(self.a, self.b, size=n)}

    def point_estimate(self):
        """Return the likelihood distribution at the posterior mean probability."""
        p = self.a / (self.a + self.b)
        if self.kind == "geometric":
            from mixle.stats.univariate.discrete.geometric import GeometricDistribution

            return GeometricDistribution(p)
        if self.kind == "binomial":
            from mixle.stats.univariate.discrete.binomial import BinomialDistribution

            return BinomialDistribution(p, self.n_trials)
        if self.kind == "negative_binomial" and getattr(self, "_nb_r", None) is not None:
            from mixle.stats.univariate.discrete.negative_binomial import NegativeBinomialDistribution

            return NegativeBinomialDistribution(self._nb_r, p)
        from mixle.stats.univariate.discrete.bernoulli import BernoulliDistribution

        return BernoulliDistribution(p)

    def posterior_predictive(self):
        """Return the exact posterior predictive distribution for a new observation.

        For a single Bernoulli trial (``n_trials == 1``, including every Bernoulli-kind posterior,
        where ``n_trials`` is never set) the Bernoulli mass is linear in ``p``, so the plug-in
        distribution at the posterior mean *is* already the exact predictive (Beta-Bernoulli
        predictive prob == mean). For a Binomial with ``n_trials > 1`` the mass is nonlinear in
        ``p``, so plugging in the posterior mean is only an approximation -- the exact predictive,
        integrating ``p`` out of the Beta posterior, is the closed-form Beta-Binomial(n, a, b).
        """
        if self.kind == "binomial" and self.n_trials > 1:
            from mixle.stats.univariate.discrete.beta_binomial import BetaBinomialDistribution

            return _mark_exact_predictive(
                BetaBinomialDistribution(self.n_trials, self.a, self.b),
                self.family,
                "Beta-Binomial compound law",
            )
        if self.kind in {"geometric", "negative_binomial"}:
            return _mark_approximate_predictive(
                self.point_estimate(),
                self.family,
                "the exact compound predictive is not implemented for %s" % self.kind,
            )
        return _mark_exact_predictive(
            self.point_estimate(),
            self.family,
            "single-observation marginal equals the posterior-mean probability",
        )

    def hyper(self) -> dict[str, Any]:
        """Return the Beta shape hyperparameters."""
        return {"a": self.a, "b": self.b}

    def _set_prior(self, a0: float, b0: float, n: float, s: float, extra: float) -> None:
        # cache enough to compute the marginal likelihood; ``extra`` carries failure mass / n_trials
        self._a0, self._b0, self._n, self._s, self._extra = a0, b0, n, s, extra

    def log_marginal_likelihood(self) -> float:
        """Return the Beta-conjugate log marginal likelihood."""
        if self._a0 is None:
            raise ValueError("prior hyperparameters were not recorded")
        # Beta-Bernoulli/Binomial evidence: B(a0+s, b0+f) / B(a0, b0); ``log_base`` carries the
        # product of binomial coefficients for Binomial data (0 for Bernoulli/Geometric).
        return float(self.log_base + betaln(self.a, self.b) - betaln(self._a0, self._b0))


def _beta_prior(prior: dict | None) -> tuple[float, float]:
    # Shared hyperparameter parsing + validation for every Beta-posterior family (Bernoulli,
    # Geometric, Binomial, NegativeBinomial): a non-positive or non-finite a/b is not a valid Beta
    # density and previously passed straight through into the posterior's own a/b unchecked.
    p = prior or {}
    a0, b0 = float(p.get("a", 1.0)), float(p.get("b", 1.0))
    if not (math.isfinite(a0) and math.isfinite(b0)) or a0 <= 0.0 or b0 <= 0.0:
        raise ValueError(
            "Beta prior requires finite positive hyperparameters a > 0 and b > 0 (got a=%r, b=%r)." % (a0, b0)
        )
    return a0, b0


def _build_bernoulli(dist, data, weights, prior) -> BetaPosterior:
    x, w = _as_weighted_array(data, weights)
    if not np.all((x == 0.0) | (x == 1.0)):
        # A value outside {0, 1} is not a valid Bernoulli outcome; letting it through silently
        # corrupts the sufficient statistics (e.g. x=2 is counted as two successes at once).
        raise ValueError("Bernoulli conjugate update requires observations in {0, 1}.")
    n = float(w.sum())
    s = float(np.dot(w, x))
    a0, b0 = _beta_prior(prior)
    post = BetaPosterior(a0 + s, b0 + n - s, kind="bernoulli")
    post._set_prior(a0, b0, n, s, n - s)
    return post


def _build_geometric(dist, data, weights, prior) -> BetaPosterior:
    # Geometric(p) on {1,2,...}: likelihood propto p^N (1-p)^(sum_x - N).
    x, w = _as_weighted_array(data, weights)
    if np.any(x < 1.0) or np.any(x != np.floor(x)):
        raise ValueError("Geometric conjugate update requires integer observations in {1, 2, ...}.")
    n = float(w.sum())
    sx = float(np.dot(w, x))
    a0, b0 = _beta_prior(prior)
    post = BetaPosterior(a0 + n, b0 + sx - n, kind="geometric")
    post._set_prior(a0, b0, n, n, sx - n)
    return post


def _build_binomial(dist, data, weights, prior) -> BetaPosterior:
    n_trials = int(getattr(dist, "n", getattr(dist, "n_trials", 1)))
    x, w = _as_weighted_array(data, weights)
    if np.any(x != np.floor(x)) or np.any(x < 0.0) or np.any(x > n_trials):
        # More successes than trials (or a fractional/negative count) is not a valid Binomial(n)
        # outcome; letting it through silently drives the posterior's failure count negative.
        raise ValueError(
            "Binomial conjugate update requires integer successes in [0, %d] (n_trials); got data "
            "outside that range." % n_trials
        )
    N = float(w.sum())
    s = float(np.dot(w, x))
    a0, b0 = _beta_prior(prior)
    failures = n_trials * N - s
    post = BetaPosterior(a0 + s, b0 + failures, kind="binomial", n_trials=n_trials)
    post.log_base = float(np.dot(w, gammaln(n_trials + 1.0) - gammaln(x + 1.0) - gammaln(n_trials - x + 1.0)))
    post._set_prior(a0, b0, N, s, failures)
    return post


# ---------------------------------------------------------------------------
# Gamma posterior (Poisson / Exponential / Gamma-known-shape rate)
# ---------------------------------------------------------------------------
class GammaRatePosterior(ConjugatePosterior):
    """Gamma posterior over a rate parameter for Poisson, exponential, or known-shape Gamma likelihoods."""

    family = "Gamma"

    def __init__(self, shape: float, rate: float, kind: str = "poisson", known_shape: float = 1.0):
        self.shape = _positive_scalar(shape, "Gamma posterior shape")
        self.rate = _positive_scalar(rate, "Gamma posterior rate")
        self.kind = kind
        self.known_shape = _positive_scalar(known_shape, "known Gamma shape")
        self._a0 = None

    def mean(self) -> dict[str, Any]:
        """Return the posterior mean rate."""
        return {"rate": self.shape / self.rate}

    def sample(self, n: int = 1, rng: np.random.RandomState | None = None) -> dict[str, np.ndarray]:
        """Draw rate samples from the Gamma posterior."""
        n = _validate_sample_size(n)
        rng = rng or np.random.RandomState()
        return {"rate": rng.gamma(self.shape, 1.0 / self.rate, size=n)}

    def point_estimate(self):
        """Return the likelihood distribution at the posterior mean rate."""
        lam = self.shape / self.rate
        if self.kind == "poisson":
            from mixle.stats.univariate.discrete.poisson import PoissonDistribution

            return PoissonDistribution(lam)
        if self.kind == "exponential":
            from mixle.stats.univariate.continuous.exponential import ExponentialDistribution

            return ExponentialDistribution(1.0 / lam)  # ExponentialDistribution is mean-parameterised
        from mixle.stats.univariate.continuous.gamma import GammaDistribution

        return GammaDistribution(self.known_shape, 1.0 / lam)  # k=shape, theta=scale=1/rate

    def posterior_predictive(self):
        """Return an exact closed-form predictive or an explicitly marked plug-in."""
        if self.kind == "poisson":
            # Poisson-Gamma predictive is Negative-Binomial(r=A, p=B/(B+1)).
            from mixle.stats.univariate.discrete.negative_binomial import NegativeBinomialDistribution

            return _mark_exact_predictive(
                NegativeBinomialDistribution(self.shape, self.rate / (self.rate + 1.0)),
                self.family,
                "Poisson-Gamma compound law",
            )
        return _mark_approximate_predictive(
            self.point_estimate(),
            self.family,
            "the exact exponential-Gamma compound predictive is not implemented",
        )

    def hyper(self) -> dict[str, Any]:
        """Return the Gamma shape and rate hyperparameters."""
        return {"shape": self.shape, "rate": self.rate}

    def _set_prior(self, a0, b0, n, sx):
        self._a0, self._b0, self._n, self._sx = a0, b0, n, sx

    def log_marginal_likelihood(self) -> float:
        """Return the Gamma-conjugate log marginal likelihood."""
        if self._a0 is None:
            raise ValueError("prior hyperparameters were not recorded")
        # Gamma-conjugate evidence; ``log_base`` carries sum_i log h(x_i) (-sum log x_i! for Poisson, 0
        # for Exponential), making this the absolute log marginal likelihood usable across families.
        return float(
            self.log_base
            + gammaln(self.shape)
            - gammaln(self._a0)
            + self._a0 * math.log(self._b0)
            - self.shape * math.log(self.rate)
        )


def _build_poisson(dist, data, weights, prior) -> GammaRatePosterior:
    x, w = _as_weighted_array(data, weights)
    if np.any(x < 0.0) or np.any(x != np.floor(x)):
        raise ValueError("Poisson conjugate update requires non-negative integer observations.")
    n = float(w.sum())
    sx = float(np.dot(w, x))
    a0, b0 = _gamma_prior(prior)
    post = GammaRatePosterior(a0 + sx, b0 + n, kind="poisson")
    post.log_base = float(-np.dot(w, gammaln(x + 1.0)))  # sum_i log(1/x_i!)
    post._set_prior(a0, b0, n, sx)
    return post


def _build_exponential(dist, data, weights, prior) -> GammaRatePosterior:
    x, w = _as_weighted_array(data, weights)
    if np.any(x < 0.0):
        raise ValueError("Exponential conjugate update requires non-negative observations.")
    n = float(w.sum())
    sx = float(np.dot(w, x))
    a0, b0 = _gamma_prior(prior)
    post = GammaRatePosterior(a0 + n, b0 + sx, kind="exponential")
    post._set_prior(a0, b0, n, sx)
    return post


# ---------------------------------------------------------------------------
# Dirichlet posterior (Categorical / IntegerCategorical)
# ---------------------------------------------------------------------------
class DirichletPosterior(ConjugatePosterior):
    """Dirichlet posterior over categorical probabilities."""

    family = "Dirichlet"

    def __init__(self, alpha: np.ndarray, support: list, kind: str = "categorical", min_val: int = 0):
        self.support = list(support)
        self.alpha = _positive_vector(alpha, "Dirichlet posterior alpha", length=len(self.support))
        self.kind = kind
        self.min_val = int(min_val)
        self._alpha0 = None

    def mean(self) -> dict[str, Any]:
        """Return posterior mean probabilities and a support-to-probability map."""
        p = self.alpha / self.alpha.sum()
        return {"probs": p, "map": dict(zip(self.support, p))}

    def sample(self, n: int = 1, rng: np.random.RandomState | None = None) -> dict[str, np.ndarray]:
        """Draw categorical probability vectors from the Dirichlet posterior."""
        n = _validate_sample_size(n)
        rng = rng or np.random.RandomState()
        return {"probs": rng.dirichlet(self.alpha, size=n)}

    def point_estimate(self):
        """Return a categorical likelihood at the posterior mean probabilities."""
        p = self.alpha / self.alpha.sum()
        if self.kind == "integer_categorical":
            from mixle.stats.univariate.discrete.integer_categorical import IntegerCategoricalDistribution

            return IntegerCategoricalDistribution(self.min_val, p)
        from mixle.stats.univariate.discrete.categorical import CategoricalDistribution

        return CategoricalDistribution(dict(zip(self.support, p)))

    def posterior_predictive(self):
        """Return the exact one-observation categorical posterior predictive."""
        return _mark_exact_predictive(
            self.point_estimate(),
            self.family,
            "single-observation marginal equals the posterior-mean probability vector",
        )

    def hyper(self) -> dict[str, Any]:
        """Return the Dirichlet concentration vector."""
        return {"alpha": self.alpha}

    def _set_prior(self, alpha0):
        self._alpha0 = np.asarray(alpha0, dtype=np.float64)

    def log_marginal_likelihood(self) -> float:
        """Return the Dirichlet-multinomial log marginal likelihood."""
        if self._alpha0 is None:
            raise ValueError("prior hyperparameters were not recorded")
        a0, an = self._alpha0, self.alpha
        # Dirichlet-multinomial evidence (multinomial coefficient dropped -- constant in the parameter)
        return float(gammaln(a0.sum()) - gammaln(an.sum()) + (gammaln(an).sum() - gammaln(a0).sum()))


def _categorical_counts(dist, data, weights):
    x = list(data)
    weights = _as_weight_vector(weights, len(x))
    counts: Counter = Counter()
    for v, w in zip(x, weights):
        try:
            hash(v)
        except TypeError as exc:
            raise ValueError("Categorical conjugate observations must be hashable.") from exc
        counts[v] += float(w)
    return counts


def _dirichlet_prior(prior: dict | None, width: int) -> np.ndarray:
    raw = (prior or {}).get("alpha", 1.0)
    if np.ndim(raw) == 0:
        value = _positive_scalar(raw, "Dirichlet prior alpha")
        return np.full(width, value, dtype=np.float64)
    return _positive_vector(raw, "Dirichlet prior alpha", length=width)


def _build_categorical(dist, data, weights, prior) -> DirichletPosterior:
    # support comes from the distribution if available, else from the data
    support = list(getattr(dist, "pmap", {}).keys()) if getattr(dist, "pmap", None) else None
    counts = _categorical_counts(dist, data, weights)
    if support is None:
        support = sorted(
            counts,
            key=lambda value: (
                type(value).__module__,
                type(value).__qualname__,
                repr(value),
            ),
        )
    else:
        unsupported = [value for value in counts if value not in dist.pmap]
        if unsupported:
            raise ValueError(
                "Categorical conjugate update received observations outside its explicit support: %r."
                % unsupported
            )
    if not support:
        raise ValueError("Categorical conjugate update requires a non-empty support.")
    alpha0 = _dirichlet_prior(prior, len(support))
    cvec = np.array([counts.get(s, 0.0) for s in support], dtype=np.float64)
    post = DirichletPosterior(alpha0 + cvec, support, kind="categorical")
    post._set_prior(alpha0)
    return post


def _build_integer_categorical(dist, data, weights, prior) -> DirichletPosterior:
    x, w = _as_weighted_array(data, weights)
    if np.any(x != np.floor(x)):
        # A fractional observation is not a valid category index; truncating it (the old
        # x.astype(int) below silently did exactly that) folds it into the next-lower category
        # instead of rejecting the malformed observation.
        raise ValueError("IntegerCategoricalDistribution conjugate update requires integer-valued observations.")
    if hasattr(dist, "min_val"):
        min_val = int(dist.min_val)
    elif x.size:
        min_val = int(x.min())
    else:
        raise ValueError("Integer-categorical conjugate update cannot infer support from empty data.")
    k = len(getattr(dist, "p_vec", getattr(dist, "prob_vec", [])))
    if k == 0:
        if not x.size:
            raise ValueError("Integer-categorical conjugate update requires a non-empty support.")
        k = int(x.max()) - min_val + 1
    support = list(range(min_val, min_val + k))
    cvec = np.zeros(k, dtype=np.float64)
    xi_arr = x.astype(int)
    offset = xi_arr - min_val
    out_of_range = (offset < 0) | (offset >= k)
    if np.any(out_of_range):
        # An out-of-support category used to be silently dropped -- the old loop's
        # ``if 0 <= xi - min_val < k`` guarded the accumulation but nothing else -- losing the
        # observation instead of rejecting the malformed input.
        bad = sorted({int(v) for v in xi_arr[out_of_range]})
        raise ValueError(
            "IntegerCategoricalDistribution conjugate update received category value(s) %r outside "
            "the supported range [%d, %d]." % (bad, min_val, min_val + k - 1)
        )
    for xi, wi in zip(xi_arr, w):
        cvec[xi - min_val] += wi
    alpha0 = _dirichlet_prior(prior, k)
    post = DirichletPosterior(alpha0 + cvec, support, kind="integer_categorical", min_val=min_val)
    post._set_prior(alpha0)
    return post


# ---------------------------------------------------------------------------
# Normal-Inverse-Gamma posterior (Gaussian, unknown mean AND variance)
# ---------------------------------------------------------------------------
class NormalInverseGammaPosterior(ConjugatePosterior):
    """Normal-Inverse-Gamma posterior for a univariate Gaussian mean and variance."""

    family = "NormalInverseGamma"

    def __init__(self, m: float, kappa: float, a: float, b: float, kind: str = "gaussian"):
        self.m = _finite_scalar(m, "Normal-Inverse-Gamma posterior m")
        self.kappa = _positive_scalar(kappa, "Normal-Inverse-Gamma posterior kappa")
        self.a = _positive_scalar(a, "Normal-Inverse-Gamma posterior a")
        self.b = _positive_scalar(b, "Normal-Inverse-Gamma posterior b")
        self.kind = kind  # "gaussian" or "log_gaussian" (the latter models log x)
        self._prior = None

    def mean(self) -> dict[str, Any]:
        """Return posterior mean parameters for ``mu`` and ``sigma2``."""
        return {"mu": self.m, "sigma2": self.b / (self.a - 1.0) if self.a > 1.0 else float("inf")}

    def sample(self, n: int = 1, rng: np.random.RandomState | None = None) -> dict[str, np.ndarray]:
        """Draw paired ``mu`` and ``sigma2`` samples from the posterior."""
        n = _validate_sample_size(n)
        rng = rng or np.random.RandomState()
        sigma2 = 1.0 / rng.gamma(self.a, 1.0 / self.b, size=n)  # InvGamma(a,b)
        mu = rng.normal(self.m, np.sqrt(sigma2 / self.kappa))
        return {"mu": mu, "sigma2": sigma2}

    def point_estimate(self):
        """Return the Gaussian-family likelihood at posterior mean parameters."""
        sigma2 = self.b / (self.a - 1.0) if self.a > 1.0 else self.b / self.a
        if self.kind == "log_gaussian":
            from mixle.stats.univariate.continuous.log_gaussian import LogGaussianDistribution

            return LogGaussianDistribution(self.m, sigma2)
        from mixle.stats.univariate.continuous.gaussian import GaussianDistribution

        return GaussianDistribution(self.m, sigma2)

    def posterior_predictive(self):
        """Return the Student-t predictive when available, otherwise the plug-in predictive."""
        # Marginalising (mu, sigma2) gives a Student-t: df=2a, loc=m, scale^2=b(kappa+1)/(a*kappa).
        # For log_gaussian the predictive is that Student-t in log-space (a log-Student-t in x); the
        # plug-in LogGaussian is returned as a usable mixle distribution.
        if self.kind == "log_gaussian":
            return _mark_approximate_predictive(
                self.point_estimate(),
                self.family,
                "the exact log-Student-t predictive distribution is not implemented",
            )
        from mixle.stats.univariate.continuous.student_t import StudentTDistribution

        scale = math.sqrt(self.b * (self.kappa + 1.0) / (self.a * self.kappa))
        return _mark_exact_predictive(
            StudentTDistribution(2.0 * self.a, self.m, scale),
            self.family,
            "Normal-Inverse-Gamma Student-t compound law",
        )

    def hyper(self) -> dict[str, Any]:
        """Return Normal-Inverse-Gamma posterior hyperparameters."""
        return {"m": self.m, "kappa": self.kappa, "a": self.a, "b": self.b}

    def _set_prior(self, m0, k0, a0, b0, n):
        self._prior = (m0, k0, a0, b0, n)

    def log_marginal_likelihood(self) -> float:
        """Return the Normal-Inverse-Gamma log marginal likelihood."""
        if self._prior is None:
            raise ValueError("prior hyperparameters were not recorded")
        m0, k0, a0, b0, n = self._prior
        return float(
            self.log_base  # Jacobian sum_i -log x_i for log_gaussian; 0 for gaussian
            + gammaln(self.a)
            - gammaln(a0)
            + a0 * math.log(b0)
            - self.a * math.log(self.b)
            + 0.5 * (math.log(k0) - math.log(self.kappa))
            - 0.5 * n * math.log(2.0 * math.pi)
        )


def _build_gaussian(dist, data, weights, prior) -> NormalInverseGammaPosterior:
    x, w = _as_weighted_array(data, weights)
    n = float(w.sum())
    sx = float(np.dot(w, x))
    xbar = sx / n if n > 0 else 0.0
    p = prior or {}
    m0 = _finite_scalar(p.get("m", 0.0), "Normal-Inverse-Gamma prior m")
    k0 = _positive_scalar(p.get("kappa", 1e-3), "Normal-Inverse-Gamma prior kappa")
    a0 = _positive_scalar(p.get("a", 1e-3), "Normal-Inverse-Gamma prior a")
    b0 = _positive_scalar(p.get("b", 1e-3), "Normal-Inverse-Gamma prior b")
    kn = k0 + n
    mn = (k0 * m0 + sx) / kn
    an = a0 + 0.5 * n
    # b_n = b0 + 0.5*sum w (x-xbar)^2 + 0.5 k0 n (xbar-m0)^2 / kn.
    # Center the scatter (the raw data is in hand here) rather than computing it as
    # sx2 - n*xbar^2, which cancels catastrophically for large-|xbar| data.
    ss = float(np.dot(w, (x - xbar) ** 2)) if n > 0 else 0.0
    bn = b0 + 0.5 * ss + 0.5 * k0 * n * (xbar - m0) ** 2 / kn
    post = NormalInverseGammaPosterior(mn, kn, an, bn)
    post._set_prior(m0, k0, a0, b0, n)
    return post


# ---------------------------------------------------------------------------
# Normal-Inverse-Wishart posterior (multivariate Gaussian, unknown mean AND covariance)
# ---------------------------------------------------------------------------
class NormalInverseWishartPosterior(ConjugatePosterior):
    """Normal-Inverse-Wishart posterior for multivariate Gaussian mean and covariance."""

    family = "NormalInverseWishart"

    def __init__(self, m: np.ndarray, kappa: float, nu: float, psi: np.ndarray):
        try:
            self.m = np.asarray(m, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("Normal-Inverse-Wishart posterior m must be a numeric vector.") from exc
        if self.m.ndim != 1 or self.m.size == 0 or np.any(~np.isfinite(self.m)):
            raise ValueError("Normal-Inverse-Wishart posterior m must be a non-empty finite vector.")
        self.d = self.m.shape[0]
        self.kappa = _positive_scalar(kappa, "Normal-Inverse-Wishart posterior kappa")
        self.nu = _finite_scalar(nu, "Normal-Inverse-Wishart posterior nu")
        if self.nu <= self.d - 1.0:
            raise ValueError("Normal-Inverse-Wishart posterior nu must exceed dimension - 1.")
        self.psi = _spd_matrix(psi, self.d, "Normal-Inverse-Wishart posterior psi")
        self._prior = None

    def mean(self) -> dict[str, Any]:
        """Return posterior mean parameters for the Gaussian mean and covariance."""
        cov = self.psi / (self.nu - self.d - 1.0) if self.nu > self.d + 1 else None
        return {"mean": self.m, "cov": cov}

    def sample(self, n: int = 1, rng: np.random.RandomState | None = None) -> dict[str, np.ndarray]:
        """Draw paired mean and covariance samples from the posterior."""
        n = _validate_sample_size(n)
        rng = rng or np.random.RandomState()
        means = np.empty((n, self.d))
        covs = np.empty((n, self.d, self.d))
        psi_inv = np.linalg.inv(self.psi)
        for i in range(n):
            sigma = _sample_inverse_wishart(self.nu, psi_inv, rng)
            covs[i] = sigma
            means[i] = rng.multivariate_normal(self.m, sigma / self.kappa)
        return {"mean": means, "cov": covs}

    def point_estimate(self):
        """Return a multivariate Gaussian at representative posterior parameters."""
        from mixle.stats.multivariate.multivariate_gaussian import MultivariateGaussianDistribution

        cov = self.psi / (self.nu - self.d - 1.0) if self.nu > self.d + 1 else self.psi / self.nu
        return MultivariateGaussianDistribution(self.m, cov)

    def posterior_predictive(self):
        """Return the multivariate Student-t posterior predictive distribution."""
        from mixle.stats.multivariate.multivariate_student_t import MultivariateStudentTDistribution

        df = self.nu - self.d + 1.0
        shape = self.psi * (self.kappa + 1.0) / (self.kappa * df)
        return _mark_exact_predictive(
            MultivariateStudentTDistribution(df, self.m, shape),
            self.family,
            "Normal-Inverse-Wishart multivariate Student-t compound law",
        )

    def hyper(self) -> dict[str, Any]:
        """Return Normal-Inverse-Wishart posterior hyperparameters."""
        return {"m": self.m, "kappa": self.kappa, "nu": self.nu, "psi": self.psi}

    def _set_prior(self, m0, k0, nu0, psi0, n):
        self._prior = (np.asarray(m0), k0, nu0, np.asarray(psi0), n)

    def log_marginal_likelihood(self) -> float:
        """Return the Normal-Inverse-Wishart log marginal likelihood."""
        if self._prior is None:
            raise ValueError("prior hyperparameters were not recorded")
        m0, k0, nu0, psi0, n = self._prior
        d = self.d
        return float(
            multigammaln(self.nu / 2.0, d)
            - multigammaln(nu0 / 2.0, d)
            + (nu0 / 2.0) * _logdet(psi0)
            - (self.nu / 2.0) * _logdet(self.psi)
            + (d / 2.0) * (math.log(k0) - math.log(self.kappa))
            - (n * d / 2.0) * math.log(math.pi)
        )


def _logdet(a: np.ndarray) -> float:
    matrix = _spd_matrix(a, len(a), "log-determinant matrix")
    chol = np.linalg.cholesky(matrix)
    return float(2.0 * np.log(np.diag(chol)).sum())


def _sample_inverse_wishart(nu: float, psi_inv: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    # Sample W ~ Wishart(nu, psi_inv) via Bartlett, then invert to get Inverse-Wishart(nu, psi).
    d = psi_inv.shape[0]
    chol = np.linalg.cholesky(psi_inv)
    a = np.zeros((d, d))
    for i in range(d):
        a[i, i] = math.sqrt(rng.chisquare(nu - i))
        for j in range(i):
            a[i, j] = rng.normal()
    w = chol @ a @ a.T @ chol.T
    return np.linalg.inv(w)


def _build_mvn(dist, data, weights, prior) -> NormalInverseWishartPosterior:
    d = int(getattr(dist, "dim", len(dist.mu)))
    if d <= 0:
        raise ValueError("Multivariate Gaussian conjugate update requires a positive dimension.")
    try:
        x = np.asarray(list(data), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Multivariate Gaussian conjugate observations must be numeric vectors.") from exc
    if x.size == 0:
        x = np.empty((0, d), dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != d:
        raise ValueError(
            "Multivariate Gaussian conjugate observations must have exact shape (n, %d), got %r."
            % (d, x.shape)
        )
    if np.any(~np.isfinite(x)):
        raise ValueError("Multivariate Gaussian conjugate observations must be finite.")
    w = _as_weight_vector(weights, x.shape[0])
    n = float(w.sum())
    p = prior or {}
    try:
        m0 = np.asarray(p.get("m", np.zeros(d)), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Normal-Inverse-Wishart prior m must be a numeric vector.") from exc
    if m0.shape != (d,) or np.any(~np.isfinite(m0)):
        raise ValueError("Normal-Inverse-Wishart prior m must be a finite vector of length %d." % d)
    k0 = _positive_scalar(p.get("kappa", 1e-3), "Normal-Inverse-Wishart prior kappa")
    nu0 = _finite_scalar(p.get("nu", d + 2.0), "Normal-Inverse-Wishart prior nu")
    if nu0 <= d - 1.0:
        raise ValueError("Normal-Inverse-Wishart prior nu must exceed dimension - 1.")
    psi0 = _spd_matrix(p.get("psi", np.eye(d) * 1e-3), d, "Normal-Inverse-Wishart prior psi")
    xbar = (w[:, None] * x).sum(axis=0) / n if n > 0.0 else m0.copy()
    diff = x - xbar
    scatter = (w[:, None] * diff).T @ diff
    kn = k0 + n
    mn = (k0 * m0 + n * xbar) / kn
    nun = nu0 + n
    dm = xbar - m0
    psin = psi0 + scatter + (k0 * n / kn) * np.outer(dm, dm)
    post = NormalInverseWishartPosterior(mn, kn, nun, psin)
    post._set_prior(m0, k0, nu0, psi0, n)
    return post


def _build_log_gaussian(dist, data, weights, prior) -> NormalInverseGammaPosterior:
    # LogGaussian models log x ~ Normal(mu, sigma2): apply the Normal-Inverse-Gamma update to log(data),
    # then add the change-of-variables Jacobian sum_i -log x_i to the evidence.
    x, w = _as_weighted_array(data, weights)
    if np.any(x <= 0.0):
        raise ValueError("Log-Gaussian conjugate update requires strictly positive observations.")
    logx = np.log(x)
    post = _build_gaussian(dist, list(logx), w, prior)
    post.kind = "log_gaussian"
    post.log_base = float(-np.dot(w, logx))
    return post


# ---------------------------------------------------------------------------
# Inverse-Gamma-on-variance posterior (Rayleigh, HalfNormal -- unknown scale sigma)
# ---------------------------------------------------------------------------
class InverseGammaVariancePosterior(ConjugatePosterior):
    """Posterior over the squared-scale ``sigma2`` of a zero-location scale family.

    Rayleigh and half-normal both have a single scale ``sigma`` with ``x^2`` as the sufficient
    statistic, so ``sigma2`` has an Inverse-Gamma conjugate posterior.
    """

    family = "InverseGamma(variance)"

    def __init__(self, a: float, b: float, kind: str):
        self.a = _positive_scalar(a, "Inverse-Gamma posterior a")
        self.b = _positive_scalar(b, "Inverse-Gamma posterior b")
        self.kind = kind  # "rayleigh" or "half_normal"
        self._prior = None

    def _sigma2_mean(self) -> float:
        return self.b / (self.a - 1.0) if self.a > 1.0 else self.b / self.a

    def mean(self) -> dict[str, Any]:
        """Return posterior mean squared scale and scale."""
        s2 = self.b / (self.a - 1.0) if self.a > 1.0 else float("inf")
        return {"sigma2": s2, "sigma": math.sqrt(s2) if math.isfinite(s2) else float("inf")}

    def sample(self, n: int = 1, rng: np.random.RandomState | None = None) -> dict[str, np.ndarray]:
        """Draw squared-scale and scale samples from the posterior."""
        n = _validate_sample_size(n)
        rng = rng or np.random.RandomState()
        s2 = 1.0 / rng.gamma(self.a, 1.0 / self.b, size=n)
        return {"sigma2": s2, "sigma": np.sqrt(s2)}

    def point_estimate(self):
        """Return the likelihood distribution at the posterior mean scale."""
        sigma = math.sqrt(self._sigma2_mean())
        if self.kind == "half_normal":
            from mixle.stats.univariate.continuous.half_normal import HalfNormalDistribution

            return HalfNormalDistribution(sigma)
        from mixle.stats.univariate.continuous.rayleigh import RayleighDistribution

        return RayleighDistribution(sigma)

    def posterior_predictive(self):
        """Return an explicitly marked plug-in scale-family predictive distribution."""
        return _mark_approximate_predictive(
            self.point_estimate(),
            self.family,
            "the exact inverse-Gamma scale-mixture predictive is not implemented",
        )

    def hyper(self) -> dict[str, Any]:
        """Return inverse-gamma scale hyperparameters."""
        return {"a": self.a, "b": self.b}

    def _set_prior(self, a0, b0):
        self._prior = (a0, b0)

    def log_marginal_likelihood(self) -> float:
        """Return the inverse-gamma scale log marginal likelihood."""
        a0, b0 = self._prior
        return float(self.log_base + a0 * math.log(b0) - gammaln(a0) + gammaln(self.a) - self.a * math.log(self.b))


def _inverse_gamma_prior(prior: dict | None) -> tuple[float, float]:
    p = prior or {}
    return (
        _positive_scalar(p.get("a", 1e-3), "Inverse-Gamma prior a"),
        _positive_scalar(p.get("b", 1e-3), "Inverse-Gamma prior b"),
    )


def _build_rayleigh(dist, data, weights, prior) -> InverseGammaVariancePosterior:
    x, w = _as_weighted_array(data, weights)
    if np.any(x < 0.0):
        raise ValueError("Rayleigh conjugate update requires non-negative observations.")
    n, sx2 = float(w.sum()), float(np.dot(w, x * x))
    a0, b0 = _inverse_gamma_prior(prior)
    post = InverseGammaVariancePosterior(a0 + n, b0 + 0.5 * sx2, kind="rayleigh")
    post.log_base = float(np.dot(w, np.log(x)))  # h(x) = x
    post._set_prior(a0, b0)
    return post


def _build_half_normal(dist, data, weights, prior) -> InverseGammaVariancePosterior:
    x, w = _as_weighted_array(data, weights)
    if np.any(x < 0.0):
        raise ValueError("Half-normal conjugate update requires non-negative observations.")
    n, sx2 = float(w.sum()), float(np.dot(w, x * x))
    a0, b0 = _inverse_gamma_prior(prior)
    post = InverseGammaVariancePosterior(a0 + 0.5 * n, b0 + 0.5 * sx2, kind="half_normal")
    post.log_base = float(0.5 * n * math.log(2.0 / math.pi))  # h(x) = sqrt(2/pi)
    post._set_prior(a0, b0)
    return post


# ---------------------------------------------------------------------------
# Gamma-on-positive-parameter posterior (Gamma/InverseGamma/InverseGaussian/Pareto,
# each conditional on its known shape/location/scale parameter, taken from the distribution)
# ---------------------------------------------------------------------------
class GammaParameterPosterior(ConjugatePosterior):
    """Gamma posterior over a positive parameter ``psi`` of a one-parameter exponential family.

    Each supported likelihood factorises (given its other, known parameter) as
    ``propto psi^P exp(-psi * C)``, so a ``Gamma(A0, B0)`` prior on ``psi`` gives ``Gamma(A0+P, B0+C)``.
    ``kind`` selects which parameter ``psi`` is and how to rebuild a distribution at its posterior
    mean (using the known parameter ``fixed``).
    """

    family = "Gamma(parameter)"

    def __init__(self, shape: float, rate: float, kind: str, fixed: float, param: str):
        self.shape = _positive_scalar(shape, "Gamma parameter posterior shape")
        self.rate = _positive_scalar(rate, "Gamma parameter posterior rate")
        self.kind = kind
        self.fixed = _positive_scalar(fixed, "known %s parameter" % kind)
        self.param = param  # the name of psi for reporting
        self._prior = None

    def mean(self) -> dict[str, Any]:
        """Return the posterior mean of the positive parameter."""
        return {self.param: self.shape / self.rate}

    def sample(self, n: int = 1, rng: np.random.RandomState | None = None) -> dict[str, np.ndarray]:
        """Draw positive-parameter samples from the Gamma posterior."""
        n = _validate_sample_size(n)
        rng = rng or np.random.RandomState()
        return {self.param: rng.gamma(self.shape, 1.0 / self.rate, size=n)}

    def point_estimate(self):
        """Return the likelihood distribution at the posterior mean parameter."""
        psi = self.shape / self.rate
        if self.kind == "gamma":
            from mixle.stats.univariate.continuous.gamma import GammaDistribution

            return GammaDistribution(self.fixed, 1.0 / psi)  # known shape k, theta = 1/rate
        if self.kind == "inverse_gamma":
            from mixle.stats.univariate.continuous.inverse_gamma import InverseGammaDistribution

            return InverseGammaDistribution(self.fixed, psi)  # known alpha, beta = psi
        if self.kind == "inverse_gaussian":
            from mixle.stats.univariate.continuous.inverse_gaussian import InverseGaussianDistribution

            return InverseGaussianDistribution(self.fixed, psi)  # known mu, lam = psi
        from mixle.stats.univariate.continuous.pareto import ParetoDistribution

        return ParetoDistribution(self.fixed, psi)  # known xm, alpha = psi

    def posterior_predictive(self):
        """Return an explicitly marked plug-in for the known-parameter family."""
        return _mark_approximate_predictive(
            self.point_estimate(),
            self.family,
            "the exact compound predictive is not implemented for %s" % self.kind,
        )

    def hyper(self) -> dict[str, Any]:
        """Return Gamma posterior hyperparameters and posterior mean parameter."""
        return {"shape": self.shape, "rate": self.rate, self.param: self.shape / self.rate}

    def _set_prior(self, a0, b0):
        self._prior = (a0, b0)

    def log_marginal_likelihood(self) -> float:
        """Return the Gamma-parameter log marginal likelihood."""
        a0, b0 = self._prior
        return float(
            self.log_base + gammaln(self.shape) - gammaln(a0) + a0 * math.log(b0) - self.shape * math.log(self.rate)
        )


def _gamma_prior(prior):
    p = prior or {}
    return (
        _positive_scalar(p.get("shape", 1e-3), "Gamma prior shape"),
        _positive_scalar(p.get("rate", 1e-3), "Gamma prior rate"),
    )


def _build_gamma(dist, data, weights, prior) -> GammaParameterPosterior:
    # Gamma(k, theta): known shape k -> rate=1/theta has a Gamma posterior.
    k = _positive_scalar(dist.k, "known Gamma likelihood shape")
    x, w = _as_weighted_array(data, weights)
    if np.any(x <= 0.0):
        raise ValueError("Gamma conjugate update requires strictly positive observations.")
    n, sx = float(w.sum()), float(np.dot(w, x))
    a0, b0 = _gamma_prior(prior)
    post = GammaParameterPosterior(a0 + n * k, b0 + sx, kind="gamma", fixed=k, param="rate")
    post.log_base = float(np.dot(w, (k - 1.0) * np.log(x)) - n * gammaln(k))  # x^{k-1}/Gamma(k)
    post._set_prior(a0, b0)
    return post


def _build_inverse_gamma(dist, data, weights, prior) -> GammaParameterPosterior:
    # InverseGamma(alpha, beta): known alpha -> beta has a Gamma posterior (suff stat 1/x).
    alpha = _positive_scalar(dist.alpha, "known inverse-Gamma likelihood shape")
    x, w = _as_weighted_array(data, weights)
    if np.any(x <= 0.0):
        raise ValueError("Inverse-Gamma conjugate update requires strictly positive observations.")
    n, s_inv = float(w.sum()), float(np.dot(w, 1.0 / x))
    a0, b0 = _gamma_prior(prior)
    post = GammaParameterPosterior(a0 + n * alpha, b0 + s_inv, kind="inverse_gamma", fixed=alpha, param="beta")
    post.log_base = float(np.dot(w, -(alpha + 1.0) * np.log(x)) - n * gammaln(alpha))  # x^{-alpha-1}/Gamma(alpha)
    post._set_prior(a0, b0)
    return post


def _build_inverse_gaussian(dist, data, weights, prior) -> GammaParameterPosterior:
    # InverseGaussian(mu, lam): known mu -> lam has a Gamma posterior.
    mu = _positive_scalar(dist.mu, "known inverse-Gaussian likelihood mean")
    x, w = _as_weighted_array(data, weights)
    if np.any(x <= 0.0):
        raise ValueError("Inverse-Gaussian conjugate update requires strictly positive observations.")
    n = float(w.sum())
    c = float(np.dot(w, (x - mu) ** 2 / (2.0 * mu * mu * x)))
    a0, b0 = _gamma_prior(prior)
    post = GammaParameterPosterior(a0 + 0.5 * n, b0 + c, kind="inverse_gaussian", fixed=mu, param="lam")
    post.log_base = float(np.dot(w, -0.5 * np.log(2.0 * math.pi * x**3)))  # sqrt(1/(2 pi x^3))
    post._set_prior(a0, b0)
    return post


def _build_pareto(dist, data, weights, prior) -> GammaParameterPosterior:
    # Pareto(xm, alpha): known scale xm -> tail index alpha has a Gamma posterior.
    xm = _positive_scalar(dist.xm, "known Pareto likelihood scale")
    x, w = _as_weighted_array(data, weights)
    if np.any(x < xm):
        raise ValueError("Pareto conjugate update requires observations greater than or equal to its scale.")
    n = float(w.sum())
    c = float(np.dot(w, np.log(x / xm)))
    a0, b0 = _gamma_prior(prior)
    post = GammaParameterPosterior(a0 + n, b0 + c, kind="pareto", fixed=xm, param="alpha")
    post.log_base = float(np.dot(w, -np.log(x)))  # density carries a 1/x factor
    post._set_prior(a0, b0)
    return post


def _build_negative_binomial(dist, data, weights, prior) -> BetaPosterior:
    # NegativeBinomial(r, p): known r -> success prob p has a Beta posterior (likelihood p^{nr}(1-p)^{sum x}).
    r = _positive_scalar(dist.r, "known negative-binomial likelihood r")
    x, w = _as_weighted_array(data, weights)
    if np.any(x < 0.0) or np.any(x != np.floor(x)):
        raise ValueError("Negative-binomial conjugate update requires non-negative integer observations.")
    n, sx = float(w.sum()), float(np.dot(w, x))
    a0, b0 = _beta_prior(prior)
    post = BetaPosterior(a0 + n * r, b0 + sx, kind="negative_binomial", n_trials=int(r))
    post.kind = "negative_binomial"
    post._nb_r = r
    post.log_base = float(np.dot(w, gammaln(x + r) - gammaln(r) - gammaln(x + 1.0)))  # binomial coefficient
    post._set_prior(a0, b0, n, sx, sx)
    return post


# ---------------------------------------------------------------------------
# Diagonal Gaussian: independent Normal-Inverse-Gamma per dimension
# ---------------------------------------------------------------------------
class DiagonalNIGPosterior(ConjugatePosterior):
    """Independent Normal-Inverse-Gamma posteriors, one per coordinate of a diagonal Gaussian."""

    family = "DiagonalNormalInverseGamma"

    def __init__(self, per_dim: list[NormalInverseGammaPosterior]):
        if not per_dim or not all(isinstance(item, NormalInverseGammaPosterior) for item in per_dim):
            raise ValueError("Diagonal Gaussian posterior requires at least one proper per-dimension posterior.")
        self.per_dim = list(per_dim)
        self.d = len(per_dim)

    def mean(self) -> dict[str, Any]:
        """Return coordinate-wise posterior mean vectors."""
        return {
            "mu": np.array([p.mean()["mu"] for p in self.per_dim]),
            "sigma2": np.array([p.mean()["sigma2"] for p in self.per_dim]),
        }

    def sample(self, n: int = 1, rng: np.random.RandomState | None = None) -> dict[str, np.ndarray]:
        """Draw diagonal Gaussian parameter samples coordinate by coordinate."""
        n = _validate_sample_size(n)
        rng = rng or np.random.RandomState()
        mus = np.empty((n, self.d))
        s2 = np.empty((n, self.d))
        for j, p in enumerate(self.per_dim):
            s = p.sample(n, rng)
            mus[:, j] = s["mu"]
            s2[:, j] = s["sigma2"]
        return {"mu": mus, "sigma2": s2}

    def point_estimate(self):
        """Return a diagonal Gaussian at coordinate-wise posterior mean parameters."""
        from mixle.stats.multivariate.diagonal_gaussian import DiagonalGaussianDistribution

        mu = np.array([p.mean()["mu"] for p in self.per_dim])
        s2 = np.array([p.b / (p.a - 1.0) if p.a > 1.0 else p.b / p.a for p in self.per_dim])
        return DiagonalGaussianDistribution(mu, s2)

    def posterior_predictive(self):
        """Return an explicitly marked plug-in diagonal Gaussian predictive."""
        return _mark_approximate_predictive(
            self.point_estimate(),
            self.family,
            "the exact product-of-Student-t predictive distribution is not implemented",
        )

    def hyper(self) -> dict[str, Any]:
        """Return per-coordinate posterior hyperparameters."""
        return {"per_dim": [p.hyper() for p in self.per_dim]}

    def log_marginal_likelihood(self) -> float:
        """Return the sum of coordinate-wise log marginal likelihoods."""
        return float(sum(p.log_marginal_likelihood() for p in self.per_dim))


def _build_diagonal_gaussian(dist, data, weights, prior) -> DiagonalNIGPosterior:
    d = int(getattr(dist, "dim", len(dist.mu)))
    if d <= 0:
        raise ValueError("Diagonal Gaussian conjugate update requires a positive dimension.")
    try:
        x = np.asarray(list(data), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Diagonal Gaussian conjugate observations must be numeric vectors.") from exc
    if x.size == 0:
        x = np.empty((0, d), dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != d:
        raise ValueError(
            "Diagonal Gaussian conjugate observations must have exact shape (n, %d), got %r."
            % (d, x.shape)
        )
    if np.any(~np.isfinite(x)):
        raise ValueError("Diagonal Gaussian conjugate observations must be finite.")
    checked_weights = _as_weight_vector(weights, x.shape[0])
    p = prior or {}
    per_dim_priors: list[dict[str, float]] = [dict() for _ in range(d)]
    for key, default in (("m", 0.0), ("kappa", 1e-3), ("a", 1e-3), ("b", 1e-3)):
        raw = p.get(key, default)
        if np.ndim(raw) == 0:
            values = np.full(d, float(raw), dtype=np.float64)
        else:
            try:
                values = np.asarray(raw, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError("Diagonal Normal-Inverse-Gamma prior %s must be numeric." % key) from exc
            if values.shape != (d,):
                raise ValueError(
                    "Diagonal Normal-Inverse-Gamma prior %s must be scalar or length %d." % (key, d)
                )
        for j, value in enumerate(values):
            per_dim_priors[j][key] = float(value)
    per_dim = [
        _build_gaussian(None, list(x[:, j]), checked_weights, per_dim_priors[j])
        for j in range(d)
    ]
    return DiagonalNIGPosterior(per_dim)


# ---------------------------------------------------------------------------
# von Mises: conjugate posterior on the mean direction (concentration kappa known)
# ---------------------------------------------------------------------------
class VonMisesMeanPosterior(ConjugatePosterior):
    """von Mises posterior over the mean direction ``mu`` (concentration ``kappa`` taken as known).

    The conjugate prior for ``mu`` is itself a von Mises ``vM(m0, R0)``; combining with the
    likelihood gives a von Mises ``vM(m_n, R_n)`` whose resultant vector adds the data's.
    """

    family = "vonMises(mean)"

    def __init__(self, m: float, r: float, kappa: float):
        self.m = _finite_scalar(m, "von Mises posterior mean direction")
        self.r = _nonnegative_scalar(r, "von Mises posterior concentration")
        self.kappa = _nonnegative_scalar(kappa, "known von Mises concentration")
        self._prior = None

    def mean(self) -> dict[str, Any]:
        """Return posterior mean direction and concentration."""
        return {"mu": self.m, "concentration": self.r}

    def sample(self, n: int = 1, rng: np.random.RandomState | None = None) -> dict[str, np.ndarray]:
        """Draw mean-direction samples from the von Mises posterior."""
        n = _validate_sample_size(n)
        rng = rng or np.random.RandomState()
        return {"mu": rng.vonmises(self.m, self.r, size=n)}

    def point_estimate(self):
        """Return a von Mises likelihood at the posterior mean direction."""
        from mixle.stats.directional.von_mises import VonMisesDistribution

        return VonMisesDistribution(self.m, self.kappa)

    def posterior_predictive(self):
        """Return an explicitly marked plug-in von Mises predictive."""
        return _mark_approximate_predictive(
            self.point_estimate(),
            self.family,
            "the exact von-Mises compound predictive distribution is not implemented",
        )

    def hyper(self) -> dict[str, Any]:
        """Return von Mises posterior hyperparameters."""
        return {"m": self.m, "R": self.r, "kappa": self.kappa}

    def _set_prior(self, r0):
        self._prior = r0

    def log_marginal_likelihood(self) -> float:
        """Return the von Mises conjugate log marginal likelihood."""
        from scipy.special import ive

        r0 = self._prior
        n = self.log_base  # carries -n (number of obs) via the (2 pi I0(kappa))^{-n} term handled below
        # evidence = I0(R_n) / [ (2 pi I0(kappa))^n * I0(R0) ]; use ive for stability: I0(x)=ive(0,x)e^{x}
        log_i0 = lambda z: math.log(ive(0, z)) + z
        return float(log_i0(self.r) - log_i0(r0) - n * (math.log(2.0 * math.pi) + log_i0(self.kappa)))


def _build_von_mises(dist, data, weights, prior) -> VonMisesMeanPosterior:
    kappa = _nonnegative_scalar(dist.kappa, "known von Mises likelihood concentration")
    x, w = _as_weighted_array(data, weights)
    p = prior or {}
    m0 = _finite_scalar(p.get("m", 0.0), "von Mises prior mean direction")
    r0 = _nonnegative_scalar(p.get("R", 1e-6), "von Mises prior concentration")
    cx = float(np.dot(w, np.cos(x)))
    sx = float(np.dot(w, np.sin(x)))
    rc = kappa * cx + r0 * math.cos(m0)
    rs = kappa * sx + r0 * math.sin(m0)
    r_n = math.hypot(rc, rs)
    m_n = math.atan2(rs, rc)
    post = VonMisesMeanPosterior(m_n, r_n, kappa)
    post.log_base = float(w.sum())  # n, used inside the evidence
    post._set_prior(r0)
    return post


# ---------------------------------------------------------------------------
# Registry + dispatch
# ---------------------------------------------------------------------------
def _registry():
    from mixle.stats.directional.von_mises import VonMisesDistribution
    from mixle.stats.multivariate.diagonal_gaussian import DiagonalGaussianDistribution
    from mixle.stats.multivariate.multivariate_gaussian import MultivariateGaussianDistribution
    from mixle.stats.univariate.continuous.exponential import ExponentialDistribution
    from mixle.stats.univariate.continuous.gamma import GammaDistribution
    from mixle.stats.univariate.continuous.gaussian import GaussianDistribution
    from mixle.stats.univariate.continuous.half_normal import HalfNormalDistribution
    from mixle.stats.univariate.continuous.inverse_gamma import InverseGammaDistribution
    from mixle.stats.univariate.continuous.inverse_gaussian import InverseGaussianDistribution
    from mixle.stats.univariate.continuous.log_gaussian import LogGaussianDistribution
    from mixle.stats.univariate.continuous.pareto import ParetoDistribution
    from mixle.stats.univariate.continuous.rayleigh import RayleighDistribution
    from mixle.stats.univariate.discrete.bernoulli import BernoulliDistribution
    from mixle.stats.univariate.discrete.binomial import BinomialDistribution
    from mixle.stats.univariate.discrete.categorical import CategoricalDistribution
    from mixle.stats.univariate.discrete.geometric import GeometricDistribution
    from mixle.stats.univariate.discrete.integer_categorical import IntegerCategoricalDistribution
    from mixle.stats.univariate.discrete.negative_binomial import NegativeBinomialDistribution
    from mixle.stats.univariate.discrete.poisson import PoissonDistribution

    return {
        BernoulliDistribution: _build_bernoulli,
        BinomialDistribution: _build_binomial,
        GeometricDistribution: _build_geometric,
        PoissonDistribution: _build_poisson,
        ExponentialDistribution: _build_exponential,
        CategoricalDistribution: _build_categorical,
        IntegerCategoricalDistribution: _build_integer_categorical,
        GaussianDistribution: _build_gaussian,
        MultivariateGaussianDistribution: _build_mvn,
        # full closed-form (all parameters)
        RayleighDistribution: _build_rayleigh,
        HalfNormalDistribution: _build_half_normal,
        LogGaussianDistribution: _build_log_gaussian,
        DiagonalGaussianDistribution: _build_diagonal_gaussian,
        # closed-form conditional on the distribution's known shape/location/scale parameter
        GammaDistribution: _build_gamma,
        InverseGammaDistribution: _build_inverse_gamma,
        InverseGaussianDistribution: _build_inverse_gaussian,
        ParetoDistribution: _build_pareto,
        NegativeBinomialDistribution: _build_negative_binomial,
        VonMisesDistribution: _build_von_mises,
    }


# Exponential-family likelihoods whose conjugate prior has no closed form (intractable normaliser),
# so there is no closed-form full-Bayesian posterior; we raise rather than return a partial answer.
_NO_CLOSED_FORM = {
    "BetaDistribution": "both shape parameters unknown (the conjugate prior normaliser is intractable)",
    "GammaDistribution_full": "both shape and scale unknown -- only known-shape is conjugate",
    "LogSeriesDistribution": "no tractable conjugate prior",
    "IntegerMultinomialDistribution": "compositional (length distribution + count vectors); use a Dirichlet on the component directly",
}


def conjugate_posterior(dist, data, prior: dict | None = None, weights: np.ndarray | None = None) -> ConjugatePosterior:
    """Closed-form conjugate posterior over the parameters of ``dist`` given ``data``.

    Every supported family returns an exact closed-form parameter posterior: exact parameter
    samples, marginal likelihood, and posterior mean / point estimate. A posterior predictive is
    exact where its compound law is implemented; otherwise the returned distribution carries
    ``is_exact_posterior_predictive=False`` and a machine-readable ``approximation_receipt``. For
    families with a multi-parameter likelihood whose conjugate is conditional (Gamma, InverseGamma,
    InverseGaussian, Pareto, NegativeBinomial, vonMises), the *non-target* parameter (shape /
    location / scale / number-of-trials / concentration) is taken as known from ``dist`` -- exactly
    as a Binomial's number of trials is. Families with no closed-form conjugate (full Beta, full
    Gamma, LogSeries, …) raise a clear error rather than returning a partial answer.

    Args:
        dist: A mixle likelihood distribution instance whose *type* selects the conjugate family.
        data: A sequence of observations of the kind ``dist`` scores.
        prior: Optional dict of conjugate-prior hyperparameters (family specific). ``None`` uses a
            weak proper prior.
        weights: Optional per-observation weights (e.g. EM responsibilities).

    Returns:
        A :class:`ConjugatePosterior` exposing ``mean``, ``sample``, ``point_estimate``,
        ``log_marginal_likelihood``, ``posterior_predictive`` and ``summary``.
    """
    if prior is not None and not isinstance(prior, dict):
        raise TypeError("conjugate_posterior prior must be a hyperparameter dictionary or None.")
    builder = _registry().get(type(dist))
    if builder is not None:
        return builder(dist, data, weights, prior)
    name = type(dist).__name__
    reason = _NO_CLOSED_FORM.get(name)
    if reason is not None:
        raise TypeError("%s has no closed-form conjugate posterior: %s." % (name, reason))
    raise TypeError(
        "%s has no registered conjugate posterior (not a conjugate exponential family, or a "
        "structured distribution such as a mixture/HMM -- use the variational machinery instead)." % name
    )


# ---------------------------------------------------------------------------
# Mixtures of conjugate priors (Diaconis-Ylvisaker: a mixture of conjugates is conjugate)
# ---------------------------------------------------------------------------
def _weighted_parameter_mean(values: list[Any], weights: np.ndarray, name: str) -> Any:
    if all(isinstance(value, dict) for value in values):
        keys = set(values[0])
        if any(set(value) != keys for value in values[1:]):
            raise ValueError("%s dictionaries do not have matching keys." % name)
        return {
            key: _weighted_parameter_mean(
                [value[key] for value in values],
                weights,
                "%s[%r]" % (name, key),
            )
            for key in values[0]
        }
    try:
        arrays = [np.asarray(value, dtype=np.float64) for value in values]
    except (TypeError, ValueError) as exc:
        raise TypeError("%s is not numerically averageable." % name) from exc
    shape = arrays[0].shape
    if any(value.shape != shape for value in arrays[1:]):
        raise ValueError("%s values have incompatible shapes." % name)
    if any(np.any(~np.isfinite(value)) for value in arrays):
        raise ValueError("%s values must be finite to compute a mixture mean." % name)
    result = np.zeros(shape, dtype=np.float64)
    for weight, value in zip(weights, arrays):
        result += weight * value
    return float(result) if result.ndim == 0 else result


class MixtureConjugatePosterior(ConjugatePosterior):
    """Posterior under a prior that is itself a mixture of conjugate priors.

    Diaconis-Ylvisaker (1979): if the prior is ``sum_m w_m * pi_m(theta)`` with each ``pi_m``
    conjugate to the likelihood, the posterior is again a mixture of the component conjugate
    posteriors, ``sum_m w'_m * pi_m^post(theta)``, with the mixing weights reweighted by each
    component's marginal likelihood ``Z_m``:

        w'_m  proportional to  w_m * Z_m.

    Everything stays closed-form, so a prior can be multimodal (e.g. "the rate is near 1 OR near
    10") or robust (a heavy-tailed prior as a mixture of conjugates) without losing exact inference.
    """

    family = "MixtureOfConjugates"

    def __init__(self, components, post_weights, prior_weights, comp_log_evidence):
        self.components: list[ConjugatePosterior] = list(components)
        if not self.components or not all(isinstance(item, ConjugatePosterior) for item in self.components):
            raise ValueError("MixtureConjugatePosterior requires at least one conjugate-posterior component.")
        count = len(self.components)
        self.weights = _probability_vector(
            post_weights,
            count,
            "posterior mixture weights",
            normalize=False,
        )
        self.prior_weights = _probability_vector(
            prior_weights,
            count,
            "prior mixture weights",
            normalize=False,
        )
        try:
            self.comp_log_evidence = np.asarray(comp_log_evidence, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("component log evidence must be a numeric vector.") from exc
        if self.comp_log_evidence.shape != (count,):
            raise ValueError("component log evidence must have exact shape (%d,)." % count)
        if np.any(np.isnan(self.comp_log_evidence)) or np.any(np.isposinf(self.comp_log_evidence)):
            raise ValueError("component log evidence may be finite or -inf, but not NaN or +inf.")
        mean_keys = set(self.components[0].mean())
        if any(set(component.mean()) != mean_keys for component in self.components[1:]):
            raise ValueError("mixture posterior components must expose identical mean parameter keys.")

    def mean(self) -> dict[str, Any]:
        """Return the posterior-weighted mean of component posterior means."""
        out: dict[str, Any] = {}
        keys = self.components[0].mean().keys()
        for k in keys:
            vals = [c.mean()[k] for c in self.components]
            out[k] = _weighted_parameter_mean(vals, self.weights, "posterior mean parameter %r" % k)
        return out

    def sample(self, n: int = 1, rng: np.random.RandomState | None = None) -> dict[str, np.ndarray]:
        """Draw parameter samples from the posterior mixture."""
        n = _validate_sample_size(n)
        rng = rng or np.random.RandomState()
        if n == 0:
            return self.components[0].sample(0, rng)
        counts = rng.multinomial(n, self.weights)  # how many draws come from each component
        gathered: dict[str, list] = {}
        for m, cm in enumerate(counts):
            if cm == 0:
                continue
            s = self.components[m].sample(int(cm), rng)
            for key, arr in s.items():
                gathered.setdefault(key, []).append(np.asarray(arr))
        return {key: np.concatenate(parts) for key, parts in gathered.items()}

    def point_estimate(self):
        """The maximum-a-posteriori *component*'s point estimate (the dominant conjugate mode)."""
        return self.components[int(np.argmax(self.weights))].point_estimate()

    def posterior_predictive(self):
        """A mixle MixtureDistribution of the component predictives, weighted by the posterior weights."""
        from mixle.stats.latent.mixture import MixtureDistribution

        predictives = [c.posterior_predictive() for c in self.components]
        result = MixtureDistribution(predictives, list(self.weights))
        if any(getattr(item, "is_exact_posterior_predictive", True) is False for item in predictives):
            return _mark_approximate_predictive(
                result,
                self.family,
                "one or more mixture component predictives use a posterior-mean plug-in",
            )
        return _mark_exact_predictive(
            result,
            self.family,
            "mixture of exact component posterior predictives",
        )

    def log_marginal_likelihood(self) -> float:
        """Return the mixture-prior log evidence by log-summing component evidences."""
        # evidence of the whole data under the mixture prior: sum_m w_m Z_m
        log_prior = np.full(len(self.prior_weights), -np.inf, dtype=np.float64)
        positive = self.prior_weights > 0.0
        log_prior[positive] = np.log(self.prior_weights[positive])
        return float(logsumexp(log_prior + self.comp_log_evidence))

    def hyper(self) -> dict[str, Any]:
        """Return posterior weights and component hyperparameters."""
        return {"weights": self.weights, "components": [c.hyper() for c in self.components]}


def mixture_conjugate_posterior(
    dist,
    data,
    priors: list[dict],
    prior_weights: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> MixtureConjugatePosterior:
    """Posterior under a prior that is a mixture of conjugate priors (Diaconis-Ylvisaker).

    Args:
        dist: The likelihood distribution (must map to a *closed-form* conjugate realiser, since the
            reweighting needs each component's marginal likelihood).
        data: The observations.
        priors: One hyperparameter dict per mixture component (same keys :func:`conjugate_posterior`
            accepts for this family).
        prior_weights: Prior mixing weights ``w_m`` (default uniform); normalised internally.
        weights: Optional per-observation weights.

    Returns:
        A :class:`MixtureConjugatePosterior`: the exact posterior, again a mixture of conjugate
        posteriors with weights ``w'_m proportional to w_m * Z_m``.
    """
    m = len(priors)
    if m == 0:
        raise ValueError("need at least one prior component")
    if not all(isinstance(prior, dict) for prior in priors):
        raise TypeError("mixture conjugate priors must be dictionaries.")
    pw = (
        np.ones(m, dtype=np.float64) / m
        if prior_weights is None
        else _probability_vector(prior_weights, m, "prior mixture weights", normalize=True)
    )
    materialized_data = list(data)
    materialized_weights = None if weights is None else list(weights)
    components = [
        conjugate_posterior(
            dist,
            materialized_data,
            prior=pr,
            weights=materialized_weights,
        )
        for pr in priors
    ]
    try:
        log_evidence = np.asarray(
            [c.log_marginal_likelihood() for c in components],
            dtype=np.float64,
        )
    except NotImplementedError as exc:
        raise TypeError(
            "mixture_conjugate_posterior needs a closed-form family (it reweights by each "
            "component's marginal likelihood); %s has only the generic posterior." % type(dist).__name__
        ) from exc
    if log_evidence.shape != (m,) or np.any(np.isnan(log_evidence)) or np.any(np.isposinf(log_evidence)):
        raise ValueError("component log evidence must be finite or -inf.")
    log_prior_weights = np.full(m, -np.inf, dtype=np.float64)
    positive = pw > 0.0
    log_prior_weights[positive] = np.log(pw[positive])
    log_post = log_prior_weights + log_evidence
    normalizer = float(logsumexp(log_post))
    if not math.isfinite(normalizer):
        raise ValueError("all conjugate-mixture components assign zero evidence to the observations.")
    post_weights = np.exp(log_post - normalizer)
    return MixtureConjugatePosterior(components, post_weights, pw, log_evidence)
