"""Generalized Gaussian (exponential-power) distribution.

A symmetric location-scale family with a tunable tail/peakedness shape ``beta`` that interpolates the
Laplace (``beta = 1``), Gaussian (``beta = 2``), and uniform (``beta -> inf``) laws. With location
``mu``, scale ``alpha > 0`` and shape ``beta > 0``,

    f(x; mu, alpha, beta) = beta / (2 alpha Gamma(1/beta)) * exp(-(|x - mu| / alpha)^beta).

The normalizer is closed form (a Gamma function), so density/CDF/quantile/moments/entropy are all exact;
it samples exactly via a Gamma draw with a random sign. Parameters are fit by the method of moments:
``mu`` is the mean, the excess kurtosis ``Gamma(5/beta)Gamma(1/beta)/Gamma(3/beta)^2 - 3`` pins ``beta``
(monotone, solved by a bracketed root find), and ``alpha`` follows from the variance
``alpha^2 Gamma(3/beta)/Gamma(1/beta)``.

References:
  - Nadarajah, "A generalized normal distribution", *J. Applied Statistics* 32 (2005).
  - Subbotin (1923), the original exponential-power family.
"""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import RandomState
from scipy.special import gamma, gammainc, gammaincinv, gammaln

from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)


def _excess_kurtosis(beta: float) -> float:
    """Excess kurtosis of the exponential-power law as a function of the shape ``beta``."""
    return float(gamma(5.0 / beta) * gamma(1.0 / beta) / gamma(3.0 / beta) ** 2 - 3.0)


class GeneralizedGaussianDistribution(SequenceEncodableProbabilityDistribution):
    """Generalized Gaussian (exponential power) with location ``mu``, scale ``alpha`` and shape ``beta``."""

    def __init__(self, mu: float, alpha: float, beta: float, name: str | None = None, keys: str | None = None) -> None:
        if alpha <= 0.0 or not np.isfinite(alpha):
            raise ValueError("GeneralizedGaussianDistribution requires finite alpha > 0.")
        if beta <= 0.0 or not np.isfinite(beta):
            raise ValueError("GeneralizedGaussianDistribution requires finite beta > 0.")
        self.mu = float(mu)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.name = name
        self.keys = keys
        self._log_norm = math.log(self.beta) - math.log(2.0 * self.alpha) - gammaln(1.0 / self.beta)

    def __str__(self) -> str:
        return "GeneralizedGaussianDistribution(%s, %s, %s, name=%s, keys=%s)" % (
            repr(self.mu),
            repr(self.alpha),
            repr(self.beta),
            repr(self.name),
            repr(self.keys),
        )

    @classmethod
    def compute_declaration(cls):
        """Return the structured compute declaration for generalized Gaussian distributions."""
        from mixle.stats.compute.declarations import DistributionDeclaration, ParameterSpec, StatisticSpec

        # declaring the engine-neutral density lets the symbolic->numba lowering compile a scalar kernel
        # for this non-exponential-family leaf (parity with Laplace/Logistic/Weibull/...).
        return DistributionDeclaration(
            name="generalized_gaussian",
            distribution_type=cls,
            parameters=(
                ParameterSpec("mu"),
                ParameterSpec("alpha", constraint="positive"),
                ParameterSpec("beta", constraint="positive"),
            ),
            statistics=(
                StatisticSpec("values", kind="raw_observations", scales=False),
                StatisticSpec("weights", kind="weights"),
            ),
            support="real",
        )

    @staticmethod
    def backend_log_density_from_params(x: Any, mu: Any, alpha: Any, beta: Any, engine: Any) -> Any:
        """Engine-neutral generalized-Gaussian log-density: log_norm - (|x-mu|/alpha)**beta."""
        log_norm = (
            engine.log(beta) - engine.log(engine.asarray(2.0) * alpha) - engine.gammaln(engine.asarray(1.0) / beta)
        )
        return log_norm - (engine.abs(x - mu) / alpha) ** beta

    def density(self, x: float) -> float:
        """Return the probability density at ``x``."""
        return math.exp(self.log_density(x))

    def log_density(self, x: float) -> float:
        """Return the log-density at ``x``."""
        return self._log_norm - (abs(float(x) - self.mu) / self.alpha) ** self.beta

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Return vectorized log-density for a sequence-encoded array of observations."""
        z = np.abs(np.asarray(x, dtype=np.float64) - self.mu) / self.alpha
        return self._log_norm - z**self.beta

    def cdf(self, x: float) -> float:
        """Cumulative distribution function P(X <= x)."""
        xv = float(x) - self.mu
        z = (abs(xv) / self.alpha) ** self.beta
        return float(0.5 + math.copysign(0.5 * gammainc(1.0 / self.beta, z), xv))

    def quantile(self, q: float) -> float:
        """Inverse CDF F^{-1}(q)."""
        qv = float(q) - 0.5
        z = gammaincinv(1.0 / self.beta, 2.0 * abs(qv))
        return float(self.mu + math.copysign(self.alpha * z ** (1.0 / self.beta), qv))

    def mean(self) -> float:
        """Mean (the location ``mu``)."""
        return self.mu

    def variance(self) -> float:
        """Variance alpha^2 Gamma(3/beta) / Gamma(1/beta)."""
        return float(self.alpha * self.alpha * gamma(3.0 / self.beta) / gamma(1.0 / self.beta))

    def skewness(self) -> float:
        """Skewness (0 -- the law is symmetric)."""
        return 0.0

    def kurtosis(self) -> float:
        """Excess kurtosis Gamma(5/beta)Gamma(1/beta)/Gamma(3/beta)^2 - 3."""
        return _excess_kurtosis(self.beta)

    def entropy(self) -> float:
        """Differential entropy 1/beta - log(beta / (2 alpha Gamma(1/beta)))."""
        return float(1.0 / self.beta - self._log_norm)

    def sampler(self, seed: int | None = None) -> "GeneralizedGaussianSampler":
        """Return a sampler (Gamma magnitude with a random sign)."""
        return GeneralizedGaussianSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "GeneralizedGaussianEstimator":
        """Return a method-of-moments estimator for ``mu``, ``alpha``, ``beta``."""
        return GeneralizedGaussianEstimator(name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> "GeneralizedGaussianDataEncoder":
        """Return the data encoder used by this distribution (the raw value)."""
        return GeneralizedGaussianDataEncoder()


class GeneralizedGaussianSampler(DistributionSampler):
    """Draw ``x = mu + sign * alpha * Gamma(1/beta)**(1/beta)``."""

    def __init__(self, dist: GeneralizedGaussianDistribution, seed: int | None = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: int | None = None) -> float | np.ndarray:
        """Draw one sample or an array of iid samples."""
        d = self.dist
        n = 1 if size is None else int(size)
        g = self.rng.gamma(1.0 / d.beta, 1.0, size=n)
        sign = self.rng.randint(0, 2, size=n) * 2 - 1
        x = d.mu + sign * d.alpha * g ** (1.0 / d.beta)
        return float(x[0]) if size is None else x


class GeneralizedGaussianAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate the weighted power sums ``(count, sum x, sum x^2, sum x^3, sum x^4)`` for the MoM."""

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.count = 0.0
        self.s1 = 0.0
        self.s2 = 0.0
        self.s3 = 0.0
        self.s4 = 0.0
        self.name = name
        self.keys = keys

    def update(self, x: float, weight: float, estimate: GeneralizedGaussianDistribution | None) -> None:
        """Accumulate weighted raw moments up to order four for one observation."""
        xv = float(x)
        self.count += weight
        self.s1 += weight * xv
        self.s2 += weight * xv**2
        self.s3 += weight * xv**3
        self.s4 += weight * xv**4

    def initialize(self, x: float, weight: float, rng: RandomState | None) -> None:
        """Initialize statistics from one observation."""
        self.update(x, weight, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: Any) -> None:
        """Accumulate weighted raw moments up to order four from encoded data."""
        xv = np.asarray(x, dtype=np.float64)
        w = np.asarray(weights, dtype=np.float64)
        self.count += float(w.sum())
        self.s1 += float(np.dot(w, xv))
        self.s2 += float(np.dot(w, xv**2))
        self.s3 += float(np.dot(w, xv**3))
        self.s4 += float(np.dot(w, xv**4))

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize statistics from encoded observations."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[float, float, float, float, float]) -> "GeneralizedGaussianAccumulator":
        """Merge another generalized-Gaussian sufficient-statistic tuple."""
        self.count += suff_stat[0]
        self.s1 += suff_stat[1]
        self.s2 += suff_stat[2]
        self.s3 += suff_stat[3]
        self.s4 += suff_stat[4]
        return self

    def value(self) -> tuple[float, float, float, float, float]:
        """Return count and raw moment sums through order four."""
        return self.count, self.s1, self.s2, self.s3, self.s4

    def from_value(self, x: tuple[float, float, float, float, float]) -> "GeneralizedGaussianAccumulator":
        """Replace accumulator contents from a sufficient-statistic tuple."""
        self.count, self.s1, self.s2, self.s3, self.s4 = (float(v) for v in x)
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Merge keyed statistics into ``stats_dict`` when keys are configured."""
        if self.keys is not None:
            if self.keys in stats_dict:
                stats_dict[self.keys].combine(self.value())
            else:
                stats_dict[self.keys] = self

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Replace this accumulator from keyed statistics when available."""
        if self.keys is not None and self.keys in stats_dict:
            self.from_value(stats_dict[self.keys].value())

    def acc_to_encoder(self) -> "GeneralizedGaussianDataEncoder":
        """Return the encoder used by this accumulator."""
        return GeneralizedGaussianDataEncoder()


class GeneralizedGaussianAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for GeneralizedGaussianAccumulator."""

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.name = name
        self.keys = keys

    def make(self) -> GeneralizedGaussianAccumulator:
        """Create a fresh generalized-Gaussian accumulator."""
        return GeneralizedGaussianAccumulator(name=self.name, keys=self.keys)


class GeneralizedGaussianEstimator(ParameterEstimator):
    """Method-of-moments estimator: ``mu`` = mean, ``beta`` from excess kurtosis, ``alpha`` from variance."""

    def __init__(
        self, beta_bounds: tuple[float, float] = (0.25, 50.0), name: str | None = None, keys: str | None = None
    ) -> None:
        self.beta_bounds = beta_bounds
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> GeneralizedGaussianAccumulatorFactory:
        """Return an accumulator factory for generalized-Gaussian raw moments."""
        return GeneralizedGaussianAccumulatorFactory(name=self.name, keys=self.keys)

    def estimate(
        self, nobs: float | None, suff_stat: tuple[float, float, float, float, float]
    ) -> GeneralizedGaussianDistribution:
        """Estimate location, scale, and shape from weighted raw moments."""
        from scipy.optimize import brentq

        count, s1, s2, s3, s4 = suff_stat
        if count <= 0.0:
            return GeneralizedGaussianDistribution(0.0, 1.0, 2.0, name=self.name, keys=self.keys)
        mu = s1 / count
        m2 = s2 / count - mu * mu
        if m2 <= 0.0:
            return GeneralizedGaussianDistribution(mu, 1.0e-6, 2.0, name=self.name, keys=self.keys)
        m4 = s4 / count - 4.0 * mu * (s3 / count) + 6.0 * mu * mu * (s2 / count) - 3.0 * mu**4
        k = m4 / (m2 * m2) - 3.0  # sample excess kurtosis
        lo, hi = self.beta_bounds
        k_lo, k_hi = _excess_kurtosis(lo), _excess_kurtosis(hi)  # k decreases as beta grows
        if k >= k_lo:
            beta = lo
        elif k <= k_hi:
            beta = hi
        else:
            beta = float(brentq(lambda b: _excess_kurtosis(b) - k, lo, hi, xtol=1.0e-8))
        alpha = math.sqrt(m2 * gamma(1.0 / beta) / gamma(3.0 / beta))
        return GeneralizedGaussianDistribution(mu, alpha, beta, name=self.name, keys=self.keys)


class GeneralizedGaussianDataEncoder(DataSequenceEncoder):
    """Encode observations as a float array."""

    def __str__(self) -> str:
        return "GeneralizedGaussianDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, GeneralizedGaussianDataEncoder)

    def seq_encode(self, x: Sequence[float]) -> np.ndarray:
        """Encode observations as a floating-point array."""
        return np.asarray(x, dtype=np.float64)
