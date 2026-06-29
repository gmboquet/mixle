"""Evaluate, estimate, and sample from a Skellam distribution (the difference of two Poissons).

Defines the SkellamDistribution, SkellamSampler, SkellamAccumulatorFactory, SkellamAccumulator,
SkellamEstimator, and SkellamDataEncoder classes for use with mixle.

Data type (int): ``K = N1 - N2`` with ``N1 ~ Poisson(mu1)``, ``N2 ~ Poisson(mu2)`` independent, so
    ``K`` ranges over all integers (negative, zero, positive). Its log-mass is

        log p(k) = -(sqrt(mu1) - sqrt(mu2))^2 + (k/2) * log(mu1/mu2) + log(I_|k|(2*sqrt(mu1*mu2))),

    where ``I_v`` is the modified Bessel function of the first kind. The exponentially-scaled
    ``ive(v, z) = I_v(z) * exp(-z)`` is used (``log I_v(z) = log(ive(v, z)) + z``) so the Bessel
    term does not overflow for large ``z``; combined with ``-(mu1+mu2) + z`` this collapses to the
    stable ``-(sqrt(mu1) - sqrt(mu2))^2`` constant above.

The MLE has no closed form, but the method of moments is exact and closed-form here: with sample
mean ``m`` and variance ``v``, ``mu1 = (v + m)/2`` and ``mu2 = (v - m)/2`` (since ``E[K] = mu1-mu2``
and ``Var[K] = mu1+mu2``), which the estimator uses (clamped to keep both rates positive).


Reference: Skellam, 'The frequency distribution of the difference between two Poisson variates', JRSS A (1946).
"""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import RandomState

from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)
from mixle.utils.special import valid_integer

_MIN_SKELLAM_RATE = 1.0e-12


class SkellamDistribution(SequenceEncodableProbabilityDistribution):
    """Skellam distribution: ``K = N1 - N2`` for independent ``N1 ~ Poisson(mu1)``, ``N2 ~ Poisson(mu2)``."""

    def __init__(self, mu1: float, mu2: float, name: str | None = None, keys: str | None = None) -> None:
        """Create a Skellam with component Poisson rates ``mu1`` and ``mu2``.

        Args:
            mu1 (float): Positive rate of the additive Poisson component ``N1``.
            mu2 (float): Positive rate of the subtractive Poisson component ``N2``.
            name (Optional[str]): Optional object name.
            keys (Optional[str]): Optional parameter key.

        Attributes:
            mu1 (float): Rate of ``N1``.
            mu2 (float): Rate of ``N2``.
            log_ratio_half (float): Cached ``0.5 * (log(mu1) - log(mu2))``.
            sqrt_diff_sq (float): Cached ``(sqrt(mu1) - sqrt(mu2))**2``.
            two_sqrt_prod (float): Cached ``2 * sqrt(mu1 * mu2)`` (the Bessel argument).

        """
        if mu1 <= 0.0 or not np.isfinite(mu1):
            raise ValueError("SkellamDistribution requires finite mu1 > 0.")
        if mu2 <= 0.0 or not np.isfinite(mu2):
            raise ValueError("SkellamDistribution requires finite mu2 > 0.")
        self.mu1 = float(mu1)
        self.mu2 = float(mu2)
        self.name = name
        self.keys = keys
        self.log_ratio_half = 0.5 * (math.log(self.mu1) - math.log(self.mu2))
        self.sqrt_diff_sq = (math.sqrt(self.mu1) - math.sqrt(self.mu2)) ** 2
        self.two_sqrt_prod = 2.0 * math.sqrt(self.mu1 * self.mu2)

    def __str__(self) -> str:
        """Returns string representation of SkellamDistribution object."""
        return "SkellamDistribution(%s, %s, name=%s, keys=%s)" % (
            repr(self.mu1),
            repr(self.mu2),
            repr(self.name),
            repr(self.keys),
        )

    def density(self, x: int) -> float:
        """Probability mass at integer ``x`` (see ``log_density``)."""
        return math.exp(self.log_density(x))

    def log_density(self, x: int) -> float:
        """Stable Skellam log-mass at integer ``x`` (``-inf`` for non-integer input)."""
        from scipy.special import ive

        if not valid_integer(x, nonneg=False):
            return -np.inf
        k = float(x)
        bessel = float(ive(abs(k), self.two_sqrt_prod))
        if bessel <= 0.0:
            return -np.inf
        return -self.sqrt_diff_sq + k * self.log_ratio_half + math.log(bessel)

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Vectorized Skellam log-mass at sequence-encoded integer counts ``x``."""
        from scipy.special import ive

        kk = np.asarray(x, dtype=np.float64)
        bessel = np.asarray(ive(np.abs(kk), self.two_sqrt_prod), dtype=np.float64)
        with np.errstate(divide="ignore"):
            log_bessel = np.log(bessel)
        return -self.sqrt_diff_sq + kk * self.log_ratio_half + log_bessel

    def mean(self) -> float:
        """Mean E[X] = mu1 - mu2."""
        return float(self.mu1 - self.mu2)

    def variance(self) -> float:
        """Variance Var[X] = mu1 + mu2."""
        return float(self.mu1 + self.mu2)

    def cdf(self, x: float) -> float:
        """Cumulative distribution function P(X <= x) (via scipy skellam)."""
        import math

        from scipy.stats import skellam

        return float(skellam.cdf(math.floor(float(x)), self.mu1, self.mu2))

    def quantile(self, q: float) -> float:
        """Inverse CDF F^{-1}(q) (via scipy skellam)."""
        from scipy.stats import skellam

        return float(skellam.ppf(float(q), self.mu1, self.mu2))

    def sampler(self, seed: int | None = None) -> "SkellamSampler":
        """Return a SkellamSampler for this distribution."""
        return SkellamSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "SkellamEstimator":
        """Return a SkellamEstimator (method of moments)."""
        return SkellamEstimator(name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> "SkellamDataEncoder":
        """Returns a SkellamDataEncoder object."""
        return SkellamDataEncoder()


class SkellamSampler(DistributionSampler):
    """Draw iid Skellam observations as the difference of two independent Poisson draws."""

    def __init__(self, dist: SkellamDistribution, seed: int | None = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: int | None = None) -> int | np.ndarray:
        """Draw ``size`` iid Skellam samples (a single int if ``size`` is None)."""
        n1 = self.rng.poisson(lam=self.dist.mu1, size=size)
        n2 = self.rng.poisson(lam=self.dist.mu2, size=size)
        rv = n1 - n2
        return int(rv) if size is None else rv.astype(np.int64)


class SkellamAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate the weighted count, sum, and sum-of-squares needed for the moment fit."""

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.count = 0.0
        self.sum = 0.0
        self.sum2 = 0.0
        self.name = name
        self.keys = keys

    def update(self, x: int, weight: float, estimate: SkellamDistribution | None) -> None:
        if not valid_integer(x, nonneg=False):
            raise ValueError("SkellamDistribution requires integer observations.")
        xw = float(x) * weight
        self.count += weight
        self.sum += xw
        self.sum2 += float(x) * xw

    def initialize(self, x: int, weight: float, rng: RandomState | None) -> None:
        self.update(x, weight, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: SkellamDistribution | None) -> None:
        xx = np.asarray(x, dtype=np.float64)
        ww = np.asarray(weights, dtype=np.float64)
        self.count += ww.sum()
        self.sum += np.dot(xx, ww)
        self.sum2 += np.dot(xx * xx, ww)

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[float, float, float]) -> "SkellamAccumulator":
        self.count += suff_stat[0]
        self.sum += suff_stat[1]
        self.sum2 += suff_stat[2]
        return self

    def value(self) -> tuple[float, float, float]:
        return self.count, self.sum, self.sum2

    def from_value(self, x: tuple[float, float, float]) -> "SkellamAccumulator":
        self.count, self.sum, self.sum2 = x
        return self

    def scale(self, c: float) -> "SkellamAccumulator":
        self.count *= c
        self.sum *= c
        self.sum2 *= c
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        if self.keys is not None:
            if self.keys in stats_dict:
                c, s, s2 = stats_dict[self.keys]
                self.count += c
                self.sum += s
                self.sum2 += s2
            else:
                stats_dict[self.keys] = (self.count, self.sum, self.sum2)

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        if self.keys is not None and self.keys in stats_dict:
            self.count, self.sum, self.sum2 = stats_dict[self.keys]

    def acc_to_encoder(self) -> "SkellamDataEncoder":
        return SkellamDataEncoder()


class SkellamAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for SkellamAccumulator."""

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.name = name
        self.keys = keys

    def make(self) -> "SkellamAccumulator":
        return SkellamAccumulator(name=self.name, keys=self.keys)


class SkellamEstimator(ParameterEstimator):
    """Estimate ``(mu1, mu2)`` by the (exact, closed-form) method of moments."""

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        """Method-of-moments Skellam estimator.

        ``E[K] = mu1 - mu2`` and ``Var[K] = mu1 + mu2``, so ``mu1 = (v + m)/2`` and
        ``mu2 = (v - m)/2`` for sample mean ``m`` and variance ``v``. Both rates are clamped to a
        small positive floor (a near-degenerate sample can drive one rate non-positive).
        """
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> "SkellamAccumulatorFactory":
        return SkellamAccumulatorFactory(name=self.name, keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat: tuple[float, float, float]) -> "SkellamDistribution":
        """Estimate a Skellam from the accumulated ``(count, sum, sum2)`` via method of moments."""
        count, xsum, xsum2 = suff_stat
        if count <= 0.0:
            return SkellamDistribution(1.0, 1.0, name=self.name, keys=self.keys)
        mean = xsum / count
        var = xsum2 / count - mean * mean
        # Var must dominate |mean| for both rates to stay positive; floor it so the fit is valid.
        if not np.isfinite(var) or var < abs(mean) + _MIN_SKELLAM_RATE:
            var = abs(mean) + _MIN_SKELLAM_RATE
        mu1 = max(0.5 * (var + mean), _MIN_SKELLAM_RATE)
        mu2 = max(0.5 * (var - mean), _MIN_SKELLAM_RATE)
        return SkellamDistribution(mu1, mu2, name=self.name, keys=self.keys)


class SkellamDataEncoder(DataSequenceEncoder):
    """Encode sequences of iid Skellam observations (integer data type, any sign)."""

    def __str__(self) -> str:
        return "SkellamDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SkellamDataEncoder)

    def seq_encode(self, x: Sequence[int]) -> np.ndarray:
        rv = np.asarray(x, dtype=np.float64)
        if rv.size and (np.any(np.isnan(rv)) or np.any(np.isinf(rv)) or np.any(np.floor(rv) != rv)):
            raise ValueError("SkellamDistribution requires integer observations.")
        return rv
