"""Evaluate, estimate, and sample from an exponentially-modified Gaussian (EMG) distribution.

Defines the ExponentiallyModifiedGaussianDistribution, ExponentiallyModifiedGaussianSampler,
ExponentiallyModifiedGaussianAccumulatorFactory, ExponentiallyModifiedGaussianAccumulator,
ExponentiallyModifiedGaussianEstimator, and ExponentiallyModifiedGaussianDataEncoder classes
for use with pysparkplug.

Data type: (float): The EMG models ``X = N(mu, sigma2) + Exp(rate=lam)`` -- a Gaussian convolved
    with a (positive-shifting) exponential, giving a positively right-skewed real-valued density.
    Its stable log-density is

        log f(x) = log(lam/2) - 0.5*u^2 + log(erfcx(z)),

    where ``u = (x - mu)/sigma`` and ``z = (lam*sigma - u)/sqrt(2)`` and ``sigma = sqrt(sigma2)``.
    Using ``log_erfcx`` keeps the right tail (large ``z``) from underflowing.

The MLE has no closed form (the score equations couple mu, sigma2, lam), so the estimator uses a
method-of-moments fit from the accumulated first three moments, which is consistent and the usual
practical choice for the EMG.
"""

import math
from collections.abc import Callable
from typing import Any, Optional

import numpy as np
from numpy.random import RandomState

from pysp.arithmetic import *
from pysp.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)
from pysp.utils.special import log_erfcx

_MIN_EMG_PARAM = 1.0e-12


class ExponentiallyModifiedGaussianDistribution(SequenceEncodableProbabilityDistribution):
    """Exponentially-modified Gaussian: ``X = N(mu, sigma2) + Exp(rate=lam)``."""

    def __init__(
        self,
        mu: float,
        sigma2: float,
        lam: float,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        """Create an EMG with Gaussian mean ``mu``, Gaussian variance ``sigma2`` and exponential rate ``lam``.

        Args:
            mu (float): Real-valued mean of the Gaussian component.
            sigma2 (float): Positive variance of the Gaussian component.
            lam (float): Positive rate of the exponential component (its mean is ``1/lam``).
            name (Optional[str]): Optional object name.
            keys (Optional[str]): Optional parameter key.

        Attributes:
            mu (float): Gaussian mean.
            sigma2 (float): Gaussian variance.
            sigma (float): Gaussian standard deviation.
            lam (float): Exponential rate.

        """
        if not np.isfinite(mu):
            raise ValueError("ExponentiallyModifiedGaussianDistribution requires finite mu.")
        if sigma2 <= 0.0 or not np.isfinite(sigma2):
            raise ValueError("ExponentiallyModifiedGaussianDistribution requires finite sigma2 > 0.")
        if lam <= 0.0 or not np.isfinite(lam):
            raise ValueError("ExponentiallyModifiedGaussianDistribution requires finite lam > 0.")
        self.mu = float(mu)
        self.sigma2 = float(sigma2)
        self.sigma = float(math.sqrt(sigma2))
        self.lam = float(lam)
        self.name = name
        self.keys = keys
        self.log_lam_half = math.log(self.lam / 2.0)

    def __str__(self) -> str:
        """Returns string representation of ExponentiallyModifiedGaussianDistribution object."""
        return "ExponentiallyModifiedGaussianDistribution(%s, %s, %s, name=%s, keys=%s)" % (
            repr(self.mu),
            repr(self.sigma2),
            repr(self.lam),
            repr(self.name),
            repr(self.keys),
        )

    def density(self, x: float) -> float:
        """Density of the EMG at observation ``x`` (see ``log_density``)."""
        return math.exp(self.log_density(x))

    def log_density(self, x: float) -> float:
        """Stable log-density of the EMG at ``x``.

        ``log f(x) = log(lam/2) - 0.5*u^2 + log_erfcx(z)`` with ``u = (x - mu)/sigma`` and
        ``z = (lam*sigma - u)/sqrt(2)``.
        """
        u = (x - self.mu) / self.sigma
        z = (self.lam * self.sigma - u) / math.sqrt(2.0)
        return self.log_lam_half - 0.5 * u * u + float(log_erfcx(z))

    def seq_ld_lambda(self) -> list[Callable]:
        """Return vectorized log-density callables for encoded data."""
        return [self.seq_log_density]

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Vectorized EMG log-density at sequence-encoded input ``x``."""
        xx = np.asarray(x, dtype=np.float64)
        u = (xx - self.mu) / self.sigma
        z = (self.lam * self.sigma - u) / math.sqrt(2.0)
        return self.log_lam_half - 0.5 * u * u + np.asarray(log_erfcx(z), dtype=np.float64)

    def cdf(self, x: float) -> float:
        """Cumulative distribution function ``P(X <= x)`` (exact, via scipy's exponnorm)."""
        from scipy.stats import exponnorm

        return float(exponnorm.cdf(x, 1.0 / (self.lam * self.sigma), loc=self.mu, scale=self.sigma))

    def quantile(self, q: float) -> float:
        """Inverse CDF ``F^{-1}(q)``."""
        from scipy.stats import exponnorm

        return float(exponnorm.ppf(q, 1.0 / (self.lam * self.sigma), loc=self.mu, scale=self.sigma))

    def sampler(self, seed: int | None = None) -> "ExponentiallyModifiedGaussianSampler":
        """Return an ExponentiallyModifiedGaussianSampler for this distribution."""
        return ExponentiallyModifiedGaussianSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "ExponentiallyModifiedGaussianEstimator":
        """Return an ExponentiallyModifiedGaussianEstimator (method-of-moments)."""
        return ExponentiallyModifiedGaussianEstimator(name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> "ExponentiallyModifiedGaussianDataEncoder":
        """Returns an ExponentiallyModifiedGaussianDataEncoder object."""
        return ExponentiallyModifiedGaussianDataEncoder()


class ExponentiallyModifiedGaussianSampler(DistributionSampler):
    def __init__(self, dist: ExponentiallyModifiedGaussianDistribution, seed: int | None = None) -> None:
        """Sampler: draw a Gaussian and add an independent Exponential.

        Args:
            dist (ExponentiallyModifiedGaussianDistribution): Distribution to sample from.
            seed (Optional[int]): Seed for the random sampler.

        """
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: int | None = None) -> float | np.ndarray:
        """Draw ``size`` iid EMG samples (a single float if ``size`` is None)."""
        d = self.dist
        normal = self.rng.normal(loc=d.mu, scale=d.sigma, size=size)
        expo = self.rng.exponential(scale=1.0 / d.lam, size=size)
        return normal + expo


class ExponentiallyModifiedGaussianAccumulator(SequenceEncodableStatisticAccumulator):
    def __init__(self, keys: str | None = None, name: str | None = None) -> None:
        """Accumulate the first three (weighted) moments needed for the MoM fit.

        Attributes:
            sum (float): sum_i w_i*x_i.
            sum2 (float): sum_i w_i*x_i^2.
            sum3 (float): sum_i w_i*x_i^3.
            count (float): sum_i w_i.

        """
        self.sum = 0.0
        self.sum2 = 0.0
        self.sum3 = 0.0
        self.count = 0.0
        self.keys = keys
        self.name = name

    def update(self, x: float, weight: float, estimate: Optional["ExponentiallyModifiedGaussianDistribution"]) -> None:
        xw = x * weight
        self.sum += xw
        self.sum2 += x * xw
        self.sum3 += x * x * xw
        self.count += weight

    def initialize(self, x: float, weight: float, rng: RandomState | None) -> None:
        self.update(x, weight, None)

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        self.seq_update(x, weights, None)

    def seq_update(
        self, x: np.ndarray, weights: np.ndarray, estimate: ExponentiallyModifiedGaussianDistribution | None
    ) -> None:
        xx = np.asarray(x, dtype=np.float64)
        ww = np.asarray(weights, dtype=np.float64)
        self.sum += np.dot(xx, ww)
        self.sum2 += np.dot(xx * xx, ww)
        self.sum3 += np.dot(xx * xx * xx, ww)
        self.count += ww.sum()

    def combine(self, suff_stat: tuple[float, float, float, float]) -> "ExponentiallyModifiedGaussianAccumulator":
        self.sum += suff_stat[0]
        self.sum2 += suff_stat[1]
        self.sum3 += suff_stat[2]
        self.count += suff_stat[3]
        return self

    def value(self) -> tuple[float, float, float, float]:
        return self.sum, self.sum2, self.sum3, self.count

    def from_value(self, x: tuple[float, float, float, float]) -> "ExponentiallyModifiedGaussianAccumulator":
        self.sum, self.sum2, self.sum3, self.count = x
        return self

    def scale(self, c: float) -> "ExponentiallyModifiedGaussianAccumulator":
        self.sum *= c
        self.sum2 *= c
        self.sum3 *= c
        self.count *= c
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        if self.keys is not None:
            if self.keys in stats_dict:
                s0, s1, s2, c = stats_dict[self.keys]
                self.sum += s0
                self.sum2 += s1
                self.sum3 += s2
                self.count += c
            else:
                stats_dict[self.keys] = (self.sum, self.sum2, self.sum3, self.count)

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        if self.keys is not None and self.keys in stats_dict:
            self.sum, self.sum2, self.sum3, self.count = stats_dict[self.keys]

    def acc_to_encoder(self) -> "ExponentiallyModifiedGaussianDataEncoder":
        return ExponentiallyModifiedGaussianDataEncoder()


class ExponentiallyModifiedGaussianAccumulatorFactory(StatisticAccumulatorFactory):
    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.keys = keys
        self.name = name

    def make(self) -> "ExponentiallyModifiedGaussianAccumulator":
        return ExponentiallyModifiedGaussianAccumulator(name=self.name, keys=self.keys)


class ExponentiallyModifiedGaussianEstimator(ParameterEstimator):
    def __init__(
        self,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        """Method-of-moments EMG estimator.

        The EMG MLE is iterative with no closed form (the score equations couple ``mu``, ``sigma2``
        and ``lam``). This estimator uses the consistent method-of-moments fit from the first three
        central moments: with sample variance ``v`` and skewness ``g``, set the exponential mean
        ``tau = 1/lam = (g*v^{3/2}/2)^{1/3}``, then ``mu = mean - tau`` and ``sigma2 = v - tau^2``.
        Degenerate moments (non-positive skew or ``tau^2 >= v``) fall back to a small positive
        exponential component so the result stays a valid EMG.
        """
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> "ExponentiallyModifiedGaussianAccumulatorFactory":
        return ExponentiallyModifiedGaussianAccumulatorFactory(self.name, self.keys)

    def estimate(
        self, nobs: float | None, suff_stat: tuple[float, float, float, float]
    ) -> "ExponentiallyModifiedGaussianDistribution":
        """Estimate an EMG from the accumulated (sum, sum2, sum3, count) via method of moments."""
        s1, s2, s3, n = suff_stat

        if n <= 0.0:
            return ExponentiallyModifiedGaussianDistribution(0.0, 1.0, 1.0, name=self.name, keys=self.keys)

        m1 = s1 / n
        m2 = s2 / n
        m3 = s3 / n
        var = m2 - m1 * m1
        if var <= _MIN_EMG_PARAM or not np.isfinite(var):
            var = _MIN_EMG_PARAM

        # third central moment and skewness
        mu3 = m3 - 3.0 * m1 * m2 + 2.0 * m1 * m1 * m1
        skew = mu3 / (var**1.5)

        if skew > _MIN_EMG_PARAM:
            tau = (0.5 * skew) ** (1.0 / 3.0) * math.sqrt(var)
        else:
            # near-symmetric data: give the exponential component a small positive share
            tau = math.sqrt(var) * 1e-3

        # keep a strictly positive Gaussian variance
        sigma2 = var - tau * tau
        if sigma2 <= _MIN_EMG_PARAM or not np.isfinite(sigma2):
            sigma2 = max(var * 0.5, _MIN_EMG_PARAM)
            tau = math.sqrt(max(var - sigma2, _MIN_EMG_PARAM))

        if tau < _MIN_EMG_PARAM:
            tau = _MIN_EMG_PARAM
        lam = 1.0 / tau
        mu = m1 - tau

        return ExponentiallyModifiedGaussianDistribution(mu, sigma2, lam, name=self.name, keys=self.keys)


class ExponentiallyModifiedGaussianDataEncoder(DataSequenceEncoder):
    """Encode sequences of iid EMG observations (data type float)."""

    def __str__(self) -> str:
        return "ExponentiallyModifiedGaussianDataEncoder"

    def __eq__(self, other) -> bool:
        return isinstance(other, ExponentiallyModifiedGaussianDataEncoder)

    def seq_encode(self, x: list[float] | np.ndarray) -> np.ndarray:
        rv = np.asarray(x, dtype=float)
        if np.any(np.isnan(rv)) or np.any(np.isinf(rv)):
            raise Exception("ExponentiallyModifiedGaussianDistribution requires support x in (-inf, inf).")
        return rv
