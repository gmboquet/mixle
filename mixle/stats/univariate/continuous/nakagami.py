"""Nakagami distribution -- the amplitude/envelope law of Nakagami-m fading.

A positive-support family for signal envelopes (wireless/radar/sonar fading, also reliability and
hydrology). With shape ``m >= 1/2`` and spread ``omega = E[X^2] > 0``,

    f(x; m, omega) = 2 m^m / (Gamma(m) omega^m) * x^(2m-1) * exp(-m x^2 / omega),  x > 0,

so ``X^2 ~ Gamma(m, omega/m)``; ``m = 1/2`` is the half-normal and ``m = 1`` the Rayleigh. The CDF is the
regularized lower incomplete gamma, it samples exactly via a Gamma draw, and it has a clean closed-form
method-of-moments fit: ``omega = E[X^2]`` and ``m = E[X^2]^2 / Var[X^2]``.

Reference: Nakagami, "The m-distribution -- a general formula of intensity distribution of rapid
fading", in *Statistical Methods in Radio Wave Propagation* (1960).
"""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import RandomState
from scipy.special import gammainc, gammaincinv, gammaln

from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)


class NakagamiDistribution(SequenceEncodableProbabilityDistribution):
    """Nakagami distribution with shape ``m >= 1/2`` and spread ``omega = E[X^2] > 0``."""

    def __init__(self, m: float, omega: float, name: str | None = None, keys: str | None = None) -> None:
        if m < 0.5 or not np.isfinite(m):
            raise ValueError("NakagamiDistribution requires finite m >= 1/2.")
        if omega <= 0.0 or not np.isfinite(omega):
            raise ValueError("NakagamiDistribution requires finite omega > 0.")
        self.m = float(m)
        self.omega = float(omega)
        self.name = name
        self.keys = keys
        self._log_const = math.log(2.0) + self.m * math.log(self.m) - gammaln(self.m) - self.m * math.log(self.omega)
        self._m_over_omega = self.m / self.omega

    def __str__(self) -> str:
        return "NakagamiDistribution(%s, %s, name=%s, keys=%s)" % (
            repr(self.m),
            repr(self.omega),
            repr(self.name),
            repr(self.keys),
        )

    def density(self, x: float) -> float:
        """Return the probability density at ``x``."""
        return math.exp(self.log_density(x))

    def log_density(self, x: float) -> float:
        """Return the log-density at ``x`` (-inf for x <= 0)."""
        xv = float(x)
        if xv <= 0.0:
            return -math.inf
        return self._log_const + (2.0 * self.m - 1.0) * math.log(xv) - self._m_over_omega * xv * xv

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Return vectorized log-density for a sequence-encoded array of observations."""
        xv = np.asarray(x, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = self._log_const + (2.0 * self.m - 1.0) * np.log(xv) - self._m_over_omega * xv * xv
        return np.where(xv > 0.0, out, -np.inf)

    def cdf(self, x: float) -> float:
        """Cumulative distribution function P(X <= x) = P(m, m x^2 / omega) (0 for x <= 0)."""
        xv = float(x)
        return float(gammainc(self.m, self._m_over_omega * xv * xv)) if xv > 0.0 else 0.0

    def quantile(self, q: float) -> float:
        """Inverse CDF F^{-1}(q)."""
        return float(math.sqrt(self.omega * gammaincinv(self.m, float(q)) / self.m))

    def mean(self) -> float:
        """Mean (Gamma(m+1/2)/Gamma(m)) sqrt(omega/m)."""
        return float(math.exp(gammaln(self.m + 0.5) - gammaln(self.m)) * math.sqrt(self.omega / self.m))

    def variance(self) -> float:
        """Variance omega - mean^2 (since E[X^2] = omega)."""
        mu = self.mean()
        return float(self.omega - mu * mu)

    def sampler(self, seed: int | None = None) -> "NakagamiSampler":
        """Return a sampler (``X = sqrt(Gamma(m, omega/m))``)."""
        return NakagamiSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "NakagamiEstimator":
        """Return a closed-form method-of-moments estimator."""
        return NakagamiEstimator(name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> "NakagamiDataEncoder":
        """Return the data encoder used by this distribution (the raw value)."""
        return NakagamiDataEncoder()


class NakagamiSampler(DistributionSampler):
    """Draw ``X = sqrt(G)`` with ``G ~ Gamma(shape=m, scale=omega/m)`` (so ``X^2 ~ Gamma(m, omega/m)``)."""

    def __init__(self, dist: NakagamiDistribution, seed: int | None = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: int | None = None) -> float | np.ndarray:
        d = self.dist
        n = 1 if size is None else int(size)
        x = np.sqrt(self.rng.gamma(d.m, d.omega / d.m, size=n))
        return float(x[0]) if size is None else x


class NakagamiAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate the weighted power sums ``(count, sum x^2, sum x^4)`` for the moment fit."""

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.count = 0.0
        self.s2 = 0.0
        self.s4 = 0.0
        self.name = name
        self.keys = keys

    def update(self, x: float, weight: float, estimate: NakagamiDistribution | None) -> None:
        x2 = float(x) ** 2
        self.count += weight
        self.s2 += weight * x2
        self.s4 += weight * x2 * x2

    def initialize(self, x: float, weight: float, rng: RandomState | None) -> None:
        self.update(x, weight, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: Any) -> None:
        x2 = np.asarray(x, dtype=np.float64) ** 2
        w = np.asarray(weights, dtype=np.float64)
        self.count += float(w.sum())
        self.s2 += float(np.dot(w, x2))
        self.s4 += float(np.dot(w, x2 * x2))

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[float, float, float]) -> "NakagamiAccumulator":
        self.count += suff_stat[0]
        self.s2 += suff_stat[1]
        self.s4 += suff_stat[2]
        return self

    def value(self) -> tuple[float, float, float]:
        return self.count, self.s2, self.s4

    def from_value(self, x: tuple[float, float, float]) -> "NakagamiAccumulator":
        self.count, self.s2, self.s4 = float(x[0]), float(x[1]), float(x[2])
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        if self.keys is not None:
            if self.keys in stats_dict:
                stats_dict[self.keys].combine(self.value())
            else:
                stats_dict[self.keys] = self

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        if self.keys is not None and self.keys in stats_dict:
            self.from_value(stats_dict[self.keys].value())

    def acc_to_encoder(self) -> "NakagamiDataEncoder":
        return NakagamiDataEncoder()


class NakagamiAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for NakagamiAccumulator."""

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.name = name
        self.keys = keys

    def make(self) -> NakagamiAccumulator:
        return NakagamiAccumulator(name=self.name, keys=self.keys)


class NakagamiEstimator(ParameterEstimator):
    """Method-of-moments estimator: ``omega = E[X^2]``, ``m = E[X^2]^2 / Var[X^2]`` (clamped m >= 1/2)."""

    def __init__(self, m_min: float = 0.5, name: str | None = None, keys: str | None = None) -> None:
        self.m_min = m_min
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> NakagamiAccumulatorFactory:
        return NakagamiAccumulatorFactory(name=self.name, keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat: tuple[float, float, float]) -> NakagamiDistribution:
        count, s2, s4 = suff_stat
        if count <= 0.0:
            return NakagamiDistribution(1.0, 1.0, name=self.name, keys=self.keys)
        omega = s2 / count
        var_x2 = s4 / count - omega * omega
        m = (omega * omega / var_x2) if var_x2 > 0.0 else 1.0e6
        m = max(m, self.m_min)
        return NakagamiDistribution(m, omega, name=self.name, keys=self.keys)


class NakagamiDataEncoder(DataSequenceEncoder):
    """Encode observations as a float array."""

    def __str__(self) -> str:
        return "NakagamiDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, NakagamiDataEncoder)

    def seq_encode(self, x: Sequence[float]) -> np.ndarray:
        return np.asarray(x, dtype=np.float64)
