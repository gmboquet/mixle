"""Wrapped Cauchy distribution -- a heavy-tailed circular law on angles.

Wrapping a Cauchy around the circle gives the wrapped Cauchy, the circular analogue of the Cauchy and
a heavier-tailed alternative to the von Mises. With mean direction ``mu`` and mean-resultant length
``rho`` in ``[0, 1)``,

    f(theta; mu, rho) = (1 - rho^2) / (2 pi (1 + rho^2 - 2 rho cos(theta - mu))),

uniform on the circle at ``rho = 0`` and increasingly peaked at ``mu`` as ``rho -> 1``. Its first
trigonometric moment is exactly ``rho e^{i mu}``, so the mean direction and concentration are estimated
in closed form from the mean resultant (the circular analogue of the sample mean). It samples exactly:
``rho`` corresponds to wrapping a Cauchy of scale ``gamma = -log rho``.
"""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import RandomState

from pysp.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)

_LOG_2PI = math.log(2.0 * math.pi)


def _wrap(theta: Any) -> Any:
    """Wrap angle(s) to ``(-pi, pi]``."""
    return (np.asarray(theta) + math.pi) % (2.0 * math.pi) - math.pi


class WrappedCauchyDistribution(SequenceEncodableProbabilityDistribution):
    """Wrapped Cauchy distribution with mean direction ``mu`` and concentration ``rho`` in ``[0, 1)``."""

    def __init__(self, mu: float, rho: float, name: str | None = None, keys: str | None = None) -> None:
        if not (0.0 <= rho < 1.0):
            raise ValueError("WrappedCauchyDistribution requires concentration rho in [0, 1).")
        self.mu = float(math.atan2(math.sin(mu), math.cos(mu)))  # wrap to (-pi, pi]
        self.rho = float(rho)
        self.name = name
        self.keys = keys
        self._log_num = math.log1p(-self.rho * self.rho) - _LOG_2PI  # log[(1-rho^2)/(2 pi)]
        self._one_plus = 1.0 + self.rho * self.rho

    def __str__(self) -> str:
        return "WrappedCauchyDistribution(%s, %s, name=%s, keys=%s)" % (
            repr(self.mu),
            repr(self.rho),
            repr(self.name),
            repr(self.keys),
        )

    def density(self, x: float) -> float:
        """Return the probability density at a single angle (radians)."""
        return math.exp(self.log_density(x))

    def log_density(self, x: float) -> float:
        """Return the log-density at a single angle (radians)."""
        return self._log_num - math.log(self._one_plus - 2.0 * self.rho * math.cos(float(x) - self.mu))

    def seq_log_density(self, x: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """Return vectorized log-density values for sequence-encoded ``(cos, sin)`` observations."""
        cos_t, sin_t = x
        cos_dev = cos_t * math.cos(self.mu) + sin_t * math.sin(self.mu)  # cos(theta - mu)
        return self._log_num - np.log(self._one_plus - 2.0 * self.rho * cos_dev)

    def sampler(self, seed: int | None = None) -> "WrappedCauchySampler":
        """Return a sampler for drawing angles from this distribution."""
        return WrappedCauchySampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "WrappedCauchyEstimator":
        """Return a closed-form (mean-resultant) estimator for ``mu`` and ``rho``."""
        return WrappedCauchyEstimator(name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> "WrappedCauchyDataEncoder":
        """Return the data encoder used by this distribution (cos/sin of the angle)."""
        return WrappedCauchyDataEncoder()


class WrappedCauchySampler(DistributionSampler):
    """Draw angles by wrapping a Cauchy of scale ``-log rho`` around the circle."""

    def __init__(self, dist: WrappedCauchyDistribution, seed: int | None = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: int | None = None) -> float | np.ndarray:
        d = self.dist
        n = 1 if size is None else int(size)
        if d.rho == 0.0:  # uniform on the circle
            theta = self.rng.uniform(-math.pi, math.pi, size=n)
        else:
            gamma = -math.log(d.rho)  # Cauchy scale of the un-wrapped variable
            c = d.mu + gamma * np.tan(math.pi * (self.rng.uniform(size=n) - 0.5))
            theta = _wrap(c)
        return float(theta[0]) if size is None else theta


class WrappedCauchyAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate the weighted circular resultant ``(sum cos, sum sin, count)``."""

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.sum_cos = 0.0
        self.sum_sin = 0.0
        self.count = 0.0
        self.name = name
        self.key = keys

    def update(self, x: float, weight: float, estimate: WrappedCauchyDistribution | None) -> None:
        self.sum_cos += weight * math.cos(float(x))
        self.sum_sin += weight * math.sin(float(x))
        self.count += weight

    def initialize(self, x: float, weight: float, rng: RandomState | None) -> None:
        self.update(x, weight, None)

    def seq_update(self, x: tuple[np.ndarray, np.ndarray], weights: np.ndarray, estimate: Any) -> None:
        cos_t, sin_t = x
        w = np.asarray(weights, dtype=np.float64)
        self.sum_cos += float(np.dot(cos_t, w))
        self.sum_sin += float(np.dot(sin_t, w))
        self.count += float(w.sum())

    def seq_initialize(self, x: tuple[np.ndarray, np.ndarray], weights: np.ndarray, rng: RandomState | None) -> None:
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[float, float, float]) -> "WrappedCauchyAccumulator":
        self.sum_cos += suff_stat[0]
        self.sum_sin += suff_stat[1]
        self.count += suff_stat[2]
        return self

    def value(self) -> tuple[float, float, float]:
        return self.sum_cos, self.sum_sin, self.count

    def from_value(self, x: tuple[float, float, float]) -> "WrappedCauchyAccumulator":
        self.sum_cos, self.sum_sin, self.count = float(x[0]), float(x[1]), float(x[2])
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        if self.key is not None:
            if self.key in stats_dict:
                stats_dict[self.key].combine(self.value())
            else:
                stats_dict[self.key] = self

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        if self.key is not None and self.key in stats_dict:
            self.from_value(stats_dict[self.key].value())

    def acc_to_encoder(self) -> "WrappedCauchyDataEncoder":
        return WrappedCauchyDataEncoder()


class WrappedCauchyAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for WrappedCauchyAccumulator."""

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.name = name
        self.keys = keys

    def make(self) -> WrappedCauchyAccumulator:
        return WrappedCauchyAccumulator(name=self.name, keys=self.keys)


class WrappedCauchyEstimator(ParameterEstimator):
    """Estimate ``mu`` and ``rho`` from the mean resultant (the first trigonometric moment)."""

    def __init__(self, rho_max: float = 1.0 - 1.0e-8, name: str | None = None, keys: str | None = None) -> None:
        self.rho_max = rho_max
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> WrappedCauchyAccumulatorFactory:
        return WrappedCauchyAccumulatorFactory(name=self.name, keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat: tuple[float, float, float]) -> WrappedCauchyDistribution:
        sum_cos, sum_sin, count = suff_stat
        if count <= 0.0:
            return WrappedCauchyDistribution(0.0, 0.0, name=self.name, keys=self.keys)
        mu = math.atan2(sum_sin, sum_cos)
        rho = math.sqrt(sum_cos * sum_cos + sum_sin * sum_sin) / count  # mean-resultant length -> rho
        rho = min(max(rho, 0.0), self.rho_max)
        return WrappedCauchyDistribution(mu, rho, name=self.name, keys=self.keys)


class WrappedCauchyDataEncoder(DataSequenceEncoder):
    """Encode angles as their cosine and sine."""

    def __str__(self) -> str:
        return "WrappedCauchyDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, WrappedCauchyDataEncoder)

    def seq_encode(self, x: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        theta = np.asarray(x, dtype=np.float64)
        return np.cos(theta), np.sin(theta)
