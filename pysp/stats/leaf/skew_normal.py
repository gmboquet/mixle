"""Skew-normal distribution -- a Gaussian with an asymmetry (shape) parameter.

The skew-normal extends the normal with a shape ``alpha`` that tilts the density without bounding it:

    f(x) = (2 / omega) phi((x - xi) / omega) Phi(alpha (x - xi) / omega),

with location ``xi``, scale ``omega > 0`` and shape ``alpha`` (``alpha = 0`` recovers the normal, the
sign of ``alpha`` sets the direction of skew). It samples exactly from two standard normals, and is fit
by method of moments: the sample skewness fixes ``alpha`` through the monotone skewness-vs-shape
relation, then the variance fixes ``omega`` and the mean fixes ``xi``.
"""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import RandomState
from scipy.special import log_ndtr

from pysp.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)

_B = math.sqrt(2.0 / math.pi)
_HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)
# largest attainable |skewness| for the skew-normal (delta -> +/-1): (4-pi)/2 * b^3 / (1-b^2)^{3/2}
_MAX_SKEW = ((4.0 - math.pi) / 2.0) * _B**3 / (1.0 - _B * _B) ** 1.5


class SkewNormalDistribution(SequenceEncodableProbabilityDistribution):
    """Skew-normal distribution with location ``loc``, scale ``> 0`` and shape ``alpha``."""

    def __init__(
        self, loc: float, scale: float, shape: float, name: str | None = None, keys: str | None = None
    ) -> None:
        if scale <= 0.0 or not (np.isfinite(loc) and np.isfinite(scale) and np.isfinite(shape)):
            raise ValueError("SkewNormalDistribution requires finite parameters and scale > 0.")
        self.loc = float(loc)  # xi
        self.scale = float(scale)  # omega
        self.shape = float(shape)  # alpha
        self.log_scale = math.log(self.scale)
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        return "SkewNormalDistribution(%s, %s, %s, name=%s, keys=%s)" % (
            repr(self.loc),
            repr(self.scale),
            repr(self.shape),
            repr(self.name),
            repr(self.keys),
        )

    def density(self, x: float) -> float:
        """Return the probability density at a single observation."""
        return math.exp(self.log_density(x))

    def log_density(self, x: float) -> float:
        """Return the log-density at a single observation."""
        z = (float(x) - self.loc) / self.scale
        return math.log(2.0) - self.log_scale - _HALF_LOG_2PI - 0.5 * z * z + float(log_ndtr(self.shape * z))

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Return vectorized log-density values for sequence-encoded observations."""
        z = (np.asarray(x, dtype=np.float64) - self.loc) / self.scale
        return math.log(2.0) - self.log_scale - _HALF_LOG_2PI - 0.5 * z * z + log_ndtr(self.shape * z)

    def cdf(self, x: float) -> float:
        """Cumulative distribution function ``P(X <= x)`` (exact)."""
        from scipy.stats import skewnorm as _sp

        return float(_sp.cdf(x, self.shape, loc=self.loc, scale=self.scale))

    def quantile(self, q: float) -> float:
        """Inverse CDF ``F^{-1}(q)``."""
        from scipy.stats import skewnorm as _sp

        return float(_sp.ppf(q, self.shape, loc=self.loc, scale=self.scale))

    def sampler(self, seed: int | None = None) -> "SkewNormalSampler":
        """Return a sampler for drawing observations from this distribution."""
        return SkewNormalSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "SkewNormalEstimator":
        """Return a method-of-moments estimator for ``loc``, ``scale`` and ``shape``."""
        return SkewNormalEstimator(name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> "SkewNormalDataEncoder":
        """Return the data encoder used by this distribution for vectorized methods."""
        return SkewNormalDataEncoder()


class SkewNormalSampler(DistributionSampler):
    """Draw observations as ``xi + omega (delta |Z0| + sqrt(1-delta^2) Z1)`` with ``Z0, Z1`` standard normal."""

    def __init__(self, dist: SkewNormalDistribution, seed: int | None = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: int | None = None) -> float | np.ndarray:
        d = self.dist
        delta = d.shape / math.sqrt(1.0 + d.shape * d.shape)
        z0 = self.rng.randn() if size is None else self.rng.randn(int(size))
        z1 = self.rng.randn() if size is None else self.rng.randn(int(size))
        return d.loc + d.scale * (delta * np.abs(z0) + math.sqrt(1.0 - delta * delta) * z1)


class SkewNormalAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate weighted central moments for skew-normal estimation.

    The sufficient statistic is stored as ``(count, mean, M2, M3)`` where
    ``M2 = sum_i w_i (x_i - mean)^2`` and ``M3 = sum_i w_i (x_i - mean)^3`` are the
    weighted central moments. This is mathematically equivalent to the raw power sums
    ``(sum x, sum x^2, sum x^3)`` but avoids the catastrophic ``E[x^2] - E[x]^2``
    cancellation when ``|mean|`` is large relative to the spread: each batch is centered
    on its own mean *before* squaring/cubing, and batches merge through the
    Pébay/West parallel-moment formulas (exact for real weights). SkewNormal is a
    host-only leaf (no exponential-family / engine-resident path), so changing the
    accumulator representation has no engine-swap parity implications.
    """

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.count = 0.0
        self.mean = 0.0
        self.m2 = 0.0
        self.m3 = 0.0
        self.name = name
        self.key = keys

    def _merge(self, c_b: float, mean_b: float, m2_b: float, m3_b: float) -> None:
        """Merge a second weighted central-moment batch into this one (parallel form)."""
        c_a, mean_a, m2_a, m3_a = self.count, self.mean, self.m2, self.m3
        count = c_a + c_b
        if count <= 0.0:
            return
        delta = mean_b - mean_a
        # mean_a stays correct when c_b == 0; otherwise shift toward the merged mean.
        self.mean = mean_a + delta * (c_b / count)
        self.m2 = m2_a + m2_b + delta * delta * (c_a * c_b / count)
        self.m3 = (
            m3_a
            + m3_b
            + delta**3 * (c_a * c_b * (c_a - c_b) / (count * count))
            + 3.0 * delta * (c_a * m2_b - c_b * m2_a) / count
        )
        self.count = count

    def update(self, x: float, weight: float, estimate: SkewNormalDistribution | None) -> None:
        # A single observation is a batch with zero internal spread (M2 = M3 = 0).
        self._merge(float(weight), float(x), 0.0, 0.0)

    def initialize(self, x: float, weight: float, rng: RandomState | None) -> None:
        self.update(x, weight, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: SkewNormalDistribution | None) -> None:
        xx = np.asarray(x, dtype=np.float64)
        ww = np.asarray(weights, dtype=np.float64)
        c_b = float(np.sum(ww))
        if c_b <= 0.0:
            return
        mean_b = float(np.dot(xx, ww) / c_b)
        dx = xx - mean_b  # center before squaring/cubing -> no cancellation
        m2_b = float(np.dot(ww, dx * dx))
        m3_b = float(np.dot(ww, dx * dx * dx))
        self._merge(c_b, mean_b, m2_b, m3_b)

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[float, float, float, float]) -> "SkewNormalAccumulator":
        self._merge(float(suff_stat[0]), float(suff_stat[1]), float(suff_stat[2]), float(suff_stat[3]))
        return self

    def value(self) -> tuple[float, float, float, float]:
        return self.count, self.mean, self.m2, self.m3

    def from_value(self, x: tuple[float, float, float, float]) -> "SkewNormalAccumulator":
        self.count, self.mean, self.m2, self.m3 = float(x[0]), float(x[1]), float(x[2]), float(x[3])
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

    def acc_to_encoder(self) -> "SkewNormalDataEncoder":
        return SkewNormalDataEncoder()


class SkewNormalAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for SkewNormalAccumulator."""

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.name = name
        self.keys = keys

    def make(self) -> SkewNormalAccumulator:
        return SkewNormalAccumulator(name=self.name, keys=self.keys)


class SkewNormalEstimator(ParameterEstimator):
    """Method-of-moments estimator for skew-normal location, scale and shape."""

    def __init__(self, min_scale: float = 1.0e-12, name: str | None = None, keys: str | None = None) -> None:
        self.min_scale = min_scale
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> SkewNormalAccumulatorFactory:
        return SkewNormalAccumulatorFactory(name=self.name, keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat: tuple[float, float, float, float]) -> SkewNormalDistribution:
        count, mean, sum_m2, sum_m3 = suff_stat
        if count <= 0.0:
            return SkewNormalDistribution(0.0, 1.0, 0.0, name=self.name, keys=self.keys)
        # ``sum_m2``/``sum_m3`` are the weighted central moments (see SkewNormalAccumulator),
        # so var/m3 are read off directly with no E[x^2]-E[x]^2 cancellation.
        var = sum_m2 / count
        if var <= 0.0:
            return SkewNormalDistribution(mean, self.min_scale, 0.0, name=self.name, keys=self.keys)
        m3 = sum_m3 / count  # central third moment
        skew = m3 / var**1.5
        skew = min(max(skew, -_MAX_SKEW * (1.0 - 1.0e-6)), _MAX_SKEW * (1.0 - 1.0e-6))
        # invert skewness -> u = b^2 delta^2 in [0,1): (1-u)/u = (((4-pi)/2)/|skew|)^{2/3}
        if skew == 0.0:
            return SkewNormalDistribution(mean, math.sqrt(var), 0.0, name=self.name, keys=self.keys)
        ratio = (((4.0 - math.pi) / 2.0) / abs(skew)) ** (2.0 / 3.0)
        u = 1.0 / (1.0 + ratio)  # = b^2 delta^2
        delta = math.copysign(math.sqrt(u * math.pi / 2.0), skew)  # b^2 = 2/pi -> delta^2 = u*pi/2
        delta = math.copysign(min(abs(delta), 1.0 - 1.0e-9), delta)
        alpha = delta / math.sqrt(1.0 - delta * delta)
        omega = math.sqrt(var / (1.0 - _B * _B * delta * delta))
        xi = mean - omega * _B * delta
        return SkewNormalDistribution(xi, max(omega, self.min_scale), alpha, name=self.name, keys=self.keys)


class SkewNormalDataEncoder(DataSequenceEncoder):
    """Encode skew-normal observations as a float array."""

    def __str__(self) -> str:
        return "SkewNormalDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SkewNormalDataEncoder)

    def seq_encode(self, x: Sequence[float]) -> np.ndarray:
        return np.asarray(x, dtype=np.float64)
