"""Two-parameter Weibull distributions over positive real values.

Reference: Johnson, Kotz & Balakrishnan, *Continuous Univariate Distributions* (2nd ed., Wiley, 1994/95).
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


def _weibull_cv2(shape: float) -> float:
    a = math.lgamma(1.0 + 2.0 / shape)
    b = math.lgamma(1.0 + 1.0 / shape)
    t = a - 2.0 * b
    if t >= math.log(np.finfo(float).max):
        return np.inf
    return math.exp(t) - 1.0


def _shape_from_moments(mean: float, var: float, min_shape: float, max_shape: float) -> float:
    if mean <= 0.0 or var <= 0.0:
        return max_shape
    target = var / (mean * mean)
    if not np.isfinite(target) or target <= 0.0:
        return max_shape

    lo = float(min_shape)
    hi = float(max_shape)
    if target >= _weibull_cv2(lo):
        return lo
    if target <= _weibull_cv2(hi):
        return hi

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _weibull_cv2(mid) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


class WeibullDistribution(SequenceEncodableProbabilityDistribution):
    """Weibull distribution with shape > 0 and scale > 0 on x >= 0."""

    @classmethod
    def compute_capabilities(cls):
        """Describe backend support for generated Weibull kernels."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch", "jax"), kernel_status="numba_adapter")

    @classmethod
    def compute_declaration(cls):
        """Return the structured compute declaration for Weibull distributions."""
        from mixle.stats.compute.declarations import DistributionDeclaration, ParameterSpec, StatisticSpec

        return DistributionDeclaration(
            name="weibull",
            distribution_type=cls,
            parameters=(
                ParameterSpec("shape", constraint="positive"),
                ParameterSpec("scale", constraint="positive"),
            ),
            statistics=(StatisticSpec("sum"), StatisticSpec("sum2"), StatisticSpec("count")),
            support="non_negative_real",
            legacy_sufficient_statistics=cls.backend_legacy_sufficient_statistics,
        )

    @staticmethod
    def backend_legacy_sufficient_statistics(
        x: tuple[Any, Any], params: dict[str, Any], engine: Any
    ) -> tuple[Any, ...]:
        """Return per-row Weibull sufficient statistics in accumulator order."""
        vals = engine.asarray(x[0])
        return vals, vals * vals, vals * 0.0 + engine.asarray(1.0)

    def __init__(self, shape: float, scale: float, name: str | None = None, keys: str | None = None) -> None:
        if shape <= 0.0 or scale <= 0.0 or not np.isfinite(shape) or not np.isfinite(scale):
            raise ValueError("WeibullDistribution requires shape > 0 and scale > 0.")
        self.shape = float(shape)
        self.scale = float(scale)
        self.log_shape = math.log(self.shape)
        self.log_scale = math.log(self.scale)
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        return "WeibullDistribution(%s, %s, name=%s, keys=%s)" % (
            repr(self.shape),
            repr(self.scale),
            repr(self.name),
            repr(self.keys),
        )

    def density(self, x: float) -> float:
        """Return the probability density or mass at a single observation."""
        return math.exp(self.log_density(x))

    def log_density(self, x: float) -> float:
        """Return the log-density or log-mass at a single observation."""
        if x < 0.0:
            return -np.inf
        if x == 0.0:
            if self.shape < 1.0:
                return np.inf
            if self.shape > 1.0:
                return -np.inf
            return -self.log_scale
        z = x / self.scale
        return self.log_shape - self.log_scale + (self.shape - 1.0) * math.log(z) - z**self.shape

    def seq_log_density(self, x: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """Return vectorized log-density values for sequence-encoded observations."""
        xx, lx = x
        z = xx / self.scale
        with np.errstate(divide="ignore", invalid="ignore"):
            rv = self.log_shape - self.log_scale + (self.shape - 1.0) * (lx - self.log_scale) - np.power(z, self.shape)
        rv = np.where(xx < 0.0, -np.inf, rv)
        if self.shape == 1.0:
            rv = np.where(xx == 0.0, -self.log_scale, rv)
        elif self.shape > 1.0:
            rv = np.where(xx == 0.0, -np.inf, rv)
        else:
            rv = np.where(xx == 0.0, np.inf, rv)
        return rv

    @staticmethod
    def backend_log_density_from_params(vals: Any, log_vals: Any, shape: Any, scale: Any, engine: Any) -> Any:
        """Engine-neutral Weibull log-density from explicit parameters."""
        z = vals / scale
        rv = (
            engine.log(shape)
            - engine.log(scale)
            + (shape - engine.asarray(1.0)) * (log_vals - engine.log(scale))
            - (z**shape)
        )
        return engine.where(vals < 0.0, engine.asarray(-np.inf), rv)

    def backend_seq_log_density(self, x: tuple[Any, Any], engine: Any) -> Any:
        """Engine-neutral vectorized log-density for encoded data."""
        vals = engine.asarray(x[0])
        log_vals = engine.asarray(x[1])
        rv = self.backend_log_density_from_params(
            vals, log_vals, engine.asarray(self.shape), engine.asarray(self.scale), engine
        )
        if engine.requires_grad(self.shape):
            return rv
        if self.shape == 1.0:
            rv = engine.where(vals == 0.0, engine.asarray(-self.log_scale), rv)
        elif self.shape > 1.0:
            rv = engine.where(vals == 0.0, engine.asarray(-np.inf), rv)
        else:
            rv = engine.where(vals == 0.0, engine.asarray(np.inf), rv)
        return rv

    @classmethod
    def backend_stacked_params(cls, dists: Sequence["WeibullDistribution"], engine: Any) -> dict[str, Any]:
        """Return stacked Weibull parameters for a homogeneous mixture kernel."""
        return {
            "shape": engine.asarray([d.shape for d in dists]),
            "scale": engine.asarray([d.scale for d in dists]),
        }

    @classmethod
    def backend_stacked_log_density(cls, x: tuple[Any, Any], params: dict[str, Any], engine: Any) -> Any:
        """Return an ``(n, k)`` matrix of Weibull log densities."""
        vals = engine.asarray(x[0])
        log_vals = engine.asarray(x[1])
        shape = params["shape"][None, :]
        scale = params["scale"][None, :]
        rv = cls.backend_log_density_from_params(vals[:, None], log_vals[:, None], shape, scale, engine)
        is_zero = vals[:, None] == 0.0
        rv = engine.where((shape == 1.0) & is_zero, -engine.log(scale), rv)
        rv = engine.where((shape > 1.0) & is_zero, engine.asarray(-np.inf), rv)
        rv = engine.where((shape < 1.0) & is_zero, engine.asarray(np.inf), rv)
        return rv

    def cdf(self, x: float) -> float:
        """Cumulative distribution function ``P(X <= x)`` (exact). The continuous 'index of' a value."""
        from scipy.stats import weibull_min as _sp

        return float(_sp.cdf(x, self.shape, scale=self.scale))

    def quantile(self, q: float) -> float:
        """Inverse CDF ``F^{-1}(q)``: the value at cumulative-probability index ``q`` (continuous unranking)."""
        from scipy.stats import weibull_min as _sp

        return float(_sp.ppf(q, self.shape, scale=self.scale))

    def mean(self) -> float:
        """Mean scale * Gamma(1 + 1/shape)."""
        import math

        return float(self.scale * math.gamma(1.0 + 1.0 / self.shape))

    def variance(self) -> float:
        """Variance scale^2 * (Gamma(1+2/shape) - Gamma(1+1/shape)^2)."""
        import math

        g1 = math.gamma(1.0 + 1.0 / self.shape)
        g2 = math.gamma(1.0 + 2.0 / self.shape)
        return float(self.scale * self.scale * (g2 - g1 * g1))

    def entropy(self) -> float:
        """Differential entropy gamma*(1 - 1/shape) + log(scale/shape) + 1."""
        import math

        import numpy as np

        return float(np.euler_gamma * (1.0 - 1.0 / self.shape) + math.log(self.scale / self.shape) + 1.0)

    def mode(self) -> float:
        """Mode scale*((k-1)/k)^(1/k) for shape k>1, else 0."""

        return float(self.scale * ((self.shape - 1.0) / self.shape) ** (1.0 / self.shape)) if self.shape > 1.0 else 0.0

    def sampler(self, seed: int | None = None) -> "WeibullSampler":
        """Return a sampler for drawing observations from this distribution."""
        return WeibullSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "WeibullEstimator":
        """Return an estimator for fitting this distribution from data."""
        if pseudo_count is None:
            return WeibullEstimator(name=self.name, keys=self.keys)
        mean = self.scale * math.exp(math.lgamma(1.0 + 1.0 / self.shape))
        second = self.scale * self.scale * math.exp(math.lgamma(1.0 + 2.0 / self.shape))
        return WeibullEstimator(pseudo_count=pseudo_count, suff_stat=(mean, second), name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> "WeibullDataEncoder":
        """Return the data encoder used by this distribution for vectorized methods."""
        return WeibullDataEncoder()


class WeibullSampler(DistributionSampler):
    """Draw iid Weibull observations."""

    def __init__(self, dist: WeibullDistribution, seed: int | None = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: int | None = None, *, batched: bool = True) -> float | np.ndarray:
        """Draw one sample or an array of iid samples."""
        return self.dist.scale * self.rng.weibull(self.dist.shape, size=size)


class WeibullAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate weighted first and second moments for Weibull estimation."""

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.sum = 0.0
        self.sum2 = 0.0
        self.count = 0.0
        self.name = name
        self.keys = keys

    def update(self, x: float, weight: float, estimate: WeibullDistribution | None) -> None:
        """Accumulate weighted first and second moments for one observation."""
        if x < 0.0:
            raise ValueError("WeibullDistribution requires observations x >= 0.")
        self.sum += x * weight
        self.sum2 += x * x * weight
        self.count += weight

    def initialize(self, x: float, weight: float, rng: RandomState | None) -> None:
        """Initialize statistics from one observation."""
        self.update(x, weight, None)

    def seq_update(
        self, x: tuple[np.ndarray, np.ndarray], weights: np.ndarray, estimate: WeibullDistribution | None
    ) -> None:
        """Accumulate weighted first and second moments from encoded data."""
        xx, _ = x
        self.sum += np.dot(xx, weights)
        self.sum2 += np.dot(xx * xx, weights)
        self.count += np.sum(weights, dtype=np.float64)

    def seq_initialize(self, x: tuple[np.ndarray, np.ndarray], weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize statistics from encoded observations."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[float, float, float]) -> "WeibullAccumulator":
        """Merge another Weibull sufficient-statistic tuple."""
        self.sum += suff_stat[0]
        self.sum2 += suff_stat[1]
        self.count += suff_stat[2]
        return self

    def value(self) -> tuple[float, float, float]:
        """Return accumulated sum, second moment sum, and count."""
        return self.sum, self.sum2, self.count

    def from_value(self, x: tuple[float, float, float]) -> "WeibullAccumulator":
        """Replace accumulator contents from a sufficient-statistic tuple."""
        self.sum = x[0]
        self.sum2 = x[1]
        self.count = x[2]
        return self

    def acc_to_encoder(self) -> "WeibullDataEncoder":
        """Return the encoder used by this accumulator."""
        return WeibullDataEncoder()


class WeibullAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for WeibullAccumulator."""

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.name = name
        self.keys = keys

    def make(self) -> WeibullAccumulator:
        """Create a fresh Weibull accumulator."""
        return WeibullAccumulator(name=self.name, keys=self.keys)


class WeibullEstimator(ParameterEstimator):
    """Moment estimator for Weibull shape and scale."""

    def __init__(
        self,
        pseudo_count: float | None = None,
        suff_stat: tuple[float, float] | None = None,
        min_shape: float = 1.0e-3,
        max_shape: float = 1.0e3,
        min_scale: float = 1.0e-12,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.min_shape = min_shape
        self.max_shape = max_shape
        self.min_scale = min_scale
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> WeibullAccumulatorFactory:
        """Return an accumulator factory for Weibull moment statistics."""
        return WeibullAccumulatorFactory(name=self.name, keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat: tuple[float, float, float]) -> WeibullDistribution:
        """Estimate shape and scale by matching weighted moments."""
        sum_x, sum_x2, count = suff_stat
        if self.pseudo_count is not None and self.suff_stat is not None:
            mean0, second0 = self.suff_stat
            sum_x += self.pseudo_count * mean0
            sum_x2 += self.pseudo_count * second0
            count += self.pseudo_count

        if count <= 0.0:
            return WeibullDistribution(1.0, 1.0, name=self.name, keys=self.keys)

        mean = max(sum_x / count, self.min_scale)
        var = max(sum_x2 / count - mean * mean, 0.0)
        shape = _shape_from_moments(mean, var, self.min_shape, self.max_shape)
        scale = mean / math.exp(math.lgamma(1.0 + 1.0 / shape))
        return WeibullDistribution(shape, max(scale, self.min_scale), name=self.name, keys=self.keys)


class WeibullDataEncoder(DataSequenceEncoder):
    """Encode Weibull observations with x and log(x)."""

    def __str__(self) -> str:
        return "WeibullDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, WeibullDataEncoder)

    def seq_encode(self, x: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        """Encode observations as values and log-values."""
        rv = np.asarray(x, dtype=np.float64)
        if rv.size and (np.any(rv < 0.0) or np.any(np.isnan(rv))):
            raise ValueError("WeibullDistribution requires observations x >= 0.")
        with np.errstate(divide="ignore"):
            lx = np.log(rv)
        return rv, lx
