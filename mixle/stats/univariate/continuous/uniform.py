"""Continuous uniform distributions over bounded intervals.

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


class UniformDistribution(SequenceEncodableProbabilityDistribution):
    """Continuous uniform distribution on [low, high]."""

    @classmethod
    def compute_capabilities(cls):
        """Describe backend support for generated uniform kernels."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch"), kernel_status="numba_adapter")

    @classmethod
    def compute_declaration(cls):
        """Return the structured compute declaration for uniform distributions."""
        from mixle.stats.compute.declarations import DistributionDeclaration, ParameterSpec, StatisticSpec

        return DistributionDeclaration(
            name="uniform",
            distribution_type=cls,
            parameters=(ParameterSpec("low"), ParameterSpec("high", constraint="greater_than:low")),
            statistics=(
                StatisticSpec("count"),
                StatisticSpec("min_val", kind="support_bound", additive=False, scales=False),
                StatisticSpec("max_val", kind="support_bound", additive=False, scales=False),
            ),
            support="bounded_real",
        )

    def __init__(self, low: float, high: float, name: str | None = None, keys: str | None = None) -> None:
        if high <= low or not np.isfinite(low) or not np.isfinite(high):
            raise ValueError("UniformDistribution requires finite low < high.")
        self.low = float(low)
        self.high = float(high)
        self.log_density_value = -math.log(self.high - self.low)
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        return "UniformDistribution(%s, %s, name=%s, keys=%s)" % (
            repr(self.low),
            repr(self.high),
            repr(self.name),
            repr(self.keys),
        )

    def density(self, x: float) -> float:
        """Return the probability density or mass at a single observation."""
        return math.exp(self.log_density(x))

    def log_density(self, x: float) -> float:
        """Return the log-density or log-mass at a single observation."""
        return self.log_density_value if self.low <= x <= self.high else -np.inf

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Return vectorized log-density values for sequence-encoded observations."""
        return np.where((x >= self.low) & (x <= self.high), self.log_density_value, -np.inf)

    @staticmethod
    def backend_log_density_from_params(x: Any, low: Any, high: Any, engine: Any) -> Any:
        """Engine-neutral uniform log-density from explicit parameters."""
        rv = -engine.log(high - low)
        return engine.where((x >= low) & (x <= high), rv + x * 0.0, engine.asarray(-np.inf))

    def backend_seq_log_density(self, x: Any, engine: Any) -> Any:
        """Engine-neutral vectorized log-density for encoded data."""
        return self.backend_log_density_from_params(
            engine.asarray(x), engine.asarray(self.low), engine.asarray(self.high), engine
        )

    @classmethod
    def backend_stacked_params(cls, dists: Sequence["UniformDistribution"], engine: Any) -> dict[str, Any]:
        """Return stacked uniform parameters for a homogeneous mixture kernel."""
        return {
            "low": engine.asarray([d.low for d in dists]),
            "high": engine.asarray([d.high for d in dists]),
        }

    @classmethod
    def backend_stacked_log_density(cls, x: Any, params: dict[str, Any], engine: Any) -> Any:
        """Return an ``(n, k)`` matrix of uniform log densities."""
        xx = engine.asarray(x)
        return cls.backend_log_density_from_params(xx[:, None], params["low"][None, :], params["high"][None, :], engine)

    @classmethod
    def backend_stacked_sufficient_statistics(
        cls, x: Any, weights: Any, params: dict[str, Any], engine: Any
    ) -> tuple[Any, Any, Any]:
        """Return stacked Uniform sufficient statistics using engine-resident arrays."""
        xx = engine.asarray(x)
        ww = engine.asarray(weights)
        mask = ww > 0.0
        vals = xx[:, None]
        count = engine.sum(ww, axis=0)
        min_val = -engine.max(engine.where(mask, -vals, engine.asarray(-np.inf)), axis=0)
        max_val = engine.max(engine.where(mask, vals, engine.asarray(-np.inf)), axis=0)
        return count, min_val, max_val

    def cdf(self, x: float) -> float:
        """Cumulative distribution function ``P(X <= x)`` (exact). The continuous 'index of' a value."""
        from scipy.stats import uniform as _sp

        return float(_sp.cdf(x, loc=self.low, scale=self.high - self.low))

    def quantile(self, q: float) -> float:
        """Inverse CDF ``F^{-1}(q)``: the value at cumulative-probability index ``q`` (continuous unranking)."""
        from scipy.stats import uniform as _sp

        return float(_sp.ppf(q, loc=self.low, scale=self.high - self.low))

    def mean(self) -> float:
        """Mean E[X] of the distribution."""
        return float(0.5 * (self.low + self.high))

    def variance(self) -> float:
        """Variance Var[X] of the distribution."""
        return float((self.high - self.low) ** 2 / 12.0)

    def entropy(self) -> float:
        """Differential entropy log(high - low)."""
        import math

        return float(math.log(self.high - self.low))

    def skewness(self) -> float:
        """Skewness (0)."""
        return 0.0

    def kurtosis(self) -> float:
        """Excess kurtosis (-6/5)."""
        return -1.2

    def sampler(self, seed: int | None = None) -> "UniformSampler":
        """Return a sampler for drawing observations from this distribution."""
        return UniformSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "UniformEstimator":
        """Return an estimator for fitting this distribution from data."""
        if pseudo_count is None:
            return UniformEstimator(name=self.name, keys=self.keys)
        return UniformEstimator(
            pseudo_count=pseudo_count, suff_stat=(self.low, self.high), name=self.name, keys=self.keys
        )

    def dist_to_encoder(self) -> "UniformDataEncoder":
        """Return the data encoder used by this distribution for vectorized methods."""
        return UniformDataEncoder()


class UniformSampler(DistributionSampler):
    """Draw iid uniform observations."""

    def __init__(self, dist: UniformDistribution, seed: int | None = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: int | None = None) -> float | np.ndarray:
        """Draw one sample or an array of iid samples."""
        return self.rng.uniform(self.dist.low, self.dist.high, size=size)


class UniformAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate weighted min/max support statistics."""

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.count = 0.0
        self.min_val = np.inf
        self.max_val = -np.inf
        self.name = name
        self.keys = keys

    def update(self, x: float, weight: float, estimate: UniformDistribution | None) -> None:
        """Accumulate support bounds and count from one weighted observation."""
        if weight > 0.0:
            self.count += weight
            self.min_val = min(self.min_val, x)
            self.max_val = max(self.max_val, x)

    def initialize(self, x: float, weight: float, rng: RandomState | None) -> None:
        """Initialize statistics from one observation."""
        self.update(x, weight, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: UniformDistribution | None) -> None:
        """Accumulate support bounds and count from encoded observations."""
        mask = weights > 0.0
        if np.any(mask):
            self.count += np.sum(weights[mask], dtype=np.float64)
            self.min_val = min(self.min_val, float(np.min(x[mask])))
            self.max_val = max(self.max_val, float(np.max(x[mask])))

    def seq_update_engine(self, x: np.ndarray, weights: Any, estimate: UniformDistribution | None, engine: Any) -> None:
        """Engine-resident accumulation of the weighted count (numpy or torch).

        The support min/max are host scalar bookkeeping over the observed values.
        """
        weights_np = np.asarray(engine.to_numpy(weights) if hasattr(engine, "to_numpy") else weights, dtype=np.float64)
        w = engine.asarray(weights_np)
        zero = engine.asarray(0.0)
        pos = w > zero
        self.count += float(engine.to_numpy(engine.sum(engine.where(pos, w, zero))))
        mask_np = weights_np > 0.0
        if np.any(mask_np):
            xv = np.asarray(x)[mask_np]
            self.min_val = min(self.min_val, float(np.min(xv)))
            self.max_val = max(self.max_val, float(np.max(xv)))

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize statistics from encoded observations."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[float, float, float]) -> "UniformAccumulator":
        """Merge another uniform sufficient-statistic tuple."""
        self.count += suff_stat[0]
        self.min_val = min(self.min_val, suff_stat[1])
        self.max_val = max(self.max_val, suff_stat[2])
        return self

    def value(self) -> tuple[float, float, float]:
        """Return count and observed support bounds."""
        return self.count, self.min_val, self.max_val

    def from_value(self, x: tuple[float, float, float]) -> "UniformAccumulator":
        """Replace accumulator contents from a sufficient-statistic tuple."""
        self.count = x[0]
        self.min_val = x[1]
        self.max_val = x[2]
        return self

    def scale(self, c: float) -> "UniformAccumulator":
        """Scale observation count while preserving support bounds."""
        self.count *= c
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

    def acc_to_encoder(self) -> "UniformDataEncoder":
        """Return the encoder used by this accumulator."""
        return UniformDataEncoder()


class UniformAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for UniformAccumulator."""

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.name = name
        self.keys = keys

    def make(self) -> UniformAccumulator:
        """Create a fresh uniform accumulator."""
        return UniformAccumulator(name=self.name, keys=self.keys)


class UniformEstimator(ParameterEstimator):
    """MLE estimator for uniform support endpoints."""

    def __init__(
        self,
        pseudo_count: float | None = None,
        suff_stat: tuple[float, float] | None = None,
        min_width: float = 1.0e-8,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.min_width = min_width
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> UniformAccumulatorFactory:
        """Return an accumulator factory for uniform support statistics."""
        return UniformAccumulatorFactory(name=self.name, keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat: tuple[float, float, float]) -> UniformDistribution:
        """Estimate support endpoints from observed weighted bounds."""
        count, low, high = suff_stat
        if count <= 0.0:
            low, high = self.suff_stat if self.suff_stat is not None else (0.0, 1.0)
        elif self.pseudo_count is not None and self.suff_stat is not None:
            low = min(low, self.suff_stat[0])
            high = max(high, self.suff_stat[1])
        if high <= low:
            mid = 0.5 * (low + high)
            low = mid - 0.5 * self.min_width
            high = mid + 0.5 * self.min_width
        return UniformDistribution(low, high, name=self.name, keys=self.keys)


class UniformDataEncoder(DataSequenceEncoder):
    """Encode uniform observations as a float array."""

    def __str__(self) -> str:
        return "UniformDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, UniformDataEncoder)

    def seq_encode(self, x: Sequence[float]) -> np.ndarray:
        """Encode observations as a floating-point array."""
        rv = np.asarray(x, dtype=np.float64)
        if rv.size and np.any(np.isnan(rv)):
            raise ValueError("UniformDistribution requires finite or infinite real-valued observations.")
        return rv
