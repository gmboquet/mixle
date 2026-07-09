"""Location-scale logistic distributions over real values.

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


class LogisticDistribution(SequenceEncodableProbabilityDistribution):
    """Logistic distribution with location loc and scale > 0."""

    @classmethod
    def compute_capabilities(cls):
        """Describe backend support for generated logistic kernels."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch"), kernel_status="numba_adapter")

    @classmethod
    def compute_declaration(cls):
        """Return the structured compute declaration for logistic distributions."""
        from mixle.stats.compute.declarations import DistributionDeclaration, ParameterSpec, StatisticSpec

        return DistributionDeclaration(
            name="logistic",
            distribution_type=cls,
            parameters=(ParameterSpec("loc"), ParameterSpec("scale", constraint="positive")),
            statistics=(StatisticSpec("sum"), StatisticSpec("sum2"), StatisticSpec("count")),
            support="real",
            legacy_sufficient_statistics=cls.backend_legacy_sufficient_statistics,
        )

    @staticmethod
    def backend_legacy_sufficient_statistics(x: Any, params: dict[str, Any], engine: Any) -> tuple[Any, ...]:
        """Return per-row Logistic sufficient statistics in accumulator order."""
        xx = engine.asarray(x)
        return xx, xx * xx, xx * 0.0 + engine.asarray(1.0)

    def __init__(self, loc: float = 0.0, scale: float = 1.0, name: str | None = None, keys: str | None = None) -> None:
        if scale <= 0.0 or not np.isfinite(scale):
            raise ValueError("LogisticDistribution requires scale > 0.")
        self.loc = float(loc)
        self.scale = float(scale)
        self.log_scale = math.log(self.scale)
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        return "LogisticDistribution(loc=%s, scale=%s, name=%s, keys=%s)" % (
            repr(self.loc),
            repr(self.scale),
            repr(self.name),
            repr(self.keys),
        )

    def density(self, x: float) -> float:
        """Return the probability density or mass at a single observation."""
        return math.exp(self.log_density(x))

    def log_density(self, x: float) -> float:
        """Return the log-density or log-mass at a single observation."""
        z = (x - self.loc) / self.scale
        return -self.log_scale - z - 2.0 * float(np.logaddexp(0.0, -z))

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Return vectorized log-density values for sequence-encoded observations."""
        z = (x - self.loc) / self.scale
        return -self.log_scale - z - 2.0 * np.logaddexp(0.0, -z)

    @staticmethod
    def backend_log_density_from_params(x: Any, loc: Any, scale: Any, engine: Any) -> Any:
        """Engine-neutral logistic log-density from explicit parameters."""
        z = (x - loc) / scale
        log_scale = engine.log(scale)
        pos = -log_scale - z - engine.asarray(2.0) * engine.log(engine.asarray(1.0) + engine.exp(-z))
        neg = -log_scale + z - engine.asarray(2.0) * engine.log(engine.asarray(1.0) + engine.exp(z))
        return engine.where(z >= 0.0, pos, neg)

    def backend_seq_log_density(self, x: Any, engine: Any) -> Any:
        """Engine-neutral vectorized log-density for encoded data."""
        return self.backend_log_density_from_params(
            engine.asarray(x), engine.asarray(self.loc), engine.asarray(self.scale), engine
        )

    @classmethod
    def backend_stacked_params(cls, dists: Sequence["LogisticDistribution"], engine: Any) -> dict[str, Any]:
        """Return stacked logistic parameters for a homogeneous mixture kernel."""
        return {
            "loc": engine.asarray([d.loc for d in dists]),
            "scale": engine.asarray([d.scale for d in dists]),
        }

    @classmethod
    def backend_stacked_log_density(cls, x: Any, params: dict[str, Any], engine: Any) -> Any:
        """Return an ``(n, k)`` matrix of logistic log densities."""
        xx = engine.asarray(x)
        return cls.backend_log_density_from_params(
            xx[:, None], params["loc"][None, :], params["scale"][None, :], engine
        )

    def cdf(self, x: float) -> float:
        """Cumulative distribution function ``P(X <= x)`` (exact). The continuous 'index of' a value."""
        from scipy.stats import logistic as _sp

        return float(_sp.cdf(x, loc=self.loc, scale=self.scale))

    def quantile(self, q: float) -> float:
        """Inverse CDF ``F^{-1}(q)``: the value at cumulative-probability index ``q`` (continuous unranking)."""
        from scipy.stats import logistic as _sp

        return float(_sp.ppf(q, loc=self.loc, scale=self.scale))

    def mean(self) -> float:
        """Mean E[X] of the distribution."""
        return float(self.loc)

    def variance(self) -> float:
        """Variance Var[X] of the distribution."""
        return float((np.pi**2 / 3.0) * self.scale * self.scale)

    def entropy(self) -> float:
        """Differential entropy log(scale) + 2."""
        import math

        return float(math.log(self.scale) + 2.0)

    def skewness(self) -> float:
        """Skewness (0)."""
        return 0.0

    def kurtosis(self) -> float:
        """Excess kurtosis (6/5)."""
        return 1.2

    def mode(self) -> float:
        """Mode (= the location loc)."""
        return float(self.loc)

    def sampler(self, seed: int | None = None) -> "LogisticSampler":
        """Return a sampler for drawing observations from this distribution."""
        return LogisticSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "LogisticEstimator":
        """Return an estimator for fitting this distribution from data."""
        if pseudo_count is None:
            return LogisticEstimator(name=self.name, keys=self.keys)
        return LogisticEstimator(
            pseudo_count=pseudo_count, suff_stat=(self.loc, self.scale), name=self.name, keys=self.keys
        )

    def dist_to_encoder(self) -> "LogisticDataEncoder":
        """Return the data encoder used by this distribution for vectorized methods."""
        return LogisticDataEncoder()


class LogisticSampler(DistributionSampler):
    """Draw iid logistic observations."""

    def __init__(self, dist: LogisticDistribution, seed: int | None = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: int | None = None) -> float | np.ndarray:
        """Draw one sample or an array of iid samples."""
        return self.rng.logistic(loc=self.dist.loc, scale=self.dist.scale, size=size)


class LogisticAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate weighted first and second moments for logistic estimation."""

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.sum = 0.0
        self.sum2 = 0.0
        self.count = 0.0
        self.name = name
        self.keys = keys

    def update(self, x: float, weight: float, estimate: LogisticDistribution | None) -> None:
        """Accumulate weighted first and second moments for one observation."""
        self.sum += x * weight
        self.sum2 += x * x * weight
        self.count += weight

    def initialize(self, x: float, weight: float, rng: RandomState | None) -> None:
        """Initialize statistics from one observation."""
        self.update(x, weight, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: LogisticDistribution | None) -> None:
        """Accumulate weighted first and second moments from encoded data."""
        self.sum += np.dot(x, weights)
        self.sum2 += np.dot(x * x, weights)
        self.count += np.sum(weights, dtype=np.float64)

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize statistics from encoded observations."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[float, float, float]) -> "LogisticAccumulator":
        """Merge another logistic sufficient-statistic tuple."""
        self.sum += suff_stat[0]
        self.sum2 += suff_stat[1]
        self.count += suff_stat[2]
        return self

    def value(self) -> tuple[float, float, float]:
        """Return accumulated sum, second moment sum, and count."""
        return self.sum, self.sum2, self.count

    def from_value(self, x: tuple[float, float, float]) -> "LogisticAccumulator":
        """Replace accumulator contents from a sufficient-statistic tuple."""
        self.sum = x[0]
        self.sum2 = x[1]
        self.count = x[2]
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

    def acc_to_encoder(self) -> "LogisticDataEncoder":
        """Return the encoder used by this accumulator."""
        return LogisticDataEncoder()


class LogisticAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for LogisticAccumulator."""

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.name = name
        self.keys = keys

    def make(self) -> LogisticAccumulator:
        """Create a fresh logistic accumulator."""
        return LogisticAccumulator(name=self.name, keys=self.keys)


class LogisticEstimator(ParameterEstimator):
    """Moment estimator for logistic location and scale.

    The likelihood MLE has no closed-form M-step. The EM estimator uses the
    identities mean=loc and var=pi^2 scale^2 / 3; torch gradient MLE can refine
    both parameters when exact likelihood optimization is desired.
    """

    def __init__(
        self,
        pseudo_count: float | None = None,
        suff_stat: tuple[float, float] | None = None,
        min_scale: float = 1.0e-8,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.min_scale = min_scale
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> LogisticAccumulatorFactory:
        """Return an accumulator factory for logistic moment statistics."""
        return LogisticAccumulatorFactory(name=self.name, keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat: tuple[float, float, float]) -> LogisticDistribution:
        """Estimate location and scale from weighted moments."""
        sum_x, sum_x2, count = suff_stat
        if self.pseudo_count is not None and self.suff_stat is not None:
            loc0, scale0 = self.suff_stat
            var0 = (math.pi * math.pi / 3.0) * scale0 * scale0
            sum_x += self.pseudo_count * loc0
            sum_x2 += self.pseudo_count * (var0 + loc0 * loc0)
            count += self.pseudo_count

        if count <= 0.0:
            return LogisticDistribution(name=self.name, keys=self.keys)

        loc = sum_x / count
        var = max(sum_x2 / count - loc * loc, 0.0)
        scale = math.sqrt(max(3.0 * var / (math.pi * math.pi), self.min_scale * self.min_scale))
        return LogisticDistribution(loc=loc, scale=scale, name=self.name, keys=self.keys)


class LogisticDataEncoder(DataSequenceEncoder):
    """Encode logistic observations as a float array."""

    def __str__(self) -> str:
        return "LogisticDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, LogisticDataEncoder)

    def seq_encode(self, x: Sequence[float]) -> np.ndarray:
        """Encode observations as a floating-point array."""
        rv = np.asarray(x, dtype=np.float64)
        if rv.size and np.any(np.isnan(rv)):
            raise ValueError("LogisticDistribution requires finite or infinite real-valued observations.")
        return rv
