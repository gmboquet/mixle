"""Create, estimate, and sample from a logarithmic (log-series) distribution on positive integers.

Defines the LogSeriesDistribution, LogSeriesSampler, LogSeriesAccumulatorFactory, LogSeriesAccumulator,
LogSeriesEstimator, and the LogSeriesDataEncoder classes for use with pysparkplug.

Data type: (int): a positive integer k >= 1. The LogSeriesDistribution with shape p in (0, 1) has
log-mass

        log(P(k; p)) = k*log(p) - log(k) - log(-log(1 - p)),    k = 1, 2, ...,

a one-parameter exponential family used for over-dispersed positive counts (species abundance,
word-frequency / type-token models). The mean is ``-p / ((1 - p) * log(1 - p))`` (a value in
``(1, inf)`` that increases with p), which the estimator inverts.

The per-row score is linear in the encoded ``k`` and ``log(k)`` fields once the scalar normalizer
``log(-log(1 - p))`` is precomputed, so the family gets generated NumPy, Torch, and Numba kernels.
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

_MIN_P = 1.0e-12
_MAX_P = 1.0 - 1.0e-12


def _mean_from_p(p: float) -> float:
    """Return the log-series mean -p / ((1 - p) * log(1 - p))."""
    return -p / ((1.0 - p) * math.log1p(-p))


def _solve_p(mean: float) -> float:
    """Invert the mean -> p for the log-series distribution (the mean increases monotonically in p)."""
    if mean <= 1.0:
        return _MIN_P
    lo, hi = _MIN_P, _MAX_P
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _mean_from_p(mid) < mean:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


class LogSeriesDistribution(SequenceEncodableProbabilityDistribution):
    """Logarithmic (log-series) distribution on k = 1, 2, ... with shape parameter p in (0, 1)."""

    @classmethod
    def compute_capabilities(cls):
        from pysp.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch"), kernel_status="numba_adapter")

    @classmethod
    def compute_declaration(cls):
        from pysp.stats.compute.declarations import DistributionDeclaration, ParameterSpec, StatisticSpec

        return DistributionDeclaration(
            name="log_series",
            distribution_type=cls,
            parameters=(
                ParameterSpec("log_p"),
                ParameterSpec("log_norm", constraint="real", differentiable=False),
            ),
            statistics=(StatisticSpec("count"), StatisticSpec("sum")),
            support="positive_integer",
            legacy_sufficient_statistics=cls.backend_legacy_sufficient_statistics,
        )

    @staticmethod
    def backend_legacy_sufficient_statistics(
        x: tuple[Any, Any], params: dict[str, Any], engine: Any
    ) -> tuple[Any, ...]:
        """Return per-row (count, k) sufficient statistics in accumulator order."""
        k = engine.asarray(x[0])
        return k * 0.0 + engine.asarray(1.0), k

    @staticmethod
    def backend_log_density_from_params(k: Any, log_k: Any, log_p: Any, log_norm: Any, engine: Any) -> Any:
        """Engine-neutral log-series log-mass from explicit parameters (linear in k and log k)."""
        return k * log_p - log_k - log_norm

    def __init__(self, p: float, name: str | None = None, keys: str | None = None) -> None:
        """LogSeriesDistribution for shape parameter p.

        Args:
            p (float): Shape parameter in (0, 1).
            name (Optional[str]): Assign a name to LogSeriesDistribution instance.
            keys (Optional[str]): Assign keys for merging sufficient statistics.

        Attributes:
            p (float): Shape parameter.
            log_p (float): Cached log(p).
            log_norm (float): Cached log(-log(1 - p)).

        """
        if not (0.0 < p < 1.0) or not np.isfinite(p):
            raise ValueError("LogSeriesDistribution requires p in (0, 1).")
        self.p = float(p)
        self.log_p = math.log(self.p)
        self.log_norm = math.log(-math.log1p(-self.p))
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        """Return string representation of LogSeriesDistribution object."""
        return "LogSeriesDistribution(%s, name=%s, keys=%s)" % (repr(self.p), repr(self.name), repr(self.keys))

    def density(self, x: int) -> float:
        """Return the probability mass at a single observation."""
        return math.exp(self.log_density(x))

    def log_density(self, x: int) -> float:
        """Return the log-mass at a single positive integer (or -inf off support)."""
        try:
            k = int(x)
        except (TypeError, ValueError):
            return -np.inf
        if k < 1 or k != x:
            return -np.inf
        return k * self.log_p - math.log(k) - self.log_norm

    def seq_log_density(self, x: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """Return vectorized log-mass values for sequence-encoded (k, log k) observations."""
        k, log_k = x
        return k * self.log_p - log_k - self.log_norm

    def backend_seq_log_density(self, x: tuple[Any, Any], engine: Any) -> Any:
        """Engine-neutral vectorized log-mass for encoded data."""
        return self.backend_log_density_from_params(
            engine.asarray(x[0]),
            engine.asarray(x[1]),
            engine.asarray(self.log_p),
            engine.asarray(self.log_norm),
            engine,
        )

    @classmethod
    def backend_stacked_params(cls, dists: Sequence["LogSeriesDistribution"], engine: Any) -> dict[str, Any]:
        """Return stacked parameters for a homogeneous mixture kernel."""
        return {
            "log_p": engine.asarray([d.log_p for d in dists]),
            "log_norm": engine.asarray([d.log_norm for d in dists]),
        }

    @classmethod
    def backend_stacked_log_density(cls, x: tuple[Any, Any], params: dict[str, Any], engine: Any) -> Any:
        """Return an ``(n, k)`` matrix of log-series log densities."""
        k = engine.asarray(x[0])[:, None]
        log_k = engine.asarray(x[1])[:, None]
        return cls.backend_log_density_from_params(
            k, log_k, params["log_p"][None, :], params["log_norm"][None, :], engine
        )

    @classmethod
    def backend_stacked_sufficient_statistics(
        cls, x: tuple[Any, Any], weights: Any, params: dict[str, Any], engine: Any
    ) -> tuple[Any, Any]:
        """Return stacked sufficient statistics using engine-resident arrays."""
        k = engine.asarray(x[0])
        ww = engine.asarray(weights)
        return engine.sum(ww, axis=0), engine.sum(ww * k[:, None], axis=0)

    def sampler(self, seed: int | None = None) -> "LogSeriesSampler":
        """Return a sampler for drawing observations from this distribution."""
        return LogSeriesSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "LogSeriesEstimator":
        """Return an estimator for fitting this distribution from data."""
        if pseudo_count is None:
            return LogSeriesEstimator(name=self.name, keys=self.keys)
        return LogSeriesEstimator(
            pseudo_count=pseudo_count, suff_stat=_mean_from_p(self.p), name=self.name, keys=self.keys
        )

    def dist_to_encoder(self) -> "LogSeriesDataEncoder":
        """Return the data encoder used by this distribution for vectorized methods."""
        return LogSeriesDataEncoder()


class LogSeriesSampler(DistributionSampler):
    """Draw iid log-series observations."""

    def __init__(self, dist: LogSeriesDistribution, seed: int | None = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist
        self.seed = seed

    def sample(self, size: int | None = None) -> int | np.ndarray:
        """Draw ``size`` iid positive integers (an int when ``size`` is None)."""
        rv = self.rng.logseries(self.dist.p, size=size)
        return int(rv) if size is None else rv


class LogSeriesAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate weighted count and sum for log-series estimation."""

    def __init__(self, keys: str | None = None) -> None:
        self.count = 0.0
        self.sum = 0.0
        self.key = keys

    def update(self, x: int, weight: float, estimate: LogSeriesDistribution | None) -> None:
        if int(x) < 1:
            raise ValueError("LogSeriesDistribution has support k >= 1.")
        self.count += weight
        self.sum += float(x) * weight

    def initialize(self, x: int, weight: float, rng: RandomState | None) -> None:
        self.update(x, weight, None)

    def seq_update(
        self, x: tuple[np.ndarray, np.ndarray], weights: np.ndarray, estimate: LogSeriesDistribution | None
    ) -> None:
        self.count += np.sum(weights, dtype=np.float64)
        self.sum += np.dot(x[0], weights)

    def seq_initialize(self, x: tuple[np.ndarray, np.ndarray], weights: np.ndarray, rng: RandomState | None) -> None:
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[float, float]) -> "LogSeriesAccumulator":
        self.count += suff_stat[0]
        self.sum += suff_stat[1]
        return self

    def value(self) -> tuple[float, float]:
        return self.count, self.sum

    def from_value(self, x: tuple[float, float]) -> "LogSeriesAccumulator":
        self.count, self.sum = x
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

    def acc_to_encoder(self) -> "LogSeriesDataEncoder":
        return LogSeriesDataEncoder()


class LogSeriesAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for LogSeriesAccumulator."""

    def __init__(self, keys: str | None = None) -> None:
        self.keys = keys

    def make(self) -> LogSeriesAccumulator:
        return LogSeriesAccumulator(keys=self.keys)


class LogSeriesEstimator(ParameterEstimator):
    """Maximum-likelihood estimator for the log-series shape p (inverts the mean -> p relation)."""

    def __init__(
        self,
        pseudo_count: float | None = None,
        suff_stat: float | None = None,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> LogSeriesAccumulatorFactory:
        return LogSeriesAccumulatorFactory(keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat: tuple[float, float]) -> LogSeriesDistribution:
        count, total = suff_stat
        if self.pseudo_count is not None and self.suff_stat is not None:
            total += self.pseudo_count * self.suff_stat
            count += self.pseudo_count
        if count <= 0.0:
            return LogSeriesDistribution(0.5, name=self.name, keys=self.keys)
        mean = total / count
        return LogSeriesDistribution(_solve_p(mean), name=self.name, keys=self.keys)


class LogSeriesDataEncoder(DataSequenceEncoder):
    """Encode log-series observations as (k, log k) pairs."""

    def __str__(self) -> str:
        return "LogSeriesDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, LogSeriesDataEncoder)

    def seq_encode(self, x: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        rv = np.asarray(x, dtype=np.float64)
        if rv.size and (np.any(rv < 1.0) or np.any(rv != np.floor(rv)) or np.any(~np.isfinite(rv))):
            raise ValueError("LogSeriesDistribution has support on the positive integers k >= 1.")
        return rv, np.log(rv)
