"""Logarithmic (log-series) distributions over positive integers.

Observations are positive integers ``k >= 1``. A log-series distribution with shape ``p in (0, 1)`` has
log-mass

        log(P(k; p)) = k*log(p) - log(k) - log(-log(1 - p)),    k = 1, 2, ...,

a one-parameter exponential family used for over-dispersed positive counts (species abundance,
word-frequency / type-token models). The mean is ``-p / ((1 - p) * log(1 - p))`` (a value in
``(1, inf)`` that increases with p), which the estimator inverts.

The per-row score is linear in the encoded ``k`` and ``log(k)`` fields once the scalar normalizer
``log(-log(1 - p))`` is precomputed, so the family gets generated NumPy, Torch, and Numba kernels.


Reference: Fisher, Corbet & Williams, 'The relation between the number of species and the number of individuals...', J. Animal Ecology (1943).
"""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import RandomState

from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionEnumerator,
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
        """Describe backend support for generated log-series kernels."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch"), kernel_status="numba_adapter")

    @classmethod
    def compute_declaration(cls):
        """Return the structured compute declaration for log-series distributions."""
        from mixle.stats.compute.declarations import (
            DistributionDeclaration,
            ExponentialFamilySpec,
            ParameterSpec,
            StatisticSpec,
        )

        return DistributionDeclaration(
            name="log_series",
            distribution_type=cls,
            parameters=(
                ParameterSpec("log_p"),
                ParameterSpec("log_norm", constraint="real", differentiable=False),
            ),
            statistics=(StatisticSpec("count"), StatisticSpec("sum")),
            support="positive_integer",
            exponential_family=ExponentialFamilySpec(
                sufficient_statistics=cls.exp_family_sufficient_statistics,
                natural_parameters=cls.exp_family_natural_parameters,
                log_partition=cls.exp_family_log_partition,
                base_measure=cls.exp_family_base_measure,
                legacy_sufficient_statistics=cls.backend_legacy_sufficient_statistics,
            ),
        )

    @staticmethod
    def backend_legacy_sufficient_statistics(
        x: tuple[Any, Any], params: dict[str, Any], engine: Any
    ) -> tuple[Any, ...]:
        """Return per-row (count, k) sufficient statistics in accumulator order."""
        k = engine.asarray(x[0])
        return k * 0.0 + engine.asarray(1.0), k

    @staticmethod
    def exp_family_sufficient_statistics(x: tuple[Any, Any], engine: Any) -> tuple[Any, ...]:
        """Return the log-series sufficient statistic ``T(k) = (k,)``."""
        return (engine.asarray(x[0]),)

    @staticmethod
    def exp_family_natural_parameters(params: dict[str, Any], engine: Any) -> tuple[Any, ...]:
        """Return the log-series natural parameter ``eta = log(p)``."""
        return (params["log_p"],)

    @staticmethod
    def exp_family_log_partition(params: dict[str, Any], engine: Any) -> Any:
        """Return the log-series log partition ``A = log(-log(1 - p))``."""
        return params["log_norm"]

    @staticmethod
    def exp_family_base_measure(x: tuple[Any, Any], engine: Any) -> Any:
        """Return the log-series base measure ``log h(k) = -log(k)`` (independent of p, fixed base)."""
        return -engine.asarray(x[1])

    @staticmethod
    def exp_family_from_natural(eta: Any) -> "LogSeriesDistribution":
        """Return the log-series with natural parameter ``eta = log(p)`` (so ``p = exp(eta)``)."""
        import numpy as _np

        return LogSeriesDistribution(float(_np.exp(float(eta[0]))))

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
        """Return a constructor-style representation of the log-series distribution."""
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

    def mean(self) -> float:
        """Mean E[X] = -p / ((1-p) log(1-p))."""
        import math

        l1m = math.log(1.0 - self.p)
        return float(-self.p / ((1.0 - self.p) * l1m))

    def variance(self) -> float:
        """Variance Var[X] = -p (p + log(1-p)) / ((1-p)^2 log(1-p)^2)."""
        import math

        p = self.p
        l1m = math.log(1.0 - p)
        return float(-p * (p + l1m) / ((1.0 - p) ** 2 * l1m * l1m))

    def cdf(self, x: float) -> float:
        """Cumulative distribution function P(X <= x), support x >= 1 (via scipy logser)."""
        import math

        from scipy.stats import logser

        k = math.floor(float(x))
        return float(logser.cdf(k, self.p)) if k >= 1 else 0.0

    def quantile(self, q: float) -> float:
        """Inverse CDF F^{-1}(q) (via scipy logser)."""
        from scipy.stats import logser

        return float(logser.ppf(float(q), self.p))

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

    def enumerator(self) -> "LogSeriesEnumerator":
        """Return an enumerator over k = 1, 2, ... in descending probability order."""
        return LogSeriesEnumerator(self)


class LogSeriesEnumerator(DistributionEnumerator):
    """Enumerate log-series support values in descending probability order.

    The log-series pmf ``p^k / (k * -log(1 - p))`` is strictly decreasing in ``k`` (each step
    multiplies by ``p * k / (k + 1) < 1``), so value order 1, 2, 3, ... IS descending-probability
    order. The iterator is infinite.
    """

    def __init__(self, dist: LogSeriesDistribution) -> None:
        super().__init__(dist)
        self._k = 1

    def __next__(self) -> tuple[int, float]:
        k = self._k
        self._k += 1
        return (k, self.dist.log_density(k))


class LogSeriesSampler(DistributionSampler):
    """Draw iid log-series observations."""

    def __init__(self, dist: LogSeriesDistribution, seed: int | None = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist
        self.seed = seed

    def sample(self, size: int | None = None, *, batched: bool = True) -> int | np.ndarray:
        """Draw ``size`` iid positive integers (an int when ``size`` is None)."""
        rv = self.rng.logseries(self.dist.p, size=size)
        return int(rv) if size is None else rv


class LogSeriesAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate weighted count and sum for log-series estimation."""

    def __init__(self, keys: str | None = None) -> None:
        self.count = 0.0
        self.sum = 0.0
        self.keys = keys

    def update(self, x: int, weight: float, estimate: LogSeriesDistribution | None) -> None:
        """Accumulate weighted count and total for one positive integer."""
        if int(x) < 1:
            raise ValueError("LogSeriesDistribution has support k >= 1.")
        self.count += weight
        self.sum += float(x) * weight

    def initialize(self, x: int, weight: float, rng: RandomState | None) -> None:
        """Initialize statistics from one observation."""
        self.update(x, weight, None)

    def seq_update(
        self, x: tuple[np.ndarray, np.ndarray], weights: np.ndarray, estimate: LogSeriesDistribution | None
    ) -> None:
        """Accumulate weighted count and total from encoded observations."""
        self.count += np.sum(weights, dtype=np.float64)
        self.sum += np.dot(x[0], weights)

    def seq_initialize(self, x: tuple[np.ndarray, np.ndarray], weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize statistics from encoded observations."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[float, float]) -> "LogSeriesAccumulator":
        """Merge another log-series sufficient-statistic tuple."""
        self.count += suff_stat[0]
        self.sum += suff_stat[1]
        return self

    def value(self) -> tuple[float, float]:
        """Return accumulated count and integer total."""
        return self.count, self.sum

    def from_value(self, x: tuple[float, float]) -> "LogSeriesAccumulator":
        """Replace accumulator contents from a sufficient-statistic tuple."""
        self.count, self.sum = x
        return self

    def acc_to_encoder(self) -> "LogSeriesDataEncoder":
        """Return the encoder used by this accumulator."""
        return LogSeriesDataEncoder()


class LogSeriesAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for LogSeriesAccumulator."""

    def __init__(self, keys: str | None = None) -> None:
        self.keys = keys

    def make(self) -> LogSeriesAccumulator:
        """Create a fresh log-series accumulator."""
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
        """Return an accumulator factory for log-series count statistics."""
        return LogSeriesAccumulatorFactory(keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat: tuple[float, float]) -> LogSeriesDistribution:
        """Estimate the shape parameter by inverting the weighted mean."""
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
        """Encode observations as integer values and log-values."""
        rv = np.asarray(x, dtype=np.float64)
        if rv.size and (np.any(rv < 1.0) or np.any(rv != np.floor(rv)) or np.any(~np.isfinite(rv))):
            raise ValueError("LogSeriesDistribution has support on the positive integers k >= 1.")
        return rv, np.log(rv)
