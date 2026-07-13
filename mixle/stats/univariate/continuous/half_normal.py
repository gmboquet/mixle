"""Half-normal distributions over non-negative real values.

Observations are floats ``x >= 0``. A half-normal distribution with scale ``sigma > 0`` has log-density

        log(f(x; sigma)) = 0.5*log(2/pi) - log(sigma) - x**2 / (2*sigma**2),    for x >= 0.0,

    and -np.inf otherwise.

The half-normal is a one-parameter exponential family with sufficient statistic x**2:

    log(f) = base(x) + eta*x**2 - A(sigma),
        base(x)   = 0.5*log(2/pi),    (a constant on the support x >= 0)
        eta       = -1 / (2*sigma**2),
        A(sigma)  = log(sigma).

Declaring those pieces gives the family generated NumPy/Torch/Numba scoring through the shared
exponential-family compute path, exactly as for the Gamma and inverse Gaussian families.


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

_MIN_SIGMA = float(np.finfo(float).tiny)
_HALF_LOG_2_OVER_PI = 0.5 * math.log(2.0 / math.pi)


class HalfNormalDistribution(SequenceEncodableProbabilityDistribution):
    """Half-normal distribution with scale sigma > 0 on x >= 0."""

    @classmethod
    def compute_capabilities(cls):
        """Describe backend support for generated half-normal kernels."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch"), kernel_status="numba_adapter")

    @classmethod
    def compute_declaration(cls):
        """Return the structured compute declaration for half-normal distributions."""
        from mixle.stats.compute.declarations import (
            DistributionDeclaration,
            ExponentialFamilySpec,
            ParameterSpec,
            StatisticSpec,
        )

        return DistributionDeclaration(
            name="half_normal",
            distribution_type=cls,
            parameters=(ParameterSpec("sigma", constraint="positive"),),
            statistics=(StatisticSpec("count"), StatisticSpec("sum2")),
            support="non_negative_real",
            exponential_family=ExponentialFamilySpec(
                sufficient_statistics=cls.exp_family_sufficient_statistics,
                natural_parameters=cls.exp_family_natural_parameters,
                log_partition=cls.exp_family_log_partition,
                base_measure=cls.exp_family_base_measure,
                legacy_sufficient_statistics=cls.exp_family_legacy_sufficient_statistics,
            ),
        )

    @staticmethod
    def exp_family_sufficient_statistics(x: tuple[Any, Any], engine: Any) -> tuple[Any, ...]:
        """Return the (x**2,) sufficient statistic for generated scoring."""
        _, sq_vals = x
        return (engine.asarray(sq_vals),)

    @staticmethod
    def exp_family_natural_parameters(params: dict[str, Any], engine: Any) -> tuple[Any, ...]:
        """Return the (-1/(2*sigma^2),) natural parameter for generated scoring."""
        sigma = params["sigma"]
        return (-engine.asarray(1.0) / (engine.asarray(2.0) * sigma * sigma),)

    @staticmethod
    def exp_family_log_partition(params: dict[str, Any], engine: Any) -> Any:
        """Return the half-normal log partition log(sigma)."""
        return engine.log(params["sigma"])

    @staticmethod
    def exp_family_base_measure(x: tuple[Any, Any], engine: Any) -> Any:
        """Return the support base measure 0.5*log(2/pi) (or -inf off support x >= 0)."""
        vals = engine.asarray(x[0])
        base = engine.asarray(_HALF_LOG_2_OVER_PI) + vals * 0.0
        return engine.where(vals >= 0.0, base, engine.asarray(-np.inf))

    @staticmethod
    def exp_family_legacy_sufficient_statistics(
        x: tuple[Any, Any], params: dict[str, Any], engine: Any
    ) -> tuple[Any, ...]:
        """Return per-row (count, x**2) sufficient statistics in accumulator order."""
        vals = engine.asarray(x[0])
        sq_vals = engine.asarray(x[1])
        return vals * 0.0 + engine.asarray(1.0), sq_vals

    def __init__(self, sigma: float, name: str | None = None, keys: str | None = None) -> None:
        """HalfNormalDistribution for scale sigma.

        Args:
            sigma (float): Positive real-valued scale parameter.
            name (Optional[str]): Assign a name to HalfNormalDistribution instance.
            keys (Optional[str]): Assign keys for merging sufficient statistics.

        Attributes:
            sigma (float): Positive real-valued scale parameter.
            log_sigma (float): Cached log(sigma).
            name (Optional[str]): Optional distribution name.
            keys (Optional[str]): Key for merging sufficient statistics.

        """
        if sigma <= 0.0 or not np.isfinite(sigma):
            raise ValueError("HalfNormalDistribution requires finite sigma > 0.")
        self.sigma = float(sigma)
        self.log_sigma = math.log(self.sigma)
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        """Return a constructor-style representation of the half-normal distribution."""
        return "HalfNormalDistribution(%s, name=%s, keys=%s)" % (
            repr(self.sigma),
            repr(self.name),
            repr(self.keys),
        )

    def density(self, x: float) -> float:
        """Return the probability density at a single observation."""
        return math.exp(self.log_density(x))

    def log_density(self, x: float) -> float:
        """Return the log-density at a single observation (or -inf off support)."""
        try:
            xx = float(x)
        except (TypeError, ValueError):
            return -np.inf
        if not np.isfinite(xx) or xx < 0.0:
            return -np.inf
        return _HALF_LOG_2_OVER_PI - self.log_sigma - xx * xx / (2.0 * self.sigma * self.sigma)

    def seq_log_density(self, x: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """Return vectorized log-density values for sequence-encoded observations.

        Args:
            x (Tuple[ndarray, ndarray]): Tuple of observations and squared observations produced by
                the HalfNormalDataEncoder.

        Returns:
            Numpy array of log-density values, with -inf entries off the non-negative support.

        """
        vals, sq_vals = x
        rv = _HALF_LOG_2_OVER_PI - self.log_sigma - sq_vals / (2.0 * self.sigma * self.sigma)
        return np.where(np.isfinite(vals) & (vals >= 0.0), rv, -np.inf)

    @staticmethod
    def backend_log_density_from_params(vals: Any, sq_vals: Any, sigma: Any, engine: Any) -> Any:
        """Engine-neutral half-normal log-density from explicit parameters."""
        rv = engine.asarray(_HALF_LOG_2_OVER_PI) - engine.log(sigma) - sq_vals / (engine.asarray(2.0) * sigma * sigma)
        return engine.where(vals >= 0.0, rv, engine.asarray(-np.inf))

    def backend_seq_log_density(self, x: tuple[Any, Any], engine: Any) -> Any:
        """Engine-neutral vectorized log-density for encoded data."""
        vals = engine.asarray(x[0])
        sq_vals = engine.asarray(x[1])
        return self.backend_log_density_from_params(vals, sq_vals, engine.asarray(self.sigma), engine)

    @classmethod
    def backend_stacked_params(cls, dists: Sequence["HalfNormalDistribution"], engine: Any) -> dict[str, Any]:
        """Return stacked parameters for a homogeneous mixture kernel."""
        return {"sigma": engine.asarray([d.sigma for d in dists])}

    @classmethod
    def backend_stacked_log_density(cls, x: tuple[Any, Any], params: dict[str, Any], engine: Any) -> Any:
        """Return an ``(n, k)`` matrix of half-normal log densities."""
        vals = engine.asarray(x[0])
        sq_vals = engine.asarray(x[1])
        return cls.backend_log_density_from_params(vals[:, None], sq_vals[:, None], params["sigma"][None, :], engine)

    @classmethod
    def backend_stacked_sufficient_statistics(
        cls, x: tuple[Any, Any], weights: Any, params: dict[str, Any], engine: Any
    ) -> tuple[Any, Any]:
        """Return stacked sufficient statistics using engine-resident arrays."""
        sq_vals = engine.asarray(x[1])
        ww = engine.asarray(weights)
        return engine.sum(ww, axis=0), engine.sum(ww * sq_vals[:, None], axis=0)

    def cdf(self, x: float) -> float:
        """Cumulative distribution function P(X <= x) (0 for x < 0)."""
        from scipy.special import erf

        x = float(x)
        return float(erf(x / (self.sigma * math.sqrt(2.0)))) if x > 0.0 else 0.0

    def quantile(self, q: float) -> float:
        """Inverse CDF F^{-1}(q)."""
        from scipy.special import erfinv

        return float(self.sigma * math.sqrt(2.0) * erfinv(float(q)))

    def mean(self) -> float:
        """Mean E[X] = sigma * sqrt(2/pi)."""
        import math

        return float(self.sigma * math.sqrt(2.0 / math.pi))

    def variance(self) -> float:
        """Variance Var[X] = sigma^2 (1 - 2/pi)."""
        import math

        return float(self.sigma * self.sigma * (1.0 - 2.0 / math.pi))

    def entropy(self) -> float:
        """Differential entropy 0.5*log(pi*sigma^2/2) + 1/2."""
        import math

        return float(0.5 * math.log(math.pi * self.sigma**2 / 2.0) + 0.5)

    def sampler(self, seed: int | None = None) -> "HalfNormalSampler":
        """Return a sampler for drawing observations from this distribution."""
        return HalfNormalSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "HalfNormalEstimator":
        """Return an estimator for fitting this distribution from data.

        Args:
            pseudo_count (Optional[float]): Re-weight the second moment toward this instance's own
                E[x**2] = sigma**2 when not None (a simple ridge toward the current parameter).

        Returns:
            HalfNormalEstimator object.

        """
        if pseudo_count is None:
            return HalfNormalEstimator(name=self.name, keys=self.keys)
        return HalfNormalEstimator(
            pseudo_count=pseudo_count, suff_stat=self.sigma * self.sigma, name=self.name, keys=self.keys
        )

    def dist_to_encoder(self) -> "HalfNormalDataEncoder":
        """Return the data encoder used by this distribution for vectorized methods."""
        return HalfNormalDataEncoder()


class HalfNormalSampler(DistributionSampler):
    """Draw iid half-normal observations as |N(0, sigma**2)|."""

    def __init__(self, dist: HalfNormalDistribution, seed: int | None = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist
        self.seed = seed

    def sample(self, size: int | None = None) -> float | np.ndarray:
        """Draw ``size`` iid observations (a float when ``size`` is None)."""
        return np.abs(self.rng.normal(0.0, self.dist.sigma, size=size))


class HalfNormalAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate weighted count and sum of squares for half-normal estimation."""

    def __init__(self, keys: str | None = None) -> None:
        self.count = 0.0
        self.sum2 = 0.0
        self.keys = keys

    def update(self, x: float, weight: float, estimate: HalfNormalDistribution | None) -> None:
        """Accumulate weighted squared observations for one non-negative sample."""
        if x < 0.0 or not np.isfinite(x):
            raise ValueError("HalfNormalDistribution has support x >= 0.")
        self.count += weight
        self.sum2 += x * x * weight

    def initialize(self, x: float, weight: float, rng: RandomState | None) -> None:
        """Initialize statistics from one observation."""
        self.update(x, weight, None)

    def seq_update(
        self, x: tuple[np.ndarray, np.ndarray], weights: np.ndarray, estimate: HalfNormalDistribution | None
    ) -> None:
        """Accumulate weighted squared observations from encoded data."""
        _, sq_vals = x
        self.sum2 += np.dot(sq_vals, weights)
        self.count += np.sum(weights, dtype=np.float64)

    def seq_initialize(self, x: tuple[np.ndarray, np.ndarray], weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize statistics from encoded observations."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[float, float]) -> "HalfNormalAccumulator":
        """Merge another half-normal sufficient-statistic tuple."""
        self.count += suff_stat[0]
        self.sum2 += suff_stat[1]
        return self

    def value(self) -> tuple[float, float]:
        """Return count and squared-observation sum."""
        return self.count, self.sum2

    def from_value(self, x: tuple[float, float]) -> "HalfNormalAccumulator":
        """Replace accumulator contents from a sufficient-statistic tuple."""
        self.count = x[0]
        self.sum2 = x[1]
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

    def acc_to_encoder(self) -> "HalfNormalDataEncoder":
        """Return the encoder used by this accumulator."""
        return HalfNormalDataEncoder()


class HalfNormalAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for HalfNormalAccumulator."""

    def __init__(self, keys: str | None = None) -> None:
        self.keys = keys

    def make(self) -> HalfNormalAccumulator:
        """Create a fresh half-normal accumulator."""
        return HalfNormalAccumulator(keys=self.keys)


class HalfNormalEstimator(ParameterEstimator):
    """Maximum-likelihood estimator for the half-normal scale: sigma = sqrt(mean(x**2))."""

    def __init__(
        self,
        pseudo_count: float | None = None,
        suff_stat: float | None = None,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        """Create an estimator for half-normal scale parameters.

        Args:
            pseudo_count (Optional[float]): Re-weight the prior second moment in ``suff_stat`` when
                not None.
            suff_stat (Optional[float]): Prior E[x**2] target for the pseudo-count ridge.
            name (Optional[str]): Assign a name to the estimator.
            keys (Optional[str]): Assign keys for combining sufficient statistics.

        """
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> HalfNormalAccumulatorFactory:
        """Return an accumulator factory for half-normal statistics."""
        return HalfNormalAccumulatorFactory(keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat: tuple[float, float]) -> HalfNormalDistribution:
        """Estimate the half-normal scale from weighted squared observations."""
        count, sum2 = suff_stat
        if self.pseudo_count is not None and self.suff_stat is not None:
            sum2 += self.pseudo_count * self.suff_stat
            count += self.pseudo_count

        if count <= 0.0 or sum2 <= 0.0 or not np.isfinite(sum2):
            return HalfNormalDistribution(1.0, name=self.name, keys=self.keys)

        sigma = max(math.sqrt(sum2 / count), _MIN_SIGMA)
        return HalfNormalDistribution(sigma, name=self.name, keys=self.keys)


class HalfNormalDataEncoder(DataSequenceEncoder):
    """Encode half-normal observations as x and x**2."""

    def __str__(self) -> str:
        return "HalfNormalDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, HalfNormalDataEncoder)

    def seq_encode(self, x: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        """Encode observations as values and squared values."""
        rv = np.asarray(x, dtype=np.float64)
        if rv.size and (np.any(rv < 0.0) or np.any(np.isnan(rv))):
            raise ValueError("HalfNormalDistribution has support x >= 0.")
        return rv, rv * rv
