"""Rayleigh distributions over non-negative real values.

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


class RayleighDistribution(SequenceEncodableProbabilityDistribution):
    """Rayleigh distribution with scale sigma > 0."""

    @classmethod
    def compute_capabilities(cls):
        """Describe backend support for generated Rayleigh kernels."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch"), kernel_status="numba_adapter")

    @classmethod
    def compute_declaration(cls):
        """Return the structured compute declaration for Rayleigh distributions."""
        from mixle.stats.compute.declarations import (
            DistributionDeclaration,
            ExponentialFamilySpec,
            ParameterSpec,
            StatisticSpec,
        )

        return DistributionDeclaration(
            name="rayleigh",
            distribution_type=cls,
            parameters=(ParameterSpec("sigma", constraint="positive"),),
            statistics=(StatisticSpec("count"), StatisticSpec("sum2")),
            support="positive_real",
            exponential_family=ExponentialFamilySpec(
                sufficient_statistics=cls.exp_family_sufficient_statistics,
                natural_parameters=cls.exp_family_natural_parameters,
                log_partition=cls.exp_family_log_partition,
                base_measure=cls.exp_family_base_measure,
                legacy_sufficient_statistics=cls.exp_family_legacy_sufficient_statistics,
            ),
        )

    @staticmethod
    def exp_family_sufficient_statistics(x: tuple[Any, Any, Any], engine: Any) -> tuple[Any, ...]:
        """Return Rayleigh sufficient statistics for generated scoring."""
        return (engine.asarray(x[1]),)

    @staticmethod
    def exp_family_legacy_sufficient_statistics(
        x: tuple[Any, Any, Any], params: dict[str, Any], engine: Any
    ) -> tuple[Any, ...]:
        """Return per-row Rayleigh sufficient statistics in accumulator order."""
        vals2 = engine.asarray(x[1])
        return vals2 * 0.0 + engine.asarray(1.0), vals2

    @staticmethod
    def exp_family_natural_parameters(params: dict[str, Any], engine: Any) -> tuple[Any, ...]:
        """Return Rayleigh natural parameters for generated scoring."""
        sigma2 = params["sigma"] * params["sigma"]
        return (-engine.asarray(0.5) / sigma2,)

    @staticmethod
    def exp_family_log_partition(params: dict[str, Any], engine: Any) -> Any:
        """Return Rayleigh log partition for generated scoring."""
        sigma2 = params["sigma"] * params["sigma"]
        return engine.log(sigma2)

    @staticmethod
    def exp_family_base_measure(x: tuple[Any, Any, Any], engine: Any) -> Any:
        """Return Rayleigh support/base measure for generated scoring."""
        vals = engine.asarray(x[0])
        log_vals = engine.asarray(x[2])
        return engine.where(vals > 0.0, log_vals, engine.asarray(-np.inf))

    def __init__(self, sigma: float, name: str | None = None, keys: str | None = None) -> None:
        if sigma <= 0.0 or not np.isfinite(sigma):
            raise ValueError("RayleighDistribution requires sigma > 0.")
        self.sigma = float(sigma)
        self.sigma2 = self.sigma * self.sigma
        self.log_sigma2 = math.log(self.sigma2)
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        return "RayleighDistribution(%s, name=%s, keys=%s)" % (repr(self.sigma), repr(self.name), repr(self.keys))

    def density(self, x: float) -> float:
        """Return the probability density or mass at a single observation."""
        return math.exp(self.log_density(x))

    def log_density(self, x: float) -> float:
        """Return the log-density or log-mass at a single observation."""
        if x < 0.0:
            return -np.inf
        if x == 0.0:
            return -np.inf
        return math.log(x) - self.log_sigma2 - x * x / (2.0 * self.sigma2)

    def seq_log_density(self, x: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
        """Return vectorized log-density values for sequence-encoded observations."""
        xx, xx2, lx = x
        rv = lx - self.log_sigma2 - xx2 / (2.0 * self.sigma2)
        return np.where(xx >= 0.0, rv, -np.inf)

    @staticmethod
    def backend_log_density_from_params(vals: Any, vals2: Any, log_vals: Any, sigma: Any, engine: Any) -> Any:
        """Engine-neutral Rayleigh log-density from explicit parameters."""
        sigma2 = sigma * sigma
        rv = log_vals - engine.log(sigma2) - vals2 / (engine.asarray(2.0) * sigma2)
        return engine.where(vals >= 0.0, rv, engine.asarray(-np.inf))

    def backend_seq_log_density(self, x: tuple[Any, Any, Any], engine: Any) -> Any:
        """Engine-neutral vectorized log-density for encoded data."""
        return self.backend_log_density_from_params(
            engine.asarray(x[0]), engine.asarray(x[1]), engine.asarray(x[2]), engine.asarray(self.sigma), engine
        )

    @classmethod
    def backend_stacked_params(cls, dists: Sequence["RayleighDistribution"], engine: Any) -> dict[str, Any]:
        """Return stacked Rayleigh parameters for a homogeneous mixture kernel."""
        return {"sigma": engine.asarray([d.sigma for d in dists])}

    @classmethod
    def backend_stacked_log_density(cls, x: tuple[Any, Any, Any], params: dict[str, Any], engine: Any) -> Any:
        """Return an ``(n, k)`` matrix of Rayleigh log densities."""
        vals = engine.asarray(x[0])
        vals2 = engine.asarray(x[1])
        log_vals = engine.asarray(x[2])
        return cls.backend_log_density_from_params(
            vals[:, None], vals2[:, None], log_vals[:, None], params["sigma"][None, :], engine
        )

    def cdf(self, x: float) -> float:
        """Cumulative distribution function ``P(X <= x)`` (exact). The continuous 'index of' a value."""
        from scipy.stats import rayleigh as _sp

        return float(_sp.cdf(x, scale=self.sigma))

    def quantile(self, q: float) -> float:
        """Inverse CDF ``F^{-1}(q)``: the value at cumulative-probability index ``q`` (continuous unranking)."""
        from scipy.stats import rayleigh as _sp

        return float(_sp.ppf(q, scale=self.sigma))

    def mean(self) -> float:
        """Mean E[X] of the distribution."""
        return float(self.sigma * np.sqrt(np.pi / 2.0))

    def variance(self) -> float:
        """Variance Var[X] of the distribution."""
        return float((2.0 - np.pi / 2.0) * self.sigma * self.sigma)

    def entropy(self) -> float:
        """Differential entropy 1 + log(sigma/sqrt(2)) + gamma/2."""
        import math

        import numpy as np

        return float(1.0 + math.log(self.sigma / math.sqrt(2.0)) + np.euler_gamma / 2.0)

    def skewness(self) -> float:
        """Skewness 2*sqrt(pi)(pi-3)/(4-pi)^1.5."""
        import math

        return float(2.0 * math.sqrt(math.pi) * (math.pi - 3.0) / (4.0 - math.pi) ** 1.5)

    def kurtosis(self) -> float:
        """Excess kurtosis -(6pi^2-24pi+16)/(4-pi)^2."""
        import math

        return float(-(6.0 * math.pi**2 - 24.0 * math.pi + 16.0) / (4.0 - math.pi) ** 2)

    def mode(self) -> float:
        """Mode (sigma)."""
        return float(self.sigma)

    def sampler(self, seed: int | None = None) -> "RayleighSampler":
        """Return a sampler for drawing observations from this distribution."""
        return RayleighSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "RayleighEstimator":
        """Return an estimator for fitting this distribution from data."""
        if pseudo_count is None:
            return RayleighEstimator(name=self.name, keys=self.keys)
        return RayleighEstimator(pseudo_count=pseudo_count, suff_stat=self.sigma, name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> "RayleighDataEncoder":
        """Return the data encoder used by this distribution for vectorized methods."""
        return RayleighDataEncoder()


class RayleighSampler(DistributionSampler):
    """Draw iid Rayleigh observations."""

    def __init__(self, dist: RayleighDistribution, seed: int | None = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: int | None = None) -> float | np.ndarray:
        """Draw one sample or an array of iid samples."""
        return self.rng.rayleigh(scale=self.dist.sigma, size=size)


class RayleighAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate weighted squared observations."""

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.count = 0.0
        self.sum2 = 0.0
        self.name = name
        self.keys = keys

    def update(self, x: float, weight: float, estimate: RayleighDistribution | None) -> None:
        """Accumulate weighted squared observations for one sample."""
        if x < 0.0:
            raise ValueError("RayleighDistribution requires observations x >= 0.")
        self.count += weight
        self.sum2 += x * x * weight

    def initialize(self, x: float, weight: float, rng: RandomState | None) -> None:
        """Initialize statistics from one observation."""
        self.update(x, weight, None)

    def seq_update(
        self,
        x: tuple[np.ndarray, np.ndarray, np.ndarray],
        weights: np.ndarray,
        estimate: RayleighDistribution | None,
    ) -> None:
        """Accumulate weighted squared observations from encoded data."""
        self.count += np.sum(weights, dtype=np.float64)
        self.sum2 += np.dot(x[1], weights)

    def seq_initialize(
        self, x: tuple[np.ndarray, np.ndarray, np.ndarray], weights: np.ndarray, rng: RandomState | None
    ) -> None:
        """Initialize statistics from encoded observations."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[float, float]) -> "RayleighAccumulator":
        """Merge another Rayleigh sufficient-statistic tuple."""
        self.count += suff_stat[0]
        self.sum2 += suff_stat[1]
        return self

    def value(self) -> tuple[float, float]:
        """Return accumulated count and squared-observation sum."""
        return self.count, self.sum2

    def from_value(self, x: tuple[float, float]) -> "RayleighAccumulator":
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

    def acc_to_encoder(self) -> "RayleighDataEncoder":
        """Return the encoder used by this accumulator."""
        return RayleighDataEncoder()


class RayleighAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for RayleighAccumulator."""

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.name = name
        self.keys = keys

    def make(self) -> RayleighAccumulator:
        """Create a fresh Rayleigh accumulator."""
        return RayleighAccumulator(name=self.name, keys=self.keys)


class RayleighEstimator(ParameterEstimator):
    """Closed-form MLE estimator for Rayleigh scale."""

    def __init__(
        self,
        pseudo_count: float | None = None,
        suff_stat: float | None = None,
        min_sigma: float = 1.0e-8,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.min_sigma = min_sigma
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> RayleighAccumulatorFactory:
        """Return an accumulator factory for Rayleigh sufficient statistics."""
        return RayleighAccumulatorFactory(name=self.name, keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat: tuple[float, float]) -> RayleighDistribution:
        """Estimate the Rayleigh scale from weighted squared observations."""
        count, sum2 = suff_stat
        if self.pseudo_count is not None and self.suff_stat is not None:
            sum2 += self.pseudo_count * 2.0 * self.suff_stat * self.suff_stat
            count += self.pseudo_count
        sigma = math.sqrt(sum2 / (2.0 * count)) if count > 0.0 else 1.0
        return RayleighDistribution(max(sigma, self.min_sigma), name=self.name, keys=self.keys)


class RayleighDataEncoder(DataSequenceEncoder):
    """Encode Rayleigh observations with x, x**2, and log(x)."""

    def __str__(self) -> str:
        return "RayleighDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, RayleighDataEncoder)

    def seq_encode(self, x: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Encode observations as values, squared values, and log-values."""
        rv = np.asarray(x, dtype=np.float64)
        if rv.size and (np.any(rv < 0.0) or np.any(np.isnan(rv))):
            raise ValueError("RayleighDistribution requires observations x >= 0.")
        with np.errstate(divide="ignore"):
            lx = np.log(rv)
        return rv, rv * rv, lx
