"""Generalized Pareto distribution (GPD): the peaks-over-threshold law of extreme exceedances.

By the Pickands-Balkema-de Haan theorem the distribution of exceedances over a high threshold of
almost any distribution converges to a GPD, which makes it the workhorse for modelling tail risk
(hydrology, finance, reliability). With threshold ``loc = mu``, scale ``sigma > 0`` and shape ``xi``,
for ``y = x - mu >= 0``:

    f(x) = (1/sigma) (1 + xi * y / sigma) ** (-1/xi - 1)   (xi != 0),
    f(x) = (1/sigma) exp(-y / sigma)                       (xi == 0, the exponential tail).

``xi > 0`` is a heavy (Pareto) tail, ``xi = 0`` exponential, ``xi < 0`` a tail with a finite upper
endpoint at ``mu - sigma/xi``. The threshold ``mu`` is treated as a *fixed, known* level (chosen, not
fit -- the standard peaks-over-threshold setup); ``sigma`` and ``xi`` are fit by method of moments,
which is closed-form: ``xi = (1 - m^2/v)/2`` and ``sigma = m (1 - xi)`` from the exceedance mean ``m``
and variance ``v`` (valid for ``xi < 1/2``, where the variance is finite).


Reference: Pickands, 'Statistical inference using extreme order statistics', Ann. Statist. (1975).
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

_XI_TOL = 1.0e-8  # |xi| below this is treated as the exponential limit


class GeneralizedParetoDistribution(SequenceEncodableProbabilityDistribution):
    """Generalized Pareto distribution with threshold ``loc``, scale ``> 0`` and shape ``xi``."""

    def __init__(
        self, scale: float, shape: float, loc: float = 0.0, name: str | None = None, keys: str | None = None
    ) -> None:
        if scale <= 0.0 or not np.isfinite(scale) or not np.isfinite(shape) or not np.isfinite(loc):
            raise ValueError("GeneralizedParetoDistribution requires finite parameters and scale > 0.")
        self.scale = float(scale)
        self.shape = float(shape)  # xi
        self.loc = float(loc)  # mu (threshold)
        self.log_scale = math.log(self.scale)
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        return "GeneralizedParetoDistribution(%s, %s, loc=%s, name=%s, keys=%s)" % (
            repr(self.scale),
            repr(self.shape),
            repr(self.loc),
            repr(self.name),
            repr(self.keys),
        )

    def _upper(self) -> float:
        """Upper endpoint of the support (``inf`` unless ``xi < 0``)."""
        return self.loc - self.scale / self.shape if self.shape < -_XI_TOL else math.inf

    def density(self, x: float) -> float:
        """Return the probability density at a single observation."""
        return math.exp(self.log_density(x))

    def log_density(self, x: float) -> float:
        """Return the log-density at a single observation (``-inf`` outside the support)."""
        y = x - self.loc
        if y < 0.0 or x > self._upper():
            return -np.inf
        if abs(self.shape) < _XI_TOL:
            return -self.log_scale - y / self.scale
        t = 1.0 + self.shape * y / self.scale
        if t <= 0.0:
            return -np.inf
        return -self.log_scale - (1.0 / self.shape + 1.0) * math.log(t)

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Return vectorized log-density values for sequence-encoded observations."""
        y = np.asarray(x, dtype=np.float64) - self.loc
        if abs(self.shape) < _XI_TOL:
            rv = -self.log_scale - y / self.scale
        else:
            t = 1.0 + self.shape * y / self.scale
            with np.errstate(divide="ignore", invalid="ignore"):
                rv = -self.log_scale - (1.0 / self.shape + 1.0) * np.log(t)
            rv = np.where(t <= 0.0, -np.inf, rv)
        return np.where(y < 0.0, -np.inf, rv)

    # --- compute-engine backend (numpy + torch/GPU): scoring + sufficient statistics in engine ops ---
    @classmethod
    def compute_capabilities(cls):
        """Describe backend support for generated generalized-Pareto kernels."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch"), kernel_status="generic")

    @classmethod
    def compute_declaration(cls):
        """Return the structured compute declaration for generalized Pareto distributions."""
        from mixle.stats.compute.declarations import DistributionDeclaration, ParameterSpec, StatisticSpec

        return DistributionDeclaration(
            name="generalized_pareto",
            distribution_type=cls,
            parameters=(ParameterSpec("scale", constraint="positive"), ParameterSpec("shape"), ParameterSpec("loc")),
            statistics=(StatisticSpec("sum"), StatisticSpec("sum2"), StatisticSpec("count")),
            support="real",
            legacy_sufficient_statistics=cls.backend_legacy_sufficient_statistics,
        )

    @staticmethod
    def backend_legacy_sufficient_statistics(x: Any, params: dict[str, Any], engine: Any) -> tuple[Any, ...]:
        """Per-row GPD moment sums in accumulator order ``(sum, sum2, count)``."""
        xx = engine.asarray(x)
        return xx, xx * xx, xx * 0.0 + engine.asarray(1.0)

    @staticmethod
    def backend_log_density_from_params(x: Any, scale: Any, shape: Any, loc: Any, engine: Any) -> Any:
        """Engine-neutral GPD log-density; the ``|xi| < tol`` exponential limit is selected per element."""
        y = x - loc
        neg_inf = engine.asarray(float("-inf"))
        is_limit = engine.abs(shape) < _XI_TOL
        xi_safe = engine.where(is_limit, engine.asarray(1.0), shape)  # keep the general branch NaN-free
        t = 1.0 + xi_safe * y / scale
        t_pos = engine.where(t > 0.0, t, engine.asarray(1.0))
        general = -engine.log(scale) - (1.0 / xi_safe + 1.0) * engine.log(t_pos)
        general = engine.where(t > 0.0, general, neg_inf)
        limit = -engine.log(scale) - y / scale
        rv = engine.where(is_limit, limit, general)
        return engine.where(y < 0.0, neg_inf, rv)

    def backend_seq_log_density(self, x: Any, engine: Any) -> Any:
        """Engine-neutral vectorized log-density for encoded data."""
        return self.backend_log_density_from_params(
            engine.asarray(x),
            engine.asarray(self.scale),
            engine.asarray(self.shape),
            engine.asarray(self.loc),
            engine,
        )

    @classmethod
    def backend_stacked_params(cls, dists: Sequence["GeneralizedParetoDistribution"], engine: Any) -> dict[str, Any]:
        """Stacked GPD parameters for a homogeneous mixture kernel."""
        return {
            "scale": engine.asarray([d.scale for d in dists]),
            "shape": engine.asarray([d.shape for d in dists]),
            "loc": engine.asarray([d.loc for d in dists]),
        }

    @classmethod
    def backend_stacked_log_density(cls, x: np.ndarray, params: dict[str, Any], engine: Any) -> Any:
        """Return an ``(n, k)`` matrix of GPD log densities."""
        xx = engine.asarray(x)
        return cls.backend_log_density_from_params(
            xx[:, None], params["scale"][None, :], params["shape"][None, :], params["loc"][None, :], engine
        )

    @classmethod
    def backend_stacked_sufficient_statistics(
        cls, x: np.ndarray, weights: Any, params: dict[str, Any], engine: Any
    ) -> tuple[Any, Any, Any]:
        """Stacked GPD moment sums ``(sum, sum2, count)`` using engine-resident arrays."""
        xx = engine.asarray(x)
        ww = engine.asarray(weights)
        return (
            engine.sum(ww * xx[:, None], axis=0),
            engine.sum(ww * (xx * xx)[:, None], axis=0),
            engine.sum(ww, axis=0),
        )

    def cdf(self, x: float) -> float:
        """Cumulative distribution function ``P(X <= x)`` (exact)."""
        from scipy.stats import genpareto as _sp

        return float(_sp.cdf(x, self.shape, loc=self.loc, scale=self.scale))

    def quantile(self, q: float) -> float:
        """Inverse CDF ``F^{-1}(q)``."""
        from scipy.stats import genpareto as _sp

        return float(_sp.ppf(q, self.shape, loc=self.loc, scale=self.scale))

    def mean(self) -> float:
        """Mean loc + scale/(1-xi) for xi < 1, else inf."""
        xi = self.shape
        return float(self.loc + self.scale / (1.0 - xi)) if xi < 1.0 else float("inf")

    def variance(self) -> float:
        """Variance scale^2 / ((1-xi)^2 (1-2xi)) for xi < 1/2, else inf."""
        xi = self.shape
        if xi < 0.5:
            return float(self.scale * self.scale / ((1.0 - xi) ** 2 * (1.0 - 2.0 * xi)))
        return float("inf")

    def entropy(self) -> float:
        """Differential entropy log(scale) + xi + 1."""
        return float(self.log_scale + self.shape + 1.0)

    def sampler(self, seed: int | None = None) -> "GeneralizedParetoSampler":
        """Return a sampler for drawing observations from this distribution."""
        return GeneralizedParetoSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "GeneralizedParetoEstimator":
        """Return a method-of-moments estimator for ``scale`` and ``shape`` at the fixed threshold ``loc``."""
        return GeneralizedParetoEstimator(loc=self.loc, pseudo_count=pseudo_count, name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> "GeneralizedParetoDataEncoder":
        """Return the data encoder used by this distribution for vectorized methods."""
        return GeneralizedParetoDataEncoder()


class GeneralizedParetoSampler(DistributionSampler):
    """Draw iid GPD observations by inverse-CDF transform."""

    def __init__(self, dist: GeneralizedParetoDistribution, seed: int | None = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: int | None = None, *, batched: bool = True) -> float | np.ndarray:
        """Draw one sample or an array of iid samples by inverse CDF."""
        d = self.dist
        u = self.rng.uniform(size=size)  # uniform; 1-U is also uniform, so use U directly below
        if abs(d.shape) < _XI_TOL:
            y = -d.scale * np.log(u)
        else:
            y = (d.scale / d.shape) * (np.power(u, -d.shape) - 1.0)
        return d.loc + y


class GeneralizedParetoAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate weighted first and second moments for GPD estimation."""

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.sum = 0.0
        self.sum2 = 0.0
        self.count = 0.0
        self.name = name
        self.keys = keys

    def update(self, x: float, weight: float, estimate: GeneralizedParetoDistribution | None) -> None:
        """Accumulate weighted first and second moments for one observation."""
        self.sum += x * weight
        self.sum2 += x * x * weight
        self.count += weight

    def initialize(self, x: float, weight: float, rng: RandomState | None) -> None:
        """Initialize statistics from one observation."""
        self.update(x, weight, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: GeneralizedParetoDistribution | None) -> None:
        """Accumulate weighted first and second moments from encoded data."""
        xx = np.asarray(x, dtype=np.float64)
        self.sum += np.dot(xx, weights)
        self.sum2 += np.dot(xx * xx, weights)
        self.count += np.sum(weights, dtype=np.float64)

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize statistics from encoded observations."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[float, float, float]) -> "GeneralizedParetoAccumulator":
        """Merge another generalized-Pareto sufficient-statistic tuple."""
        self.sum += suff_stat[0]
        self.sum2 += suff_stat[1]
        self.count += suff_stat[2]
        return self

    def value(self) -> tuple[float, float, float]:
        """Return accumulated sum, second moment sum, and count."""
        return self.sum, self.sum2, self.count

    def from_value(self, x: tuple[float, float, float]) -> "GeneralizedParetoAccumulator":
        """Replace accumulator contents from a sufficient-statistic tuple."""
        self.sum, self.sum2, self.count = float(x[0]), float(x[1]), float(x[2])
        return self

    def acc_to_encoder(self) -> "GeneralizedParetoDataEncoder":
        """Return the encoder used by this accumulator."""
        return GeneralizedParetoDataEncoder()


class GeneralizedParetoAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for GeneralizedParetoAccumulator."""

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.name = name
        self.keys = keys

    def make(self) -> GeneralizedParetoAccumulator:
        """Create a fresh generalized-Pareto accumulator."""
        return GeneralizedParetoAccumulator(name=self.name, keys=self.keys)


class GeneralizedParetoEstimator(ParameterEstimator):
    """Method-of-moments estimator for GPD scale and shape at a fixed threshold ``loc``."""

    def __init__(
        self,
        loc: float = 0.0,
        pseudo_count: float | None = None,
        min_scale: float = 1.0e-12,
        xi_max: float = 0.5 - 1.0e-6,
        xi_min: float = -10.0,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        self.loc = float(loc)
        self.pseudo_count = pseudo_count
        self.min_scale = min_scale
        self.xi_max = xi_max  # method of moments needs a finite variance (xi < 1/2)
        self.xi_min = xi_min
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> GeneralizedParetoAccumulatorFactory:
        """Return an accumulator factory for generalized-Pareto moments."""
        return GeneralizedParetoAccumulatorFactory(name=self.name, keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat: tuple[float, float, float]) -> GeneralizedParetoDistribution:
        """Estimate scale and shape from exceedance moments at the fixed location."""
        sum_x, sum_x2, count = suff_stat
        if count <= 0.0:
            return GeneralizedParetoDistribution(1.0, 0.0, loc=self.loc, name=self.name, keys=self.keys)
        mean_x = sum_x / count
        var = sum_x2 / count - mean_x * mean_x
        m = mean_x - self.loc  # exceedance mean
        if m <= 0.0 or var <= 0.0:
            return GeneralizedParetoDistribution(
                max(m, self.min_scale), 0.0, loc=self.loc, name=self.name, keys=self.keys
            )
        xi = 0.5 * (1.0 - m * m / var)
        xi = min(max(xi, self.xi_min), self.xi_max)
        scale = max(m * (1.0 - xi), self.min_scale)
        return GeneralizedParetoDistribution(scale, xi, loc=self.loc, name=self.name, keys=self.keys)


class GeneralizedParetoDataEncoder(DataSequenceEncoder):
    """Encode GPD observations as a float array."""

    def __str__(self) -> str:
        return "GeneralizedParetoDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, GeneralizedParetoDataEncoder)

    def seq_encode(self, x: Sequence[float]) -> np.ndarray:
        """Encode observations as a floating-point array."""
        return np.asarray(x, dtype=np.float64)
