"""Generalized Extreme Value distribution (GEV): the limit law of block maxima.

By the Fisher-Tippett-Gnedenko theorem the normalized maximum of a large block of iid observations
converges to a GEV, the standard model for extremes (flood levels, wind speeds, record losses). With
location ``mu``, scale ``sigma > 0`` and shape ``xi`` (the EVT sign convention; ``scipy``'s
``genextreme`` uses ``c = -xi``), for ``z = (x - mu)/sigma``:

    log f = -log sigma - (1/xi + 1) log s - s ** (-1/xi),   s = 1 + xi z > 0   (xi != 0),
    log f = -log sigma - z - exp(-z)                                            (xi == 0, Gumbel).

``xi > 0`` is the heavy-tailed Frechet type (support ``x >= mu - sigma/xi``), ``xi = 0`` the Gumbel
type (all reals), ``xi < 0`` the bounded Weibull type (``x <= mu - sigma/xi``). All three parameters
are fit by method of moments: the shape is solved from the (monotone) skewness-vs-``xi`` relation,
then scale from the variance and location from the mean.


Reference: Coles, *An Introduction to Statistical Modeling of Extreme Values* (Springer, 2001).
"""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import RandomState
from scipy.special import gamma as _gamma

from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)

_XI_TOL = 1.0e-8  # |xi| below this is treated as the Gumbel limit
_EULER = 0.5772156649015329
_GUMBEL_SKEW = 12.0 * math.sqrt(6.0) * 1.2020569031595943 / math.pi**3  # 12 sqrt(6) zeta(3) / pi^3


def _gev_skewness(xi: float) -> float:
    """Theoretical skewness of a GEV with shape ``xi`` (monotone increasing; defined for ``xi < 1/3``)."""
    if abs(xi) < _XI_TOL:
        return _GUMBEL_SKEW
    g1, g2, g3 = _gamma(1.0 - xi), _gamma(1.0 - 2.0 * xi), _gamma(1.0 - 3.0 * xi)
    return float(np.sign(xi) * (g3 - 3.0 * g1 * g2 + 2.0 * g1**3) / (g2 - g1 * g1) ** 1.5)


def _xi_from_skewness(skew: float, xi_min: float, xi_max: float) -> float:
    """Invert the monotone skewness-vs-``xi`` relation by bisection."""
    if skew <= _gev_skewness(xi_min):
        return xi_min
    if skew >= _gev_skewness(xi_max):
        return xi_max
    lo, hi = xi_min, xi_max
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if _gev_skewness(mid) < skew:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


class GeneralizedExtremeValueDistribution(SequenceEncodableProbabilityDistribution):
    """Generalized Extreme Value distribution with location ``loc``, scale ``> 0`` and shape ``xi``."""

    def __init__(
        self, loc: float, scale: float, shape: float, name: str | None = None, keys: str | None = None
    ) -> None:
        if scale <= 0.0 or not np.isfinite(scale) or not np.isfinite(loc) or not np.isfinite(shape):
            raise ValueError("GeneralizedExtremeValueDistribution requires finite parameters and scale > 0.")
        self.loc = float(loc)  # mu
        self.scale = float(scale)  # sigma
        self.shape = float(shape)  # xi
        self.log_scale = math.log(self.scale)
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        return "GeneralizedExtremeValueDistribution(%s, %s, %s, name=%s, keys=%s)" % (
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
        """Return the log-density at a single observation (``-inf`` outside the support)."""
        z = (x - self.loc) / self.scale
        if abs(self.shape) < _XI_TOL:
            return -self.log_scale - z - math.exp(-z)
        s = 1.0 + self.shape * z
        if s <= 0.0:
            return -np.inf
        return -self.log_scale - (1.0 / self.shape + 1.0) * math.log(s) - s ** (-1.0 / self.shape)

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Return vectorized log-density values for sequence-encoded observations."""
        z = (np.asarray(x, dtype=np.float64) - self.loc) / self.scale
        if abs(self.shape) < _XI_TOL:
            return -self.log_scale - z - np.exp(-z)
        s = 1.0 + self.shape * z
        with np.errstate(divide="ignore", invalid="ignore"):
            rv = -self.log_scale - (1.0 / self.shape + 1.0) * np.log(s) - np.power(s, -1.0 / self.shape)
        return np.where(s <= 0.0, -np.inf, rv)

    # --- compute-engine backend (numpy + torch/GPU): scoring + sufficient statistics in engine ops ---
    @classmethod
    def compute_capabilities(cls):
        """Describe backend support for generated GEV kernels."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch"), kernel_status="generic")

    @classmethod
    def compute_declaration(cls):
        """Return the structured compute declaration for GEV distributions."""
        from mixle.stats.compute.declarations import DistributionDeclaration, ParameterSpec, StatisticSpec

        return DistributionDeclaration(
            name="generalized_extreme_value",
            distribution_type=cls,
            parameters=(ParameterSpec("loc"), ParameterSpec("scale", constraint="positive"), ParameterSpec("shape")),
            statistics=(StatisticSpec("sum"), StatisticSpec("sum2"), StatisticSpec("sum3"), StatisticSpec("count")),
            support="real",
            legacy_sufficient_statistics=cls.backend_legacy_sufficient_statistics,
        )

    @staticmethod
    def backend_legacy_sufficient_statistics(x: Any, params: dict[str, Any], engine: Any) -> tuple[Any, ...]:
        """Per-row GEV moment sums in accumulator order ``(sum, sum2, sum3, count)``."""
        xx = engine.asarray(x)
        x2 = xx * xx
        return xx, x2, x2 * xx, xx * 0.0 + engine.asarray(1.0)

    @staticmethod
    def backend_log_density_from_params(x: Any, loc: Any, scale: Any, shape: Any, engine: Any) -> Any:
        """Engine-neutral GEV log-density; the ``|xi| < tol`` Gumbel limit is selected per element.

        ``s^{-1/xi}`` is computed as ``exp(-log(s)/xi)`` so the whole expression stays on engine ops."""
        z = (x - loc) / scale
        neg_inf = engine.asarray(float("-inf"))
        is_limit = engine.abs(shape) < _XI_TOL
        xi_safe = engine.where(is_limit, engine.asarray(1.0), shape)
        s = 1.0 + xi_safe * z
        s_pos = engine.where(s > 0.0, s, engine.asarray(1.0))  # keep log/exp NaN-free off-support
        log_s = engine.log(s_pos)
        general = -engine.log(scale) - (1.0 / xi_safe + 1.0) * log_s - engine.exp(-log_s / xi_safe)
        general = engine.where(s > 0.0, general, neg_inf)
        limit = -engine.log(scale) - z - engine.exp(-z)
        return engine.where(is_limit, limit, general)

    def backend_seq_log_density(self, x: Any, engine: Any) -> Any:
        """Engine-neutral vectorized log-density for encoded data."""
        return self.backend_log_density_from_params(
            engine.asarray(x),
            engine.asarray(self.loc),
            engine.asarray(self.scale),
            engine.asarray(self.shape),
            engine,
        )

    @classmethod
    def backend_stacked_params(
        cls, dists: Sequence["GeneralizedExtremeValueDistribution"], engine: Any
    ) -> dict[str, Any]:
        """Stacked GEV parameters for a homogeneous mixture kernel."""
        return {
            "loc": engine.asarray([d.loc for d in dists]),
            "scale": engine.asarray([d.scale for d in dists]),
            "shape": engine.asarray([d.shape for d in dists]),
        }

    @classmethod
    def backend_stacked_log_density(cls, x: np.ndarray, params: dict[str, Any], engine: Any) -> Any:
        """Return an ``(n, k)`` matrix of GEV log densities."""
        xx = engine.asarray(x)
        return cls.backend_log_density_from_params(
            xx[:, None], params["loc"][None, :], params["scale"][None, :], params["shape"][None, :], engine
        )

    @classmethod
    def backend_stacked_sufficient_statistics(
        cls, x: np.ndarray, weights: Any, params: dict[str, Any], engine: Any
    ) -> tuple[Any, Any, Any, Any]:
        """Stacked GEV moment sums ``(sum, sum2, sum3, count)`` using engine-resident arrays."""
        xx = engine.asarray(x)
        ww = engine.asarray(weights)
        x2 = xx * xx
        return (
            engine.sum(ww * xx[:, None], axis=0),
            engine.sum(ww * x2[:, None], axis=0),
            engine.sum(ww * (x2 * xx)[:, None], axis=0),
            engine.sum(ww, axis=0),
        )

    def cdf(self, x: float) -> float:
        """Cumulative distribution function ``P(X <= x)`` (exact)."""
        from scipy.stats import genextreme as _sp

        return float(_sp.cdf(x, -self.shape, loc=self.loc, scale=self.scale))

    def quantile(self, q: float) -> float:
        """Inverse CDF ``F^{-1}(q)``."""
        from scipy.stats import genextreme as _sp

        return float(_sp.ppf(q, -self.shape, loc=self.loc, scale=self.scale))

    def mean(self) -> float:
        """Mean: loc + scale*(Gamma(1-xi)-1)/xi (loc+scale*euler_gamma at xi=0); inf for xi>=1."""
        from scipy.special import gamma as _gamma

        xi = self.shape
        if abs(xi) < 1.0e-12:
            return float(self.loc + self.scale * np.euler_gamma)
        if xi < 1.0:
            return float(self.loc + self.scale * (_gamma(1.0 - xi) - 1.0) / xi)
        return float("inf")

    def variance(self) -> float:
        """Variance: scale^2 (Gamma(1-2xi)-Gamma(1-xi)^2)/xi^2 (scale^2 pi^2/6 at xi=0); inf for xi>=1/2."""
        import math

        from scipy.special import gamma as _gamma

        xi = self.shape
        if abs(xi) < 1.0e-12:
            return float(self.scale * self.scale * math.pi * math.pi / 6.0)
        if xi < 0.5:
            g1 = _gamma(1.0 - xi)
            g2 = _gamma(1.0 - 2.0 * xi)
            return float(self.scale * self.scale * (g2 - g1 * g1) / (xi * xi))
        return float("inf")

    def entropy(self) -> float:
        """Differential entropy log(scale) + euler_gamma * xi + euler_gamma + 1."""
        return float(self.log_scale + np.euler_gamma * self.shape + np.euler_gamma + 1.0)

    def sampler(self, seed: int | None = None) -> "GeneralizedExtremeValueSampler":
        """Return a sampler for drawing observations from this distribution."""
        return GeneralizedExtremeValueSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "GeneralizedExtremeValueEstimator":
        """Return a method-of-moments estimator for ``loc``, ``scale`` and ``shape``."""
        return GeneralizedExtremeValueEstimator(pseudo_count=pseudo_count, name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> "GeneralizedExtremeValueDataEncoder":
        """Return the data encoder used by this distribution for vectorized methods."""
        return GeneralizedExtremeValueDataEncoder()


class GeneralizedExtremeValueSampler(DistributionSampler):
    """Draw iid GEV observations by inverse-CDF transform."""

    def __init__(self, dist: GeneralizedExtremeValueDistribution, seed: int | None = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: int | None = None, *, batched: bool = True) -> float | np.ndarray:
        """Draw one sample or an array of iid samples by inverse CDF."""
        d = self.dist
        e = -np.log(self.rng.uniform(size=size))  # -log U ~ Exp(1) = the standard Gumbel core
        if abs(d.shape) < _XI_TOL:
            z = -np.log(e)
        else:
            z = (np.power(e, -d.shape) - 1.0) / d.shape
        return d.loc + d.scale * z


class GeneralizedExtremeValueAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate weighted first three moments for GEV estimation."""

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.sum = 0.0
        self.sum2 = 0.0
        self.sum3 = 0.0
        self.count = 0.0
        self.name = name
        self.keys = keys

    def update(self, x: float, weight: float, estimate: GeneralizedExtremeValueDistribution | None) -> None:
        """Accumulate weighted first three raw moments for one observation."""
        self.sum += x * weight
        self.sum2 += x * x * weight
        self.sum3 += x * x * x * weight
        self.count += weight

    def initialize(self, x: float, weight: float, rng: RandomState | None) -> None:
        """Initialize statistics from one observation."""
        self.update(x, weight, None)

    def seq_update(
        self, x: np.ndarray, weights: np.ndarray, estimate: GeneralizedExtremeValueDistribution | None
    ) -> None:
        """Accumulate weighted first three raw moments from encoded data."""
        xx = np.asarray(x, dtype=np.float64)
        self.sum += np.dot(xx, weights)
        self.sum2 += np.dot(xx * xx, weights)
        self.sum3 += np.dot(xx * xx * xx, weights)
        self.count += np.sum(weights, dtype=np.float64)

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize statistics from encoded observations."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[float, float, float, float]) -> "GeneralizedExtremeValueAccumulator":
        """Merge another GEV sufficient-statistic tuple."""
        self.sum += suff_stat[0]
        self.sum2 += suff_stat[1]
        self.sum3 += suff_stat[2]
        self.count += suff_stat[3]
        return self

    def value(self) -> tuple[float, float, float, float]:
        """Return raw moment sums and observation count."""
        return self.sum, self.sum2, self.sum3, self.count

    def from_value(self, x: tuple[float, float, float, float]) -> "GeneralizedExtremeValueAccumulator":
        """Replace accumulator contents from a sufficient-statistic tuple."""
        self.sum, self.sum2, self.sum3, self.count = float(x[0]), float(x[1]), float(x[2]), float(x[3])
        return self

    def acc_to_encoder(self) -> "GeneralizedExtremeValueDataEncoder":
        """Return the encoder used by this accumulator."""
        return GeneralizedExtremeValueDataEncoder()


class GeneralizedExtremeValueAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for GeneralizedExtremeValueAccumulator."""

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.name = name
        self.keys = keys

    def make(self) -> GeneralizedExtremeValueAccumulator:
        """Create a fresh GEV accumulator."""
        return GeneralizedExtremeValueAccumulator(name=self.name, keys=self.keys)


class GeneralizedExtremeValueEstimator(ParameterEstimator):
    """Method-of-moments estimator for GEV location, scale and shape."""

    def __init__(
        self,
        pseudo_count: float | None = None,
        min_scale: float = 1.0e-12,
        xi_max: float = 1.0 / 3.0 - 1.0e-4,  # third moment finite only for xi < 1/3
        xi_min: float = -1.0,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        self.pseudo_count = pseudo_count
        self.min_scale = min_scale
        self.xi_max = xi_max
        self.xi_min = xi_min
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> GeneralizedExtremeValueAccumulatorFactory:
        """Return an accumulator factory for GEV raw-moment statistics."""
        return GeneralizedExtremeValueAccumulatorFactory(name=self.name, keys=self.keys)

    def estimate(
        self, nobs: float | None, suff_stat: tuple[float, float, float, float]
    ) -> GeneralizedExtremeValueDistribution:
        """Estimate location, scale, and shape from weighted moments."""
        sum_x, sum_x2, sum_x3, count = suff_stat
        if count <= 0.0:
            return GeneralizedExtremeValueDistribution(0.0, 1.0, 0.0, name=self.name, keys=self.keys)
        mean = sum_x / count
        var = sum_x2 / count - mean * mean
        if var <= 0.0:
            return GeneralizedExtremeValueDistribution(mean, self.min_scale, 0.0, name=self.name, keys=self.keys)
        m3 = sum_x3 / count - 3.0 * mean * (sum_x2 / count) + 2.0 * mean**3  # central third moment
        skew = m3 / var**1.5
        xi = _xi_from_skewness(skew, self.xi_min, self.xi_max)
        if abs(xi) < _XI_TOL:  # Gumbel limit
            scale = math.sqrt(6.0 * var) / math.pi
            loc = mean - scale * _EULER
        else:
            g1, g2 = _gamma(1.0 - xi), _gamma(1.0 - 2.0 * xi)
            scale = math.sqrt(var) * abs(xi) / math.sqrt(g2 - g1 * g1)
            loc = mean - scale * (g1 - 1.0) / xi
        return GeneralizedExtremeValueDistribution(loc, max(scale, self.min_scale), xi, name=self.name, keys=self.keys)


class GeneralizedExtremeValueDataEncoder(DataSequenceEncoder):
    """Encode GEV observations as a float array."""

    def __str__(self) -> str:
        return "GeneralizedExtremeValueDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, GeneralizedExtremeValueDataEncoder)

    def seq_encode(self, x: Sequence[float]) -> np.ndarray:
        """Encode observations as a floating-point array."""
        return np.asarray(x, dtype=np.float64)
