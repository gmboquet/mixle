"""Location-scale Student's t distributions over real values.

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
from mixle.utils.special import gammaln


class StudentTDistribution(SequenceEncodableProbabilityDistribution):
    """Student's t distribution with degrees of freedom df, location loc, and scale > 0."""

    @classmethod
    def compute_capabilities(cls):
        """Describe backend support for generated Student-t kernels."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch"), kernel_status="numba_adapter")

    @classmethod
    def compute_declaration(cls):
        """Return the structured compute declaration for Student-t distributions."""
        from mixle.stats.compute.declarations import DistributionDeclaration, ParameterSpec, StatisticSpec

        return DistributionDeclaration(
            name="student_t",
            distribution_type=cls,
            parameters=(
                ParameterSpec("df", constraint="positive"),
                ParameterSpec("loc"),
                ParameterSpec("scale", constraint="positive"),
            ),
            statistics=(StatisticSpec("sum"), StatisticSpec("sum2"), StatisticSpec("count")),
            support="real",
            legacy_sufficient_statistics=cls.backend_legacy_sufficient_statistics,
        )

    @staticmethod
    def backend_legacy_sufficient_statistics(x: Any, params: dict[str, Any], engine: Any) -> tuple[Any, ...]:
        """Return per-row Student-t sufficient statistics in accumulator order."""
        xx = engine.asarray(x)
        return xx, xx * xx, xx * 0.0 + engine.asarray(1.0)

    def __init__(
        self, df: float, loc: float = 0.0, scale: float = 1.0, name: str | None = None, keys: str | None = None
    ) -> None:
        if df <= 0.0 or scale <= 0.0 or not np.isfinite(df) or not np.isfinite(scale):
            raise ValueError("StudentTDistribution requires df > 0 and scale > 0.")
        self.df = float(df)
        self.loc = float(loc)
        self.scale = float(scale)
        self.log_const = float(
            gammaln((self.df + 1.0) / 2.0)
            - gammaln(self.df / 2.0)
            - 0.5 * math.log(self.df * math.pi)
            - math.log(self.scale)
        )
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        return "StudentTDistribution(%s, loc=%s, scale=%s, name=%s, keys=%s)" % (
            repr(self.df),
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
        return self.log_const - 0.5 * (self.df + 1.0) * math.log1p((z * z) / self.df)

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Return vectorized log-density values for sequence-encoded observations."""
        z = (x - self.loc) / self.scale
        return self.log_const - 0.5 * (self.df + 1.0) * np.log1p((z * z) / self.df)

    @staticmethod
    def backend_log_density_from_params(x: Any, df: Any, loc: Any, scale: Any, engine: Any) -> Any:
        """Engine-neutral Student-t log-density from explicit parameters."""
        z = (x - loc) / scale
        half = engine.asarray(0.5)
        one = engine.asarray(1.0)
        log_const = (
            engine.gammaln((df + one) * half)
            - engine.gammaln(df * half)
            - half * engine.log(df * engine.asarray(math.pi))
            - engine.log(scale)
        )
        return log_const - half * (df + one) * engine.log(one + (z * z) / df)

    def backend_seq_log_density(self, x: Any, engine: Any) -> Any:
        """Engine-neutral vectorized log-density for encoded data."""
        return self.backend_log_density_from_params(
            engine.asarray(x), engine.asarray(self.df), engine.asarray(self.loc), engine.asarray(self.scale), engine
        )

    @classmethod
    def backend_stacked_params(cls, dists: Sequence["StudentTDistribution"], engine: Any) -> dict[str, Any]:
        """Return stacked Student-t parameters for a homogeneous mixture kernel."""
        return {
            "df": engine.asarray([d.df for d in dists]),
            "loc": engine.asarray([d.loc for d in dists]),
            "scale": engine.asarray([d.scale for d in dists]),
        }

    @classmethod
    def backend_stacked_log_density(cls, x: Any, params: dict[str, Any], engine: Any) -> Any:
        """Return an ``(n, k)`` matrix of Student-t log densities."""
        xx = engine.asarray(x)
        return cls.backend_log_density_from_params(
            xx[:, None], params["df"][None, :], params["loc"][None, :], params["scale"][None, :], engine
        )

    def cdf(self, x: float) -> float:
        """Cumulative distribution function ``P(X <= x)`` (exact). The continuous 'index of' a value."""
        from scipy.stats import t as _sp

        return float(_sp.cdf(x, self.df, loc=self.loc, scale=self.scale))

    def quantile(self, q: float) -> float:
        """Inverse CDF ``F^{-1}(q)``: the value at cumulative-probability index ``q`` (continuous unranking)."""
        from scipy.stats import t as _sp

        return float(_sp.ppf(q, self.df, loc=self.loc, scale=self.scale))

    def mean(self) -> float:
        """Mean (loc) for df > 1, else inf (undefined)."""
        return float(self.loc) if self.df > 1.0 else float("inf")

    def variance(self) -> float:
        """Variance scale^2 * df/(df-2) for df > 2, else inf."""
        return float(self.scale * self.scale * self.df / (self.df - 2.0)) if self.df > 2.0 else float("inf")

    def entropy(self) -> float:
        """Differential entropy log(scale) + (df+1)/2 [psi((df+1)/2) - psi(df/2)] + log(sqrt(df) B(df/2, 1/2))."""
        from scipy.special import digamma

        nu = self.df
        return float(
            math.log(self.scale)
            + 0.5 * (nu + 1.0) * (digamma(0.5 * (nu + 1.0)) - digamma(0.5 * nu))
            + 0.5 * math.log(nu)
            + gammaln(0.5 * nu)
            + gammaln(0.5)
            - gammaln(0.5 * (nu + 1.0))
        )

    def sampler(self, seed: int | None = None) -> "StudentTSampler":
        """Return a sampler for drawing observations from this distribution."""
        return StudentTSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "StudentTEstimator":
        """Return an estimator for fitting this distribution from data."""
        if pseudo_count is None:
            return StudentTEstimator(df=self.df, name=self.name, keys=self.keys)
        return StudentTEstimator(
            df=self.df, pseudo_count=pseudo_count, suff_stat=(self.loc, self.scale), name=self.name, keys=self.keys
        )

    def dist_to_encoder(self) -> "StudentTDataEncoder":
        """Return the data encoder used by this distribution for vectorized methods."""
        return StudentTDataEncoder()


class StudentTSampler(DistributionSampler):
    """Draw iid Student's t observations."""

    def __init__(self, dist: StudentTDistribution, seed: int | None = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: int | None = None) -> float | np.ndarray:
        """Draw one sample or an array of iid samples."""
        return self.rng.standard_t(self.dist.df, size=size) * self.dist.scale + self.dist.loc


class StudentTAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate weighted first and second moments for fixed-df t estimation."""

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.sum = 0.0
        self.sum2 = 0.0
        self.count = 0.0
        self.name = name
        self.keys = keys

    def update(self, x: float, weight: float, estimate: StudentTDistribution | None) -> None:
        """Accumulate weighted first and second moments for one observation."""
        self.sum += x * weight
        self.sum2 += x * x * weight
        self.count += weight

    def initialize(self, x: float, weight: float, rng: RandomState | None) -> None:
        """Initialize statistics from one observation."""
        self.update(x, weight, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: StudentTDistribution | None) -> None:
        """Accumulate weighted first and second moments from encoded data."""
        self.sum += np.dot(x, weights)
        self.sum2 += np.dot(x * x, weights)
        self.count += np.sum(weights, dtype=np.float64)

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize statistics from encoded observations."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[float, float, float]) -> "StudentTAccumulator":
        """Merge another Student-t sufficient-statistic tuple."""
        self.sum += suff_stat[0]
        self.sum2 += suff_stat[1]
        self.count += suff_stat[2]
        return self

    def value(self) -> tuple[float, float, float]:
        """Return accumulated sum, second moment sum, and count."""
        return self.sum, self.sum2, self.count

    def from_value(self, x: tuple[float, float, float]) -> "StudentTAccumulator":
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

    def acc_to_encoder(self) -> "StudentTDataEncoder":
        """Return the encoder used by this accumulator."""
        return StudentTDataEncoder()


class StudentTAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for StudentTAccumulator."""

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.name = name
        self.keys = keys

    def make(self) -> StudentTAccumulator:
        """Create a fresh Student-t accumulator."""
        return StudentTAccumulator(name=self.name, keys=self.keys)


class StudentTEstimator(ParameterEstimator):
    """Moment-style fixed-df estimator for Student's t location and scale.

    The exact MLE has no simple closed-form update. This estimator keeps df fixed
    and uses weighted moments, while generic gradient optimizers such as
    ``mixle.inference.gradient_fit.fit_mle`` / ``fit_map`` can fit all three
    parameters through distribution-owned backend math.
    """

    def __init__(
        self,
        df: float = 5.0,
        pseudo_count: float | None = None,
        suff_stat: tuple[float, float] | None = None,
        min_scale: float = 1.0e-8,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        if df <= 0.0 or not np.isfinite(df):
            raise ValueError("StudentTEstimator requires df > 0.")
        self.df = float(df)
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.min_scale = min_scale
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> StudentTAccumulatorFactory:
        """Return an accumulator factory for fixed-df Student-t moments."""
        return StudentTAccumulatorFactory(name=self.name, keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat: tuple[float, float, float]) -> StudentTDistribution:
        """Estimate location and scale while keeping degrees of freedom fixed."""
        sum_x, sum_x2, count = suff_stat
        if self.pseudo_count is not None and self.suff_stat is not None:
            loc0, scale0 = self.suff_stat
            var0 = scale0 * scale0 * self.df / (self.df - 2.0) if self.df > 2.0 else scale0 * scale0
            sum_x += self.pseudo_count * loc0
            sum_x2 += self.pseudo_count * (var0 + loc0 * loc0)
            count += self.pseudo_count

        if count <= 0.0:
            return StudentTDistribution(self.df, name=self.name, keys=self.keys)

        loc = sum_x / count
        var = max(sum_x2 / count - loc * loc, self.min_scale * self.min_scale)
        scale2 = var * (self.df - 2.0) / self.df if self.df > 2.0 else var
        scale = math.sqrt(max(scale2, self.min_scale * self.min_scale))
        return StudentTDistribution(self.df, loc=loc, scale=scale, name=self.name, keys=self.keys)


class StudentTDataEncoder(DataSequenceEncoder):
    """Encode Student's t observations as a float array."""

    def __str__(self) -> str:
        return "StudentTDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, StudentTDataEncoder)

    def seq_encode(self, x: Sequence[float]) -> np.ndarray:
        """Encode observations as a floating-point array."""
        rv = np.asarray(x, dtype=np.float64)
        if rv.size and np.any(np.isnan(rv)):
            raise ValueError("StudentTDistribution requires finite or infinite real-valued observations.")
        return rv
