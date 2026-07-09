"""Inverse Gaussian (Wald) distributions over positive real values.

Observations are floats ``x > 0``. An inverse Gaussian distribution with mean ``mu > 0`` and shape
``lam > 0`` has log-density

        log(f(x; mu, lam)) = 0.5 * (log(lam) - log(2*pi) - 3*log(x)) - lam*(x - mu)**2 / (2*mu**2*x),

    for x > 0.0, else -np.inf.

The inverse Gaussian is a two-parameter exponential family with sufficient statistics (x, 1/x):

    log(f) = base(x) + eta1*x + eta2*(1/x) - A(mu, lam),
        base(x)        = -0.5*log(2*pi) - 1.5*log(x),
        eta1           = -lam / (2*mu**2),
        eta2           = -lam / 2,
        A(mu, lam)     = -0.5*log(lam) - lam/mu.

Declaring those pieces gives the family generated NumPy/Torch/Numba scoring through the shared
exponential-family compute path, exactly as for the Gamma family.


Reference: Chhikara & Folks, *The Inverse Gaussian Distribution* (Dekker, 1989).
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

_MIN_IG_PARAM = 1.0e-12
_MAX_IG_PARAM = 1.0e12
_HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)


class InverseGaussianDistribution(SequenceEncodableProbabilityDistribution):
    """Inverse Gaussian (Wald) distribution with mean mu > 0 and shape lam > 0 on x > 0."""

    @classmethod
    def compute_capabilities(cls):
        """Describe backend support for generated inverse-Gaussian kernels."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch"), kernel_status="numba_adapter")

    @classmethod
    def compute_declaration(cls):
        """Return the structured compute declaration for inverse Gaussian distributions."""
        from mixle.stats.compute.declarations import (
            DistributionDeclaration,
            ExponentialFamilySpec,
            ParameterSpec,
            StatisticSpec,
        )

        return DistributionDeclaration(
            name="inverse_gaussian",
            distribution_type=cls,
            parameters=(
                ParameterSpec("mu", constraint="positive"),
                ParameterSpec("lam", constraint="positive"),
            ),
            statistics=(StatisticSpec("count"), StatisticSpec("sum"), StatisticSpec("sum_inv")),
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
        """Return the (x, 1/x) sufficient statistics for generated scoring."""
        vals, inv_vals, _ = x
        return engine.asarray(vals), engine.asarray(inv_vals)

    @staticmethod
    def exp_family_natural_parameters(params: dict[str, Any], engine: Any) -> tuple[Any, ...]:
        """Return the (-lam/(2*mu^2), -lam/2) natural parameters for generated scoring."""
        mu = params["mu"]
        lam = params["lam"]
        return -lam / (engine.asarray(2.0) * mu * mu), -lam / engine.asarray(2.0)

    @staticmethod
    def exp_family_log_partition(params: dict[str, Any], engine: Any) -> Any:
        """Return the inverse Gaussian log partition -0.5*log(lam) - lam/mu."""
        mu = params["mu"]
        lam = params["lam"]
        return -engine.asarray(0.5) * engine.log(lam) - lam / mu

    @staticmethod
    def exp_family_base_measure(x: tuple[Any, Any, Any], engine: Any) -> Any:
        """Return the support base measure -0.5*log(2*pi) - 1.5*log(x) (or -inf off support)."""
        vals = engine.asarray(x[0])
        log_vals = engine.asarray(x[2])
        base = engine.asarray(-_HALF_LOG_2PI) - engine.asarray(1.5) * log_vals
        return engine.where(vals > 0.0, base, engine.asarray(-np.inf))

    @staticmethod
    def exp_family_legacy_sufficient_statistics(
        x: tuple[Any, Any, Any], params: dict[str, Any], engine: Any
    ) -> tuple[Any, ...]:
        """Return per-row (count, x, 1/x) sufficient statistics in accumulator order."""
        vals = engine.asarray(x[0])
        inv_vals = engine.asarray(x[1])
        return vals * 0.0 + engine.asarray(1.0), vals, inv_vals

    def __init__(self, mu: float, lam: float, name: str | None = None, keys: str | None = None) -> None:
        """InverseGaussianDistribution for mean mu and shape lam.

        Args:
            mu (float): Positive real-valued mean.
            lam (float): Positive real-valued shape parameter.
            name (Optional[str]): Assign a name to InverseGaussianDistribution instance.
            keys (Optional[str]): Assign keys for merging sufficient statistics.

        Attributes:
            mu (float): Positive real-valued mean.
            lam (float): Positive real-valued shape parameter.
            log_lam (float): Cached log(lam).
            name (Optional[str]): Optional distribution name.
            keys (Optional[str]): Key for merging sufficient statistics.

        """
        if mu <= 0.0 or not np.isfinite(mu):
            raise ValueError("InverseGaussianDistribution requires finite mu > 0.")
        if lam <= 0.0 or not np.isfinite(lam):
            raise ValueError("InverseGaussianDistribution requires finite lam > 0.")
        self.mu = float(mu)
        self.lam = float(lam)
        self.log_lam = math.log(self.lam)
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        """Return a constructor-style representation of the inverse Gaussian distribution."""
        return "InverseGaussianDistribution(%s, %s, name=%s, keys=%s)" % (
            repr(self.mu),
            repr(self.lam),
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
        if not np.isfinite(xx) or xx <= 0.0:
            return -np.inf
        z = xx - self.mu
        return 0.5 * (self.log_lam - math.log(2.0 * math.pi) - 3.0 * math.log(xx)) - (
            self.lam * z * z / (2.0 * self.mu * self.mu * xx)
        )

    def seq_log_density(self, x: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
        """Return vectorized log-density values for sequence-encoded observations.

        Args:
            x (Tuple[ndarray, ndarray, ndarray]): Tuple of observations, reciprocals, and log values
                produced by the InverseGaussianDataEncoder.

        Returns:
            Numpy array of log-density values, with -inf entries off the positive support.

        """
        vals, inv_vals, log_vals = x
        rv = (
            0.5 * (self.log_lam - math.log(2.0 * math.pi))
            - 1.5 * log_vals
            - self.lam * (vals - 2.0 * self.mu + self.mu * self.mu * inv_vals) / (2.0 * self.mu * self.mu)
        )
        return np.where(np.isfinite(vals) & (vals > 0.0), rv, -np.inf)

    @staticmethod
    def backend_log_density_from_params(vals: Any, inv_vals: Any, log_vals: Any, mu: Any, lam: Any, engine: Any) -> Any:
        """Engine-neutral inverse Gaussian log-density from explicit parameters."""
        two = engine.asarray(2.0)
        rv = (
            engine.asarray(0.5) * (engine.log(lam) - engine.asarray(math.log(2.0 * math.pi)))
            - engine.asarray(1.5) * log_vals
            - lam * (vals - two * mu + mu * mu * inv_vals) / (two * mu * mu)
        )
        return engine.where(vals > 0.0, rv, engine.asarray(-np.inf))

    def backend_seq_log_density(self, x: tuple[Any, Any, Any], engine: Any) -> Any:
        """Engine-neutral vectorized log-density for encoded data."""
        vals = engine.asarray(x[0])
        inv_vals = engine.asarray(x[1])
        log_vals = engine.asarray(x[2])
        return self.backend_log_density_from_params(
            vals, inv_vals, log_vals, engine.asarray(self.mu), engine.asarray(self.lam), engine
        )

    @classmethod
    def backend_stacked_params(cls, dists: Sequence["InverseGaussianDistribution"], engine: Any) -> dict[str, Any]:
        """Return stacked parameters for a homogeneous mixture kernel."""
        return {
            "mu": engine.asarray([d.mu for d in dists]),
            "lam": engine.asarray([d.lam for d in dists]),
        }

    @classmethod
    def backend_stacked_log_density(cls, x: tuple[Any, Any, Any], params: dict[str, Any], engine: Any) -> Any:
        """Return an ``(n, k)`` matrix of inverse Gaussian log densities."""
        vals = engine.asarray(x[0])
        inv_vals = engine.asarray(x[1])
        log_vals = engine.asarray(x[2])
        return cls.backend_log_density_from_params(
            vals[:, None],
            inv_vals[:, None],
            log_vals[:, None],
            params["mu"][None, :],
            params["lam"][None, :],
            engine,
        )

    @classmethod
    def backend_stacked_sufficient_statistics(
        cls, x: tuple[Any, Any, Any], weights: Any, params: dict[str, Any], engine: Any
    ) -> tuple[Any, Any, Any]:
        """Return stacked sufficient statistics using engine-resident arrays."""
        vals = engine.asarray(x[0])
        inv_vals = engine.asarray(x[1])
        ww = engine.asarray(weights)
        return (
            engine.sum(ww, axis=0),
            engine.sum(ww * vals[:, None], axis=0),
            engine.sum(ww * inv_vals[:, None], axis=0),
        )

    def cdf(self, x: float) -> float:
        """Cumulative distribution function P(X <= x) (Wald, via scipy invgauss)."""
        from scipy.stats import invgauss

        return float(invgauss.cdf(float(x), mu=self.mu / self.lam, scale=self.lam))

    def quantile(self, q: float) -> float:
        """Inverse CDF F^{-1}(q)."""
        from scipy.stats import invgauss

        return float(invgauss.ppf(float(q), mu=self.mu / self.lam, scale=self.lam))

    def sampler(self, seed: int | None = None) -> "InverseGaussianSampler":
        """Return a sampler for drawing observations from this distribution."""
        return InverseGaussianSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "InverseGaussianEstimator":
        """Return an estimator for fitting this distribution from data.

        Args:
            pseudo_count (Optional[float]): Re-weight the sufficient statistics of this instance toward
                its own moments when not None (a simple ridge toward the current parameters).

        Returns:
            InverseGaussianEstimator object.

        """
        if pseudo_count is None:
            return InverseGaussianEstimator(name=self.name, keys=self.keys)
        # E[x] = mu, E[1/x] = 1/mu + 1/lam for the inverse Gaussian.
        suff_stat = (self.mu, 1.0 / self.mu + 1.0 / self.lam)
        return InverseGaussianEstimator(pseudo_count=pseudo_count, suff_stat=suff_stat, name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> "InverseGaussianDataEncoder":
        """Return the data encoder used by this distribution for vectorized methods."""
        return InverseGaussianDataEncoder()


class InverseGaussianSampler(DistributionSampler):
    """Draw iid inverse Gaussian (Wald) observations."""

    def __init__(self, dist: InverseGaussianDistribution, seed: int | None = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist
        self.seed = seed

    def sample(self, size: int | None = None) -> float | np.ndarray:
        """Draw ``size`` iid observations (a float when ``size`` is None)."""
        return self.rng.wald(self.dist.mu, self.dist.lam, size=size)


class InverseGaussianAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate weighted count, sum, and sum of reciprocals for inverse Gaussian estimation."""

    def __init__(self, keys: str | None = None) -> None:
        self.count = 0.0
        self.sum = 0.0
        self.sum_inv = 0.0
        self.keys = keys

    def update(self, x: float, weight: float, estimate: InverseGaussianDistribution | None) -> None:
        """Accumulate count, sum, and reciprocal sum for one positive observation."""
        if x <= 0.0 or not np.isfinite(x):
            raise ValueError("InverseGaussianDistribution has support x > 0.")
        self.count += weight
        self.sum += x * weight
        self.sum_inv += weight / x

    def initialize(self, x: float, weight: float, rng: RandomState | None) -> None:
        """Initialize statistics from one observation."""
        self.update(x, weight, None)

    def seq_update(
        self,
        x: tuple[np.ndarray, np.ndarray, np.ndarray],
        weights: np.ndarray,
        estimate: InverseGaussianDistribution | None,
    ) -> None:
        """Accumulate transformed sufficient statistics from encoded data."""
        vals, inv_vals, _ = x
        self.sum += np.dot(vals, weights)
        self.sum_inv += np.dot(inv_vals, weights)
        self.count += np.sum(weights, dtype=np.float64)

    def seq_initialize(
        self, x: tuple[np.ndarray, np.ndarray, np.ndarray], weights: np.ndarray, rng: RandomState | None
    ) -> None:
        """Initialize statistics from encoded observations."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[float, float, float]) -> "InverseGaussianAccumulator":
        """Merge another inverse-Gaussian sufficient-statistic tuple."""
        self.count += suff_stat[0]
        self.sum += suff_stat[1]
        self.sum_inv += suff_stat[2]
        return self

    def value(self) -> tuple[float, float, float]:
        """Return count, sum, and reciprocal sum."""
        return self.count, self.sum, self.sum_inv

    def from_value(self, x: tuple[float, float, float]) -> "InverseGaussianAccumulator":
        """Replace accumulator contents from a sufficient-statistic tuple."""
        self.count = x[0]
        self.sum = x[1]
        self.sum_inv = x[2]
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

    def acc_to_encoder(self) -> "InverseGaussianDataEncoder":
        """Return the encoder used by this accumulator."""
        return InverseGaussianDataEncoder()


class InverseGaussianAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for InverseGaussianAccumulator."""

    def __init__(self, keys: str | None = None) -> None:
        self.keys = keys

    def make(self) -> InverseGaussianAccumulator:
        """Create a fresh inverse-Gaussian accumulator."""
        return InverseGaussianAccumulator(keys=self.keys)


class InverseGaussianEstimator(ParameterEstimator):
    """Maximum-likelihood estimator for the inverse Gaussian mean and shape.

    The MLE is closed form: ``mu = mean(x)`` and ``1/lam = mean(1/x) - 1/mu``.
    """

    def __init__(
        self,
        pseudo_count: float | None = None,
        suff_stat: tuple[float, float] | None = None,
        min_param: float = _MIN_IG_PARAM,
        max_param: float = _MAX_IG_PARAM,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        """Create an estimator for inverse-Gaussian parameters.

        Args:
            pseudo_count (Optional[float]): Re-weight the prior moments in ``suff_stat`` when not None.
            suff_stat (Optional[Tuple[float, float]]): Prior (mean, inverse-mean) targets for the
                pseudo-count ridge.
            min_param (float): Lower clamp for the estimated mu and lam.
            max_param (float): Upper clamp for the estimated lam.
            name (Optional[str]): Assign a name to the estimator.
            keys (Optional[str]): Assign keys for combining sufficient statistics.

        """
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.min_param = min_param
        self.max_param = max_param
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> InverseGaussianAccumulatorFactory:
        """Return an accumulator factory for inverse-Gaussian statistics."""
        return InverseGaussianAccumulatorFactory(keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat: tuple[float, float, float]) -> InverseGaussianDistribution:
        """Estimate mean and shape from count, sum, and reciprocal sum."""
        count, sum_x, sum_inv = suff_stat
        if self.pseudo_count is not None and self.suff_stat is not None:
            mean0, inv_mean0 = self.suff_stat
            sum_x += self.pseudo_count * mean0
            sum_inv += self.pseudo_count * inv_mean0
            count += self.pseudo_count

        if count <= 0.0 or sum_x <= 0.0 or not np.isfinite(sum_x):
            return InverseGaussianDistribution(1.0, 1.0, name=self.name, keys=self.keys)

        mu = max(sum_x / count, self.min_param)
        # 1/lam = mean(1/x) - 1/mean(x); guard the harmonic gap against round-off.
        inv_lam = sum_inv / count - 1.0 / mu
        if not np.isfinite(inv_lam) or inv_lam <= 0.0:
            lam = self.max_param
        else:
            lam = min(max(1.0 / inv_lam, self.min_param), self.max_param)
        return InverseGaussianDistribution(mu, lam, name=self.name, keys=self.keys)


class InverseGaussianDataEncoder(DataSequenceEncoder):
    """Encode inverse Gaussian observations as x, 1/x, and log(x)."""

    def __str__(self) -> str:
        return "InverseGaussianDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, InverseGaussianDataEncoder)

    def seq_encode(self, x: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Encode observations as values, reciprocal values, and log-values."""
        rv = np.asarray(x, dtype=np.float64)
        if rv.size and (np.any(rv <= 0.0) or np.any(~np.isfinite(rv))):
            raise ValueError("InverseGaussianDistribution has support x > 0.")
        with np.errstate(divide="ignore"):
            inv = 1.0 / rv
            lx = np.log(rv)
        return rv, inv, lx
