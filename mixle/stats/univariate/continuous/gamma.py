"""Gamma distributions, estimators, accumulators, samplers, and encoders.

For positive real-valued observations, ``GammaDistribution(k, theta)`` uses
shape ``k > 0`` and scale ``theta > 0`` with log-density:

    log(f(x;k,theta)) = -gammaln(k) - k*log(theta) + (k-1) * log(x) - x / theta.
Values outside the positive support score ``-inf``.


Reference: Johnson, Kotz & Balakrishnan, *Continuous Univariate Distributions* (2nd ed., Wiley, 1994/95).
"""

import math
from collections.abc import Sequence
from typing import Any, Optional

import numpy as np
from numpy.random import RandomState

from mixle.engines.arithmetic import *
from mixle.inference.fisher import FixedFisherView
from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)
from mixle.utils.aliasing import broadcast_pseudo_count
from mixle.utils.special import digamma, gammaln, trigamma

_MIN_GAMMA_PARAM = 1.0e-12
_MIN_GAMMA_SCALE = float(np.finfo(float).tiny)
_MAX_GAMMA_SHAPE = 1.0e12


class GammaFisherView(FixedFisherView):
    """Expose Gamma sufficient statistics for Fisher-information utilities."""

    def __init__(self, dist: Any) -> None:
        super().__init__(dist, [("count",), ("sum_log",), ("sum",)])

    @staticmethod
    def _matrix_from_values(x: Any, log_x: Any | None = None) -> np.ndarray:
        xx = np.asarray(x, dtype=np.float64).reshape(-1)
        lx = np.log(xx) if log_x is None else np.asarray(log_x, dtype=np.float64).reshape(-1)
        return np.column_stack((np.ones_like(xx, dtype=np.float64), lx, xx))

    def _statistics_from_data(self, data: Sequence[Any], estimate: Any | None = None) -> np.ndarray:
        return self._matrix_from_values(data)

    def _statistics_from_encoded(self, enc_data: Any, estimate: Any | None = None) -> np.ndarray:
        if isinstance(enc_data, tuple):
            return self._matrix_from_values(enc_data[0], enc_data[1])
        return self._matrix_from_values(enc_data)

    def _model_mean(self) -> np.ndarray:
        k = float(self.dist.k)
        theta = float(self.dist.theta)
        return np.asarray([1.0, digamma(k) + math.log(theta), k * theta], dtype=np.float64)

    def _model_fisher(self) -> np.ndarray:
        k = float(self.dist.k)
        theta = float(self.dist.theta)
        out = np.zeros((3, 3), dtype=np.float64)
        out[1, 1] = trigamma(k)
        out[1, 2] = theta
        out[2, 1] = theta
        out[2, 2] = k * theta * theta
        return out


class GammaDistribution(SequenceEncodableProbabilityDistribution):
    """Gamma distribution parameterized by shape and scale."""

    @classmethod
    def compute_capabilities(cls):
        """Declare backend support for generated Gamma density kernels."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch", "jax"), kernel_status="numba_adapter")

    @classmethod
    def compute_declaration(cls):
        """Return the generated-compute declaration for the Gamma distribution."""
        from mixle.stats.compute.declarations import (
            DistributionDeclaration,
            ExponentialFamilySpec,
            ParameterSpec,
            StatisticSpec,
        )

        return DistributionDeclaration(
            name="gamma",
            distribution_type=cls,
            parameters=(
                ParameterSpec("k", constraint="positive"),
                ParameterSpec("theta", constraint="positive"),
            ),
            statistics=(
                StatisticSpec("count"),
                StatisticSpec("sum"),
                StatisticSpec("sum_of_logs"),
            ),
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
    def exp_family_sufficient_statistics(x: tuple[Any, Any], engine: Any) -> tuple[Any, ...]:
        """Return Gamma sufficient statistics for generated scoring."""
        vals, log_vals = x
        return engine.asarray(log_vals), engine.asarray(vals)

    @staticmethod
    def exp_family_legacy_sufficient_statistics(
        x: tuple[Any, Any], params: dict[str, Any], engine: Any
    ) -> tuple[Any, ...]:
        """Return per-row Gamma sufficient statistics in accumulator order."""
        vals = engine.asarray(x[0])
        log_vals = engine.asarray(x[1])
        return vals * 0.0 + engine.asarray(1.0), vals, log_vals

    @staticmethod
    def exp_family_natural_parameters(params: dict[str, Any], engine: Any) -> tuple[Any, ...]:
        """Return Gamma natural parameters for generated scoring."""
        return params["k"] - engine.asarray(1.0), -engine.asarray(1.0) / params["theta"]

    @staticmethod
    def exp_family_log_partition(params: dict[str, Any], engine: Any) -> Any:
        """Return Gamma log partition for generated scoring."""
        k = params["k"]
        theta = params["theta"]
        return engine.gammaln(k) + k * engine.log(theta)

    @staticmethod
    def exp_family_base_measure(x: tuple[Any, Any], engine: Any) -> Any:
        """Return Gamma support base measure for generated scoring."""
        vals = engine.asarray(x[0])
        return engine.where(vals > 0.0, vals * 0.0, engine.asarray(-np.inf))

    def __init__(self, k: float, theta: float, name: str | None = None) -> None:
        """Create a Gamma distribution.

        Args:
            k: Positive finite shape parameter.
            theta: Positive finite scale parameter.
            name: Optional diagnostic name.

        Attributes:
            k: Shape parameter.
            theta: Scale parameter.
            name: Optional diagnostic name.
            log_const: Log normalizing constant.

        """
        if k <= 0.0 or not np.isfinite(k):
            raise ValueError("GammaDistribution requires finite k > 0.")
        if theta <= 0.0 or not np.isfinite(theta):
            raise ValueError("GammaDistribution requires finite theta > 0.")
        self.k = float(k)
        self.theta = float(theta)
        self.log_const = -(gammaln(self.k) + self.k * log(self.theta))
        self.name = name

    def __str__(self) -> str:
        """Return a readable distribution summary."""
        return "GammaDistribution(%s, %s, name=%s)" % (repr(self.k), repr(self.theta), repr(self.name))

    def get_parameters(self) -> tuple[float, float]:
        """Return the (shape k, scale theta) pair.

        Lets a GammaDistribution serve as a conjugate prior (on a Poisson/Exponential rate, or a
        Gamma scale) under the unified Bayesian estimation protocol.
        """
        return self.k, self.theta

    def cross_entropy(self, dist: "GammaDistribution") -> float:
        """Cross entropy -E_self[log dist(x)] for a Gamma argument (closed form).

        Used as the conjugate prior/posterior cross-entropy term in variational Bayes (e.g. the
        ELBO global term in DPM for Poisson/Exponential-rate components).
        """
        if isinstance(dist, GammaDistribution):
            k = self.k
            t = self.theta
            kk = dist.k
            tt = dist.theta
            return float(-((kk - 1) * (digamma(k) + np.log(t)) - gammaln(kk) - np.log(tt) * kk - k * t / tt))
        raise NotImplementedError(
            "GammaDistribution.cross_entropy is only implemented for Gamma arguments (got %s)." % type(dist).__name__
        )

    def entropy(self) -> float:
        """Return the differential entropy in nats."""
        return float(self.k + np.log(self.theta) + gammaln(self.k) + (1 - self.k) * digamma(self.k))

    def density(self, x: float) -> float:
        """Density of gamma distribution evaluated at x.

        See log_density() for details.

        Args:
            x (float): Positive real-valued number.

        Returns:
            Density of gamma distribution evaluated at x.

        """
        try:
            xx = float(x)
        except Exception:  # noqa: BLE001
            return 0.0
        if not np.isfinite(xx) or xx <= 0.0:
            return 0.0
        return exp(self.log_const + (self.k - one) * log(xx) - xx / self.theta)

    def log_density(self, x: float) -> float:
        """Log-density of gamma distribution evaluated at x.

        Log-density given by,
        If x > 0.0,
            log(f(x;k,theta)) = -gammaln(k) - k*log(theta) + (k-1) * log(x) - x / theta,
        else,
            -np.inf
        Args:
            x (float): Positive real-valued number.

        Returns:
            Log-density of gamma distribution evaluated at x.

        """
        try:
            xx = float(x)
        except Exception:  # noqa: BLE001
            return -np.inf
        if not np.isfinite(xx) or xx <= 0.0:
            return -np.inf
        return self.log_const + (self.k - one) * log(xx) - xx / self.theta

    def seq_log_density(self, x: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """Vectorized evaluation of sequence encoded observations from gamma distribution.

        Input must be x (Tuple[ndarray, ndarray]):
            x[0]: Numpy array of floats containing observations from gamma distribution.
            x[1]: Numpy array of floats containing log of observation values.

        Args:
            x (Tuple[np.ndarray, np.ndarray]): See above for details.

        Returns:
            Numpy array containing log-density evaluated at all observations of encoded sequence x.

        """
        rv = x[0] * (-1.0 / self.theta)
        if self.k != 1.0:
            rv += x[1] * (self.k - 1.0)
        rv += self.log_const

        return np.where(np.isfinite(x[0]) & (x[0] > 0.0), rv, -np.inf)

    @staticmethod
    def backend_log_density_from_params(vals: Any, log_vals: Any, k: Any, theta: Any, engine: Any) -> Any:
        """Engine-neutral gamma log-density from explicit parameters."""
        rv = -engine.gammaln(k) - k * engine.log(theta) + (k - engine.asarray(1.0)) * log_vals - vals / theta
        return engine.where(vals > 0.0, rv, engine.asarray(-np.inf))

    def backend_seq_log_density(self, x: tuple[Any, Any], engine: Any) -> Any:
        """Engine-neutral vectorized log-density for encoded data."""
        vals = engine.asarray(x[0])
        log_vals = engine.asarray(x[1])
        k = engine.asarray(self.k)
        theta = engine.asarray(self.theta)
        return self.backend_log_density_from_params(vals, log_vals, k, theta, engine)

    @classmethod
    def backend_stacked_params(cls, dists: Sequence["GammaDistribution"], engine: Any) -> dict[str, Any]:
        """Return stacked Gamma parameters for a homogeneous mixture kernel."""
        return {
            "k": engine.asarray([d.k for d in dists]),
            "theta": engine.asarray([d.theta for d in dists]),
        }

    @classmethod
    def backend_stacked_log_density(cls, x: tuple[Any, Any], params: dict[str, Any], engine: Any) -> Any:
        """Return an ``(n, k)`` matrix of Gamma log densities."""
        vals = engine.asarray(x[0])
        log_vals = engine.asarray(x[1])
        return cls.backend_log_density_from_params(
            vals[:, None], log_vals[:, None], params["k"][None, :], params["theta"][None, :], engine
        )

    @classmethod
    def backend_stacked_sufficient_statistics(
        cls, x: tuple[Any, Any], weights: Any, params: dict[str, Any], engine: Any
    ) -> tuple[Any, Any, Any]:
        """Return stacked Gamma sufficient statistics using engine-resident arrays."""
        vals = engine.asarray(x[0])
        log_vals = engine.asarray(x[1])
        ww = engine.asarray(weights)
        return (
            engine.sum(ww, axis=0),
            engine.sum(ww * vals[:, None], axis=0),
            engine.sum(ww * log_vals[:, None], axis=0),
        )

    def cdf(self, x: float) -> float:
        """Cumulative distribution function ``P(X <= x)`` (exact). The continuous 'index of' a value."""
        from scipy.stats import gamma as _sp

        return float(_sp.cdf(x, self.k, scale=self.theta))

    def quantile(self, q: float) -> float:
        """Inverse CDF ``F^{-1}(q)``: the value at cumulative-probability index ``q`` (continuous unranking)."""
        from scipy.stats import gamma as _sp

        return float(_sp.ppf(q, self.k, scale=self.theta))

    def to_fisher(self, **kwargs):
        """Return this distribution's own Fisher view."""
        return GammaFisherView(self)

    def mean(self) -> float:
        """Mean E[X] of the distribution."""
        return float(self.k * self.theta)

    def variance(self) -> float:
        """Variance Var[X] of the distribution."""
        return float(self.k * self.theta * self.theta)

    def skewness(self) -> float:
        """Skewness 2/sqrt(k)."""
        import math

        return float(2.0 / math.sqrt(self.k))

    def kurtosis(self) -> float:
        """Excess kurtosis 6/k."""
        return float(6.0 / self.k)

    def mode(self) -> float:
        """Mode (k-1)*theta for k>=1, else 0."""
        return float((self.k - 1.0) * self.theta) if self.k >= 1.0 else 0.0

    def sampler(self, seed: int | None = None) -> "GammaSampler":
        """Return a sampler for iid draws from this distribution.

        Args:
            seed: Optional seed for the sampler's random state.

        Returns:
            A configured ``GammaSampler``.

        """
        return GammaSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "GammaEstimator":
        """Return an estimator initialized from this distribution's shape.

        Args:
            pseudo_count: Optional smoothing count applied to the current
                distribution moments.

        Returns:
            A ``GammaEstimator``.

        """
        if pseudo_count is None:
            return GammaEstimator(name=self.name)
        else:
            suff_stat = (self.k * self.theta, digamma(self.k) + log(self.theta))
            return GammaEstimator(pseudo_count=(pseudo_count, pseudo_count), suff_stat=suff_stat, name=self.name)

    def dist_to_encoder(self) -> "GammaDataEncoder":
        """Return an encoder for iid Gamma observations."""
        return GammaDataEncoder()


class GammaSampler(DistributionSampler):
    """Draw independent samples from a :class:`GammaDistribution`."""

    def __init__(self, dist: "GammaDistribution", seed: int | None = None) -> None:
        """Create a sampler bound to ``dist``.

        Args:
            dist: Distribution to sample from.
            seed: Optional seed for the sampler's random state.

        Attributes:
            rng: Random state used for draws.
            dist: Distribution being sampled.
            seed: Seed used to initialize the random state.


        """
        self.rng = RandomState(seed)
        self.dist = dist
        self.seed = seed

    def sample(self, size: int | None = None) -> float | np.ndarray:
        """Draw iid observations from the Gamma distribution.

        Args:
            size: Number of iid samples to draw. ``None`` returns a scalar sample.

        Returns:
            A scalar draw when ``size`` is ``None``; otherwise an array of draws.

        """
        return self.rng.gamma(shape=self.dist.k, scale=self.dist.theta, size=size)


class GammaAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate weighted count, sum, and log-sum statistics for Gamma estimation."""

    def __init__(self, keys: str | None = None) -> None:
        """Create an accumulator for Gamma sufficient statistics.

        Args:
            keys: Optional key for merging sufficient statistics.

        Attributes:
            count: Sum of observation weights.
            sum: Weighted sum of observations.
            sum_of_logs: Weighted sum of log-observations.
            keys: Optional sufficient-statistic key.

        """
        self.count = zero
        self.sum = zero
        self.sum_of_logs = zero
        self.keys = keys

    def initialize(self, x: float, weight: float, rng: RandomState | None) -> None:
        """Initialize sufficient statistics of GammaAccumulator with weighted observation.

        This delegates to :meth:`update`.

        Args:
            x (float): Positive real-valued observation of gamma.
            weight (float): Positive real-valued weight for observation x.
            rng (Optional[RandomState]): Kept for consistency with SequenceEncodableStatisticAccumulator.

        Returns:
            None.

        """
        self.update(x, weight, None)

    def seq_initialize(self, x: tuple[np.ndarray, np.ndarray], weights: np.ndarray, rng: RandomState | None) -> None:
        """Vectorized initialization of GammaAccumulator sufficient statistics with weighted observations.

        This delegates to :meth:`seq_update`.

        Args:
            x (Tuple[ndarray, ndarray]): Tuple of Numpy array of observations and log(observations).
            weights (ndarray): Numpy array of positive floats.
            rng (Optional[RandomState]): Kept for consistency with SequenceEncodableStatisticAccumulator.

        Returns:
            None.

        """
        self.seq_update(x, weights, None)

    def update(self, x: float, weight: float, estimate: Optional["GammaDistribution"]) -> None:
        """Update sufficient statistics for GammaAccumulator with one weighted observation.

        Args:
            x (float): Observation from gamma distribution.
            weight (float): Weight for observation.
            estimate (Optional[GammaDistribution]): Kept for consistency with SequenceEncodableStatisticAccumulator.

        Returns:
            None

        """
        if x <= 0.0 or not np.isfinite(x):
            raise ValueError("GammaDistribution has support x > 0.")
        self.count += weight
        self.sum += x * weight
        self.sum_of_logs += log(x) * weight

    def seq_update(
        self, x: tuple[np.ndarray, np.ndarray], weights: np.ndarray, estimate: Optional["GammaDistribution"]
    ) -> None:
        """Vectorized update of sufficient statistics from encoded sequence x.

        Args:
            x (Tuple[ndarray, ndarray]): Tuple of Numpy array of observations and log(observations).
            weights (ndarray): Numpy array of positive floats.
            estimate (Optional[GammaDistribution]): Kept for consistency with SequenceEncodableStatisticAccumulator.

        Returns:
            None.

        """
        self.sum += np.dot(x[0], weights)
        self.sum_of_logs += np.dot(x[1], weights)
        self.count += np.sum(weights)

    def combine(self, suff_stat: tuple[float, float, float]) -> "GammaAccumulator":
        """Aggregates sufficient statistics with GammaAccumulator member sufficient statistics.

        Args:
            suff_stat (Tuple[float, float, float]): Aggregated count, sum, sum_of_logs.

        Returns:
            ExponentialAccumulator

        """
        self.count += suff_stat[0]
        self.sum += suff_stat[1]
        self.sum_of_logs += suff_stat[2]

        return self

    def value(self) -> tuple[float, float, float]:
        """Return ``(count, sum, sum_of_logs)`` sufficient statistics."""
        return self.count, self.sum, self.sum_of_logs

    def from_value(self, x: tuple[float, float, float]) -> "GammaAccumulator":
        """Sets sufficient statistics GammaAccumulator to x.

        Args:
            x (Tuple[float, float, float]): Sufficient statistics tuple of length three..

        Returns:
            ExponentialAccumulator

        """
        self.count = x[0]
        self.sum = x[1]
        self.sum_of_logs = x[2]

        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Merge sufficient statistics from ``stats_dict`` when this accumulator's key is present.

        Args:
            stats_dict (Dict[str, Any]): Dict mapping keys to sufficient statistics.

        Returns:
            None.

        """
        if self.keys is not None:
            if self.keys in stats_dict:
                x0, x1, x2 = stats_dict[self.keys]
                self.count += x0
                self.sum += x1
                self.sum_of_logs += x2

            else:
                stats_dict[self.keys] = (self.count, self.sum, self.sum_of_logs)

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Replace sufficient statistics from ``suff_stats`` when this accumulator's key is present.

        Args:
            stats_dict (Dict[str, Any]): Dict mapping keys to sufficient statistics.

        Returns:
            None.

        """
        if self.keys is not None:
            if self.keys in stats_dict:
                x0, x1, x2 = stats_dict[self.keys]
                self.count = x0
                self.sum = x1
                self.sum_of_logs = x2

    def acc_to_encoder(self) -> "GammaDataEncoder":
        """Return an encoder compatible with Gamma observations."""
        return GammaDataEncoder()


class GammaAccumulatorFactory(StatisticAccumulatorFactory):
    """Create Gamma accumulators with a shared optional merge key."""

    def __init__(self, keys: str | None = None) -> None:
        """Create an accumulator factory.

        Args:
            keys: Optional key for merging sufficient statistics.

        Attributes:
            keys: Optional sufficient-statistic key.

        """
        self.keys = keys

    def make(self) -> "GammaAccumulator":
        """Return a fresh Gamma accumulator."""
        return GammaAccumulator(keys=self.keys)


class GammaEstimator(ParameterEstimator):
    """Estimate Gamma shape and scale parameters from sufficient statistics."""

    def __init__(
        self,
        pseudo_count: float | tuple[float, float] = (0.0, 0.0),
        suff_stat: tuple[float, float] = (1.0, 0.0),
        threshold: float = 1.0e-8,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        """Create an estimator for Gamma sufficient statistics.

        Args:
            pseudo_count: Smoothing weights for the mean and log-mean statistics. A scalar
                is broadcast to both slots.
            suff_stat: Prior mean and log-mean statistics used with ``pseudo_count``.
            threshold: Convergence threshold for shape estimation.
            name: Optional diagnostic name.
            keys: Optional key for merging sufficient statistics.

        Attributes:
            pseudo_count: Smoothing weights for sufficient statistics.
            suff_stat: Prior sufficient statistics.
            threshold: Shape-estimation convergence threshold.
            name: Optional diagnostic name.
            keys: Optional sufficient-statistic key.

        """
        pseudo_count = broadcast_pseudo_count(pseudo_count, 2)
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.threshold = threshold
        self.keys = keys
        self.name = name

    def accumulator_factory(self) -> "GammaAccumulatorFactory":
        """Return an accumulator factory matching this estimator."""
        return GammaAccumulatorFactory(keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat: tuple[float, float, float]) -> "GammaDistribution":
        """Estimate a Gamma distribution from aggregated sufficient statistics.

        The tuple is interpreted as ``(count, sum, sum_of_logs)``.

        Args:
            nobs: Unused; accepted for the ``ParameterEstimator`` interface.
            suff_stat: Aggregated Gamma sufficient statistics.

        Returns:
            A fitted Gamma distribution.

        """
        pc1, pc2 = self.pseudo_count
        ss1, ss2 = self.suff_stat

        if suff_stat[0] <= 0:
            return GammaDistribution(1.0, 1.0, name=self.name)

        adj_sum = suff_stat[1] + ss1 * pc1
        adj_cnt = suff_stat[0] + pc1
        if adj_cnt <= 0.0 or adj_sum <= 0.0 or not np.isfinite(adj_sum):
            return GammaDistribution(1.0, 1.0, name=self.name)
        adj_mean = adj_sum / adj_cnt

        adj_lsum = suff_stat[2] + ss2 * pc2
        adj_lcnt = suff_stat[0] + pc2
        if adj_lcnt <= 0.0 or not np.isfinite(adj_lsum):
            return GammaDistribution(1.0, adj_mean, name=self.name)
        adj_lmean = adj_lsum / adj_lcnt

        k = self.estimate_shape(adj_mean, adj_lmean, self.threshold)

        # theta = mean / k, where the mean uses the count adjusted by pc1 (adj_lcnt
        # uses pc2 and is only valid for the log-mean).
        return GammaDistribution(k, max(_MIN_GAMMA_SCALE, adj_mean / k), name=self.name)

    @staticmethod
    def estimate_shape(avg_sum: float, avg_sum_of_logs: float, threshold: float) -> float:
        """Estimates the shape parameter of GammaDistribution.

        Args:
            avg_sum (float): Weighted sum of gamma observations.
            avg_sum_of_logs (float): Weighted log sum of gamma observations.
            threshold (float): Threshold used for assessing convergence of shape estimation.

        Returns:
            Estimate of shape parameter 'k'.

        """
        avg_sum = float(avg_sum)
        avg_sum_of_logs = float(avg_sum_of_logs)
        if avg_sum <= 0.0 or not np.isfinite(avg_sum) or not np.isfinite(avg_sum_of_logs):
            return 1.0

        s = float(math.log(avg_sum) - avg_sum_of_logs)
        if not np.isfinite(s):
            return 1.0
        if s <= 0.0:
            return _MAX_GAMMA_SHAPE

        threshold = max(float(threshold), 1.0e-12)

        def shape_eq(k: float) -> float:
            return float(math.log(k) - digamma(k) - s)

        lo = _MIN_GAMMA_PARAM
        hi = min(_MAX_GAMMA_SHAPE, max(1.0, 1.0 / (2.0 * s)))
        f_lo = shape_eq(lo)
        if not np.isfinite(f_lo) or f_lo <= 0.0:
            return lo

        f_hi = shape_eq(hi)
        while np.isfinite(f_hi) and f_hi > 0.0 and hi < _MAX_GAMMA_SHAPE:
            hi = min(_MAX_GAMMA_SHAPE, hi * 2.0)
            f_hi = shape_eq(hi)
        if not np.isfinite(f_hi):
            return 1.0
        if f_hi > 0.0:
            return _MAX_GAMMA_SHAPE

        for _ in range(200):
            mid = 0.5 * (lo + hi)
            f_mid = shape_eq(mid)
            if not np.isfinite(f_mid):
                break
            if f_mid > 0.0:
                lo = mid
            else:
                hi = mid
            if hi - lo <= threshold * max(1.0, hi):
                break

        return min(_MAX_GAMMA_SHAPE, max(_MIN_GAMMA_PARAM, 0.5 * (lo + hi)))


class GammaDataEncoder(DataSequenceEncoder):
    """Encoder for iid positive Gamma observations."""

    def __str__(self) -> str:
        """Return a readable encoder summary."""
        return "GammaDataEncoder"

    def __eq__(self, other: object) -> bool:
        """Return true when ``other`` is a gamma data encoder.

        Args:
            other (object): An object to check for equality.

        Returns:
            True if object is instance of GammaDataEncoder, else False.

        """
        return isinstance(other, GammaDataEncoder)

    def seq_encode(self, x: list[float] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Encode iid sequence of gamma observations for vectorized "seq_" function calls.

        Note: Each entry of x must be positive float.

        Args:
            x (Union[List[float], np.ndarray]): IID sequence of gamma distributed observations.

        Returns:
            Tuple of x as numpy array and log(x).

        """
        rv1 = np.asarray(x, dtype=float)

        if np.any(rv1 <= 0) or np.any(~np.isfinite(rv1)):
            raise ValueError("GammaDistribution has support x > 0.")
        else:
            rv2 = np.log(rv1)
            return rv1, rv2
