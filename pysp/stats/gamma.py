"""Create, estimate, and sample from a gamma distribution with shape k and scale theta.

Defines the GammaDistribution, GammaSampler, GammaAccumulatorFactory, GammaAccumulator, GammaEstimator,
and the GammaDataEncoder classes for use with pysparkplug.

Data type: (float): The GammaDistribution with shape k > 0.0 and scale theta > 0.0, has log-density
    log(f(x;k,theta)) = -gammaln(k) - k*log(theta) + (k-1) * log(x) - x / theta, for x > 0.0, else -np.inf

"""

import math
from collections.abc import Sequence
from typing import Any, Optional

import numpy as np
from numpy.random import RandomState

from pysp.arithmetic import *
from pysp.stats.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)
from pysp.utils.special import digamma, gammaln

_MIN_GAMMA_PARAM = 1.0e-12
_MIN_GAMMA_SCALE = float(np.finfo(float).tiny)
_MAX_GAMMA_SHAPE = 1.0e12


class GammaDistribution(SequenceEncodableProbabilityDistribution):
    """Gamma distribution parameterized by shape and scale."""

    @classmethod
    def compute_capabilities(cls):
        from pysp.stats.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch"), kernel_status="numba_adapter")

    @classmethod
    def compute_declaration(cls):
        from pysp.stats.declarations import DistributionDeclaration, ExponentialFamilySpec, ParameterSpec, StatisticSpec

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
        """GammaDistribution for shape k and scale theta.

        Args:
            k (float): Positive real-valued number.
            theta (float): Positive real-valued number.
            name (Optional[str]): Assign a name to GammaDistribution instance.

        Attributes:
            k (float): Positive real-valued number.
            theta (float): Positive real-valued number.
            name (Optional[str]): Assign a name to GammaDistribution instance.
            log_const (float): Normalizing constant of gamma distribution.

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
        """Return string representation of GammaDistribution object."""
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
        """Returns the differential entropy in nats."""
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
        except Exception:
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
        except Exception:
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

    def sampler(self, seed: int | None = None) -> "GammaSampler":
        """Create a GammaSampler object from GammaDistribution.

        Args:
            seed (Optional[int]): Set seed on random number generator.

        Returns:
            GammaSampler object.

        """
        return GammaSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "GammaEstimator":
        """Creates GammaEstimator object from GammaDistribution instance.

        Args:
            pseudo_count (Optional[float]): Re-weight the sufficient statistics of GammaDistribution instance if not
                None.

        Returns:
            GammaEstimator object.

        """
        if pseudo_count is None:
            return GammaEstimator(name=self.name)
        else:
            suff_stat = (self.k * self.theta, exp(digamma(self.k) + log(self.theta)))
            return GammaEstimator(pseudo_count=(pseudo_count, pseudo_count), suff_stat=suff_stat, name=self.name)

    def dist_to_encoder(self) -> "GammaDataEncoder":
        """Returns GammaDataEncoder object for encoding sequence of GammaDistribution observations."""
        return GammaDataEncoder()


class GammaSampler(DistributionSampler):
    def __init__(self, dist: "GammaDistribution", seed: int | None = None) -> None:
        """GammaSampler object used to draw samples from GammaDistribution.

        Args:
            dist (GammaDistribution): GammaDistribution to sample from.
            seed (Optional[int]): Used to set seed on random number generator used in sampling.

        Attributes:
            rng (RandomState): RandomState with seed set for sampling.
            dist (GammaDistribution): GammaDistribution to sample from.
            seed (Optional[int]): Used to set seed on random number generator used in sampling.


        """
        self.rng = RandomState(seed)
        self.dist = dist
        self.seed = seed

    def sample(self, size: int | None = None) -> float | np.ndarray:
        """Draw 'size'-iid observations from GammaSampler.

        Args:
            size (Optional[int]): Number of iid samples to draw from GammaSampler.

        Returns:
            Single sample (float) if size is None, else a numpy array of floats containing iid samples from
            GammaDistribution.

        """
        return self.rng.gamma(shape=self.dist.k, scale=self.dist.theta, size=size)


class GammaAccumulator(SequenceEncodableStatisticAccumulator):
    def __init__(self, keys: str | None = None) -> None:
        """GammaAccumulator object used to accumulate sufficient statistics from observations.

        Args:
            keys (Optional[str]): GammaAccumulator objects with same key merge sufficient statistics.

        Attributes:
            nobs (float): Number of observations accumulated.
            sum (float): Weighted-sum of observations accumulated.
            sum_of_logs (float): log weighted sum of weighted log(observations).
            key (Optional[str]): GammaAccumulator objects with same key merge sufficient statistics.

        """
        self.nobs = zero
        self.sum = zero
        self.sum_of_logs = zero
        self.key = keys

    def initialize(self, x: float, weight: float, rng: RandomState | None) -> None:
        """Initialize sufficient statistics of GammaAccumulator with weighted observation.

        Note: Just calls update.

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

        Note: Just calls seq_update().

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
        self.nobs += weight
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
        self.nobs += np.sum(weights)

    def combine(self, suff_stat: tuple[float, float, float]) -> "GammaAccumulator":
        """Aggregates sufficient statistics with GammaAccumulator member sufficient statistics.

        Args:
            suff_stat (Tuple[float, float, float]): Aggregated sum, sum_of_logs, nobs.

        Returns:
            ExponentialAccumulator

        """
        self.nobs += suff_stat[0]
        self.sum += suff_stat[1]
        self.sum_of_logs += suff_stat[2]

        return self

    def value(self) -> tuple[float, float, float]:
        """Returns Tuple[float, float, float] containing sufficient statistics of GammaAccumulator."""
        return self.nobs, self.sum, self.sum_of_logs

    def from_value(self, x: tuple[float, float, float]) -> "GammaAccumulator":
        """Sets sufficient statistics GammaAccumulator to x.

        Args:
            x (Tuple[float, float, float]): Sufficient statistics tuple of length three..

        Returns:
            ExponentialAccumulator

        """
        self.nobs = x[0]
        self.sum = x[1]
        self.sum_of_logs = x[2]

        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Merge sufficient statistics of object instance with suff stats containing matching keys.

        Args:
            stats_dict (Dict[str, Any]): Dict mapping keys to sufficient statistics.

        Returns:
            None.

        """
        if self.key is not None:
            if self.key in stats_dict:
                x0, x1, x2 = stats_dict[self.key]
                self.nobs += x0
                self.sum += x1
                self.sum_of_logs += x2

            else:
                stats_dict[self.key] = (self.nobs, self.sum, self.sum_of_logs)

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Set sufficient statistics of object instance to suff_stats with matching keys.

        Args:
            stats_dict (Dict[str, Any]): Dict mapping keys to sufficient statistics.

        Returns:
            None.

        """
        if self.key is not None:
            if self.key in stats_dict:
                x0, x1, x2 = stats_dict[self.key]
                self.nobs = x0
                self.sum = x1
                self.sum_of_logs = x2

    def acc_to_encoder(self) -> "GammaDataEncoder":
        """Return GammaDataEncoder for encoding sequence of data."""
        return GammaDataEncoder()


class GammaAccumulatorFactory(StatisticAccumulatorFactory):
    def __init__(self, keys: str | None = None) -> None:
        """GammaAccumulatorFactory object for creating GammaAccumulator objects.

        Args:
            keys (Optional[str]): Used for merging sufficient statistics of GammaAccumulator.

        Attributes:
            keys (Optional[str]): Used for merging sufficient statistics of GammaAccumulator.

        """
        self.keys = keys

    def make(self) -> "GammaAccumulator":
        """Returns GammaAccumulator object with keys passed."""
        return GammaAccumulator(keys=self.keys)


class GammaEstimator(ParameterEstimator):
    def __init__(
        self,
        pseudo_count: tuple[float, float] = (0.0, 0.0),
        suff_stat: tuple[float, float] = (1.0, 0.0),
        threshold: float = 1.0e-8,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        """GammaEstimator object used for estimating GammaDistribution from aggregated data.

        Args:
            pseudo_count (Tuple[float, float]): Values used to re-weight member instances of sufficient statistics.
            suff_stat (Tuple[float, float]):  shape 'k' and scale 'theta'.
            threshold (float): Threshold used for estimating the shape of gamma.
            name (Optional[str]): Assign a name to GammaEstimator.
            keys (Optional[str]): Assign keys to GammaEstimator for combining sufficient statistics.

        Attributes:
            pseudo_count (Tuple[float, float]): Values used to re-weight member instances of sufficient statistics.
            suff_stat (Tuple[float, float]):  shape 'k' and scale 'theta'.
            threshold (float): Threshold used for estimating the shape of gamma.
            name (Optional[str]): Assign a name to GammaEstimator.
            keys (Optional[str]): Assign keys to GammaEstimator for combining sufficient statistics.

        """
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.threshold = threshold
        self.keys = keys
        self.name = name

    def accumulator_factory(self) -> "GammaAccumulatorFactory":
        """Create GammaAccumulatorFactory with keys passed."""
        return GammaAccumulatorFactory(keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat: tuple[float, float, float]) -> "GammaDistribution":
        """Obtain GammaDistribution from aggregated sufficient statistics of observed data.

        Takes sufficient statistic aggregated from observed data:
            suff_stat[0]: weighted sum of observations
            suff_stat[1]: weighted sum of log-observations
            suff_stat[2]: weighted observation count.

        Args:
            nobs (Optional[float]): Not used. Kept for consistency with ParameterEstimator.
            suff_stat: See description above for details.

        Returns:
            GammaDistribution object.

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
    """GammaDataEncoder object for encoding sequences of iid Gamma observations with data type float."""

    def __str__(self) -> str:
        """Return string representation of GammaDataEncoder."""
        return "GammaDataEncoder"

    def __eq__(self, other: object) -> bool:
        """Check if object is instance of GammaDataEncoder.

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
