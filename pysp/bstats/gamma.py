"""Gamma distribution for positive real-valued data (floats x > 0), in the
shape/scale parameterization

    p(x | k, theta) = x^(k-1) exp(-x/theta) / (Gamma(k) theta^k).

Estimation is maximum likelihood (optionally smoothed with pseudo-counts):
the shape k solves log(k) - digamma(k) = log(mean(x)) - mean(log(x)) via
Newton iterations from the standard Minka starting point, and the scale is
theta = mean(x)/k.
"""

import math

import numpy as np
import scipy.integrate
from numpy.random import RandomState

from pysp.arithmetic import *
from pysp.bstats.nulldist import null_dist
from pysp.bstats.pdist import ParameterEstimator, ProbabilityDistribution, StatisticAccumulator
from pysp.utils.special import digamma, gammaln

_MIN_GAMMA_PARAM = 1.0e-12
_MIN_GAMMA_SCALE = float(np.finfo(float).tiny)
_MAX_GAMMA_SHAPE = 1.0e12


def _finite_positive(x: float, default: float = 1.0) -> float:
    x = float(x)
    return x if np.isfinite(x) and x > 0.0 else default


class GammaDistribution(ProbabilityDistribution):
    """Gamma distribution with shape k > 0 and scale theta > 0 for positive
    real-valued observations."""

    def __init__(self, k: float, theta: float, name: str | None = None, prior: ProbabilityDistribution = null_dist):
        """GammaDistribution object.

        Args:
                k (float): Shape parameter (k > 0).
                theta (float): Scale parameter (theta > 0).
                name (Optional[str]): Name for the distribution.
                prior (ProbabilityDistribution): Prior on (k, theta). Defaults to null_dist.
        """
        self.k = _finite_positive(k)
        self.theta = _finite_positive(theta)
        self.log_const = -(gammaln(self.k) + self.k * log(self.theta))
        self.prior = prior
        self.name = name
        self.parents = []

    def __str__(self):
        return "GammaDistribution(%f, %f, name=%s, prior=%s)" % (self.k, self.theta, str(self.name), str(self.prior))

    def get_parameters(self):
        """Returns the parameter tuple (k, theta)."""
        return self.k, self.theta

    def set_parameters(self, params):
        """Sets the shape and scale parameters.

        Args:
                params (Tuple[float, float]): Tuple (k, theta).
        """
        self.k = params[0]
        self.theta = params[1]

    def get_prior(self):
        """Returns the prior distribution on (k, theta)."""
        return self.prior

    def set_prior(self, prior):
        """Sets the prior distribution on (k, theta).

        Args:
                prior (ProbabilityDistribution): New prior distribution.
        """
        self.prior = prior

    def cross_entropy(self, dist: ProbabilityDistribution):
        """Cross entropy -E_self[log dist(x)].

        Closed form for Gamma arguments; numeric quadrature otherwise.

        Args:
                dist (ProbabilityDistribution): Distribution evaluated under this one.

        Returns:
                float: Cross entropy value in nats.
        """
        if isinstance(dist, GammaDistribution):
            k = self.k
            t = self.theta
            kk = dist.k
            tt = dist.theta
            return -((kk - 1) * (digamma(k) + np.log(t)) - gammaln(kk) - np.log(tt) * kk - k * t / tt)
        else:
            return -scipy.integrate.quad(lambda x: dist.log_density(x) * self.density(x), 0, np.inf)[0]

    def entropy(self):
        """Returns the differential entropy in nats."""
        return self.k + np.log(self.theta) + gammaln(self.k) + (1 - self.k) * digamma(self.k)

    def density(self, x):
        """Density of the gamma distribution at x.

        Args:
                x (float): Positive observation.

        Returns:
                float: Density value exp(log_density(x)).
        """
        if x <= 0.0:
            return 0.0
        return exp(self.log_const + (self.k - one) * log(x) - x / self.theta)

    def log_density(self, x):
        """Log density of the gamma distribution at x.

        Args:
                x (float): Positive observation.

        Returns:
                float: Log density value.
        """
        if x <= 0.0:
            return -np.inf
        return self.log_const + (self.k - one) * log(x) - x / self.theta

    def seq_log_density(self, x):
        """Vectorized log density for sequence-encoded observations.

        Args:
                x: Encoding (values, log values) from seq_encode.

        Returns:
                np.ndarray: Log density for each encoded observation.
        """
        rv = x[0] * (-1.0 / (self.theta))
        if self.k != 1.0:
            rv += x[1] * (self.k - 1.0)
        rv += self.log_const
        rv[x[0] <= 0.0] = -np.inf
        return rv

    def seq_encode(self, x):
        """Encodes an iterable of positive floats for vectorized evaluation.

        Args:
                x: Iterable of positive observations.

        Returns:
                Tuple[np.ndarray, np.ndarray]: (values, log values).
        """
        rv1 = np.asarray(x)
        rv2 = np.log(rv1)
        return rv1, rv2

    def sampler(self, seed=None):
        """Returns a GammaSampler for this distribution.

        Args:
                seed (Optional[int]): Random seed.

        Returns:
                GammaSampler object.
        """
        return GammaSampler(self, seed)

    def estimator(self):
        """Returns a GammaEstimator carrying this distribution's name and prior."""
        return GammaEstimator(name=self.name, prior=self.prior)


class GammaSampler:
    """Sampler for GammaDistribution. Draws positive floats."""

    def __init__(self, dist, seed=None):
        """GammaSampler object.

        Args:
                dist (GammaDistribution): Distribution to sample from.
                seed (Optional[int]): Random seed.
        """
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size=None):
        """Draw gamma-distributed samples.

        Args:
                size (Optional[int]): Number of samples to draw.

        Returns:
                float if size is None, else np.ndarray of length size.
        """
        return self.rng.gamma(shape=self.dist.k, scale=self.dist.theta, size=size)


class GammaAccumulator(StatisticAccumulator):
    """Accumulates the gamma sufficient statistics (count, sum, sum of logs)."""

    def __init__(self, keys):
        """GammaAccumulator object.

        Args:
                keys (Optional[str]): Key for merging statistics across accumulators.
        """
        self.nobs = zero
        self.sum = zero
        self.sum_of_logs = zero
        self.key = keys

    def initialize(self, x, weight, rng):
        """Initializes the accumulator with a weighted observation.

        Args:
                x (float): Positive observation.
                weight (float): Observation weight.
                rng: Random number generator. Unused.
        """
        self.update(x, weight, None)

    def seq_initialize(self, x, weights, rng):
        """Initializes the accumulator with sequence-encoded observations.

        Args:
                x: Encoding (values, log values) from GammaDistribution.seq_encode.
                weights (np.ndarray): Observation weights.
                rng: Random number generator. Unused.
        """
        self.seq_update(x, weights, None)

    def update(self, x, weight, estimate):
        """Adds a weighted observation to the sufficient statistics.

        Args:
                x (float): Positive observation.
                weight (float): Observation weight.
                estimate (Optional[GammaDistribution]): Current estimate. Unused.
        """
        self.nobs += weight
        self.sum += x * weight
        self.sum_of_logs += log(x) * weight

    def seq_update(self, x, weights, estimate):
        """Adds sequence-encoded weighted observations to the statistics.

        Args:
                x: Encoding (values, log values) from GammaDistribution.seq_encode.
                weights (np.ndarray): Observation weights.
                estimate (Optional[GammaDistribution]): Current estimate. Unused.
        """
        self.sum += np.dot(x[0], weights)
        self.sum_of_logs += np.dot(x[1], weights)
        self.nobs += np.sum(weights)

    def combine(self, suff_stat):
        """Adds another accumulator's sufficient statistics to this one.

        Args:
                suff_stat (Tuple[float, float, float]): (count, sum, sum of logs).

        Returns:
                GammaAccumulator: This accumulator.
        """
        self.nobs += suff_stat[0]
        self.sum += suff_stat[1]
        self.sum_of_logs += suff_stat[2]

        return self

    def value(self):
        """Returns the sufficient statistic tuple (count, sum, sum of logs)."""
        return self.nobs, self.sum, self.sum_of_logs

    def from_value(self, x):
        """Sets the sufficient statistics from a value() tuple.

        Args:
                x (Tuple[float, float, float]): (count, sum, sum of logs).

        Returns:
                GammaAccumulator: This accumulator.
        """
        self.nobs = x[0]
        self.sum = x[1]
        self.sum_of_logs = x[2]
        return self

    def key_merge(self, stats_dict):
        """Merges this accumulator into stats_dict under its key (if keyed)."""
        if self.key is not None:
            if self.key in stats_dict:
                stats_dict[self.key].combine(self.value())
            else:
                stats_dict[self.key] = self

    def key_replace(self, stats_dict):
        """Replaces this accumulator's statistics from stats_dict (if keyed)."""
        if self.key is not None:
            if self.key in stats_dict:
                self.from_value(stats_dict[self.key].value())


class GammaEstimator(ParameterEstimator):
    """Maximum-likelihood estimator for GammaDistribution, with optional
    pseudo-count smoothing of the mean and mean-log statistics."""

    def __init__(
        self,
        pseudo_count=(0.0, 0.0),
        suff_stat=(1.0, 0.0),
        threshold=1.0e-8,
        name: str | None = None,
        prior: ProbabilityDistribution = null_dist,
        keys=None,
    ):
        """GammaEstimator object.

        Args:
                pseudo_count (Tuple[float, float]): Pseudo-counts blended into the
                        (sum, sum-of-logs) statistics respectively.
                suff_stat (Tuple[float, float]): Per-pseudo-observation (mean,
                        mean-log) values used with pseudo_count.
                threshold (float): Convergence threshold for the shape iteration.
                name (Optional[str]): Name passed to estimated distributions.
                prior (ProbabilityDistribution): Prior on (k, theta) passed to
                        estimated distributions. Defaults to null_dist.
                keys (Optional[str]): Key for merging statistics across accumulators.
        """
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.threshold = threshold
        self.name = name
        self.prior = prior
        self.keys = keys

    def accumulator_factory(self):
        """Returns a factory whose make() creates a GammaAccumulator."""
        obj = type("", (object,), {"make": lambda o: GammaAccumulator(self.keys)})()
        return obj

    def get_prior(self):
        """Returns the prior distribution on (k, theta)."""
        return self.prior

    def set_prior(self, prior):
        """Sets the prior distribution on (k, theta).

        Args:
                prior (ProbabilityDistribution): New prior distribution.
        """
        self.prior = prior

    def estimate(self, suff_stat):
        """Estimates a GammaDistribution from sufficient statistics.

        Args:
                suff_stat (Tuple[float, float, float]): (count, sum, sum of logs)
                        as produced by GammaAccumulator.value().

        Returns:
                GammaDistribution: Maximum-likelihood estimate, smoothed by any
                configured pseudo-counts.
        """
        pc1, pc2 = self.pseudo_count
        ss1, ss2 = self.suff_stat

        if suff_stat[0] <= 0:
            return GammaDistribution(1.0, 1.0, name=self.name, prior=self.prior)

        adj_sum = suff_stat[1] + ss1 * pc1
        adj_cnt = suff_stat[0] + pc1
        if adj_cnt <= 0.0 or adj_sum <= 0.0 or not np.isfinite(adj_sum):
            return GammaDistribution(1.0, 1.0, name=self.name, prior=self.prior)
        adj_mean = adj_sum / adj_cnt

        adj_lsum = suff_stat[2] + ss2 * pc2
        adj_lcnt = suff_stat[0] + pc2
        if adj_lcnt <= 0.0 or not np.isfinite(adj_lsum):
            return GammaDistribution(1.0, adj_mean, name=self.name, prior=self.prior)
        adj_lmean = adj_lsum / adj_lcnt

        k = self.estimate_shape(adj_mean, adj_lmean, self.threshold)

        return GammaDistribution(k, max(_MIN_GAMMA_SCALE, adj_sum / (k * adj_cnt)), name=self.name, prior=self.prior)

    @staticmethod
    def estimate_shape(avg_sum, avg_sum_of_logs, threshold):
        """Solves for the ML shape parameter k.

        Newton iterations on log(k) - digamma(k) = s where
        s = log(avg_sum) - avg_sum_of_logs, started from Minka's approximation.

        Args:
                avg_sum (float): Mean of the observations.
                avg_sum_of_logs (float): Mean of the log observations.
                threshold (float): Convergence threshold on successive k values.

        Returns:
                float: Estimated shape parameter k.
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

        def shape_eq(k):
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
