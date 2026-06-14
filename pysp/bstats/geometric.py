"""Bayesian geometric distribution with success probability p.

Defines the GeometricDistribution, GeometricSampler, GeometricAccumulator,
GeometricAccumulatorFactory, and GeometricEstimator classes for use with
pysp.bstats.

Data type: (int): The GeometricDistribution with success probability
    p in [0, 1] has log-density
    log(f(x; p)) = (x-1)*log(1-p) + log(p), for integer x >= 1
    (number of trials up to and including the first success).

Conjugate prior: BetaDistribution on p. With prior Beta(a, b) and
observations x_1..x_m, the posterior is Beta(a + m, b + sum(x_i - 1)).
Estimation returns the posterior mode (MAP) clamped to the boundary values
0, 1, or 1/2 when the mode is undefined, carrying the posterior as the new
prior. expected_log_density evaluates the variational Bayes expectation
E_q[log p(x | p)] via digamma terms.
"""

import mpmath
import numpy as np
from numpy.random import RandomState
from scipy.special import digamma

from pysp.bstats.beta import BetaDistribution
from pysp.bstats.nulldist import NullDistribution
from pysp.bstats.pdist import ParameterEstimator, ProbabilityDistribution, StatisticAccumulator

default_prior = BetaDistribution(1.0, 1.0)


class GeometricDistribution(ProbabilityDistribution):
    """Geometric distribution with success probability p, optionally carrying
    a Beta conjugate prior on p."""

    def __init__(
        self,
        p: float,
        name: str | None = None,
        prior: ProbabilityDistribution = default_prior,
        keys: str | None = None,
    ):
        """GeometricDistribution object with success probability p.

        Args:
            p (float): Success probability in [0, 1].
            name (Optional[str]): Name of object.
            prior (ProbabilityDistribution): Prior on p; a BetaDistribution
                enables the conjugate machinery (see set_prior()).
            keys (Optional[str]): Key for sharing sufficient statistics.

        """
        self.parents = []
        self.name = name
        self.keys = keys
        self.set_parameters(p)
        self.set_prior(prior)

    def __str__(self):
        return "GeometricDistribution(%f, prior=%s, keys=%s, name=%s)" % (
            self.p,
            str(self.prior),
            str(self.keys),
            str(self.name),
        )

    def set_prior(self, dist: ProbabilityDistribution) -> None:
        """Set the prior and precompute conjugate-prior expectations.

        If dist is a BetaDistribution(a, b), this caches
        (digamma(a), digamma(b), digamma(a+b)), the terms needed for
        E[log p] = digamma(a) - digamma(a+b) and
        E[log(1-p)] = digamma(b) - digamma(a+b) in expected_log_density.
        Sets has_conj_prior accordingly.

        Args:
            dist (ProbabilityDistribution): Prior on p.

        """
        self.prior = dist

        if isinstance(dist, BetaDistribution):
            self.has_conj_prior = True
            self.conj_prior_params = (digamma(dist.a), digamma(dist.b), digamma(dist.a + dist.b))
        else:
            self.has_conj_prior = False
            self.conj_prior_params = (0, 0, 0)

    def get_prior(self) -> ProbabilityDistribution:
        """Returns the prior distribution on p."""
        return self.prior

    def get_parameters(self) -> float:
        """Returns the success probability p."""
        return self.p

    def set_parameters(self, params: float) -> None:
        """Set the success probability p.

        Args:
            params (float): Success probability in [0, 1].

        """
        self.p = params
        self.log_p = np.log(params) if params > 0.0 else -np.inf
        self.log_1p = np.log1p(-params) if params < 1.0 else -np.inf

    def entropy(self) -> float:
        """Returns the entropy of the geometric distribution (in nats)."""
        p = self.p
        return -(np.log1p(-p) * (1 - p) / p + np.log(p))

    def cross_entropy(self, dist) -> float:
        """Cross-entropy H(self, dist) for a GeometricDistribution argument.

        Args:
            dist: Distribution to evaluate against.

        Returns:
            Cross-entropy in nats, or None if dist is not geometric.

        """
        if isinstance(dist, GeometricDistribution):
            pp = dist.p
            p = self.p
            return -(np.log1p(-pp) * (1 - p) / p + np.log(pp))
        else:
            return None

    def moment(self, p):
        """Returns the p-th moment of the distribution (closed form for
        p in {1, 2}; polylogarithm evaluation otherwise).

        Args:
            p (int): Moment order.

        Returns:
            The p-th moment as a float.

        """
        order = int(p)
        p_loc = self.p
        if order == 0:
            return 1.0
        if p_loc <= 0.0:
            return np.inf
        if p_loc == 1.0:
            return 1.0
        if order == 1:
            return 1.0 / p_loc
        elif order == 2:
            return (2 - p_loc) / (p_loc * p_loc)
        else:
            q = 1.0 - p_loc
            aa = mpmath.polylog(-order, q)
            return float(aa) * p_loc / q

    def density(self, x: int) -> float:
        """Density of the geometric distribution at observation x.

        Args:
            x (int): Observation (number of trials, x >= 1).

        Returns:
            Density at observation x.

        """
        return np.exp(self.log_density(x))

    def log_density(self, x: int) -> float:
        """Log-density of the geometric distribution at observation x.

        Args:
            x (int): Observation (number of trials, x >= 1).

        Returns:
            Log-density at observation x.

        """
        if x < 1:
            return -np.inf
        if self.p == 1.0:
            return 0.0 if x == 1 else -np.inf
        elif self.p == 0.0:
            return -np.inf
        else:
            return (x - 1) * self.log_1p + self.log_p

    def expected_log_density(self, x: int) -> float:
        """Variational expectation E_q[log p(x | p)] under the Beta prior.

        Uses the cached digamma expectations of log p and log(1-p); falls
        back to the plug-in log_density(x) without a conjugate prior.

        Args:
            x (int): Observation (number of trials, x >= 1).

        Returns:
            Expected log-density at observation x.

        """
        if self.has_conj_prior:
            ga, gb, gab = self.conj_prior_params
            if x < 1:
                return -np.inf
            else:
                return (gb - gab) * (x - 1) + (ga - gab)
        else:
            return self.log_density(x)

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Vectorized log-density at sequence-encoded input x.

        Args:
            x (np.ndarray): Encoded observations from seq_encode().

        Returns:
            Numpy array of log-densities, one per observation.

        """
        rv = np.zeros_like(x, dtype=np.float64)
        invalid = x < 1
        if self.p == 1.0:
            rv.fill(-np.inf)
            rv[x == 1] = 0.0
        elif self.p == 0.0:
            rv.fill(-np.inf)
        else:
            rv = (x - 1.0) * self.log_1p + self.log_p
            rv[invalid] = -np.inf
        return rv

    def seq_expected_log_density(self, x: np.ndarray) -> np.ndarray:
        """Vectorized expected_log_density() at sequence-encoded input x.

        Args:
            x (np.ndarray): Encoded observations from seq_encode().

        Returns:
            Numpy array of expected log-densities, one per observation.

        """
        if self.has_conj_prior:
            ga, gb, gab = self.conj_prior_params
            rv = x - 1
            rv *= gb - gab
            rv += ga - gab
            rv[x < 1] = -np.inf
            return rv
        else:
            return self.seq_log_density(x)

    def seq_encode(self, x):
        """Encode an iterable of observations into a float numpy array.

        Args:
            x: Iterable of integer observations.

        Returns:
            Numpy array of floats for use with seq_ methods.

        """
        rv = np.asarray(x, dtype=float)
        return rv

    def sampler(self, seed=None):
        """Create a GeometricSampler for this distribution.

        Args:
            seed (Optional[int]): Seed for the random number generator.

        Returns:
            GeometricSampler object.

        """
        return GeometricSampler(self, seed)

    def estimator(self, pseudo_count=None):
        """Create a GeometricEstimator with this distribution's name, keys, and prior.

        Args:
            pseudo_count (Optional[float]): Unused; kept for interface compatibility.

        Returns:
            GeometricEstimator object.

        """
        return GeometricEstimator(name=self.name, keys=self.keys, prior=self.prior)


class GeometricSampler:
    """Draws samples from a GeometricDistribution."""

    def __init__(self, dist, seed=None):
        """GeometricSampler object.

        Args:
            dist (GeometricDistribution): Distribution to sample from.
            seed (Optional[int]): Seed for the random number generator.

        """
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size=None):
        """Draw size samples (a single int when size is None).

        Args:
            size (Optional[int]): Number of samples to draw.

        Returns:
            An int if size is None, else a numpy array of length size.

        """
        return self.rng.geometric(p=self.dist.p, size=size)


class GeometricAccumulator(StatisticAccumulator):
    """Accumulates geometric sufficient statistics (weighted observation
    count and weighted sum of trial counts)."""

    def __init__(self, keys, name):
        """GeometricAccumulator object.

        Args:
            keys (Optional[str]): Key for sharing sufficient statistics.
            name (Optional[str]): Name of the accumulator.

        """
        self.sum = 0.0
        self.count = 0.0
        self.key = keys
        self.name = name

    def update(self, x, weight, estimate):
        """Accumulate the weighted sufficient statistics of observation x.

        Args:
            x (int): Observation (non-negative values are accumulated).
            weight (float): Weight of the observation.
            estimate: Current distribution estimate (unused).

        """
        if x >= 0:
            self.sum += x * weight
            self.count += weight

    def seq_update(self, x, weights, estimate):
        """Vectorized update() on sequence-encoded data.

        Args:
            x (np.ndarray): Encoded observations.
            weights (np.ndarray): Weight per observation.
            estimate: Current distribution estimate (unused).

        """
        self.sum += np.dot(x, weights)
        self.count += np.sum(weights)

    def initialize(self, x, weight, rng):
        """Initialize the accumulator with observation x (delegates to update).

        Args:
            x (int): Observation.
            weight (float): Weight of the observation.
            rng: Random number generator (unused).

        """
        self.update(x, weight, None)

    def seq_initialize(self, x, weights, rng):
        """Vectorized initialize() on sequence-encoded data (delegates to seq_update).

        Args:
            x (np.ndarray): Encoded observations.
            weights (np.ndarray): Weight per observation.
            rng: Random number generator (unused).

        """
        self.seq_update(x, weights, None)

    def combine(self, suff_stat):
        """Add another accumulator's sufficient-statistic value into this one.

        Args:
            suff_stat: Tuple (count, sum) as returned by value().

        Returns:
            This accumulator.

        """
        self.sum += suff_stat[1]
        self.count += suff_stat[0]
        return self

    def value(self):
        """Returns the sufficient statistics (count, sum)."""
        return self.count, self.sum

    def from_value(self, x):
        """Set the sufficient statistics from a value() tuple.

        Args:
            x: Tuple (count, sum) as returned by value().

        Returns:
            This accumulator.

        """
        self.count = x[0]
        self.sum = x[1]
        return self

    def key_merge(self, stats_dict):
        """Merge this accumulator's keyed statistics into a shared dict.

        Args:
            stats_dict (dict): Shared key-to-statistics dictionary.

        """
        if self.key is not None:
            if self.key in stats_dict:
                vals = stats_dict[self.key]
                stats_dict[self.key] = (vals[0] + self.count, vals[1] + self.sum)
            else:
                stats_dict[self.key] = (self.count, self.sum)

    def key_replace(self, stats_dict):
        """Replace this accumulator's statistics with the pooled keyed values.

        Args:
            stats_dict (dict): Shared key-to-statistics dictionary.

        """
        if self.key is not None:
            if self.key in stats_dict:
                vals = stats_dict[self.key]
                self.count = vals[0]
                self.sum = vals[1]


class GeometricAccumulatorFactory:
    """Factory that creates GeometricAccumulator objects."""

    def __init__(self, keys, name):
        """GeometricAccumulatorFactory object.

        Args:
            keys (Optional[str]): Key passed to created accumulators.
            name (Optional[str]): Name passed to created accumulators.

        """
        self.keys = keys
        self.name = name

    def make(self):
        """Returns a new GeometricAccumulator."""
        return GeometricAccumulator(self.keys, self.name)


class GeometricEstimator(ParameterEstimator):
    """Estimates a GeometricDistribution from sufficient statistics, using a
    conjugate Beta posterior update when the prior allows it."""

    name: str | None
    has_conj_prior: bool
    has_prior: bool
    keys: str | None
    prior: ProbabilityDistribution

    def __init__(
        self, name: str | None = None, keys: str | None = None, prior: ProbabilityDistribution = default_prior
    ):
        """GeometricEstimator object.

        Args:
            name (Optional[str]): Name of the estimated distribution.
            keys (Optional[str]): Key for sharing sufficient statistics.
            prior (ProbabilityDistribution): Prior on p; a BetaDistribution
                enables the conjugate update.

        """
        self.keys = keys
        self.name = name
        self.set_prior(prior)

    def accumulator_factory(self):
        """Returns a GeometricAccumulatorFactory for this estimator."""
        return GeometricAccumulatorFactory(self.keys, self.name)

    def get_prior(self) -> ProbabilityDistribution:
        """Returns the prior distribution on p."""
        return self.prior

    def set_prior(self, prior) -> None:
        """Set the prior and flag whether it admits the conjugate update.

        Sets has_conj_prior when prior is a BetaDistribution and has_prior
        when it is any non-null distribution.

        Args:
            prior (ProbabilityDistribution): Prior on p.

        """
        self.prior = prior
        if isinstance(prior, BetaDistribution):
            self.has_conj_prior = True
            self.has_prior = True
        elif isinstance(prior, NullDistribution) or prior is None:
            self.has_conj_prior = False
            self.has_prior = False
        else:
            self.has_conj_prior = False
            self.has_prior = True

    def estimate(self, suff_stat: (float, float)) -> GeometricDistribution:
        """Estimate a GeometricDistribution from sufficient statistics.

        With a Beta(a, b) prior the posterior is
        Beta(a + count, b + sum - count) and the MAP estimate
        p = (a'-1)/(a'+b'-2) is returned, clamped to 0, 1, or 1/2 on the
        boundary where the mode is undefined; the posterior is carried as
        the new prior. Otherwise the maximum likelihood estimate count/sum
        is returned.

        Args:
            suff_stat: Tuple (count, sum) as returned by
                GeometricAccumulator.value().

        Returns:
            GeometricDistribution object.

        """
        ocnt, osum = suff_stat

        if self.has_conj_prior:
            old_a = self.prior.a
            old_b = self.prior.b

            a = old_a + ocnt
            b = old_b + osum - ocnt

            if a > 1 and b > 1:
                p = (a - 1) / (a + b - 2)
            elif a <= 1 and b > 1:
                p = 0.0
            elif a > 1 and b <= 1:
                p = 1.0
            else:
                p = 0.5

            return GeometricDistribution(p, name=self.name, prior=BetaDistribution(a, b), keys=self.keys)

        else:
            return GeometricDistribution(ocnt / osum, name=self.name, keys=self.keys)


if __name__ == "__main__":
    dist = GeometricDistribution(0.2)

    data = dist.sampler(seed=1).sample(100)
    enc_data = dist.seq_encode(data)

    est = GeometricEstimator()
    acc = est.accumulator_factory().make()
    acc.seq_update(enc_data, np.ones(len(data)), None)
    model = est.estimate(acc.value())

    print(str(model))
