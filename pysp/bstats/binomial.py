"""Binomial distribution with conjugate Beta prior on the success probability.

The number of trials n is treated as known. With prior p ~ Beta(a, b) and
observations x_1..x_m, the posterior is Beta(a + sum x_i, b + sum (n - x_i)).
Estimation returns the posterior mode (MAP) for p, carrying the posterior as
the new prior, matching the conventions of the other pysp.bstats modules.
"""
from typing import Optional, Tuple

from numpy.random import RandomState
from scipy.special import gammaln, digamma
import numpy as np

from pysp.bstats.pdist import ProbabilityDistribution, StatisticAccumulator, ParameterEstimator
from pysp.bstats.beta import BetaDistribution

default_prior = BetaDistribution(1.0001, 1.0001)


class BinomialDistribution(ProbabilityDistribution):
    """Binomial distribution with n trials and success probability p,
    optionally carrying a Beta conjugate prior on p."""

    def __init__(self, n: int, p: float, name: Optional[str] = None,
                 prior: ProbabilityDistribution = default_prior, keys: Optional[str] = None):
        """BinomialDistribution object with n trials and success probability p.

        Args:
            n (int): Number of trials, n >= 1.
            p (float): Success probability in [0, 1].
            name (Optional[str]): Name of object.
            prior (ProbabilityDistribution): Prior on p; a BetaDistribution
                enables the conjugate machinery (see set_prior()).
            keys (Optional[str]): Key for sharing sufficient statistics.

        """
        assert 0.0 <= p <= 1.0 and n >= 1
        self.name = name
        self.keys = keys
        self.n = int(n)
        self.set_parameters(p)
        self.set_prior(prior)

    def __str__(self):
        return 'BinomialDistribution(%d, %f, name=%s, prior=%s, keys=%s)' % (
            self.n, self.p, str(self.name), str(self.prior), str(self.keys))

    def get_parameters(self) -> float:
        """Returns the success probability p."""
        return self.p

    def set_parameters(self, params: float) -> None:
        """Set the success probability and refresh the cached log terms.

        Args:
            params (float): Success probability in [0, 1].

        """
        self.p = params
        self.log_p = np.log(params) if params > 0 else -np.inf
        self.log_1p = np.log1p(-params) if params < 1 else -np.inf

    def get_prior(self) -> ProbabilityDistribution:
        """Returns the prior distribution on p."""
        return self.prior

    def set_prior(self, prior: ProbabilityDistribution) -> None:
        """Set the prior and precompute conjugate-prior expectations.

        If prior is a BetaDistribution(a, b), this caches
        (E[log p], E[log(1-p)]) = (digamma(a) - digamma(a+b),
        digamma(b) - digamma(a+b)), the terms used by
        expected_log_density. Sets has_conj_prior accordingly.

        Args:
            prior (ProbabilityDistribution): Prior on p.

        """
        self.prior = prior

        if isinstance(prior, BetaDistribution):
            a, b = prior.get_parameters()
            self.conj_prior_params = (a, b)
            # E[log p] and E[log(1-p)] under the Beta prior
            self.expected_nparams = (digamma(a) - digamma(a + b), digamma(b) - digamma(a + b))
            self.has_conj_prior = True
        else:
            self.conj_prior_params = None
            self.expected_nparams = None
            self.has_conj_prior = False

    def density(self, x: int) -> float:
        """Density of the binomial distribution at observation x.

        Args:
            x (int): Number of successes, 0 <= x <= n.

        Returns:
            Density at observation x.

        """
        return np.exp(self.log_density(x))

    def log_density(self, x: int) -> float:
        """Log-density of the binomial distribution at observation x.

        Args:
            x (int): Number of successes; -inf outside [0, n].

        Returns:
            Log-density at observation x.

        """
        n = self.n
        if x < 0 or x > n:
            return -np.inf
        cc = gammaln(n + 1) - gammaln(x + 1) - gammaln(n - x + 1)
        return cc + x*self.log_p + (n - x)*self.log_1p

    def expected_log_density(self, x: int) -> float:
        """Variational expectation E_q[log p(x | p)] under the Beta prior.

        Uses the cached digamma expectations of log p and log(1-p); falls
        back to the plug-in log_density(x) without a conjugate prior.

        Args:
            x (int): Number of successes; -inf outside [0, n].

        Returns:
            Expected log-density at observation x.

        """
        if self.has_conj_prior:
            n = self.n
            if x < 0 or x > n:
                return -np.inf
            e1, e2 = self.expected_nparams
            cc = gammaln(n + 1) - gammaln(x + 1) - gammaln(n - x + 1)
            return cc + x*e1 + (n - x)*e2
        else:
            return self.log_density(x)

    def entropy(self) -> float:
        """Returns the entropy of the binomial distribution (in nats)."""
        x = np.arange(self.n + 1)
        ll = self.seq_log_density(self.seq_encode(x))
        return -np.dot(np.exp(ll), ll)

    def cross_entropy(self, dist: ProbabilityDistribution) -> float:
        """Cross-entropy H(self, dist) = -E_self[log dist] over {0..n}.

        Args:
            dist (ProbabilityDistribution): Distribution to evaluate against.

        Returns:
            Cross-entropy in nats.

        """
        x = np.arange(self.n + 1)
        pp = np.exp(self.seq_log_density(self.seq_encode(x)))
        return -np.dot(pp, np.asarray([dist.log_density(u) for u in x]))

    def seq_log_density(self, x):
        """Vectorized log-density at sequence-encoded input x.

        Args:
            x: Encoded tuple (values, log binomial coefficients) from
                seq_encode().

        Returns:
            Numpy array of log-densities, one per observation.

        """
        xv, cc = x
        rv = xv*self.log_p + (self.n - xv)*self.log_1p + cc
        rv[np.bitwise_or(xv < 0, xv > self.n)] = -np.inf
        return rv

    def seq_expected_log_density(self, x):
        """Vectorized expected_log_density() at sequence-encoded input x.

        Args:
            x: Encoded tuple (values, log binomial coefficients) from
                seq_encode().

        Returns:
            Numpy array of expected log-densities, one per observation.

        """
        if self.has_conj_prior:
            xv, cc = x
            e1, e2 = self.expected_nparams
            rv = xv*e1 + (self.n - xv)*e2 + cc
            rv[np.bitwise_or(xv < 0, xv > self.n)] = -np.inf
            return rv
        else:
            return self.seq_log_density(x)

    def seq_encode(self, x):
        """Encode observations into (values, log binomial coefficients).

        Args:
            x: Iterable of integer success counts.

        Returns:
            Tuple (values array, log n-choose-x array) for use with seq_ methods.

        """
        xv = np.asarray(x, dtype=float)
        cc = gammaln(self.n + 1) - gammaln(xv + 1) - gammaln(self.n - xv + 1)
        return xv, cc

    def sampler(self, seed: Optional[int] = None):
        """Create a BinomialSampler for this distribution.

        Args:
            seed (Optional[int]): Seed for the random number generator.

        Returns:
            BinomialSampler object.

        """
        return BinomialSampler(self, seed)

    def estimator(self):
        """Create a BinomialEstimator with this distribution's n, name, keys, and prior.

        Returns:
            BinomialEstimator object.

        """
        return BinomialEstimator(self.n, name=self.name, keys=self.keys, prior=self.prior)


class BinomialSampler(object):
    """Draws samples from a BinomialDistribution."""

    def __init__(self, dist: BinomialDistribution, seed: Optional[int] = None):
        """BinomialSampler object.

        Args:
            dist (BinomialDistribution): Distribution to sample from.
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
        return self.rng.binomial(self.dist.n, self.dist.p, size=size)


class BinomialAccumulator(StatisticAccumulator):
    """Accumulates binomial sufficient statistics (weighted observation
    count and weighted sum of success counts)."""

    def __init__(self, n: int, name=None, keys=None):
        """BinomialAccumulator object.

        Args:
            n (int): Number of trials.
            name (Optional[str]): Name of the accumulator.
            keys (Optional[str]): Key for sharing sufficient statistics.

        """
        self.n = n
        self.name = name
        self.key = keys
        self.sum = 0.0
        self.count = 0.0

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
            x: Encoded observations.
            weights (np.ndarray): Weight per observation.
            rng: Random number generator (unused).

        """
        self.seq_update(x, weights, None)

    def update(self, x, weight, estimate):
        """Accumulate the weighted sufficient statistics of observation x.

        Args:
            x (int): Observation (number of successes).
            weight (float): Weight of the observation.
            estimate: Current distribution estimate (unused).

        """
        self.sum += x*weight
        self.count += weight

    def seq_update(self, x, weights, estimate):
        """Vectorized update() on sequence-encoded data.

        Args:
            x: Encoded tuple (values, log binomial coefficients).
            weights (np.ndarray): Weight per observation.
            estimate: Current distribution estimate (unused).

        """
        self.sum += np.dot(x[0], weights)
        self.count += weights.sum()

    def combine(self, suff_stat):
        """Add another accumulator's sufficient-statistic value into this one.

        Args:
            suff_stat: Tuple (count, sum) as returned by value().

        Returns:
            This accumulator.

        """
        self.count += suff_stat[0]
        self.sum += suff_stat[1]
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
                stats_dict[self.key].combine(self.value())
            else:
                stats_dict[self.key] = self

    def key_replace(self, stats_dict):
        """Replace this accumulator's statistics with the pooled keyed values.

        Args:
            stats_dict (dict): Shared key-to-statistics dictionary.

        """
        if self.key is not None:
            if self.key in stats_dict:
                self.from_value(stats_dict[self.key].value())


class BinomialEstimatorAccumulatorFactory(object):
    """Factory that creates BinomialAccumulator objects."""

    def __init__(self, n, name, keys):
        """BinomialEstimatorAccumulatorFactory object.

        Args:
            n (int): Number of trials passed to created accumulators.
            name (Optional[str]): Name passed to created accumulators.
            keys (Optional[str]): Key passed to created accumulators.

        """
        self.n = n
        self.name = name
        self.keys = keys

    def make(self):
        """Returns a new BinomialAccumulator."""
        return BinomialAccumulator(self.n, name=self.name, keys=self.keys)


class BinomialEstimator(ParameterEstimator):
    """Estimates a BinomialDistribution from sufficient statistics, using a
    conjugate Beta posterior update when the prior allows it."""

    def __init__(self, n: int, name: Optional[str] = None, keys: Optional[str] = None,
                 prior: ProbabilityDistribution = default_prior):
        """BinomialEstimator object.

        Args:
            n (int): Number of trials (treated as known).
            name (Optional[str]): Name of the estimated distribution.
            keys (Optional[str]): Key for sharing sufficient statistics.
            prior (ProbabilityDistribution): Prior on p; a BetaDistribution
                enables the conjugate update.

        """
        self.n = int(n)
        self.name = name
        self.keys = keys
        self.set_prior(prior)

    def accumulator_factory(self):
        """Returns a BinomialEstimatorAccumulatorFactory for this estimator."""
        return BinomialEstimatorAccumulatorFactory(self.n, self.name, self.keys)

    def get_prior(self):
        """Returns the prior distribution on p."""
        return self.prior

    def set_prior(self, prior):
        """Set the prior and flag whether it admits the conjugate update.

        Args:
            prior (ProbabilityDistribution): Prior on p.

        """
        self.prior = prior
        self.has_conj_prior = isinstance(prior, BetaDistribution)

    def model_log_density(self, model):
        """Log-density of the model's success probability under the prior.

        Args:
            model (BinomialDistribution): Model to score.

        Returns:
            Prior log-density of the model parameters.

        """
        if self.has_conj_prior:
            return float(self.prior.log_density(model.p))
        return super().model_log_density(model)

    def estimate(self, suff_stat: Tuple[float, float]) -> BinomialDistribution:
        """Estimate a BinomialDistribution from sufficient statistics.

        With a Beta(a, b) prior the posterior is Beta(a + successes,
        b + failures) and the MAP estimate p = (a'-1)/(a'+b'-2) is returned
        when a', b' > 1, falling back to the posterior mean a'/(a'+b') on
        the boundary; the posterior is carried as the new prior. Otherwise
        the maximum likelihood estimate is returned (p = 1/2 when there is
        no data).

        Args:
            suff_stat (Tuple[float, float]): Tuple (count, sum) as returned
                by BinomialAccumulator.value().

        Returns:
            BinomialDistribution object.

        """
        count, psum = suff_stat
        fsum = count*self.n - psum

        if self.has_conj_prior:

            a, b = self.prior.get_parameters()
            new_a = a + psum
            new_b = b + fsum

            # posterior mode for new_a, new_b > 1; mean on the boundary
            if new_a > 1.0 and new_b > 1.0:
                p = (new_a - 1.0)/(new_a + new_b - 2.0)
            else:
                p = new_a/(new_a + new_b)

            return BinomialDistribution(self.n, p, name=self.name, keys=self.keys,
                                        prior=BetaDistribution(new_a, new_b))

        else:
            p = psum/(count*self.n) if count > 0 else 0.5
            return BinomialDistribution(self.n, p, name=self.name, keys=self.keys, prior=self.prior)

# --- API naming aliases (notes/distribution_api_naming_accounting.md) ---
BinomialAccumulatorFactory = BinomialEstimatorAccumulatorFactory
