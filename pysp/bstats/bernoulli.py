"""Bayesian Bernoulli distribution with success probability p.

Defines the BernoulliDistribution, BernoulliSampler,
BernoulliEstimatorAccumulator, BernoulliEstimatorAccumulatorFactory, and
BernoulliEstimator classes for use with pysp.bstats.

Data type: (bool): The BernoulliDistribution with success probability
    p in [0, 1] has log-density
    log(f(x; p)) = log(p) if x is true, log(1-p) otherwise.

Conjugate prior: BetaDistribution on p. With prior Beta(a, b) and weighted
counts (psum, nsum) of true/false observations, the posterior is
Beta(a + psum, b + nsum). Estimation returns the posterior mode (MAP),
carrying the posterior as the new prior; with a non-conjugate prior the
penalized likelihood is maximized numerically, and with no prior the maximum
likelihood estimate psum/(psum + nsum) is returned. expected_log_density
evaluates the variational Bayes expectation E_q[log p(x | p)] via digamma
terms.
"""
from typing import Optional, Any, Dict, Union
from numpy.random import RandomState
from pysp.bstats.pdist import ParameterEstimator, ProbabilityDistribution, StatisticAccumulator
from pysp.bstats.beta import BetaDistribution
from pysp.bstats.nulldist import NullDistribution, null_dist
import numpy as np
from scipy.special import gammaln, digamma, exp1
from scipy.optimize import minimize_scalar
import scipy.integrate


default_prior = BetaDistribution(1.000001, 1.000001)


class BernoulliDistribution(ProbabilityDistribution):
    """Bernoulli distribution with success probability p, optionally carrying
    a Beta conjugate prior on p."""

    def __init__(self, p: float, name: Optional[str] = None, prior: ProbabilityDistribution = default_prior, keys: Optional[str] = None):
        """BernoulliDistribution object with success probability p.

        Args:
            p (float): Success probability in [0, 1].
            name (Optional[str]): Name of object.
            prior (ProbabilityDistribution): Prior on p; a BetaDistribution
                enables the conjugate machinery (see set_prior()).
            keys (Optional[str]): Key for sharing sufficient statistics.
        """

        self.p = p
        self.log_p0 = np.log(p)
        self.log_p1 = np.log1p(-p)

        self.name = name
        self.keys = keys
        self.set_parameters(p)
        self.set_prior(prior)

    def __str__(self) -> str:
        return 'BernoulliDistribution(%s, name=%s, prior=%s, keys=%s)' % (repr(self.p), repr(self.name), str(self.prior), repr(self.keys))

    def get_parameters(self) -> float:
        """Returns the success probability p."""
        return self.p

    def set_parameters(self, params: float) -> None:
        """Sets the success probability and refreshes the cached log terms.

        Args:
            params (float): New success probability in [0, 1].
        """
        self.p = params
        self.log_p0 = np.log(params)
        self.log_p1 = np.log1p(-params)

    def get_prior(self) -> ProbabilityDistribution:
        """Returns the prior distribution on p."""
        return self.prior

    def set_prior(self, prior: ProbabilityDistribution):
        """Set the prior and precompute conjugate-prior expectations.

        If prior is a BetaDistribution(a, b), this caches
        (digamma(a), digamma(b), digamma(a+b)), the terms needed for
        E[log p] = digamma(a) - digamma(a+b) and
        E[log(1-p)] = digamma(b) - digamma(a+b) in expected_log_density.
        Sets has_conj_prior and has_prior accordingly.

        Args:
            prior (ProbabilityDistribution): Prior on p.
        """
        self.prior = prior

        if isinstance(prior, BetaDistribution):
            a, b = self.prior.get_parameters()
            self.conj_prior_params = (digamma(a), digamma(b), digamma(a+b))
            self.has_conj_prior = True
            self.has_prior = True
        elif isinstance(prior, NullDistribution) or prior is None:
            self.conj_prior_params = None
            self.has_conj_prior = False
            self.has_prior = False
        else:
            self.conj_prior_params = None
            self.has_conj_prior = False
            self.has_prior = True

    def get_data_type(self):
        """Returns the observation data type (bool)."""
        return bool

    def density(self, x: bool) -> float:
        """Density at observation x (exp of log_density).

        Args:
            x (bool): Observation.

        Returns:
            Density at observation x.
        """
        return np.exp(self.log_density(x))

    def log_density(self, x: bool) -> float:
        """Log-density at observation x.

        Args:
            x (bool): Observation.

        Returns:
            log(p) if x is true, log(1-p) otherwise.
        """
        if x:
            return self.log_p0
        else:
            return self.log_p1

    def expected_log_density(self, x: bool) -> float:
        """Prior-expected log-density E_q[log p(x|p)] at observation x.

        Falls back to log_density when no conjugate prior is set.

        Args:
            x (bool): Observation.

        Returns:
            Expected log-density (float) at x.
        """
        if self.has_conj_prior:
            da, db, dab = self.conj_prior_params
            if x:
                return da - dab
            else:
                return db - dab
        else:
            return self.log_density(x)

    def cross_entropy(self, dist: ProbabilityDistribution) -> float:
        """Cross entropy -E_self[log dist(x)] over {True, False}.

        Args:
            dist (ProbabilityDistribution): Distribution evaluated under this
                one.

        Returns:
            float: Cross entropy value in nats.
        """
        a = dist.log_density(True)
        b = dist.log_density(False)
        return (a-b)*self.p + b

    def entropy(self) -> float:
        """Returns the entropy -p log(p) - (1-p) log(1-p) in nats."""
        return self.p * (self.log_p0 - self.log_p1) + self.log_p1

    def moment(self, p: int) -> float:
        # X takes values in {0, 1}, so E[X^p] = P(X=1) for any p >= 1
        return 1.0 if p == 0 else self.p

    def seq_log_density(self, x):
        """Vectorized log-density at sequence-encoded input x.

        Args:
            x: Encoded data from seq_encode().

        Returns:
            Numpy array of log-densities, one entry per observation.
        """
        return np.where(x, self.log_p0, self.log_p1)

    def seq_expected_log_density(self, x):
        """Vectorized expected log-density at sequence-encoded input x
        (requires a conjugate prior).

        Args:
            x: Encoded data from seq_encode().

        Returns:
            Numpy array of expected log-densities, one entry per observation.
        """
        da, db, dab = self.conj_prior_params
        return np.where(x, da - dab, db - dab)

    def seq_encode(self, x):
        """Encode a sequence of observations for vectorized evaluation.

        Args:
            x: Iterable of boolean observations.

        Returns:
            Numpy boolean array of the observations.
        """
        return np.asarray(x, dtype=bool)

    def sampler(self, seed: Optional[int] = None):
        """Return a BernoulliSampler for this distribution.

        Args:
            seed (Optional[int]): Seed for the random number generator.
        """
        return BernoulliSampler(self, seed)

    def estimator(self):
        """Return a BernoulliEstimator matching this distribution."""
        return BernoulliEstimator(name=self.name, keys=self.keys, prior=self.prior)


class BernoulliSampler(object):
    """Draws boolean observations from a BernoulliDistribution."""

    def __init__(self, dist, seed=None):
        """BernoulliSampler object.

        Args:
            dist (BernoulliDistribution): Distribution to sample from.
            seed (Optional[int]): Seed for the random number generator.
        """
        self.rng  = np.random.RandomState(seed)
        self.dist = dist

    def sample(self, size=None):
        """Draw size samples (or one sample when size is None).

        Args:
            size (Optional[int]): Number of samples to draw.

        Returns:
            A single bool when size is None, otherwise a list of bools.
        """
        if size is None:
            return self.rng.rand() < self.dist.p
        else:
            return (self.rng.rand(size) < self.dist.p).tolist()


class BernoulliEstimatorAccumulator(StatisticAccumulator):
    """Accumulates the weighted counts of true (psum) and false (nsum)
    observations for Bernoulli estimation."""

    def __init__(self, name, keys):
        """BernoulliEstimatorAccumulator object.

        Args:
            name (Optional[str]): Name of the accumulated statistics.
            keys (Optional[str]): Key for sharing sufficient statistics.
        """
        self.name  = name
        self.key   = keys
        self.psum  = 0.0
        self.nsum  = 0.0
        self.count = 0.0

    def initialize(self, x, weight, rng):
        """Initialize with one weighted observation (delegates to update)."""
        self.update(x, weight, None)

    def seq_initialize(self, x, weights, rng):
        """Vectorized initialization (delegates to seq_update).

        Args:
            x: Encoded data from BernoulliDistribution.seq_encode().
            weights (np.ndarray): Observation weights.
            rng: Unused (kept for protocol consistency).
        """
        self.seq_update(x, weights, None)

    def update(self, x, weight, estimate):
        """Accumulate one weighted observation.

        Args:
            x (bool): Observation.
            weight (float): Observation weight.
            estimate: Unused (kept for protocol consistency).
        """
        if x:
            self.psum += weight
        else:
            self.nsum += weight

    def seq_update(self, x, weights, estimate):
        """Vectorized update from sequence-encoded data.

        Args:
            x: Encoded data from BernoulliDistribution.seq_encode().
            weights (np.ndarray): Observation weights.
            estimate: Unused (kept for protocol consistency).
        """
        n = weights.sum()
        p = weights[x].sum()
        self.psum += p
        self.nsum += n - p

    def combine(self, suff_stat):
        """Merge another accumulator's value() into this one.

        Args:
            suff_stat: Tuple (psum, nsum).

        Returns:
            This accumulator.
        """
        self.psum += suff_stat[0]
        self.nsum += suff_stat[1]
        return self

    def value(self):
        """Return (psum, nsum)."""
        return self.psum, self.nsum

    def from_value(self, x):
        """Set this accumulator's state from a value() tuple.

        Args:
            x: Tuple (psum, nsum).
        """
        self.psum = x[0]
        self.nsum = x[1]

    def key_merge(self, stats_dict: Dict[str, Any]):
        """Merge this accumulator into stats_dict under its key (if keyed)."""
        if self.key is not None:
            if self.key in stats_dict:
                stats_dict[self.key].combine(self.value())
            else:
                stats_dict[self.key] = self

    def key_replace(self, stats_dict: Dict[str, Any]):
        """Replace this accumulator's statistics from stats_dict (if keyed)."""
        if self.key is not None:
            if self.key in stats_dict:
                self.from_value(stats_dict[self.key].value())


class BernoulliEstimatorAccumulatorFactory(object):
    """Factory for creating BernoulliEstimatorAccumulator objects."""

    def __init__(self, name, keys):
        """BernoulliEstimatorAccumulatorFactory object.

        Args:
            name (Optional[str]): Name passed to the accumulators.
            keys (Optional[str]): Key passed to the accumulators.
        """
        self.name = name
        self.keys = keys

    def make(self):
        """Returns a new BernoulliEstimatorAccumulator."""
        return BernoulliEstimatorAccumulator(self.name, self.keys)


class BernoulliEstimator(ParameterEstimator):
    """Estimates a BernoulliDistribution from accumulated true/false counts,
    using the Beta posterior mode when a conjugate prior is set."""

    def __init__(self, name: Optional[str] = None, keys: Optional[str] = None, prior: ProbabilityDistribution = default_prior):
        """BernoulliEstimator object.

        Args:
            name (Optional[str]): Name of the estimated distribution.
            keys (Optional[str]): Key for sharing sufficient statistics.
            prior (ProbabilityDistribution): Prior on p; a BetaDistribution
                enables conjugate MAP estimation.
        """

        self.prior = prior
        self.name  = name
        self.keys  = keys
        self.has_conj_prior = isinstance(prior, BetaDistribution)
        self.has_prior = not isinstance(prior, NullDistribution) and prior is not None

    def accumulator_factory(self) -> BernoulliEstimatorAccumulatorFactory:
        """Returns a BernoulliEstimatorAccumulatorFactory for this estimator."""
        return BernoulliEstimatorAccumulatorFactory(self.name, self.keys)

    def accumulatorFactory(self) -> BernoulliEstimatorAccumulatorFactory:
        """Deprecated alias for accumulator_factory()."""
        return self.accumulator_factory()

    def set_prior(self, prior) -> None:
        """Set the prior on p.

        Args:
            prior (ProbabilityDistribution): New prior distribution; a
                BetaDistribution enables conjugate MAP estimation.
        """
        self.prior = prior
        self.has_conj_prior = isinstance(prior, BetaDistribution)
        self.has_prior = not isinstance(prior, NullDistribution) and prior is not None

    def get_prior(self) -> ProbabilityDistribution:
        """Returns the prior distribution on p."""
        return self.prior

    def estimate(self, suff_stat: (float, float)) -> BernoulliDistribution:
        """Estimate a BernoulliDistribution from sufficient statistics.

        With a Beta(a, b) prior this returns the posterior mode
        (psum + a - 1)/(psum + nsum + a + b - 2) carrying the posterior
        Beta(a + psum, b + nsum) as the new prior. With a non-conjugate prior
        the penalized log-likelihood is maximized numerically, and with no
        prior the maximum-likelihood estimate psum/(psum + nsum) is returned.

        Args:
            suff_stat: Tuple (psum, nsum) as returned by
                BernoulliEstimatorAccumulator.value().

        Returns:
            BernoulliDistribution estimate.
        """

        psum, nsum = suff_stat

        if self.has_conj_prior:
            a, b = self.prior.get_parameters()
            new_a = a + psum
            new_b = b + nsum
            p = (psum + a - 1.0)/(psum + nsum + a + b - 2.0)
            return BernoulliDistribution(p, name=self.name, prior=BetaDistribution(new_a, new_b), keys=self.keys)

        elif self.has_prior:

            ll_fun = lambda x: np.log(x)*psum + np.log1p(-x)*nsum + self.prior.log_density(x)
            eps = np.sqrt(np.finfo(float).eps)
            sol = minimize_scalar(lambda x: -ll_fun(x), bounds=(eps, 1.0 - eps), method='bounded')
            return BernoulliDistribution(float(sol.x), name=self.name, prior=self.prior, keys=self.keys)

        else:
            return BernoulliDistribution(psum/(psum + nsum), name=self.name, prior=null_dist, keys=self.keys)
