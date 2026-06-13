"""Poisson distribution over non-negative integer counts with a conjugate
Gamma prior on the rate.

Data type: (int). The PoissonDistribution with rate lam > 0 has log-density

    log f(x; lam) = x*log(lam) - lam - log(x!), for x = 0, 1, 2, ...

Defines the PoissonDistribution, PoissonSampler, PoissonEstimatorAccumulator,
PoissonEstimatorAccumulatorFactory, and PoissonEstimator classes for use with
pysparkplug. With a GammaDistribution prior, estimation is MAP (the Gamma
posterior mode, falling back to the posterior mean at the boundary) and the
posterior Gamma is carried forward as the new prior. expected_log_density
uses the Gamma expectations E[lam] = k*theta and E[ln lam] = psi(k) +
ln(theta).
"""
#import copyreg, copy, pickle, dill
from typing import Optional, Any, Dict
from pysp.arithmetic import *
from numpy.random import RandomState
from pysp.bstats.pdist import ParameterEstimator, ProbabilityDistribution, StatisticAccumulator
from pysp.bstats.gamma import GammaDistribution
from pysp.bstats.nulldist import NullDistribution, null_dist
from pysp.utils.special import stirling2
import numpy as np
from scipy.special import gammaln, digamma, exp1
from scipy.optimize import minimize_scalar
import scipy.integrate


default_prior = GammaDistribution(1.0001, 1.0e6)


class PoissonDistribution(ProbabilityDistribution):
    """
    A Poisson distributed random variable has the likelihood function

    l(x | lambda) = (lambda ** x) * exp(-lambda) / x!

    where x is a non-negative integer and lambda is a positive real number.
    """

    lam: float
    log_lambda: float
    conj_prior_params: (float, float)
    prior: ProbabilityDistribution
    has_conj_prior: bool
    has_prior: bool

    def __init__(self, lam: float, name: Optional[str] = None, prior: ProbabilityDistribution = default_prior, keys: Optional[str] = None):
        """Create a Poisson distribution.

        Args:
            lam (float): Positive rate parameter.
            name (Optional[str]): Name of the distribution.
            prior (ProbabilityDistribution): Prior on the rate
                (GammaDistribution for conjugacy).
            keys (Optional[str]): Key for sharing sufficient statistics.
        """

        self.name = name
        self.keys = keys
        self.set_parameters(lam)
        self.set_prior(prior)

    def __str__(self) -> str:
        return 'PoissonDistribution(%f, name=%s, prior=%s, keys=%s)' % (self.lam, str(self.name), str(self.prior), str(self.keys))

    def get_parameters(self) -> float:
        """Return the rate parameter lam."""
        return self.lam

    def set_parameters(self, params: float) -> None:
        """Set the rate parameter.

        Args:
            params (float): Positive rate parameter.
        """
        self.lam = params
        self.log_lambda = np.log(self.lam)

    def get_prior(self) -> ProbabilityDistribution:
        """Return the prior on the rate."""
        return self.prior

    def set_prior(self, prior: ProbabilityDistribution):
        """Set the prior on the rate and cache the conjugate Gamma
        parameters when applicable.

        Args:
            prior (ProbabilityDistribution): New prior distribution.
        """
        self.prior = prior

        if isinstance(prior, GammaDistribution):
            k, theta = self.prior.get_parameters()
            self.conj_prior_params = (k, theta)
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
        """Return the data type accepted by this distribution (int)."""
        return int

    def density(self, x: int) -> float:
        """Density of the Poisson distribution at observation x.

        Args:
            x (int): Non-negative integer observation.

        Returns:
            Density (float) at x.
        """
        return np.exp(self.log_density(x))

    def log_density(self, x: int) -> float:
        """Log-density x*log(lam) - lam - log(x!) at observation x.

        Args:
            x (int): Non-negative integer observation.

        Returns:
            Log-density (float) at x.
        """
        if x < 0:
            return -np.inf
        return x*self.log_lambda - float(gammaln(x + 1.0)) - self.lam

    def expected_log_density(self, x: float) -> float:
        """Prior-expected log-density at observation x.

        Uses E[ln lam] and E[lam] under a conjugate Gamma prior, falling
        back to log_density when no conjugate prior is set.

        Args:
            x (float): Non-negative integer observation.

        Returns:
            Expected log-density (float) at x.
        """

        if self.has_conj_prior:

            k, theta = self.conj_prior_params
            e1 = (digamma(k)+np.log(theta))*x
            e2 = k*theta
            e3 = gammaln(x+1)
            return e1 - e2 - e3

        else:
            return self.log_density(x)

    def cross_entropy(self, dist: ProbabilityDistribution) -> float:
        """Cross entropy H(self, dist) for another PoissonDistribution.

        Args:
            dist (PoissonDistribution): Distribution to evaluate against.

        Returns:
            H(self) + KL(self || dist).
        """
        if isinstance(dist, PoissonDistribution):
            lam1 = self.lam
            lam2 = dist.lam
            rv = self.entropy()
            rv += (np.log(lam1)-np.log(lam2))*lam1 + lam2 - lam1
            return rv
        else:
            raise NotImplementedError('PoissonDistribution.cross_entropy is only implemented for PoissonDistribution arguments (got %s).' % type(dist).__name__)

    def entropy(self) -> float:
        """Return the entropy of the Poisson distribution (numerically
        integrated for small rates, asymptotic expansion for large)."""

        if self.lam > 450:
            l = self.lam
            rv = 0.5*np.log(2.0*np.pi*l) + 0.5 - 1/(12.0*l) - 1.0/(24.0*l*l) - 19.0/(360.0*l*l*l)
        else:
            lam = self.lam
            rv0 = 0.5 * np.log(2.0 * np.pi * lam) + 0.5 + (lam + 0.5) * exp1(lam) - np.exp(-lam)
            rterm = lambda x: (np.exp(-lam * x) / x) * ((1 / x) - 0.5 + (1 / np.log1p(-x)))
            rv1 = scipy.integrate.quad(rterm, 0, 1)[0]
            rv = rv0 - rv1
        return rv

    def moment(self, p: int) -> float:
        """Return the p-th moment of the Poisson distribution.

        Args:
            p (int): Moment order.

        Returns:
            E[X^p] computed via Stirling numbers of the second kind.
        """
        if p == 0:
            return 1.0
        elif p == 1:
            return self.lam
        else:
            rv = 0
            for i in range(p + 1):
                rv += np.power(self.lam, i) * stirling2(p, i)
            return rv

    def seq_log_density(self, x):
        """Vectorized log-density at sequence-encoded input x.

        Args:
            x: Encoded data from seq_encode().

        Returns:
            Numpy array of log-densities, one entry per observation.
        """
        invalid = x[0] < 0
        rv = x[0]*self.log_lambda
        rv -= x[1]
        rv -= self.lam
        rv[invalid] = -np.inf
        return rv

    def seq_expected_log_density(self, x):
        """Vectorized expected log-density at sequence-encoded input x.

        Falls back to seq_log_density when no conjugate prior is set.

        Args:
            x: Encoded data from seq_encode().

        Returns:
            Numpy array of expected log-densities, one entry per observation.
        """
        if not self.has_conj_prior:
            return self.seq_log_density(x)

        k, theta = self.conj_prior_params
        e1 = (digamma(k) + np.log(theta))*x[0]
        e2 = k*theta
        e3 = x[1]
        rv = e1 - e2 - e3
        rv[x[0] < 0] = -np.inf
        return rv

    def seq_encode(self, x):
        """Encode a sequence of observations for vectorized evaluation.

        Args:
            x: Iterable of non-negative integers.

        Returns:
            Tuple (observation array, gammaln(x+1) array).
        """
        rv1 = np.asarray(x, dtype=float)
        rv2 = gammaln(rv1 + 1.0)
        return rv1, rv2

    def sampler(self, seed: Optional[int] = None):
        """Return a PoissonSampler for this distribution.

        Args:
            seed (Optional[int]): Seed for the random number generator.
        """
        return PoissonSampler(self, seed)

    def estimator(self):
        """Return a PoissonEstimator matching this distribution."""
        return PoissonEstimator(name=self.name, keys=self.keys, prior=self.prior)



class PoissonSampler(object):
    """Draws observations from a PoissonDistribution."""

    def __init__(self, dist, seed=None):
        """Create a sampler for a PoissonDistribution.

        Args:
            dist (PoissonDistribution): Distribution to sample from.
            seed (Optional[int]): Seed for the random number generator.
        """
        self.rng  = RandomState(seed)
        self.dist = dist

    def sample(self, size=None):
        """Draw size samples (or one sample when size is None).

        Args:
            size (Optional[int]): Number of samples to draw.

        Returns:
            An int when size is None, otherwise a numpy array of ints.
        """
        return self.rng.poisson(lam=self.dist.lam, size=size)


class PoissonEstimatorAccumulator(StatisticAccumulator):
    """Accumulates the weighted count and sum of observations for Poisson
    estimation. The sufficient statistic is the tuple (count, sum)."""

    def __init__(self, name, keys):
        """Create a Poisson accumulator.

        Args:
            name: Name of the corresponding estimator.
            keys: Key for sharing statistics across accumulators.
        """
        self.name  = name
        self.key   = keys
        self.sum   = 0.0
        self.count = 0.0

    def initialize(self, x, weight, rng):
        """Initialize with one weighted observation (delegates to update)."""
        self.update(x, weight, None)

    def seq_initialize(self, x, weights, rng):
        """Vectorized initialization (delegates to seq_update)."""
        self.seq_update(x, weights, None)

    def update(self, x, weight, estimate):
        """Accumulate one weighted observation.

        Args:
            x (int): Observation.
            weight (float): Observation weight.
            estimate: Unused (kept for protocol consistency).
        """
        self.sum  += x*weight
        self.count += weight

    def seq_update(self, x, weights, estimate):
        """Vectorized update from sequence-encoded data.

        Args:
            x: Encoded data from PoissonDistribution.seq_encode().
            weights (np.ndarray): Observation weights.
            estimate: Unused (kept for protocol consistency).
        """
        self.sum   += np.dot(x[0], weights)
        self.count += weights.sum()

    def combine(self, suff_stat):
        """Merge another accumulator's value() into this one.

        Args:
            suff_stat: Tuple (count, sum).

        Returns:
            This accumulator.
        """
        self.sum  += suff_stat[1]
        self.count += suff_stat[0]
        return self

    def value(self):
        """Return (count, sum)."""
        return self.count, self.sum

    def from_value(self, x):
        """Set this accumulator's state from a value() tuple.

        Args:
            x: Tuple (count, sum).
        """
        self.count = x[0]
        self.sum = x[1]

    def key_merge(self, stats_dict: Dict[str, Any]):
        """Merge keyed statistics into stats_dict.

        Args:
            stats_dict: Mapping from key to shared statistics.
        """
        if self.key is not None:
            if self.key in stats_dict:
                stats_dict[self.key].combine(self.value())
            else:
                stats_dict[self.key] = self

    def key_replace(self, stats_dict: Dict[str, Any]):
        """Replace this accumulator's statistics with keyed entries from
        stats_dict.

        Args:
            stats_dict: Mapping from key to shared statistics.
        """
        if self.key is not None:
            if self.key in stats_dict:
                self.from_value(stats_dict[self.key].value())


class PoissonEstimatorAccumulatorFactory(object):
    """Factory for creating PoissonEstimatorAccumulator objects."""

    def __init__(self, name, keys):
        """Create a Poisson accumulator factory.

        Args:
            name: Name passed to the accumulators.
            keys: Key passed to the accumulators.
        """
        self.name = name
        self.keys = keys

    def make(self):
        """Return a new PoissonEstimatorAccumulator."""
        return PoissonEstimatorAccumulator(self.name, self.keys)


class PoissonEstimator(ParameterEstimator):
    """Estimates a PoissonDistribution from accumulated sufficient
    statistics, using the Gamma posterior mode when a conjugate prior is
    set."""

    def __init__(self, name: Optional[str] = None, keys: Optional[str] = None, prior: ProbabilityDistribution = default_prior):
        """Create a Poisson estimator.

        Args:
            name (Optional[str]): Name of the estimated distribution.
            keys (Optional[str]): Key for sharing statistics.
            prior (ProbabilityDistribution): Prior on the rate
                (GammaDistribution for conjugacy).
        """

        self.prior = prior
        self.name  = name
        self.keys  = keys
        self.has_conj_prior = isinstance(prior, GammaDistribution)
        self.has_prior = not isinstance(prior, NullDistribution) and prior is not None

    def accumulator_factory(self) -> PoissonEstimatorAccumulatorFactory:
        """Return a PoissonEstimatorAccumulatorFactory for this estimator."""
        return PoissonEstimatorAccumulatorFactory(self.name, self.keys)

    def set_prior(self, prior) -> None:
        """Set the prior on the rate.

        Args:
            prior (ProbabilityDistribution): New prior distribution.
        """
        self.prior = prior

    def get_prior(self) -> ProbabilityDistribution:
        """Return the prior on the rate."""
        return self.prior

    def estimate(self, suff_stat: (float, float)) -> PoissonDistribution:
        """Estimate a PoissonDistribution from sufficient statistics.

        Args:
            suff_stat: Tuple (count, sum) as returned by
                PoissonEstimatorAccumulator.value().

        Returns:
            PoissonDistribution with the Gamma-posterior-mode rate when a
            conjugate prior is set, otherwise the maximum-likelihood rate.
        """

        nobs, psum = suff_stat

        if self.has_conj_prior:

            k, theta = self.prior.get_parameters()

            new_k     = k + psum
            new_theta = theta/(nobs*theta + 1)

            # posterior mode of Gamma(k, theta) is (k-1)*theta for k >= 1; fall
            # back to the posterior mean when the mode is at the boundary
            if new_k >= 1.0:
                posterior_mode = (new_k - 1.0)*new_theta
            else:
                posterior_mode = new_k*new_theta

            posterior_mode = max(posterior_mode, 1.0e-128)

            return PoissonDistribution(posterior_mode, name=self.name, prior=GammaDistribution(new_k, new_theta))

        else:
            lam = psum/nobs if nobs > 0 else 1.0
            return PoissonDistribution(max(lam, 1.0e-128), name=self.name, prior=self.prior if self.has_prior else null_dist)

# --- API naming aliases (notes/distribution_api_naming_accounting.md) ---
PoissonAccumulator = PoissonEstimatorAccumulator
PoissonAccumulatorFactory = PoissonEstimatorAccumulatorFactory
