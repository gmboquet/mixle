"""Evaluate, estimate, and sample from a distribution with missing values.

Defines the OptionalDistribution, OptionalSampler, OptionalEstimatorAccumulator,
OptionalEstimatorAccumulatorFactory, and OptionalEstimator classes for use with
pysparkplug's Bayesian estimation (pysp.bstats).

Data type: Union[missing-value type, inner data type]. An observation equals
the designated missing value (default None, NaN also recognized for scalar
missing values) with probability p, and is otherwise drawn from the wrapped
distribution, giving log-density
    log(f(x)) = log(p)                            if x is missing,
    log(f(x)) = log(1-p) + log(f_dist(x))         otherwise.
A Beta prior on p is conjugate and enables variational (expected log-density)
updates.
"""
from typing import Optional, Any, Dict, Union
from numpy.random import RandomState
from pysp.bstats.pdist import ParameterEstimator, ProbabilityDistribution, StatisticAccumulator
from pysp.bstats.beta import BetaDistribution
from pysp.bstats.composite import CompositeDistribution
from pysp.bstats.nulldist import NullDistribution, null_dist
from pysp.utils.special import stirling2
import numpy as np
from scipy.special import gammaln, digamma, exp1
from scipy.optimize import minimize_scalar
import scipy.integrate


default_prior = BetaDistribution(1.0001, 1.0001)


class OptionalDistribution(ProbabilityDistribution):
    """Distribution over observations that are missing with probability p and
    otherwise drawn from a wrapped distribution."""

    def __init__(self, dist: ProbabilityDistribution, p: float = 0.5, missing_value: Any = None, name: Optional[str] = None, prior: ProbabilityDistribution = default_prior, keys: Optional[str] = None):
        """OptionalDistribution object for data with missing values.

        Args:
            dist (ProbabilityDistribution): Distribution of non-missing
                observations.
            p (float): Probability that an observation is missing.
            missing_value (Any): Value marking a missing observation. NaN is
                also recognized when missing_value is a scalar NaN.
            name (Optional[str]): String for name of object.
            prior (ProbabilityDistribution): Prior on p (Beta is conjugate).
            keys (Optional[str]): Key used to merge sufficient statistics
                across accumulators.
        """
        self.name = name
        self.keys = keys
        self.dist = dist
        self.p    = p
        self.log_p0 = np.log(p)
        self.log_p1 = np.log1p(-p)
        self.missing_value = missing_value
        self._set_prior(prior)
        self.mv_is_nan = False if not np.isscalar(missing_value) else np.isnan(missing_value)

    def __str__(self) -> str:
        """Returns string representation of OptionalDistribution object."""
        return 'OptionalDistribution(%s, p=%s, missing_value=%s, name=%s, prior=%s, keys=%s)' % (str(self.dist), repr(self.p), repr(self.missing_value), repr(self.name), str(self.prior), repr(self.keys))

    def get_parameters(self):
        """Returns tuple (p, parameters of the wrapped distribution)."""
        return self.p, self.dist.get_parameters()

    def set_parameters(self, params) -> None:
        """Sets the missing probability and wrapped-distribution parameters.

        Args:
            params: Tuple (p, wrapped-distribution parameters).
        """
        self.p = params[0]
        self.log_p0 = np.log(params[0])
        self.log_p1 = np.log1p(-params[0])
        #self.missing_value = params[0][1]
        self.dist.set_parameters(params[1])

    def get_prior(self) -> ProbabilityDistribution:
        """Returns the joint prior as a CompositeDistribution of (prior on p,
        prior of the wrapped distribution)."""
        return CompositeDistribution((self.prior, self.dist.get_prior()))

    def set_prior(self, prior: ProbabilityDistribution):
        """Sets the joint prior from its composable form.

        Args:
            prior (CompositeDistribution): Pair (prior on p, prior of the
                wrapped distribution).
        """
        self.dist.set_prior(prior.dists[1])
        self._set_prior(prior.dists[0])

    def _set_prior(self, prior: ProbabilityDistribution):
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
        """Returns the accepted data type: Union of the missing-value type
        and the wrapped distribution's data type.

        Uses the bstats get_data_type() convention, falling back to the
        legacy get_type() name, and to Any when the wrapped distribution
        declares neither.
        """
        get_type_fn = getattr(self.dist, 'get_data_type', None)
        if get_type_fn is None:
            get_type_fn = getattr(self.dist, 'get_type', None)
        base_type = Any if get_type_fn is None else get_type_fn()
        return Union[type(self.missing_value), base_type]

    def density(self, x) -> float:
        """Density at observation x.

        See log_density() for details.

        Args:
            x: Observation (missing value or wrapped-distribution value).

        Returns:
            Density at observation x.
        """
        return np.exp(self.log_density(x))

    def log_density(self, x) -> float:
        """Log-density at observation x.

        Returns log(p) when x is the missing value (or NaN for scalar NaN
        missing values), otherwise log(1-p) plus the wrapped distribution's
        log-density at x.

        Args:
            x: Observation (missing value or wrapped-distribution value).

        Returns:
            Log-density at observation x.
        """
        if (x is self.missing_value) or (self.mv_is_nan and np.isscalar(x) and np.isnan(x)):
            return self.log_p0
        else:
            return self.log_p1 + self.dist.log_density(x)

    def expected_log_density(self, x) -> float:
        """Posterior-expected log-density E_q[log p(x|theta)] at x.

        With a conjugate Beta prior on p the expectation over p is available
        in closed form via digamma terms; otherwise this falls back to the
        plug-in log_density.

        Args:
            x: Observation (missing value or wrapped-distribution value).

        Returns:
            Expected log-density at observation x.
        """
        if self.has_conj_prior:
            da, db, dab = self.conj_prior_params
            if (x is self.missing_value) or (self.mv_is_nan and np.isscalar(x) and np.isnan(x)):
                return da - dab
            else:
                return db - dab + self.dist.expected_log_density(x)
        else:
            return self.log_density(x)

    def cross_entropy(self, dist: ProbabilityDistribution) -> float:
        """Cross entropy H(self, dist) of this distribution with another.

        Args:
            dist (ProbabilityDistribution): Distribution to evaluate against.

        Returns:
            Cross entropy in nats.
        """
        if isinstance(dist, OptionalDistribution):
            v1 = -self.p * dist.log_p0
            v2 = (1.0 - self.p) * (-dist.log_p1 + self.dist.cross_entropy(dist.dist))
            return v1 + v2
        else:
            v1 = -self.p * dist.log_density(self.missing_value)
            v2 = (1.0 - self.p) * self.dist.cross_entropy(dist)
            return v1 + v2

    def entropy(self) -> float:
        """Returns the entropy of this distribution in nats."""
        v1 = -self.p * self.log_p0
        v2 = (1.0 - self.p) * (-self.log_p1 + self.dist.entropy())
        return v1 + v2

    def seq_log_density(self, x):
        """Vectorized log-density at sequence-encoded input x.

        Args:
            x: Sequence-encoded data from seq_encode().

        Returns:
            Numpy array of log-densities, one per encoded observation.
        """
        rv = np.empty(x[0], dtype=np.float64)
        rv.fill(self.log_p0)
        rv[x[1]] = self.dist.seq_log_density(x[3]) + self.log_p1
        return rv

    def seq_expected_log_density(self, x):
        """Vectorized posterior-expected log-density at sequence-encoded x.

        Falls back to seq_log_density when no conjugate prior on p is set.

        Args:
            x: Sequence-encoded data from seq_encode().

        Returns:
            Numpy array of expected log-densities.
        """
        if not self.has_conj_prior:
            return self.seq_log_density(x)
        da, db, dab = self.conj_prior_params
        aa = da - dab
        bb = db - dab

        rv = np.empty(x[0], dtype=np.float64)
        rv.fill(aa)
        rv[x[1]] = self.dist.seq_expected_log_density(x[3]) + bb
        return rv

    def seq_encode(self, x):
        """Sequence-encode iterable of observations for vectorized methods.

        Args:
            x: Iterable of observations (missing or wrapped-distribution
                values).

        Returns:
            Tuple (count, non-missing indices, missing indices, encoding of
            the non-missing values by the wrapped distribution).
        """
        nz_idx = []
        iz_idx = []
        nz_val = []
        cnt = 0

        for i,xx in enumerate(x):
            cnt += 1

            if  (xx is self.missing_value) or (self.mv_is_nan and np.isscalar(xx) and np.isnan(xx)):
                iz_idx.append(i)
            else:
                nz_idx.append(i)
                nz_val.append(xx)

        nz_idx = np.asarray(nz_idx, dtype=np.int32)
        iz_idx = np.asarray(iz_idx, dtype=np.int32)
        nz_val = self.dist.seq_encode(nz_val)

        return cnt, nz_idx, iz_idx, nz_val

    def sampler(self, seed: Optional[int] = None):
        """Create an OptionalSampler from this distribution.

        Args:
            seed (Optional[int]): Used to set seed in random sampler.

        Returns:
            OptionalSampler object.
        """
        return OptionalSampler(self, seed)

    def estimator(self):
        """Create an OptionalEstimator matching this distribution's
        configuration.

        The wrapped distribution's estimator() is used for the non-missing
        observations, and the missing value and prior on p are carried over.

        Returns:
            OptionalEstimator object.
        """
        return OptionalEstimator(self.dist.estimator(), missing_value=self.missing_value, name=self.name, keys=self.keys, prior=self.prior)


class OptionalSampler(object):
    """Sampler that emits the missing value with probability p and otherwise
    samples from the wrapped distribution."""

    def __init__(self, dist, seed=None):
        """OptionalSampler object.

        Args:
            dist (OptionalDistribution): Distribution to sample from.
            seed (Optional[int]): Used to set seed in random sampler.
        """
        rng  = np.random.RandomState(seed)
        self.dist = dist
        self.obs_sampler = dist.dist.sampler(rng.randint(0, 2**31))
        self.mis_sampler = np.random.RandomState(rng.randint(0, 2**31))

    def sample(self, size=None):
        """Draw size samples (missing value or wrapped-distribution sample).

        Args:
            size (Optional[int]): Number of samples; a single sample is
                returned when None.

        Returns:
            A single sample or a list of size samples.
        """
        if size is None:
            if self.mis_sampler.rand() <= self.dist.p:
                return self.dist.missing_value
            else:
                return self.obs_sampler.sample()
        else:
            return [self.sample() for i in range(size)]


class OptionalEstimatorAccumulator(StatisticAccumulator):
    """Accumulates missing/non-missing weights plus the wrapped accumulator's
    sufficient statistics."""

    def __init__(self, accumulator, missing_value, name, keys):
        """OptionalEstimatorAccumulator object.

        Args:
            accumulator (StatisticAccumulator): Accumulator for the
                non-missing observations.
            missing_value (Any): Value marking a missing observation.
            name (Optional[str]): String for name of object.
            keys (Optional[str]): Key used to merge sufficient statistics
                across accumulators.
        """
        self.acc   = accumulator
        self.name  = name
        self.key   = keys
        self.psum  = 0.0
        self.nsum  = 0.0
        self.missing_value = missing_value
        self.mv_is_nan = False if not np.isscalar(missing_value) else np.isnan(missing_value)

    def initialize(self, x, weight, rng):
        self.update(x, weight, None)

    def seq_initialize(self, x, weights, rng):
        self.seq_update(x, weights, None)

    def update(self, x, weight, estimate):
        """Accumulate one observation with the given weight.

        Missing observations add weight to the missing count; non-missing
        observations add weight to the observed count and are passed to the
        wrapped accumulator.

        Args:
            x: Observation (missing value or wrapped-distribution value).
            weight (float): Weight of the observation.
            estimate (Optional[OptionalDistribution]): Previous estimate; its
                wrapped distribution is forwarded to the inner accumulator.
        """
        if (x is self.missing_value) or (self.mv_is_nan and np.isscalar(x) and np.isnan(x)):
            self.psum += weight
        else:
            self.nsum += weight
            self.acc.update(x, weight, None if estimate is None else estimate.dist)

    def seq_update(self, x, weights, estimate):
        """Vectorized update from sequence-encoded data.

        Args:
            x: Sequence-encoded data from OptionalDistribution.seq_encode().
            weights (np.ndarray): Weight per encoded observation.
            estimate (Optional[OptionalDistribution]): Previous estimate; its
                wrapped distribution is forwarded to the inner accumulator.
        """
        cnt, nz_idx, iz_idx, nz_val = x

        self.psum += weights[iz_idx].sum()
        self.nsum += weights[nz_idx].sum()
        self.acc.seq_update(nz_val, weights[nz_idx], None if estimate is None else estimate.dist)

    def combine(self, suff_stat):
        """Merge another accumulator's value() into this accumulator.

        Args:
            suff_stat: Tuple (missing weight, observed weight, wrapped
                sufficient statistics).

        Returns:
            This accumulator.
        """
        self.psum += suff_stat[0]
        self.nsum += suff_stat[1]
        self.acc.combine(suff_stat[2])
        return self

    def value(self):
        """Returns tuple (missing weight, observed weight, wrapped
        sufficient statistics)."""
        return self.psum, self.nsum, self.acc.value()

    def from_value(self, x):
        """Set this accumulator's state from a value() tuple.

        Args:
            x: Tuple (missing weight, observed weight, wrapped sufficient
                statistics).

        Returns:
            This accumulator.
        """
        self.psum = x[0]
        self.nsum = x[1]
        self.acc.from_value(x[2])
        return self

    def key_merge(self, stats_dict: Dict[str, Any]):
        if self.key is not None:
            if self.key in stats_dict:
                stats_dict[self.key].combine(self.value())
            else:
                stats_dict[self.key] = self

    def key_replace(self, stats_dict: Dict[str, Any]):
        if self.key is not None:
            if self.key in stats_dict:
                self.from_value(stats_dict[self.key].value())


class OptionalEstimatorAccumulatorFactory(object):
    """Factory for creating OptionalEstimatorAccumulator objects."""

    def __init__(self, acc_factory, missing_value, name, keys):
        """OptionalEstimatorAccumulatorFactory object.

        Args:
            acc_factory: Accumulator factory of the wrapped estimator (or
                None).
            missing_value (Any): Value marking a missing observation.
            name (Optional[str]): String for name of object.
            keys (Optional[str]): Key used to merge sufficient statistics
                across accumulators.
        """
        self.name = name
        self.keys = keys
        self.acc_factory = acc_factory
        self.missing_value = missing_value

    def make(self):
        """Returns a new OptionalEstimatorAccumulator object."""
        acc = None if self.acc_factory is None else self.acc_factory.make()
        return OptionalEstimatorAccumulator(acc, self.missing_value, self.name, self.keys)


class OptionalEstimator(ParameterEstimator):
    """Estimator for an OptionalDistribution: missing probability p (Beta
    posterior when the prior is conjugate) plus the wrapped estimator."""

    def __init__(self, estimator: ParameterEstimator, missing_value: Any = None, fixed_prob: Optional[float] = None, name: Optional[str] = None, keys: Optional[str] = None, prior: ProbabilityDistribution = default_prior):
        """OptionalEstimator object.

        Args:
            estimator (ParameterEstimator): Estimator for the non-missing
                observations.
            missing_value (Any): Value marking a missing observation.
            fixed_prob (Optional[float]): If given, the estimated missing
                probability is overridden with this value.
            name (Optional[str]): String for name of object.
            keys (Optional[str]): Key used to merge sufficient statistics
                across accumulators.
            prior (ProbabilityDistribution): Prior on the missing probability
                (Beta is conjugate).
        """
        self.estimator = estimator
        self.name  = name
        self.keys  = keys
        self.prior = prior
        self.fixed_prob = fixed_prob
        self.missing_value = missing_value
        self.has_conj_prior = isinstance(prior, BetaDistribution)
        self.has_prior = not isinstance(prior, NullDistribution) and prior is not None

    def accumulator_factory(self) -> OptionalEstimatorAccumulatorFactory:
        """Returns an OptionalEstimatorAccumulatorFactory wrapping the inner
        estimator's accumulator factory."""
        acc_factory = None if self.estimator is None else self.estimator.accumulator_factory()
        return OptionalEstimatorAccumulatorFactory(acc_factory, self.missing_value, self.name, self.keys)

    def set_prior(self, prior) -> None:
        """Sets the joint prior from its composable form.

        Args:
            prior (CompositeDistribution): Pair (prior on p, prior of the
                wrapped estimator).
        """
        if isinstance(prior, CompositeDistribution):
            self.prior = prior.dists[0]
            self.has_conj_prior = isinstance(self.prior, BetaDistribution)
            self.has_prior = not isinstance(self.prior, NullDistribution) and self.prior is not None
            self.estimator.set_prior(prior.dists[1])

    def get_prior(self) -> ProbabilityDistribution:
        """Returns the joint prior as a CompositeDistribution of (prior on p,
        prior of the wrapped estimator)."""
        return CompositeDistribution((self.prior, self.estimator.get_prior()))

    def scale_suff_stat(self, suff_stat, c):
        """Scale missing/observed weights and delegate wrapped statistics."""
        psum, nsum, dist_suff_stat = suff_stat
        return psum * c, nsum * c, self.estimator.scale_suff_stat(dist_suff_stat, c)

    def estimate(self, suff_stat: (float, float)) -> OptionalDistribution:
        """Estimate an OptionalDistribution from sufficient statistics.

        With a conjugate Beta prior the posterior hyperparameters are updated
        with the missing/observed weights and p is set to the posterior mode;
        otherwise p is the empirical missing fraction.

        Args:
            suff_stat: Tuple (missing weight, observed weight, wrapped
                sufficient statistics) from the accumulator's value().

        Returns:
            OptionalDistribution estimate.
        """
        psum, nsum, dist_suff_stat = suff_stat

        dist = self.estimator.estimate(dist_suff_stat)

        if self.has_conj_prior:

            a, b = self.prior.get_parameters()
            new_a = a + psum
            new_b = b + nsum
            new_p = (psum + a - 1.0)/(psum + nsum + a + b - 2.0)
            new_prior = BetaDistribution(new_a, new_b)

        else:
            new_p = psum/(psum + nsum)
            new_prior = self.prior

        if self.fixed_prob is not None:
            new_p = self.fixed_prob

        return OptionalDistribution(dist, p=new_p, missing_value=self.missing_value, name=self.name, prior=new_prior, keys=self.keys)
