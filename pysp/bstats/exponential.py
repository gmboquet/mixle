"""Exponential distribution with rate lambda and a conjugate Gamma prior on
the rate.

Data type: (float). The ExponentialDistribution with rate lam > 0 has
log-density

	log f(x; lam) = log(lam) - lam*x, for x >= 0 (and -inf for x < 0).

Defines the ExponentialDistribution, ExponentialSampler,
ExponentialAccumulator, ExponentialAccumulatorFactory, and
ExponentialEstimator classes for use with pysparkplug. With a
GammaDistribution prior on the rate, estimation is MAP and the posterior
Gamma is carried forward as the new prior. expected_log_density uses the
Gamma expectations E[lam] = k*theta and E[ln lam] = psi(k) + ln(theta).
"""
from typing import Dict, Tuple, Sequence, Optional

from pysp.arithmetic import *
from pysp.bstats.pdist import ParameterEstimator, ProbabilityDistribution, StatisticAccumulator
import numpy as np
from scipy.special import gammaln, digamma

from pysp.bstats.gamma import GammaDistribution

default_prior = GammaDistribution(1.0001, 1.0e6)


class ExponentialDistribution(ProbabilityDistribution):
	"""Exponential distribution with rate parameter lam."""

	def __init__(self, lam: float, name: Optional[str] = None, prior: ProbabilityDistribution = default_prior):
		"""Create an exponential distribution.

		Args:
			lam (float): Positive rate parameter.
			name (Optional[str]): Name of the distribution.
			prior (ProbabilityDistribution): Prior on the rate
				(GammaDistribution for conjugacy).
		"""

		self.name  = name
		self.set_parameters(lam)
		self.set_prior(prior)

	def __str__(self):
		return 'ExponentialDistribution(%f, name=%s, prior=%s)'%(self.lam, self.name, str(self.prior))

	def get_parameters(self) -> float:
		"""Return the rate parameter lam."""
		return self.lam

	def set_parameters(self, params: float) -> None:
		"""Set the rate parameter.

		Args:
			params (float): Positive rate parameter.
		"""
		self.lam = params
		self.log_lam = np.log(params)
		self.params = params

	def get_prior(self) -> ProbabilityDistribution:
		"""Return the prior on the rate."""
		return self.prior

	def set_prior(self, prior: ProbabilityDistribution):
		"""Set the prior on the rate and cache the natural-parameter
		expectations when the prior is a conjugate Gamma.

		Args:
			prior (ProbabilityDistribution): New prior distribution.
		"""

		self.prior = prior

		if isinstance(prior, GammaDistribution):
			self.conj_prior_params = [prior.k, 1/prior.theta]

			a, b = self.conj_prior_params
			# eta = -lambda
			# E[ eta ] = -k * theta
			# E[ a( eta ) ] = E[ -ln( -eta ) ] = digamma(k) + ln(theta)
			# E[ ln h(x) ] = 0

			e1 = -a / b
			ea = -(digamma(a) - log(b))

			self.expected_nparams = [ea, 0, e1]

		else:
			self.conj_prior_params = None
			self.expected_nparams = None

	def log_density(self, x: float) -> float:
		"""Log-density at observation x.

		Args:
			x (float): Non-negative observation.

		Returns:
			log(lam) - lam*x for x >= 0, -inf otherwise.
		"""
		if x < 0:
			return -inf
		else:
			return -x*self.lam + self.log_lam

	def expected_log_density(self, x):
		"""Prior-expected log-density at observation x.

		Falls back to log_density when no conjugate prior is set.

		Args:
			x (float): Non-negative observation.

		Returns:
			Expected log-density (float) at x.
		"""

		if self.expected_nparams is None:
			return self.log_density(x)

		ea, eb, e1 = self.expected_nparams

		return -inf if x < 0 else e1*x + (eb - ea)

	def seq_log_density(self, x):
		"""Vectorized log-density at sequence-encoded input x.

		Args:
			x (np.ndarray): Encoded data from seq_encode().

		Returns:
			Numpy array of log-densities, one entry per observation.
		"""
		rv = x*(-self.lam)
		rv += self.log_lam
		rv[x < 0] = -inf
		return rv

	def seq_expected_log_density(self, x):
		"""Vectorized expected log-density at sequence-encoded input x.

		Falls back to seq_log_density when no conjugate prior is set.

		Args:
			x (np.ndarray): Encoded data from seq_encode().

		Returns:
			Numpy array of expected log-densities, one entry per observation.
		"""

		if self.expected_nparams is None:
			return self.seq_log_density(x)

		ea, eb, e1 = self.expected_nparams

		rv = e1*x + (eb - ea)
		rv[x < 0] = -inf
		return rv

	def seq_encode(self, x):
		"""Encode a sequence of observations for vectorized evaluation.

		Args:
			x: Iterable of non-negative floats.

		Returns:
			Numpy array of the observations.
		"""
		rv = np.asarray(x)
		return rv

	def value(self):
		"""Return the parameters as a list [lam]."""
		return [self.lam]

	def sampler(self, seed=None):
		"""Return an ExponentialSampler for this distribution.

		Args:
			seed (Optional[int]): Seed for the random number generator.
		"""
		return ExponentialSampler(self, seed)

	def estimator(self):
		"""Return an ExponentialEstimator matching this distribution."""
		return ExponentialEstimator(prior=self.prior)


class ExponentialSampler(object):
	"""Draws observations from an ExponentialDistribution."""

	def __init__(self, dist, seed=None):
		"""Create a sampler for an ExponentialDistribution.

		Args:
			dist (ExponentialDistribution): Distribution to sample from.
			seed (Optional[int]): Seed for the random number generator.
		"""
		self.rng  = np.random.RandomState(seed)
		self.dist = dist

	def sample(self, size=None):
		"""Draw size samples (or one sample when size is None).

		Args:
			size (Optional[int]): Number of samples to draw.

		Returns:
			A float when size is None, otherwise a numpy array of floats.
		"""
		return self.rng.exponential(scale=1/self.dist.lam, size=size)


class ExponentialAccumulator(StatisticAccumulator):
	"""Accumulates the weighted count and sum of observations for
	exponential estimation."""

	def __init__(self, keys):
		"""Create an exponential accumulator.

		Args:
			keys: Tuple whose first entry keys this accumulator for
				statistic sharing.
		"""
		self.sum  = 0.0
		self.count = 0.0
		self.key   = keys[0]

	def update(self, x, weight, estimate):
		"""Accumulate one weighted observation (ignores negative x).

		Args:
			x (float): Observation.
			weight (float): Observation weight.
			estimate: Unused (kept for protocol consistency).
		"""
		if x >= 0:
			self.sum  += x*weight
			self.count += weight

	def seq_update(self, x, weights, estimate):
		"""Vectorized update from sequence-encoded data.

		Args:
			x (np.ndarray): Encoded data from seq_encode().
			weights (np.ndarray): Observation weights.
			estimate: Unused (kept for protocol consistency).
		"""
		self.sum += np.dot(x, weights)
		self.count += np.sum(weights)

	def initialize(self, x, weight, rng):
		"""Initialize with one weighted observation (delegates to update)."""
		self.update(x, weight, None)

	def seq_initialize(self, x, weights, rng):
		"""Vectorized initialization (delegates to seq_update)."""
		self.seq_update(x, weights, None)

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

		Returns:
			This accumulator.
		"""
		self.count = x[0]
		self.sum = x[1]
		return self

	def key_merge(self, stats_dict):
		"""Merge keyed statistics into stats_dict.

		Args:
			stats_dict: Mapping from key to shared statistics.
		"""
		if self.key is not None:
			if self.key in stats_dict:
				vals = stats_dict[self.key]
				stats_dict[self.key] = (vals[0] + self.count, vals[1] + self.sum)
			else:
				stats_dict[self.key] = (self.count, self.sum)

	def key_replace(self, stats_dict):
		"""Replace this accumulator's statistics with keyed entries from
		stats_dict.

		Args:
			stats_dict: Mapping from key to shared statistics.
		"""
		if self.key is not None:
			if self.key in stats_dict:
				vals = stats_dict[self.key]
				self.count = vals[0]
				self.sum   = vals[1]


class ExponentialAccumulatorFactory(object):
	"""Factory for creating ExponentialAccumulator objects."""

	def __init__(self, keys):
		"""Create an exponential accumulator factory.

		Args:
			keys: Key tuple passed to the accumulators.
		"""
		self.keys = keys

	def make(self):
		"""Return a new ExponentialAccumulator."""
		return ExponentialAccumulator(self.keys)


class ExponentialEstimator(ParameterEstimator):
	"""Estimates an ExponentialDistribution from accumulated sufficient
	statistics, using the Gamma posterior mode when a conjugate prior is
	set."""

	def __init__(self, prior=default_prior, name=None,  keys=(None,)):
		"""Create an exponential estimator.

		Args:
			prior: Prior on the rate (GammaDistribution for conjugacy).
			name (Optional[str]): Name of the estimated distribution.
			keys: Key tuple for sharing statistics.
		"""

		self.keys   = keys
		self.name   = name
		self.set_prior(prior)

	def accumulator_factory(self):
		"""Return an ExponentialAccumulatorFactory for this estimator."""
		return ExponentialAccumulatorFactory(self.keys)

	def get_prior(self):
		"""Return the prior on the rate."""
		return self.prior

	def set_prior(self, prior):
		"""Set the prior on the rate.

		Args:
			prior: GammaDistribution, [shape, rate] list, or other prior.
		"""
		self.prior  = prior

		if isinstance(prior, GammaDistribution):
			self.conj_prior_params = [prior.k, 1/prior.theta]
			pass
		elif isinstance(prior, list):
			self.conj_prior_params = prior
		else:
			self.conj_prior_params = None


	def estimate(self, suff_stat):
		"""Estimate an ExponentialDistribution from sufficient statistics.

		Args:
			suff_stat: Tuple (count, sum) as returned by
				ExponentialAccumulator.value().

		Returns:
			ExponentialDistribution with the Gamma-posterior-mode rate when
			a conjugate prior is set, otherwise the maximum-likelihood rate.
		"""

		if self.conj_prior_params is not None:

			a, b = self.conj_prior_params

			n = suff_stat[0] + a
			s = suff_stat[1] + b

			#conj_prior_params = (n + 1, s)

			return ExponentialDistribution((n - 1) / s, name=self.name, prior=GammaDistribution(n,1/s))


		else:

			n = suff_stat[0]
			s = suff_stat[1]

			return ExponentialDistribution(n / s, name=self.name)
