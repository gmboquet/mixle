"""Wrap a distribution so that its data is scored but never re-estimated.

Defines the IgnoredDistribution, IgnoredSampler, IgnoredAccumulator,
IgnoredAccumulatorFactory, and IgnoredEstimator classes for use with
pysparkplug's Bayesian estimation (pysp.bstats).

Data type: whatever the wrapped distribution accepts. The IgnoredDistribution
delegates density evaluation and sampling to the wrapped distribution, while
its accumulator discards all observations, so estimation always returns the
wrapped distribution unchanged. Useful for holding part of a composite model
fixed during EM/VB updates.
"""
from typing import Optional
from pysp.arithmetic import *
from pysp.bstats.pdist import StatisticAccumulator, ParameterEstimator, ProbabilityDistribution
from numpy.random import RandomState
from pysp.bstats.nulldist import NullDistribution
import numpy as np

null_dist = NullDistribution()

class IgnoredDistribution(ProbabilityDistribution):
	"""Distribution wrapper that delegates scoring/sampling to dist but whose
	estimator leaves dist unchanged."""

	def __init__(self, dist : ProbabilityDistribution = null_dist):
		"""IgnoredDistribution object wrapping a fixed distribution.

		Args:
			dist (ProbabilityDistribution): Distribution used for density
				evaluation and sampling. Defaults to the null distribution
				(log-density 0 for any input).
		"""
		self.dist = dist

	def __str__(self):
		"""Returns string representation of IgnoredDistribution object."""
		return 'IgnoredDistribution(%s)'%(str(self.dist))

	def get_prior(self):
		"""Returns the prior of the wrapped distribution."""
		return self.dist.get_prior()

	def set_prior(self, dist):
		"""Sets the prior of the wrapped distribution.

		Args:
			dist (ProbabilityDistribution): New prior for the wrapped
				distribution.
		"""
		self.dist.set_prior(dist)

	def set_parameters(self, params):
		"""Sets the parameters of the wrapped distribution.

		Args:
			params: Parameter value accepted by the wrapped distribution.
		"""
		self.dist.set_parameters(params)

	def get_parameters(self):
		"""Returns the parameters of the wrapped distribution."""
		return self.dist.get_parameters()

	def cross_entropy(self, dist):
		return self.dist.cross_entropy(dist)

	def entropy(self):
		return self.dist.entropy()

	def density(self, x):
		"""Density of the wrapped distribution at observation x.

		Args:
			x: Observation accepted by the wrapped distribution.

		Returns:
			Density at observation x.
		"""
		return np.exp(self.log_density(x))

	def log_density(self, x):
		"""Log-density of the wrapped distribution at observation x.

		Args:
			x: Observation accepted by the wrapped distribution.

		Returns:
			Log-density at observation x.
		"""
		return self.dist.log_density(x)

	def expected_log_density(self, x):
		"""Posterior-expected log-density of the wrapped distribution at x.

		Args:
			x: Observation accepted by the wrapped distribution.

		Returns:
			E_q[log p(x|theta)] under the wrapped distribution's prior.
		"""
		return self.dist.expected_log_density(x)

	def seq_log_density(self, x):
		"""Vectorized log-density at sequence-encoded input x.

		Args:
			x: Sequence-encoded data from seq_encode().

		Returns:
			Numpy array of log-densities.
		"""
		return self.dist.seq_log_density(x)

	def seq_expected_log_density(self, x):
		"""Vectorized posterior-expected log-density at sequence-encoded x.

		Args:
			x: Sequence-encoded data from seq_encode().

		Returns:
			Numpy array of expected log-densities.
		"""
		return self.dist.seq_expected_log_density(x)

	def seq_encode(self, x):
		"""Sequence-encode iterable of observations with the wrapped distribution.

		Args:
			x: Iterable of observations.

		Returns:
			Encoding produced by the wrapped distribution's seq_encode().
		"""
		return self.dist.seq_encode(x)

	def sampler(self, seed=None):
		"""Create an IgnoredSampler delegating to the wrapped distribution.

		Args:
			seed (Optional[int]): Used to set seed in random sampler.

		Returns:
			IgnoredSampler object.
		"""
		return IgnoredSampler(self, seed)

	def estimator(self):
		"""Create an IgnoredEstimator that returns this wrapped distribution.

		Returns:
			IgnoredEstimator object.
		"""
		return IgnoredEstimator(dist=self.dist)


class IgnoredSampler(object):
	"""Sampler that draws from the wrapped distribution of an IgnoredDistribution."""

	def __init__(self, dist, seed=None):
		"""IgnoredSampler object.

		Args:
			dist (IgnoredDistribution): Distribution whose wrapped member is
				sampled from.
			seed (Optional[int]): Used to set seed in random sampler.
		"""
		self.dist_sampler = dist.dist.sampler(seed)

	def sample(self, size=None):
		"""Draw size samples from the wrapped distribution.

		Args:
			size (Optional[int]): Number of samples; a single sample is
				returned when None.

		Returns:
			A single sample or a list of size samples.
		"""
		return self.dist_sampler.sample(size=size)


class IgnoredAccumulator(StatisticAccumulator):
	"""Accumulator that discards all observations (no sufficient statistics)."""

	def __init__(self):
		pass

	def update(self, x, weight, estimate):
		"""No-op update; the observation is discarded."""
		pass

	def seq_update(self, x, weights, estimate):
		"""No-op vectorized update; the encoded observations are discarded."""
		pass

	def seq_initialize(self, x, weights, rng):
		"""No-op vectorized initialization."""
		pass

	def initialize(self, x, weight, rng):
		"""No-op initialization; the observation is discarded."""
		pass

	def combine(self, suff_stat):
		"""No-op combine; returns self."""
		return self

	def value(self):
		"""Returns None (there are no sufficient statistics)."""
		return None

	def from_value(self, x):
		"""No-op restore; returns self."""
		return self

	def key_merge(self, stats_dict):
		pass

	def key_replace(self, stats_dict):
		pass


class IgnoredAccumulatorFactory(object):
	"""Factory for creating IgnoredAccumulator objects."""

	def make(self):
		"""Returns a new IgnoredAccumulator object."""
		return IgnoredAccumulator()


class IgnoredEstimator(ParameterEstimator):
	"""Estimator that ignores all data and returns a fixed distribution."""

	def __init__(self, dist: ProbabilityDistribution = null_dist, prior: ProbabilityDistribution = null_dist, keys=None):
		"""IgnoredEstimator object holding the distribution to return.

		Args:
			dist (ProbabilityDistribution): Fixed distribution returned (in
				an IgnoredDistribution wrapper) by estimate().
			prior (ProbabilityDistribution): Unused placeholder kept for
				protocol compatibility.
			keys (Optional[str]): Unused placeholder kept for protocol
				compatibility.
		"""
		self.dist   = dist
		self.prior  = prior
		self.keys   = keys

	def accumulator_factory(self):
		"""Returns an IgnoredAccumulatorFactory object."""
		return IgnoredAccumulatorFactory()

	def get_prior(self):
		"""Returns the prior of the wrapped distribution."""
		return self.dist.get_prior()

	def set_prior(self, prior):
		"""Sets the prior of the wrapped distribution.

		Args:
			prior (ProbabilityDistribution): New prior for the wrapped
				distribution.
		"""
		self.dist.set_prior(prior)

	def estimate(self, suff_stat):
		"""Returns an IgnoredDistribution around the fixed distribution.

		Args:
			suff_stat: Ignored (the accumulator collects no statistics).

		Returns:
			IgnoredDistribution wrapping this estimator's dist.
		"""
		return IgnoredDistribution(self.dist)
