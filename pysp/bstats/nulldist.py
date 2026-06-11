"""The null distribution: a placeholder that models nothing and accepts any
data type.

Every observation has density 1 (log density 0), so plugging null_dist into a
slot of a compound model (e.g. the len_dist of a SequenceDistribution, or the
prior of a distribution with no usable prior) leaves the model's likelihood
unchanged. NullSampler always samples None, NullAccumulator gathers no
statistics, and NullEstimator always estimates the shared null_dist instance.

The module-level singletons null_dist and null_estimator should be preferred
over creating new instances.
"""
from typing import Optional, Any
from pysp.bstats.pdist import StatisticAccumulator, ParameterEstimator, ProbabilityDistribution
import numpy as np



class NullDistribution(ProbabilityDistribution[Any,None,None]):
	"""Distribution assigning density 1 to every observation. Serves as the
	'nothing is modeled here' placeholder for optional model components."""

	def __init__(self):
		self.parents = []
		pass

	def __str__(self):
		return 'NullDistribution()'

	def get_prior(self):
		"""Returns this distribution (the null distribution is its own prior)."""
		return self

	def set_prior(self, prior):
		"""Ignores the prior (nothing is modeled).

		Args:
			prior (ProbabilityDistribution): Ignored.
		"""
		pass

	def get_parameters(self):
		"""Returns None (the null distribution has no parameters)."""
		return None

	def set_parameters(self, params):
		"""Ignores the parameters (nothing is modeled).

		Args:
			params (Any): Ignored.
		"""
		pass

	def moments(self, p, o):
		"""Returns 1.0 for any requested moment.

		Args:
			p (Any): Moment order. Ignored.
			o (Any): Moment origin. Ignored.

		Returns:
			float: Always 1.0.
		"""
		return 1.0

	def cross_entropy(self, dist):
		"""Returns 0.0 (the null distribution carries no information).

		Args:
			dist (ProbabilityDistribution): Ignored.

		Returns:
			float: Always 0.0.
		"""
		return 0.0

	def entropy(self):
		"""Returns 0.0 (the null distribution carries no information)."""
		return 0.0

	def density(self, x):
		"""Returns 1.0 for any observation.

		Args:
			x (Any): Observation. Ignored.

		Returns:
			float: Always 1.0.
		"""
		return 1.0

	def log_density(self, x):
		"""Returns 0.0 for any observation.

		Args:
			x (Any): Observation. Ignored.

		Returns:
			float: Always 0.0.
		"""
		return 0.0

	def seq_log_density(self, x):
		"""Returns a zero vector with one entry per encoded observation.

		Args:
			x: Sequence-encoded observations (any sized object).

		Returns:
			np.ndarray: Zeros of length len(x).
		"""
		return np.zeros(len(x))

	def seq_encode(self, x):
		"""Returns the observations unchanged.

		Args:
			x: Iterable of observations.

		Returns:
			The input x, unmodified.
		"""
		return x

	def expected_log_density(self, x):
		"""Returns 0.0 for any observation (matches log_density).

		Args:
			x (Any): Observation. Ignored.

		Returns:
			float: Always 0.0.
		"""
		return 0.0

	def seq_expected_log_density(self, x):
		"""Returns a zero vector with one entry per encoded observation.

		Args:
			x: Sequence-encoded observations (any sized object).

		Returns:
			np.ndarray: Zeros of length len(x).
		"""
		return np.zeros(len(x))

	def sampler(self, seed=None):
		"""Returns a NullSampler (samples are always None).

		Args:
			seed (Optional[int]): Ignored (sampling is deterministic).

		Returns:
			NullSampler: Sampler producing None.
		"""
		return NullSampler(self, seed)

	def estimator(self):
		"""Returns a NullEstimator (always estimates null_dist)."""
		return NullEstimator()

null_dist = NullDistribution()


class NullSampler(object):
	"""Sampler for NullDistribution. Every draw is None."""

	def __init__(self, dist=None, seed=None):
		"""NullSampler object.

		Args:
			dist (Optional[NullDistribution]): Distribution being sampled. Unused.
			seed (Optional[int]): Random seed. Unused (sampling is deterministic).
		"""
		self.dist = dist
		self.seed = seed

	def sample(self, size=None):
		"""Draw None values.

		Args:
			size (Optional[int]): Number of samples to draw.

		Returns:
			None if size is None, else a list of size None values.
		"""
		if size is None:
			return None
		else:
			return [None]*size

class NullAccumulator(StatisticAccumulator):
	"""Accumulator for NullDistribution. Gathers no statistics."""

	def __init__(self):
		pass

	def update(self, x, weight, estimate):
		"""Does nothing (no statistics are gathered)."""
		pass

	def seq_update(self, x, weights, estimate):
		"""Does nothing (no statistics are gathered)."""
		pass

	def initialize(self, x, weight, rng):
		"""Does nothing (no statistics are gathered)."""
		pass

	def seq_initialize(self, x, weights, rng):
		"""Does nothing (no statistics are gathered)."""
		pass

	def combine(self, suff_stat):
		"""Returns self unchanged (there is nothing to combine).

		Args:
			suff_stat: Ignored.

		Returns:
			NullAccumulator: This accumulator.
		"""
		return self

	def value(self):
		"""Returns None (the null sufficient statistic)."""
		return None

	def from_value(self, x):
		"""Returns self unchanged (there is no state to set).

		Args:
			x: Ignored.

		Returns:
			NullAccumulator: This accumulator.
		"""
		return self

	def key_merge(self, stats_dict):
		"""Does nothing (null accumulators are keyless)."""
		pass

	def key_replace(self, stats_dict):
		"""Does nothing (null accumulators are keyless)."""
		pass


class NullEstimator(ParameterEstimator):
	"""Estimator for NullDistribution. Always estimates the shared null_dist."""

	def __init__(self, prior=None, keys=None):
		"""NullEstimator object.

		Args:
			prior (Optional[ProbabilityDistribution]): Ignored (kept for API compatibility).
			keys (Optional[str]): Ignored (kept for API compatibility).
		"""
		self.prior = prior
		self.keys = keys

	def accumulator_factory(self):
		"""Returns a factory whose make() creates a NullAccumulator."""
		obj = type('', (object,), {'make': lambda o: NullAccumulator()})()
		return obj

	def get_prior(self):
		"""Returns the shared null_dist instance."""
		return null_dist

	def set_prior(self, prior):
		"""Ignores the prior (nothing is estimated).

		Args:
			prior (ProbabilityDistribution): Ignored.
		"""
		pass

	def estimate(self, suff_stat):
		"""Returns the shared null_dist instance.

		Args:
			suff_stat: Ignored (the null sufficient statistic).

		Returns:
			NullDistribution: The module-level null_dist singleton.
		"""
		return null_dist


null_estimator = NullEstimator()
