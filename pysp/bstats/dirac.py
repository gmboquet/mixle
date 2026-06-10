"""Dirac (point-mass) distribution placing all probability on a single fixed
value. Accepts observations of any equality-comparable type: the log density
is 0 when x equals the fixed value and -inf otherwise.

There are no free parameters, so DiracAccumulator gathers no statistics and
DiracEstimator always returns the configured point mass.
"""
from typing import Optional, Any
from pysp.arithmetic import *
from pysp.bstats.pdist import ProbabilityDistribution, SequenceEncodableAccumulator, ParameterEstimator
from numpy.random import RandomState
from pysp.bstats.nulldist import NullDistribution, null_dist
import numpy as np


class DiracDistribution(ProbabilityDistribution):
	"""Point-mass distribution: density 1 at the fixed value, 0 elsewhere."""

	def __init__(self, value: Any, name: Optional[str] = None, prior: ProbabilityDistribution = null_dist):
		"""DiracDistribution object.

		Args:
			value (Any): The single value carrying all probability mass.
			name (Optional[str]): Name for the distribution.
			prior (ProbabilityDistribution): Prior on the (empty) parameter set.
				Defaults to null_dist.
		"""
		self.value = value
		self.name = name
		self.prior = prior

	def __str__(self):
		return 'DiracDistribution(%s)'%(str(self.value))

	def get_prior(self):
		"""Returns the prior distribution (null_dist unless one was supplied)."""
		return self.prior

	def set_prior(self, prior):
		"""Sets the prior distribution.

		Args:
			prior (ProbabilityDistribution): New prior distribution.
		"""
		self.prior = prior

	def get_parameters(self):
		"""Returns the point-mass value."""
		return self.value

	def set_parameters(self, params):
		"""Sets the point-mass value.

		Args:
			params (Any): New point-mass value.
		"""
		self.value = params

	def density(self, x):
		"""Density at x: 1.0 if x equals the point-mass value, else 0.0.

		Args:
			x (Any): Observation.

		Returns:
			float: 1.0 or 0.0.
		"""
		return np.exp(self.log_density(x))

	def log_density(self, x):
		"""Log density at x: 0.0 if x equals the point-mass value, else -inf.

		Args:
			x (Any): Observation.

		Returns:
			float: 0.0 or -inf.
		"""
		return 0.0 if x == self.value else -inf

	def seq_log_density(self, x):
		"""Vectorized log density for sequence-encoded observations.

		Args:
			x (np.ndarray): Encoded observations from seq_encode.

		Returns:
			np.ndarray: 0.0 where the entry equals the point-mass value, -inf
			elsewhere.
		"""
		return np.asarray([0.0 if u == self.value else -inf for u in x], dtype=float)

	def seq_encode(self, x):
		"""Encodes an iterable of observations for vectorized evaluation.

		Args:
			x: Iterable of observations.

		Returns:
			np.ndarray: Object array of the observations.
		"""
		return np.asarray(x, dtype=object)

	def sampler(self, seed=None):
		"""Returns a DiracSampler (every draw is the point-mass value).

		Args:
			seed (Optional[int]): Ignored (sampling is deterministic).

		Returns:
			DiracSampler object.
		"""
		return DiracSampler(self, seed)

	def estimator(self):
		"""Returns a DiracEstimator fixed at this distribution's value."""
		return DiracEstimator(self.value, prior=self.prior)


class DiracSampler(object):
	"""Sampler for DiracDistribution. Every draw is the fixed value."""

	def __init__(self, dist, seed=None):
		"""DiracSampler object.

		Args:
			dist (DiracDistribution): Distribution to sample from.
			seed (Optional[int]): Ignored (sampling is deterministic).
		"""
		self.dist = dist
		self.value = dist.value

	def sample(self, size=None):
		"""Draw the point-mass value.

		Args:
			size (Optional[int]): Number of samples to draw.

		Returns:
			The fixed value if size is None, else a list of size copies.
		"""
		if size is None:
			return self.value
		else:
			return [self.value]*size


class DiracAccumulator(SequenceEncodableAccumulator):
	"""Accumulator for DiracDistribution. Gathers no statistics (no free
	parameters)."""

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
			DiracAccumulator: This accumulator.
		"""
		return self

	def value(self):
		"""Returns None (the empty sufficient statistic)."""
		return None

	def from_value(self, x):
		"""Returns self unchanged (there is no state to set).

		Args:
			x: Ignored.

		Returns:
			DiracAccumulator: This accumulator.
		"""
		return self

	def key_merge(self, stats_dict):
		"""Does nothing (dirac accumulators are keyless)."""
		pass

	def key_replace(self, stats_dict):
		"""Does nothing (dirac accumulators are keyless)."""
		pass


class DiracEstimator(ParameterEstimator):
	"""Estimator for DiracDistribution. Always estimates the configured point
	mass (the data cannot move it)."""

	def __init__(self, value, prior: ProbabilityDistribution = null_dist, keys=None):
		"""DiracEstimator object.

		Args:
			value (Any): The point-mass value of the estimated distribution.
			prior (ProbabilityDistribution): Prior on the (empty) parameter
				set. Defaults to null_dist.
			keys (Optional[str]): Ignored (kept for API compatibility).
		"""
		self.value  = value
		self.prior  = prior
		self.keys   = keys

	def accumulator_factory(self):
		"""Returns a factory whose make() creates a DiracAccumulator."""
		obj = type('', (object,), {'make': lambda o: DiracAccumulator()})()
		return(obj)

	def get_prior(self):
		"""Returns the prior distribution."""
		return self.prior

	def set_prior(self, prior):
		"""Sets the prior distribution.

		Args:
			prior (ProbabilityDistribution): New prior distribution.
		"""
		self.prior = prior

	def estimate(self, suff_stat):
		"""Returns the point-mass distribution at the configured value.

		Args:
			suff_stat: Ignored (the empty sufficient statistic).

		Returns:
			DiracDistribution: Point mass at self.value.
		"""
		return DiracDistribution(self.value, prior=self.prior)
