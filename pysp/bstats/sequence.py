"""Sequence distribution: i.i.d. draws from a base distribution with an
optional length distribution.

Data type: List[X], where X is the data type of the base distribution. The
log-density of a sequence x = [x_1, ..., x_n] is

	log f(x) = sum_i log f_base(x_i) + log f_len(n),

optionally normalized by the sequence length when len_normalized is set.
Defines the SequenceDistribution, SequenceSampler,
SequenceEstimatorAccumulator, SequenceEstimatorAccumulatorFactory, and
SequenceEstimator classes for use with pysparkplug. Priors compose: the joint
prior is a CompositeDistribution of (base prior, length prior).
"""
from typing import Optional, Tuple, TypeVar, List
import numpy as np
import pandas as pd
from numpy.random import RandomState
from pysp.arithmetic import maxint
from pysp.bstats.nulldist import null_dist, null_estimator
from pysp.bstats.composite import CompositeDistribution
from pysp.bstats.pdist import ProbabilityDistribution, SequenceEncodableAccumulator, ParameterEstimator

X  = TypeVar('X') # Observation type
P1 = TypeVar('P1') # Sequence parameter type
P2 = TypeVar('P2') # Length parameter type
V1 = TypeVar('V1') # Data encoding type
V2 = TypeVar('V2') # Length encoding type


class SequenceDistribution(ProbabilityDistribution[List[X],Tuple[P1,P2],Tuple[V1,V2]]):
	"""Distribution over variable-length sequences of i.i.d. draws from a
	base distribution, with an optional length distribution."""

	def __init__(self, dist: ProbabilityDistribution[X,P1,V1], len_dist: ProbabilityDistribution[int,P2,V2] = null_dist, name: Optional[str] = None, len_normalized: bool = False):
		"""Create a sequence distribution.

		Args:
			dist (ProbabilityDistribution): Base distribution of the entries.
			len_dist (ProbabilityDistribution): Distribution of the sequence
				length (null_dist treats lengths as exogenous).
			name (Optional[str]): Name of the distribution.
			len_normalized (bool): If True, normalize the entry log-density
				sum by the sequence length.
		"""
		self.dist = dist
		self.len_dist = len_dist
		self.len_normalized = len_normalized
		self.name = name
		self.parents = []
		dist.add_parent(self)
		len_dist.add_parent(self)

	def __str__(self):
		return 'SequenceDistribution(%s, len_dist=%s, name=%s)'%(str(self.dist), str(self.len_dist), str(self.name))

	def get_parameters(self) -> Tuple[P1,P2]:
		"""Return (base parameters, length parameters)."""
		return self.dist.get_parameters(), self.len_dist.get_parameters()

	def set_parameters(self, params: Tuple[P1,P2]) -> None:
		"""Set parameters from (base parameters, length parameters).

		Args:
			params: Tuple of base and length distribution parameters.
		"""
		self.dist.set_parameters(params[0])
		self.len_dist.set_parameters(params[1])

	def get_prior(self):
		"""Return the joint prior as a CompositeDistribution of (base prior,
		length prior)."""
		return CompositeDistribution((self.dist.get_prior(), self.len_dist.get_prior()))

	def set_prior(self, prior):
		"""Set the joint prior from a CompositeDistribution of (base prior,
		length prior).

		Args:
			prior: CompositeDistribution matching get_prior() structure.
		"""
		self.dist.set_prior(prior.dists[0])
		self.len_dist.set_prior(prior.dists[1])

	def cross_entropy(self, dist: ProbabilityDistribution):
		"""Cross entropy H(self, dist) for another SequenceDistribution.

		Args:
			dist (SequenceDistribution): Distribution to evaluate against.

		Returns:
			E[len] * H(base, base') + H(len, len').
		"""
		if isinstance(dist, SequenceDistribution):
			v1 = self.dist.cross_entropy(dist.dist)
			v2 = self.len_dist.cross_entropy(dist.len_dist)
			v3 = self.len_dist.moment(1)
			return v3*v1 + v2
		else:
			raise NotImplementedError('SequenceDistribution.cross_entropy is only implemented for SequenceDistribution arguments (got %s).' % type(dist).__name__)

	def entropy(self):
		"""Return E[len] * H(base) + H(len)."""
		v1 = self.dist.entropy()
		v2 = self.len_dist.entropy()
		v3 = self.len_dist.moment(1)
		return v3 * v1 + v2

	def density(self, x) -> float:
		"""Density of the sequence x (excluding the length term).

		Args:
			x: Sequence of base-distribution observations.

		Returns:
			Product of entry densities (length-normalized if configured).
		"""
		rv = 1.0
		for i in range(len(x)):
			rv *= self.dist.density(x[i])

		if self.len_normalized and len(x) > 0:
			rv = np.power(rv, 1.0/len(x))

		return rv

	def log_density(self, x) -> float:
		"""Log-density of the sequence x including the length term.

		Args:
			x: Sequence of base-distribution observations.

		Returns:
			Sum of entry log-densities (length-normalized if configured)
			plus the length log-density.
		"""
		rv = 0.0
		for i in range(len(x)):
			rv += self.dist.log_density(x[i])

		if self.len_normalized and len(x) > 0:
			rv /= len(x)

		rv += self.len_dist.log_density(len(x))

		return rv

	def expected_log_density(self, x) -> float:
		"""Prior-expected log-density of the sequence x.

		Args:
			x: Sequence of base-distribution observations.

		Returns:
			Sum of entry expected log-densities (length-normalized if
			configured) plus the length expected log-density.
		"""
		rv = 0.0
		for i in range(len(x)):
			rv += self.dist.expected_log_density(x[i])

		if self.len_normalized and len(x) > 0:
			rv /= len(x)

		if self.len_dist is not None:
			rv += self.len_dist.expected_log_density(len(x))

		return rv

	def seq_log_density(self, x) -> float:
		"""Vectorized log-density at sequence-encoded input x.

		Args:
			x: Encoded data from seq_encode().

		Returns:
			Numpy array of log-densities, one entry per sequence.
		"""

		idx, icnt, inz, enc_seq, enc_nseq = x

		if np.all(icnt==0):
			ll_sum = np.zeros(len(icnt), dtype=float)

		else:
			ll = self.dist.seq_log_density(enc_seq)
			ll_sum = np.bincount(idx, weights=ll, minlength=len(icnt))

			if self.len_normalized:
				ll_sum *= icnt

		if self.len_dist is not None and enc_nseq is not None:
			nll = self.len_dist.seq_log_density(enc_nseq)
			ll_sum += nll

		return ll_sum

	def seq_expected_log_density(self, x):
		"""Vectorized expected log-density at sequence-encoded input x.

		Args:
			x: Encoded data from seq_encode().

		Returns:
			Numpy array of expected log-densities, one entry per sequence.
		"""

		idx, icnt, inz, enc_seq, enc_nseq = x

		if np.all(icnt==0):
			ll_sum = np.zeros(len(icnt), dtype=float)

		else:
			ll = self.dist.seq_expected_log_density(enc_seq)
			ll_sum = np.bincount(idx, weights=ll, minlength=len(icnt))

			if self.len_normalized:
				ll_sum *= icnt

		if self.len_dist is not None and enc_nseq is not None:
			nll = self.len_dist.seq_expected_log_density(enc_nseq)
			ll_sum += nll

		return ll_sum


	def seq_encode(self, x):
		"""Encode a list of sequences for vectorized evaluation.

		Args:
			x: List of sequences of base-distribution observations.

		Returns:
			Tuple (sequence index per entry, reciprocal lengths, nonzero
			length mask, encoded entries, encoded lengths).
		"""

		tx   = []
		nx   = []
		tidx = []

		for i in range(len(x)):
			m = len(x[i])
			nx.append(m)
			tx.extend(x[i])
			tidx.extend([i]*m)

		rv1 = np.asarray(tidx, dtype=int)
		rv2 = np.asarray(nx, dtype=float)
		rv3 = (rv2 != 0)

		rv2[rv3] = 1.0/rv2[rv3]
		#rv2[rv3] = 1.0

		rv4 = self.dist.seq_encode(tx)
		rv5 = self.len_dist.seq_encode(nx)

		return rv1, rv2, rv3, rv4, rv5

	def sampler(self, seed=None):
		"""Return a SequenceSampler for this distribution.

		Args:
			seed (Optional[int]): Seed for the random number generator.
		"""
		return SequenceSampler(self, seed)

	def estimator(self):
		"""Return a SequenceEstimator matching this distribution."""
		return SequenceEstimator(self.dist.estimator(), self.len_dist.estimator(), name=self.name)



class SequenceSampler(object):
	"""Draws variable-length sequences from a SequenceDistribution."""

	def __init__(self, dist, seed=None):
		"""Create a sampler for a SequenceDistribution.

		Args:
			dist (SequenceDistribution): Distribution to sample from.
			seed (Optional[int]): Seed for the random number generator.
		"""
		self.dist        = dist
		self.rng         = RandomState(seed)
		self.distSampler = self.dist.dist.sampler(seed=self.rng.randint(maxint))
		self.lenSampler  = self.dist.len_dist.sampler(seed=self.rng.randint(maxint))

	def sample(self, size=None):
		"""Draw size sequences (or one sequence when size is None).

		Args:
			size (Optional[int]): Number of sequences to draw.

		Returns:
			A single sequence when size is None, otherwise a list of
			sequences.
		"""

		if size is None:
			n = self.lenSampler.sample()
			return [self.distSampler.sample() for i in range(n)]
		else:
			return [self.sample() for i in range(size)]


class SequenceEstimatorAccumulator(SequenceEncodableAccumulator):
	"""Accumulates entry and length sufficient statistics for sequence
	estimation."""

	def __init__(self, accumulator, len_normalized, len_accumulator, keys):
		"""Create a sequence accumulator.

		Args:
			accumulator: StatisticAccumulator for the entry distribution.
			len_normalized (bool): If True, weight entries by 1/length.
			len_accumulator: Optional StatisticAccumulator for the length
				distribution (None disables length accumulation).
			keys: Tuple (dist_key, len_key) for sharing statistics.
		"""
		self.accumulator = accumulator
		self.len_accumulator = len_accumulator
		self.dist_key = keys[0]
		self.len_key = keys[1]
		self.len_normalized = len_normalized

	def update(self, x, weight, estimate):
		"""Accumulate one weighted sequence observation.

		Args:
			x: Sequence of base-distribution observations.
			weight (float): Observation weight.
			estimate (Optional[SequenceDistribution]): Current model
				estimate or None.
		"""

		if estimate is None:
			w = weight / len(x) if (self.len_normalized and len(x) > 0) else weight

			for i in range(len(x)):
				self.accumulator.update(x[i], w, None)

			if self.len_accumulator is not None:
				self.len_accumulator.update(len(x), weight, None)

		else:
			w = weight / len(x) if (self.len_normalized and len(x) > 0) else weight

			for i in range(len(x)):
				self.accumulator.update(x[i], w, estimate.dist)

			if self.len_accumulator is not None:
				self.len_accumulator.update(len(x), weight, estimate.len_dist)

	def initialize(self, x, weight, rng):
		"""Initialize with one weighted sequence observation.

		Args:
			x: Sequence of base-distribution observations.
			weight (float): Observation weight.
			rng (RandomState): Random number generator.
		"""

		if len(x) > 0:
			w = weight/len(x) if self.len_normalized else weight
			for xx in x:
				self.accumulator.initialize(xx, w, rng)

		if self.len_accumulator is not None:
			self.len_accumulator.initialize(len(x), weight, rng)


	def combine(self, suff_stat):
		"""Merge another accumulator's value() into this one.

		Args:
			suff_stat: Tuple (entry suff stats, length suff stats or None).

		Returns:
			This accumulator.
		"""
		self.accumulator.combine(suff_stat[0])
		if self.len_accumulator is not None:
			self.len_accumulator.combine(suff_stat[1])
		return self

	def value(self):
		"""Return (entry suff stats, length suff stats or None)."""
		if self.len_accumulator is not None:
			return self.accumulator.value(), self.len_accumulator.value()
		else:
			return self.accumulator.value(), None

	def from_value(self, x):
		"""Set this accumulator's state from a value() tuple.

		Args:
			x: Tuple (entry suff stats, length suff stats or None).

		Returns:
			This accumulator.
		"""
		self.accumulator.from_value(x[0])
		if self.len_accumulator is not None:
			self.len_accumulator.from_value(x[1])
		return self

	def get_seq_lambda(self):
		rv = self.accumulator.get_seq_lambda()
		if self.len_accumulator is not None:
			rv.extend(self.len_accumulator.get_seq_lambda())
		return rv

	def seq_initialize(self, x, weights, rng):
		"""Vectorized initialization from sequence-encoded data.

		Args:
			x: Encoded data from SequenceDistribution.seq_encode().
			weights (np.ndarray): Per-sequence observation weights.
			rng (RandomState): Random number generator.
		"""
		idx, icnt, inz, enc_seq, enc_nseq = x

		w = weights[idx]*icnt[idx] if self.len_normalized else weights[idx]

		self.accumulator.seq_initialize(enc_seq, w, rng)

		if self.len_accumulator is not None:
			self.len_accumulator.seq_initialize(enc_nseq, weights, rng)

	def seq_update(self, x, weights, estimate):
		"""Vectorized update from sequence-encoded data.

		Args:
			x: Encoded data from SequenceDistribution.seq_encode().
			weights (np.ndarray): Per-sequence observation weights.
			estimate (SequenceDistribution): Current model estimate.
		"""
		idx, icnt, inz, enc_seq, enc_nseq = x

		w = weights[idx]*icnt[idx] if self.len_normalized else weights[idx]

		self.accumulator.seq_update(enc_seq, w, estimate.dist)

		if self.len_accumulator is not None:
			self.len_accumulator.seq_update(enc_nseq, weights, estimate.len_dist)

	def key_merge(self, stats_dict):
		"""Merge keyed statistics into stats_dict.

		Args:
			stats_dict: Mapping from key to shared statistics.
		"""

		if self.dist_key is not None:
			if self.dist_key in stats_dict:
				stats_dict[self.dist_key].combine(self.value())
			else:
				stats_dict[self.dist_key] = self

		if self.len_key is not None:
			if self.len_key in stats_dict:
				stats_dict[self.len_key].combine(self.value())
			else:
				stats_dict[self.len_key] = self

		self.accumulator.key_merge(stats_dict)

		if self.len_accumulator is not None:
			self.len_accumulator.key_merge(stats_dict)


	def key_replace(self, stats_dict):
		"""Replace this accumulator's statistics with keyed entries from
		stats_dict.

		Args:
			stats_dict: Mapping from key to shared statistics.
		"""

		if self.dist_key is not None:
			if self.dist_key in stats_dict:
				self.from_value(stats_dict[self.dist_key].value())
		if self.len_key is not None:
			if self.len_key in stats_dict:
				self.from_value(stats_dict[self.len_key].value())

		self.accumulator.key_replace(stats_dict)

		if self.len_accumulator is not None:
			self.len_accumulator.key_replace(stats_dict)


class SequenceEstimatorAccumulatorFactory(object):
	"""Factory for creating SequenceEstimatorAccumulator objects."""

	def __init__(self, dist_factory, len_normalized, len_factory, keys):
		"""Create a sequence accumulator factory.

		Args:
			dist_factory: Accumulator factory for the entry estimator.
			len_normalized (bool): Passed to the accumulators.
			len_factory: Accumulator factory for the length estimator, or
				None to disable length accumulation.
			keys: Tuple (dist_key, len_key) passed to the accumulators.
		"""
		self.dist_factory = dist_factory
		self.len_normalized = len_normalized
		self.len_factory = len_factory
		self.keys = keys

	def make(self):
		"""Return a new SequenceEstimatorAccumulator."""
		len_acc = None if self.len_factory is None else self.len_factory.make()
		return SequenceEstimatorAccumulator(self.dist_factory.make(), self.len_normalized, len_acc, self.keys)


class SequenceEstimator(ParameterEstimator):
	"""Estimates a SequenceDistribution from accumulated entry and length
	sufficient statistics."""

	def __init__(self, estimator: ParameterEstimator, len_estimator: ParameterEstimator = null_estimator, len_normalized=False, name=None, keys: Tuple[Optional[str], Optional[str]] = (None, None)):
		"""Create a sequence estimator.

		Args:
			estimator (ParameterEstimator): Estimator for the entry
				distribution.
			len_estimator (ParameterEstimator): Estimator for the length
				distribution.
			len_normalized (bool): If True, weight entries by 1/length.
			name (Optional[str]): Name of the estimated distribution.
			keys: Tuple (dist_key, len_key) for sharing statistics.
		"""
		self.name = name
		self.estimator = estimator
		self.len_estimator = len_estimator
		self.keys = keys
		self.len_normalized=len_normalized

	def get_prior(self):
		"""Return the joint prior as a CompositeDistribution of (entry
		prior, length prior)."""
		return CompositeDistribution((self.estimator.get_prior(), self.len_estimator.get_prior()))

	def set_prior(self, prior):
		"""Set the joint prior from a CompositeDistribution of (entry prior,
		length prior).

		Args:
			prior: CompositeDistribution matching get_prior() structure.
		"""
		self.estimator.set_prior(prior.dists[0])
		self.len_estimator.set_prior(prior.dists[1])

	def model_log_density(self, model: CompositeDistribution) -> float:
		"""Log density of the model parameters under this estimator's prior.

		Args:
			model: Model whose parameters are evaluated.

		Returns:
			Prior log-density at the model parameters.
		"""
		prior = self.get_prior()
		params = model.get_parameters()

		return prior.log_density(params)

	def accumulator_factory(self):
		"""Return a SequenceEstimatorAccumulatorFactory for this
		estimator."""

		if self.len_estimator is None:
			len_factory = None
		else:
			len_factory = self.len_estimator.accumulator_factory()

		return SequenceEstimatorAccumulatorFactory(self.estimator.accumulator_factory(), self.len_normalized, len_factory, self.keys)

	def scale_suff_stat(self, suff_stat, c):
		"""Scale entry and length sufficient statistics through child estimators."""
		entry_stat = self.estimator.scale_suff_stat(suff_stat[0], c)
		if self.len_estimator is None or suff_stat[1] is None:
			return entry_stat, None
		return entry_stat, self.len_estimator.scale_suff_stat(suff_stat[1], c)

	def estimate(self, suff_stat):
		"""Estimate a SequenceDistribution from sufficient statistics.

		Args:
			suff_stat: Tuple (entry suff stats, length suff stats or None).

		Returns:
			SequenceDistribution with estimated entry and length
			distributions.
		"""

		if self.len_estimator is None:
			return SequenceDistribution(self.estimator.estimate(suff_stat[0]), None, len_normalized=self.len_normalized)
		else:
			return SequenceDistribution(self.estimator.estimate(suff_stat[0]), self.len_estimator.estimate(suff_stat[1]), len_normalized=self.len_normalized)
