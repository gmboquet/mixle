"""Conditional distribution: a lookup table mapping a conditioning value to a
distribution for the dependent value.

Observations are pairs (c, y) where c selects the distribution dmap[c] used to
score y (when pass_value is True the whole pair is scored instead).
Conditioning values without an entry in dmap fall back to default_dist (the
null distribution by default, contributing log density 0); when default_dist
is None unmatched values score -inf under seq_log_density.

The conditioning value itself is not modeled: sampling is only defined given a
conditioning value (ConditionalDistributionSampler.sample_given), and
estimation fits each entry's distribution from the observations routed to it.
"""
from typing import Dict, Optional
from pysp.arithmetic import maxint
from pysp.bstats.pdist import ProbabilityDistribution, SequenceEncodableAccumulator, \
	ParameterEstimator
from numpy.random import RandomState
import numpy as np
from pysp.bstats.nulldist import null_dist


class ConditionalDistribution(ProbabilityDistribution):
	"""Maps a conditioning value to a member distribution: p(y | c) = dmap[c](y),
	falling back to default_dist for unmatched conditioning values."""

	def __init__(self, dmap, cond_dist=None, default_dist=null_dist, pass_value=False):
		"""ConditionalDistribution object.

		Args:
			dmap (Dict): Map from conditioning value to ProbabilityDistribution.
			cond_dist (Optional[ProbabilityDistribution]): Optional distribution
				for the conditioning value itself. Carried but not evaluated.
			default_dist (Optional[ProbabilityDistribution]): Distribution used
				for conditioning values missing from dmap. Defaults to the null
				distribution (log density 0). When None, unmatched values score
				-inf in seq_log_density.
			pass_value (bool): If True, member distributions score the whole
				pair x; otherwise they score the dependent value x[1].
		"""
		self.dmap         = dmap
		self.cond_dist    = cond_dist
		self.default_dist = default_dist
		self.pass_value   = pass_value
		self.has_default  = default_dist is not None

	def __str__(self):
		return 'ConditionalDistribution(%s, default_dist=%s)' % (str({k: str(v) for k,v in self.dmap.items()}), str(self.default_dist))

	def get_parameters(self):
		"""Returns the parameter map {conditioning value: member parameters}."""
		return {k: v.get_parameters() for k, v in self.dmap.items()}

	def set_parameters(self, params):
		"""Sets member-distribution parameters from a map keyed like dmap.

		Args:
			params (Dict): Map from conditioning value to member parameters.
		"""
		for k, v in params.items():
			self.dmap[k].set_parameters(v)

	def log_density(self, x):
		"""Log density of the pair x = (c, y) under the member selected by c.

		Args:
			x: Pair whose first entry is the conditioning value.

		Returns:
			float: Log density of y (or of x when pass_value is set) under
			dmap[c], or under default_dist when c is unmatched.
		"""
		dist = self.dmap.get(x[0], self.default_dist)
		if dist is None:
			return -np.inf
		if self.pass_value:
			return dist.log_density(x)
		else:
			return dist.log_density(x[1])

	def seq_log_density(self, x):
		"""Vectorized log density for sequence-encoded observations.

		Args:
			x: Encoding from seq_encode.

		Returns:
			np.ndarray: Log density per observation. Without a default
			distribution, unmatched conditioning values score -inf.
		"""
		sz, cond_vals, idx_vals, eobs_vals = x

		rv = np.zeros(sz)

		if self.has_default:
			for i in range(len(cond_vals)):
				rv[idx_vals[i]] = self.dmap.get(cond_vals[i], self.default_dist).seq_log_density(eobs_vals[i])
		else:
			for i in range(len(cond_vals)):
				if cond_vals[i] in self.dmap:
					rv[idx_vals[i]] = self.dmap[cond_vals[i]].seq_log_density(eobs_vals[i])
				else:
					rv[idx_vals[i]] = -np.inf

		return rv


	def seq_encode(self, x):
		"""Encodes an iterable of (c, y) pairs for vectorized evaluation.

		Observations are grouped by conditioning value and each group is
		encoded with its member distribution's seq_encode. Unmatched groups
		are encoded with the default distribution, or with the null
		distribution when no default is set (seq_log_density scores those
		observations -inf without touching their encoding).

		Args:
			x: Iterable of pairs whose first entry is the conditioning value.

		Returns:
			Tuple: (number of observations, conditioning values, per-group
			index arrays, per-group member encodings).
		"""
		cond_enc = dict()

		for i in range(len(x)):
			xx = x[i]
			vv = xx if self.pass_value else xx[1]
			if xx[0] not in cond_enc:
				cond_enc[xx[0]] = [[vv], [i]]
			else:
				cond_enc_loc = cond_enc[xx[0]]
				cond_enc_loc[0].append(vv)
				cond_enc_loc[1].append(i)

		cond_enc = list(cond_enc.items())

		fallback_dist = self.default_dist if self.default_dist is not None else null_dist

		cond_vals  = tuple([u[0] for u in cond_enc])
		eobs_vals  = tuple([self.dmap.get(u[0], fallback_dist).seq_encode(u[1][0]) for u in cond_enc])
		idx_vals   = tuple([np.asarray(u[1][1]) for u in cond_enc])

		return len(x), cond_vals, idx_vals, eobs_vals


	def sampler(self, seed=None):
		"""Returns a ConditionalDistributionSampler.

		The conditioning value is not modeled, so only conditional sampling via
		sample_given(x) is supported.

		Args:
			seed (Optional[int]): Random seed.

		Returns:
			ConditionalDistributionSampler object.
		"""
		return ConditionalDistributionSampler(self, seed)

	def estimator(self, pseudo_count=None):
		"""Returns a ConditionalDistributionEstimator with one member estimator
		per dmap entry (and a default estimator when a default is modeled).

		Args:
			pseudo_count (Optional[float]): Accepted for API compatibility.
				Ignored (member estimators are created with their defaults).

		Returns:
			ConditionalDistributionEstimator object.
		"""
		emap = {k: v.estimator() for k, v in self.dmap.items()}
		default_est = None if self.default_dist is None else self.default_dist.estimator()
		return ConditionalDistributionEstimator(emap, default_estimator=default_est)


class ConditionalDistributionSampler(object):
	"""Sampler for ConditionalDistribution. Supports sampling the dependent
	value given a conditioning value; unconditional sampling is undefined."""

	def __init__(self, dist, seed=None):
		"""ConditionalDistributionSampler object.

		Args:
			dist (ConditionalDistribution): Distribution to sample from.
			seed (Optional[int]): Random seed.
		"""
		self.dist = dist
		self.rng = RandomState(seed)
		self._samplers = dict()

	def sample_given(self, x, size=None):
		"""Draw dependent values from the member selected by the conditioning value.

		Args:
			x: Conditioning value (selects dmap[x], or the default distribution
				when unmatched).
			size (Optional[int]): Number of samples to draw.

		Returns:
			A sample from the selected member distribution (or a size-length
			collection of samples).

		Raises:
			KeyError: If x is unmatched and no default distribution is modeled.
		"""
		if x not in self._samplers:
			if x in self.dist.dmap:
				member = self.dist.dmap[x]
			elif self.dist.default_dist is not None:
				member = self.dist.default_dist
			else:
				raise KeyError('ConditionalDistributionSampler has no distribution for conditioning value %s and no default_dist is set.' % repr(x))
			self._samplers[x] = member.sampler(seed=self.rng.randint(maxint))

		return self._samplers[x].sample(size=size)

	def sample(self, size=None):
		"""Unconditional sampling is not defined for ConditionalDistribution.

		Raises:
			NotImplementedError: Always; the conditioning value is not modeled.
				Use sample_given(x) instead.
		"""
		raise NotImplementedError('ConditionalDistribution does not model the conditioning value, so unconditional sampling is undefined. Use ConditionalDistributionSampler.sample_given(x) with a conditioning value.')


class ConditionalDistributionEstimatorAccumulator(SequenceEncodableAccumulator):
	"""Routes each observation's sufficient statistics to the accumulator of
	its conditioning value (or to the default accumulator when unmatched)."""

	def __init__(self, accumulator_map, default_accumulator, keys=None):
		"""ConditionalDistributionEstimatorAccumulator object.

		Args:
			accumulator_map (Dict): Map from conditioning value to member accumulator.
			default_accumulator: Accumulator for unmatched conditioning values
				(None to drop them).
			keys (Optional[str]): Key for merging statistics across accumulators.
		"""
		self.accumulator_map     = accumulator_map
		self.default_accumulator = default_accumulator
		self.key                 = keys

	def update(self, x, weight, estimate):
		"""Adds a weighted pair observation to the matching member accumulator.

		Args:
			x: Pair whose first entry is the conditioning value.
			weight (float): Observation weight.
			estimate (Optional[ConditionalDistribution]): Current estimate
				passed through to the member accumulator (may be None).
		"""
		if x[0] in self.accumulator_map:
			member_estimate = None if estimate is None else estimate.dmap.get(x[0], None)
			self.accumulator_map[x[0]].update(x[1], weight, member_estimate)
		else:
			if self.default_accumulator is not None:
				member_estimate = None if estimate is None else estimate.default_dist
				self.default_accumulator.update(x[1], weight, member_estimate)

	def initialize(self, x, weight, rng):
		"""Initializes the matching member accumulator with a weighted observation.

		Args:
			x: Pair whose first entry is the conditioning value.
			weight (float): Observation weight.
			rng: Random number generator passed to the member accumulator.
		"""
		if x[0] in self.accumulator_map:
			self.accumulator_map[x[0]].initialize(x[1], weight, rng)
		else:
			if self.default_accumulator is not None:
				self.default_accumulator.initialize(x[1], weight, rng)

	def seq_initialize(self, x, weights, rng):
		"""Initializes the member accumulators with sequence-encoded observations.

		Args:
			x: Encoding from ConditionalDistribution.seq_encode.
			weights (np.ndarray): Observation weights.
			rng: Random number generator passed to the member accumulators.
		"""
		sz, cond_vals, idx_vals, eobs_vals = x

		for i in range(len(cond_vals)):
			if cond_vals[i] in self.accumulator_map:
				self.accumulator_map[cond_vals[i]].seq_initialize(eobs_vals[i], weights[idx_vals[i]], rng)
			else:
				if self.default_accumulator is not None:
					self.default_accumulator.seq_initialize(eobs_vals[i], weights[idx_vals[i]], rng)

	def seq_update(self, x, weights, estimate):
		"""Adds sequence-encoded weighted observations to the member accumulators.

		Args:
			x: Encoding from ConditionalDistribution.seq_encode.
			weights (np.ndarray): Observation weights.
			estimate (Optional[ConditionalDistribution]): Current estimate
				passed through to the member accumulators (may be None).
		"""
		sz, cond_vals, idx_vals, eobs_vals = x

		for i in range(len(cond_vals)):
			if cond_vals[i] in self.accumulator_map:
				member_estimate = None if estimate is None else estimate.dmap.get(cond_vals[i], None)
				self.accumulator_map[cond_vals[i]].seq_update(eobs_vals[i], weights[idx_vals[i]], member_estimate)
			else:
				if self.default_accumulator is not None:
					member_estimate = None if estimate is None else estimate.default_dist
					self.default_accumulator.seq_update(eobs_vals[i], weights[idx_vals[i]], member_estimate)

	def combine(self, suff_stat):
		"""Adds another accumulator's sufficient statistics to this one.

		Args:
			suff_stat: Pair (member statistic map, default statistic) as
				produced by value().

		Returns:
			ConditionalDistributionEstimatorAccumulator: This accumulator.
		"""
		for k,v in suff_stat[0].items():
			if k in self.accumulator_map:
				self.accumulator_map[k].combine(v)
			else:
				self.accumulator_map[k] = v

		if self.default_accumulator is not None and suff_stat[1] is not None:
			self.default_accumulator.combine(suff_stat[1])

		return self

	def value(self):
		"""Returns the pair (member statistic map, default statistic)."""
		rv2 = None if self.default_accumulator is None else self.default_accumulator.value()
		rv1 = {k: v.value() for k, v in self.accumulator_map.items()}
		return rv1, rv2

	def from_value(self, x):
		"""Sets the sufficient statistics from a value() pair.

		Args:
			x: Pair (member statistic map, default statistic).

		Returns:
			ConditionalDistributionEstimatorAccumulator: This accumulator.
		"""
		for k,v in x[0].items():
			self.accumulator_map[k].from_value(v)

		if self.default_accumulator is not None and x[1] is not None:
			self.default_accumulator.from_value(x[1])

		return self

	def key_merge(self, stats_dict):
		"""Delegates key merging to the member accumulators."""
		for k,v in self.accumulator_map.items():
			v.key_merge(stats_dict)

	def key_replace(self, stats_dict):
		"""Delegates key replacement to the member accumulators."""
		for k,v in self.accumulator_map.items():
			v.key_replace(stats_dict)



class ConditionalDistributionEstimator(ParameterEstimator):
	"""Estimator for ConditionalDistribution: fits one member distribution per
	conditioning value (plus an optional default for unmatched values)."""

	def __init__(self, estimator_map, default_estimator=None, keys=None):
		"""ConditionalDistributionEstimator object.

		Args:
			estimator_map (Dict): Map from conditioning value to member estimator.
			default_estimator (Optional[ParameterEstimator]): Estimator for
				unmatched conditioning values.
			keys (Optional[str]): Key for merging statistics across accumulators.
		"""
		self.estimator_map = estimator_map
		self.default_estimator = default_estimator
		self.keys = keys

	def accumulator_factory(self):
		"""Returns a factory whose make() creates an accumulator with one member
		accumulator per estimator-map entry."""
		emap_items = self.estimator_map.items()

		obj = type('', (object,), {'make': lambda o: ConditionalDistributionEstimatorAccumulator({k : v.accumulator_factory().make() for k,v in emap_items}, None if self.default_estimator is None else self.default_estimator.accumulator_factory().make(), self.keys)})()
		# def makeL():
		#	return(CompositeEstimatorAccumulator([x.accumulatorFactory().make() for x in self.estimators]))
		# obj = AccumulatorFactory(makeL)
		return (obj)

	def estimate(self, suff_stat):
		"""Estimates a ConditionalDistribution from sufficient statistics.

		Args:
			suff_stat: Pair (member statistic map, default statistic) as
				produced by the accumulator's value().

		Returns:
			ConditionalDistribution: Member distributions estimated per
			conditioning value, with the estimated default (or None).
		"""
		if self.default_estimator is not None:
			default_dist = self.default_estimator.estimate(suff_stat[1])
		else:
			default_dist = None

		dist_map = {k : self.estimator_map[k].estimate(v) for k,v in suff_stat[0].items()}


		return ConditionalDistribution(dist_map, default_dist=default_dist)

# --- API naming aliases (notes/distribution_api_naming_accounting.md) ---
ConditionalDistributionAccumulator = ConditionalDistributionEstimatorAccumulator
