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

	def __init__(self, dist: ProbabilityDistribution[X,P1,V1], len_dist: ProbabilityDistribution[int,P2,V2] = null_dist, name: Optional[str] = None, len_normalized: bool = False):
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
		return self.dist.get_parameters(), self.len_dist.get_parameters()

	def set_parameters(self, params: Tuple[P1,P2]) -> None:
		self.dist.set_parameters(params[0])
		self.len_dist.set_parameters(params[1])

	def get_prior(self):
		return CompositeDistribution((self.dist.get_prior(), self.len_dist.get_prior()))

	def set_prior(self, prior):
		self.dist.set_prior(prior.dists[0])
		self.len_dist.set_prior(prior.dists[1])

	def cross_entropy(self, dist: ProbabilityDistribution):
		if isinstance(dist, SequenceDistribution):
			v1 = self.dist.cross_entropy(dist.dist)
			v2 = self.len_dist.cross_entropy(dist.len_dist)
			v3 = self.len_dist.moment(1)
			return v3*v1 + v2
		else:
			raise NotImplementedError('SequenceDistribution.cross_entropy is only implemented for SequenceDistribution arguments (got %s).' % type(dist).__name__)

	def entropy(self):
		v1 = self.dist.entropy()
		v2 = self.len_dist.entropy()
		v3 = self.len_dist.moment(1)
		return v3 * v1 + v2

	def density(self, x) -> float:
		rv = 1.0
		for i in range(len(x)):
			rv *= self.dist.density(x[i])

		if self.len_normalized and len(x) > 0:
			rv = np.power(rv, 1.0/len(x))

		return rv

	def log_density(self, x) -> float:
		rv = 0.0
		for i in range(len(x)):
			rv += self.dist.log_density(x[i])

		if self.len_normalized and len(x) > 0:
			rv /= len(x)

		rv += self.len_dist.log_density(len(x))

		return rv

	def expected_log_density(self, x) -> float:
		rv = 0.0
		for i in range(len(x)):
			rv += self.dist.expected_log_density(x[i])

		if self.len_normalized and len(x) > 0:
			rv /= len(x)

		if self.len_dist is not None:
			rv += self.len_dist.expected_log_density(len(x))

		return rv

	def seq_log_density(self, x) -> float:

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
		return SequenceSampler(self, seed)

	def estimator(self):
		return SequenceEstimator(self.dist.estimator(), self.len_dist.estimator(), name=self.name)



class SequenceSampler(object):
	def __init__(self, dist, seed=None):
		self.dist        = dist
		self.rng         = RandomState(seed)
		self.distSampler = self.dist.dist.sampler(seed=self.rng.randint(maxint))
		self.lenSampler  = self.dist.len_dist.sampler(seed=self.rng.randint(maxint))

	def sample(self, size=None):

		if size is None:
			n = self.lenSampler.sample()
			return [self.distSampler.sample() for i in range(n)]
		else:
			return [self.sample() for i in range(size)]


class SequenceEstimatorAccumulator(SequenceEncodableAccumulator):

	def __init__(self, accumulator, len_normalized, len_accumulator, keys):
		self.accumulator = accumulator
		self.len_accumulator = len_accumulator
		self.dist_key = keys[0]
		self.len_key = keys[1]
		self.len_normalized = len_normalized

	def update(self, x, weight, estimate):

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

		if len(x) > 0:
			w = weight/len(x) if self.len_normalized else weight
			for xx in x:
				self.accumulator.initialize(xx, w, rng)

		if self.len_accumulator is not None:
			self.len_accumulator.initialize(len(x), weight, rng)


	def combine(self, suff_stat):
		self.accumulator.combine(suff_stat[0])
		if self.len_accumulator is not None:
			self.len_accumulator.combine(suff_stat[1])
		return self

	def value(self):
		if self.len_accumulator is not None:
			return self.accumulator.value(), self.len_accumulator.value()
		else:
			return self.accumulator.value(), None

	def from_value(self, x):
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
		idx, icnt, inz, enc_seq, enc_nseq = x

		w = weights[idx]*icnt[idx] if self.len_normalized else weights[idx]

		self.accumulator.seq_initialize(enc_seq, w, rng)

		if self.len_accumulator is not None:
			self.len_accumulator.seq_initialize(enc_nseq, weights, rng)

	def seq_update(self, x, weights, estimate):
		idx, icnt, inz, enc_seq, enc_nseq = x

		w = weights[idx]*icnt[idx] if self.len_normalized else weights[idx]

		self.accumulator.seq_update(enc_seq, w, estimate.dist)

		if self.len_accumulator is not None:
			self.len_accumulator.seq_update(enc_nseq, weights, estimate.len_dist)

	def key_merge(self, stats_dict):

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

		if self.dist_key is not None:
			if self.dist_key in stats_dict:
				self.from_value(stats_dict[self.dist_key].value())
		if self.len_key is not None:
			if self.len_key in stats_dict:
				self.from_value(stats_dict[self.len_key].value())

		self.accumulator.key_replace(stats_dict)

		if self.len_accumulator is not None:
			self.len_accumulator.key_replace(stats_dict)


class SequenceEstimator(ParameterEstimator):

	def __init__(self, estimator: ParameterEstimator, len_estimator: ParameterEstimator = null_estimator, len_normalized=False, name=None, keys: Tuple[Optional[str], Optional[str]] = (None, None)):
		self.name = name
		self.estimator = estimator
		self.len_estimator = len_estimator
		self.keys = keys
		self.len_normalized=len_normalized

	def get_prior(self):
		return CompositeDistribution((self.estimator.get_prior(), self.len_estimator.get_prior()))

	def set_prior(self, prior):
		self.dist.set_prior(prior.dists[0])
		self.len_estimator.set_prior(prior.dists[1])

	def model_log_density(self, model: CompositeDistribution) -> float:
		prior = self.get_prior()
		params = model.get_parameters()

		return prior.log_density(params)

	def accumulator_factory(self):

		if self.len_estimator is None:
			obj = type('', (object,),{'make': lambda o: SequenceEstimatorAccumulator(self.estimator.accumulator_factory().make(), self.len_normalized, self.keys)})()
		else:
			obj = type('', (object,),{'make': lambda o: SequenceEstimatorAccumulator(self.estimator.accumulator_factory().make(), self.len_normalized, self.len_estimator.accumulator_factory().make(), self.keys)})()

		return obj

	def estimate(self, suff_stat):

		if self.len_estimator is None:
			return SequenceDistribution(self.estimator.estimate(suff_stat[0]), None, len_normalized=self.len_normalized)
		else:
			return SequenceDistribution(self.estimator.estimate(suff_stat[0]), self.len_estimator.estimate(suff_stat[1]), len_normalized=self.len_normalized)

