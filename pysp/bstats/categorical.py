from typing import Optional, Union, List, Any, Dict, Tuple, NoReturn

from pysp.arithmetic import *
from pysp.bstats.pdist import SequenceEncodableAccumulator, ParameterEstimator, DataFrameEncodableAccumulator, ProbabilityDistribution
from pysp.bstats.catdirichlet import DictDirichletDistribution
from pysp.bstats.symdirichlet import SymmetricDirichletDistribution
from pysp.bstats.dirichlet import DirichletDistribution

from collections import defaultdict
from pysp.bstats.nulldist import null_dist
import numpy as np
from scipy.special import digamma


default_prior = DictDirichletDistribution(1.0 + 1.0e-12)

class CategoricalDistribution(ProbabilityDistribution):

	def __init__(self, prob_map: Dict[Any, float], default_value: float = 0.0, name: Optional[str] = None, prior: Optional[ProbabilityDistribution] = default_prior):

		with np.errstate(divide='ignore'):
			self.prob_map = prob_map
			#self.prob_vec = np.asarray(u[1] for u in prob_list)
			self.name     = name

			self.default_value = default_value
			self.log_default_value = np.log(default_value)
			self.log1p_default_value = np.log1p(default_value)
			self.set_prior(prior)

	def __str__(self):
		return 'CategoricalDistribution(%s, default_value=%s, name=%s, prior=%s)' % (str(self.prob_map), str(self.default_value), str(self.name), str(self.prior))

	def get_parameters(self):
		return self.prob_map

	def set_parameters(self, params):
		self.prob_map = params

	def get_prior(self) -> ProbabilityDistribution:
		return self.prior

	def set_prior(self, prior: ProbabilityDistribution):
		self.prior = prior

		if isinstance(prior, DictDirichletDistribution):
			a = self.prior.get_parameters()
			n = len(self.prob_map)
			if isinstance(a, float):
				bb = digamma(a) - digamma(n*a)
				b = {k: bb for k in self.prob_map.keys()}
			else:
				b = digamma(sum(a.values()))
				b = {k: digamma(v) - b for k, v in a.items()}
			self.conj_prior_params = a
			self.expected_nparams  = b
			self.has_conj_prior    = True

		else:
			self.conj_prior_params = None
			self.expected_nparams  = None
			self.has_conj_prior    = False

	def entropy(self) -> float:
		rv = 0.0
		for v in self.prob_map.values():
			if v > 0:
				rv += np.log(v)*v
		return rv

	def log_density(self, x) -> float:
		return np.log(self.prob_map.get(x, self.default_value)) - self.log1p_default_value

	def expected_log_density(self, x) -> float:

		if not self.has_conj_prior:
			return self.log_density(x)

		if x not in self.prob_map:
			return self.log_default_value - self.log1p_default_value

		if self.has_conj_prior:
			return self.expected_nparams[x] - self.log1p_default_value

	def seq_log_density(self, x) -> float:
		xs, val_map_inv = x
		mapped_probs = np.log([self.prob_map.get(u,self.default_value) for u in val_map_inv])

		return mapped_probs[xs]

	def seq_expected_log_density(self, x):
		xs, val_map_inv = x
		rv = np.asarray([self.expected_log_density(u) for u in val_map_inv])

		return rv[xs]

	def seq_encode(self, x):
		val_map_inv, xs = np.unique(x, return_inverse=True)
		return xs, val_map_inv

	def sampler(self, seed=None):
		return CategoricalSampler(self, seed)

	def estimator(self):
		return CategoricalEstimator(name=self.name, prior=self.prior)



class CategoricalSampler(object):

	def __init__(self, dist, seed=None):
		self.rng = np.random.RandomState(seed)

		temp = dist.prob_map.items()
		self.levels = [u[0] for u in temp]
		self.probs  = [u[1] for u in temp]
		self.num_levels = len(self.levels)

	def sample(self, size=None):

		if size is None:
			return self.rng.choice(self.levels, p=self.probs, size=size)

		else:
			levels = self.levels
			rv = self.rng.choice(self.num_levels, p=self.probs, size=size)
			return [levels[i] for i in rv]


class CategoricalEstimatorAccumulator(SequenceEncodableAccumulator, DataFrameEncodableAccumulator):

	def __init__(self, name, keys):
		self.name = name
		self.key  = keys[0]
		self.count_map = defaultdict(float)
		self.count_sum = 0.0

	def update(self, x, weight, estimate):
		self.count_map[x] += weight
		self.count_sum += weight

	def initialize(self, x, weight, rng):
		self.update(x, weight, None)

	def seq_initialize(self, x, weights, rng):
		inv_key_map = x[1]
		bcnt = np.bincount(x[0], weights=weights)
		self.count_sum += np.sum(bcnt)
		for i in range(0, len(bcnt)):
			self.count_map[inv_key_map[i]] += bcnt[i]

	def seq_update(self, x, weights, estimate):
		inv_key_map = x[1]
		bcnt = np.bincount(x[0], weights=weights)
		self.count_sum += np.sum(bcnt)
		for i in range(0, len(bcnt)):
			self.count_map[inv_key_map[i]] += bcnt[i]

	def df_initialize(self, df, weights, rng):
		self.df_update(df, weights, None)

	def df_update(self, df, weights, estimate):
		gb = df.groupby([self.name])
		for k,idx in gb.indices:
			loc_sum = np.sum(weights[idx])
			self.count_map[k] += loc_sum
			self.count_sum += loc_sum

	def combine(self, suff_stat):
		self.count_sum += suff_stat[1]
		for item in suff_stat[0].items():
			self.count_map[item[0]] = self.count_map.get(item[0], 0.0) + item[1]
		return self

	def value(self):
		return self.count_map, self.count_sum

	def from_value(self, x):
		self.count_map = x[0]
		self.count_sum = x[1]
		return self


class CategoricalEstimatorAccumulatorFactory(object):

	def __init__(self, name, keys):
		self.name = name
		self.keys = keys

	def make(self):
		return CategoricalEstimatorAccumulator(self.name, self.keys)


class CategoricalEstimator(ParameterEstimator):

	def __init__(self, default_value: float = 0.0, name=None, prior=default_prior, keys=(None,)):
		self.keys = keys
		self.name = name
		self.prior = prior
		self.default_value = default_value

	def accumulator_factory(self):
		return CategoricalEstimatorAccumulatorFactory(self.name, self.keys)

	def get_prior(self):
		return self.prior

	def set_prior(self, prior):
		self.prior = prior

	def estimate(self, suff_stat):


		count_map, stats_sum = suff_stat
		stats_sum = sum(count_map.values())

		#if self.default_value:
		#	if stats_sum > 0:
		#		default_value = 1.0/stats_sum
		#		default_value *= default_value
		#	else:
		#		default_value = 0.5
		#else:
		#	default_value = 0.0
		default_value = self.default_value

		if isinstance(self.prior, DictDirichletDistribution):

			conj_prior_params = self.prior.get_parameters()

			if isinstance(conj_prior_params, float):
				alpha = conj_prior_params

				keys = count_map.keys()
				# Dirichlet MAP sits on the boundary when alpha_k + n_k < 1
				num  = {k: max((alpha-1) + count_map[k], 0.0) for k in keys}
				cpp  = {k: (alpha + count_map[k]) for k in keys}
			else:
				alpha_sum = sum(conj_prior_params.values())

				keys = set(conj_prior_params.keys()).union(count_map.keys())
				num  = {k: max((conj_prior_params.get(k, 0.0)-1) + count_map.get(k, 0.0), 0.0) for k in keys}
				cpp  = {k: (conj_prior_params.get(k, 0.0) + count_map.get(k, 0.0)) for k in keys}

			norm_const = sum(num.values())

			if norm_const > 0:
				pMap = {k: v/norm_const for k, v in num.items()}
			else:
				# fall back to the posterior mean when the MAP is degenerate
				cpp_sum = sum(cpp.values())
				pMap = {k: v/cpp_sum for k, v in cpp.items()}

			return CategoricalDistribution(pMap, default_value=default_value, name=self.name, prior=DictDirichletDistribution(cpp))

		else:

			nobs_loc = stats_sum

			if nobs_loc == 0:
				pMap = {k: 1.0 / float(len(count_map)) for k in count_map.keys()}
			else:
				pMap = {k: v / nobs_loc for k, v in count_map.items()}


			return CategoricalDistribution(pMap, default_value=default_value, name=self.name)