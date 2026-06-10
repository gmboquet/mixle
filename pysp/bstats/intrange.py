from typing import Optional, List, Union, Tuple
from numpy.random import RandomState

import pysp.utils.vector as vec
from pysp.arithmetic import *
from pysp.bstats.pdist import ProbabilityDistribution, SequenceEncodableAccumulator, ParameterEstimator
from pysp.bstats.symdirichlet import SymmetricDirichletDistribution
from pysp.bstats.dirichlet import DirichletDistribution
import numpy as np
from scipy.special import gammaln, digamma
import scipy.sparse as sp

default_prior = DirichletDistribution(1.0 + 1.0e-12)

class IntegerCategoricalDistribution(ProbabilityDistribution):

    def __init__(self, prob_vec: Union[np.ndarray, List[float], sp.spmatrix], default_value: float = 0.0, min_index: int = 0, name: Optional[str] = None, prior: ProbabilityDistribution = default_prior):
        self.min_index = min_index
        self.name = name
        self.set_parameters(prob_vec)#, default_value, min_index))
        self.set_prior(prior)


    def __str__(self):
        pstr = ','.join(map(str, self.prob_vec))
        astr = 'default_value=%s, min_index=%s, prior=%s'%(str(self.default_value), str(self.min_index), str(self.prior))
        return 'IntegerRangeDistribution([%s], %s)' % (pstr, astr)

    def get_parameters(self):
        return self.prob_vec#, self.default_value, self.min_index

    def set_parameters(self, params):

        with np.errstate(divide='ignore'):

            self.prob_vec      = params
            #self.default_value = params[1]
            #self.min_index     = int(params[2])
            self.default_value = 0.0
            #self.min_index = 0
            self.max_index     = int(self.min_index + len(self.prob_vec) - 1)

            self.num_vals      = len(self.prob_vec)
            self.log_prob_vec  = np.log(self.prob_vec)
            self.log_default_value = np.log(self.default_value)
            self.log_const = np.log1p(1 + self.default_value)

    def set_prior(self, prior):
        self.prior = prior

        if isinstance(self.prior, DirichletDistribution):
            cpp = prior.get_parameters()
            if np.ndim(cpp) == 0:
                cpp = np.ones(self.num_vals)*cpp
            self.conj_prior_params = cpp
            self.expected_nparams = digamma(self.conj_prior_params) - digamma(np.sum(self.conj_prior_params))
        elif isinstance(self.prior, SymmetricDirichletDistribution):
            self.conj_prior_params = np.ones(self.num_vals)*prior.get_parameters()
            self.expected_nparams = digamma(self.conj_prior_params) - digamma(np.sum(self.conj_prior_params))
        else:
            self.conj_prior_params = None
            self.expected_nparams = None

    def get_prior(self):
        return self.prior

    def get_data_type(self):
        return int

    def entropy(self):
        p = self.prob_vec
        g = p > 0
        log_p = self.log_prob_vec

        return -np.dot(p[g], log_p[g])

    def cross_entropy(self, dist: ProbabilityDistribution):
        if isinstance(dist, IntegerCategoricalDistribution):
            p = self.prob_vec
            g = p > 0
            gg = np.flatnonzero(g) + (self.min_index - dist.min_index)
            log_p = dist.log_prob_vec
            return -np.dot(p[g], log_p[gg])
        else:
            rv = 0
            for x,p in enumerate(self.prob_vec):
                if p > 0:
                    rv += dist.log_density(x+self.min_index) * p
            return -rv

    def moment(self, p, o=0):
        return np.dot(np.power(np.arange(self.min_index, self.max_index+1)-o, p), self.prob_vec)

    def log_density(self, x: int) -> float:

        if (x < self.min_index) or (x > self.max_index):
            return self.log_default_value
        else:
            return self.log_prob_vec[x - self.min_index] - self.log_const

    def expected_log_density(self, x: int) -> float:

        # E[ params ]*x
        #e_x = digamma(self.conj_prior_params[idx]) - digamma(np.sum(self.conj_prior_params))
        # E[ A(params) ] = 0
        # E[ ln(B(x)) ] = 0

        if (x < self.min_index) or (x > self.max_index):
            return self.log_default_value - self.log_const
        else:
            idx = int(x - self.min_index)
            return self.expected_nparams[idx] - self.log_const

    def seq_log_density(self, x):

        v  = x - self.min_index
        u  = np.bitwise_and(v >= 0, v < self.num_vals)
        rv = np.zeros(len(x))
        rv.fill(self.log_default_value)
        rv[u] = self.log_prob_vec[v[u]]
        rv -= self.log_const

        return rv

    def seq_expected_log_density(self, x):
        idx = x - self.min_index
        in_range = np.bitwise_and(idx >= 0, idx < self.num_vals)
        rv = np.zeros(len(x))
        rv.fill(self.log_default_value)
        rv[in_range] = self.expected_nparams[idx[in_range]]
        rv -= self.log_const
        return rv

    def seq_encode(self, x):
        return np.asarray(x, dtype=int)

    def sampler(self, seed: Optional[int] = None):
        return IntegerCategoricalSampler(self, seed)

    def estimator(self):
        return IntegerCategoricalEstimator(name=self.name, prior=self.prior)



class IntegerCategoricalSampler(object):

    def __init__(self, dist: IntegerCategoricalDistribution, seed: Optional[int] = None):
        self.rng  = RandomState(seed)
        self.dist = dist

    def sample(self, size=None):

        if size is None:
            return self.rng.choice(range(self.dist.min_index, self.dist.max_index+1), p=self.dist.prob_vec)
        else:
            return self.rng.choice(range(self.dist.min_index, self.dist.max_index + 1), p=self.dist.prob_vec, size=size).tolist()


class IntegerCategoricalAccumulator(SequenceEncodableAccumulator):

    def __init__(self, min_val: int, max_val: int, keys: Tuple[str,]):

        self.minVal = min_val
        self.maxVal = max_val

        if min_val is not None and max_val is not None:
            self.countVec = vec.zeros(max_val-min_val+1)
        else:
            self.countVec = None

        self.key = keys[0]

    def update(self, x, weight, estimate):

        if self.countVec is None:
            self.minVal   = x
            self.maxVal   = x
            self.countVec = vec.make([weight])

        elif self.maxVal < x:
            tempVec = self.countVec
            self.maxVal   = x
            self.countVec = vec.zeros(self.maxVal - self.minVal + 1)
            self.countVec[:len(tempVec)] = tempVec
            self.countVec[x-self.minVal] += weight
        elif self.minVal > x:
            tempVec  = self.countVec
            tempDiff = self.minVal - x
            self.minVal   = x
            self.countVec = vec.zeros(self.maxVal - self.minVal + 1)
            self.countVec[tempDiff:] = tempVec
            self.countVec[x-self.minVal] += weight
        else:
            self.countVec[x-self.minVal] += weight


    def seq_update(self, x, weights, estimate):

        min_x = x.min()
        max_x = x.max()

        loc_cnt = np.bincount(x-min_x, weights=weights)

        if self.countVec is None:
            self.countVec = np.zeros(max_x-min_x+1)
            self.minVal = min_x
            self.maxVal = max_x

        if self.minVal > min_x or self.maxVal < max_x:
            prev_min    = self.minVal
            self.minVal = min(min_x, self.minVal)
            self.maxVal = max(max_x, self.maxVal)
            temp        = self.countVec
            prev_diff   = prev_min - self.minVal
            self.countVec = np.zeros(self.maxVal - self.minVal + 1)
            self.countVec[prev_diff:(prev_diff + len(temp))] = temp

        min_diff = min_x - self.minVal
        self.countVec[min_diff:(min_diff+len(loc_cnt))] += loc_cnt

    def seq_initialize(self, x, weights, rng):
        self.seq_update(x, weights, None)

    def combine(self, suff_stat):

        if self.countVec is None and suff_stat[1] is not None:
            self.minVal   = suff_stat[0]
            self.maxVal   = suff_stat[0] + len(suff_stat[1]) - 1
            self.countVec = suff_stat[1]

        elif self.countVec is not None and suff_stat[1] is not None:

            if self.minVal == suff_stat[0] and len(self.countVec) == len(suff_stat[1]):
                self.countVec += suff_stat[1]

            else:
                minVal = min(self.minVal, suff_stat[0])
                maxVal = max(self.maxVal, suff_stat[0] + len(suff_stat[1]) - 1)

                countVec = vec.zeros(maxVal-minVal+1)

                i0 = self.minVal - minVal
                i1 = self.maxVal - minVal + 1
                countVec[i0:i1] = self.countVec

                i0 = suff_stat[0] - minVal
                i1 = (suff_stat[0] + len(suff_stat[1]) - 1) - minVal + 1
                countVec[i0:i1] += suff_stat[1]

                self.minVal   = minVal
                self.maxVal   = maxVal
                self.countVec = countVec

        return self

    def value(self):
        return self.minVal, self.countVec

    def from_value(self, x):
        self.minVal   = x[0]
        self.maxVal   = x[0] + len(x[1]) - 1
        self.countVec = x[1]


    def key_merge(self, stats_dict):
        if self.key is not None:
            if self.key in stats_dict:
                stats_dict[self.key].combine(self.value())
            else:
                stats_dict[self.key] = self

    def key_replace(self, stats_dict):
        if self.key is not None:
            if self.key in stats_dict:
                self.from_value(stats_dict[self.key].value())


class IntegerCategoricalAccumulatorFactory(object):

    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

    def make(self):
        return IntegerCategoricalAccumulator(min_val=self.min_val, max_val=self.max_val)


class IntegerCategoricalEstimator(object):

    def __init__(self, min_index: Optional[int] = None, max_index: Optional[int] = None, default_value: float = 0.0, name: Optional[str] = None, prior: ProbabilityDistribution = default_prior, keys: Tuple[Optional[str], ] = (None,)):

        self.minVal        = min_index
        self.maxVal        = max_index
        self.default_value = default_value
        self.keys          = keys
        self.name          = name

        self.set_prior(prior)

    def get_prior(self):
        return self.prior

    def set_prior(self, prior):
        self.prior = prior

        if isinstance(self.prior, (DirichletDistribution, SymmetricDirichletDistribution)):
            self.has_conj_prior = True
        else:
            self.has_conj_prior = False


    def accumulator_factory(self):

        minVal = self.minVal
        maxVal = self.maxVal

        obj = type('', (object,), {'make': lambda o: IntegerCategoricalAccumulator(minVal, maxVal, self.keys)})()
        return obj

    def estimate(self, suff_stat):


        if self.has_conj_prior:

            min_val, count_vec = suff_stat
            alpha0 = self.prior.get_parameters()
            #alpha = count_vec + alpha0 - 1
            alpha = count_vec + (alpha0 - 1)
            prob_vec = alpha / np.sum(alpha)


            hyper_posterior = DirichletDistribution(alpha + 1)

            return IntegerCategoricalDistribution(prob_vec, min_index=min_val, default_value=self.default_value, name=self.name, prior=hyper_posterior)

        else:

            return IntegerCategoricalDistribution(suff_stat[0], suff_stat[1] / (suff_stat[1].sum()), name=self.name, prior=self.prior)
