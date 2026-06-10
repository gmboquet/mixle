from typing import Optional, Any, Dict, Union
from numpy.random import RandomState
from pysp.bstats.pdist import ParameterEstimator, ProbabilityDistribution, StatisticAccumulator
from pysp.bstats.beta import BetaDistribution
from pysp.bstats.nulldist import NullDistribution, null_dist
import numpy as np
from scipy.special import gammaln, digamma, exp1
from scipy.optimize import minimize_scalar
import scipy.integrate


default_prior = BetaDistribution(1.000001, 1.000001)


class BernoulliDistribution(ProbabilityDistribution):

    def __init__(self, p: float, name: Optional[str] = None, prior: ProbabilityDistribution = default_prior, keys: Optional[str] = None):

        self.p = p
        self.log_p0 = np.log(p)
        self.log_p1 = np.log1p(-p)

        self.name = name
        self.keys = keys
        self.set_parameters(p)
        self.set_prior(prior)

    def __str__(self) -> str:
        return 'BernoulliDistribution(%s, name=%s, prior=%s, keys=%s)' % (repr(self.p), repr(self.name), str(self.prior), repr(self.keys))

    def get_parameters(self) -> float:
        return self.p

    def set_parameters(self, params: float) -> None:
        self.p = params
        self.log_p0 = np.log(params)
        self.log_p1 = np.log1p(-params)

    def get_prior(self) -> ProbabilityDistribution:
        return self.prior

    def set_prior(self, prior: ProbabilityDistribution):
        self.prior = prior

        if isinstance(prior, BetaDistribution):
            a, b = self.prior.get_parameters()
            self.conj_prior_params = (digamma(a), digamma(b), digamma(a+b))
            self.has_conj_prior = True
            self.has_prior = True
        elif isinstance(prior, NullDistribution) or prior is None:
            self.has_prior = False
        else:
            self.conj_prior_params = None
            self.has_conj_prior = False
            self.has_prior = True

    def get_data_type(self):
        return bool

    def density(self, x: bool) -> float:
        return np.exp(self.log_density(x))

    def log_density(self, x: bool) -> float:
        if x:
            return self.log_p0
        else:
            return self.log_p1

    def expected_log_density(self, x: bool) -> float:
        if self.has_conj_prior:
            da, db, dab = self.conj_prior_params
            if x:
                return da - dab
            else:
                return db - dab
        else:
            return self.log_density(x)

    def cross_entropy(self, dist: ProbabilityDistribution) -> float:
        a = dist.log_density(True)
        b = dist.log_density(False)
        return (a-b)*self.p + b

    def entropy(self) -> float:
        return self.p * (self.log_p0 - self.log_p1) + self.log_p1

    def moment(self, p: int) -> float:
        # X takes values in {0, 1}, so E[X^p] = P(X=1) for any p >= 1
        return 1.0 if p == 0 else self.p

    def seq_log_density(self, x):
        return np.where(x, self.log_p0, self.log_p1)

    def seq_expected_log_density(self, x):
        da, db, dab = self.conj_prior_params
        return np.where(x, da - dab, db - dab)

    def seq_encode(self, x):
        return np.asarray(x, dtype=bool)

    def sampler(self, seed: Optional[int] = None):
        return BernoulliSampler(self, seed)

    def estimator(self):
        return BernoulliEstimator(name=self.name, keys=self.keys, prior=self.prior)


class BernoulliSampler(object):

    def __init__(self, dist, seed=None):
        self.rng  = np.random.RandomState(seed)
        self.dist = dist

    def sample(self, size=None):
        if size is None:
            return self.rng.rand() < self.dist.p
        else:
            return (self.rng.rand(size) < self.dist.p).tolist()


class BernoulliEstimatorAccumulator(StatisticAccumulator):

    def __init__(self, name, keys):
        self.name  = name
        self.key   = keys
        self.psum  = 0.0
        self.nsum  = 0.0
        self.count = 0.0

    def initialize(self, x, weight, rng):
        self.update(x, weight, None)

    def seq_initialize(self, x, weights, rng):
        self.seq_update(x, weights, None)

    def update(self, x, weight, estimate):
        if x:
            self.psum += weight
        else:
            self.nsum += weight

    def seq_update(self, x, weights, estimate):
        n = weights.sum()
        p = weights[x].sum()
        self.psum += p
        self.nsum += n - p

    def combine(self, suff_stat):
        self.psum += suff_stat[0]
        self.nsum += suff_stat[1]
        return self

    def value(self):
        return self.psum, self.nsum

    def from_value(self, x):
        self.psum = x[0]
        self.nsum = x[1]

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


class BernoulliEstimatorAccumulatorFactory(object):

    def __init__(self, name, keys):
        self.name = name
        self.keys = keys

    def make(self):
        return BernoulliEstimatorAccumulator(self.name, self.keys)


class BernoulliEstimator(ParameterEstimator):

    def __init__(self, name: Optional[str] = None, keys: Optional[str] = None, prior: ProbabilityDistribution = default_prior):

        self.prior = prior
        self.name  = name
        self.keys  = keys
        self.has_conj_prior = isinstance(prior, BetaDistribution)
        self.has_prior = not isinstance(prior, NullDistribution) and prior is not None

    def accumulator_factory(self) -> BernoulliEstimatorAccumulatorFactory:
        return BernoulliEstimatorAccumulatorFactory(self.name, self.keys)

    def set_prior(self, prior) -> None:
        self.prior = prior
        self.has_conj_prior = isinstance(prior, BetaDistribution)
        self.has_prior = not isinstance(prior, NullDistribution) and prior is not None

    def get_prior(self) -> ProbabilityDistribution:
        return self.prior

    def estimate(self, suff_stat: (float, float)) -> BernoulliDistribution:

        psum, nsum = suff_stat

        if self.has_conj_prior:
            a, b = self.prior.get_parameters()
            new_a = a + psum
            new_b = b + nsum
            p = (psum + a - 1.0)/(psum + nsum + a + b - 2.0)
            return BernoulliDistribution(p, name=self.name, prior=BetaDistribution(new_a, new_b), keys=self.keys)

        elif self.has_prior:

            ll_fun = lambda x: np.log(x)*psum + np.log1p(-x)*nsum + self.prior.log_density(x)
            sol = minimize_scalar(ll_fun)

        else:
            return BernoulliDistribution(psum/(psum + nsum), name=self.name, prior=null_dist, keys=self.keys)
