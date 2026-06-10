from typing import Optional
from pysp.bstats.pdist import ProbabilityDistribution, StatisticAccumulator, ParameterEstimator
from pysp.bstats.beta import BetaDistribution
from pysp.bstats.nulldist import NullDistribution, null_dist
from numpy.random import RandomState
from scipy.special import digamma
import mpmath
import numpy as np

default_prior = BetaDistribution(1.0, 1.0)

class GeometricDistribution(ProbabilityDistribution):

    def __init__(self, p: float, name: Optional[str] = None, prior: ProbabilityDistribution = default_prior, keys: Optional[str] = None):
        self.parents = []
        self.p      = p
        self.log_p  = np.log(p)
        self.log_1p = np.log1p(-p)
        self.name = name
        self.keys = keys
        self.set_prior(prior)

    def __str__(self):
        return 'GeometricDistribution(%f, prior=%s, keys=%s, name=%s)'%(self.p, str(self.prior), str(self.keys), str(self.name))

    def set_prior(self, dist: ProbabilityDistribution) -> None:
        self.prior = dist

        if isinstance(dist, BetaDistribution):
            self.has_conj_prior = True
            self.conj_prior_params = (digamma(dist.a), digamma(dist.b), digamma(dist.a + dist.b))
        else:
            self.has_conj_prior = False
            self.conj_prior_params = (0, 0, 0)

    def get_prior(self) -> ProbabilityDistribution:
        return self.prior

    def get_parameters(self) -> float:
        return self.p

    def set_parameters(self, params: float) -> None:
        self.p = params

    def entropy(self) -> float:
        p = self.p
        return -(np.log1p(-p)*(1-p)/p + np.log(p))

    def cross_entropy(self, dist) -> float:
        if isinstance(dist, GeometricDistribution):
            pp = dist.p
            p = self.p
            return -(np.log1p(-pp)*(1-p)/p + np.log(pp))
        else:
            return None

    def moment(self, p):
        p_loc = self.p
        if p == 1:
            return 1.0/p_loc
        elif p == 2:
            return (2-p_loc)/(p_loc*p_loc)
        else:
            aa = mpmath.polylog(-1, 1 - p)
            return float(mpmath.nstr(aa, 12)) * p / (1 - p)

    def density(self, x: int) -> float:
        return np.exp(self.log_density(x))

    def log_density(self, x: int) -> float:
        if self.p == 1.0:
            return 0.0 if x == 1 else -np.inf
        elif self.p == 0.0:
            return -np.inf
        else:
            return (x-1)*self.log_1p + self.log_p

    def expected_log_density(self, x: int) -> float:
        if self.has_conj_prior:
            ga, gb, gab = self.conj_prior_params
            if x < 1:
                return -np.inf
            else:
                return (gb-gab)*(x-1) + (ga - gab)
        else:
            return self.log_density(x)

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        rv = x-1
        rv *= self.log_1p
        rv += self.log_p
        return rv

    def seq_expected_log_density(self, x: np.ndarray) -> np.ndarray:

        if self.has_conj_prior:
            ga, gb, gab = self.conj_prior_params
            rv = x-1
            rv *= (gb - gab)
            rv += (ga - gab)
            rv[x < 1] = -np.inf
            return rv
        else:
            return self.seq_log_density(x)


    def seq_encode(self, x):
        rv = np.asarray(x, dtype=float)
        return rv

    def sampler(self, seed=None):
        return GeometricSampler(self, seed)

    def estimator(self, pseudo_count=None):
        return GeometricEstimator(name=self.name, keys=self.keys, prior=self.prior)


class GeometricSampler(object):

    def __init__(self, dist, seed=None):
        self.rng  = RandomState(seed)
        self.dist = dist

    def sample(self, size=None):
        return self.rng.geometric(p=self.dist.p, size=size)


class GeometricAccumulator(StatisticAccumulator):

    def __init__(self, keys, name):
        self.sum   = 0.0
        self.count = 0.0
        self.key   = keys
        self.name  = name

    def update(self, x, weight, estimate):
        if x >= 0:
            self.sum  += x*weight
            self.count += weight

    def seq_update(self, x, weights, estimate):
        self.sum += np.dot(x, weights)
        self.count += np.sum(weights)

    def initialize(self, x, weight, rng):
        self.update(x, weight, None)

    def seq_initialize(self, x, weights, rng):
        self.seq_update(x, weights, None)

    def combine(self, suff_stat):
        self.sum   += suff_stat[1]
        self.count += suff_stat[0]
        return self

    def value(self):
        return self.count, self.sum

    def from_value(self, x):
        self.count = x[0]
        self.sum = x[1]
        return self

    def key_merge(self, stats_dict):
        if self.key is not None:
            if self.key in stats_dict:
                vals = stats_dict[self.key]
                stats_dict[self.key] = (vals[0] + self.count, vals[1] + self.sum)
            else:
                stats_dict[self.key] = (self.count, self.sum)

    def key_replace(self, stats_dict):
        if self.key is not None:
            if self.key in stats_dict:
                vals = stats_dict[self.key]
                self.count = vals[0]
                self.sum   = vals[1]

class GeometricAccumulatorFactory():
    def __init__(self, keys, name):
        self.keys = keys
        self.name = name

    def make(self):
        return GeometricAccumulator(self.keys, self.name)

class GeometricEstimator(ParameterEstimator):

    name: Optional[str]
    has_conj_prior: bool
    has_prior: bool
    keys: Optional[str]
    prior: ProbabilityDistribution

    def __init__(self, name: Optional[str] = None, keys: Optional[str] = None, prior: ProbabilityDistribution = default_prior):

        self.keys  = keys
        self.name = name
        self.set_prior(prior)

    def accumulator_factory(self):
        return GeometricAccumulatorFactory(self.keys, self.name)

    def get_prior(self) -> ProbabilityDistribution:
        return self.prior

    def set_prior(self, prior) -> None:
        self.prior = prior
        if isinstance(prior, BetaDistribution):
            self.has_conj_prior = True
            self.has_prior = True
        elif isinstance(prior, NullDistribution) or prior is None:
            self.has_conj_prior = False
            self.has_prior = False
        else:
            self.has_conj_prior = False
            self.has_prior = True

    def estimate(self, suff_stat: (float, float)) -> GeometricDistribution:

        ocnt, osum = suff_stat

        if self.has_conj_prior:

            old_a = self.prior.a
            old_b = self.prior.b

            a = old_a + ocnt
            b = old_b + osum - ocnt

            if a > 1 and b > 1:
                p = (a-1)/(a+b-2)
            elif a <= 1 and b > 1:
                p = 0.0
            elif a > 1 and b <= 1:
                p = 1.0
            else:
                p = 0.5

            return GeometricDistribution(p, name=self.name, prior=BetaDistribution(a,b), keys=self.keys)

        else:

            return GeometricDistribution(ocnt/osum, name=self.name, keys=self.keys)


if __name__ == '__main__':

    dist = GeometricDistribution(0.2)

    data = dist.sampler(seed=1).sample(100)
    enc_data = dist.seq_encode(data)

    est = GeometricEstimator()
    acc = est.accumulator_factory().make()
    acc.seq_update(enc_data, np.ones(len(data)), None)
    model = est.estimate(acc.value())

    print(str(model))