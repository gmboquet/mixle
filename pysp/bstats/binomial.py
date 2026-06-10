"""Binomial distribution with conjugate Beta prior on the success probability.

The number of trials n is treated as known. With prior p ~ Beta(a, b) and
observations x_1..x_m, the posterior is Beta(a + sum x_i, b + sum (n - x_i)).
Estimation returns the posterior mode (MAP) for p, carrying the posterior as
the new prior, matching the conventions of the other pysp.bstats modules.
"""
from typing import Optional, Tuple

from numpy.random import RandomState
from scipy.special import gammaln, digamma
import numpy as np

from pysp.bstats.pdist import ProbabilityDistribution, StatisticAccumulator, ParameterEstimator
from pysp.bstats.beta import BetaDistribution

default_prior = BetaDistribution(1.0001, 1.0001)


class BinomialDistribution(ProbabilityDistribution):

    def __init__(self, n: int, p: float, name: Optional[str] = None,
                 prior: ProbabilityDistribution = default_prior, keys: Optional[str] = None):

        assert 0.0 <= p <= 1.0 and n >= 1
        self.name = name
        self.keys = keys
        self.n = int(n)
        self.set_parameters(p)
        self.set_prior(prior)

    def __str__(self):
        return 'BinomialDistribution(%d, %f, name=%s, prior=%s, keys=%s)' % (
            self.n, self.p, str(self.name), str(self.prior), str(self.keys))

    def get_parameters(self) -> float:
        return self.p

    def set_parameters(self, params: float) -> None:
        self.p = params
        self.log_p = np.log(params) if params > 0 else -np.inf
        self.log_1p = np.log1p(-params) if params < 1 else -np.inf

    def get_prior(self) -> ProbabilityDistribution:
        return self.prior

    def set_prior(self, prior: ProbabilityDistribution) -> None:
        self.prior = prior

        if isinstance(prior, BetaDistribution):
            a, b = prior.get_parameters()
            self.conj_prior_params = (a, b)
            # E[log p] and E[log(1-p)] under the Beta prior
            self.expected_nparams = (digamma(a) - digamma(a + b), digamma(b) - digamma(a + b))
            self.has_conj_prior = True
        else:
            self.conj_prior_params = None
            self.expected_nparams = None
            self.has_conj_prior = False

    def density(self, x: int) -> float:
        return np.exp(self.log_density(x))

    def log_density(self, x: int) -> float:
        n = self.n
        if x < 0 or x > n:
            return -np.inf
        cc = gammaln(n + 1) - gammaln(x + 1) - gammaln(n - x + 1)
        return cc + x*self.log_p + (n - x)*self.log_1p

    def expected_log_density(self, x: int) -> float:
        if self.has_conj_prior:
            n = self.n
            if x < 0 or x > n:
                return -np.inf
            e1, e2 = self.expected_nparams
            cc = gammaln(n + 1) - gammaln(x + 1) - gammaln(n - x + 1)
            return cc + x*e1 + (n - x)*e2
        else:
            return self.log_density(x)

    def entropy(self) -> float:
        x = np.arange(self.n + 1)
        ll = self.seq_log_density(self.seq_encode(x))
        return -np.dot(np.exp(ll), ll)

    def cross_entropy(self, dist: ProbabilityDistribution) -> float:
        x = np.arange(self.n + 1)
        pp = np.exp(self.seq_log_density(self.seq_encode(x)))
        return -np.dot(pp, np.asarray([dist.log_density(u) for u in x]))

    def seq_log_density(self, x):
        xv, cc = x
        rv = xv*self.log_p + (self.n - xv)*self.log_1p + cc
        rv[np.bitwise_or(xv < 0, xv > self.n)] = -np.inf
        return rv

    def seq_expected_log_density(self, x):
        if self.has_conj_prior:
            xv, cc = x
            e1, e2 = self.expected_nparams
            rv = xv*e1 + (self.n - xv)*e2 + cc
            rv[np.bitwise_or(xv < 0, xv > self.n)] = -np.inf
            return rv
        else:
            return self.seq_log_density(x)

    def seq_encode(self, x):
        xv = np.asarray(x, dtype=float)
        cc = gammaln(self.n + 1) - gammaln(xv + 1) - gammaln(self.n - xv + 1)
        return xv, cc

    def sampler(self, seed: Optional[int] = None):
        return BinomialSampler(self, seed)

    def estimator(self):
        return BinomialEstimator(self.n, name=self.name, keys=self.keys, prior=self.prior)


class BinomialSampler(object):

    def __init__(self, dist: BinomialDistribution, seed: Optional[int] = None):
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size=None):
        return self.rng.binomial(self.dist.n, self.dist.p, size=size)


class BinomialAccumulator(StatisticAccumulator):

    def __init__(self, n: int, name=None, keys=None):
        self.n = n
        self.name = name
        self.key = keys
        self.sum = 0.0
        self.count = 0.0

    def initialize(self, x, weight, rng):
        self.update(x, weight, None)

    def seq_initialize(self, x, weights, rng):
        self.seq_update(x, weights, None)

    def update(self, x, weight, estimate):
        self.sum += x*weight
        self.count += weight

    def seq_update(self, x, weights, estimate):
        self.sum += np.dot(x[0], weights)
        self.count += weights.sum()

    def combine(self, suff_stat):
        self.count += suff_stat[0]
        self.sum += suff_stat[1]
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
                stats_dict[self.key].combine(self.value())
            else:
                stats_dict[self.key] = self

    def key_replace(self, stats_dict):
        if self.key is not None:
            if self.key in stats_dict:
                self.from_value(stats_dict[self.key].value())


class BinomialEstimatorAccumulatorFactory(object):

    def __init__(self, n, name, keys):
        self.n = n
        self.name = name
        self.keys = keys

    def make(self):
        return BinomialAccumulator(self.n, name=self.name, keys=self.keys)


class BinomialEstimator(ParameterEstimator):

    def __init__(self, n: int, name: Optional[str] = None, keys: Optional[str] = None,
                 prior: ProbabilityDistribution = default_prior):

        self.n = int(n)
        self.name = name
        self.keys = keys
        self.set_prior(prior)

    def accumulator_factory(self):
        return BinomialEstimatorAccumulatorFactory(self.n, self.name, self.keys)

    def get_prior(self):
        return self.prior

    def set_prior(self, prior):
        self.prior = prior
        self.has_conj_prior = isinstance(prior, BetaDistribution)

    def model_log_density(self, model):
        if self.has_conj_prior:
            return float(self.prior.log_density(model.p))
        return super().model_log_density(model)

    def estimate(self, suff_stat: Tuple[float, float]) -> BinomialDistribution:

        count, psum = suff_stat
        fsum = count*self.n - psum

        if self.has_conj_prior:

            a, b = self.prior.get_parameters()
            new_a = a + psum
            new_b = b + fsum

            # posterior mode for new_a, new_b > 1; mean on the boundary
            if new_a > 1.0 and new_b > 1.0:
                p = (new_a - 1.0)/(new_a + new_b - 2.0)
            else:
                p = new_a/(new_a + new_b)

            return BinomialDistribution(self.n, p, name=self.name, keys=self.keys,
                                        prior=BetaDistribution(new_a, new_b))

        else:
            p = psum/(count*self.n) if count > 0 else 0.5
            return BinomialDistribution(self.n, p, name=self.name, keys=self.keys, prior=self.prior)
