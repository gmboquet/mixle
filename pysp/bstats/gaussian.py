from typing import Optional, List, Iterable, TypeVar, Tuple, Union
from pysp.arithmetic import *
from pysp.bstats.normgamma import NormalGammaDistribution
from pysp.bstats.pdist import ProbabilityDistribution, StatisticAccumulator, ParameterEstimator
from numpy.random import RandomState
from pysp.utils.special import digamma, gammaln
import numpy as np

default_prior = NormalGammaDistribution(0.0, 1.0e-8, 0.500001, 1.0)


class GaussianDistribution(ProbabilityDistribution):

    def __init__(self, mu: float, sigma2: float, name: Optional[str] = None, prior: ProbabilityDistribution = default_prior):

        assert sigma2 > 0 and np.isfinite(sigma2)
        assert np.isfinite(mu)
        self.parents = []
        self.set_parameters((mu, sigma2))
        self.set_prior(prior)
        self.set_name(name)
        #self.prior = prior # normal-gamma with lambda = 1

    def __str__(self):
        return 'GaussianDistribution(%f, %f, name=%s, prior=%s)' % (self.mu, self.sigma2, self.name, str(self.prior))

    def get_parameters(self) -> Tuple[float, float]:
        return self.mu, self.sigma2

    def set_parameters(self, params: Tuple[float, float]) -> None:
        self.mu = params[0]
        self.sigma2 = params[1]
        self.logConst = -0.5*log(2.0*pi*self.sigma2)
        self.const    = 1.0/sqrt(2.0*pi*self.sigma2)

    def get_prior(self) -> ProbabilityDistribution:
        return self.prior

    def set_prior(self, prior: ProbabilityDistribution) -> None:
        self.prior = prior

        if isinstance(prior, NormalGammaDistribution):
            self.conj_prior_params = prior.get_parameters()

            mu, lam, a, b = self.conj_prior_params

            ea = ((mu*mu)*(a/b)*0.5 + (0.5/lam) + 0.5*(np.log(b) - digamma(a)))
            e1 = mu*a/b
            e2 = -0.5*a/b
            eb = -0.5*np.log(2*np.pi)

            self.expected_nparams = [ea, eb, e1, e2]
            self.has_conj_prior = True

        else:
            self.conj_prior_params = None
            self.expected_nparams = None
            self.has_conj_prior = False

    def log_density(self, x: float) -> float:
        return self.logConst - 0.5*(x-self.mu)*(x-self.mu)/self.sigma2

    def expected_log_density(self, x: float) -> float:

        if self.has_conj_prior:
            ea, eb, e1, e2 = self.expected_nparams
            return x*(e1 + x*e2) - ea + eb
        else:
            return self.log_density(x)

    def seq_encode(self, x: Iterable[float]) -> np.ndarray:
        rv = np.asarray(x, dtype=float)
        return rv

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        rv = x - self.mu
        rv *= rv
        rv *= -0.5/self.sigma2
        rv += self.logConst
        return rv

    def seq_expected_log_density(self, x: np.ndarray) -> np.ndarray:

        if self.conj_prior_params is not None:

            ea, eb, e1, e2 = self.expected_nparams

            return x*(e1 + x*e2) - ea + eb

        else:
            return self.seq_log_density(x)

    def sampler(self, seed: Optional[int] = None):
        return GaussianSampler(self, seed)

    def estimator(self):
        return GaussianEstimator(name=self.name, prior=self.prior)


class GaussianSampler(object):

    def __init__(self, dist: GaussianDistribution, seed: Optional[int] = None):
        self.rng  = RandomState(seed)
        self.dist = dist

    def sample(self, size=None):
        return self.rng.normal(loc=self.dist.mu, scale=sqrt(self.dist.sigma2), size=size)


class GaussianAccumulator(StatisticAccumulator):

    def __init__(self, name=None, keys=(None, None)):

        self.name = name
        self.sum  = 0.0
        self.sum2 = 0.0
        self.sum3 = 0.0
        self.count = 0.0
        self.count2 = 0.0
        self.sum_key = keys[0]
        self.sum2_key = keys[1]

    def initialize(self, x, weight, rng):
        self.update(x, weight, None)

    def update(self, x, weight, estimate):
        xWeight   = x*weight
        self.sum  += xWeight
        self.sum2 += x*xWeight
        self.sum3 += xWeight
        self.count += weight
        self.count2 += weight

    def seq_initialize(self, x, weights, rng):
        self.seq_update(x, weights, None)

    def seq_update(self, x, weights, estimate):
        temp = np.dot(x, weights)
        self.sum += temp
        self.sum2 += np.dot(x*x, weights)
        self.sum3 += temp
        w_sum = weights.sum()
        self.count += w_sum
        self.count2 += w_sum

    def df_initialize(self, df, weights, rng):
        self.df_update(df, weights, None)

    def df_update(self, df, weights, estimate):
        col = df[self.name]
        self.sum += col.dot(weights)
        self.sum2 += col.pow(2.0).dot(weights)
        self.sum3 += col.dot(weights)
        w_sum = weights.sum()
        self.count += w_sum
        self.count2 += w_sum

    def combine(self, suff_stat):
        self.sum  += suff_stat[0]
        self.sum2 += suff_stat[1]
        self.sum3 += suff_stat[2]
        self.count += suff_stat[3]
        self.count2 += suff_stat[4]

        return self

    def value(self):
        return self.sum, self.sum2, self.sum3, self.count, self.count2

    def from_value(self, x):
        self.sum = x[0]
        self.sum2 = x[1]
        self.sum3 = x[2]
        self.count = x[3]
        self.count2 = x[4]
        return self

    def key_merge(self, stats_dict):
        if self.sum_key is not None:
            if self.sum_key in stats_dict:
                vals = stats_dict[self.sum_key]
                stats_dict[self.sum_key] = (vals[0] + self.count, vals[1] + self.sum)
            else:
                stats_dict[self.sum_key] = (self.count, self.sum)

        if self.sum2_key is not None:
            if self.sum2_key in stats_dict and self.count2 > 0:
                vals = stats_dict[self.sum2_key]

                m0 = self.sum3/self.count2
                m1 = 0 if vals[0] == 0 else vals[2]/vals[0]
                m2 = (self.sum3 + vals[2])/(self.count2 + vals[0])
                b0 = self.sum2 - m0*self.sum3
                b1 = (vals[0]*self.count2/(vals[0] + self.count2))*np.power(m0 - m1, 2.0)
                b2 = vals[1] - m1*vals[2] + b0 + m2*(self.sum3 + vals[2])

                stats_dict[self.sum2_key] = (vals[0] + self.count2, b2, vals[2] + self.sum3)
            else:
                stats_dict[self.sum2_key] = (self.count2, self.sum2, self.sum3)

    def key_replace(self, stats_dict):
        if self.sum_key is not None:
            if self.sum_key in stats_dict:
                vals = stats_dict[self.sum_key]
                self.count = vals[0]
                self.sum = vals[1]

        if self.sum2_key is not None:
            if self.sum2_key in stats_dict:
                vals = stats_dict[self.sum2_key]
                self.count2 = vals[0]
                self.sum2 = vals[1]
                self.sum3 = vals[2]


class GaussianEstimatorAccumulatorFactory(object):

    def __init__(self, name, keys):
        self.name = name
        self.keys = keys

    def make(self):
        return GaussianAccumulator(name=self.name, keys=self.keys)


class GaussianEstimator(ParameterEstimator):

    def __init__(self, name=None, prior=default_prior, keys=(None, None)):

        self.keys  = keys
        self.name  = name
        self.set_prior(prior)

    def accumulator_factory(self):
        return GaussianEstimatorAccumulatorFactory(self.name, self.keys)

    def set_prior(self, prior):
        self.prior = prior
        self.has_conj_prior = isinstance(prior, NormalGammaDistribution)

    def get_prior(self):
        return self.prior

    def model_log_density(self, model):
        # the normal-gamma prior is over (mu, tau) with tau = 1/sigma2
        if self.has_conj_prior:
            mu, sigma2 = model.get_parameters()
            return float(self.prior.log_density((mu, 1.0/sigma2)))
        return super().model_log_density(model)

    def estimate(self, suff_stat):

        sum_x, sum_xx, sum_xxx, nobs_loc1, nobs_loc2 = suff_stat

        if self.has_conj_prior:

            old_mu, old_lam, old_a, old_b = self.prior.get_parameters()

            new_n  = old_lam + nobs_loc1
            new_a  = old_a + (nobs_loc2 / 2.0)
            new_nn = old_lam + nobs_loc2

            if nobs_loc1 > 0:
                sample_mean1 = (sum_x/nobs_loc1)
            else:
                sample_mean1 = 0

            if nobs_loc2 > 0:
                sample_mean2 = (sum_xxx/nobs_loc2)
            else:
                sample_mean2 = 0

            new_mu = (sum_x + old_mu*old_lam)/(old_lam + nobs_loc1)

            new_b0 = (sum_xx - sample_mean2*sum_xxx)
            new_b1 = (old_lam*nobs_loc1/new_n)*np.power(sample_mean1-old_mu,2)
            new_b  = old_b + 0.5*(new_b0 + new_b1)

            new_sigma2 = (new_b/(new_a - 0.5))

            new_prior  = NormalGammaDistribution(new_mu, new_n, new_a, new_b)

            return GaussianDistribution(new_mu, new_sigma2, name=self.name, prior=new_prior)

        else:


            if nobs_loc1 == 0:
                mu = 0.0
            else:
                mu = sum_x / nobs_loc1

            if nobs_loc2 == 0:
                sigma2 = 0
            else:
                mu2 = sum_xxx/nobs_loc2
                sigma2 = (sum_xx / nobs_loc2) - mu2*mu2

            return GaussianDistribution(mu, sigma2, name=self.name)