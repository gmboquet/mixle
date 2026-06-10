#import copyreg, copy, pickle, dill
from typing import Optional, Any, Dict
from pysp.arithmetic import *
from numpy.random import RandomState
from pysp.bstats.pdist import ParameterEstimator, ProbabilityDistribution, StatisticAccumulator
from pysp.bstats.gamma import GammaDistribution
from pysp.bstats.nulldist import NullDistribution, null_dist
from pysp.utils.special import stirling2
import numpy as np
from scipy.special import gammaln, digamma, exp1
from scipy.optimize import minimize_scalar
import scipy.integrate


default_prior = GammaDistribution(1.0001, 1.0e6)


class PoissonDistribution(ProbabilityDistribution):
    """
    A Poisson distributed random variable has the likelihood function

    l(x | lambda) = (lambda ** x) * exp(-lambda) / x!

    where x is a non-negative integer and lambda is a positive real number.
    """

    lam: float
    log_lambda: float
    conj_prior_params: (float, float)
    prior: ProbabilityDistribution
    has_conj_prior: bool
    has_prior: bool

    def __init__(self, lam: float, name: Optional[str] = None, prior: ProbabilityDistribution = default_prior, keys: Optional[str] = None):

        self.name = name
        self.keys = keys
        self.set_parameters(lam)
        self.set_prior(prior)

    def __str__(self) -> str:
        return 'PoissonDistribution(%f, name=%s, prior=%s, keys=%s)' % (self.lam, str(self.name), str(self.prior), str(self.keys))

    def get_parameters(self) -> float:
        return self.lam

    def set_parameters(self, params: float) -> None:
        self.lam = params
        self.log_lambda = np.log(self.lam)

    def get_prior(self) -> ProbabilityDistribution:
        return self.prior

    def set_prior(self, prior: ProbabilityDistribution):
        self.prior = prior

        if isinstance(prior, GammaDistribution):
            k, theta = self.prior.get_parameters()
            self.conj_prior_params = (k, theta)
            self.has_conj_prior = True
            self.has_prior = True
        elif isinstance(prior, NullDistribution) or prior is None:
            self.has_prior = False
        else:
            self.conj_prior_params = None
            self.has_conj_prior = False
            self.has_prior = True

    def get_data_type(self):
        return int

    def density(self, x: int) -> float:
        return np.exp(self.log_density(x))

    def log_density(self, x: int) -> float:
        return x*self.log_lambda - float(gammaln(x + 1.0)) - self.lam

    def expected_log_density(self, x: float) -> float:

        if self.has_conj_prior:

            k, theta = self.conj_prior_params
            e1 = (digamma(k)+np.log(theta))*x
            e2 = k*theta
            e3 = gammaln(x+1)
            return e1 - e2 - e3

        else:
            pass

    def cross_entropy(self, dist: ProbabilityDistribution) -> float:
        if isinstance(dist, PoissonDistribution):
            lam1 = self.lam
            lam2 = dist.lam
            rv = self.entropy()
            rv += lam1 - lam2 - (np.log(lam1)-np.log(lam2))*lam1
        else:
            pass

    def entropy(self) -> float:

        if self.lam > 450:
            l = self.lam
            rv = -(0.5*np.log(2.0*np.pi*l) + 0.5 - 1/(12.0*l) - 1.0/(24.0*l*l) - 19.0/(360.0*l*l*l))
        else:
            lam = self.lam
            rv0 = 0.5 * np.log(2.0 * np.pi * lam) + 0.5 + (lam + 0.5) * exp1(lam) - np.exp(-lam)
            rterm = lambda x: (np.exp(-lam * x) / x) * ((1 / x) - 0.5 + (1 / np.log1p(-x)))
            rv1 = scipy.integrate.quad(rterm, 0, 1)[0]
            rv = -rv0 + rv1
        return rv

    def moment(self, p: int) -> float:
        if p == 0:
            return 0
        elif p == 1:
            return self.lam
        else:
            rv = 0
            for i in range(p + 1):
                rv += np.power(self.lam, i) * stirling2(p, i)
            return rv

    def seq_log_density(self, x):
        rv = x[0]*self.log_lambda
        rv -= x[1]
        rv -= self.lam
        return rv

    def seq_expected_log_density(self, x):
        k, theta = self.conj_prior_params
        e1 = (digamma(k) + np.log(theta))*x[0]
        e2 = k*theta
        e3 = x[1]
        return e1 - e2 - e3

    def seq_encode(self, x):
        rv1 = np.asarray(x, dtype=float)
        rv2 = gammaln(rv1 + 1.0)
        return rv1, rv2

    def sampler(self, seed: Optional[int] = None):
        return PoissonSampler(self, seed)

    def estimator(self):
        return PoissonEstimator(name=self.name, keys=self.keys, prior=self.prior)



class PoissonSampler(object):

    def __init__(self, dist, seed=None):
        self.rng  = RandomState(seed)
        self.dist = dist

    def sample(self, size=None):
        return self.rng.poisson(lam=self.dist.lam, size=size)


class PoissonEstimatorAccumulator(StatisticAccumulator):

    def __init__(self, name, keys):
        self.name  = name
        self.key   = keys
        self.sum   = 0.0
        self.gsum  = 0.0
        self.count = 0.0

    def initialize(self, x, weight, rng):
        self.update(x, weight, None)

    def seq_initialize(self, x, weights, rng):
        self.seq_update(x, weights, None)

    def update(self, x, weight, estimate):
        self.sum  += x*weight
        self.count += weight
        self.gsum += gammaln(x+1)

    def seq_update(self, x, weights, estimate):
        self.sum   += np.dot(x[0], weights)
        self.count += weights.sum()

    def combine(self, suff_stat):
        self.sum  += suff_stat[1]
        self.count += suff_stat[0]
        return self

    def value(self):
        return self.count, self.sum

    def from_value(self, x):
        self.count = x[0]
        self.sum = x[1]

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


class PoissonEstimatorAccumulatorFactory(object):

    def __init__(self, name, keys):
        self.name = name
        self.keys = keys

    def make(self):
        return PoissonEstimatorAccumulator(self.name, self.keys)


class PoissonEstimator(ParameterEstimator):

    def __init__(self, name: Optional[str] = None, keys: Optional[str] = None, prior: ProbabilityDistribution = default_prior):

        self.prior = prior
        self.name  = name
        self.keys  = keys
        self.has_conj_prior = isinstance(prior, GammaDistribution)
        self.has_prior = not isinstance(prior, NullDistribution) and prior is not None

    def accumulator_factory(self) -> PoissonEstimatorAccumulatorFactory:
        return PoissonEstimatorAccumulatorFactory(self.name, self.keys)

    def set_prior(self, prior) -> None:
        self.prior = prior

    def get_prior(self) -> ProbabilityDistribution:
        return self.prior

    def estimate(self, suff_stat: (float, float)) -> PoissonDistribution:

        nobs, psum = suff_stat

        if self.has_conj_prior:

            k, theta = self.prior.get_parameters()

            new_k     = k + psum
            new_theta = theta/(nobs*theta + 1)

            # posterior mode of Gamma(k, theta) is (k-1)*theta for k >= 1; fall
            # back to the posterior mean when the mode is at the boundary
            if new_k >= 1.0:
                posterior_mode = (new_k - 1.0)*new_theta
            else:
                posterior_mode = new_k*new_theta

            posterior_mode = max(posterior_mode, 1.0e-128)

            return PoissonDistribution(posterior_mode, name=self.name, prior=GammaDistribution(new_k, new_theta))

        else:
            lam = psum/nobs if nobs > 0 else 1.0
            return PoissonDistribution(max(lam, 1.0e-128), name=self.name, prior=self.prior if self.has_prior else null_dist)
