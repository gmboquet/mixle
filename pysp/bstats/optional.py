from typing import Optional, Any, Dict, Union
from numpy.random import RandomState
from pysp.bstats.pdist import ParameterEstimator, ProbabilityDistribution, StatisticAccumulator
from pysp.bstats.beta import BetaDistribution
from pysp.bstats.composite import CompositeDistribution
from pysp.bstats.nulldist import NullDistribution, null_dist
from pysp.utils.special import stirling2
import numpy as np
from scipy.special import gammaln, digamma, exp1
from scipy.optimize import minimize_scalar
import scipy.integrate


default_prior = BetaDistribution(1.0001, 1.0001)


class OptionalDistribution(ProbabilityDistribution):


    def __init__(self, dist: ProbabilityDistribution, p: float = 0.5, missing_value: Any = None, name: Optional[str] = None, prior: ProbabilityDistribution = default_prior, keys: Optional[str] = None):

        self.name = name
        self.keys = keys
        self.dist = dist
        self.p    = p
        self.log_p0 = np.log(p)
        self.log_p1 = np.log1p(-p)
        self.missing_value = missing_value
        self._set_prior(prior)
        self.mv_is_nan = False if not np.isscalar(missing_value) else np.isnan(missing_value)

    def __str__(self) -> str:
        return 'OptionalDistribution(%s, p=%s, missing_value=%s, name=%s, prior=%s, keys=%s)' % (str(self.dist), repr(self.p), repr(self.missing_value), repr(self.name), str(self.prior), repr(self.keys))

    def get_parameters(self):
        return self.p, self.dist.get_parameters()

    def set_parameters(self, params) -> None:
        self.p = params[0]
        self.log_p0 = np.log(params[0])
        self.log_p1 = np.log1p(-params[0])
        #self.missing_value = params[0][1]
        self.dist.set_parameters(params[1])

    def get_prior(self) -> ProbabilityDistribution:
        return CompositeDistribution((self.prior, self.dist.get_prior()))

    def set_prior(self, prior: ProbabilityDistribution):
        self.dist.set_prior(prior.dists[1])
        self._set_prior(prior.dists[0])

    def _set_prior(self, prior: ProbabilityDistribution):
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
        return Union[type(self.missing_value), self.dist.get_type()]

    def density(self, x) -> float:
        return np.exp(self.log_density(x))

    def log_density(self, x) -> float:
        if (x is self.missing_value) or (self.mv_is_nan and np.isscalar(x) and np.isnan(x)):
            return self.log_p0
        else:
            return self.log_p1 + self.dist.log_density(x)

    def expected_log_density(self, x) -> float:

        if self.has_conj_prior:
            da, db, dab = self.conj_prior_params
            if (x is self.missing_value) or (self.mv_is_nan and np.isscalar(x) and np.isnan(x)):
                return da - dab
            else:
                return db - dab + self.dist.expected_log_density(x)
        else:
            return self.log_density(x)

    def cross_entropy(self, dist: ProbabilityDistribution) -> float:
        if isinstance(dist, OptionalDistribution):
            v1 = -self.p * dist.log_p0
            v2 = (1.0 - self.p) * (-dist.log_p1 + self.dist.cross_entropy(dist.dist))
            return v1 + v2
        else:
            v1 = -self.p * dist.log_density(self.missing_value)
            v2 = (1.0 - self.p) * self.dist.cross_entropy(dist)
            return v1 + v2

    def entropy(self) -> float:
        v1 = -self.p * self.log_p0
        v2 = (self.p - 1.0) * (-self.log_p1 + self.dist.entropy())
        return v1 + v2

    def seq_log_density(self, x):
        rv = np.empty(x[0], dtype=np.float64)
        rv.fill(self.log_p0)
        rv[x[1]] = self.dist.seq_log_density(x[3]) + self.log_p1
        return rv

    def seq_expected_log_density(self, x):
        da, db, dab = self.conj_prior_params
        aa = da - dab
        bb = db - dab

        rv = np.empty(x[0], dtype=np.float64)
        rv.fill(aa)
        rv[x[1]] = self.dist.seq_expected_log_density(x[3]) + bb
        return rv

    def seq_encode(self, x):
        nz_idx = []
        iz_idx = []
        nz_val = []
        cnt = 0

        for i,xx in enumerate(x):
            cnt += 1

            if  (xx is self.missing_value) or (self.mv_is_nan and np.isscalar(xx) and np.isnan(xx)):
                iz_idx.append(i)
            else:
                nz_idx.append(i)
                nz_val.append(xx)

        nz_idx = np.asarray(nz_idx, dtype=np.int32)
        iz_idx = np.asarray(iz_idx, dtype=np.int32)
        nz_val = self.dist.seq_encode(nz_val)

        return cnt, nz_idx, iz_idx, nz_val

    def sampler(self, seed: Optional[int] = None):
        return OptionalSampler(self, seed)

    def estimator(self):
        return OptionalEstimator(name=self.name, keys=self.keys, prior=self.prior)


class OptionalSampler(object):

    def __init__(self, dist, seed=None):
        rng  = np.random.RandomState(seed)
        self.dist = dist
        self.obs_sampler = dist.dist.sampler(rng.randint(0, 2**31))
        self.mis_sampler = np.random.RandomState(rng.randint(0, 2**31))

    def sample(self, size=None):
        if size is None:
            if self.mis_sampler.rand() <= self.dist.p:
                return self.dist.missing_value
            else:
                return self.obs_sampler.sample()
        else:
            return [self.sample() for i in range(size)]


class OptionalEstimatorAccumulator(StatisticAccumulator):

    def __init__(self, accumulator, missing_value, name, keys):
        self.acc   = accumulator
        self.name  = name
        self.key   = keys
        self.psum  = 0.0
        self.nsum  = 0.0
        self.missing_value = missing_value
        self.mv_is_nan = False if not np.isscalar(missing_value) else np.isnan(missing_value)

    def initialize(self, x, weight, rng):
        self.update(x, weight, None)

    def seq_initialize(self, x, weights, rng):
        self.seq_update(x, weights, None)

    def update(self, x, weight, estimate):
        if (x is self.missing_value) or (self.mv_is_nan and np.isscalar(x) and np.isnan(x)):
            self.psum += weight
        else:
            self.nsum += weight
            self.acc.update(x, weight, None if estimate is None else estimate.dist)

    def seq_update(self, x, weights, estimate):
        cnt, nz_idx, iz_idx, nz_val = x

        self.psum += weights[iz_idx].sum()
        self.nsum += weights[nz_idx].sum()
        self.acc.seq_update(nz_val, weights[nz_idx], None if estimate is None else estimate.dist)

    def combine(self, suff_stat):
        self.psum += suff_stat[0]
        self.nsum += suff_stat[1]
        self.acc.combine(suff_stat[2])
        return self

    def value(self):
        return self.psum, self.nsum, self.acc.value()

    def from_value(self, x):
        self.psum = x[0]
        self.nsum = x[1]
        self.acc.from_value(x[2])
        return self

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


class OptionalEstimatorAccumulatorFactory(object):

    def __init__(self, acc_factory, missing_value, name, keys):
        self.name = name
        self.keys = keys
        self.acc_factory = acc_factory
        self.missing_value = missing_value

    def make(self):
        acc = None if self.acc_factory is None else self.acc_factory.make()
        return OptionalEstimatorAccumulator(acc, self.missing_value, self.name, self.keys)


class OptionalEstimator(ParameterEstimator):

    def __init__(self, estimator: ParameterEstimator, missing_value: Any = None, fixed_prob: Optional[float] = None, name: Optional[str] = None, keys: Optional[str] = None, prior: ProbabilityDistribution = default_prior):

        self.estimator = estimator
        self.name  = name
        self.keys  = keys
        self.prior = prior
        self.fixed_prob = fixed_prob
        self.missing_value = missing_value
        self.has_conj_prior = isinstance(prior, BetaDistribution)
        self.has_prior = not isinstance(prior, NullDistribution) and prior is not None

    def accumulator_factory(self) -> OptionalEstimatorAccumulatorFactory:
        acc_factory = None if self.estimator is None else self.estimator.accumulator_factory()
        return OptionalEstimatorAccumulatorFactory(acc_factory, self.missing_value, self.name, self.keys)

    def set_prior(self, prior) -> None:
        if isinstance(prior, CompositeDistribution):
            self.prior = prior.dists[0]
            self.has_conj_prior = isinstance(self.prior, BetaDistribution)
            self.has_prior = not isinstance(self.prior, NullDistribution) and self.prior is not None
            self.estimator.set_prior(prior.dists[1])

    def get_prior(self) -> ProbabilityDistribution:
        return CompositeDistribution((self.prior, self.estimator.get_prior()))

    def estimate(self, suff_stat: (float, float)) -> OptionalDistribution:

        psum, nsum, dist_suff_stat = suff_stat

        dist = self.estimator.estimate(dist_suff_stat)

        if self.has_conj_prior:

            a, b = self.prior.get_parameters()
            new_a = a + psum
            new_b = b + nsum
            new_p = (psum + a - 1.0)/(psum + nsum + a + b - 2.0)
            new_prior = BetaDistribution(new_a, new_b)

        else:
            new_p = psum/(psum + nsum)
            new_prior = self.prior

        if self.fixed_prob is not None:
            new_p = self.fixed_prob

        return OptionalDistribution(dist, p=new_p, missing_value=self.missing_value, name=self.name, prior=new_prior, keys=self.keys)