"""Multivariate Gaussian with conjugate Normal-Wishart prior.

With prior (mu, Lambda) ~ NW(m0, kappa0, W0, nu0) and observations x_1..x_n
(sample mean xbar, scatter S = sum (x_i - xbar)(x_i - xbar)'), the posterior is

    kappa_n = kappa0 + n
    m_n     = (kappa0 m0 + n xbar) / kappa_n
    nu_n    = nu0 + n
    W_n^-1  = W0^-1 + S + (kappa0 n / kappa_n)(xbar - m0)(xbar - m0)'

Estimation returns the joint MAP: mu = m_n and Sigma = W_n^-1 / (nu_n - d)
(the d-dimensional analogue of sigma2 = b/(a - 1/2)), carrying the posterior
as the new prior. expected_log_density implements the standard variational
expectations (Bishop 10.64-10.66).
"""
from typing import Optional, Tuple

import numpy as np
from numpy.random import RandomState

from pysp.bstats.pdist import ProbabilityDistribution, StatisticAccumulator, ParameterEstimator
from pysp.bstats.normwishart import NormalWishartDistribution


def default_prior(dim: int) -> NormalWishartDistribution:
    # d-dimensional analogue of NormalGamma(0, 1e-8, 0.500001, 1.0):
    # nu = 2a + (d-1), W = (2b)^-1 * I
    return NormalWishartDistribution(np.zeros(dim), 1.0e-8, np.eye(dim)*0.5, dim + 2.0e-6)


class MultivariateGaussianDistribution(ProbabilityDistribution):

    def __init__(self, mu, covar, name: Optional[str] = None,
                 prior: Optional[ProbabilityDistribution] = None):

        self.name = name
        self.set_parameters((mu, covar))
        self.set_prior(prior if prior is not None else default_prior(self.dim))

    def __str__(self):
        mu = ','.join(map(str, self.mu.tolist()))
        co = ','.join(map(str, self.covar.flatten().tolist()))
        return 'MultivariateGaussianDistribution([%s], [%s], name=%s, prior=%s)' % (mu, co, self.name, str(self.prior))

    def get_parameters(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.mu, self.covar

    def set_parameters(self, params) -> None:
        mu, covar = params

        self.mu = np.asarray(mu, dtype=float)
        self.covar = np.asarray(covar, dtype=float)
        self.dim = len(self.mu)

        sgn, self.log_det_covar = np.linalg.slogdet(self.covar)
        assert sgn > 0, 'Covariance matrix must be positive definite.'
        self.precision = np.linalg.inv(self.covar)
        self.log_const = -0.5*(self.dim*np.log(2.0*np.pi) + self.log_det_covar)

    def get_prior(self) -> ProbabilityDistribution:
        return self.prior

    def set_prior(self, prior: ProbabilityDistribution) -> None:
        self.prior = prior
        self.has_conj_prior = isinstance(prior, NormalWishartDistribution)

        if self.has_conj_prior:
            self.conj_prior_params = prior.get_parameters()
            # E[ln|Lambda|] and the data-independent parts of E[log p(x|mu,Lambda)]
            self.e_log_det = prior.expected_log_det()
        else:
            self.conj_prior_params = None
            self.e_log_det = None

    def density(self, x) -> float:
        return np.exp(self.log_density(x))

    def log_density(self, x) -> float:
        diff = np.asarray(x, dtype=float) - self.mu
        return self.log_const - 0.5*float(np.dot(diff, np.dot(self.precision, diff)))

    def expected_log_density(self, x) -> float:
        if self.has_conj_prior:
            m0, kappa, w_mat, nu = self.conj_prior_params
            diff = np.asarray(x, dtype=float) - m0
            e_quad = self.dim/kappa + nu*float(np.dot(diff, np.dot(w_mat, diff)))
            return 0.5*self.e_log_det - 0.5*self.dim*np.log(2.0*np.pi) - 0.5*e_quad
        else:
            return self.log_density(x)

    def seq_encode(self, x) -> np.ndarray:
        rv = np.reshape(np.asarray(x, dtype=float), (-1, self.dim))
        return rv

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        diff = x - self.mu
        rv = -0.5*np.sum(np.dot(diff, self.precision)*diff, axis=1)
        rv += self.log_const
        return rv

    def seq_expected_log_density(self, x: np.ndarray) -> np.ndarray:
        if self.has_conj_prior:
            m0, kappa, w_mat, nu = self.conj_prior_params
            diff = x - m0
            e_quad = self.dim/kappa + nu*np.sum(np.dot(diff, w_mat)*diff, axis=1)
            return 0.5*self.e_log_det - 0.5*self.dim*np.log(2.0*np.pi) - 0.5*e_quad
        else:
            return self.seq_log_density(x)

    def sampler(self, seed: Optional[int] = None):
        return MultivariateGaussianSampler(self, seed)

    def estimator(self):
        return MultivariateGaussianEstimator(self.dim, name=self.name, prior=self.prior)


class MultivariateGaussianSampler(object):

    def __init__(self, dist: MultivariateGaussianDistribution, seed: Optional[int] = None):
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size=None):
        rv = self.rng.multivariate_normal(self.dist.mu, self.dist.covar, size=size)
        if size is None:
            return rv
        return list(rv)


class MultivariateGaussianAccumulator(StatisticAccumulator):

    def __init__(self, dim: int, name=None, keys=None):
        self.dim = dim
        self.name = name
        self.key = keys
        self.sum = np.zeros(dim)
        self.sum_outer = np.zeros((dim, dim))
        self.count = 0.0

    def initialize(self, x, weight, rng):
        self.update(x, weight, None)

    def seq_initialize(self, x, weights, rng):
        self.seq_update(x, weights, None)

    def update(self, x, weight, estimate):
        xv = np.asarray(x, dtype=float)
        self.sum += xv*weight
        self.sum_outer += np.outer(xv, xv)*weight
        self.count += weight

    def seq_update(self, x, weights, estimate):
        self.sum += np.dot(x.T, weights)
        self.sum_outer += np.dot(x.T*weights, x)
        self.count += weights.sum()

    def combine(self, suff_stat):
        self.count += suff_stat[0]
        self.sum += suff_stat[1]
        self.sum_outer += suff_stat[2]
        return self

    def value(self):
        return self.count, self.sum, self.sum_outer

    def from_value(self, x):
        self.count = x[0]
        self.sum = x[1]
        self.sum_outer = x[2]
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


class MultivariateGaussianAccumulatorFactory(object):

    def __init__(self, dim, name, keys):
        self.dim = dim
        self.name = name
        self.keys = keys

    def make(self):
        return MultivariateGaussianAccumulator(self.dim, name=self.name, keys=self.keys)


class MultivariateGaussianEstimator(ParameterEstimator):

    def __init__(self, dim: int, name: Optional[str] = None, keys: Optional[str] = None,
                 prior: Optional[ProbabilityDistribution] = None):

        self.dim = int(dim)
        self.name = name
        self.keys = keys
        self.set_prior(prior if prior is not None else default_prior(self.dim))

    def accumulator_factory(self):
        return MultivariateGaussianAccumulatorFactory(self.dim, self.name, self.keys)

    def get_prior(self):
        return self.prior

    def set_prior(self, prior):
        self.prior = prior
        self.has_conj_prior = isinstance(prior, NormalWishartDistribution)

    def model_log_density(self, model):
        # the normal-Wishart prior is over (mu, Lambda) with Lambda = Sigma^-1
        if self.has_conj_prior:
            mu, covar = model.get_parameters()
            return float(self.prior.log_density((mu, np.linalg.inv(covar))))
        return super().model_log_density(model)

    def estimate(self, suff_stat) -> MultivariateGaussianDistribution:

        count, xsum, outer_sum = suff_stat
        d = self.dim

        if self.has_conj_prior:

            m0, kappa0, w0, nu0 = self.prior.get_parameters()

            kappa_n = kappa0 + count
            nu_n = nu0 + count
            m_n = (kappa0*m0 + xsum)/kappa_n

            if count > 0:
                xbar = xsum/count
                scatter = outer_sum - count*np.outer(xbar, xbar)
                dmu = xbar - m0
                w_n_inv = np.linalg.inv(w0) + scatter + (kappa0*count/kappa_n)*np.outer(dmu, dmu)
            else:
                w_n_inv = np.linalg.inv(w0)

            # keep the inverse-scale symmetric despite accumulation round-off
            w_n_inv = 0.5*(w_n_inv + w_n_inv.T)
            w_n = np.linalg.inv(w_n_inv)

            # joint MAP precision is (nu_n - d) W_n for nu_n > d; fall back to
            # the posterior mean nu_n W_n at the boundary
            if nu_n > d:
                covar = w_n_inv/(nu_n - d)
            else:
                covar = w_n_inv/nu_n

            posterior = NormalWishartDistribution(m_n, kappa_n, w_n, nu_n)

            return MultivariateGaussianDistribution(m_n, covar, name=self.name, prior=posterior)

        else:

            if count > 0:
                mu = xsum/count
                covar = outer_sum/count - np.outer(mu, mu)
            else:
                mu = np.zeros(d)
                covar = np.eye(d)

            return MultivariateGaussianDistribution(mu, covar, name=self.name, prior=self.prior)
