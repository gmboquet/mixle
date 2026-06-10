"""Log-Gaussian (log-normal) distribution with a conjugate NormalGamma prior.

If X ~ LogGaussian(mu, sigma2) then log(X) ~ Gaussian(mu, sigma2), so the
NormalGamma prior on (mu, tau = 1/sigma2) is conjugate for the log of the
data. Encoded sequences store log(x), which makes the sufficient statistics
and variational expectations identical to the Gaussian case up to the
Jacobian term -log(x) in the density.
"""
from typing import Optional, Tuple

from numpy.random import RandomState
from scipy.special import digamma
import numpy as np

from pysp.bstats.pdist import ProbabilityDistribution, StatisticAccumulator, ParameterEstimator
from pysp.bstats.normgamma import NormalGammaDistribution
from pysp.bstats.gaussian import GaussianAccumulator, GaussianEstimator

default_prior = NormalGammaDistribution(0.0, 1.0e-8, 0.500001, 1.0)


class LogGaussianDistribution(ProbabilityDistribution):

    def __init__(self, mu: float, sigma2: float, name: Optional[str] = None,
                 prior: ProbabilityDistribution = default_prior):

        assert sigma2 > 0 and np.isfinite(sigma2)
        assert np.isfinite(mu)
        self.name = name
        self.set_parameters((mu, sigma2))
        self.set_prior(prior)

    def __str__(self):
        return 'LogGaussianDistribution(%f, %f, name=%s, prior=%s)' % (self.mu, self.sigma2, self.name, str(self.prior))

    def get_parameters(self) -> Tuple[float, float]:
        return self.mu, self.sigma2

    def set_parameters(self, params: Tuple[float, float]) -> None:
        self.mu = params[0]
        self.sigma2 = params[1]
        self.log_const = -0.5*np.log(2.0*np.pi*self.sigma2)

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

    def density(self, x: float) -> float:
        return np.exp(self.log_density(x))

    def log_density(self, x: float) -> float:
        if x <= 0:
            return -np.inf
        y = np.log(x)
        return self.log_const - 0.5*(y - self.mu)*(y - self.mu)/self.sigma2 - y

    def expected_log_density(self, x: float) -> float:
        if self.has_conj_prior:
            if x <= 0:
                return -np.inf
            y = np.log(x)
            ea, eb, e1, e2 = self.expected_nparams
            return y*(e1 + y*e2) - ea + eb - y
        else:
            return self.log_density(x)

    def seq_encode(self, x) -> np.ndarray:
        rv = np.log(np.asarray(x, dtype=float))
        if np.any(np.isnan(rv)) or np.any(np.isinf(rv)):
            raise Exception('LogGaussianDistribution requires support x in (0,inf).')
        return rv

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        rv = x - self.mu
        rv *= rv
        rv *= -0.5/self.sigma2
        rv += self.log_const
        rv -= x
        return rv

    def seq_expected_log_density(self, x: np.ndarray) -> np.ndarray:
        if self.has_conj_prior:
            ea, eb, e1, e2 = self.expected_nparams
            return x*(e1 + x*e2) - ea + eb - x
        else:
            return self.seq_log_density(x)

    def sampler(self, seed: Optional[int] = None):
        return LogGaussianSampler(self, seed)

    def estimator(self):
        return LogGaussianEstimator(name=self.name, prior=self.prior)


class LogGaussianSampler(object):

    def __init__(self, dist: LogGaussianDistribution, seed: Optional[int] = None):
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size=None):
        return self.rng.lognormal(mean=self.dist.mu, sigma=np.sqrt(self.dist.sigma2), size=size)


class LogGaussianAccumulator(GaussianAccumulator):
    """Accumulates Gaussian sufficient statistics of log(x).

    seq_update receives encoded (already logged) values, so only the scalar
    update path needs the log transform.
    """

    def update(self, x, weight, estimate):
        super().update(np.log(x), weight, estimate)


class LogGaussianEstimatorAccumulatorFactory(object):

    def __init__(self, name, keys):
        self.name = name
        self.keys = keys

    def make(self):
        return LogGaussianAccumulator(name=self.name, keys=self.keys)


class LogGaussianEstimator(GaussianEstimator):
    """NormalGamma-conjugate estimator for log-normal data.

    The conjugate update is the Gaussian one applied to log-scale sufficient
    statistics; only the returned distribution type differs.
    """

    def accumulator_factory(self):
        return LogGaussianEstimatorAccumulatorFactory(self.name, self.keys)

    def estimate(self, suff_stat):
        gd = super().estimate(suff_stat)
        sigma2 = gd.sigma2 if gd.sigma2 > 0 else 1.0
        return LogGaussianDistribution(gd.mu, sigma2, name=self.name, prior=gd.prior)
