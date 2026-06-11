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
    """Log-Gaussian distribution with log-scale mean mu and variance sigma2,
    optionally carrying a NormalGamma conjugate prior over (mu, 1/sigma2)."""

    def __init__(self, mu: float, sigma2: float, name: Optional[str] = None,
                 prior: ProbabilityDistribution = default_prior):
        """LogGaussianDistribution object with log-scale parameters (mu, sigma2).

        Args:
            mu (float): Mean of log(X). Must be finite.
            sigma2 (float): Variance of log(X). Must be positive and finite.
            name (Optional[str]): Name of object.
            prior (ProbabilityDistribution): Prior on the parameters;
                a NormalGammaDistribution over (mu, tau=1/sigma2) enables the
                conjugate machinery (see set_prior()).

        """
        assert sigma2 > 0 and np.isfinite(sigma2)
        assert np.isfinite(mu)
        self.name = name
        self.set_parameters((mu, sigma2))
        self.set_prior(prior)

    def __str__(self):
        return 'LogGaussianDistribution(%f, %f, name=%s, prior=%s)' % (self.mu, self.sigma2, self.name, str(self.prior))

    def get_parameters(self) -> Tuple[float, float]:
        """Returns the parameter tuple (mu, sigma2)."""
        return self.mu, self.sigma2

    def set_parameters(self, params: Tuple[float, float]) -> None:
        """Set the parameters and refresh the cached normalizing constant.

        Args:
            params (Tuple[float, float]): Tuple (mu, sigma2).

        """
        self.mu = params[0]
        self.sigma2 = params[1]
        self.log_const = -0.5*np.log(2.0*np.pi*self.sigma2)

    def get_prior(self) -> ProbabilityDistribution:
        """Returns the prior distribution on (mu, tau=1/sigma2)."""
        return self.prior

    def set_prior(self, prior: ProbabilityDistribution) -> None:
        """Set the prior and precompute conjugate-prior expectations.

        If prior is a NormalGammaDistribution(mu0, lam, a, b) over
        (mu, tau=1/sigma2), this caches the expected natural parameters
        [ea, eb, e1, e2] exactly as in the Gaussian case (see
        pysp.bstats.gaussian), so that expected_log_density(x) =
        y*(e1 + y*e2) - ea + eb - y with y = log(x). Sets has_conj_prior
        accordingly.

        Args:
            prior (ProbabilityDistribution): Prior on the parameters.

        """
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
        """Density of the log-Gaussian at observation x.

        Args:
            x (float): Positive real-valued observation.

        Returns:
            Density at observation x.

        """
        return np.exp(self.log_density(x))

    def log_density(self, x: float) -> float:
        """Log-density of the log-Gaussian at observation x.

        Equals the Gaussian log-density of log(x) plus the Jacobian term
        -log(x); -inf for x <= 0.

        Args:
            x (float): Positive real-valued observation.

        Returns:
            Log-density at observation x.

        """
        if x <= 0:
            return -np.inf
        y = np.log(x)
        return self.log_const - 0.5*(y - self.mu)*(y - self.mu)/self.sigma2 - y

    def expected_log_density(self, x: float) -> float:
        """Variational expectation E_q[log p(x | mu, tau)] under the prior.

        With a NormalGamma conjugate prior q this is the Gaussian VB
        expected log-likelihood evaluated at log(x), plus the Jacobian term
        -log(x); without a conjugate prior it falls back to the plug-in
        log_density(x).

        Args:
            x (float): Positive real-valued observation.

        Returns:
            Expected log-density at observation x.

        """
        if self.has_conj_prior:
            if x <= 0:
                return -np.inf
            y = np.log(x)
            ea, eb, e1, e2 = self.expected_nparams
            return y*(e1 + y*e2) - ea + eb - y
        else:
            return self.log_density(x)

    def seq_encode(self, x) -> np.ndarray:
        """Encode positive observations into a numpy array of log(x).

        Args:
            x: Iterable of positive observations.

        Returns:
            Numpy array of log-values for use with seq_ methods.

        """
        rv = np.log(np.asarray(x, dtype=float))
        if np.any(np.isnan(rv)) or np.any(np.isinf(rv)):
            raise Exception('LogGaussianDistribution requires support x in (0,inf).')
        return rv

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Vectorized log-density at sequence-encoded (logged) input x.

        Args:
            x (np.ndarray): Encoded observations from seq_encode().

        Returns:
            Numpy array of log-densities, one per observation.

        """
        rv = x - self.mu
        rv *= rv
        rv *= -0.5/self.sigma2
        rv += self.log_const
        rv -= x
        return rv

    def seq_expected_log_density(self, x: np.ndarray) -> np.ndarray:
        """Vectorized expected_log_density() at sequence-encoded (logged) input x.

        Args:
            x (np.ndarray): Encoded observations from seq_encode().

        Returns:
            Numpy array of expected log-densities, one per observation.

        """
        if self.has_conj_prior:
            ea, eb, e1, e2 = self.expected_nparams
            return x*(e1 + x*e2) - ea + eb - x
        else:
            return self.seq_log_density(x)

    def sampler(self, seed: Optional[int] = None):
        """Create a LogGaussianSampler for this distribution.

        Args:
            seed (Optional[int]): Seed for the random number generator.

        Returns:
            LogGaussianSampler object.

        """
        return LogGaussianSampler(self, seed)

    def estimator(self):
        """Create a LogGaussianEstimator with this distribution's name and prior.

        Returns:
            LogGaussianEstimator object.

        """
        return LogGaussianEstimator(name=self.name, prior=self.prior)


class LogGaussianSampler(object):
    """Draws samples from a LogGaussianDistribution."""

    def __init__(self, dist: LogGaussianDistribution, seed: Optional[int] = None):
        """LogGaussianSampler object.

        Args:
            dist (LogGaussianDistribution): Distribution to sample from.
            seed (Optional[int]): Seed for the random number generator.

        """
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size=None):
        """Draw size samples (a single float when size is None).

        Args:
            size (Optional[int]): Number of samples to draw.

        Returns:
            A float if size is None, else a numpy array of length size.

        """
        return self.rng.lognormal(mean=self.dist.mu, sigma=np.sqrt(self.dist.sigma2), size=size)


class LogGaussianAccumulator(GaussianAccumulator):
    """Accumulates Gaussian sufficient statistics of log(x).

    seq_update receives encoded (already logged) values, so only the scalar
    update path needs the log transform.
    """

    def update(self, x, weight, estimate):
        """Accumulate the weighted Gaussian sufficient statistics of log(x).

        Args:
            x (float): Positive real-valued observation.
            weight (float): Weight of the observation.
            estimate: Current distribution estimate (unused).

        """
        super().update(np.log(x), weight, estimate)


class LogGaussianEstimatorAccumulatorFactory(object):
    """Factory that creates LogGaussianAccumulator objects."""

    def __init__(self, name, keys):
        """LogGaussianEstimatorAccumulatorFactory object.

        Args:
            name (Optional[str]): Name passed to created accumulators.
            keys: Keys passed to created accumulators.

        """
        self.name = name
        self.keys = keys

    def make(self):
        """Returns a new LogGaussianAccumulator."""
        return LogGaussianAccumulator(name=self.name, keys=self.keys)


class LogGaussianEstimator(GaussianEstimator):
    """NormalGamma-conjugate estimator for log-normal data.

    The conjugate update is the Gaussian one applied to log-scale sufficient
    statistics; only the returned distribution type differs.
    """

    def accumulator_factory(self):
        """Returns a LogGaussianEstimatorAccumulatorFactory for this estimator."""
        return LogGaussianEstimatorAccumulatorFactory(self.name, self.keys)

    def estimate(self, suff_stat):
        """Estimate a LogGaussianDistribution from log-scale sufficient statistics.

        Performs the Gaussian estimate (conjugate NormalGamma update with
        posterior-carried-as-prior when available) on the log-scale
        statistics and rewraps the result as a LogGaussianDistribution,
        guarding sigma2 to stay positive.

        Args:
            suff_stat: Tuple as returned by LogGaussianAccumulator.value().

        Returns:
            LogGaussianDistribution object.

        """
        gd = super().estimate(suff_stat)
        sigma2 = gd.sigma2 if gd.sigma2 > 0 else 1.0
        return LogGaussianDistribution(gd.mu, sigma2, name=self.name, prior=gd.prior)
