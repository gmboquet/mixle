"""Bayesian Gaussian distribution with mean mu and variance sigma2.

Defines the GaussianDistribution, GaussianSampler, GaussianAccumulator,
GaussianEstimatorAccumulatorFactory, and GaussianEstimator classes for use
with pysp.bstats.

Data type: (float): The GaussianDistribution with mean mu and variance
    sigma2 > 0.0 has log-density
    log(f(x; mu, sigma2)) = -0.5*log(2*pi*sigma2) - 0.5*(x-mu)^2/sigma2,
    for real-valued x.

Conjugate prior: NormalGammaDistribution over (mu, tau) with precision
tau = 1/sigma2. When the prior is a NormalGamma, estimation performs the
closed-form conjugate posterior update, returns the joint MAP estimate
(sigma2 = b/(a - 1/2)), and carries the posterior forward as the prior of
the returned distribution. expected_log_density evaluates the variational
Bayes expectation E_q[log p(x | mu, tau)] under the prior in place of the
plug-in log-density.
"""
from typing import Optional, List, Iterable, TypeVar, Tuple, Union
from pysp.arithmetic import *
from pysp.bstats.normgamma import NormalGammaDistribution
from pysp.bstats.pdist import ProbabilityDistribution, StatisticAccumulator, ParameterEstimator
from numpy.random import RandomState
from pysp.utils.special import digamma, gammaln
import numpy as np

default_prior = NormalGammaDistribution(0.0, 1.0e-8, 0.500001, 1.0)


class GaussianDistribution(ProbabilityDistribution):
    """Gaussian distribution with mean mu and variance sigma2, optionally
    carrying a NormalGamma conjugate prior over (mu, 1/sigma2)."""

    def __init__(self, mu: float, sigma2: float, name: Optional[str] = None, prior: ProbabilityDistribution = default_prior):
        """GaussianDistribution object with mean mu and variance sigma2.

        Args:
            mu (float): Mean. Must be finite.
            sigma2 (float): Variance. Must be positive and finite.
            name (Optional[str]): Name of object.
            prior (ProbabilityDistribution): Prior on the parameters;
                a NormalGammaDistribution over (mu, tau=1/sigma2) enables the
                conjugate machinery (see set_prior()).

        """
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
        """Returns the parameter tuple (mu, sigma2)."""
        return self.mu, self.sigma2

    def set_parameters(self, params: Tuple[float, float]) -> None:
        """Set the parameters and refresh the cached normalizing constants.

        Args:
            params (Tuple[float, float]): Tuple (mu, sigma2).

        """
        self.mu = params[0]
        self.sigma2 = params[1]
        self.logConst = -0.5*log(2.0*pi*self.sigma2)
        self.const    = 1.0/sqrt(2.0*pi*self.sigma2)

    def get_prior(self) -> ProbabilityDistribution:
        """Returns the prior distribution on (mu, tau=1/sigma2)."""
        return self.prior

    def set_prior(self, prior: ProbabilityDistribution) -> None:
        """Set the prior and precompute conjugate-prior expectations.

        If prior is a NormalGammaDistribution(mu0, lam, a, b) over
        (mu, tau=1/sigma2), this caches the expected natural parameters
        [ea, eb, e1, e2] with e1 = E[mu*tau], e2 = -0.5*E[tau],
        ea = E[0.5*mu^2*tau] + 0.5*(1/lam) + 0.5*E[log(1/tau)] and
        eb = -0.5*log(2*pi), so that expected_log_density(x) =
        x*(e1 + x*e2) - ea + eb. Sets has_conj_prior accordingly.

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

    def log_density(self, x: float) -> float:
        """Log-density of the Gaussian at observation x.

        Args:
            x (float): Real-valued observation.

        Returns:
            Log-density at observation x.

        """
        return self.logConst - 0.5*(x-self.mu)*(x-self.mu)/self.sigma2

    def expected_log_density(self, x: float) -> float:
        """Variational expectation E_q[log p(x | mu, tau)] under the prior.

        With a NormalGamma conjugate prior q this is the standard VB
        expected log-likelihood term; without a conjugate prior it falls
        back to the plug-in log_density(x).

        Args:
            x (float): Real-valued observation.

        Returns:
            Expected log-density at observation x.

        """
        if self.has_conj_prior:
            ea, eb, e1, e2 = self.expected_nparams
            return x*(e1 + x*e2) - ea + eb
        else:
            return self.log_density(x)

    def seq_encode(self, x: Iterable[float]) -> np.ndarray:
        """Encode an iterable of observations into a float numpy array.

        Args:
            x (Iterable[float]): Observations.

        Returns:
            Numpy array of floats for use with seq_ methods.

        """
        rv = np.asarray(x, dtype=float)
        return rv

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Vectorized log-density at sequence-encoded input x.

        Args:
            x (np.ndarray): Encoded observations from seq_encode().

        Returns:
            Numpy array of log-densities, one per observation.

        """
        rv = x - self.mu
        rv *= rv
        rv *= -0.5/self.sigma2
        rv += self.logConst
        return rv

    def seq_expected_log_density(self, x: np.ndarray) -> np.ndarray:
        """Vectorized expected_log_density() at sequence-encoded input x.

        Args:
            x (np.ndarray): Encoded observations from seq_encode().

        Returns:
            Numpy array of expected log-densities, one per observation.

        """

        if self.conj_prior_params is not None:

            ea, eb, e1, e2 = self.expected_nparams

            return x*(e1 + x*e2) - ea + eb

        else:
            return self.seq_log_density(x)

    def sampler(self, seed: Optional[int] = None):
        """Create a GaussianSampler for this distribution.

        Args:
            seed (Optional[int]): Seed for the random number generator.

        Returns:
            GaussianSampler object.

        """
        return GaussianSampler(self, seed)

    def estimator(self):
        """Create a GaussianEstimator with this distribution's name and prior.

        Returns:
            GaussianEstimator object.

        """
        return GaussianEstimator(name=self.name, prior=self.prior)


class GaussianSampler(object):
    """Draws samples from a GaussianDistribution."""

    def __init__(self, dist: GaussianDistribution, seed: Optional[int] = None):
        """GaussianSampler object.

        Args:
            dist (GaussianDistribution): Distribution to sample from.
            seed (Optional[int]): Seed for the random number generator.

        """
        self.rng  = RandomState(seed)
        self.dist = dist

    def sample(self, size=None):
        """Draw size samples (a single float when size is None).

        Args:
            size (Optional[int]): Number of samples to draw.

        Returns:
            A float if size is None, else a numpy array of length size.

        """
        return self.rng.normal(loc=self.dist.mu, scale=sqrt(self.dist.sigma2), size=size)


class GaussianAccumulator(StatisticAccumulator):
    """Accumulates Gaussian sufficient statistics (weighted sums of x and
    x^2 with their observation counts, tracked separately per keyed group)."""

    def __init__(self, name=None, keys=(None, None)):
        """GaussianAccumulator object.

        Args:
            name (Optional[str]): Name of the accumulator.
            keys (Tuple[Optional[str], Optional[str]]): Keys for sharing the
                mean statistics (sum, count) and the variance statistics
                (sum2, sum3, count2) across accumulators.

        """
        self.name = name
        self.sum  = 0.0
        self.sum2 = 0.0
        self.sum3 = 0.0
        self.count = 0.0
        self.count2 = 0.0
        self.sum_key = keys[0]
        self.sum2_key = keys[1]

    def initialize(self, x, weight, rng):
        """Initialize the accumulator with observation x (delegates to update).

        Args:
            x (float): Observation.
            weight (float): Weight of the observation.
            rng: Random number generator (unused).

        """
        self.update(x, weight, None)

    def update(self, x, weight, estimate):
        """Accumulate the weighted sufficient statistics of observation x.

        Args:
            x (float): Observation.
            weight (float): Weight of the observation.
            estimate: Current distribution estimate (unused).

        """
        xWeight   = x*weight
        self.sum  += xWeight
        self.sum2 += x*xWeight
        self.sum3 += xWeight
        self.count += weight
        self.count2 += weight

    def seq_initialize(self, x, weights, rng):
        """Vectorized initialize() on sequence-encoded data (delegates to seq_update).

        Args:
            x (np.ndarray): Encoded observations.
            weights (np.ndarray): Weight per observation.
            rng: Random number generator (unused).

        """
        self.seq_update(x, weights, None)

    def seq_update(self, x, weights, estimate):
        """Vectorized update() on sequence-encoded data.

        Args:
            x (np.ndarray): Encoded observations.
            weights (np.ndarray): Weight per observation.
            estimate: Current distribution estimate (unused).

        """
        temp = np.dot(x, weights)
        self.sum += temp
        self.sum2 += np.dot(x*x, weights)
        self.sum3 += temp
        w_sum = weights.sum()
        self.count += w_sum
        self.count2 += w_sum

    def df_initialize(self, df, weights, rng):
        """DataFrame variant of initialize() (delegates to df_update).

        Args:
            df: DataFrame with a column named after this accumulator.
            weights: Weight per row.
            rng: Random number generator (unused).

        """
        self.df_update(df, weights, None)

    def df_update(self, df, weights, estimate):
        """DataFrame variant of update(), reading column self.name.

        Args:
            df: DataFrame with a column named after this accumulator.
            weights: Weight per row.
            estimate: Current distribution estimate (unused).

        """
        col = df[self.name]
        self.sum += col.dot(weights)
        self.sum2 += col.pow(2.0).dot(weights)
        self.sum3 += col.dot(weights)
        w_sum = weights.sum()
        self.count += w_sum
        self.count2 += w_sum

    def combine(self, suff_stat):
        """Add another accumulator's sufficient-statistic value into this one.

        Args:
            suff_stat: Tuple as returned by value().

        Returns:
            This accumulator.

        """
        self.sum  += suff_stat[0]
        self.sum2 += suff_stat[1]
        self.sum3 += suff_stat[2]
        self.count += suff_stat[3]
        self.count2 += suff_stat[4]

        return self

    def value(self):
        """Returns the sufficient statistics (sum, sum2, sum3, count, count2)."""
        return self.sum, self.sum2, self.sum3, self.count, self.count2

    def from_value(self, x):
        """Set the sufficient statistics from a value() tuple.

        Args:
            x: Tuple as returned by value().

        Returns:
            This accumulator.

        """
        self.sum = x[0]
        self.sum2 = x[1]
        self.sum3 = x[2]
        self.count = x[3]
        self.count2 = x[4]
        return self

    def key_merge(self, stats_dict):
        """Merge this accumulator's keyed statistics into a shared dict.

        Statistics under sum_key are pooled additively; statistics under
        sum2_key are pooled with the parallel-variance correction so the
        merged scatter matches a single-pass computation.

        Args:
            stats_dict (dict): Shared key-to-statistics dictionary.

        """
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
        """Replace this accumulator's statistics with the pooled keyed values.

        Args:
            stats_dict (dict): Shared key-to-statistics dictionary.

        """
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
    """Factory that creates GaussianAccumulator objects."""

    def __init__(self, name, keys):
        """GaussianEstimatorAccumulatorFactory object.

        Args:
            name (Optional[str]): Name passed to created accumulators.
            keys: Keys passed to created accumulators.

        """
        self.name = name
        self.keys = keys

    def make(self):
        """Returns a new GaussianAccumulator."""
        return GaussianAccumulator(name=self.name, keys=self.keys)


class GaussianEstimator(ParameterEstimator):
    """Estimates a GaussianDistribution from sufficient statistics, using a
    conjugate NormalGamma posterior update when the prior allows it."""

    def __init__(self, name=None, prior=default_prior, keys=(None, None)):
        """GaussianEstimator object.

        Args:
            name (Optional[str]): Name of the estimated distribution.
            prior (ProbabilityDistribution): Prior on (mu, tau=1/sigma2);
                a NormalGammaDistribution enables the conjugate update.
            keys (Tuple[Optional[str], Optional[str]]): Keys for sharing
                mean and variance statistics across accumulators.

        """
        self.keys  = keys
        self.name  = name
        self.set_prior(prior)

    def accumulator_factory(self):
        """Returns a GaussianEstimatorAccumulatorFactory for this estimator."""
        return GaussianEstimatorAccumulatorFactory(self.name, self.keys)

    def set_prior(self, prior):
        """Set the prior and flag whether it admits the conjugate update.

        Args:
            prior (ProbabilityDistribution): Prior on (mu, tau=1/sigma2).

        """
        self.prior = prior
        self.has_conj_prior = isinstance(prior, NormalGammaDistribution)

    def get_prior(self):
        """Returns the prior distribution on (mu, tau=1/sigma2)."""
        return self.prior

    def model_log_density(self, model):
        """Log-density of the model parameters under this estimator's prior.

        The NormalGamma prior is over (mu, tau) with tau = 1/sigma2, so the
        model's parameters are mapped to that parameterization first.

        Args:
            model (GaussianDistribution): Model to score.

        Returns:
            Prior log-density of the model parameters.

        """
        if self.has_conj_prior:
            mu, sigma2 = model.get_parameters()
            return float(self.prior.log_density((mu, 1.0/sigma2)))
        return super().model_log_density(model)

    def estimate(self, suff_stat):
        """Estimate a GaussianDistribution from sufficient statistics.

        With a NormalGamma(mu0, lam, a, b) prior this performs the conjugate
        posterior update, returns the joint MAP estimate (posterior-mean mu
        and sigma2 = b_n/(a_n - 1/2)), and carries the posterior forward as
        the prior of the returned distribution. Otherwise the maximum
        likelihood estimates are returned.

        Args:
            suff_stat: Tuple (sum_x, sum_xx, sum_xxx, count1, count2) as
                returned by GaussianAccumulator.value().

        Returns:
            GaussianDistribution object.

        """
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

# --- API naming aliases (notes/distribution_api_naming_accounting.md) ---
GaussianAccumulatorFactory = GaussianEstimatorAccumulatorFactory
