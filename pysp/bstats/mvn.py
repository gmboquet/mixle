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

import numpy as np
from numpy.random import RandomState

from pysp.bstats.normwishart import NormalWishartDistribution
from pysp.bstats.pdist import ParameterEstimator, ProbabilityDistribution, StatisticAccumulator


def default_prior(dim: int) -> NormalWishartDistribution:
    # d-dimensional analogue of NormalGamma(0, 1e-8, 0.500001, 1.0):
    # nu = 2a + (d-1), W = (2b)^-1 * I
    return NormalWishartDistribution(np.zeros(dim), 1.0e-8, np.eye(dim) * 0.5, dim + 2.0e-6)


class MultivariateGaussianDistribution(ProbabilityDistribution):
    """Multivariate Gaussian with mean mu and covariance covar, optionally
    carrying a NormalWishart conjugate prior over (mu, Lambda=Sigma^-1)."""

    def __init__(self, mu, covar, name: str | None = None, prior: ProbabilityDistribution | None = None):
        """MultivariateGaussianDistribution object with mean mu and covariance covar.

        Args:
            mu: Length-d mean vector.
            covar: (d, d) positive-definite covariance matrix.
            name (Optional[str]): Name of object.
            prior (Optional[ProbabilityDistribution]): Prior on the
                parameters; defaults to a vague NormalWishart, which enables
                the conjugate machinery (see set_prior()).

        """
        self.name = name
        self.set_parameters((mu, covar))
        self.set_prior(prior if prior is not None else default_prior(self.dim))

    def __str__(self):
        mu = ",".join(map(str, self.mu.tolist()))
        co = ",".join(map(str, self.covar.flatten().tolist()))
        return "MultivariateGaussianDistribution([%s], [%s], name=%s, prior=%s)" % (mu, co, self.name, str(self.prior))

    def get_parameters(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns the parameter tuple (mu, covar)."""
        return self.mu, self.covar

    def set_parameters(self, params) -> None:
        """Set the parameters and refresh the cached precision and constants.

        Args:
            params: Tuple (mu, covar) with covar positive definite.

        """
        mu, covar = params

        self.mu = np.asarray(mu, dtype=float)
        self.covar = np.asarray(covar, dtype=float)
        self.dim = len(self.mu)

        sgn, self.log_det_covar = np.linalg.slogdet(self.covar)
        assert sgn > 0, "Covariance matrix must be positive definite."
        self.precision = np.linalg.inv(self.covar)
        self.log_const = -0.5 * (self.dim * np.log(2.0 * np.pi) + self.log_det_covar)

    def get_prior(self) -> ProbabilityDistribution:
        """Returns the prior distribution on (mu, Lambda=Sigma^-1)."""
        return self.prior

    def set_prior(self, prior: ProbabilityDistribution) -> None:
        """Set the prior and precompute conjugate-prior expectations.

        If prior is a NormalWishartDistribution(m0, kappa, W, nu) over
        (mu, Lambda=Sigma^-1), this caches its parameters and
        E[ln|Lambda|], the quantities needed by expected_log_density.
        Sets has_conj_prior accordingly.

        Args:
            prior (ProbabilityDistribution): Prior on the parameters.

        """
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
        """Density of the multivariate Gaussian at observation x.

        Args:
            x: Length-d observation vector.

        Returns:
            Density at observation x.

        """
        return np.exp(self.log_density(x))

    def log_density(self, x) -> float:
        """Log-density of the multivariate Gaussian at observation x.

        Args:
            x: Length-d observation vector.

        Returns:
            Log-density at observation x.

        """
        diff = np.asarray(x, dtype=float) - self.mu
        return self.log_const - 0.5 * float(np.dot(diff, np.dot(self.precision, diff)))

    def expected_log_density(self, x) -> float:
        """Variational expectation E_q[log p(x | mu, Lambda)] under the prior.

        With a NormalWishart prior this is the standard VB expected
        log-likelihood term (Bishop 10.64/10.67): 0.5*E[ln|Lambda|]
        - 0.5*d*ln(2*pi) - 0.5*(d/kappa + nu*(x-m0)'W(x-m0)). Without a
        conjugate prior it falls back to the plug-in log_density(x).

        Args:
            x: Length-d observation vector.

        Returns:
            Expected log-density at observation x.

        """
        if self.has_conj_prior:
            m0, kappa, w_mat, nu = self.conj_prior_params
            diff = np.asarray(x, dtype=float) - m0
            e_quad = self.dim / kappa + nu * float(np.dot(diff, np.dot(w_mat, diff)))
            return 0.5 * self.e_log_det - 0.5 * self.dim * np.log(2.0 * np.pi) - 0.5 * e_quad
        else:
            return self.log_density(x)

    def seq_encode(self, x) -> np.ndarray:
        """Encode observations into an (n, d) float numpy array.

        Args:
            x: Iterable of length-d observation vectors.

        Returns:
            (n, d) numpy array for use with seq_ methods.

        """
        rv = np.reshape(np.asarray(x, dtype=float), (-1, self.dim))
        return rv

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Vectorized log-density at sequence-encoded input x.

        Args:
            x (np.ndarray): Encoded observations from seq_encode().

        Returns:
            Numpy array of log-densities, one per observation.

        """
        diff = x - self.mu
        rv = -0.5 * np.sum(np.dot(diff, self.precision) * diff, axis=1)
        rv += self.log_const
        return rv

    def seq_expected_log_density(self, x: np.ndarray) -> np.ndarray:
        """Vectorized expected_log_density() at sequence-encoded input x.

        Args:
            x (np.ndarray): Encoded observations from seq_encode().

        Returns:
            Numpy array of expected log-densities, one per observation.

        """
        if self.has_conj_prior:
            m0, kappa, w_mat, nu = self.conj_prior_params
            diff = x - m0
            e_quad = self.dim / kappa + nu * np.sum(np.dot(diff, w_mat) * diff, axis=1)
            return 0.5 * self.e_log_det - 0.5 * self.dim * np.log(2.0 * np.pi) - 0.5 * e_quad
        else:
            return self.seq_log_density(x)

    def sampler(self, seed: int | None = None):
        """Create a MultivariateGaussianSampler for this distribution.

        Args:
            seed (Optional[int]): Seed for the random number generator.

        Returns:
            MultivariateGaussianSampler object.

        """
        return MultivariateGaussianSampler(self, seed)

    def estimator(self):
        """Create a MultivariateGaussianEstimator with this distribution's
        dimension, name, and prior.

        Returns:
            MultivariateGaussianEstimator object.

        """
        return MultivariateGaussianEstimator(self.dim, name=self.name, prior=self.prior)


class MultivariateGaussianSampler:
    """Draws samples from a MultivariateGaussianDistribution."""

    def __init__(self, dist: MultivariateGaussianDistribution, seed: int | None = None):
        """MultivariateGaussianSampler object.

        Args:
            dist (MultivariateGaussianDistribution): Distribution to sample from.
            seed (Optional[int]): Seed for the random number generator.

        """
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size=None):
        """Draw size samples (a single vector when size is None).

        Args:
            size (Optional[int]): Number of samples to draw.

        Returns:
            A length-d numpy array if size is None, else a list of size such arrays.

        """
        rv = self.rng.multivariate_normal(self.dist.mu, self.dist.covar, size=size)
        if size is None:
            return rv
        return list(rv)


class MultivariateGaussianAccumulator(StatisticAccumulator):
    """Accumulates multivariate Gaussian sufficient statistics (weighted sum
    of x, weighted sum of outer products xx', and the total weight)."""

    def __init__(self, dim: int, name=None, keys=None):
        """MultivariateGaussianAccumulator object.

        Args:
            dim (int): Dimension d of the observations.
            name (Optional[str]): Name of the accumulator.
            keys (Optional[str]): Key for sharing sufficient statistics.

        """
        self.dim = dim
        self.name = name
        self.key = keys
        self.sum = np.zeros(dim)
        self.sum_outer = np.zeros((dim, dim))
        self.count = 0.0

    def initialize(self, x, weight, rng):
        """Initialize the accumulator with observation x (delegates to update).

        Args:
            x: Length-d observation vector.
            weight (float): Weight of the observation.
            rng: Random number generator (unused).

        """
        self.update(x, weight, None)

    def seq_initialize(self, x, weights, rng):
        """Vectorized initialize() on sequence-encoded data (delegates to seq_update).

        Args:
            x (np.ndarray): Encoded observations.
            weights (np.ndarray): Weight per observation.
            rng: Random number generator (unused).

        """
        self.seq_update(x, weights, None)

    def update(self, x, weight, estimate):
        """Accumulate the weighted sufficient statistics of observation x.

        Args:
            x: Length-d observation vector.
            weight (float): Weight of the observation.
            estimate: Current distribution estimate (unused).

        """
        xv = np.asarray(x, dtype=float)
        self.sum += xv * weight
        self.sum_outer += np.outer(xv, xv) * weight
        self.count += weight

    def seq_update(self, x, weights, estimate):
        """Vectorized update() on sequence-encoded data.

        Args:
            x (np.ndarray): Encoded observations.
            weights (np.ndarray): Weight per observation.
            estimate: Current distribution estimate (unused).

        """
        self.sum += np.dot(x.T, weights)
        self.sum_outer += np.dot(x.T * weights, x)
        self.count += weights.sum()

    def combine(self, suff_stat):
        """Add another accumulator's sufficient-statistic value into this one.

        Args:
            suff_stat: Tuple (count, sum, sum_outer) as returned by value().

        Returns:
            This accumulator.

        """
        self.count += suff_stat[0]
        self.sum += suff_stat[1]
        self.sum_outer += suff_stat[2]
        return self

    def value(self):
        """Returns the sufficient statistics (count, sum, sum_outer)."""
        return self.count, self.sum, self.sum_outer

    def from_value(self, x):
        """Set the sufficient statistics from a value() tuple.

        Args:
            x: Tuple (count, sum, sum_outer) as returned by value().

        Returns:
            This accumulator.

        """
        self.count = x[0]
        self.sum = x[1]
        self.sum_outer = x[2]
        return self

    def key_merge(self, stats_dict):
        """Merge this accumulator's keyed statistics into a shared dict.

        Args:
            stats_dict (dict): Shared key-to-statistics dictionary.

        """
        if self.key is not None:
            if self.key in stats_dict:
                stats_dict[self.key].combine(self.value())
            else:
                stats_dict[self.key] = self

    def key_replace(self, stats_dict):
        """Replace this accumulator's statistics with the pooled keyed values.

        Args:
            stats_dict (dict): Shared key-to-statistics dictionary.

        """
        if self.key is not None:
            if self.key in stats_dict:
                self.from_value(stats_dict[self.key].value())


class MultivariateGaussianAccumulatorFactory:
    """Factory that creates MultivariateGaussianAccumulator objects."""

    def __init__(self, dim, name, keys):
        """MultivariateGaussianAccumulatorFactory object.

        Args:
            dim (int): Dimension passed to created accumulators.
            name (Optional[str]): Name passed to created accumulators.
            keys (Optional[str]): Key passed to created accumulators.

        """
        self.dim = dim
        self.name = name
        self.keys = keys

    def make(self):
        """Returns a new MultivariateGaussianAccumulator."""
        return MultivariateGaussianAccumulator(self.dim, name=self.name, keys=self.keys)


class MultivariateGaussianEstimator(ParameterEstimator):
    """Estimates a MultivariateGaussianDistribution from sufficient
    statistics, using a conjugate NormalWishart posterior update when the
    prior allows it."""

    def __init__(
        self,
        dim: int,
        name: str | None = None,
        keys: str | None = None,
        prior: ProbabilityDistribution | None = None,
    ):
        """MultivariateGaussianEstimator object.

        Args:
            dim (int): Dimension d of the observations.
            name (Optional[str]): Name of the estimated distribution.
            keys (Optional[str]): Key for sharing sufficient statistics.
            prior (Optional[ProbabilityDistribution]): Prior on
                (mu, Lambda=Sigma^-1); defaults to a vague NormalWishart,
                which enables the conjugate update.

        """
        self.dim = int(dim)
        self.name = name
        self.keys = keys
        self.set_prior(prior if prior is not None else default_prior(self.dim))

    def accumulator_factory(self):
        """Returns a MultivariateGaussianAccumulatorFactory for this estimator."""
        return MultivariateGaussianAccumulatorFactory(self.dim, self.name, self.keys)

    def get_prior(self):
        """Returns the prior distribution on (mu, Lambda=Sigma^-1)."""
        return self.prior

    def set_prior(self, prior):
        """Set the prior and flag whether it admits the conjugate update.

        Args:
            prior (ProbabilityDistribution): Prior on (mu, Lambda=Sigma^-1).

        """
        self.prior = prior
        self.has_conj_prior = isinstance(prior, NormalWishartDistribution)

    def model_log_density(self, model):
        """Log-density of the model parameters under this estimator's prior.

        The NormalWishart prior is over (mu, Lambda) with Lambda = Sigma^-1,
        so the model's covariance is inverted before scoring.

        Args:
            model (MultivariateGaussianDistribution): Model to score.

        Returns:
            Prior log-density of the model parameters.

        """
        # the normal-Wishart prior is over (mu, Lambda) with Lambda = Sigma^-1
        if self.has_conj_prior:
            mu, covar = model.get_parameters()
            return float(self.prior.log_density((mu, np.linalg.inv(covar))))
        return super().model_log_density(model)

    def estimate(self, suff_stat) -> MultivariateGaussianDistribution:
        """Estimate a MultivariateGaussianDistribution from sufficient statistics.

        With a NormalWishart(m0, kappa0, W0, nu0) prior this performs the
        conjugate posterior update (see the module docstring), returns the
        joint MAP estimate (mu = m_n, Sigma = W_n^-1/(nu_n - d), falling
        back to the posterior-mean form W_n^-1/nu_n when nu_n <= d), and
        carries the posterior forward as the prior of the returned
        distribution. Otherwise the maximum likelihood estimates are
        returned (identity covariance when count is zero).

        Args:
            suff_stat: Tuple (count, sum, sum_outer) as returned by
                MultivariateGaussianAccumulator.value().

        Returns:
            MultivariateGaussianDistribution object.

        """
        count, xsum, outer_sum = suff_stat
        d = self.dim

        if self.has_conj_prior:
            m0, kappa0, w0, nu0 = self.prior.get_parameters()

            kappa_n = kappa0 + count
            nu_n = nu0 + count
            m_n = (kappa0 * m0 + xsum) / kappa_n

            if count > 0:
                xbar = xsum / count
                scatter = outer_sum - count * np.outer(xbar, xbar)
                dmu = xbar - m0
                w_n_inv = np.linalg.inv(w0) + scatter + (kappa0 * count / kappa_n) * np.outer(dmu, dmu)
            else:
                w_n_inv = np.linalg.inv(w0)

            # keep the inverse-scale symmetric despite accumulation round-off
            w_n_inv = 0.5 * (w_n_inv + w_n_inv.T)
            w_n = np.linalg.inv(w_n_inv)

            # joint MAP precision is (nu_n - d) W_n for nu_n > d; fall back to
            # the posterior mean nu_n W_n at the boundary
            if nu_n > d:
                covar = w_n_inv / (nu_n - d)
            else:
                covar = w_n_inv / nu_n

            posterior = NormalWishartDistribution(m_n, kappa_n, w_n, nu_n)

            return MultivariateGaussianDistribution(m_n, covar, name=self.name, prior=posterior)

        else:
            if count > 0:
                mu = xsum / count
                covar = outer_sum / count - np.outer(mu, mu)
            else:
                mu = np.zeros(d)
                covar = np.eye(d)

            return MultivariateGaussianDistribution(mu, covar, name=self.name, prior=self.prior)
