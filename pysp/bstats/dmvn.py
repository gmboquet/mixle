"""Bayesian diagonal-covariance multivariate Gaussian distribution.

Defines the DiagonalGaussianDistribution, DiagonalGaussianSampler,
DiagonalGaussianAccumulator, and DiagonalGaussianEstimator classes for use
with pysp.bstats.

Data type: (Sequence[float] or np.ndarray): A length-d vector x with
    independent components x_i ~ Gaussian(mu_i, covar_i), i.e. log-density
    log(f(x; mu, covar)) = sum_i [-0.5*log(2*pi*covar_i)
    - 0.5*(x_i - mu_i)^2/covar_i].

Conjugate prior: MultivariateNormalGammaDistribution, a vector of
independent NormalGamma priors over each (mu_i, tau_i = 1/covar_i).
When the prior is conjugate, estimation performs the per-component
NormalGamma posterior update, returns the joint MAP estimate
(covar_i = b_i/(a_i - 1/2)), and carries the posterior as the new prior.
expected_log_density evaluates the variational Bayes expectation
E_q[log p(x | mu, tau)] under the prior.
"""

from collections.abc import Sequence

import numpy as np

import pysp.utils.vector as vec
from pysp.bstats.mvngamma import MultivariateNormalGammaDistribution
from pysp.bstats.pdist import ParameterEstimator, ProbabilityDistribution, StatisticAccumulator
from pysp.utils.special import digamma

DatumType = Sequence[float] | np.ndarray
ParamType = tuple[np.ndarray, np.ndarray]


class DiagonalGaussianDistribution(ProbabilityDistribution):
    """Multivariate Gaussian with diagonal covariance, optionally carrying a
    MultivariateNormalGamma conjugate prior over (mu, 1/covar)."""

    def __init__(
        self,
        mu: Sequence[float] | np.ndarray,
        covariance: Sequence[float] | np.ndarray,
        name: str | None = None,
        prior: ProbabilityDistribution | None = None,
    ):
        """DiagonalGaussianDistribution object with mean mu and diagonal covariance.

        Args:
            mu (Union[Sequence[float], np.ndarray]): Length-d mean vector.
            covariance (Union[Sequence[float], np.ndarray]): Length-d vector
                of positive component variances (the covariance diagonal).
            name (Optional[str]): Name of object.
            prior (Optional[ProbabilityDistribution]): Prior on the
                parameters; defaults to a vague MultivariateNormalGamma,
                which enables the conjugate machinery (see set_prior()).

        """
        if prior is None:
            n = len(mu)
            prior = MultivariateNormalGammaDistribution(
                np.zeros(n), np.ones(n) * 1.0e-8, np.ones(n) * 0.500001, np.ones(n) * 1.0
            )

        self.conj_prior_params = None
        self.expected_nparams = None
        self.has_conj_prior = False
        self.prior = None

        self.dim = len(mu)
        self.mu = np.asarray(mu, dtype=float)
        self.covar = np.asarray(covariance, dtype=float)
        self.name = name
        self.log_c = -0.5 * (np.log(2.0 * np.pi) * self.dim + np.log(self.covar).sum())

        self.ca = -0.5 / self.covar
        self.cb = self.mu / self.covar
        self.cc = (-0.5 * self.mu * self.mu / self.covar).sum() + self.log_c

        self.set_prior(prior)

    def __str__(self):
        mu_str = ",".join(map(str, self.mu.flatten()))
        co_str = ",".join(map(str, self.covar.flatten()))
        return "DiagonalGaussianDistribution([%s], [%s], name=%s, prior=%s)" % (
            mu_str,
            co_str,
            str(self.name),
            str(self.prior),
        )

    def get_prior(self) -> ProbabilityDistribution:
        """Returns the prior distribution on (mu, tau=1/covar)."""
        return self.prior

    def set_prior(self, prior: ProbabilityDistribution) -> None:
        """Set the prior and precompute conjugate-prior expectations.

        If prior is a MultivariateNormalGammaDistribution(mu0, lam, a, b),
        this caches the expected natural parameters [ea, eb, e1, e2] with
        e1 = E[mu*tau] and e2 = -0.5*E[tau] per component (ea, eb scalars
        summed over components), so that expected_log_density(x) =
        x.e1 + (x*x).e2 - ea + eb. Sets has_conj_prior accordingly.

        Args:
            prior (ProbabilityDistribution): Prior on the parameters.

        """
        self.prior = prior

        if isinstance(prior, MultivariateNormalGammaDistribution):
            mu, lam, a, b = prior.get_parameters()

            ea = np.sum((mu * mu) * (a / b) * 0.5 + (0.5 / lam) + 0.5 * (np.log(b) - digamma(a)))
            e1 = mu * a / b
            e2 = -0.5 * a / b
            eb = -0.5 * np.log(2 * np.pi) * self.dim

            self.conj_prior_params = [mu, lam, a, b]
            self.expected_nparams = [ea, eb, e1, e2]
            self.has_conj_prior = True

        else:
            self.conj_prior_params = None
            self.expected_nparams = None
            self.has_conj_prior = False

    def get_parameters(self) -> ParamType:
        """Returns the parameter tuple (mu, covar)."""
        return self.mu, self.covar

    def set_parameters(self, value: ParamType) -> None:
        """Set the parameters and refresh the cached natural-form constants.

        Args:
            value (ParamType): Tuple (mu, covariance diagonal).

        """
        mu, covariance = value

        self.dim = len(mu)
        self.mu = np.asarray(mu, dtype=float)
        self.covar = np.asarray(covariance, dtype=float)
        self.log_c = -0.5 * (np.log(2.0 * np.pi) * self.dim + np.log(self.covar).sum())

        self.ca = -0.5 / self.covar
        self.cb = self.mu / self.covar
        self.cc = (-0.5 * self.mu * self.mu / self.covar).sum() + self.log_c

    def log_density(self, x):
        """Log-density of the diagonal Gaussian at observation x.

        Args:
            x: Length-d observation vector.

        Returns:
            Log-density at observation x.

        """
        rv = np.dot(x * x, self.ca)
        rv += np.dot(x, self.cb)
        rv += self.cc
        return rv

    def expected_log_density(self, x) -> float:
        """Variational expectation E_q[log p(x | mu, tau)] under the prior.

        Requires a MultivariateNormalGamma conjugate prior; raises otherwise.

        Args:
            x: Length-d observation vector.

        Returns:
            Expected log-density at observation x.

        """
        if self.has_conj_prior:
            ea, eb, e1, e2 = self.expected_nparams
            return np.dot(x, e1) + np.dot(np.power(x, 2), e2) - ea + eb
        else:
            raise Exception("dmvn expected_log_density not implemented.")

    def seq_log_density(self, x):
        """Vectorized log-density at sequence-encoded input x.

        Args:
            x: Encoded tuple (X, X*X) from seq_encode().

        Returns:
            Numpy array of log-densities, one per observation.

        """
        rv = np.dot(x[1], self.ca)
        rv += np.dot(x[0], self.cb)
        rv += self.cc
        return rv

    def seq_expected_log_density(self, x):
        """Vectorized expected_log_density() at sequence-encoded input x.

        Requires a MultivariateNormalGamma conjugate prior; raises otherwise.

        Args:
            x: Encoded tuple (X, X*X) from seq_encode().

        Returns:
            Numpy array of expected log-densities, one per observation.

        """
        if self.has_conj_prior:
            ea, eb, e1, e2 = self.expected_nparams
            return np.dot(x[0], e1) + np.dot(x[1], e2) - ea + eb
        else:
            raise Exception("General seq_expected_log_density not implemented.")

    def seq_encode(self, x) -> tuple[np.ndarray, np.ndarray]:
        """Encode observations into the pair (X, X*X) of (n, d) arrays.

        Args:
            x: Iterable of length-d observation vectors.

        Returns:
            Tuple (X, X*X) for use with seq_ methods.

        """
        xv = np.reshape(x, (-1, self.dim))
        return xv, xv * xv

    def sampler(self, seed=None):
        """Create a DiagonalGaussianSampler for this distribution.

        Args:
            seed (Optional[int]): Seed for the random number generator.

        Returns:
            DiagonalGaussianSampler object.

        """
        return DiagonalGaussianSampler(self, seed)

    def estimator(self):
        """Returns a default-constructed DiagonalGaussianEstimator."""
        return DiagonalGaussianEstimator()


class DiagonalGaussianSampler:
    """Draws samples from a DiagonalGaussianDistribution."""

    def __init__(self, dist, seed=None):
        """DiagonalGaussianSampler object.

        Args:
            dist (DiagonalGaussianDistribution): Distribution to sample from.
            seed (Optional[int]): Seed for the random number generator.

        """
        self.rng = np.random.RandomState(seed)
        self.dist = dist

    def sample(self, size=None):
        """Draw size samples (a single vector when size is None).

        Args:
            size (Optional[int]): Number of samples to draw.

        Returns:
            A list of d floats if size is None, else a list of size such lists.

        """
        if size is None:
            rv = self.rng.standard_normal(size=self.dist.dim) * np.sqrt(self.dist.covar) + self.dist.mu
            return rv.tolist()
        else:
            rv = self.rng.standard_normal(size=(size, self.dist.dim)) * np.sqrt(self.dist.covar) + self.dist.mu
            return [u.tolist() for u in rv]


class DiagonalGaussianAccumulator(StatisticAccumulator):
    """Accumulates diagonal-Gaussian sufficient statistics (per-component
    weighted sums of x and x^2 plus the total weight)."""

    def __init__(self, dim=None):
        """DiagonalGaussianAccumulator object.

        Args:
            dim (Optional[int]): Dimension d; inferred from the first
                observation when None.

        """
        self.dim = dim
        self.count = 0.0

        if dim is not None:
            self.sum = vec.zeros(dim)
            self.sum2 = vec.zeros(dim)
        else:
            self.sum = None
            self.sum2 = None

    def update(self, x, weight, estimate):
        """Accumulate the weighted sufficient statistics of observation x.

        Args:
            x: Length-d observation vector.
            weight (float): Weight of the observation.
            estimate: Current distribution estimate (unused).

        """
        if self.dim is None:
            self.dim = len(x)
            self.sum = vec.zeros(self.dim)
            self.sum2 = vec.zeros(self.dim)

        xWeight = x * weight
        self.count += weight
        self.sum += xWeight
        xWeight *= x
        self.sum2 += xWeight

    def initialize(self, x, weight, rng):
        """Initialize the accumulator with observation x (delegates to update).

        Args:
            x: Length-d observation vector.
            weight (float): Weight of the observation.
            rng: Random number generator (unused).

        """
        self.update(x, weight, None)

    def seq_update(self, x, weights, estimate):
        """Vectorized update() on sequence-encoded data.

        Args:
            x: Encoded tuple (X, X*X) from seq_encode().
            weights (np.ndarray): Weight per observation.
            estimate: Current distribution estimate (unused).

        """
        if self.dim is None:
            self.dim = x[0].shape[1]
            self.sum = vec.zeros(self.dim)
            self.sum2 = vec.zeros(self.dim)

        self.count += weights.sum()
        self.sum += np.dot(x[0].T, weights)
        self.sum2 += np.dot(x[1].T, weights)

    def combine(self, suff_stat):
        """Add another accumulator's sufficient-statistic value into this one.

        Args:
            suff_stat: Tuple (sum, sum2, count) as returned by value().

        Returns:
            This accumulator.

        """
        if suff_stat[0] is not None and self.sum is not None:
            self.sum += suff_stat[0]
            self.sum2 += suff_stat[1]
            self.count += suff_stat[2]

        elif suff_stat[0] is not None and self.sum is None:
            self.sum = suff_stat[0]
            self.sum2 = suff_stat[1]
            self.count = suff_stat[2]

        return self

    def value(self):
        """Returns the sufficient statistics (sum, sum2, count)."""
        return self.sum, self.sum2, self.count

    def from_value(self, x):
        """Set the sufficient statistics from a value() tuple.

        Args:
            x: Tuple (sum, sum2, count) as returned by value().

        """
        self.sum = x[0]
        self.sum2 = x[1]
        self.count = x[2]


class DiagonalGaussianEstimator(ParameterEstimator):
    """Estimates a DiagonalGaussianDistribution from sufficient statistics,
    using per-component conjugate NormalGamma updates when the prior allows it."""

    def __init__(self, dim: int | None = None, name: str | None = None, prior: ProbabilityDistribution | None = None):
        """DiagonalGaussianEstimator object.

        Args:
            dim (Optional[int]): Dimension d; when given without a prior, a
                vague MultivariateNormalGamma prior of that dimension is used.
            name (Optional[str]): Name of the estimated distribution.
            prior (Optional[ProbabilityDistribution]): Prior on the
                parameters; a MultivariateNormalGammaDistribution enables
                the conjugate update.

        """
        if (prior is None) and (dim is not None):
            prior = MultivariateNormalGammaDistribution(
                np.zeros(dim), np.ones(dim) * 1.0e-8, np.ones(dim) * 0.500001, np.ones(dim) * 1.0
            )

        self.dim = dim
        self.name = name
        self.prior = None
        self.has_conj_prior = None

        self.set_prior(prior)

    def accumulator_factory(self):
        """Returns a factory whose make() creates DiagonalGaussianAccumulator objects."""
        dim = self.dim
        obj = type("", (object,), {"make": lambda o: DiagonalGaussianAccumulator(dim=dim)})()
        return obj

    def set_prior(self, prior):
        """Set the prior and flag whether it admits the conjugate update.

        Args:
            prior (ProbabilityDistribution): Prior on the parameters.

        """
        self.prior = prior
        self.has_conj_prior = isinstance(prior, MultivariateNormalGammaDistribution)

    def get_prior(self):
        """Returns the prior distribution on (mu, tau=1/covar)."""
        return self.prior

    def estimate(self, suff_stat):
        """Estimate a DiagonalGaussianDistribution from sufficient statistics.

        With a MultivariateNormalGamma prior this performs the per-component
        conjugate posterior update, returns the joint MAP estimate
        (posterior-mean mu and covar_i = b_i/(a_i - 1/2)), and carries the
        posterior forward as the prior of the returned distribution.
        Otherwise the maximum likelihood estimates are returned.

        Args:
            suff_stat: Tuple (sum, sum2, count) as returned by
                DiagonalGaussianAccumulator.value().

        Returns:
            DiagonalGaussianDistribution object.

        """
        sum_x, sum_xx, nobs_loc1 = suff_stat
        sum_xxx = sum_x
        nobs_loc2 = nobs_loc1

        if self.has_conj_prior:
            old_mu, old_lam, old_a, old_b = self.prior.get_parameters()

            new_n = old_lam + nobs_loc1
            new_a = old_a + (nobs_loc2 / 2.0)
            new_nn = old_lam + nobs_loc2

            if nobs_loc1 > 0:
                sample_mean1 = sum_x / nobs_loc1
            else:
                sample_mean1 = 0

            if nobs_loc2 > 0:
                sample_mean2 = sum_xxx / nobs_loc2
            else:
                sample_mean2 = 0

            new_mu = (sum_x + old_mu * old_lam) / (old_lam + nobs_loc1)

            new_b0 = sum_xx - sample_mean2 * sum_xxx
            new_b1 = (old_lam * nobs_loc1 / new_n) * np.power(sample_mean1 - old_mu, 2)
            new_b = old_b + 0.5 * (new_b0 + new_b1)

            new_sigma2 = new_b / (new_a - 0.5)

            new_prior = MultivariateNormalGammaDistribution(new_mu, new_n, new_a, new_b)

            return DiagonalGaussianDistribution(new_mu, new_sigma2, name=self.name, prior=new_prior)

        else:
            if nobs_loc1 == 0:
                mu = 0.0
            else:
                mu = sum_x / nobs_loc1

            if nobs_loc2 == 0:
                sigma2 = 0
            else:
                mu2 = sum_xxx / nobs_loc2
                sigma2 = (sum_xx / nobs_loc2) - mu2 * mu2

            return DiagonalGaussianDistribution(mu, sigma2, name=self.name)
