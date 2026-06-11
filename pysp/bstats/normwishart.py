"""Normal-Wishart distribution over (mu, Lambda) for a d-dimensional Gaussian
with unknown mean and precision matrix.

q(mu, Lambda) = N(mu | m, (kappa*Lambda)^{-1}) * Wishart(Lambda | W, nu)

with scale matrix W (d x d positive definite) and degrees of freedom
nu > d - 1. This is the conjugate prior for the multivariate Gaussian and the
d-dimensional generalization of NormalGammaDistribution (d=1: nu = 2a,
W = 1/(2b)).
"""
from typing import Optional

import numpy as np
from scipy.special import gammaln, digamma

from pysp.bstats.pdist import ProbabilityDistribution


def _multigammaln(a: float, d: int) -> float:
    """Log of the d-dimensional multivariate gamma function at a."""
    return d*(d - 1)/4.0*np.log(np.pi) + sum(gammaln(a + (1.0 - i)/2.0) for i in range(1, d + 1))


def _multidigamma(a: float, d: int) -> float:
    """Derivative of _multigammaln with respect to a."""
    return sum(digamma(a + (1.0 - i)/2.0) for i in range(1, d + 1))


class NormalWishartDistribution(ProbabilityDistribution):
    """Normal-Wishart distribution over (mu, Lambda); conjugate prior for the
    multivariate Gaussian with unknown mean and precision matrix."""

    def __init__(self, mu, kappa: float, w_mat, nu: float, name: Optional[str] = None,
                 prior: Optional[ProbabilityDistribution] = None):
        """NormalWishartDistribution object.

        Args:
            mu: Length-d prior mean m.
            kappa (float): Mean-precision scale kappa > 0.
            w_mat: (d, d) positive-definite Wishart scale matrix W.
            nu (float): Degrees of freedom nu > d - 1.
            name (Optional[str]): Name of object.
            prior (Optional[ProbabilityDistribution]): Hyper-prior (stored
                for interface compatibility).

        """
        self.name = name
        self.prior = prior
        self.set_parameters((mu, kappa, w_mat, nu))

    def __str__(self):
        mu = ','.join(map(str, self.mu.tolist()))
        w = ','.join(map(str, self.w_mat.flatten().tolist()))
        return 'NormalWishartDistribution([%s], %f, [%s], %f, name=%s, prior=%s)' % (
            mu, self.kappa, w, self.nu, self.name, str(self.prior))

    def get_parameters(self):
        """Returns the parameter tuple (mu, kappa, w_mat, nu)."""
        return self.mu, self.kappa, self.w_mat, self.nu

    def set_parameters(self, params):
        """Set the parameters and refresh the cached Wishart log-normalizer.

        Args:
            params: Tuple (mu, kappa, w_mat, nu) with w_mat positive
                definite and nu > d - 1.

        """
        mu, kappa, w_mat, nu = params

        self.mu = np.asarray(mu, dtype=float)
        self.kappa = float(kappa)
        self.w_mat = np.asarray(w_mat, dtype=float)
        self.nu = float(nu)
        self.dim = len(self.mu)

        d = self.dim
        assert self.nu > d - 1, 'NormalWishart requires nu > dim - 1.'

        sgn, self.log_det_w = np.linalg.slogdet(self.w_mat)
        assert sgn > 0, 'NormalWishart scale matrix must be positive definite.'
        self.w_inv = np.linalg.inv(self.w_mat)

        # log normalizer of the Wishart factor
        self.log_z = (self.nu*d/2.0)*np.log(2.0) + (self.nu/2.0)*self.log_det_w + _multigammaln(self.nu/2.0, d)

    def expected_log_det(self) -> float:
        """E[ln |Lambda|] under the Wishart factor."""
        return _multidigamma(self.nu/2.0, self.dim) + self.dim*np.log(2.0) + self.log_det_w

    def expected_precision(self) -> np.ndarray:
        """E[Lambda] = nu * W."""
        return self.nu*self.w_mat

    def density(self, x) -> float:
        """Density at x = (mu, Lambda); see log_density().

        Args:
            x: Tuple (mu, Lambda) with Lambda a positive-definite matrix.

        Returns:
            Density at x.

        """
        return np.exp(self.log_density(x))

    def log_density(self, x) -> float:
        """Log density at x = (mu, Lambda) with Lambda a precision matrix.

        Args:
            x: Tuple (mu, Lambda); returns -inf when Lambda is not
                positive definite.

        Returns:
            Log-density at x.

        """
        mu, lam = x
        mu = np.asarray(mu, dtype=float)
        lam = np.asarray(lam, dtype=float)
        d = self.dim

        sgn, log_det_lam = np.linalg.slogdet(lam)
        if sgn <= 0:
            return -np.inf

        diff = mu - self.mu
        c_norm = (d/2.0)*np.log(self.kappa/(2.0*np.pi)) + 0.5*log_det_lam \
                 - 0.5*self.kappa*float(np.dot(diff, np.dot(lam, diff)))
        c_wish = ((self.nu - d - 1.0)/2.0)*log_det_lam - 0.5*float(np.trace(np.dot(self.w_inv, lam))) - self.log_z

        return c_norm + c_wish

    def cross_entropy(self, dist: ProbabilityDistribution) -> float:
        """H(self, dist) = -E_self[log dist] for NormalWishart dist.

        Args:
            dist (ProbabilityDistribution): NormalWishartDistribution to
                evaluate against (other types raise NotImplementedError).

        Returns:
            Cross-entropy in nats.

        """
        if not isinstance(dist, NormalWishartDistribution):
            raise NotImplementedError(
                'NormalWishartDistribution.cross_entropy is only implemented for NormalWishart arguments (got %s).'
                % type(dist).__name__)

        d = self.dim
        e_log_det = self.expected_log_det()
        e_lam = self.expected_precision()

        # E[(mu - m_p)' Lambda (mu - m_p)] under self
        diff = self.mu - dist.mu
        e_quad = d/self.kappa + self.nu*float(np.dot(diff, np.dot(self.w_mat, diff)))

        c_norm = (d/2.0)*np.log(dist.kappa/(2.0*np.pi)) + 0.5*e_log_det - 0.5*dist.kappa*e_quad
        c_wish = ((dist.nu - d - 1.0)/2.0)*e_log_det - 0.5*float(np.trace(np.dot(dist.w_inv, e_lam))) - dist.log_z

        return -(c_norm + c_wish)

    def entropy(self) -> float:
        """Returns the entropy of the Normal-Wishart distribution (in nats)."""
        return self.cross_entropy(self)

    def sampler(self, seed: Optional[int] = None):
        """Create a NormalWishartSampler for this distribution.

        Args:
            seed (Optional[int]): Seed for the random number generator.

        Returns:
            NormalWishartSampler object.

        """
        return NormalWishartSampler(self, seed)


class NormalWishartSampler(object):
    """Draws (mu, Lambda) samples from a NormalWishartDistribution."""

    def __init__(self, dist: NormalWishartDistribution, seed: Optional[int] = None):
        """NormalWishartSampler object.

        Args:
            dist (NormalWishartDistribution): Distribution to sample from.
            seed (Optional[int]): Seed for the random number generator.

        """
        self.dist = dist
        self.rng = np.random.RandomState(seed)

    def sample(self, size=None):
        """Draw size samples (a single (mu, Lambda) pair when size is None).

        Lambda is drawn from the Wishart factor, then mu from
        N(m, (kappa*Lambda)^-1).

        Args:
            size (Optional[int]): Number of samples to draw.

        Returns:
            A tuple (mu, Lambda) if size is None, else a list of size such tuples.

        """
        if size is None:
            d = self.dist.dim
            lam = scipy_wishart_sample(self.rng, self.dist.nu, self.dist.w_mat)
            covar = np.linalg.inv(lam*self.dist.kappa)
            mu = self.rng.multivariate_normal(self.dist.mu, covar)
            return mu, lam
        else:
            return [self.sample() for _ in range(size)]


def scipy_wishart_sample(rng: np.random.RandomState, nu: float, w_mat: np.ndarray) -> np.ndarray:
    """Draw one Wishart(nu, W) sample via the Bartlett decomposition."""
    d = w_mat.shape[0]
    chol = np.linalg.cholesky(w_mat)
    a_mat = np.zeros((d, d))
    for i in range(d):
        a_mat[i, i] = np.sqrt(rng.chisquare(nu - i))
        for j in range(i):
            a_mat[i, j] = rng.normal()
    la = np.dot(chol, a_mat)
    return np.dot(la, la.T)
