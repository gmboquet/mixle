"""Normal-Gamma distribution over (mu, tau) for a Gaussian with unknown mean
and precision.

    tau ~ Gamma(a, b),  mu | tau ~ Gaussian(mu0, 1/(lam*tau))

Data type: (Tuple[float, float]): A pair (mu, tau) with tau > 0; the
    log-density is
    log(f(mu, tau)) = a*log(b) + 0.5*log(lam/(2*pi)) - gammaln(a)
    + (a - 0.5)*log(tau) - b*tau - 0.5*lam*tau*(mu - mu0)^2.

This is the conjugate prior for the univariate Gaussian (see
pysp.bstats.gaussian) and the d=1 special case of NormalWishartDistribution
(nu = 2a, W = 1/(2b)).
"""

import numpy as np
import scipy.integrate

from pysp.bstats.pdist import ProbabilityDistribution
from pysp.utils.special import digamma, gammaln


class NormalGammaDistribution(ProbabilityDistribution):
    """Normal-Gamma distribution over (mu, tau); conjugate prior for the
    univariate Gaussian with unknown mean and precision."""

    def __init__(
        self,
        mu: float,
        lam: float,
        a: float,
        b: float,
        name: str | None = None,
        prior: ProbabilityDistribution | None = None,
    ):
        """NormalGammaDistribution object.

        Args:
                mu (float): Prior mean mu0.
                lam (float): Mean-precision scale lam > 0.
                a (float): Gamma shape a > 0.
                b (float): Gamma rate b > 0.
                name (Optional[str]): Name of object.
                prior (Optional[ProbabilityDistribution]): Hyper-prior (stored
                        for interface compatibility).

        """
        self.mu = mu
        self.lam = lam
        self.a = a
        self.b = b
        self.parents = []
        self.name = name
        self.prior = prior

    def __str__(self):
        return "NormalGammaDistribution(%f, %f, %f, %f, name=%s, prior=%s)" % (
            self.mu,
            self.lam,
            self.a,
            self.b,
            self.name,
            str(self.prior),
        )

    def get_parameters(self):
        """Returns the parameter tuple (mu, lam, a, b)."""
        return self.mu, self.lam, self.a, self.b

    def set_parameters(self, params):
        """Set the parameters from a tuple.

        Args:
                params: Tuple (mu, lam, a, b).

        """
        self.mu = params[0]
        self.lam = params[1]
        self.a = params[2]
        self.b = params[3]

    def cross_entropy(self, dist):
        """Cross-entropy H(self, dist) = -E_self[log dist].

        Closed form for a NormalGamma argument; numerical double
        integration otherwise.

        Args:
                dist: Distribution to evaluate against.

        Returns:
                Cross-entropy in nats.

        """
        if isinstance(dist, NormalGammaDistribution):
            a = self.a
            b = self.b
            m = self.mu
            l = self.lam

            aa = dist.a
            bb = dist.b
            mm = dist.mu
            ll = dist.lam

            c1 = np.log(bb) * aa + 0.5 * np.log(ll) - gammaln(aa) - 0.5 * np.log(2 * np.pi)
            c2 = (aa - 0.5) * (digamma(a) - np.log(b)) - bb * (a / b)
            c3 = -0.5 * ll * ((1 / l) + m * m * a / b - 2 * mm * m * a / b + mm * mm * a / b)
            return -(c1 + c2 + c3)
        else:
            lf2 = lambda x, y: dist.log_density((x, y)) * self.density((x, y))
            lf1 = lambda x, y: dist.log_density((-x, y)) * self.density((-x, y))
            a1 = scipy.integrate.dblquad(lf1, 0, np.inf, lambda u: 0, lambda u: np.inf)
            a2 = scipy.integrate.dblquad(lf2, 0, np.inf, lambda u: 0, lambda u: np.inf)
            return -(a1[0] + a2[0])

    def entropy(self) -> float:
        """Returns the entropy of the Normal-Gamma distribution (in nats)."""
        a = self.a
        b = self.b
        lam = self.lam

        return -(
            (a - 0.5) * (digamma(a) - np.log(b))
            - a
            - 0.5
            + np.log(b) * a
            + 0.5 * np.log(lam)
            - gammaln(a)
            - 0.5 * np.log(2 * np.pi)
        )

    def density(self, x: (float, float)) -> float:
        """Density at x = (mu, tau); see log_density().

        Args:
                x: Tuple (mu, tau) with tau > 0.

        Returns:
                Density at x.

        """
        return np.exp(self.log_density(x))

    def log_density(self, x: (float, float)) -> float:
        """Log-density at x = (mu, tau).

        Args:
                x: Tuple (mu, tau) with tau > 0.

        Returns:
                Log-density at x.

        """
        a = self.a
        b = self.b
        mu = self.mu
        lam = self.lam

        c0 = np.log(b) * a + 0.5 * np.log(lam / (2 * np.pi)) - gammaln(a)
        c1 = np.log(x[1]) * (a - 0.5) - b * x[1]
        c2 = -lam * x[1] * (x[0] - mu) * (x[0] - mu) / 2
        return c0 + c1 + c2

    def sampler(self, seed: int | None = None):
        """Create a NormalGammaSampler for this distribution.

        Args:
                seed (Optional[int]): Seed for the random number generator.

        Returns:
                NormalGammaSampler object.

        """
        return NormalGammaSampler(self, seed)


class NormalGammaSampler:
    """Draws (mu, tau) samples from a NormalGammaDistribution."""

    def __init__(self, dist: NormalGammaDistribution, seed: int | None = None):
        """NormalGammaSampler object.

        Args:
                dist (NormalGammaDistribution): Distribution to sample from.
                seed (Optional[int]): Seed for the random number generator.

        """
        self.dist = dist
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.grng = np.random.RandomState(self.rng.randint(0, 2**31 - 1))
        self.nrng = np.random.RandomState(self.rng.randint(0, 2**31 - 1))

    def sample(self, size=None):
        """Draw size samples (a single (mu, tau) pair when size is None).

        Args:
                size (Optional[int]): Number of samples to draw.

        Returns:
                A tuple (mu, tau) if size is None, else a list of size such tuples.

        """
        if size is None:
            t = self.grng.gamma(self.dist.a, 1 / self.dist.b)
            x = self.nrng.normal(self.dist.mu, np.sqrt(1 / (self.dist.lam * t)))
            return x, t
        else:
            return [self.sample() for i in range(size)]
