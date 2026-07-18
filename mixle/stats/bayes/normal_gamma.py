"""Normal-Gamma distribution over (mu, tau) for a Gaussian with unknown mean and precision.

    tau ~ Gamma(a, b),  mu | tau ~ Gaussian(mu0, 1/(lam*tau))

Data type: (Tuple[float, float]): A pair (mu, tau) with tau > 0; the log-density is
    log(f(mu, tau)) = a*log(b) + 0.5*log(lam/(2*pi)) - gammaln(a)
    + (a - 0.5)*log(tau) - b*tau - 0.5*lam*tau*(mu - mu0)^2.

This is the conjugate prior for the univariate :class:`~mixle.stats.univariate.continuous.gaussian.GaussianDistribution`
(see its ``prior=`` argument) and the d=1 special case of NormalWishart (nu = 2a, W = 1/(2b)).
It is a parameter prior: it is scored on ``(mu, tau)`` parameter pairs, not fit from data by EM.
"""

from typing import Any, Optional

import numpy as np

from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
)
from mixle.utils.special import digamma, gammaln


class NormalGammaDistribution(SequenceEncodableProbabilityDistribution):
    """Normal-Gamma distribution over (mu, tau); conjugate prior for the univariate Gaussian."""

    def __init__(
        self,
        mu: float,
        lam: float,
        a: float,
        b: float,
        name: str | None = None,
        prior: Optional["SequenceEncodableProbabilityDistribution"] = None,
    ) -> None:
        """Create a normal-gamma prior over scalar Gaussian mean and precision.

        Args:
            mu (float): Prior mean mu0.
            lam (float): Mean-precision scale lam > 0.
            a (float): Gamma shape a > 0.
            b (float): Gamma rate b > 0.
            name (Optional[str]): Name of object.
            prior (Optional): Hyper-prior (stored for interface compatibility).

        """
        self.mu = float(mu)
        self.lam = float(lam)
        self.a = float(a)
        self.b = float(b)
        self.name = name
        self.prior = prior

    def __str__(self) -> str:
        return "NormalGammaDistribution(%s, %s, %s, %s, name=%s, prior=%s)" % (
            repr(self.mu),
            repr(self.lam),
            repr(self.a),
            repr(self.b),
            repr(self.name),
            str(self.prior),
        )

    def get_parameters(self) -> tuple[float, float, float, float]:
        """Returns the parameter tuple (mu, lam, a, b)."""
        return self.mu, self.lam, self.a, self.b

    def set_parameters(self, params: tuple[float, float, float, float]) -> None:
        """Set the parameters from a tuple (mu, lam, a, b)."""
        self.mu = float(params[0])
        self.lam = float(params[1])
        self.a = float(params[2])
        self.b = float(params[3])

    def cross_entropy(self, dist: "NormalGammaDistribution") -> float:
        """Cross-entropy H(self, dist) = -E_self[log dist].

        Closed form for a NormalGamma argument; numerical double integration otherwise.
        """
        if isinstance(dist, NormalGammaDistribution):
            a = self.a
            b = self.b
            m = self.mu
            lam = self.lam

            aa = dist.a
            bb = dist.b
            mm = dist.mu
            ll = dist.lam

            c1 = np.log(bb) * aa + 0.5 * np.log(ll) - gammaln(aa) - 0.5 * np.log(2 * np.pi)
            c2 = (aa - 0.5) * (digamma(a) - np.log(b)) - bb * (a / b)
            c3 = -0.5 * ll * ((1 / lam) + m * m * a / b - 2 * mm * m * a / b + mm * mm * a / b)
            return -(c1 + c2 + c3)
        else:
            import scipy.integrate

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

    def density(self, x: tuple[float, float]) -> float:
        """Density at x = (mu, tau); see log_density()."""
        return float(np.exp(self.log_density(x)))

    def log_density(self, x: tuple[float, float]) -> float:
        """Log-density at x = (mu, tau) with tau > 0."""
        a = self.a
        b = self.b
        mu = self.mu
        lam = self.lam

        c0 = np.log(b) * a + 0.5 * np.log(lam / (2 * np.pi)) - gammaln(a)
        c1 = np.log(x[1]) * (a - 0.5) - b * x[1]
        c2 = -lam * x[1] * (x[0] - mu) * (x[0] - mu) / 2
        return float(c0 + c1 + c2)

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Vectorized log-density at sequence-encoded (n, 2) array of (mu, tau) rows."""
        mu0 = self.mu
        b = self.b
        a = self.a
        lam = self.lam
        m = x[:, 0]
        tau = x[:, 1]
        c0 = np.log(b) * a + 0.5 * np.log(lam / (2 * np.pi)) - gammaln(a)
        c1 = np.log(tau) * (a - 0.5) - b * tau
        c2 = -lam * tau * (m - mu0) * (m - mu0) / 2
        return c0 + c1 + c2

    def sampler(self, seed: int | None = None) -> "NormalGammaSampler":
        """Create a NormalGammaSampler for this distribution."""
        return NormalGammaSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "ParameterEstimator":
        """NormalGamma is a parameter prior and is not fit from data by EM."""
        raise NotImplementedError("NormalGammaDistribution is a parameter prior; it has no data estimator.")

    def dist_to_encoder(self) -> "NormalGammaDataEncoder":
        """Return the encoder for ``(mu, tau)`` normal-gamma observations."""
        return NormalGammaDataEncoder()


class NormalGammaSampler(DistributionSampler):
    """Draws (mu, tau) samples from a NormalGammaDistribution."""

    def __init__(self, dist: NormalGammaDistribution, seed: int | None = None) -> None:
        self.dist = dist
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.grng = np.random.RandomState(self.rng.randint(0, 2**31 - 1))
        self.nrng = np.random.RandomState(self.rng.randint(0, 2**31 - 1))

    def sample(self, size: int | None = None, *, batched: bool = True) -> Any:
        """Draw size samples (a single (mu, tau) pair when size is None)."""
        if size is None:
            t = self.grng.gamma(self.dist.a, 1 / self.dist.b)
            x = self.nrng.normal(self.dist.mu, np.sqrt(1 / (self.dist.lam * t)))
            return x, t
        else:
            return [self.sample() for _ in range(size)]


class NormalGammaDataEncoder(DataSequenceEncoder):
    """Encodes a sequence of (mu, tau) parameter pairs into an (n, 2) float array."""

    def __str__(self) -> str:
        return "NormalGammaDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, NormalGammaDataEncoder)

    def seq_encode(self, x: Any) -> np.ndarray:
        """Encode Normal-Gamma observations as a floating array."""
        return np.asarray(x, dtype=float)
