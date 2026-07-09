"""Multivariate (factorized) Normal-Gamma distribution over (mu, tau) for a
vector of independent Gaussians with unknown means and precisions.

Each component i is an independent NormalGamma:

    tau_i ~ Gamma(a_i, b_i),  mu_i | tau_i ~ Gaussian(mu0_i, 1/(lam_i*tau_i))

Data type: (Tuple[np.ndarray, np.ndarray]): A pair (mu, tau) of length-d
    vectors; the log-density is the sum of the d univariate NormalGamma
    log-densities.

This is the conjugate prior used by the diagonal
:class:`~mixle.stats.multivariate.diagonal_gaussian.DiagonalGaussianDistribution` (see its ``prior=``
argument) and the vectorized counterpart of NormalGamma. It is a parameter
prior: it is scored on ``(mu, tau)`` parameter pairs, not fit from data by EM.
"""

from collections.abc import Sequence
from typing import Any, Optional

import numpy as np

from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
)
from mixle.utils.special import digamma, gammaln

FlexDatumType = tuple[Sequence[float] | np.ndarray, Sequence[float] | np.ndarray]
FlexParamType = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]

DatumType = tuple[np.ndarray, np.ndarray]
ParamType = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


class MultivariateNormalGammaDistribution(SequenceEncodableProbabilityDistribution):
    """Vector of independent NormalGamma distributions over per-component
    (mu_i, tau_i) pairs; conjugate prior for diagonal Gaussians."""

    def __init__(
        self,
        mu: np.ndarray,
        lam: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        name: str | None = None,
        prior: Optional["SequenceEncodableProbabilityDistribution"] = None,
    ) -> None:
        """Create independent normal-gamma priors for vector-valued Gaussian coordinates.

        Args:
            mu (np.ndarray): Length-d vector of prior means mu0_i.
            lam (np.ndarray): Length-d vector of mean-precision scales lam_i > 0.
            a (np.ndarray): Length-d vector of Gamma shapes a_i > 0.
            b (np.ndarray): Length-d vector of Gamma rates b_i > 0.
            name (Optional[str]): Name of object.
            prior (Optional): Hyper-prior (stored for interface compatibility).

        """
        self.name = name
        self.prior = prior
        self.set_parameters((mu, lam, a, b))

    def __str__(self) -> str:
        mu = ",".join(map(str, self.mu.tolist()))
        lam = ",".join(map(str, self.lam.tolist()))
        a = ",".join(map(str, self.a.tolist()))
        b = ",".join(map(str, self.b.tolist()))

        return "MultivariateNormalGammaDistribution([%s], [%s], [%s], [%s], name=%s, prior=%s)" % (
            mu,
            lam,
            a,
            b,
            self.name,
            str(self.prior),
        )

    def get_parameters(self):
        """Returns the parameter tuple (mu, lam, a, b) of vectors."""
        return self.mu, self.lam, self.a, self.b

    def set_parameters(self, value) -> None:
        """Set the parameters from a tuple of vectors.

        Args:
            value: Tuple (mu, lam, a, b) of length-d arrays.

        """
        mu, lam, a, b = value

        self.mu = np.asarray(mu, dtype=float)
        self.lam = np.asarray(lam, dtype=float)
        self.a = np.asarray(a, dtype=float)
        self.b = np.asarray(b, dtype=float)

    def cross_entropy(self, dist: "MultivariateNormalGammaDistribution") -> float:
        """Cross-entropy H(self, dist) = -E_self[log dist], summed over
        components, for a MultivariateNormalGamma argument."""
        if isinstance(dist, MultivariateNormalGammaDistribution):
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
            return -np.sum(c1 + c2 + c3)
        else:
            raise NotImplementedError(
                "MultivariateNormalGammaDistribution.cross_entropy is only implemented for "
                "MultivariateNormalGammaDistribution arguments (got %s)." % type(dist).__name__
            )

    def entropy(self) -> float:
        """Returns the entropy (in nats), summed over components."""
        a = self.a
        b = self.b
        lam = self.lam

        return -np.sum(
            (a - 0.5) * (digamma(a) - np.log(b))
            - a
            - 0.5
            + np.log(b) * a
            + 0.5 * np.log(lam)
            - gammaln(a)
            - 0.5 * np.log(2 * np.pi)
        )

    def density(self, x: FlexDatumType) -> float:
        """Density at x = (mu, tau); see log_density()."""
        return np.exp(self.log_density(x))

    def log_density(self, x: FlexDatumType) -> float:
        """Log-density at x = (mu, tau), summed over the d components.

        Args:
            x (FlexDatumType): Tuple (mu, tau) of length-d vectors with
                tau_i > 0.

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
        return float(np.sum(c0 + c1 + c2))

    def seq_log_density(self, x) -> np.ndarray:
        """Vectorized log-density over a sequence of (mu, tau) pairs."""
        return np.asarray([self.log_density(xx) for xx in x], dtype=float)

    def sampler(self, seed: int | None = None) -> "MultivariateNormalGammaSampler":
        """Create a MultivariateNormalGammaSampler for this distribution."""
        return MultivariateNormalGammaSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "ParameterEstimator":
        """MultivariateNormalGamma is a parameter prior; not fit from data by EM."""
        raise NotImplementedError("MultivariateNormalGammaDistribution is a parameter prior; it has no data estimator.")

    def dist_to_encoder(self) -> "MultivariateNormalGammaDataEncoder":
        """Returns a MultivariateNormalGammaDataEncoder for encoding (mu, tau) pairs."""
        return MultivariateNormalGammaDataEncoder()


class MultivariateNormalGammaSampler(DistributionSampler):
    """Draws (mu, tau) samples from a MultivariateNormalGammaDistribution."""

    def __init__(self, dist: MultivariateNormalGammaDistribution, seed: int | None = None) -> None:
        self.dist = dist
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.grng = np.random.RandomState(self.rng.randint(0, 2**31 - 1))
        self.nrng = np.random.RandomState(self.rng.randint(0, 2**31 - 1))

    def sample(self, size=None) -> Any:
        """Draw size samples (a single (mu, tau) pair when size is None)."""
        if size is None:
            t = self.grng.gamma(self.dist.a, 1 / self.dist.b)
            x = self.nrng.normal(self.dist.mu, np.sqrt(1 / (self.dist.lam * t)))
            return x, t
        else:
            return [self.sample() for _ in range(size)]


class MultivariateNormalGammaDataEncoder(DataSequenceEncoder):
    """Encodes a sequence of (mu, tau) parameter pairs (identity passthrough)."""

    def __str__(self) -> str:
        return "MultivariateNormalGammaDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, MultivariateNormalGammaDataEncoder)

    def seq_encode(self, x: Any) -> Any:
        """Encode multivariate Normal-Gamma observations as a list payload."""
        return list(x)
