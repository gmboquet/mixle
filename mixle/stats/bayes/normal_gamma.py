"""Normal-Gamma distribution over (mu, tau) for a Gaussian with unknown mean and precision.

    tau ~ Gamma(a, b),  mu | tau ~ Gaussian(mu0, 1/(lam*tau))

Data type: (Tuple[float, float]): A pair (mu, tau) with tau > 0; the log-density is
    log(f(mu, tau)) = a*log(b) + 0.5*log(lam/(2*pi)) - gammaln(a)
    + (a - 0.5)*log(tau) - b*tau - 0.5*lam*tau*(mu - mu0)^2.

This is the conjugate prior for the univariate :class:`~mixle.stats.univariate.continuous.gaussian.GaussianDistribution`
(see its ``prior=`` argument) and the d=1 special case of NormalWishart (nu = 2a, W = 1/(2b)).
It is a parameter prior: it is scored on ``(mu, tau)`` parameter pairs, not fit from data by EM.
"""

import operator
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
        self.mu, self.lam, self.a, self.b = self._validated_parameters((mu, lam, a, b))
        self.name = name
        self.prior = prior

    @staticmethod
    def _validated_parameters(params: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        if not isinstance(params, (tuple, list)) or len(params) != 4:
            raise ValueError("NormalGammaDistribution parameters must be a four-item tuple.")
        if any(np.ndim(value) != 0 for value in params):
            raise ValueError("NormalGammaDistribution parameters must be scalars.")
        try:
            mu, lam, a, b = (float(value) for value in params)
        except (TypeError, ValueError) as exc:
            raise ValueError("NormalGammaDistribution parameters must be numeric scalars.") from exc
        if not np.isfinite(mu):
            raise ValueError("NormalGammaDistribution requires a finite prior mean.")
        if not np.isfinite(lam) or lam <= 0.0:
            raise ValueError("NormalGammaDistribution requires finite lam > 0.")
        if not np.isfinite(a) or a <= 0.0:
            raise ValueError("NormalGammaDistribution requires finite a > 0.")
        if not np.isfinite(b) or b <= 0.0:
            raise ValueError("NormalGammaDistribution requires finite b > 0.")
        return mu, lam, a, b

    @staticmethod
    def _validated_observation(x: Any) -> tuple[float, float]:
        if not isinstance(x, (tuple, list, np.ndarray)):
            raise ValueError("NormalGammaDistribution observations must be (mu, tau) pairs.")
        try:
            values = np.asarray(x, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("NormalGammaDistribution observations must be numeric.") from exc
        if values.shape != (2,) or np.any(~np.isfinite(values)):
            raise ValueError("NormalGammaDistribution observations must be finite vectors of length 2.")
        if values[1] <= 0.0:
            raise ValueError("NormalGammaDistribution observation precision tau must be positive.")
        return float(values[0]), float(values[1])

    def _validated_state(self) -> tuple[float, float, float, float]:
        return self._validated_parameters(self.get_parameters())

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
        self.mu, self.lam, self.a, self.b = self._validated_parameters(params)

    def cross_entropy(self, dist: "NormalGammaDistribution") -> float:
        """Cross-entropy H(self, dist) = -E_self[log dist].

        Closed form for a NormalGamma argument; numerical double integration otherwise.
        """
        if isinstance(dist, NormalGammaDistribution):
            m, lam, a, b = self._validated_state()
            mm, ll, aa, bb = dist._validated_state()

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
        _, lam, a, b = self._validated_state()

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
        mu, lam, a, b = self._validated_state()
        observed_mu, tau = self._validated_observation(x)

        c0 = np.log(b) * a + 0.5 * np.log(lam / (2 * np.pi)) - gammaln(a)
        c1 = np.log(tau) * (a - 0.5) - b * tau
        c2 = -lam * tau * (observed_mu - mu) * (observed_mu - mu) / 2
        return float(c0 + c1 + c2)

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Vectorized log-density at sequence-encoded (n, 2) array of (mu, tau) rows."""
        mu0, lam, a, b = self._validated_state()
        try:
            values = np.asarray(x, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("NormalGammaDistribution encoded observations must be numeric.") from exc
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError("NormalGammaDistribution encoded observations must have exact shape (n, 2).")
        if np.any(~np.isfinite(values)):
            raise ValueError("NormalGammaDistribution encoded observations must be finite.")
        m = values[:, 0]
        tau = values[:, 1]
        if np.any(tau <= 0.0):
            raise ValueError("NormalGammaDistribution encoded precisions must be positive.")
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
        mu, lam, a, b = self.dist._validated_state()
        if size is None:
            t = self.grng.gamma(a, 1 / b)
            x = self.nrng.normal(mu, np.sqrt(1 / (lam * t)))
            return x, t
        if isinstance(size, (bool, np.bool_)):
            raise TypeError("NormalGammaSampler size must be a non-negative integer.")
        try:
            count = operator.index(size)
        except TypeError as exc:
            raise TypeError("NormalGammaSampler size must be a non-negative integer.") from exc
        if count < 0:
            raise ValueError("NormalGammaSampler size must be non-negative.")
        return [self.sample() for _ in range(count)]


class NormalGammaDataEncoder(DataSequenceEncoder):
    """Encodes a sequence of (mu, tau) parameter pairs into an (n, 2) float array."""

    def __str__(self) -> str:
        return "NormalGammaDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, NormalGammaDataEncoder)

    def seq_encode(self, x: Any) -> np.ndarray:
        """Encode Normal-Gamma observations as a floating array."""
        try:
            values = np.asarray(x, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("NormalGammaDataEncoder observations must be numeric.") from exc
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError("NormalGammaDataEncoder observations must have exact shape (n, 2).")
        if np.any(~np.isfinite(values)) or np.any(values[:, 1] <= 0.0):
            raise ValueError("NormalGammaDataEncoder requires finite means and positive precisions.")
        return values
