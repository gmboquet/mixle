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

import operator
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

    @staticmethod
    def _validated_parameters(value: Any) -> ParamType:
        if not isinstance(value, (tuple, list)) or len(value) != 4:
            raise ValueError("MultivariateNormalGammaDistribution parameters must be a four-item tuple of vectors.")
        converted: list[np.ndarray] = []
        for name, raw in zip(("mu", "lam", "a", "b"), value):
            try:
                vector = np.asarray(raw, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError("MultivariateNormalGammaDistribution %s must be a numeric vector." % name) from exc
            if vector.ndim != 1 or vector.size == 0:
                raise ValueError(
                    "MultivariateNormalGammaDistribution %s must be a non-empty one-dimensional vector." % name
                )
            converted.append(vector.copy())
        expected = converted[0].shape
        if any(vector.shape != expected for vector in converted[1:]):
            raise ValueError("MultivariateNormalGammaDistribution parameter vectors must have equal lengths.")
        mu, lam, a, b = converted
        if np.any(~np.isfinite(mu)):
            raise ValueError("MultivariateNormalGammaDistribution requires finite means.")
        for name, vector in (("lam", lam), ("a", a), ("b", b)):
            if np.any(~np.isfinite(vector)) or np.any(vector <= 0.0):
                raise ValueError("MultivariateNormalGammaDistribution requires finite positive %s values." % name)
        return mu, lam, a, b

    @staticmethod
    def _validated_observation(value: Any, dimension: int) -> DatumType:
        if not isinstance(value, (tuple, list, np.ndarray)) or len(value) != 2:
            raise ValueError("MultivariateNormalGammaDistribution observations must be (mu, tau) vector pairs.")
        converted: list[np.ndarray] = []
        for name, raw in zip(("mu", "tau"), value):
            try:
                vector = np.asarray(raw, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError("MultivariateNormalGammaDistribution observation %s must be numeric." % name) from exc
            if vector.shape != (dimension,) or np.any(~np.isfinite(vector)):
                raise ValueError(
                    "MultivariateNormalGammaDistribution observation %s must be a finite vector "
                    "of length %d." % (name, dimension)
                )
            converted.append(vector)
        if np.any(converted[1] <= 0.0):
            raise ValueError("MultivariateNormalGammaDistribution observation precisions tau must be positive.")
        return converted[0], converted[1]

    def _validated_state(self) -> ParamType:
        return self._validated_parameters((self.mu, self.lam, self.a, self.b))

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
        return self.mu.copy(), self.lam.copy(), self.a.copy(), self.b.copy()

    def set_parameters(self, value) -> None:
        """Set the parameters from a tuple of vectors.

        Args:
            value: Tuple (mu, lam, a, b) of length-d arrays.

        """
        mu, lam, a, b = self._validated_parameters(value)
        self.mu, self.lam, self.a, self.b = mu, lam, a, b

    def cross_entropy(self, dist: "MultivariateNormalGammaDistribution") -> float:
        """Cross-entropy H(self, dist) = -E_self[log dist], summed over
        components, for a MultivariateNormalGamma argument."""
        if isinstance(dist, MultivariateNormalGammaDistribution):
            m, l, a, b = self._validated_state()
            mm, ll, aa, bb = dist._validated_state()
            if m.shape != mm.shape:
                raise ValueError("MultivariateNormalGammaDistribution cross-entropy requires equal dimensions.")

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
        _, lam, a, b = self._validated_state()

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
        mu, lam, a, b = self._validated_state()
        observed_mu, tau = self._validated_observation(x, len(mu))

        c0 = np.log(b) * a + 0.5 * np.log(lam / (2 * np.pi)) - gammaln(a)
        c1 = np.log(tau) * (a - 0.5) - b * tau
        c2 = -lam * tau * (observed_mu - mu) * (observed_mu - mu) / 2
        return float(np.sum(c0 + c1 + c2))

    def seq_log_density(self, x) -> np.ndarray:
        """Vectorized log-density over a sequence of (mu, tau) pairs."""
        self._validated_state()
        return np.asarray([self.log_density(xx) for xx in x], dtype=float)

    def sampler(self, seed: int | None = None) -> "MultivariateNormalGammaSampler":
        """Create a MultivariateNormalGammaSampler for this distribution."""
        return MultivariateNormalGammaSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "ParameterEstimator":
        """MultivariateNormalGamma is a parameter prior; not fit from data by EM."""
        raise NotImplementedError("MultivariateNormalGammaDistribution is a parameter prior; it has no data estimator.")

    def dist_to_encoder(self) -> "MultivariateNormalGammaDataEncoder":
        """Returns a MultivariateNormalGammaDataEncoder for encoding (mu, tau) pairs."""
        return MultivariateNormalGammaDataEncoder(len(self.mu))


class MultivariateNormalGammaSampler(DistributionSampler):
    """Draws (mu, tau) samples from a MultivariateNormalGammaDistribution."""

    def __init__(self, dist: MultivariateNormalGammaDistribution, seed: int | None = None) -> None:
        self.dist = dist
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.grng = np.random.RandomState(self.rng.randint(0, 2**31 - 1))
        self.nrng = np.random.RandomState(self.rng.randint(0, 2**31 - 1))

    def sample(self, size=None, *, batched: bool = True) -> Any:
        """Draw size samples (a single (mu, tau) pair when size is None)."""
        mu, lam, a, b = self.dist._validated_state()
        if size is None:
            t = self.grng.gamma(a, 1 / b)
            x = self.nrng.normal(mu, np.sqrt(1 / (lam * t)))
            return x, t
        if isinstance(size, (bool, np.bool_)):
            raise TypeError("MultivariateNormalGammaSampler size must be a non-negative integer.")
        try:
            count = operator.index(size)
        except TypeError as exc:
            raise TypeError("MultivariateNormalGammaSampler size must be a non-negative integer.") from exc
        if count < 0:
            raise ValueError("MultivariateNormalGammaSampler size must be non-negative.")
        return [self.sample() for _ in range(count)]


class MultivariateNormalGammaDataEncoder(DataSequenceEncoder):
    """Encodes a sequence of (mu, tau) parameter pairs (identity passthrough)."""

    def __init__(self, dimension: int | None = None) -> None:
        self.dimension = dimension

    def __str__(self) -> str:
        return "MultivariateNormalGammaDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, MultivariateNormalGammaDataEncoder) and self.dimension == other.dimension

    def seq_encode(self, x: Any) -> Any:
        """Encode multivariate Normal-Gamma observations as a list payload."""
        values = list(x)
        dimension = self.dimension
        if dimension is None:
            if not values:
                raise ValueError("MultivariateNormalGammaDataEncoder needs a dimension to encode an empty batch.")
            first = values[0]
            if not isinstance(first, (tuple, list, np.ndarray)) or len(first) != 2:
                raise ValueError("MultivariateNormalGammaDataEncoder observations must be (mu, tau) pairs.")
            try:
                dimension = len(first[0])
            except TypeError as exc:
                raise ValueError("MultivariateNormalGammaDataEncoder observation means must be vectors.") from exc
        return [MultivariateNormalGammaDistribution._validated_observation(value, dimension) for value in values]
