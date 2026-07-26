"""Normal-Wishart distribution over (mu, Lambda) for a d-dimensional Gaussian
with unknown mean and precision matrix.

    q(mu, Lambda) = N(mu | m, (kappa*Lambda)^{-1}) * Wishart(Lambda | W, nu)

with scale matrix W (d x d positive definite) and degrees of freedom nu > d - 1.
This is the conjugate prior for the multivariate
:class:`~mixle.stats.multivariate.multivariate_gaussian.MultivariateGaussianDistribution` (see its ``prior=``
argument) and the d-dimensional generalization of NormalGamma (d=1: nu = 2a,
W = 1/(2b)). It is a parameter prior: it is scored on ``(mu, Lambda)`` parameter
pairs, not fit from data by EM.
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
from mixle.utils.vector import cholesky_logdet


def _multigammaln(a: float, d: int) -> float:
    """Log of the d-dimensional multivariate gamma function at a."""
    return d * (d - 1) / 4.0 * np.log(np.pi) + sum(gammaln(a + (1.0 - i) / 2.0) for i in range(1, d + 1))


def _multidigamma(a: float, d: int) -> float:
    """Derivative of _multigammaln with respect to a."""
    return sum(digamma(a + (1.0 - i) / 2.0) for i in range(1, d + 1))


class NormalWishartDistribution(SequenceEncodableProbabilityDistribution):
    """Normal-Wishart distribution over (mu, Lambda); conjugate prior for the
    multivariate Gaussian with unknown mean and precision matrix."""

    def __init__(
        self,
        mu,
        kappa: float,
        w_mat,
        nu: float,
        name: str | None = None,
        prior: Optional["SequenceEncodableProbabilityDistribution"] = None,
    ) -> None:
        """Create a normal-Wishart prior over Gaussian mean and precision matrix.

        Args:
            mu: Length-d prior mean m.
            kappa (float): Mean-precision scale kappa > 0.
            w_mat: (d, d) positive-definite Wishart scale matrix W.
            nu (float): Degrees of freedom nu > d - 1.
            name (Optional[str]): Name of object.
            prior (Optional): Hyper-prior (stored for interface compatibility).

        """
        self.name = name
        self.prior = prior
        self.set_parameters((mu, kappa, w_mat, nu))

    def __str__(self) -> str:
        mu = ",".join(map(str, self.mu.tolist()))
        w = ",".join(map(str, self.w_mat.flatten().tolist()))
        return "NormalWishartDistribution([%s], %f, [%s], %f, name=%s, prior=%s)" % (
            mu,
            self.kappa,
            w,
            self.nu,
            self.name,
            str(self.prior),
        )

    def get_parameters(self):
        """Returns the parameter tuple (mu, kappa, w_mat, nu)."""
        return self.mu.copy(), self.kappa, self.w_mat.copy(), self.nu

    @staticmethod
    def _validated_parameters(params):
        if not isinstance(params, (tuple, list)) or len(params) != 4:
            raise ValueError("NormalWishartDistribution parameters must be a four-item tuple.")
        mu_raw, kappa_raw, w_raw, nu_raw = params
        try:
            mu = np.asarray(mu_raw, dtype=np.float64)
            w_mat = np.asarray(w_raw, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("NormalWishartDistribution mean and scale must be numeric arrays.") from exc
        if mu.ndim != 1 or mu.size == 0:
            raise ValueError("NormalWishartDistribution requires mu to be a non-empty vector.")
        if np.any(~np.isfinite(mu)):
            raise ValueError("NormalWishartDistribution requires mu to be finite.")
        dimension = len(mu)
        if w_mat.shape != (dimension, dimension):
            raise ValueError(
                "NormalWishartDistribution requires w_mat shape (%d, %d)." % (dimension, dimension)
            )
        if np.any(~np.isfinite(w_mat)):
            raise ValueError("NormalWishartDistribution requires a finite scale matrix w_mat.")
        if not np.allclose(w_mat, w_mat.T, rtol=1.0e-10, atol=1.0e-12):
            raise ValueError("NormalWishartDistribution requires a symmetric scale matrix w_mat.")
        w_mat = 0.5 * (w_mat + w_mat.T)
        if np.ndim(kappa_raw) != 0 or np.ndim(nu_raw) != 0:
            raise ValueError("NormalWishartDistribution kappa and nu must be scalars.")
        try:
            kappa = float(kappa_raw)
            nu = float(nu_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("NormalWishartDistribution kappa and nu must be numeric scalars.") from exc
        if kappa <= 0.0 or not np.isfinite(kappa):
            raise ValueError("NormalWishartDistribution requires finite kappa > 0.")
        if not nu > dimension - 1 or not np.isfinite(nu):
            raise ValueError(
                "NormalWishartDistribution requires a finite nu > dim - 1 (got nu=%s, dim=%d)."
                % (nu, dimension)
            )
        log_det_w = cholesky_logdet(w_mat)
        if log_det_w is None:
            raise ValueError("NormalWishartDistribution requires a positive-definite scale matrix w_mat.")
        w_inv = np.linalg.inv(w_mat)
        log_z = (
            (nu * dimension / 2.0) * np.log(2.0)
            + (nu / 2.0) * log_det_w
            + _multigammaln(nu / 2.0, dimension)
        )
        if not np.isfinite(log_z):
            raise ValueError("NormalWishartDistribution parameters produced a non-finite normalizer.")
        return (
            mu.copy(),
            kappa,
            w_mat.copy(),
            nu,
            dimension,
            float(log_det_w),
            w_inv,
            float(log_z),
        )

    def _validated_state(self):
        return self._validated_parameters((self.mu, self.kappa, self.w_mat, self.nu))

    @staticmethod
    def _validated_observation(value: Any, dimension: int):
        if not isinstance(value, (tuple, list, np.ndarray)) or len(value) != 2:
            raise ValueError("NormalWishartDistribution observations must be (mu, Lambda) pairs.")
        try:
            mu = np.asarray(value[0], dtype=np.float64)
            precision = np.asarray(value[1], dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("NormalWishartDistribution observations must be numeric.") from exc
        if mu.shape != (dimension,) or np.any(~np.isfinite(mu)):
            raise ValueError(
                "NormalWishartDistribution observation mu must be a finite vector of length %d."
                % dimension
            )
        if precision.shape != (dimension, dimension) or np.any(~np.isfinite(precision)):
            raise ValueError(
                "NormalWishartDistribution observation precision must be a finite (%d, %d) matrix."
                % (dimension, dimension)
            )
        if not np.allclose(precision, precision.T, rtol=1.0e-10, atol=1.0e-12):
            raise ValueError("NormalWishartDistribution observation precision must be symmetric.")
        return mu, 0.5 * (precision + precision.T)

    def set_parameters(self, params) -> None:
        """Set the parameters and refresh the cached Wishart log-normalizer.

        Args:
            params: Tuple (mu, kappa, w_mat, nu) with w_mat positive
                definite and nu > d - 1.

        """
        (
            mu,
            kappa,
            w_mat,
            nu,
            dimension,
            log_det_w,
            w_inv,
            log_z,
        ) = self._validated_parameters(params)
        self.mu = mu
        self.kappa = kappa
        self.w_mat = w_mat
        self.nu = nu
        self.dim = dimension
        self.log_det_w = log_det_w
        self.w_inv = w_inv
        self.log_z = log_z

    def expected_log_det(self) -> float:
        """E[ln |Lambda|] under the Wishart factor."""
        _, _, _, nu, dimension, log_det_w, _, _ = self._validated_state()
        return _multidigamma(nu / 2.0, dimension) + dimension * np.log(2.0) + log_det_w

    def expected_precision(self) -> np.ndarray:
        """E[Lambda] = nu * W."""
        _, _, w_mat, nu, _, _, _, _ = self._validated_state()
        return nu * w_mat

    def density(self, x) -> float:
        """Density at x = (mu, Lambda); see log_density()."""
        return np.exp(self.log_density(x))

    def log_density(self, x) -> float:
        """Log density at x = (mu, Lambda) with Lambda a precision matrix.

        Returns -inf when Lambda is not positive definite.
        """
        model_mu, kappa, _, nu, d, _, w_inv, log_z = self._validated_state()
        mu, lam = self._validated_observation(x, d)

        log_det_lam = cholesky_logdet(lam)
        if log_det_lam is None:
            return -np.inf

        diff = mu - model_mu
        c_norm = (
            (d / 2.0) * np.log(kappa / (2.0 * np.pi))
            + 0.5 * log_det_lam
            - 0.5 * kappa * float(np.dot(diff, np.dot(lam, diff)))
        )
        c_wish = ((nu - d - 1.0) / 2.0) * log_det_lam - 0.5 * float(np.trace(np.dot(w_inv, lam))) - log_z

        return c_norm + c_wish

    def cross_entropy(self, dist: "NormalWishartDistribution") -> float:
        """H(self, dist) = -E_self[log dist] for a NormalWishart argument."""
        if not isinstance(dist, NormalWishartDistribution):
            raise NotImplementedError(
                "NormalWishartDistribution.cross_entropy is only implemented for NormalWishart arguments (got %s)."
                % type(dist).__name__
            )

        self_mu, self_kappa, self_w, self_nu, d, self_log_det, _, _ = self._validated_state()
        dist_mu, dist_kappa, _, dist_nu, dist_d, _, dist_w_inv, dist_log_z = dist._validated_state()
        if d != dist_d:
            raise ValueError("NormalWishartDistribution cross-entropy requires equal dimensions.")
        e_log_det = _multidigamma(self_nu / 2.0, d) + d * np.log(2.0) + self_log_det
        e_lam = self_nu * self_w

        # E[(mu - m_p)' Lambda (mu - m_p)] under self
        diff = self_mu - dist_mu
        e_quad = d / self_kappa + self_nu * float(np.dot(diff, np.dot(self_w, diff)))

        c_norm = (d / 2.0) * np.log(dist_kappa / (2.0 * np.pi)) + 0.5 * e_log_det - 0.5 * dist_kappa * e_quad
        c_wish = (
            ((dist_nu - d - 1.0) / 2.0) * e_log_det
            - 0.5 * float(np.trace(np.dot(dist_w_inv, e_lam)))
            - dist_log_z
        )

        return -(c_norm + c_wish)

    def entropy(self) -> float:
        """Returns the entropy of the Normal-Wishart distribution (in nats)."""
        return self.cross_entropy(self)

    def seq_log_density(self, x) -> np.ndarray:
        """Vectorized log-density over a sequence of (mu, Lambda) pairs."""
        self._validated_state()
        return np.asarray([self.log_density(xx) for xx in x], dtype=float)

    def sampler(self, seed: int | None = None) -> "NormalWishartSampler":
        """Create a NormalWishartSampler for this distribution."""
        return NormalWishartSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "ParameterEstimator":
        """NormalWishart is a parameter prior and is not fit from data by EM."""
        raise NotImplementedError("NormalWishartDistribution is a parameter prior; it has no data estimator.")

    def dist_to_encoder(self) -> "NormalWishartDataEncoder":
        """Return the encoder for ``(mu, Lambda)`` normal-Wishart observations."""
        return NormalWishartDataEncoder(self.dim)


class NormalWishartSampler(DistributionSampler):
    """Draws (mu, Lambda) samples from a NormalWishartDistribution."""

    def __init__(self, dist: NormalWishartDistribution, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = np.random.RandomState(seed)

    def sample(self, size=None, *, batched: bool = True) -> Any:
        """Draw size samples (a single (mu, Lambda) pair when size is None).

        Lambda is drawn from the Wishart factor, then mu from
        N(m, (kappa*Lambda)^-1).
        """
        model_mu, kappa, w_mat, nu, _, _, _, _ = self.dist._validated_state()
        if size is None:
            lam = scipy_wishart_sample(self.rng, nu, w_mat)
            covar = np.linalg.inv(lam * kappa)
            mu = self.rng.multivariate_normal(model_mu, covar)
            return mu, lam
        if isinstance(size, (bool, np.bool_)):
            raise TypeError("NormalWishartSampler size must be a non-negative integer.")
        try:
            count = operator.index(size)
        except TypeError as exc:
            raise TypeError("NormalWishartSampler size must be a non-negative integer.") from exc
        if count < 0:
            raise ValueError("NormalWishartSampler size must be non-negative.")
        return [self.sample() for _ in range(count)]


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


class NormalWishartDataEncoder(DataSequenceEncoder):
    """Encodes a sequence of (mu, Lambda) parameter pairs (identity passthrough)."""

    def __init__(self, dimension: int | None = None) -> None:
        self.dimension = dimension

    def __str__(self) -> str:
        return "NormalWishartDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, NormalWishartDataEncoder) and self.dimension == other.dimension

    def seq_encode(self, x: Any) -> Any:
        """Encode Normal-Wishart observations as a list payload."""
        values = list(x)
        if self.dimension is None:
            if not values:
                raise ValueError("NormalWishartDataEncoder needs a dimension to encode an empty batch.")
            first = values[0]
            if not isinstance(first, (tuple, list, np.ndarray)) or len(first) != 2:
                raise ValueError("NormalWishartDataEncoder observations must be (mu, Lambda) pairs.")
            try:
                dimension = len(first[0])
            except TypeError as exc:
                raise ValueError("NormalWishartDataEncoder observation means must be vectors.") from exc
        else:
            dimension = self.dimension
        return [
            NormalWishartDistribution._validated_observation(value, dimension)
            for value in values
        ]
