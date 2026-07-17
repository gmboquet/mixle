"""Probabilistic PCA latent-factor distributions and estimators.

Observations are length-``d`` real vectors represented as ``np.ndarray`` or a
compatible sequence of floats.

Probabilistic PCA is the latent linear-Gaussian model

    z ~ N(0, I_q),    x | z ~ N(W z + mu, sigma2 * I_d),

so marginally ``x ~ N(mu, C)`` with the structured covariance ``C = W W^T + sigma2 * I_d`` (a rank-q
factor structure plus isotropic noise). It is the probabilistic foundation of PCA / factor analysis and
gives a generative model, a likelihood, and a posterior over the latent factors
``E[z | x] = M^{-1} W^T (x - mu)`` (the low-dimensional embedding, exposed by ``transform``), with
``M = W^T W + sigma2 * I_q``.

Scoring uses the Woodbury identity, so the d-by-d inverse and log-determinant are obtained from a small
q-by-q solve (``C^{-1} = (I_d - W M^{-1} W^T) / sigma2`` and ``log|C| = (d-q) log sigma2 + log|M|``); the
reduction is engine-neutral, so the model scores on NumPy and Torch. Estimation is the **closed-form**
maximum-likelihood solution of Tipping & Bishop (1999): ``sigma2`` is the mean of the discarded
eigenvalues of the sample covariance and ``W`` is built from its top-q eigenpairs -- no EM iteration.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import RandomState

from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)

_MIN_SIGMA2 = 1.0e-12
_LOG_2PI = float(np.log(2.0 * np.pi))


class ProbabilisticPCADistribution(SequenceEncodableProbabilityDistribution):
    """Probabilistic PCA: x ~ N(mu, W W^T + sigma2 I) with q latent factors."""

    @classmethod
    def compute_capabilities(cls):
        """Return compute-backend metadata for PPCA scoring."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch"), kernel_status="generic")

    def __init__(
        self,
        w: Sequence[Sequence[float]] | np.ndarray,
        mu: Sequence[float] | np.ndarray,
        sigma2: float,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        """Create a probabilistic PCA distribution.

        Args:
            w: ``d`` by ``q`` factor-loading matrix.
            mu: Length-``d`` mean vector.
            sigma2: Positive isotropic noise variance.
            name: Optional diagnostic name.
            keys: Optional key for merging sufficient statistics.

        Attributes:
            w: Factor-loading matrix.
            mu: Mean vector.
            sigma2: Isotropic noise variance.
            dim: Observation dimension.
            latent_dim: Number of latent factors.
            inv_covar: Cached covariance inverse via Woodbury.
            log_det: Cached covariance log-determinant.

        """
        w = np.asarray(w, dtype=float)
        mu = np.asarray(mu, dtype=float).copy()
        if w.ndim != 2 or w.shape[0] != len(mu):
            raise ValueError("ProbabilisticPCADistribution requires W of shape (d, q) matching mu of length d.")
        if sigma2 <= 0.0 or not np.isfinite(sigma2):
            raise ValueError("ProbabilisticPCADistribution requires sigma2 > 0.")
        self.w = w
        self.mu = mu
        self.sigma2 = float(sigma2)
        self.dim = w.shape[0]
        self.latent_dim = w.shape[1]

        q = self.latent_dim
        m = w.T @ w + self.sigma2 * np.eye(q)  # (q, q)
        self._m_inv = np.linalg.inv(m)
        # Woodbury: C^{-1} = (I_d - W M^{-1} W^T) / sigma2
        self.inv_covar = (np.eye(self.dim) - w @ self._m_inv @ w.T) / self.sigma2
        sign, log_det_m = np.linalg.slogdet(m)
        self.log_det = float((self.dim - q) * np.log(self.sigma2) + log_det_m)
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        """Return a readable distribution summary."""
        return "ProbabilisticPCADistribution(%s, %s, %s, name=%s, keys=%s)" % (
            repr([[float(v) for v in row] for row in self.w]),
            repr([float(v) for v in self.mu]),
            repr(self.sigma2),
            repr(self.name),
            repr(self.keys),
        )

    def transform(self, x: Sequence[float] | np.ndarray) -> np.ndarray:
        """Return the posterior mean of the latent factors E[z | x] = M^{-1} W^T (x - mu)."""
        diff = np.asarray(x, dtype=float) - self.mu
        return self._m_inv @ (self.w.T @ diff.T)

    def density(self, x: Sequence[float] | np.ndarray) -> float:
        """Return the probability density at a single observation."""
        return float(np.exp(self.log_density(x)))

    def log_density(self, x: Sequence[float] | np.ndarray) -> float:
        """Return the log-density at a single observation."""
        diff = np.asarray(x, dtype=float) - self.mu
        mahal = float(diff @ self.inv_covar @ diff)
        return -0.5 * (self.dim * _LOG_2PI + self.log_det + mahal)

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Return vectorized log-density values for sequence-encoded observations."""
        diff = x - self.mu
        mahal = np.einsum("ij,jk,ik->i", diff, self.inv_covar, diff)
        return -0.5 * (self.dim * _LOG_2PI + self.log_det + mahal)

    def backend_seq_log_density(self, x: np.ndarray, engine: Any) -> Any:
        """Engine-neutral vectorized log-density for encoded data."""
        diff = engine.asarray(x) - engine.asarray(self.mu)
        mahal = engine.sum(engine.matmul(diff, engine.asarray(self.inv_covar)) * diff, axis=-1)
        const = engine.asarray(self.dim * _LOG_2PI + self.log_det)
        return engine.asarray(-0.5) * (const + mahal)

    def sampler(self, seed: int | None = None) -> "ProbabilisticPCASampler":
        """Return a sampler for drawing observations from this distribution."""
        return ProbabilisticPCASampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "ProbabilisticPCAEstimator":
        """Return a closed-form ML estimator with the latent dimension fixed at this model's q."""
        return ProbabilisticPCAEstimator(latent_dim=self.latent_dim, dim=self.dim, name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> "ProbabilisticPCADataEncoder":
        """Return the data encoder used by this distribution for vectorized methods."""
        return ProbabilisticPCADataEncoder()


class ProbabilisticPCASampler(DistributionSampler):
    """Draw iid observations x = mu + W z + sigma * eps from a PPCA model."""

    def __init__(self, dist: ProbabilisticPCADistribution, seed: int | None = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: int | None = None, *, batched: bool = True) -> np.ndarray:
        """Draw ``size`` iid vectors (shape (d,) when size is None, else (size, d))."""
        sz = 1 if size is None else size
        d, q = self.dist.dim, self.dist.latent_dim
        z = self.rng.standard_normal(size=(sz, q))
        noise = np.sqrt(self.dist.sigma2) * self.rng.standard_normal(size=(sz, d))
        rv = self.dist.mu[None, :] + z @ self.dist.w.T + noise
        return rv[0] if size is None else rv


class ProbabilisticPCAAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate the weighted count, mean, and second-moment matrix (the PPCA sufficient statistics)."""

    def __init__(self, dim: int | None = None, keys: str | None = None) -> None:
        self.dim = dim
        self.count = 0.0
        self.sum = np.zeros(dim) if dim is not None else None
        self.sum2 = np.zeros((dim, dim)) if dim is not None else None
        self.keys = keys

    def _ensure_dim(self, d: int) -> None:
        if self.dim is None:
            self.dim = d
        if self.sum is None:
            self.sum = np.zeros(self.dim)
            self.sum2 = np.zeros((self.dim, self.dim))

    def update(self, x: np.ndarray, weight: float, estimate: ProbabilisticPCADistribution | None) -> None:
        """Accumulate weighted count, sum, and second moment for one vector."""
        xx = np.asarray(x, dtype=float)
        self._ensure_dim(len(xx))
        self.count += weight
        self.sum += weight * xx
        self.sum2 += weight * np.outer(xx, xx)

    def initialize(self, x: np.ndarray, weight: float, rng: RandomState | None) -> None:
        """Initialize the sufficient statistics with one weighted vector."""
        self.update(x, weight, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: ProbabilisticPCADistribution | None) -> None:
        """Accumulate weighted count, sum, and second moment for encoded vectors."""
        self._ensure_dim(x.shape[1])
        self.count += float(np.sum(weights, dtype=np.float64))
        self.sum += x.T @ weights
        self.sum2 += (x * weights[:, None]).T @ x

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize the sufficient statistics from encoded vectors."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[float, np.ndarray | None, np.ndarray | None]) -> "ProbabilisticPCAAccumulator":
        """Merge serialized PPCA sufficient statistics into this accumulator."""
        count, s, s2 = suff_stat
        if s is not None:
            self._ensure_dim(len(s))
            self.sum += s
            self.sum2 += s2
        self.count += count
        return self

    def value(self) -> tuple[float, np.ndarray | None, np.ndarray | None]:
        """Return the total weight, weighted sum, and weighted second moment."""
        return self.count, self.sum, self.sum2

    def from_value(self, x: tuple[float, np.ndarray | None, np.ndarray | None]) -> "ProbabilisticPCAAccumulator":
        """Restore the accumulator from serialized PPCA sufficient statistics."""
        self.count, self.sum, self.sum2 = x
        self.dim = None if x[1] is None else len(x[1])
        return self

    def acc_to_encoder(self) -> "ProbabilisticPCADataEncoder":
        """Return an encoder compatible with PPCA vector observations."""
        return ProbabilisticPCADataEncoder()


class ProbabilisticPCAAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for ProbabilisticPCAAccumulator."""

    def __init__(self, dim: int | None = None, keys: str | None = None) -> None:
        self.dim = dim
        self.keys = keys

    def make(self) -> ProbabilisticPCAAccumulator:
        """Create an empty PPCA accumulator."""
        return ProbabilisticPCAAccumulator(dim=self.dim, keys=self.keys)


class ProbabilisticPCAEstimator(ParameterEstimator):
    """Closed-form maximum-likelihood estimator for PPCA (Tipping & Bishop eigen-solution)."""

    def __init__(
        self,
        latent_dim: int,
        dim: int | None = None,
        min_sigma2: float = _MIN_SIGMA2,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        if latent_dim is None or latent_dim < 1:
            raise ValueError("ProbabilisticPCAEstimator requires latent_dim >= 1.")
        self.latent_dim = int(latent_dim)
        self.dim = dim
        self.min_sigma2 = min_sigma2
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> ProbabilisticPCAAccumulatorFactory:
        """Return a factory for PPCA sufficient-statistic accumulators."""
        return ProbabilisticPCAAccumulatorFactory(dim=self.dim, keys=self.keys)

    def estimate(
        self, nobs: float | None, suff_stat: tuple[float, np.ndarray | None, np.ndarray | None]
    ) -> ProbabilisticPCADistribution:
        """Estimate PPCA parameters from weighted first and second moments."""
        count, s, s2 = suff_stat
        if s is None or count <= 0.0:
            d = self.dim if self.dim is not None else self.latent_dim
            return ProbabilisticPCADistribution(
                np.zeros((d, self.latent_dim)), np.zeros(d), 1.0, name=self.name, keys=self.keys
            )

        d = len(s)
        q = min(self.latent_dim, d)
        mu = s / count
        cov = s2 / count - np.outer(mu, mu)
        cov = 0.5 * (cov + cov.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = np.clip(eigvals[order], 0.0, None)
        eigvecs = eigvecs[:, order]

        # sigma2 = mean of the discarded eigenvalues (the isotropic residual variance).
        sigma2 = float(np.mean(eigvals[q:])) if q < d else 0.0
        sigma2 = max(sigma2, self.min_sigma2)
        # W = U_q (Lambda_q - sigma2 I)^{1/2}; padded with zero columns if q < latent_dim.
        scale = np.sqrt(np.clip(eigvals[:q] - sigma2, 0.0, None))
        w = eigvecs[:, :q] * scale[None, :]
        if q < self.latent_dim:
            w = np.hstack([w, np.zeros((d, self.latent_dim - q))])
        return ProbabilisticPCADistribution(w, mu, sigma2, name=self.name, keys=self.keys)


class ProbabilisticPCADataEncoder(DataSequenceEncoder):
    """Encode a sequence of length-d real vectors into an (n, d) float array."""

    def __str__(self) -> str:
        return "ProbabilisticPCADataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ProbabilisticPCADataEncoder)

    def seq_encode(self, x: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        """Validate and encode observations as a two-dimensional float array."""
        rv = np.asarray(x, dtype=np.float64)
        if rv.ndim != 2:
            rv = rv.reshape((len(x), -1))
        if rv.size and not np.all(np.isfinite(rv)):
            raise ValueError("ProbabilisticPCADistribution requires finite real-vector observations.")
        return rv
