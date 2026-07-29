"""Gaussian copula: dependence structure on ``(0,1)^d`` decoupled from the marginals.

A copula is the joint distribution of ``U = (F_1(X_1), ..., F_d(X_d))`` -- each coordinate is its own
marginal CDF, so every marginal is Uniform(0,1) and all that remains is the *dependence*. The Gaussian
copula puts that dependence in a correlation matrix ``R``: pull each uniform back to a standard normal
``z_i = Phi^{-1}(u_i)`` and let ``z ~ N(0, R)``. Its density on ``(0,1)^d`` is

    c(u) = |R|^{-1/2} exp(-1/2 z^T (R^{-1} - I) z),   z = Phi^{-1}(u),

(the ``Phi`` Jacobians cancel the standard-normal part of the multivariate normal). Modelling the
dependence separately from the marginals is the whole point of copulas -- couple any marginals you
like (fit each separately) through one ``R``. ``R`` is fit by the standard inversion estimator: the
sample correlation of the transformed ``z``.


Reference: Nelsen, *An Introduction to Copulas* (2nd ed., Springer, 2006).
"""

import math
from collections.abc import Sequence

import numpy as np
from numpy.random import RandomState
from scipy.stats import norm

from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)
from mixle.stats.multivariate._copula_common import (
    reject_unsupported_pseudo_count,
    u_score_batch,
    u_score_event,
    validated_dimension,
    validated_finite_scalar,
    validated_sample_size,
    validated_weight,
    validated_weights,
)
from mixle.utils.vector import cholesky_logdet


class GaussianCopulaDistribution(SequenceEncodableProbabilityDistribution):
    """Gaussian copula on ``(0,1)^d`` with dependence given by a correlation matrix."""

    def __init__(self, corr: np.ndarray, name: str | None = None, keys: str | None = None) -> None:
        r = np.asarray(corr, dtype=np.float64)
        if r.ndim != 2 or r.shape[0] != r.shape[1] or r.shape[0] < 2:
            raise ValueError("corr must be a square correlation matrix of size at least two")
        if np.any(~np.isfinite(r)):
            raise ValueError("corr must contain only finite values")
        if not np.allclose(r, r.T):
            raise ValueError("corr must be symmetric")
        if not np.allclose(np.diag(r), 1.0):
            raise ValueError("corr must have a unit diagonal (it is a correlation, not a covariance, matrix)")
        r = r.copy()
        r.setflags(write=False)
        self._corr = r
        self.dim = r.shape[0]
        self.name = name
        self.keys = keys
        logdet = cholesky_logdet(r)
        if logdet is None:
            raise ValueError("corr must be positive definite")
        self._logdet = logdet
        self._inv_minus_i = np.linalg.inv(r) - np.eye(self.dim)

    @property
    def corr(self) -> np.ndarray:
        """Return an owned copy so cached scoring state cannot be desynchronized."""
        return self._corr.copy()

    def __str__(self) -> str:
        return "GaussianCopulaDistribution(%s, name=%s, keys=%s)" % (
            repr(self._corr.tolist()),
            repr(self.name),
            repr(self.keys),
        )

    def density(self, x: np.ndarray) -> float:
        """Return the copula density at a single point ``u`` in ``(0,1)^d``."""
        return math.exp(self.log_density(x))

    def log_density(self, x: np.ndarray) -> float:
        """Return the log copula density at a single point ``u`` in ``(0,1)^d``."""
        x = u_score_event(x, self.dim)
        z = norm.ppf(x)
        return -0.5 * self._logdet - 0.5 * float(z @ self._inv_minus_i @ z)

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Vectorized log copula density for sequence-encoded observations (``z = Phi^{-1}(u)`` rows)."""
        z = np.asarray(x, dtype=np.float64)
        if z.ndim != 2 or z.shape[1] != self.dim:
            raise ValueError("encoded Gaussian copula observations must have exact shape (N, %d)" % self.dim)
        if np.any(~np.isfinite(z)):
            raise ValueError("encoded Gaussian copula observations must be finite")
        quad = np.einsum("ni,ij,nj->n", z, self._inv_minus_i, z)
        return -0.5 * self._logdet - 0.5 * quad

    def sampler(self, seed: int | None = None) -> "GaussianCopulaSampler":
        """Return a sampler for drawing observations from this copula."""
        return GaussianCopulaSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "GaussianCopulaEstimator":
        """Return an estimator that fits the correlation matrix by the inversion estimator."""
        reject_unsupported_pseudo_count(pseudo_count, family="Gaussian copula")
        return GaussianCopulaEstimator(dim=self.dim, name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> "GaussianCopulaDataEncoder":
        """Return the data encoder (stores the normal-score transform ``z = Phi^{-1}(u)``)."""
        return GaussianCopulaDataEncoder(self.dim)


class GaussianCopulaSampler(DistributionSampler):
    """Draw ``u`` by sampling ``z ~ N(0, R)`` and mapping through the standard-normal CDF."""

    def __init__(self, dist: GaussianCopulaDistribution, seed: int | None = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: int | None = None, *, batched: bool = True) -> np.ndarray:
        """Draw one copula sample or a batch of independent copula samples."""
        n = 1 if size is None else validated_sample_size(size)
        z = self.rng.multivariate_normal(np.zeros(self.dist.dim), self.dist._corr, size=n)
        u = norm.cdf(z)
        return u[0] if size is None else u


class GaussianCopulaAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate the weighted first and second moments of the normal scores ``z``."""

    def __init__(self, dim: int, name: str | None = None, keys: str | None = None) -> None:
        self.dim = validated_dimension(dim, label="Gaussian copula dimension")
        self.sum_z = np.zeros(dim, dtype=np.float64)
        self.sum_zz = np.zeros((dim, dim), dtype=np.float64)
        self.count = 0.0
        self.name = name
        self.keys = keys

    def update(self, x: np.ndarray, weight: float, estimate: GaussianCopulaDistribution | None) -> None:
        """Accumulate weighted normal-score moments for one copula observation."""
        x = u_score_event(x, self.dim)
        checked_weight = validated_weight(weight)
        z = norm.ppf(x)
        self.sum_z += checked_weight * z
        self.sum_zz += checked_weight * np.outer(z, z)
        self.count += checked_weight

    def initialize(self, x: np.ndarray, weight: float, rng: RandomState | None) -> None:
        """Initialize the sufficient statistics with one weighted observation."""
        self.update(x, weight, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: GaussianCopulaDistribution | None) -> None:
        """Accumulate weighted moments from encoded normal-score observations."""
        z = np.asarray(x, dtype=np.float64)  # already normal-scored by the encoder
        if z.ndim != 2 or z.shape[1] != self.dim:
            raise ValueError("encoded Gaussian copula observations must have exact shape (N, %d)" % self.dim)
        if np.any(~np.isfinite(z)):
            raise ValueError("encoded Gaussian copula observations must be finite")
        w = validated_weights(weights, len(z))
        self.sum_z += z.T @ w
        self.sum_zz += (z * w[:, None]).T @ z
        self.count += float(w.sum())

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize the sufficient statistics from encoded observations."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[np.ndarray, np.ndarray, float]) -> "GaussianCopulaAccumulator":
        """Merge serialized normal-score moments into this accumulator."""
        sum_z, sum_zz, count = _validated_gaussian_statistic(suff_stat, self.dim, require_positive=False)
        self.sum_z += sum_z
        self.sum_zz += sum_zz
        self.count += count
        return self

    def value(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Return the weighted first moments, second moments, and total weight."""
        return self.sum_z.copy(), self.sum_zz.copy(), self.count

    def from_value(self, x: tuple[np.ndarray, np.ndarray, float]) -> "GaussianCopulaAccumulator":
        """Restore the accumulator from serialized normal-score moments."""
        self.sum_z, self.sum_zz, self.count = _validated_gaussian_statistic(x, self.dim, require_positive=False)
        return self

    def acc_to_encoder(self) -> "GaussianCopulaDataEncoder":
        """Return an encoder that produces normal-score observations."""
        return GaussianCopulaDataEncoder(self.dim)


class GaussianCopulaAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for GaussianCopulaAccumulator."""

    def __init__(self, dim: int, name: str | None = None, keys: str | None = None) -> None:
        self.dim = dim
        self.name = name
        self.keys = keys

    def make(self) -> GaussianCopulaAccumulator:
        """Create an empty Gaussian copula accumulator."""
        return GaussianCopulaAccumulator(self.dim, name=self.name, keys=self.keys)


class GaussianCopulaEstimator(ParameterEstimator):
    """Inversion estimator: the correlation of the normal scores ``z = Phi^{-1}(u)``."""

    def __init__(self, dim: int, min_eig: float = 1.0e-8, name: str | None = None, keys: str | None = None) -> None:
        self.dim = validated_dimension(dim, label="Gaussian copula dimension")
        self.min_eig = validated_finite_scalar(min_eig, label="Gaussian copula min_eig")
        if not 0.0 < self.min_eig < 1.0:
            raise ValueError("Gaussian copula min_eig must lie strictly between zero and one")
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> GaussianCopulaAccumulatorFactory:
        """Return a factory for Gaussian copula sufficient-statistic accumulators."""
        return GaussianCopulaAccumulatorFactory(self.dim, name=self.name, keys=self.keys)

    def estimate(
        self, nobs: float | None, suff_stat: tuple[np.ndarray, np.ndarray, float]
    ) -> GaussianCopulaDistribution:
        """Estimate the copula correlation matrix from normal-score moments."""
        sum_z, sum_zz, count = _validated_gaussian_statistic(suff_stat, self.dim, require_positive=True)
        mean = sum_z / count
        cov = sum_zz / count - np.outer(mean, mean)
        d = np.sqrt(np.clip(np.diag(cov), 1.0e-12, None))
        corr = cov / np.outer(d, d)  # normalize to unit diagonal
        corr = 0.5 * (corr + corr.T)
        np.fill_diagonal(corr, 1.0)
        # project to a valid (positive-definite) correlation matrix if needed
        w, v = np.linalg.eigh(corr)
        if w.min() < self.min_eig:
            corr = v @ np.diag(np.clip(w, self.min_eig, None)) @ v.T
            dd = np.sqrt(np.diag(corr))
            corr = corr / np.outer(dd, dd)
            np.fill_diagonal(corr, 1.0)
        return GaussianCopulaDistribution(corr, name=self.name, keys=self.keys)


class GaussianCopulaDataEncoder(DataSequenceEncoder):
    """Encode each ``u`` row as its normal score ``z = Phi^{-1}(u)``."""

    def __init__(self, dim: int) -> None:
        self.dim = validated_dimension(dim, label="Gaussian copula dimension")

    def __str__(self) -> str:
        return "GaussianCopulaDataEncoder(dim=%d)" % self.dim

    def __eq__(self, other: object) -> bool:
        return isinstance(other, GaussianCopulaDataEncoder) and self.dim == other.dim

    def seq_encode(self, x: Sequence[np.ndarray]) -> np.ndarray:
        """Encode copula observations by clipping and applying the normal quantile."""
        return norm.ppf(u_score_batch(x, self.dim))


def _validated_gaussian_statistic(
    value: object,
    dim: int,
    *,
    require_positive: bool,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Validate serialized normal-score moments before accumulation or fitting."""
    if not isinstance(value, tuple) or len(value) != 3:
        raise ValueError("Gaussian copula statistic must be a (sum_z, sum_zz, count) tuple")
    sum_z = np.asarray(value[0], dtype=np.float64)
    sum_zz = np.asarray(value[1], dtype=np.float64)
    count = validated_weight(value[2])
    if sum_z.shape != (dim,) or sum_zz.shape != (dim, dim):
        raise ValueError("Gaussian copula statistic must have shapes (%d,) and (%d, %d)" % (dim, dim, dim))
    if np.any(~np.isfinite(sum_z)) or np.any(~np.isfinite(sum_zz)):
        raise ValueError("Gaussian copula statistic moments must be finite")
    if not np.allclose(sum_zz, sum_zz.T):
        raise ValueError("Gaussian copula second-moment statistic must be symmetric")
    if require_positive and not count > 0.0:
        raise ValueError("Gaussian copula fit requires positive total observation weight")
    return sum_z.copy(), sum_zz.copy(), count
