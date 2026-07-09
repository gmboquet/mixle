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
from typing import Any

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

_CLIP = 1.0e-12  # keep Phi^{-1}(u) finite at the open-interval boundary


class GaussianCopulaDistribution(SequenceEncodableProbabilityDistribution):
    """Gaussian copula on ``(0,1)^d`` with dependence given by a correlation matrix."""

    def __init__(self, corr: np.ndarray, name: str | None = None, keys: str | None = None) -> None:
        r = np.asarray(corr, dtype=np.float64)
        if r.ndim != 2 or r.shape[0] != r.shape[1]:
            raise ValueError("corr must be a square correlation matrix")
        self.corr = r
        self.dim = r.shape[0]
        self.name = name
        self.keys = keys
        sign, logdet = np.linalg.slogdet(r)
        if sign <= 0:
            raise ValueError("corr must be positive definite")
        self._logdet = float(logdet)
        self._inv_minus_i = np.linalg.inv(r) - np.eye(self.dim)

    def __str__(self) -> str:
        return "GaussianCopulaDistribution(%s, name=%s, keys=%s)" % (
            repr(self.corr.tolist()),
            repr(self.name),
            repr(self.keys),
        )

    def density(self, x: np.ndarray) -> float:
        """Return the copula density at a single point ``u`` in ``(0,1)^d``."""
        return math.exp(self.log_density(x))

    def log_density(self, x: np.ndarray) -> float:
        """Return the log copula density at a single point ``u`` in ``(0,1)^d``."""
        z = norm.ppf(np.clip(np.asarray(x, dtype=np.float64), _CLIP, 1.0 - _CLIP))
        return -0.5 * self._logdet - 0.5 * float(z @ self._inv_minus_i @ z)

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Vectorized log copula density for sequence-encoded observations (``z = Phi^{-1}(u)`` rows)."""
        z = np.asarray(x, dtype=np.float64)
        quad = np.einsum("ni,ij,nj->n", z, self._inv_minus_i, z)
        return -0.5 * self._logdet - 0.5 * quad

    def sampler(self, seed: int | None = None) -> "GaussianCopulaSampler":
        """Return a sampler for drawing observations from this copula."""
        return GaussianCopulaSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "GaussianCopulaEstimator":
        """Return an estimator that fits the correlation matrix by the inversion estimator."""
        return GaussianCopulaEstimator(dim=self.dim, name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> "GaussianCopulaDataEncoder":
        """Return the data encoder (stores the normal-score transform ``z = Phi^{-1}(u)``)."""
        return GaussianCopulaDataEncoder()


class GaussianCopulaSampler(DistributionSampler):
    """Draw ``u`` by sampling ``z ~ N(0, R)`` and mapping through the standard-normal CDF."""

    def __init__(self, dist: GaussianCopulaDistribution, seed: int | None = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: int | None = None) -> np.ndarray:
        """Draw one copula sample or a batch of independent copula samples."""
        n = 1 if size is None else int(size)
        z = self.rng.multivariate_normal(np.zeros(self.dist.dim), self.dist.corr, size=n)
        u = norm.cdf(z)
        return u[0] if size is None else u


class GaussianCopulaAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate the weighted first and second moments of the normal scores ``z``."""

    def __init__(self, dim: int, name: str | None = None, keys: str | None = None) -> None:
        self.dim = dim
        self.sum_z = np.zeros(dim, dtype=np.float64)
        self.sum_zz = np.zeros((dim, dim), dtype=np.float64)
        self.count = 0.0
        self.name = name
        self.keys = keys

    def update(self, x: np.ndarray, weight: float, estimate: GaussianCopulaDistribution | None) -> None:
        """Accumulate weighted normal-score moments for one copula observation."""
        z = norm.ppf(np.clip(np.asarray(x, dtype=np.float64), _CLIP, 1.0 - _CLIP))
        self.sum_z += weight * z
        self.sum_zz += weight * np.outer(z, z)
        self.count += weight

    def initialize(self, x: np.ndarray, weight: float, rng: RandomState | None) -> None:
        """Initialize the sufficient statistics with one weighted observation."""
        self.update(x, weight, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: GaussianCopulaDistribution | None) -> None:
        """Accumulate weighted moments from encoded normal-score observations."""
        z = np.asarray(x, dtype=np.float64)  # already normal-scored by the encoder
        w = np.asarray(weights, dtype=np.float64)
        self.sum_z += z.T @ w
        self.sum_zz += (z * w[:, None]).T @ z
        self.count += float(w.sum())

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize the sufficient statistics from encoded observations."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[np.ndarray, np.ndarray, float]) -> "GaussianCopulaAccumulator":
        """Merge serialized normal-score moments into this accumulator."""
        self.sum_z += suff_stat[0]
        self.sum_zz += suff_stat[1]
        self.count += suff_stat[2]
        return self

    def value(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Return the weighted first moments, second moments, and total weight."""
        return self.sum_z.copy(), self.sum_zz.copy(), self.count

    def from_value(self, x: tuple[np.ndarray, np.ndarray, float]) -> "GaussianCopulaAccumulator":
        """Restore the accumulator from serialized normal-score moments."""
        self.sum_z = np.asarray(x[0], dtype=np.float64).copy()
        self.sum_zz = np.asarray(x[1], dtype=np.float64).copy()
        self.count = float(x[2])
        self.dim = self.sum_z.shape[0]
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Merge this accumulator into a keyed statistics dictionary."""
        if self.keys is not None:
            if self.keys in stats_dict:
                stats_dict[self.keys].combine(self.value())
            else:
                stats_dict[self.keys] = self

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Replace this accumulator from a keyed statistics dictionary."""
        if self.keys is not None and self.keys in stats_dict:
            self.from_value(stats_dict[self.keys].value())

    def acc_to_encoder(self) -> "GaussianCopulaDataEncoder":
        """Return an encoder that produces normal-score observations."""
        return GaussianCopulaDataEncoder()


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
        self.dim = dim
        self.min_eig = min_eig
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> GaussianCopulaAccumulatorFactory:
        """Return a factory for Gaussian copula sufficient-statistic accumulators."""
        return GaussianCopulaAccumulatorFactory(self.dim, name=self.name, keys=self.keys)

    def estimate(
        self, nobs: float | None, suff_stat: tuple[np.ndarray, np.ndarray, float]
    ) -> GaussianCopulaDistribution:
        """Estimate the copula correlation matrix from normal-score moments."""
        sum_z, sum_zz, count = suff_stat
        if count <= 0.0:
            return GaussianCopulaDistribution(np.eye(self.dim), name=self.name, keys=self.keys)
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

    def __str__(self) -> str:
        return "GaussianCopulaDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, GaussianCopulaDataEncoder)

    def seq_encode(self, x: Sequence[np.ndarray]) -> np.ndarray:
        """Encode copula observations by clipping and applying the normal quantile."""
        u = np.asarray(x, dtype=np.float64)
        return norm.ppf(np.clip(u, _CLIP, 1.0 - _CLIP))
