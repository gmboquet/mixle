"""Ledoit-Wolf covariance-shrinkage estimator for the multivariate Gaussian.

The sample covariance is a poor estimate when the number of observations is not large relative to the
dimension: its extreme eigenvalues are biased (the largest too large, the smallest too small), which is
exactly what wrecks anything that inverts it -- Markowitz portfolios, Mahalanobis distances, GP precisions.
Ledoit and Wolf (2004, *A well-conditioned estimator for large-dimensional covariance matrices*) give the
optimal convex combination of the sample covariance ``S`` and a well-conditioned target ``F`` (a scaled
identity ``F = (tr S / d) I``):

    Sigma_hat = (1 - delta) S + delta F,    delta = clip( b^2 / d^2 , 0, 1 ),

where ``d^2 = ||S - F||_F^2`` and ``b^2 = (1/n^2) sum_t ||y_t y_t^T - S||_F^2`` (``y_t`` the centered
observations) -- a data-driven shrinkage intensity, no cross-validation needed.

``LedoitWolfEstimator`` is a first-class mixle estimator: it follows the accumulator/factory/encoder
contract, so it composes with ``estimate``, mixtures, HMMs, and anything else that takes a
``ParameterEstimator``, and it returns an ordinary :class:`MultivariateGaussianDistribution`. The shrinkage
intensity is computed *exactly* from streaming sufficient statistics -- the centered 4th moment
``sum_t (y_t . y_t)^2`` decomposes into ``sum x``, ``sum x x^T``, ``sum x ||x||^2`` and ``sum ||x||^4`` --
so it works under ``seq_update`` and ``combine`` (distributed) without holding the data.


Reference: Ledoit & Wolf, 'A well-conditioned estimator for large-dimensional covariance matrices', J. Multivariate Anal. (2004).
"""

from __future__ import annotations

import numpy as np
from numpy.random import RandomState

from mixle.stats.compute.pdist import (
    ParameterEstimator,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)
from mixle.stats.multivariate.multivariate_gaussian import (
    MultivariateGaussianDataEncoder,
    MultivariateGaussianDistribution,
)

__all__ = ["LedoitWolfEstimator"]


def _shrink(s1: np.ndarray, s2: np.ndarray, s3: np.ndarray, s4: float, n: float):
    """Sample mean, sample covariance, and Ledoit-Wolf-shrunk covariance from sufficient statistics."""
    mean = s1 / n
    cov = s2 / n - np.outer(mean, mean)  # MLE sample covariance
    d = len(mean)
    target_scale = np.trace(cov) / d
    target = target_scale * np.eye(d)
    d2 = float(np.sum((cov - target) ** 2))  # ||S - F||_F^2
    # sum_t (y_t . y_t)^2 with y_t = x_t - mean, expanded into the accumulated moments
    c = float(mean @ mean)
    sum_yy2 = (
        s4 + 4.0 * float(mean @ s2 @ mean) - 4.0 * float(s3 @ mean) + 2.0 * c * float(np.trace(s2)) - 3.0 * n * c * c
    )
    b2 = sum_yy2 / n**2 - float(np.sum(cov**2)) / n  # (1/n^2) sum ||y y^T - S||_F^2
    delta = float(np.clip(b2 / d2, 0.0, 1.0)) if d2 > 0 else 0.0
    shrunk = (1.0 - delta) * cov + delta * target
    return mean, cov, shrunk, delta


class LedoitWolfEstimator(ParameterEstimator):
    """Estimate a multivariate Gaussian with a Ledoit-Wolf-shrunk covariance.

    Returns a :class:`MultivariateGaussianDistribution` whose mean is the sample mean and whose covariance
    is shrunk toward a scaled identity by the data-driven Ledoit-Wolf intensity. The chosen intensity is
    exposed on the returned distribution as ``dist.shrinkage`` for inspection.
    """

    def __init__(self, dim: int | None = None, name: str | None = None, keys: str | None = None):
        self.dim = dim
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> LedoitWolfAccumulatorFactory:
        """Return an accumulator factory for streaming Ledoit-Wolf sufficient statistics."""
        return LedoitWolfAccumulatorFactory(self.dim, self.keys, self.name)

    def estimate(self, nobs, suff_stat) -> MultivariateGaussianDistribution:
        """Estimate a Gaussian distribution from accumulated Ledoit-Wolf sufficient statistics."""
        s1, s2, s3, s4, n = suff_stat
        mean, _cov, shrunk, delta = _shrink(np.asarray(s1), np.asarray(s2), np.asarray(s3), float(s4), float(n))
        dist = MultivariateGaussianDistribution(mean, shrunk, name=self.name)
        dist.shrinkage = delta
        return dist


class LedoitWolfAccumulator(SequenceEncodableStatisticAccumulator):
    """Aggregate the sufficient statistics (sum x, sum xx^T, sum x||x||^2, sum ||x||^4, count)."""

    def __init__(self, dim: int | None = None, keys: str | None = None, name: str | None = None):
        self.dim = dim
        self.keys = keys
        self.name = name
        self.count = 0.0
        self.s4 = 0.0
        if dim is not None:
            self.s1 = np.zeros(dim)
            self.s2 = np.zeros((dim, dim))
            self.s3 = np.zeros(dim)
        else:
            self.s1 = self.s2 = self.s3 = None

    def _ensure(self, dim: int) -> None:
        if self.s1 is None:
            self.dim = dim
            self.s1 = np.zeros(dim)
            self.s2 = np.zeros((dim, dim))
            self.s3 = np.zeros(dim)

    def update(self, x: np.ndarray, weight: float, estimate=None) -> None:
        """Add one weighted observation to the streaming covariance-shrinkage statistics."""
        x = np.asarray(x, dtype=float)
        self._ensure(len(x))
        sq = float(x @ x)
        self.s1 += weight * x
        self.s2 += weight * np.outer(x, x)
        self.s3 += weight * sq * x
        self.s4 += weight * sq * sq
        self.count += weight

    def initialize(self, x: np.ndarray, weight: float, rng: RandomState | None) -> None:
        """Initialize the accumulator from one observation using the standard update path."""
        self.update(x, weight, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate=None) -> None:
        """Add a batch of weighted observations to the streaming sufficient statistics."""
        x = np.asarray(x, dtype=float)
        self._ensure(x.shape[1])
        sq = np.einsum("ij,ij->i", x, x)  # ||x_t||^2 per row
        xw = x.T * weights
        self.s1 += xw.sum(axis=1)
        self.s2 += np.einsum("ji,ik->jk", xw, x)
        self.s3 += (x.T * (weights * sq)).sum(axis=1)
        self.s4 += float(np.sum(weights * sq * sq))
        self.count += float(weights.sum())

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize the accumulator from a weighted batch using the standard batch update path."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat) -> LedoitWolfAccumulator:
        """Merge sufficient statistics from another Ledoit-Wolf accumulator."""
        s1, s2, s3, s4, count = suff_stat
        if s1 is None:
            return self
        if self.s1 is None:
            self.dim = len(s1)
            self.s1, self.s2, self.s3, self.s4, self.count = (np.array(s1), np.array(s2), np.array(s3), s4, count)
        else:
            self.s1 += s1
            self.s2 += s2
            self.s3 += s3
            self.s4 += s4
            self.count += count
        return self

    def value(self):
        """Return the accumulated ``(sum_x, sum_xx, sum_x_norm2, sum_norm4, count)`` tuple."""
        return self.s1, self.s2, self.s3, self.s4, self.count

    def from_value(self, x) -> LedoitWolfAccumulator:
        """Restore accumulator state from a value tuple produced by :meth:`value`."""
        self.s1, self.s2, self.s3, self.s4, self.count = x
        self.dim = None if x[0] is None else len(x[0])
        return self

    def acc_to_encoder(self) -> MultivariateGaussianDataEncoder:
        """Return the encoder expected by this accumulator."""
        return MultivariateGaussianDataEncoder(dim=self.dim)


class LedoitWolfAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for Ledoit-Wolf accumulators with fixed dimensional metadata."""

    def __init__(self, dim: int | None = None, keys: str | None = None, name: str | None = None):
        self.dim = dim
        self.keys = keys
        self.name = name

    def make(self) -> LedoitWolfAccumulator:
        """Create a fresh accumulator instance."""
        return LedoitWolfAccumulator(self.dim, self.keys, self.name)
