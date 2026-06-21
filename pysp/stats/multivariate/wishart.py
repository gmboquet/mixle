"""Wishart distribution -- a distribution over symmetric positive-definite ``p``-by-``p`` matrices.

The Wishart is the distribution of a scatter matrix ``X = sum_{i=1}^{df} z_i z_i^T`` with
``z_i ~ N(0, scale)``; it is the matrix generalisation of the chi-square / gamma and the standard model
for random covariance matrices (and the conjugate prior for a Gaussian precision). With ``df >= p``
degrees of freedom and scale matrix ``V``,

    log f(X) = (df-p-1)/2 log|X| - 1/2 tr(V^{-1} X) - df p/2 log 2 - df/2 log|V| - log Gamma_p(df/2),

where ``Gamma_p`` is the multivariate gamma. ``df`` is a fixed, known parameter; since ``E[X] = df V``
the scale ``V`` is estimated in closed form as the mean scatter divided by ``df``.
"""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import RandomState
from scipy.special import multigammaln

from pysp.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)


class WishartDistribution(SequenceEncodableProbabilityDistribution):
    """Wishart distribution with ``df`` degrees of freedom and scale matrix ``scale`` (p, p)."""

    def __init__(self, df: float, scale: np.ndarray, name: str | None = None, keys: str | None = None) -> None:
        v = np.asarray(scale, dtype=np.float64)
        if v.ndim != 2 or v.shape[0] != v.shape[1]:
            raise ValueError("scale must be a square matrix")
        self.dim = v.shape[0]
        if df < self.dim:
            raise ValueError("df must be >= the matrix dimension p")
        sign, logdet = np.linalg.slogdet(v)
        if sign <= 0:
            raise ValueError("scale must be positive definite")
        self.df = float(df)
        self.scale = v
        self.name = name
        self.keys = keys
        self._scale_inv = np.linalg.inv(v)
        self._chol = np.linalg.cholesky(v)
        p = self.dim
        self._log_norm = (
            -(self.df * p / 2.0) * math.log(2.0) - (self.df / 2.0) * logdet - multigammaln(self.df / 2.0, p)
        )

    def __str__(self) -> str:
        return "WishartDistribution(%s, %s, name=%s, keys=%s)" % (
            repr(self.df),
            repr(self.scale.tolist()),
            repr(self.name),
            repr(self.keys),
        )

    def density(self, x: np.ndarray) -> float:
        """Return the density at a single ``(p, p)`` SPD matrix."""
        return math.exp(self.log_density(x))

    def log_density(self, x: np.ndarray) -> float:
        """Return the log-density at a single ``(p, p)`` SPD matrix (``-inf`` if not positive definite)."""
        xx = np.asarray(x, dtype=np.float64)
        sign, logdet = np.linalg.slogdet(xx)
        if sign <= 0:
            return -np.inf
        tr = np.trace(self._scale_inv @ xx)
        return float(self._log_norm + (self.df - self.dim - 1.0) / 2.0 * logdet - 0.5 * tr)

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Vectorized log-density for a stack of SPD matrices, shape ``(N, p, p)``."""
        xx = np.asarray(x, dtype=np.float64)
        sign, logdet = np.linalg.slogdet(xx)
        tr = np.einsum("ab,nba->n", self._scale_inv, xx, optimize=True)
        rv = self._log_norm + (self.df - self.dim - 1.0) / 2.0 * logdet - 0.5 * tr
        return np.where(sign <= 0, -np.inf, rv)

    def sampler(self, seed: int | None = None) -> "WishartSampler":
        """Return a sampler for drawing SPD matrices from this distribution."""
        return WishartSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "WishartEstimator":
        """Return a closed-form estimator for the scale at the fixed degrees of freedom ``df``."""
        return WishartEstimator(self.dim, self.df, name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> "WishartDataEncoder":
        """Return the data encoder used by this distribution for vectorized methods."""
        return WishartDataEncoder()


class WishartSampler(DistributionSampler):
    """Draw SPD matrices by the Bartlett decomposition ``X = L A A^T L^T`` with ``V = L L^T``."""

    def __init__(self, dist: WishartDistribution, seed: int | None = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist

    def _one(self) -> np.ndarray:
        d = self.dist
        p = d.dim
        a = np.zeros((p, p))
        for i in range(p):
            a[i, i] = math.sqrt(self.rng.chisquare(d.df - i))
            for j in range(i):
                a[i, j] = self.rng.randn()
        la = d._chol @ a
        return la @ la.T

    def sample(self, size: int | None = None) -> np.ndarray:
        if size is None:
            return self._one()
        return np.stack([self._one() for _ in range(int(size))])


class WishartAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate the weighted sum of matrices ``sum_i w_i X_i`` and the total weight."""

    def __init__(self, dim: int, name: str | None = None, keys: str | None = None) -> None:
        self.dim = dim
        self.sum_x = np.zeros((dim, dim), dtype=np.float64)
        self.count = 0.0
        self.name = name
        self.key = keys

    def update(self, x: np.ndarray, weight: float, estimate: WishartDistribution | None) -> None:
        self.sum_x += weight * np.asarray(x, dtype=np.float64)
        self.count += weight

    def initialize(self, x: np.ndarray, weight: float, rng: RandomState | None) -> None:
        self.update(x, weight, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: WishartDistribution | None) -> None:
        xx = np.asarray(x, dtype=np.float64)
        w = np.asarray(weights, dtype=np.float64)
        self.sum_x += np.einsum("n,nab->ab", w, xx, optimize=True)
        self.count += float(w.sum())

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[np.ndarray, float]) -> "WishartAccumulator":
        self.sum_x += suff_stat[0]
        self.count += suff_stat[1]
        return self

    def value(self) -> tuple[np.ndarray, float]:
        return self.sum_x.copy(), self.count

    def from_value(self, x: tuple[np.ndarray, float]) -> "WishartAccumulator":
        self.sum_x = np.asarray(x[0], dtype=np.float64).copy()
        self.count = float(x[1])
        self.dim = self.sum_x.shape[0]
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        if self.key is not None:
            if self.key in stats_dict:
                stats_dict[self.key].combine(self.value())
            else:
                stats_dict[self.key] = self

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        if self.key is not None and self.key in stats_dict:
            self.from_value(stats_dict[self.key].value())

    def acc_to_encoder(self) -> "WishartDataEncoder":
        return WishartDataEncoder()


class WishartAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for WishartAccumulator."""

    def __init__(self, dim: int, name: str | None = None, keys: str | None = None) -> None:
        self.dim = dim
        self.name = name
        self.keys = keys

    def make(self) -> WishartAccumulator:
        return WishartAccumulator(self.dim, name=self.name, keys=self.keys)


class WishartEstimator(ParameterEstimator):
    """Closed-form scale estimator at fixed ``df``: ``V = (1/df) * mean(X)`` since ``E[X] = df V``."""

    def __init__(self, dim: int, df: float, name: str | None = None, keys: str | None = None) -> None:
        self.dim = dim
        self.df = float(df)
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> WishartAccumulatorFactory:
        return WishartAccumulatorFactory(self.dim, name=self.name, keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat: tuple[np.ndarray, float]) -> WishartDistribution:
        sum_x, count = suff_stat
        if count <= 0.0:
            return WishartDistribution(self.df, np.eye(self.dim), name=self.name, keys=self.keys)
        scale = (sum_x / count) / self.df  # E[X] = df V
        scale = 0.5 * (scale + scale.T)
        return WishartDistribution(self.df, scale, name=self.name, keys=self.keys)


class WishartDataEncoder(DataSequenceEncoder):
    """Encode a sequence of ``(p, p)`` matrices as an ``(N, p, p)`` float array."""

    def __str__(self) -> str:
        return "WishartDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, WishartDataEncoder)

    def seq_encode(self, x: Sequence[np.ndarray]) -> np.ndarray:
        return np.asarray(x, dtype=np.float64)
