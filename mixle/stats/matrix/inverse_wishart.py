"""Inverse-Wishart distribution -- a distribution over symmetric positive-definite matrices.

If ``X^{-1} ~ Wishart(df, scale^{-1})`` then ``X ~ InverseWishart(df, scale)``; it is the conjugate
prior for a multivariate-normal covariance and the standard model for a random covariance matrix
(rather than a random precision). With ``df > p - 1`` and scale matrix ``Psi``,

    log f(X) = df/2 log|Psi| - df p/2 log 2 - log Gamma_p(df/2)
               - (df+p+1)/2 log|X| - 1/2 tr(Psi X^{-1}).

``df`` is a fixed, known parameter; since ``E[X] = Psi / (df - p - 1)`` the scale is estimated in closed
form as ``Psi = (df - p - 1) * mean(X)`` (for ``df > p + 1``).


Reference: Mardia, Kent & Bibby, *Multivariate Analysis* (Academic Press, 1979).
"""

import math
from collections.abc import Sequence

import numpy as np
from numpy.random import RandomState
from scipy.special import multigammaln

from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    StatisticAccumulatorFactory,
)
from mixle.stats.matrix.wishart import WishartDistribution, _MeanScatterAccumulator


class InverseWishartDistribution(SequenceEncodableProbabilityDistribution):
    """Inverse-Wishart distribution with ``df`` degrees of freedom and scale matrix ``scale`` (p, p)."""

    def __init__(self, df: float, scale: np.ndarray, name: str | None = None, keys: str | None = None) -> None:
        v = np.asarray(scale, dtype=np.float64)
        if v.ndim != 2 or v.shape[0] != v.shape[1]:
            raise ValueError("scale must be a square matrix")
        self.dim = v.shape[0]
        if df <= self.dim - 1:
            raise ValueError("df must be > p - 1")
        sign, logdet = np.linalg.slogdet(v)
        if sign <= 0:
            raise ValueError("scale must be positive definite")
        self.df = float(df)
        self.scale = v
        self.name = name
        self.keys = keys
        p = self.dim
        self._log_norm = (self.df / 2.0) * logdet - (self.df * p / 2.0) * math.log(2.0) - multigammaln(self.df / 2.0, p)

    def __str__(self) -> str:
        return "InverseWishartDistribution(%s, %s, name=%s, keys=%s)" % (
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
        tr = np.trace(self.scale @ np.linalg.inv(xx))
        return float(self._log_norm - (self.df + self.dim + 1.0) / 2.0 * logdet - 0.5 * tr)

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Vectorized log-density for a stack of SPD matrices, shape ``(N, p, p)``."""
        xx = np.asarray(x, dtype=np.float64)
        sign, logdet = np.linalg.slogdet(xx)
        x_inv = np.linalg.inv(xx)
        tr = np.einsum("ab,nba->n", self.scale, x_inv, optimize=True)
        rv = self._log_norm - (self.df + self.dim + 1.0) / 2.0 * logdet - 0.5 * tr
        return np.where(sign <= 0, -np.inf, rv)

    def sampler(self, seed: int | None = None) -> "InverseWishartSampler":
        """Return a sampler for drawing SPD matrices from this distribution."""
        return InverseWishartSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "InverseWishartEstimator":
        """Return a closed-form estimator for the scale at the fixed degrees of freedom ``df``."""
        return InverseWishartEstimator(self.dim, self.df, name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> "InverseWishartDataEncoder":
        """Return the data encoder used by this distribution for vectorized methods."""
        return InverseWishartDataEncoder()


class InverseWishartSampler(DistributionSampler):
    """Draw SPD matrices by inverting a ``Wishart(df, scale^{-1})`` draw."""

    def __init__(self, dist: InverseWishartDistribution, seed: int | None = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist
        self._wishart = WishartDistribution(dist.df, np.linalg.inv(dist.scale)).sampler(
            seed=self.rng.randint(0, 2**31 - 1)
        )

    def sample(self, size: int | None = None, *, batched: bool = True) -> np.ndarray:
        """Draw one or more inverse-Wishart SPD matrix samples."""
        w = self._wishart.sample(size=size)
        if size is None:
            return np.linalg.inv(w)
        return np.linalg.inv(w)


class InverseWishartAccumulator(_MeanScatterAccumulator):
    """Accumulate the weighted sum of matrices ``sum_i w_i X_i`` and the total weight."""

    def acc_to_encoder(self) -> "InverseWishartDataEncoder":
        """Return the encoder compatible with the accumulated matrix statistics."""
        return InverseWishartDataEncoder()


class InverseWishartAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for InverseWishartAccumulator."""

    def __init__(self, dim: int, name: str | None = None, keys: str | None = None) -> None:
        self.dim = dim
        self.name = name
        self.keys = keys

    def make(self) -> InverseWishartAccumulator:
        """Create an accumulator for weighted inverse-Wishart matrix observations."""
        return InverseWishartAccumulator(self.dim, name=self.name, keys=self.keys)


class InverseWishartEstimator(ParameterEstimator):
    """Closed-form scale estimator at fixed ``df``: ``Psi = (df-p-1) * mean(X)`` since ``E[X] = Psi/(df-p-1)``."""

    def __init__(self, dim: int, df: float, name: str | None = None, keys: str | None = None) -> None:
        self.dim = dim
        self.df = float(df)
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> InverseWishartAccumulatorFactory:
        """Return an accumulator factory for estimating the fixed-df scale matrix."""
        return InverseWishartAccumulatorFactory(self.dim, name=self.name, keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat: tuple[np.ndarray, float]) -> InverseWishartDistribution:
        """Estimate the inverse-Wishart scale matrix from weighted matrix means."""
        sum_x, count = suff_stat
        factor = self.df - self.dim - 1.0
        if count <= 0.0 or factor <= 0.0:
            return InverseWishartDistribution(self.df, np.eye(self.dim), name=self.name, keys=self.keys)
        scale = factor * (sum_x / count)  # E[X] = Psi/(df-p-1)
        scale = 0.5 * (scale + scale.T)
        return InverseWishartDistribution(self.df, scale, name=self.name, keys=self.keys)


class InverseWishartDataEncoder(DataSequenceEncoder):
    """Encode a sequence of ``(p, p)`` matrices as an ``(N, p, p)`` float array."""

    def __str__(self) -> str:
        return "InverseWishartDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, InverseWishartDataEncoder)

    def seq_encode(self, x: Sequence[np.ndarray]) -> np.ndarray:
        """Encode a sequence of SPD matrices as a floating matrix stack."""
        return np.asarray(x, dtype=np.float64)
