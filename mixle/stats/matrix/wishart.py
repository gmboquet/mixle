"""Wishart distribution -- a distribution over symmetric positive-definite ``p``-by-``p`` matrices.

The Wishart is the distribution of a scatter matrix ``X = sum_{i=1}^{df} z_i z_i^T`` with
``z_i ~ N(0, scale)``; it is the matrix generalisation of the chi-square / gamma and the standard model
for random covariance matrices (and the conjugate prior for a Gaussian precision). With ``df >= p``
degrees of freedom and scale matrix ``V``,

    log f(X) = (df-p-1)/2 log|X| - 1/2 tr(V^{-1} X) - df p/2 log 2 - df/2 log|V| - log Gamma_p(df/2),

where ``Gamma_p`` is the multivariate gamma. Since ``E[X] = df V`` the scale ``V`` is estimated in closed
form as the mean scatter divided by ``df``. The degrees of freedom may be supplied (``WishartEstimator(dim,
df=value)``) or estimated by maximum likelihood (``WishartEstimator(dim, df=None)``) -- the latter adds the
``sum log det(X)`` sufficient statistic and solves the profile-likelihood score for ``df`` by Newton's
method.


Reference: Wishart, 'The generalised product moment distribution in samples...', Biometrika (1928).
"""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import RandomState
from scipy.special import digamma, multigammaln, polygamma

from mixle.stats.compute.pdist import (
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
        """Draw one SPD matrix or a stacked batch of independent Wishart samples."""
        if size is None:
            return self._one()
        return np.stack([self._one() for _ in range(int(size))])


class _MeanScatterAccumulator(SequenceEncodableStatisticAccumulator):
    """Shared accumulator for (inverse-)Wishart: weighted matrix sum ``sum_i w_i X_i`` and total weight.

    Subclasses override :meth:`acc_to_encoder` to return the matching :class:`DataSequenceEncoder`;
    the weighted sufficient-statistic update path is shared.
    """

    def __init__(self, dim: int, name: str | None = None, keys: str | None = None) -> None:
        self.dim = dim
        self.sum_x = np.zeros((dim, dim), dtype=np.float64)
        self.count = 0.0
        self.name = name
        self.keys = keys

    def update(self, x: np.ndarray, weight: float, estimate: Any | None) -> None:
        self.sum_x += weight * np.asarray(x, dtype=np.float64)
        self.count += weight

    def initialize(self, x: np.ndarray, weight: float, rng: RandomState | None) -> None:
        self.update(x, weight, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: Any | None) -> None:
        xx = np.asarray(x, dtype=np.float64)
        w = np.asarray(weights, dtype=np.float64)
        self.sum_x += np.einsum("n,nab->ab", w, xx, optimize=True)
        self.count += float(w.sum())

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[np.ndarray, float]) -> "_MeanScatterAccumulator":
        self.sum_x += suff_stat[0]
        self.count += suff_stat[1]
        return self

    def value(self) -> tuple[np.ndarray, float]:
        return self.sum_x.copy(), self.count

    def from_value(self, x: tuple[np.ndarray, float]) -> "_MeanScatterAccumulator":
        self.sum_x = np.asarray(x[0], dtype=np.float64).copy()
        self.count = float(x[1])
        self.dim = self.sum_x.shape[0]
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        if self.keys is not None:
            if self.keys in stats_dict:
                stats_dict[self.keys].combine(self.value())
            else:
                stats_dict[self.keys] = self

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        if self.keys is not None and self.keys in stats_dict:
            self.from_value(stats_dict[self.keys].value())

    def acc_to_encoder(self) -> DataSequenceEncoder:
        raise NotImplementedError


class WishartAccumulator(_MeanScatterAccumulator):
    """Accumulate ``sum_i w_i X_i``, the total weight, and ``sum_i w_i log det(X_i)``.

    The extra ``sum_logdet`` statistic is what enables maximum-likelihood estimation of the degrees of
    freedom (the inverse-Wishart base accumulator, which does not need it, is left untouched).
    """

    def __init__(self, dim: int, name: str | None = None, keys: str | None = None) -> None:
        super().__init__(dim, name=name, keys=keys)
        self.sum_logdet = 0.0

    @staticmethod
    def _logdet(xx: np.ndarray) -> np.ndarray:
        sign, logdet = np.linalg.slogdet(xx)
        return np.where(sign > 0, logdet, -np.inf)

    def update(self, x: np.ndarray, weight: float, estimate: Any | None) -> None:
        """Accumulate matrix scatter and log-determinant statistics for one observation."""
        super().update(x, weight, estimate)
        self.sum_logdet += weight * float(self._logdet(np.asarray(x, dtype=np.float64)))

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: Any | None) -> None:
        """Accumulate matrix scatter and log-determinants from encoded observations."""
        super().seq_update(x, weights, estimate)
        w = np.asarray(weights, dtype=np.float64)
        self.sum_logdet += float(np.dot(w, self._logdet(np.asarray(x, dtype=np.float64))))

    def combine(self, suff_stat: tuple[np.ndarray, float, float]) -> "WishartAccumulator":
        """Merge serialized Wishart sufficient statistics into this accumulator."""
        super().combine((suff_stat[0], suff_stat[1]))
        self.sum_logdet += float(suff_stat[2])
        return self

    def value(self) -> tuple[np.ndarray, float, float]:
        """Return scatter, total weight, and weighted log-determinant sum."""
        return self.sum_x.copy(), self.count, self.sum_logdet

    def from_value(self, x: tuple[np.ndarray, float, float]) -> "WishartAccumulator":
        """Restore the accumulator from serialized Wishart sufficient statistics."""
        super().from_value((x[0], x[1]))
        self.sum_logdet = float(x[2])
        return self

    def scale(self, c: float) -> "WishartAccumulator":
        """Scale accumulated Wishart sufficient statistics by a constant."""
        self.sum_x *= c
        self.count *= c
        self.sum_logdet *= c
        return self

    def acc_to_encoder(self) -> "WishartDataEncoder":
        """Return an encoder for SPD matrix observations."""
        return WishartDataEncoder()


class WishartAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for WishartAccumulator."""

    def __init__(self, dim: int, name: str | None = None, keys: str | None = None) -> None:
        self.dim = dim
        self.name = name
        self.keys = keys

    def make(self) -> WishartAccumulator:
        """Create an empty Wishart accumulator."""
        return WishartAccumulator(self.dim, name=self.name, keys=self.keys)


def _solve_wishart_df(mean_logdet: float, logdet_scatter: float, dim: int, df0: float | None = None) -> float:
    """MLE of the Wishart degrees of freedom by Newton's method on the profile log-likelihood.

    With the scale profiled out (``V = mean(X)/df``), the profile score in ``df`` is
    ``g(df) = -p/2 log2 - logdet(S)/2 + p/2 log df - 1/2 sum_j psi((df+1-j)/2) + mean_logdet/2`` where
    ``S = mean(X)`` and ``mean_logdet = mean(log det X)``; ``g'(df) = p/(2 df) - 1/4 sum_j psi'((df+1-j)/2)``.
    The root is the MLE; ``df`` is constrained to ``> p - 1`` (the Wishart density's existence bound).
    """
    p = dim
    lo = float(p - 1) + 1e-6
    df = float(df0) if df0 is not None else float(p + 1.0)
    df = max(df, lo + 1e-3)
    for _ in range(100):
        j = np.arange(1, p + 1)
        g = (
            -p / 2.0 * math.log(2.0)
            - logdet_scatter / 2.0
            + p / 2.0 * math.log(df)
            - 0.5 * float(np.sum(digamma((df + 1.0 - j) / 2.0)))
            + mean_logdet / 2.0
        )
        gp = p / (2.0 * df) - 0.25 * float(np.sum(polygamma(1, (df + 1.0 - j) / 2.0)))
        if not np.isfinite(g) or not np.isfinite(gp) or gp == 0.0:
            break
        step = g / gp
        df_new = df - step
        if df_new <= lo:  # keep inside the feasible region with a damped step
            df_new = 0.5 * (df + lo)
        if abs(df_new - df) < 1e-8:
            df = df_new
            break
        df = df_new
    return df


class WishartEstimator(ParameterEstimator):
    """Closed-form scale estimator (``V = mean(X)/df``); ``df=None`` also fits the degrees of freedom by MLE.

    With a fixed ``df`` the estimator returns only the closed-form scale (``E[X] = df V``). With ``df=None``
    it additionally estimates the degrees of freedom from ``sum_i w_i log det(X_i)`` via Newton's method on
    the profile log-likelihood (:func:`_solve_wishart_df`).
    """

    def __init__(self, dim: int, df: float | None = None, name: str | None = None, keys: str | None = None) -> None:
        self.dim = dim
        self.df = None if df is None else float(df)
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> WishartAccumulatorFactory:
        """Return a factory for Wishart sufficient-statistic accumulators."""
        return WishartAccumulatorFactory(self.dim, name=self.name, keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat: tuple[np.ndarray, float, float]) -> WishartDistribution:
        """Estimate the Wishart scale and optionally degrees of freedom."""
        sum_x, count, sum_logdet = suff_stat
        if count <= 0.0:
            df = self.df if self.df is not None else float(self.dim + 1.0)
            return WishartDistribution(df, np.eye(self.dim), name=self.name, keys=self.keys)
        scatter = sum_x / count  # E[X] = df V, so mean(X) = df V
        if self.df is not None:
            df = self.df
        else:
            sign, logdet_scatter = np.linalg.slogdet(scatter)
            df = _solve_wishart_df(sum_logdet / count, float(logdet_scatter), self.dim, df0=self.dim + 1.0)
        scale = scatter / df
        scale = 0.5 * (scale + scale.T)
        return WishartDistribution(df, scale, name=self.name, keys=self.keys)


class WishartDataEncoder(DataSequenceEncoder):
    """Encode a sequence of ``(p, p)`` matrices as an ``(N, p, p)`` float array."""

    def __str__(self) -> str:
        return "WishartDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, WishartDataEncoder)

    def seq_encode(self, x: Sequence[np.ndarray]) -> np.ndarray:
        """Encode SPD matrix observations as a floating-point stack."""
        return np.asarray(x, dtype=np.float64)
