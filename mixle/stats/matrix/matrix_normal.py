"""Matrix normal distribution MN(M, U, V) -- a distribution over n-by-p real matrices.

The matrix normal is the Gaussian on matrices with a *separable* (Kronecker) covariance: a draw ``X``
has mean matrix ``M`` (n, p), an among-row covariance ``U`` (n, n) and an among-column covariance ``V``
(p, p), and ``vec(X) ~ N(vec(M), V (x) U)`` (column-stacking ``vec``). Equivalently ``X = M + A Z B^T``
with ``U = A A^T``, ``V = B B^T`` and ``Z`` standard normal. Its density is

    log p(X) = -np/2 log(2pi) - n/2 log|V| - p/2 log|U| - 1/2 tr(U^{-1} (X-M) V^{-1} (X-M)^T).

``U`` and ``V`` are identifiable only through their Kronecker product ``U (x) V`` (scaling ``U`` by ``c``
and ``V`` by ``1/c`` is the same law), so the estimator anchors ``V[0,0] = 1``. It is fit by the
standard flip-flop MLE: alternate the closed-form updates of ``U`` given ``V`` and ``V`` given ``U`` to
convergence -- here from a fixed sufficient statistic (the row-blocked second moment), so it converges
inside a single ``estimate`` call.


Reference: Dawid, 'Some matrix-variate distribution theory', Biometrika (1981).
"""

import math
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


class MatrixNormalDistribution(SequenceEncodableProbabilityDistribution):
    """Matrix normal distribution over ``(n, p)`` matrices with row covariance ``U`` and column covariance ``V``."""

    def __init__(
        self,
        mean: np.ndarray,
        row_covar: np.ndarray,
        col_covar: np.ndarray,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        m = np.asarray(mean, dtype=np.float64)
        u = np.asarray(row_covar, dtype=np.float64)
        v = np.asarray(col_covar, dtype=np.float64)
        if m.ndim != 2:
            raise ValueError("mean must be a 2-D (n, p) matrix")
        self.n, self.p = m.shape
        if u.shape != (self.n, self.n) or v.shape != (self.p, self.p):
            raise ValueError("row_covar must be (n, n) and col_covar (p, p) to match the (n, p) mean")
        self.mean = m
        self.row_covar = u
        self.col_covar = v
        self.name = name
        self.keys = keys
        su, logdet_u = np.linalg.slogdet(u)
        sv, logdet_v = np.linalg.slogdet(v)
        if su <= 0 or sv <= 0:
            raise ValueError("row_covar and col_covar must be positive definite")
        self._u_inv = np.linalg.inv(u)
        self._v_inv = np.linalg.inv(v)
        self._chol_u = np.linalg.cholesky(u)
        self._chol_v = np.linalg.cholesky(v)
        self._log_norm = -0.5 * (self.n * self.p * math.log(2.0 * math.pi) + self.n * logdet_v + self.p * logdet_u)

    def __str__(self) -> str:
        return "MatrixNormalDistribution(%s, %s, %s, name=%s, keys=%s)" % (
            repr(self.mean.tolist()),
            repr(self.row_covar.tolist()),
            repr(self.col_covar.tolist()),
            repr(self.name),
            repr(self.keys),
        )

    def density(self, x: np.ndarray) -> float:
        """Return the matrix-normal density at a single ``(n, p)`` matrix."""
        return math.exp(self.log_density(x))

    def log_density(self, x: np.ndarray) -> float:
        """Return the log-density at a single ``(n, p)`` matrix."""
        c = np.asarray(x, dtype=np.float64) - self.mean
        quad = np.trace(self._u_inv @ c @ self._v_inv @ c.T)
        return float(self._log_norm - 0.5 * quad)

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Vectorized log-density for a stack of matrices, shape ``(N, n, p)``."""
        c = np.asarray(x, dtype=np.float64) - self.mean
        quad = np.einsum("ab,nbc,cd,nad->n", self._u_inv, c, self._v_inv, c, optimize=True)
        return self._log_norm - 0.5 * quad

    def sampler(self, seed: int | None = None) -> "MatrixNormalSampler":
        """Return a sampler for drawing matrices from this distribution."""
        return MatrixNormalSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "MatrixNormalEstimator":
        """Return a flip-flop MLE estimator for the mean and the two covariance factors."""
        return MatrixNormalEstimator(self.n, self.p, name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> "MatrixNormalDataEncoder":
        """Return the data encoder used by this distribution for vectorized methods."""
        return MatrixNormalDataEncoder()


class MatrixNormalSampler(DistributionSampler):
    """Draw matrices by ``X = M + chol(U) Z chol(V)^T`` with ``Z`` standard normal."""

    def __init__(self, dist: MatrixNormalDistribution, seed: int | None = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: int | None = None) -> np.ndarray:
        """Draw one matrix or a batch of independent matrix-normal samples."""
        d = self.dist
        n = 1 if size is None else int(size)
        z = self.rng.randn(n, d.n, d.p)
        x = d.mean[None, :, :] + np.einsum("ab,nbc,dc->nad", d._chol_u, z, d._chol_v, optimize=True)
        return x[0] if size is None else x


class MatrixNormalAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate the mean and the row-blocked second moment ``T[a,b,c,d] = sum_i X_i[a,c] X_i[b,d]``."""

    def __init__(self, n: int, p: int, name: str | None = None, keys: str | None = None) -> None:
        self.n = n
        self.p = p
        self.sum_x = np.zeros((n, p), dtype=np.float64)
        self.t = np.zeros((n, n, p, p), dtype=np.float64)
        self.count = 0.0
        self.name = name
        self.keys = keys

    def update(self, x: np.ndarray, weight: float, estimate: MatrixNormalDistribution | None) -> None:
        """Accumulate weighted mean and row-blocked second moments for one matrix."""
        xx = np.asarray(x, dtype=np.float64)
        self.sum_x += weight * xx
        self.t += weight * np.einsum("ac,bd->abcd", xx, xx, optimize=True)
        self.count += weight

    def initialize(self, x: np.ndarray, weight: float, rng: RandomState | None) -> None:
        """Initialize the sufficient statistics with one weighted matrix."""
        self.update(x, weight, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: MatrixNormalDistribution | None) -> None:
        """Accumulate weighted mean and block moments for encoded matrices."""
        xx = np.asarray(x, dtype=np.float64)
        w = np.asarray(weights, dtype=np.float64)
        xw = xx * w[:, None, None]
        self.sum_x += xw.sum(axis=0)
        self.t += np.einsum("iac,ibd->abcd", xw, xx, optimize=True)
        self.count += float(w.sum())

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize the sufficient statistics from encoded matrices."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[np.ndarray, np.ndarray, float]) -> "MatrixNormalAccumulator":
        """Merge serialized matrix-normal sufficient statistics into this accumulator."""
        self.sum_x += suff_stat[0]
        self.t += suff_stat[1]
        self.count += suff_stat[2]
        return self

    def value(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Return the weighted sum, block second moment, and total weight."""
        return self.sum_x.copy(), self.t.copy(), self.count

    def from_value(self, x: tuple[np.ndarray, np.ndarray, float]) -> "MatrixNormalAccumulator":
        """Restore the accumulator from serialized matrix-normal statistics."""
        self.sum_x = np.asarray(x[0], dtype=np.float64).copy()
        self.t = np.asarray(x[1], dtype=np.float64).copy()
        self.count = float(x[2])
        self.n, self.p = self.sum_x.shape
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

    def acc_to_encoder(self) -> "MatrixNormalDataEncoder":
        """Return an encoder compatible with matrix-normal vectorized updates."""
        return MatrixNormalDataEncoder()


class MatrixNormalAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for MatrixNormalAccumulator."""

    def __init__(self, n: int, p: int, name: str | None = None, keys: str | None = None) -> None:
        self.n = n
        self.p = p
        self.name = name
        self.keys = keys

    def make(self) -> MatrixNormalAccumulator:
        """Create an empty matrix-normal accumulator."""
        return MatrixNormalAccumulator(self.n, self.p, name=self.name, keys=self.keys)


class MatrixNormalEstimator(ParameterEstimator):
    """Flip-flop maximum-likelihood estimator for the matrix-normal parameters."""

    def __init__(
        self, n: int, p: int, max_iter: int = 100, tol: float = 1.0e-9, name: str | None = None, keys: str | None = None
    ) -> None:
        self.n = n
        self.p = p
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> MatrixNormalAccumulatorFactory:
        """Return a factory for matrix-normal sufficient-statistic accumulators."""
        return MatrixNormalAccumulatorFactory(self.n, self.p, name=self.name, keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat: tuple[np.ndarray, np.ndarray, float]) -> MatrixNormalDistribution:
        """Estimate the mean and covariance factors using flip-flop updates."""
        sum_x, t, count = suff_stat
        n, p = self.n, self.p
        if count <= 0.0:
            return MatrixNormalDistribution(np.zeros((n, p)), np.eye(n), np.eye(p), name=self.name, keys=self.keys)
        mean = sum_x / count
        # centered row-blocked second moment: T_c[a,b,c,d] = sum_i (X_i-M)[a,c] (X_i-M)[b,d]
        tc = t - count * np.einsum("ac,bd->abcd", mean, mean, optimize=True)
        u = np.eye(n)
        v = np.eye(p)
        for _ in range(self.max_iter):
            v_inv = np.linalg.inv(v)
            u_new = np.einsum("abcd,cd->ab", tc, v_inv, optimize=True) / (count * p)
            u_new = 0.5 * (u_new + u_new.T)
            u_inv = np.linalg.inv(u_new)
            v_new = np.einsum("abcd,ab->cd", tc, u_inv, optimize=True) / (count * n)
            v_new = 0.5 * (v_new + v_new.T)
            scale = v_new[0, 0]  # anchor V[0,0]=1 to fix the U<->V scale ambiguity
            v_new = v_new / scale
            u_new = u_new * scale
            if np.max(np.abs(u_new - u)) < self.tol and np.max(np.abs(v_new - v)) < self.tol:
                u, v = u_new, v_new
                break
            u, v = u_new, v_new
        return MatrixNormalDistribution(mean, u, v, name=self.name, keys=self.keys)


class MatrixNormalDataEncoder(DataSequenceEncoder):
    """Encode a sequence of ``(n, p)`` matrices as an ``(N, n, p)`` float array."""

    def __str__(self) -> str:
        return "MatrixNormalDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, MatrixNormalDataEncoder)

    def seq_encode(self, x: Sequence[np.ndarray]) -> np.ndarray:
        """Encode matrices as a floating-point stack for vectorized evaluation."""
        return np.asarray(x, dtype=np.float64)
