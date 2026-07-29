"""Matrix normal distribution MN(M, U, V) -- a distribution over n-by-p real matrices.

The matrix normal is the Gaussian on matrices with a *separable* (Kronecker) covariance: a draw ``X``
has mean matrix ``M`` (n, p), an among-row covariance ``U`` (n, n) and an among-column covariance ``V``
(p, p), and ``vec(X) ~ N(vec(M), V (x) U)`` (column-stacking ``vec``). Equivalently ``X = M + A Z B^T``
with ``U = A A^T``, ``V = B B^T`` and ``Z`` standard normal. Its density is

    log p(X) = -np/2 log(2pi) - n/2 log|V| - p/2 log|U| - 1/2 tr(U^{-1} (X-M) V^{-1} (X-M)^T).

``U`` and ``V`` are identifiable only through their Kronecker product ``U (x) V`` (scaling ``U`` by ``c``
and ``V`` by ``1/c`` is the same law), so the estimator anchors ``V[0,0] = 1``. It is fit by the
standard flip-flop MLE: alternate the closed-form updates of ``U`` given ``V``
and ``V`` given ``U``. Fitting returns only after a finite positive-definite
convergence certificate; low-rank, non-identifiable, or exhausted fits raise a
typed diagnostic error instead of fabricating covariance factors.


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
from mixle.stats.matrix.wishart import (
    _validated_dimension,
    _validated_sample_size,
    _validated_weight,
    _validated_weights,
)
from mixle.utils.vector import cholesky_logdet


class MatrixNormalFitError(RuntimeError):
    """Raised when a matrix-normal MLE is invalid or non-identifiable."""


def _matrix_normal_event(value: Any, n: int, p: int) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("matrix-normal observation must be numeric") from exc
    if result.shape != (n, p):
        raise ValueError("matrix-normal observation must have exact shape (%d, %d)" % (n, p))
    if np.any(~np.isfinite(result)):
        raise ValueError("matrix-normal observation must be finite")
    return result


def _matrix_normal_batch(value: Any, n: int, p: int) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("matrix-normal observations must be numeric") from exc
    if result.shape == (0,):
        return np.empty((0, n, p), dtype=np.float64)
    if result.ndim != 3 or result.shape[1:] != (n, p):
        raise ValueError("matrix-normal observations must have exact shape (N, %d, %d)" % (n, p))
    if np.any(~np.isfinite(result)):
        raise ValueError("matrix-normal observations must be finite")
    return result


def _validated_matrix_normal_statistics(
    value: Any,
    n: int,
    p: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError("matrix-normal sufficient statistics must be a three-item tuple")
    if isinstance(value[2], (bool, np.bool_)) or np.ndim(value[2]) != 0:
        raise TypeError("matrix-normal sufficient-statistic count must be a real scalar")
    try:
        sum_x = np.asarray(value[0], dtype=np.float64)
        moment = np.asarray(value[1], dtype=np.float64)
        count = float(value[2])
    except (TypeError, ValueError) as exc:
        raise ValueError("matrix-normal sufficient statistics must be numeric") from exc
    if sum_x.shape != (n, p) or np.any(~np.isfinite(sum_x)):
        raise ValueError("matrix-normal sum must be finite with exact shape (%d, %d)" % (n, p))
    if moment.shape != (n, n, p, p) or np.any(~np.isfinite(moment)):
        raise ValueError("matrix-normal second moment must be finite with exact shape (%d, %d, %d, %d)" % (n, n, p, p))
    if not np.isfinite(count) or count < 0.0:
        raise ValueError("matrix-normal sufficient-statistic count must be finite and non-negative")
    if count == 0.0 and (np.any(sum_x != 0.0) or np.any(moment != 0.0)):
        raise ValueError("empty matrix-normal sufficient statistics must have zero moments")
    if not np.array_equal(moment, moment.transpose(1, 0, 3, 2)):
        raise ValueError("matrix-normal second moment must have exact paired symmetry")
    return sum_x.copy(), moment.copy(), count


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
        try:
            m = np.asarray(mean, dtype=np.float64)
            u = np.asarray(row_covar, dtype=np.float64)
            v = np.asarray(col_covar, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("matrix-normal parameters must be numeric") from exc
        if m.ndim != 2 or 0 in m.shape:
            raise ValueError("mean must be a non-empty 2-D (n, p) matrix")
        if np.any(~np.isfinite(m)):
            raise ValueError("mean must be finite")
        self.n, self.p = m.shape
        if u.shape != (self.n, self.n) or v.shape != (self.p, self.p):
            raise ValueError("row_covar must be (n, n) and col_covar (p, p) to match the (n, p) mean")
        if np.any(~np.isfinite(u)) or np.any(~np.isfinite(v)):
            raise ValueError("row_covar and col_covar must be finite")
        if not np.array_equal(u, u.T) or not np.array_equal(v, v.T):
            raise ValueError("row_covar and col_covar must be exactly symmetric")
        self.mean = m.copy()
        self.row_covar = u.copy()
        self.col_covar = v.copy()
        self.mean.setflags(write=False)
        self.row_covar.setflags(write=False)
        self.col_covar.setflags(write=False)
        self.name = name
        self.keys = keys
        logdet_u = cholesky_logdet(self.row_covar)
        logdet_v = cholesky_logdet(self.col_covar)
        if logdet_u is None or logdet_v is None:
            raise ValueError("row_covar and col_covar must be positive definite")
        self._u_inv = np.linalg.inv(self.row_covar)
        self._v_inv = np.linalg.inv(self.col_covar)
        self._chol_u = np.linalg.cholesky(self.row_covar)
        self._chol_v = np.linalg.cholesky(self.col_covar)
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
        c = _matrix_normal_event(x, self.n, self.p) - self.mean
        quad = np.trace(self._u_inv @ c @ self._v_inv @ c.T)
        return float(self._log_norm - 0.5 * quad)

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Vectorized log-density for a stack of matrices, shape ``(N, n, p)``."""
        c = _matrix_normal_batch(x, self.n, self.p) - self.mean
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
        return MatrixNormalDataEncoder(self.n, self.p)


class MatrixNormalSampler(DistributionSampler):
    """Draw matrices by ``X = M + chol(U) Z chol(V)^T`` with ``Z`` standard normal."""

    def __init__(self, dist: MatrixNormalDistribution, seed: int | None = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: int | None = None, *, batched: bool = True) -> np.ndarray:
        """Draw one matrix or a batch of independent matrix-normal samples."""
        d = self.dist
        sample_count = 1 if size is None else _validated_sample_size(size)
        z = self.rng.randn(sample_count, d.n, d.p)
        x = d.mean[None, :, :] + np.einsum("ab,nbc,dc->nad", d._chol_u, z, d._chol_v, optimize=True)
        return x[0] if size is None else x


class MatrixNormalAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate the mean and the row-blocked second moment ``T[a,b,c,d] = sum_i X_i[a,c] X_i[b,d]``."""

    def __init__(self, n: int, p: int, name: str | None = None, keys: str | None = None) -> None:
        self.n = _validated_dimension(n, "matrix-normal row dimension")
        self.p = _validated_dimension(p, "matrix-normal column dimension")
        self.sum_x = np.zeros((self.n, self.p), dtype=np.float64)
        self.t = np.zeros(
            (self.n, self.n, self.p, self.p),
            dtype=np.float64,
        )
        self.count = 0.0
        self.name = name
        self.keys = keys

    def update(self, x: np.ndarray, weight: float, estimate: MatrixNormalDistribution | None) -> None:
        """Accumulate weighted mean and row-blocked second moments for one matrix."""
        xx = _matrix_normal_event(x, self.n, self.p)
        checked_weight = _validated_weight(weight)
        self.sum_x += checked_weight * xx
        self.t += checked_weight * np.einsum(
            "ac,bd->abcd",
            xx,
            xx,
            optimize=True,
        )
        self.count += checked_weight

    def initialize(self, x: np.ndarray, weight: float, rng: RandomState | None) -> None:
        """Initialize the sufficient statistics with one weighted matrix."""
        self.update(x, weight, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: MatrixNormalDistribution | None) -> None:
        """Accumulate weighted mean and block moments for encoded matrices."""
        xx = _matrix_normal_batch(x, self.n, self.p)
        w = _validated_weights(weights, len(xx))
        xw = xx * w[:, None, None]
        self.sum_x += xw.sum(axis=0)
        self.t += np.einsum("iac,ibd->abcd", xw, xx, optimize=True)
        self.count += float(w.sum())

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize the sufficient statistics from encoded matrices."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[np.ndarray, np.ndarray, float]) -> "MatrixNormalAccumulator":
        """Merge serialized matrix-normal sufficient statistics into this accumulator."""
        sum_x, moment, count = _validated_matrix_normal_statistics(
            suff_stat,
            self.n,
            self.p,
        )
        self.sum_x += sum_x
        self.t += moment
        self.count += count
        return self

    def value(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Return the weighted sum, block second moment, and total weight."""
        return self.sum_x.copy(), self.t.copy(), self.count

    def from_value(self, x: tuple[np.ndarray, np.ndarray, float]) -> "MatrixNormalAccumulator":
        """Restore the accumulator from serialized matrix-normal statistics."""
        sum_x, moment, count = _validated_matrix_normal_statistics(
            x,
            self.n,
            self.p,
        )
        self.sum_x = sum_x
        self.t = moment
        self.count = count
        return self

    def scale(self, c: float) -> "MatrixNormalAccumulator":
        """Scale all linear matrix-normal sufficient statistics."""
        checked_scale = _validated_weight(c)
        self.sum_x *= checked_scale
        self.t *= checked_scale
        self.count *= checked_scale
        return self

    def acc_to_encoder(self) -> "MatrixNormalDataEncoder":
        """Return an encoder compatible with matrix-normal vectorized updates."""
        return MatrixNormalDataEncoder(self.n, self.p)


class MatrixNormalAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for MatrixNormalAccumulator."""

    def __init__(self, n: int, p: int, name: str | None = None, keys: str | None = None) -> None:
        self.n = _validated_dimension(n, "matrix-normal row dimension")
        self.p = _validated_dimension(p, "matrix-normal column dimension")
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
        self.n = _validated_dimension(n, "matrix-normal row dimension")
        self.p = _validated_dimension(p, "matrix-normal column dimension")
        self.max_iter = _validated_dimension(max_iter, "matrix-normal max_iter")
        if isinstance(tol, (bool, np.bool_)) or np.ndim(tol) != 0:
            raise TypeError("matrix-normal tolerance must be a finite positive scalar")
        try:
            self.tol = float(tol)
        except (TypeError, ValueError) as exc:
            raise TypeError("matrix-normal tolerance must be a finite positive scalar") from exc
        if not np.isfinite(self.tol) or self.tol <= 0.0:
            raise ValueError("matrix-normal tolerance must be a finite positive scalar")
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> MatrixNormalAccumulatorFactory:
        """Return a factory for matrix-normal sufficient-statistic accumulators."""
        return MatrixNormalAccumulatorFactory(self.n, self.p, name=self.name, keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat: tuple[np.ndarray, np.ndarray, float]) -> MatrixNormalDistribution:
        """Estimate the mean and covariance factors using flip-flop updates."""
        sum_x, t, count = _validated_matrix_normal_statistics(
            suff_stat,
            self.n,
            self.p,
        )
        n, p = self.n, self.p
        if count == 0.0:
            raise MatrixNormalFitError("matrix-normal fitting requires positive observation weight")
        mean = sum_x / count
        # centered row-blocked second moment: T_c[a,b,c,d] = sum_i (X_i-M)[a,c] (X_i-M)[b,d]
        tc = t - count * np.einsum("ac,bd->abcd", mean, mean, optimize=True)
        u = np.eye(n)
        v = np.eye(p)
        converged = False
        delta = np.inf
        for iteration in range(1, self.max_iter + 1):
            v_inv = np.linalg.inv(v)
            u_new = np.einsum("abcd,cd->ab", tc, v_inv, optimize=True) / (count * p)
            u_new = 0.5 * (u_new + u_new.T)
            if np.any(~np.isfinite(u_new)) or cholesky_logdet(u_new) is None:
                raise MatrixNormalFitError(
                    "matrix-normal row covariance is non-identifiable at flip-flop iteration %d" % iteration
                )
            u_inv = np.linalg.inv(u_new)
            v_new = np.einsum("abcd,ab->cd", tc, u_inv, optimize=True) / (count * n)
            v_new = 0.5 * (v_new + v_new.T)
            scale = v_new[0, 0]  # anchor V[0,0]=1 to fix the U<->V scale ambiguity
            if not np.isfinite(scale) or scale <= 0.0:
                raise MatrixNormalFitError(
                    "matrix-normal scale anchor is invalid at flip-flop iteration %d" % iteration
                )
            v_new = v_new / scale
            u_new = u_new * scale
            if np.any(~np.isfinite(v_new)) or cholesky_logdet(v_new) is None or cholesky_logdet(u_new) is None:
                raise MatrixNormalFitError(
                    "matrix-normal covariance factors are invalid at flip-flop iteration %d" % iteration
                )
            delta = max(
                float(np.max(np.abs(u_new - u))),
                float(np.max(np.abs(v_new - v))),
            )
            if delta < self.tol:
                u, v = u_new, v_new
                converged = True
                break
            u, v = u_new, v_new
        if not converged:
            raise MatrixNormalFitError(
                "matrix-normal flip-flop fit did not converge in %d iterations "
                "(last parameter delta %.6g)" % (self.max_iter, delta)
            )
        result = MatrixNormalDistribution(
            mean,
            u,
            v,
            name=self.name,
            keys=self.keys,
        )
        result.fit_metadata = {
            "converged": True,
            "solver": "flip-flop",
            "iterations": iteration,
            "parameter_delta": delta,
            "tolerance": self.tol,
            "repairs": (),
        }
        return result


class MatrixNormalDataEncoder(DataSequenceEncoder):
    """Encode a sequence of ``(n, p)`` matrices as an ``(N, n, p)`` float array."""

    def __init__(self, n: int, p: int) -> None:
        self.n = _validated_dimension(n, "matrix-normal encoder row dimension")
        self.p = _validated_dimension(p, "matrix-normal encoder column dimension")

    def __str__(self) -> str:
        return "MatrixNormalDataEncoder(%d, %d)" % (self.n, self.p)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, MatrixNormalDataEncoder) and self.n == other.n and self.p == other.p

    def seq_encode(self, x: Sequence[np.ndarray]) -> np.ndarray:
        """Encode matrices as a floating-point stack for vectorized evaluation."""
        return _matrix_normal_batch(x, self.n, self.p).copy()

    def row_count(self, x: np.ndarray) -> int:
        """Return the number of matrices after validating encoded geometry."""
        return len(_matrix_normal_batch(x, self.n, self.p))
