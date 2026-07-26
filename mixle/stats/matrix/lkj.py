"""LKJ distribution over correlation matrices.

The Lewandowski-Kurowicka-Joe (LKJ) law places density ``f(R) = c_d(eta) * det(R)^(eta-1)`` on ``d x d``
correlation matrices (symmetric, unit diagonal, positive definite). The concentration ``eta > 0`` tilts
mass toward the identity: ``eta = 1`` is uniform over correlation matrices, ``eta > 1`` favours weak
correlations (``R`` near ``I``), and ``eta < 1`` favours strong ones. It is the standard prior on
correlation matrices in hierarchical Bayesian models (Stan's default), separating a covariance into
scales times a correlation.

Normalizer (C-vine derivation, verified to high precision against arbitrary-precision integration over
the correlation elliptope for ``d = 2, 3``):

    Z_d(eta) = prod_{k=1}^{d-1} B(eta + (d-1-k)/2, 1/2)^(d-k),   c_d(eta) = 1 / Z_d(eta).

It samples exactly by the onion method (Lewandowski et al. 2009, sec. 3.2): each off-diagonal entry then
has the exact marginal ``(r + 1)/2 ~ Beta(eta + (d-2)/2, eta + (d-2)/2)``. Because the density depends on
``R`` only through ``det(R)``, observations are encoded as ``log det(R)``, and ``eta`` is fit by maximum
likelihood (a 1-D root find: the mean log-determinant equals ``sum_k (d-k)[psi(eta+e_k) -
psi(eta+e_k+1/2)]`` with ``e_k = (d-1-k)/2``).

Reference: Lewandowski, Kurowicka & Joe, "Generating random correlation matrices based on vines and
extended onion method", *J. Multivariate Analysis* 100 (2009).
"""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import RandomState
from scipy.special import gammaln

from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)
from mixle.stats.matrix.wishart import (
    _matrix_batch,
    _matrix_event,
    _support_mask_and_logdet,
    _validated_dimension,
    _validated_sample_size,
    _validated_weight,
    _validated_weights,
)
from mixle.utils.vector import cholesky_logdet

_LOG_SQRT_PI = 0.5 * math.log(math.pi)


class LKJFitError(RuntimeError):
    """Raised when an LKJ concentration fit has no certified solution."""


def _is_corr_like(r: np.ndarray, dim: int | None = None) -> bool:
    """True iff r is a square, symmetric, unit-diagonal matrix (a correlation matrix modulo
    positive-definiteness, which is checked separately via cholesky_logdet since that call also
    yields the log-determinant callers need). If dim is given, r must also be exactly that size --
    callers that don't track a dimension (the accumulator and encoder, which only ever store
    (count, sum_log_det) / a reduced log-determinant) pass dim=None and skip that check.
    """
    if r.ndim != 2 or r.shape[0] != r.shape[1]:
        return False
    if dim is not None and r.shape[0] != dim:
        return False
    return bool(
        np.all(np.isfinite(r))
        and np.array_equal(r, r.T)
        and np.array_equal(np.diag(r), np.ones(r.shape[0]))
    )


def _correlation_batch(
    value: Any,
    dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return matrix data, exact support mask, and safe log determinants."""
    result = _matrix_batch(value, dim, "LKJ observations")
    spd, logdet = _support_mask_and_logdet(result)
    unit_diag = np.all(
        np.diagonal(result, axis1=-2, axis2=-1) == 1.0,
        axis=-1,
    )
    valid = spd & unit_diag & (logdet <= 1.0e-10)
    checked_logdet = np.where(valid, np.minimum(logdet, 0.0), -np.inf)
    return result, valid, checked_logdet


def _validated_correlation(
    value: Any,
    dim: int,
    name: str = "LKJ observation",
) -> tuple[np.ndarray, float]:
    result = _matrix_event(value, dim, name)
    if not _is_corr_like(result, dim):
        raise ValueError(
            "%s must be a finite, exactly symmetric matrix with unit diagonal"
            % name
        )
    logdet = cholesky_logdet(result)
    if logdet is None:
        raise ValueError("%s must be positive definite" % name)
    if logdet > 1.0e-10:
        raise ValueError("%s has an invalid positive log determinant" % name)
    return result.copy(), min(float(logdet), 0.0)


def _validated_encoded_lkj(
    value: Any,
    expected_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError(
            "LKJ encoded data must be (dimension, log determinants, validity mask)"
        )
    encoded_dim = _validated_dimension(value[0], "LKJ encoded dimension")
    if encoded_dim != expected_dim:
        raise ValueError(
            "LKJ encoded dimension %d does not match model dimension %d"
            % (encoded_dim, expected_dim)
        )
    try:
        logdet = np.asarray(value[1], dtype=np.float64)
        valid = np.asarray(value[2])
    except (TypeError, ValueError) as exc:
        raise ValueError("LKJ encoded data must be numeric") from exc
    if logdet.ndim != 1 or valid.shape != logdet.shape or valid.dtype != np.bool_:
        raise ValueError(
            "LKJ encoded log determinants and boolean validity mask must be aligned vectors"
        )
    if np.any(np.isnan(logdet)) or np.any(np.isposinf(logdet)):
        raise ValueError("LKJ encoded log determinants cannot contain NaN or +inf")
    if np.any(valid & ~np.isfinite(logdet)):
        raise ValueError("valid LKJ encoded rows require finite log determinants")
    if np.any(valid & (logdet > 1.0e-10)):
        raise ValueError("valid LKJ log determinants cannot exceed zero")
    if np.any(~valid & ~np.isneginf(logdet)):
        raise ValueError("invalid LKJ encoded rows must carry -inf log determinant")
    return logdet.copy(), valid.copy()


def _validated_lkj_statistics(value: Any) -> tuple[float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError("LKJ sufficient statistics must be a two-item tuple")
    if any(
        isinstance(item, (bool, np.bool_)) or np.ndim(item) != 0
        for item in value
    ):
        raise TypeError("LKJ sufficient statistics must be real scalars")
    try:
        count = float(value[0])
        sum_log_det = float(value[1])
    except (TypeError, ValueError) as exc:
        raise ValueError("LKJ sufficient statistics must be numeric") from exc
    if not np.isfinite(count) or count < 0.0:
        raise ValueError("LKJ count must be finite and non-negative")
    if not np.isfinite(sum_log_det):
        raise ValueError("LKJ log-determinant statistic must be finite")
    if sum_log_det > 0.0:
        raise ValueError("LKJ log-determinant statistic cannot exceed zero")
    if count == 0.0 and sum_log_det != 0.0:
        raise ValueError("empty LKJ sufficient statistics must have zero log determinant")
    return count, sum_log_det


def _log_beta_half(a: float) -> float:
    """``log B(a, 1/2) = lgamma(a) + lgamma(1/2) - lgamma(a + 1/2)`` (``lgamma(1/2) = log sqrt(pi)``)."""
    return float(gammaln(a) + _LOG_SQRT_PI - gammaln(a + 0.5))


def _log_normalizer(dim: int, eta: float) -> float:
    """``log c_d(eta) = -log Z_d(eta) = -sum_{k=1}^{d-1} (d-k) log B(eta + (d-1-k)/2, 1/2)``."""
    return -sum((dim - k) * _log_beta_half(eta + (dim - 1 - k) / 2.0) for k in range(1, dim))


class LKJDistribution(SequenceEncodableProbabilityDistribution):
    """LKJ distribution over ``dim x dim`` correlation matrices with concentration ``eta > 0``."""

    def __init__(self, dim: int, eta: float, name: str | None = None, keys: str | None = None) -> None:
        checked_dim = _validated_dimension(dim, "LKJ dimension")
        if checked_dim < 2:
            raise ValueError("LKJDistribution requires dim >= 2.")
        if isinstance(eta, (bool, np.bool_)) or np.ndim(eta) != 0:
            raise TypeError("LKJDistribution requires scalar eta.")
        try:
            checked_eta = float(eta)
        except (TypeError, ValueError) as exc:
            raise TypeError("LKJDistribution requires scalar eta.") from exc
        if checked_eta <= 0.0 or not np.isfinite(checked_eta):
            raise ValueError("LKJDistribution requires finite eta > 0.")
        self.dim = checked_dim
        self.eta = checked_eta
        self.name = name
        self.keys = keys
        self._log_c = _log_normalizer(self.dim, self.eta)
        if not np.isfinite(self._log_c):
            raise ValueError("LKJ normalizer must be finite")

    def __str__(self) -> str:
        return "LKJDistribution(%s, %s, name=%s, keys=%s)" % (
            repr(self.dim),
            repr(self.eta),
            repr(self.name),
            repr(self.keys),
        )

    def density(self, x: Any) -> float:
        """Return the probability density at a correlation matrix ``x``."""
        return math.exp(self.log_density(x))

    def log_density(self, x: Any) -> float:
        """Return the log-density at a ``dim x dim`` correlation matrix (``-inf`` if ``x`` is the
        wrong size, not symmetric, does not have a unit diagonal, or is not positive definite)."""
        try:
            r = np.asarray(x, dtype=np.float64)
        except (TypeError, ValueError):
            return -math.inf
        if not _is_corr_like(r, self.dim):
            return -math.inf
        logdet = cholesky_logdet(r)
        if logdet is None:
            return -math.inf
        if logdet > 1.0e-10:
            return -math.inf
        logdet = min(float(logdet), 0.0)
        return self._log_c + (self.eta - 1.0) * logdet

    def seq_log_density(self, x: Any) -> np.ndarray:
        """Return vectorized log density for dimension-bound encoded LKJ data.

        Invalid rows remain ``-inf`` regardless of eta, including at eta=1
        where direct multiplication by their ``-inf`` sentinel would be NaN.
        """
        log_det, valid = _validated_encoded_lkj(x, self.dim)
        safe_log_det = np.where(valid, log_det, 0.0)
        rv = self._log_c + (self.eta - 1.0) * safe_log_det
        return np.where(valid, rv, -np.inf)

    def sampler(self, seed: int | None = None) -> "LKJSampler":
        """Return an onion-method sampler for correlation matrices."""
        return LKJSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "LKJEstimator":
        """Return a maximum-likelihood estimator for the concentration ``eta`` (``dim`` fixed)."""
        return LKJEstimator(self.dim, name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> "LKJDataEncoder":
        """Return the data encoder (a correlation matrix is encoded as its log-determinant)."""
        return LKJDataEncoder(self.dim)


class LKJSampler(DistributionSampler):
    """Sample correlation matrices by the onion method (Lewandowski-Kurowicka-Joe 2009)."""

    def __init__(self, dist: LKJDistribution, seed: int | None = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist

    def _batch(self, n: int) -> np.ndarray:
        """Sample ``n`` correlation matrices by the onion method, vectorized across samples (~30x faster)."""
        d = self.dist.dim
        eta = self.dist.eta
        beta = eta + (d - 2) / 2.0
        r = 2.0 * self.rng.beta(beta, beta, size=n) - 1.0
        corr = np.zeros((n, 2, 2))
        corr[:, 0, 0] = corr[:, 1, 1] = 1.0
        corr[:, 0, 1] = corr[:, 1, 0] = r
        for k in range(2, d):
            beta -= 0.5
            y = self.rng.beta(k / 2.0, beta, size=n)  # squared norm of the partial-correlation vectors
            u = self.rng.standard_normal((n, k))
            u /= np.linalg.norm(u, axis=1, keepdims=True)  # uniform directions on the (k-1)-sphere
            w = np.sqrt(y)[:, None] * u
            z = np.einsum("nij,nj->ni", np.linalg.cholesky(corr), w)  # batched Cholesky + matvec
            nxt = np.zeros((n, k + 1, k + 1))
            nxt[:, :k, :k] = corr
            nxt[:, :k, k] = z
            nxt[:, k, :k] = z
            nxt[:, k, k] = 1.0
            corr = nxt
        return corr

    def sample(self, size: int | None = None, *, batched: bool = True) -> Any:
        """Draw one correlation matrix or a list of independent correlation matrices."""
        if size is None:
            return self._batch(1)[0]
        return list(self._batch(_validated_sample_size(size)))


class LKJAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate ``(count, sum of log det(R))`` -- the sufficient statistics for the eta-MLE."""

    def __init__(
        self,
        dim: int,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        self.dim = _validated_dimension(dim, "LKJ accumulator dimension")
        if self.dim < 2:
            raise ValueError("LKJ accumulator dimension must be at least two")
        self.count = 0.0
        self.sum_log_det = 0.0
        self.name = name
        self.keys = keys

    def update(self, x: Any, weight: float, estimate: LKJDistribution | None) -> None:
        """Accumulate the weighted log determinant for one correlation matrix.

        Invalid events are rejected before either statistic is mutated.
        """
        _, logdet = _validated_correlation(x, self.dim)
        checked_weight = _validated_weight(weight)
        self.count += checked_weight
        self.sum_log_det += checked_weight * logdet

    def initialize(self, x: Any, weight: float, rng: RandomState | None) -> None:
        """Initialize the sufficient statistics with one weighted matrix."""
        self.update(x, weight, None)

    def seq_update(self, x: Any, weights: np.ndarray, estimate: Any) -> None:
        """Accumulate valid, dimension-bound encoded log determinants."""
        log_det, valid = _validated_encoded_lkj(x, self.dim)
        if not np.all(valid):
            raise ValueError("LKJ accumulation rejects invalid encoded observations")
        w = _validated_weights(weights, len(log_det))
        self.count += float(w.sum())
        self.sum_log_det += float(np.dot(w, log_det))

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize the sufficient statistics from encoded log determinants."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[float, float]) -> "LKJAccumulator":
        """Merge serialized LKJ sufficient statistics into this accumulator."""
        count, sum_log_det = _validated_lkj_statistics(suff_stat)
        self.count += count
        self.sum_log_det += sum_log_det
        return self

    def value(self) -> tuple[float, float]:
        """Return the total weight and weighted sum of log determinants."""
        return self.count, self.sum_log_det

    def from_value(self, x: tuple[float, float]) -> "LKJAccumulator":
        """Restore the accumulator from serialized LKJ sufficient statistics."""
        self.count, self.sum_log_det = _validated_lkj_statistics(x)
        return self

    def scale(self, c: float) -> "LKJAccumulator":
        """Scale LKJ sufficient statistics by a non-negative factor."""
        checked_scale = _validated_weight(c)
        self.count *= checked_scale
        self.sum_log_det *= checked_scale
        return self

    def acc_to_encoder(self) -> "LKJDataEncoder":
        """Return an encoder that reduces correlation matrices to log determinants."""
        return LKJDataEncoder(self.dim)


class LKJAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for LKJAccumulator."""

    def __init__(
        self,
        dim: int,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        self.dim = _validated_dimension(dim, "LKJ accumulator dimension")
        if self.dim < 2:
            raise ValueError("LKJ accumulator dimension must be at least two")
        self.name = name
        self.keys = keys

    def make(self) -> LKJAccumulator:
        """Create an empty LKJ accumulator."""
        return LKJAccumulator(self.dim, name=self.name, keys=self.keys)


class LKJEstimator(ParameterEstimator):
    """Maximum-likelihood estimator for the concentration ``eta`` at fixed dimension ``dim``."""

    def __init__(
        self,
        dim: int,
        eta_bounds: tuple[float, float] = (0.05, 1.0e4),
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        self.dim = _validated_dimension(dim, "LKJ estimator dimension")
        if self.dim < 2:
            raise ValueError("LKJ estimator dimension must be at least two")
        if not isinstance(eta_bounds, (tuple, list)) or len(eta_bounds) != 2:
            raise ValueError("LKJ eta_bounds must be a two-item sequence")
        try:
            lower, upper = (float(value) for value in eta_bounds)
        except (TypeError, ValueError) as exc:
            raise ValueError("LKJ eta_bounds must be numeric") from exc
        if (
            not np.isfinite(lower)
            or not np.isfinite(upper)
            or lower <= 0.0
            or lower >= upper
        ):
            raise ValueError(
                "LKJ eta_bounds must be finite, positive, and strictly ordered"
            )
        self.eta_bounds = (lower, upper)
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> LKJAccumulatorFactory:
        """Return a factory for LKJ sufficient-statistic accumulators."""
        return LKJAccumulatorFactory(self.dim, name=self.name, keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat: tuple[float, float]) -> LKJDistribution:
        """Estimate the LKJ concentration from the mean log determinant."""
        from scipy.optimize import brentq
        from scipy.special import digamma

        count, sum_log_det = _validated_lkj_statistics(suff_stat)
        if count == 0.0:
            raise LKJFitError("LKJ fitting requires positive observation weight")
        mean_log_det = sum_log_det / count
        d = self.dim

        def score(eta: float) -> float:
            # d/d eta of the log-likelihood per observation: mean_log_det - sum_k (d-k)[psi(.)-psi(.+1/2)]
            return mean_log_det - sum(
                (d - k) * (digamma(eta + (d - 1 - k) / 2.0) - digamma(eta + (d - 1 - k) / 2.0 + 0.5))
                for k in range(1, d)
            )

        lo, hi = self.eta_bounds
        lower_score = float(score(lo))
        upper_score = float(score(hi))
        if not np.isfinite(lower_score) or not np.isfinite(upper_score):
            raise LKJFitError("LKJ profile score is non-finite at its bounds")
        if lower_score <= 0.0:
            eta = lo
            boundary = "lower"
            iterations = 0
            residual = lower_score
        elif upper_score >= 0.0:
            eta = hi
            boundary = "upper"
            iterations = 0
            residual = upper_score
        else:
            try:
                eta, result = brentq(
                    score,
                    lo,
                    hi,
                    xtol=1.0e-10,
                    full_output=True,
                    disp=False,
                )
            except (ValueError, RuntimeError) as exc:
                raise LKJFitError("LKJ profile root solver failed") from exc
            if not result.converged:
                raise LKJFitError("LKJ profile root solver did not converge")
            eta = float(eta)
            boundary = None
            iterations = int(result.iterations)
            residual = float(score(eta))
            if not np.isfinite(residual) or abs(residual) > 1.0e-7:
                raise LKJFitError(
                    "LKJ profile root lacks a finite optimality certificate"
                )
        fitted = LKJDistribution(
            self.dim,
            eta,
            name=self.name,
            keys=self.keys,
        )
        fitted.fit_metadata = {
            "converged": True,
            "solver": "profile-boundary" if boundary is not None else "brentq",
            "boundary": boundary,
            "iterations": iterations,
            "score": residual,
            "bracket": self.eta_bounds,
            "repairs": (),
        }
        return fitted


class LKJDataEncoder(DataSequenceEncoder):
    """Encode each correlation matrix as its log-determinant (the only data the density depends on)."""

    def __init__(self, dim: int) -> None:
        self.dim = _validated_dimension(dim, "LKJ encoder dimension")
        if self.dim < 2:
            raise ValueError("LKJ encoder dimension must be at least two")

    def __str__(self) -> str:
        return "LKJDataEncoder(%d)" % self.dim

    def __eq__(self, other: object) -> bool:
        return isinstance(other, LKJDataEncoder) and self.dim == other.dim

    def seq_encode(self, x: Sequence[Any]) -> tuple[int, np.ndarray, np.ndarray]:
        """Encode dimension, log determinants, and a per-row support mask."""
        _xx, valid, logdet = _correlation_batch(x, self.dim)
        return self.dim, logdet, valid

    def row_count(self, x: Any) -> int:
        """Return the row count after validating dimension-bound encoded data."""
        logdet, _valid = _validated_encoded_lkj(x, self.dim)
        return len(logdet)
