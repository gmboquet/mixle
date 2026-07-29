"""Wishart distribution -- a distribution over symmetric positive-definite ``p``-by-``p`` matrices.

The Wishart is the distribution of a scatter matrix ``X = sum_{i=1}^{df} z_i z_i^T`` with
``z_i ~ N(0, scale)``; it is the matrix generalisation of the chi-square / gamma and the standard model
for random covariance matrices (and the conjugate prior for a Gaussian precision). With ``df >= p``
degrees of freedom and scale matrix ``V``,

    log f(X) = (df-p-1)/2 log|X| - 1/2 tr(V^{-1} X) - df p/2 log 2 - df/2 log|V| - log Gamma_p(df/2),

where ``Gamma_p`` is the multivariate gamma. Since ``E[X] = df V`` the scale ``V`` is estimated in closed
form as the mean scatter divided by ``df``. The degrees of freedom may be supplied (``WishartEstimator(dim,
df=value)``) or estimated by maximum likelihood (``WishartEstimator(dim, df=None)``) -- the latter adds the
``sum log det(X)`` sufficient statistic and solves the profile-likelihood score
for ``df`` with a certified bracketed bisection.


Reference: Wishart, 'The generalised product moment distribution in samples...', Biometrika (1928).
"""

import math
import operator
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import RandomState
from scipy.special import digamma, multigammaln

from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)
from mixle.utils.vector import batched_pd_logdet, cholesky_logdet


class WishartFitError(RuntimeError):
    """Raised when a Wishart fit has no valid, certified optimum."""


def _validated_dimension(value: Any, name: str = "dimension") -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("%s must be a positive integer" % name)
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError("%s must be a positive integer" % name) from exc
    if result <= 0:
        raise ValueError("%s must be a positive integer" % name)
    return result


def _validated_sample_size(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("sample size must be a non-negative integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError("sample size must be a non-negative integer") from exc
    if result < 0:
        raise ValueError("sample size must be non-negative")
    return result


def _validated_weight(value: Any) -> float:
    if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0:
        raise TypeError("matrix observation weight must be a real scalar")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError("matrix observation weight must be a real scalar") from exc
    if not np.isfinite(result) or result < 0.0:
        raise ValueError("matrix observation weight must be finite and non-negative")
    return result


def _validated_df(value: Any, dim: int, name: str = "df") -> float:
    if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0:
        raise TypeError("%s must be a real scalar" % name)
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError("%s must be a real scalar" % name) from exc
    if not np.isfinite(result) or result < dim:
        raise ValueError("%s must be a finite value >= the matrix dimension p" % name)
    return result


def _validated_weights(value: Any, rows: int) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("matrix observation weights must be numeric") from exc
    if result.shape != (rows,):
        raise ValueError("matrix observation weights must have exact shape (%d,)" % rows)
    if np.any(~np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError("matrix observation weights must be finite and non-negative")
    return result.copy()


def _matrix_event(value: Any, dim: int, name: str = "matrix observation") -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("%s must be numeric" % name) from exc
    if result.shape != (dim, dim):
        raise ValueError("%s must have exact shape (%d, %d)" % (name, dim, dim))
    return result


def _matrix_batch(value: Any, dim: int, name: str = "matrix observations") -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("%s must be numeric" % name) from exc
    if result.shape == (0,):
        return np.empty((0, dim, dim), dtype=np.float64)
    if result.ndim != 3 or result.shape[1:] != (dim, dim):
        raise ValueError("%s must have exact shape (N, %d, %d)" % (name, dim, dim))
    return result


def _is_symmetric(x: np.ndarray) -> bool:
    """True iff ``x`` is a symmetric matrix.

    A Wishart/inverse-Wishart random variable is by definition a symmetric positive-definite
    matrix, so an asymmetric observation is not a member of the support at all.
    ``batched_pd_logdet`` (like ``np.linalg.eigvalsh``, which it wraps) reads one triangle only
    (``UPLO='L'`` by default) and never inspects the other, so an asymmetric matrix with a
    positive-definite-looking triangle would otherwise pass straight through
    ``log_density``/``seq_log_density`` and receive a finite score instead of being rejected.
    """
    return bool(np.array_equal(x, x.T))


def _batched_is_symmetric(x: np.ndarray) -> np.ndarray:
    """Vectorized symmetry check for a stack of matrices, shape ``(..., p, p)`` -- see
    :func:`_is_symmetric`. Folded into the same validity mask as the positive-definite check in
    ``seq_log_density`` (rather than raising), matching the pattern already used for correlation
    matrices in ``mixle.stats.matrix.lkj._batched_is_corr_like``: a structurally invalid row in a
    batch scores ``-inf`` without disturbing the rest.
    """
    return np.all(x == np.swapaxes(x, -1, -2), axis=(-1, -2))


def _support_mask_and_logdet(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return exact SPD support and safe log determinants for a matrix batch."""
    finite = np.all(np.isfinite(value), axis=(-1, -2))
    symmetric = _batched_is_symmetric(value)
    preliminary = finite & symmetric
    safe = np.where(preliminary[:, None, None], value, np.eye(value.shape[-1]))
    with np.errstate(divide="ignore", invalid="ignore"):
        is_pd, logdet = batched_pd_logdet(safe)
    valid = preliminary & is_pd
    return valid, np.where(valid, logdet, -np.inf)


def _validated_spd_matrix(
    value: Any,
    dim: int,
    name: str = "matrix observation",
) -> tuple[np.ndarray, float]:
    result = _matrix_event(value, dim, name)
    if np.any(~np.isfinite(result)):
        raise ValueError("%s must be finite" % name)
    if not _is_symmetric(result):
        raise ValueError("%s must be exactly symmetric" % name)
    logdet = cholesky_logdet(result)
    if logdet is None:
        raise ValueError("%s must be positive definite" % name)
    return result.copy(), float(logdet)


def _validated_spd_batch(
    value: Any,
    dim: int,
    name: str = "matrix observations",
) -> tuple[np.ndarray, np.ndarray]:
    result = _matrix_batch(value, dim, name)
    valid, logdet = _support_mask_and_logdet(result)
    if not np.all(valid):
        raise ValueError("%s must contain only finite, exactly symmetric positive-definite matrices" % name)
    return result.copy(), logdet


def _validated_mean_scatter_statistics(
    value: Any,
    dim: int,
) -> tuple[np.ndarray, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError("matrix sufficient statistics must be a two-item tuple")
    if isinstance(value[1], (bool, np.bool_)) or np.ndim(value[1]) != 0:
        raise TypeError("matrix sufficient-statistic count must be a real scalar")
    try:
        scatter = np.asarray(value[0], dtype=np.float64)
        count = float(value[1])
    except (TypeError, ValueError) as exc:
        raise ValueError("matrix sufficient statistics must be numeric") from exc
    if scatter.shape != (dim, dim) or np.any(~np.isfinite(scatter)):
        raise ValueError("matrix sufficient-statistic scatter must be finite with exact shape (%d, %d)" % (dim, dim))
    if not _is_symmetric(scatter):
        raise ValueError("matrix sufficient-statistic scatter must be exactly symmetric")
    if not np.isfinite(count) or count < 0.0:
        raise ValueError("matrix sufficient-statistic count must be finite and non-negative")
    if count == 0.0:
        if np.any(scatter != 0.0):
            raise ValueError("empty matrix sufficient statistics must have zero scatter")
    elif cholesky_logdet(scatter) is None:
        raise ValueError("non-empty matrix sufficient-statistic scatter must be positive definite")
    return scatter.copy(), count


def _validated_wishart_statistics(
    value: Any,
    dim: int,
) -> tuple[np.ndarray, float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError("Wishart sufficient statistics must be a three-item tuple")
    scatter, count = _validated_mean_scatter_statistics(value[:2], dim)
    try:
        sum_logdet = float(value[2])
    except (TypeError, ValueError) as exc:
        raise ValueError("Wishart log-determinant statistic must be numeric") from exc
    if not np.isfinite(sum_logdet):
        raise ValueError("Wishart log-determinant statistic must be finite")
    if count == 0.0 and sum_logdet != 0.0:
        raise ValueError("empty Wishart sufficient statistics must have zero log determinant")
    return scatter, count, sum_logdet


class WishartDistribution(SequenceEncodableProbabilityDistribution):
    """Wishart distribution with ``df`` degrees of freedom and scale matrix ``scale`` (p, p)."""

    def __init__(self, df: float, scale: np.ndarray, name: str | None = None, keys: str | None = None) -> None:
        v = np.asarray(scale, dtype=np.float64)
        if v.ndim != 2 or v.shape[0] != v.shape[1] or v.shape[0] == 0:
            raise ValueError("scale must be a non-empty square matrix")
        if np.any(~np.isfinite(v)):
            raise ValueError("scale must be finite")
        if not _is_symmetric(v):
            raise ValueError("scale must be exactly symmetric")
        self.dim = _validated_dimension(v.shape[0], "matrix dimension")
        checked_df = _validated_df(df, self.dim)
        logdet = cholesky_logdet(v)
        if logdet is None:
            raise ValueError("scale must be positive definite")
        self.df = checked_df
        self.scale = v.copy()
        self.scale.setflags(write=False)
        self.name = name
        self.keys = keys
        self._scale_inv = np.linalg.inv(self.scale)
        self._chol = np.linalg.cholesky(self.scale)
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
        """Return the log-density at a single ``(p, p)`` SPD matrix (``-inf`` if not symmetric or
        not positive definite).

        Mirrors :meth:`seq_log_density`'s routines exactly rather than merely approximately: uses
        :func:`batched_pd_logdet` (the same eigenvalue-based check) instead of :func:`cholesky_logdet`
        for positive-definiteness/log-determinant, and the same ``einsum`` trace contraction instead of
        ``trace(A @ B)``. Near the positive-definiteness boundary a Cholesky factorization and an
        eigendecomposition can round differently and even disagree on whether a matrix is PD at all, and
        a full matrix product summed via ``trace`` accumulates in a different order than the direct
        ``einsum`` contraction -- both previously made this scalar path and the vectorized path diverge
        (in value, and occasionally in support) on the same input.
        """
        xx = _matrix_event(x, self.dim)
        if np.any(~np.isfinite(xx)) or not _is_symmetric(xx):
            return -np.inf
        with np.errstate(divide="ignore", invalid="ignore"):
            is_pd, logdet = batched_pd_logdet(xx)
        if not is_pd:
            return -np.inf
        tr = np.einsum("ab,ba->", self._scale_inv, xx, optimize=True)
        return float(self._log_norm + (self.df - self.dim - 1.0) / 2.0 * logdet - 0.5 * tr)

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Vectorized log-density for a stack of SPD matrices, shape ``(N, p, p)`` (``-inf`` per
        row that is not symmetric or not positive definite)."""
        xx = _matrix_batch(x, self.dim)
        is_valid, logdet = _support_mask_and_logdet(xx)
        safe_xx = np.where(is_valid[:, None, None], xx, np.eye(self.dim))
        tr = np.einsum("ab,nba->n", self._scale_inv, safe_xx, optimize=True)
        rv = self._log_norm + (self.df - self.dim - 1.0) / 2.0 * logdet - 0.5 * tr
        return np.where(is_valid, rv, -np.inf)

    def sampler(self, seed: int | None = None) -> "WishartSampler":
        """Return a sampler for drawing SPD matrices from this distribution."""
        return WishartSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "WishartEstimator":
        """Return a closed-form estimator for the scale at the fixed degrees of freedom ``df``."""
        return WishartEstimator(self.dim, self.df, name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> "WishartDataEncoder":
        """Return the data encoder used by this distribution for vectorized methods."""
        return WishartDataEncoder(self.dim)


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

    def sample(self, size: int | None = None, *, batched: bool = True) -> np.ndarray:
        """Draw one SPD matrix or a stacked batch of independent Wishart samples."""
        if size is None:
            return self._one()
        checked_size = _validated_sample_size(size)
        if checked_size == 0:
            return np.empty((0, self.dist.dim, self.dist.dim))
        return np.stack([self._one() for _ in range(checked_size)])


class _MeanScatterAccumulator(SequenceEncodableStatisticAccumulator):
    """Shared accumulator for (inverse-)Wishart: weighted matrix sum ``sum_i w_i X_i`` and total weight.

    Subclasses override :meth:`acc_to_encoder` to return the matching :class:`DataSequenceEncoder`;
    the weighted sufficient-statistic update path is shared.
    """

    def __init__(self, dim: int, name: str | None = None, keys: str | None = None) -> None:
        self.dim = _validated_dimension(dim, "matrix dimension")
        self.sum_x = np.zeros((self.dim, self.dim), dtype=np.float64)
        self.count = 0.0
        self.name = name
        self.keys = keys

    def update(self, x: np.ndarray, weight: float, estimate: Any | None) -> None:
        xx, _ = _validated_spd_matrix(x, self.dim)
        checked_weight = _validated_weight(weight)
        self.sum_x += checked_weight * xx
        self.count += checked_weight

    def initialize(self, x: np.ndarray, weight: float, rng: RandomState | None) -> None:
        self.update(x, weight, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: Any | None) -> None:
        xx, _ = _validated_spd_batch(x, self.dim)
        w = _validated_weights(weights, len(xx))
        self.sum_x += np.einsum("n,nab->ab", w, xx, optimize=True)
        self.count += float(w.sum())

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[np.ndarray, float]) -> "_MeanScatterAccumulator":
        if not isinstance(suff_stat, (tuple, list)) or len(suff_stat) != 2:
            raise ValueError("matrix sufficient statistics must be a two-item tuple")
        sum_x, count = _validated_mean_scatter_statistics(suff_stat, self.dim)
        self.sum_x += sum_x
        self.count += count
        return self

    def value(self) -> tuple[np.ndarray, float]:
        return self.sum_x.copy(), self.count

    def from_value(self, x: tuple[np.ndarray, float]) -> "_MeanScatterAccumulator":
        if not isinstance(x, (tuple, list)) or len(x) != 2:
            raise ValueError("matrix sufficient statistics must be a two-item tuple")
        sum_x, count = _validated_mean_scatter_statistics(x, self.dim)
        self.sum_x = sum_x
        self.count = count
        return self

    def scale(self, c: float) -> "_MeanScatterAccumulator":
        """Scale linear matrix sufficient statistics by a non-negative factor."""
        checked_scale = _validated_weight(c)
        self.sum_x *= checked_scale
        self.count *= checked_scale
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

    def update(self, x: np.ndarray, weight: float, estimate: Any | None) -> None:
        """Accumulate matrix scatter and log-determinant statistics for one observation."""
        xx, logdet = _validated_spd_matrix(x, self.dim)
        checked_weight = _validated_weight(weight)
        self.sum_x += checked_weight * xx
        self.count += checked_weight
        self.sum_logdet += checked_weight * logdet

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: Any | None) -> None:
        """Accumulate matrix scatter and log-determinants from encoded observations."""
        xx, logdet = _validated_spd_batch(x, self.dim)
        w = _validated_weights(weights, len(xx))
        self.sum_x += np.einsum("n,nab->ab", w, xx, optimize=True)
        self.count += float(w.sum())
        self.sum_logdet += float(np.dot(w, logdet))

    def combine(self, suff_stat: tuple[np.ndarray, float, float]) -> "WishartAccumulator":
        """Merge serialized Wishart sufficient statistics into this accumulator."""
        sum_x, count, sum_logdet = _validated_wishart_statistics(
            suff_stat,
            self.dim,
        )
        self.sum_x += sum_x
        self.count += count
        self.sum_logdet += sum_logdet
        return self

    def value(self) -> tuple[np.ndarray, float, float]:
        """Return scatter, total weight, and weighted log-determinant sum."""
        return self.sum_x.copy(), self.count, self.sum_logdet

    def from_value(self, x: tuple[np.ndarray, float, float]) -> "WishartAccumulator":
        """Restore the accumulator from serialized Wishart sufficient statistics."""
        sum_x, count, sum_logdet = _validated_wishart_statistics(x, self.dim)
        self.sum_x = sum_x
        self.count = count
        self.sum_logdet = sum_logdet
        return self

    def scale(self, c: float) -> "WishartAccumulator":
        """Scale accumulated Wishart sufficient statistics by a constant."""
        checked_scale = _validated_weight(c)
        self.sum_x *= checked_scale
        self.count *= checked_scale
        self.sum_logdet *= checked_scale
        return self

    def acc_to_encoder(self) -> "WishartDataEncoder":
        """Return an encoder for SPD matrix observations."""
        return WishartDataEncoder(self.dim)


class WishartAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for WishartAccumulator."""

    def __init__(self, dim: int, name: str | None = None, keys: str | None = None) -> None:
        self.dim = _validated_dimension(dim, "Wishart dimension")
        self.name = name
        self.keys = keys

    def make(self) -> WishartAccumulator:
        """Create an empty Wishart accumulator."""
        return WishartAccumulator(self.dim, name=self.name, keys=self.keys)


def _wishart_profile_score(
    df: float,
    mean_logdet: float,
    logdet_scatter: float,
    dim: int,
) -> float:
    j = np.arange(1, dim + 1)
    return float(
        -dim / 2.0 * math.log(2.0)
        - logdet_scatter / 2.0
        + dim / 2.0 * math.log(df)
        - 0.5 * np.sum(digamma((df + 1.0 - j) / 2.0))
        + mean_logdet / 2.0
    )


def _solve_wishart_df(
    mean_logdet: float,
    logdet_scatter: float,
    dim: int,
    df0: float | None = None,
) -> tuple[float, dict[str, Any]]:
    """Certified constrained MLE of the Wishart degrees of freedom.

    With the scale profiled out (``V = mean(X)/df``), the profile score in ``df`` is
    ``g(df) = -p/2 log2 - logdet(S)/2 + p/2 log df - 1/2 sum_j psi((df+1-j)/2) + mean_logdet/2`` where
    ``S = mean(X)`` and ``mean_logdet = mean(log det X)``. The public
    distribution uses the nonsingular ``df >= p`` domain. A monotone score
    bracket is expanded and bisected; a lower-bound optimum is accepted only
    with the corresponding gradient sign.
    """
    p = _validated_dimension(dim, "Wishart dimension")
    if not np.isfinite(mean_logdet) or not np.isfinite(logdet_scatter):
        raise WishartFitError("Wishart degree-of-freedom fit requires finite determinant statistics.")
    if mean_logdet > logdet_scatter + 1.0e-10 * max(
        1.0,
        abs(logdet_scatter),
    ):
        raise WishartFitError("Wishart determinant statistics violate log-determinant concavity.")
    if df0 is not None:
        _validated_df(df0, p, "initial df")

    lower = float(p)
    lower_score = _wishart_profile_score(
        lower,
        mean_logdet,
        logdet_scatter,
        p,
    )
    if not np.isfinite(lower_score):
        raise WishartFitError("Wishart profile score is non-finite at its lower bound.")
    if lower_score <= 0.0:
        return lower, {
            "converged": True,
            "boundary": "lower",
            "iterations": 0,
            "score": lower_score,
            "bracket": (lower, lower),
        }

    upper = max(float(p + 1), float(df0) if df0 is not None else 0.0)
    upper_score = _wishart_profile_score(
        upper,
        mean_logdet,
        logdet_scatter,
        p,
    )
    bracket_iterations = 0
    while np.isfinite(upper_score) and upper_score > 0.0 and upper < 1.0e12:
        upper *= 2.0
        upper_score = _wishart_profile_score(
            upper,
            mean_logdet,
            logdet_scatter,
            p,
        )
        bracket_iterations += 1
    if not np.isfinite(upper_score):
        raise WishartFitError("Wishart profile score became non-finite while bracketing.")
    if upper_score > 0.0:
        raise WishartFitError("Wishart profile likelihood has no certified finite degrees-of-freedom optimum.")

    left, right = lower, upper
    score = lower_score
    for iteration in range(1, 201):
        midpoint = 0.5 * (left + right)
        score = _wishart_profile_score(
            midpoint,
            mean_logdet,
            logdet_scatter,
            p,
        )
        if not np.isfinite(score):
            raise WishartFitError("Wishart profile score became non-finite during bisection.")
        if abs(score) <= 1.0e-10 or right - left <= 1.0e-10 * max(
            1.0,
            midpoint,
        ):
            return midpoint, {
                "converged": True,
                "boundary": None,
                "iterations": bracket_iterations + iteration,
                "score": score,
                "bracket": (left, right),
            }
        if score > 0.0:
            left = midpoint
        else:
            right = midpoint
    raise WishartFitError("Wishart degrees-of-freedom bisection did not converge.")


class WishartEstimator(ParameterEstimator):
    """Closed-form scale estimator (``V = mean(X)/df``); ``df=None`` also fits the degrees of freedom by MLE.

    With a fixed ``df`` the estimator returns only the closed-form scale (``E[X] = df V``). With ``df=None``
    it additionally estimates the degrees of freedom from
    ``sum_i w_i log det(X_i)`` with a bracketed, certified root of the profile
    score (:func:`_solve_wishart_df`).
    """

    def __init__(self, dim: int, df: float | None = None, name: str | None = None, keys: str | None = None) -> None:
        self.dim = _validated_dimension(dim, "Wishart dimension")
        self.df = None if df is None else _validated_df(df, self.dim)
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> WishartAccumulatorFactory:
        """Return a factory for Wishart sufficient-statistic accumulators."""
        return WishartAccumulatorFactory(self.dim, name=self.name, keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat: tuple[np.ndarray, float, float]) -> WishartDistribution:
        """Estimate the Wishart scale and optionally degrees of freedom."""
        sum_x, count, sum_logdet = _validated_wishart_statistics(
            suff_stat,
            self.dim,
        )
        if count == 0.0:
            raise WishartFitError("Wishart fitting requires positive observation weight.")
        scatter = sum_x / count  # E[X] = df V, so mean(X) = df V
        _, logdet_scatter = _validated_spd_matrix(
            scatter,
            self.dim,
            "Wishart mean scatter",
        )
        if self.df is not None:
            df = self.df
            diagnostics = {
                "converged": True,
                "boundary": None,
                "iterations": 0,
                "score": None,
                "bracket": None,
            }
            solver = "fixed"
        else:
            df, diagnostics = _solve_wishart_df(
                sum_logdet / count,
                logdet_scatter,
                self.dim,
                df0=self.dim + 1.0,
            )
            solver = "bracketed-bisection"
        scale = scatter / df
        result = WishartDistribution(
            df,
            scale,
            name=self.name,
            keys=self.keys,
        )
        result.fit_metadata = {
            **diagnostics,
            "solver": solver,
            "repairs": (),
        }
        return result


class WishartDataEncoder(DataSequenceEncoder):
    """Encode a sequence of ``(p, p)`` matrices as an ``(N, p, p)`` float array."""

    def __init__(self, dim: int) -> None:
        self.dim = _validated_dimension(dim, "Wishart encoder dimension")

    def __str__(self) -> str:
        return "WishartDataEncoder(%d)" % self.dim

    def __eq__(self, other: object) -> bool:
        return isinstance(other, WishartDataEncoder) and self.dim == other.dim

    def seq_encode(self, x: Sequence[np.ndarray]) -> np.ndarray:
        """Encode SPD matrix observations as a floating-point stack."""
        encoded, _ = _validated_spd_batch(x, self.dim)
        return encoded

    def row_count(self, x: np.ndarray) -> int:
        """Return the number of validated encoded SPD matrices."""
        encoded, _ = _validated_spd_batch(x, self.dim)
        return len(encoded)
