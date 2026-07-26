"""Inverse-Wishart distribution -- a distribution over symmetric positive-definite matrices.

If ``X^{-1} ~ Wishart(df, scale^{-1})`` then ``X ~ InverseWishart(df, scale)``; it is the conjugate
prior for a multivariate-normal covariance and the standard model for a random covariance matrix
(rather than a random precision). With ``df > p - 1`` and scale matrix ``Psi``,

    log f(X) = df/2 log|Psi| - df p/2 log 2 - log Gamma_p(df/2)
               - (df+p+1)/2 log|X| - 1/2 tr(Psi X^{-1}).

``df`` is a fixed, known parameter. When ``df > p + 1``, the scale can be
estimated by the explicit method-of-moments identity
``Psi = (df - p - 1) * mean(X)``. This package labels that operation as a
mean-moment fit; it is not the inverse-Wishart likelihood MLE, and it is
rejected where the required mean does not exist.


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
from mixle.stats.matrix.wishart import (
    _is_symmetric,
    _matrix_batch,
    _matrix_event,
    _MeanScatterAccumulator,
    _support_mask_and_logdet,
    _validated_dimension,
    _validated_mean_scatter_statistics,
    _validated_sample_size,
    _validated_spd_batch,
)
from mixle.utils.vector import batched_pd_logdet, cholesky_logdet


class InverseWishartMomentFitError(RuntimeError):
    """Raised when the inverse-Wishart mean-moment fit is undefined."""


def _validated_inverse_wishart_df(value: object, dim: int) -> float:
    if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0:
        raise TypeError("df must be a real scalar")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError("df must be a real scalar") from exc
    if not np.isfinite(result) or result <= dim - 1:
        raise ValueError("df must be a finite value > p - 1")
    return result


class InverseWishartDistribution(SequenceEncodableProbabilityDistribution):
    """Inverse-Wishart distribution with ``df`` degrees of freedom and scale matrix ``scale`` (p, p)."""

    def __init__(self, df: float, scale: np.ndarray, name: str | None = None, keys: str | None = None) -> None:
        try:
            v = np.asarray(scale, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("scale must be numeric") from exc
        if v.ndim != 2 or v.shape[0] != v.shape[1] or v.shape[0] == 0:
            raise ValueError("scale must be a non-empty square matrix")
        if np.any(~np.isfinite(v)):
            raise ValueError("scale must be finite")
        if not _is_symmetric(v):
            raise ValueError("scale must be exactly symmetric")
        self.dim = _validated_dimension(v.shape[0], "matrix dimension")
        checked_df = _validated_inverse_wishart_df(df, self.dim)
        logdet = cholesky_logdet(v)
        if logdet is None:
            raise ValueError("scale must be positive definite")
        self.df = checked_df
        self.scale = v.copy()
        self.scale.setflags(write=False)
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
        x_inv = np.linalg.inv(xx)
        tr = np.einsum("ab,ba->", self.scale, x_inv, optimize=True)
        return float(self._log_norm - (self.df + self.dim + 1.0) / 2.0 * logdet - 0.5 * tr)

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Vectorized log-density for a stack of SPD matrices, shape ``(N, p, p)`` (``-inf`` per
        row that is not symmetric or not positive definite).

        ``np.linalg.inv``, like ``np.linalg.cholesky`` (see :func:`batched_pd_logdet`'s
        docstring), raises ``LinAlgError`` for the WHOLE batch if even one matrix is singular --
        singular is just the not-positive-definite boundary case, already scored ``-inf`` below via
        ``is_valid``, so its inverse is never actually used. Substitute the identity for invalid
        rows before inverting so the call always succeeds, mirroring how the scalar
        :meth:`log_density` path never calls ``inv`` on a matrix it is about to reject.
        """
        xx = _matrix_batch(x, self.dim)
        is_valid, logdet = _support_mask_and_logdet(xx)
        safe_xx = np.where(is_valid[:, None, None], xx, np.eye(self.dim))
        x_inv = np.linalg.inv(safe_xx)
        tr = np.einsum("ab,nba->n", self.scale, x_inv, optimize=True)
        rv = self._log_norm - (self.df + self.dim + 1.0) / 2.0 * logdet - 0.5 * tr
        return np.where(is_valid, rv, -np.inf)

    def sampler(self, seed: int | None = None) -> "InverseWishartSampler":
        """Return a sampler for drawing SPD matrices from this distribution."""
        return InverseWishartSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "InverseWishartEstimator":
        """Return the fixed-df mean-moment estimator when that moment exists."""
        return InverseWishartEstimator(self.dim, self.df, name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> "InverseWishartDataEncoder":
        """Return the data encoder used by this distribution for vectorized methods."""
        return InverseWishartDataEncoder(self.dim)


class InverseWishartSampler(DistributionSampler):
    """Draw SPD matrices by inverting a Bartlett-factorized precision draw."""

    def __init__(self, dist: InverseWishartDistribution, seed: int | None = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist
        wishart_scale = np.linalg.inv(dist.scale)
        wishart_scale = 0.5 * (wishart_scale + wishart_scale.T)
        self._chol_precision_scale = np.linalg.cholesky(wishart_scale)

    def _one(self) -> np.ndarray:
        p = self.dist.dim
        a = np.zeros((p, p))
        for i in range(p):
            a[i, i] = math.sqrt(self.rng.chisquare(self.dist.df - i))
            for j in range(i):
                a[i, j] = self.rng.randn()
        la = self._chol_precision_scale @ a
        result = np.linalg.inv(la @ la.T)
        return 0.5 * (result + result.T)

    def sample(self, size: int | None = None, *, batched: bool = True) -> np.ndarray:
        """Draw one or more inverse-Wishart SPD matrix samples."""
        if size is None:
            return self._one()
        count = _validated_sample_size(size)
        if count == 0:
            return np.empty((0, self.dist.dim, self.dist.dim))
        return np.stack([self._one() for _ in range(count)])


class InverseWishartAccumulator(_MeanScatterAccumulator):
    """Accumulate the weighted sum of matrices ``sum_i w_i X_i`` and the total weight."""

    def acc_to_encoder(self) -> "InverseWishartDataEncoder":
        """Return the encoder compatible with the accumulated matrix statistics."""
        return InverseWishartDataEncoder(self.dim)


class InverseWishartAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for InverseWishartAccumulator."""

    def __init__(self, dim: int, name: str | None = None, keys: str | None = None) -> None:
        self.dim = _validated_dimension(dim, "inverse-Wishart dimension")
        self.name = name
        self.keys = keys

    def make(self) -> InverseWishartAccumulator:
        """Create an accumulator for weighted inverse-Wishart matrix observations."""
        return InverseWishartAccumulator(self.dim, name=self.name, keys=self.keys)


class InverseWishartMeanMomentEstimator(ParameterEstimator):
    """Fixed-df method-of-moments estimator based on the inverse-Wishart mean.

    This is not a likelihood MLE. It is defined only for ``df > p + 1``, where
    ``E[X] = Psi / (df-p-1)`` exists.
    """

    def __init__(self, dim: int, df: float, name: str | None = None, keys: str | None = None) -> None:
        self.dim = _validated_dimension(dim, "inverse-Wishart dimension")
        self.df = _validated_inverse_wishart_df(df, self.dim)
        if self.df <= self.dim + 1.0:
            raise InverseWishartMomentFitError(
                "inverse-Wishart mean-moment fitting requires df > p + 1"
            )
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> InverseWishartAccumulatorFactory:
        """Return an accumulator factory for estimating the fixed-df scale matrix."""
        return InverseWishartAccumulatorFactory(self.dim, name=self.name, keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat: tuple[np.ndarray, float]) -> InverseWishartDistribution:
        """Estimate scale by the inverse-Wishart mean moment, not likelihood."""
        sum_x, count = _validated_mean_scatter_statistics(
            suff_stat,
            self.dim,
        )
        factor = self.df - self.dim - 1.0
        if count == 0.0:
            raise InverseWishartMomentFitError(
                "inverse-Wishart mean-moment fitting requires positive observation weight"
            )
        scale = factor * (sum_x / count)  # E[X] = Psi/(df-p-1)
        result = InverseWishartDistribution(
            self.df,
            scale,
            name=self.name,
            keys=self.keys,
        )
        result.fit_metadata = {
            "converged": True,
            "method": "mean-moment",
            "moment": "E[X]",
            "repairs": (),
        }
        return result


class InverseWishartDataEncoder(DataSequenceEncoder):
    """Encode a sequence of ``(p, p)`` matrices as an ``(N, p, p)`` float array."""

    def __init__(self, dim: int) -> None:
        self.dim = _validated_dimension(
            dim,
            "inverse-Wishart encoder dimension",
        )

    def __str__(self) -> str:
        return "InverseWishartDataEncoder(%d)" % self.dim

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, InverseWishartDataEncoder)
            and self.dim == other.dim
        )

    def seq_encode(self, x: Sequence[np.ndarray]) -> np.ndarray:
        """Encode a sequence of SPD matrices as a floating matrix stack."""
        encoded, _ = _validated_spd_batch(x, self.dim)
        return encoded

    def row_count(self, x: np.ndarray) -> int:
        """Return the number of matrices after validating encoded geometry."""
        encoded, _ = _validated_spd_batch(x, self.dim)
        return len(encoded)


# Backward-compatible name. The concrete class name makes the method explicit.
InverseWishartEstimator = InverseWishartMeanMomentEstimator
