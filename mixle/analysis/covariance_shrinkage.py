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

__all__ = ["LedoitWolfEstimator", "LedoitWolfInsufficientData"]


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


_MIN_EFFECTIVE_COUNT = 2.0
"""Minimum accumulated observation weight :meth:`LedoitWolfEstimator.estimate` requires before it will
compute a shrinkage estimate.

Below 2, the raw (unshrunk) sample covariance ``S = E[xx^T] - mean*mean^T`` is not merely noisy but
*identically zero*: mean-centering a single effective observation always cancels exactly, in exact
arithmetic, whatever the data is, so there is no variance information to shrink at any dimension --
dividing through anyway is what previously produced NaN/Inf sufficient statistics (zero weight) or a
covariance that is purely a numerical-healing artifact rather than an estimate (one observation's
exactly-zero sample covariance, silently jittered into a tiny-but-"valid" positive-definite matrix by
:class:`~mixle.stats.multivariate.multivariate_gaussian.MultivariateGaussianDistribution`'s own
construction healing). This is deliberately a much weaker floor than the classical "n > dim" rule of
thumb for the plain sample covariance: Ledoit-Wolf's entire purpose is a well-conditioned estimate
precisely when n is small relative to dim (see module docstring), so gating on n > dim here would
defeat the reason this estimator exists.
"""


class LedoitWolfInsufficientData(ValueError):
    """Specific estimation error raised when the data cannot identify covariance.

    ``effective_count`` and ``dim`` let a caller decide whether to accumulate more data and retry.
    Raising, rather than returning a foreign non-distribution object, preserves the common
    :class:`~mixle.stats.compute.pdist.ParameterEstimator` result protocol.
    """

    def __init__(self, reason: str, effective_count: float, dim: int | None):
        super().__init__(reason)
        self.reason = reason
        self.effective_count = effective_count
        self.dim = dim


class LedoitWolfEstimator(ParameterEstimator):
    """Estimate a multivariate Gaussian with a Ledoit-Wolf-shrunk covariance.

    Returns a :class:`MultivariateGaussianDistribution` whose mean is the sample mean and whose covariance
    is shrunk toward a scaled identity by the data-driven Ledoit-Wolf intensity. The chosen intensity is
    exposed on the returned distribution as ``dist.shrinkage`` for inspection. When the accumulated
    sufficient statistics carry too little effective weight or no identifiable dispersion, raises
    :class:`LedoitWolfInsufficientData`.
    """

    def __init__(self, dim: int | None = None, name: str | None = None, keys: str | None = None):
        self.dim = dim
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> LedoitWolfAccumulatorFactory:
        """Return an accumulator factory for streaming Ledoit-Wolf sufficient statistics."""
        return LedoitWolfAccumulatorFactory(self.dim, self.keys, self.name)

    def estimate(self, nobs, suff_stat) -> MultivariateGaussianDistribution:
        """Estimate a Gaussian distribution from accumulated Ledoit-Wolf sufficient statistics.

        Raises:
            LedoitWolfInsufficientData: If the accumulated weight is below
                :data:`_MIN_EFFECTIVE_COUNT` or the observations contain no numerically identifiable
                dispersion.
            ValueError: If the shrinkage computation itself produces a non-finite mean or covariance
                despite adequate, already-validated sufficient statistics (a numerical-overflow defect,
                not an insufficient-data condition). A substantively non-positive-semidefinite result is
                rejected by :class:`MultivariateGaussianDistribution`'s own construction, which this
                relies on as the final PSD gate.
        """
        s1, s2, s3, s4, n = suff_stat
        n = float(n) if n is not None else 0.0
        if s1 is None or not np.isfinite(n) or n < _MIN_EFFECTIVE_COUNT:
            raise LedoitWolfInsufficientData(
                reason=(
                    f"need at least {_MIN_EFFECTIVE_COUNT:g} of accumulated observation weight to "
                    f"estimate a covariance, got {n:g}"
                ),
                effective_count=n,
                dim=None if s1 is None else len(np.asarray(s1)),
            )
        s1 = np.asarray(s1)
        s2 = np.asarray(s2)
        mean, cov, shrunk, delta = _shrink(s1, s2, np.asarray(s3), float(s4), n)
        if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(shrunk))):
            raise ValueError(
                "Ledoit-Wolf shrinkage produced a non-finite mean or covariance from finite, "
                f"adequately-weighted sufficient statistics (effective count {n:g}); this indicates a "
                "numerical overflow in the shrinkage computation, not insufficient data."
            )
        raw_scale = max(
            float(np.max(np.abs(s2 / n), initial=0.0)),
            float(np.max(np.abs(mean), initial=0.0)) ** 2,
            np.finfo(np.float64).tiny,
        )
        dispersion = float(np.trace(cov))
        resolution = 16.0 * np.finfo(np.float64).eps * raw_scale * max(1, len(mean))
        if not np.isfinite(dispersion) or dispersion <= resolution:
            raise LedoitWolfInsufficientData(
                reason=(
                    "observations contain no covariance dispersion distinguishable from sufficient-statistic "
                    f"roundoff (trace={dispersion!r}, resolution={resolution!r})"
                ),
                effective_count=n,
                dim=len(mean),
            )
        # A finite-sample Ledoit-Wolf optimum can still land exactly on a rank-deficient sample
        # covariance (notably two distinct centered observations, whose outer products coincide and
        # make the estimated shrinkage numerator zero). The distribution contract is strictly
        # positive-definite. Apply only a scale-relative numerical eigenvalue floor and expose its
        # exact contribution; this is never used to rescue the zero-dispersion case rejected above.
        shrunk = 0.5 * (shrunk + shrunk.T)
        eigenvalues = np.linalg.eigvalsh(shrunk)
        variance_scale = dispersion / len(mean)
        eigenvalue_floor = np.sqrt(np.finfo(np.float64).eps) * variance_scale
        regularization = max(0.0, eigenvalue_floor - float(eigenvalues[0]))
        regularized = shrunk + regularization * np.eye(len(mean))
        dist = MultivariateGaussianDistribution(mean, regularized, name=self.name)
        dist.shrinkage = delta
        dist.regularization = regularization
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
        """Add one weighted observation to the streaming covariance-shrinkage statistics.

        Raises:
            ValueError: If ``weight`` is negative or non-finite, ``x`` contains a non-finite entry, or
                ``x``'s dimension does not match every previously accumulated observation's.
        """
        weight = float(weight)
        if not np.isfinite(weight) or weight < 0.0:
            raise ValueError(f"observation weight must be finite and nonnegative, got {weight!r}")
        x = np.asarray(x, dtype=float)
        if x.ndim != 1:
            raise ValueError(f"observation must be one-dimensional, got shape {x.shape}")
        if not np.all(np.isfinite(x)):
            n_bad = int(np.sum(~np.isfinite(x)))
            raise ValueError(f"observation must be finite, got {n_bad} of {len(x)} non-finite entries")
        if self.dim is not None and len(x) != self.dim:
            raise ValueError(f"expected an observation of dimension {self.dim}, got dimension {len(x)}")
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
        """Add a batch of weighted observations to the streaming sufficient statistics.

        Raises:
            ValueError: If any weight is negative or non-finite, any observation contains a non-finite
                entry, or the batch's dimension does not match every previously accumulated
                observation's.
        """
        x = np.asarray(x, dtype=float)
        weights = np.asarray(weights, dtype=float)
        if x.ndim != 2:
            raise ValueError(f"observations must be a two-dimensional (rows, dimensions) matrix, got shape {x.shape}")
        if weights.ndim != 1 or weights.shape != (x.shape[0],):
            raise ValueError(
                f"weights must have exact shape ({x.shape[0]},), one value per observation; got {weights.shape}"
            )
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("observation weights must be finite and nonnegative")
        if not np.all(np.isfinite(x)):
            n_bad = int(np.sum(~np.isfinite(x)))
            raise ValueError(f"observations must be finite, got {n_bad} non-finite entries across the batch")
        if self.dim is not None and x.shape[1] != self.dim:
            raise ValueError(f"expected observations of dimension {self.dim}, got dimension {x.shape[1]}")
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
        """Merge sufficient statistics from another Ledoit-Wolf accumulator.

        Raises:
            ValueError: If the incoming statistics' dimension does not match this accumulator's.
        """
        s1, s2, s3, s4, count = suff_stat
        if s1 is None:
            return self
        if self.s1 is None:
            self.dim = len(s1)
            self.s1, self.s2, self.s3, self.s4, self.count = (np.array(s1), np.array(s2), np.array(s3), s4, count)
        else:
            if len(s1) != self.dim:
                raise ValueError(
                    f"cannot combine accumulators of mismatched dimension: expected {self.dim}, got {len(s1)}"
                )
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
