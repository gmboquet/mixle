"""Shared fail-closed contracts for finite vector-valued probability families."""

from __future__ import annotations

from operator import index
from typing import Any

import numpy as np

from mixle.utils.aliasing import broadcast_pseudo_count


def dimension(value: Any, *, label: str, allow_none: bool = False) -> int | None:
    if value is None and allow_none:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("%s must be a positive integer" % label)
    try:
        result = index(value)
    except TypeError as exc:
        raise TypeError("%s must be a positive integer" % label) from exc
    if result <= 0:
        raise ValueError("%s must be a positive integer" % label)
    return result


def finite_scalar(value: Any, *, label: str, nonnegative: bool = False, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0:
        raise TypeError("%s must be a real scalar" % label)
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("%s must be a real scalar" % label) from exc
    if not np.isfinite(result):
        raise ValueError("%s must be finite" % label)
    if positive and not result > 0.0:
        raise ValueError("%s must be positive" % label)
    if nonnegative and result < 0.0:
        raise ValueError("%s must be non-negative" % label)
    return result


def vector(value: Any, *, label: str, dim: int | None = None, positive: bool = False) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("%s must be a finite numeric vector" % label) from exc
    expected = None if dim is None else (dim,)
    if result.ndim != 1 or result.size == 0 or (expected is not None and result.shape != expected):
        shape_text = "a nonempty one-dimensional vector" if dim is None else "exact shape (%d,)" % dim
        raise ValueError("%s must have %s" % (label, shape_text))
    if np.any(~np.isfinite(result)):
        raise ValueError("%s must contain only finite values" % label)
    if positive and np.any(result <= 0.0):
        raise ValueError("%s must contain only positive values" % label)
    return result.copy()


def matrix(
    value: Any,
    *,
    label: str,
    dim: int,
    symmetric: bool = False,
) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("%s must be a finite numeric matrix" % label) from exc
    if result.shape != (dim, dim):
        raise ValueError("%s must have exact shape (%d, %d)" % (label, dim, dim))
    if np.any(~np.isfinite(result)):
        raise ValueError("%s must contain only finite values" % label)
    if symmetric and not np.allclose(result, result.T):
        raise ValueError("%s must be symmetric" % label)
    return result.copy()


def event(value: Any, dim: int, *, label: str) -> np.ndarray:
    return vector(value, label=label, dim=dim)


def batch(value: Any, dim: int, *, label: str, allow_empty: bool = True) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("%s must be a finite numeric matrix" % label) from exc
    if allow_empty and result.shape == (0,):
        return np.empty((0, dim), dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != dim:
        raise ValueError("%s must have exact shape (N, %d)" % (label, dim))
    if not allow_empty and len(result) == 0:
        raise ValueError("%s must contain at least one row" % label)
    if np.any(~np.isfinite(result)):
        raise ValueError("%s must contain only finite values" % label)
    return result


def weight(value: Any, *, label: str = "observation weight") -> float:
    return finite_scalar(value, label=label, nonnegative=True)


def weights(value: Any, rows: int, *, label: str = "observation weights") -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("%s must be numeric" % label) from exc
    if result.shape != (rows,):
        raise ValueError("%s must have exact shape (%d,)" % (label, rows))
    if np.any(~np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError("%s must be finite and non-negative" % label)
    return result


def pseudo_counts(value: Any, *, label: str) -> tuple[float | None, float | None]:
    raw = broadcast_pseudo_count(value, 2)
    if raw is None:
        raw = (None, None)
    if not isinstance(raw, (tuple, list)) or len(raw) != 2:
        raise ValueError("%s must be a scalar or a two-item sequence" % label)
    checked = tuple(None if item is None else finite_scalar(item, label=label, nonnegative=True) for item in raw)
    return checked[0], checked[1]


def gaussian_prior_statistics(
    value: Any,
    dim: int | None,
    *,
    diagonal: bool,
) -> tuple[np.ndarray | None, np.ndarray | None, int | None]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError("Gaussian prior sufficient statistic must be a (mean, covariance) tuple")
    checked_dim = dimension(dim, label="Gaussian dimension", allow_none=True)
    prior_mean = None
    prior_covar = None
    if value[0] is not None:
        prior_mean = vector(value[0], label="Gaussian prior mean", dim=checked_dim)
        if checked_dim is None:
            checked_dim = len(prior_mean)
    if value[1] is not None:
        if diagonal:
            prior_covar = vector(
                value[1],
                label="diagonal Gaussian prior covariance",
                dim=checked_dim,
                positive=True,
            )
            if checked_dim is None:
                checked_dim = len(prior_covar)
        else:
            try:
                raw = np.asarray(value[1], dtype=np.float64)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("Gaussian prior covariance must be a finite numeric matrix") from exc
            if checked_dim is None:
                if raw.ndim != 2 or raw.shape[0] == 0 or raw.shape[0] != raw.shape[1]:
                    raise ValueError("Gaussian prior covariance must be a nonempty square matrix")
                checked_dim = raw.shape[0]
            prior_covar = matrix(
                raw,
                label="Gaussian prior covariance",
                dim=checked_dim,
                symmetric=True,
            )
            try:
                np.linalg.cholesky(prior_covar)
            except np.linalg.LinAlgError as exc:
                raise ValueError("Gaussian prior covariance must be positive definite") from exc
    for parameter in (prior_mean, prior_covar):
        if parameter is not None:
            parameter.setflags(write=False)
    return prior_mean, prior_covar, checked_dim


def require_pseudo_moments(
    counts: tuple[float | None, float | None],
    prior_mean: np.ndarray | None,
    prior_covar: np.ndarray | None,
) -> None:
    # An unpaired pseudo-count is not an error, it just contributes nothing: the mean and
    # covariance estimators each fall through to the plain maximum-likelihood quantity when their
    # prior is absent, exactly as the univariate family documents and does. Rejecting construction
    # here instead broke the cross-family contract that a scalar ``pseudo_count`` is accepted by
    # every tuple-arity estimator, and it rejected a configuration whose prior can still be
    # supplied later through ``set_prior``. The real defect the raise was standing in for -- the
    # estimator evaluating ``pseudo_count * None`` -- is fixed at the use sites themselves.
    del counts, prior_mean, prior_covar


def pooled_gaussian_covariance(
    sum_x: np.ndarray,
    sum_xx: np.ndarray,
    count: float,
    mean: np.ndarray,
    pseudo_count: float | None,
    prior_mean: np.ndarray | None,
    prior_covar: np.ndarray | None,
    *,
    diagonal: bool,
) -> np.ndarray:
    if diagonal:
        observed_scatter = sum_xx - 2.0 * mean * sum_x + count * mean * mean
        scale = np.maximum.reduce(
            (
                np.abs(sum_xx),
                np.abs(2.0 * mean * sum_x),
                np.abs(count * mean * mean),
                np.ones_like(mean),
            )
        )
        if np.any(observed_scatter < -1.0e-6 * scale):
            raise ValueError("Gaussian sufficient statistics imply a negative centered second moment")
        observed_scatter = np.maximum(observed_scatter, 0.0)
        if pseudo_count not in (None, 0.0) and prior_covar is not None:
            offset = 0.0 if prior_mean is None else (prior_mean - mean) ** 2
            prior_scatter = pseudo_count * (prior_covar + offset)
            return (observed_scatter + prior_scatter) / (count + pseudo_count)
    else:
        observed_scatter = sum_xx - np.outer(mean, sum_x) - np.outer(sum_x, mean) + count * np.outer(mean, mean)
        observed_scatter = 0.5 * (observed_scatter + observed_scatter.T)
        eigenvalues = np.linalg.eigvalsh(observed_scatter)
        scale = max(
            float(np.linalg.norm(sum_xx, ord=2)),
            float(np.linalg.norm(np.outer(mean, sum_x), ord=2)),
            float(np.linalg.norm(count * np.outer(mean, mean), ord=2)),
            1.0,
        )
        if eigenvalues[0] < -1.0e-6 * scale:
            raise ValueError("Gaussian sufficient statistics imply a non-positive-semidefinite scatter")
        if pseudo_count not in (None, 0.0) and prior_covar is not None:
            offset = 0.0 if prior_mean is None else np.outer(prior_mean - mean, prior_mean - mean)
            prior_scatter = pseudo_count * (prior_covar + offset)
            return (observed_scatter + prior_scatter) / (count + pseudo_count)
    if count == 0.0:
        return np.zeros_like(sum_xx)
    return observed_scatter / count


def marginal_indices(value: Any, dim: int) -> np.ndarray:
    try:
        raw = list(value)
    except TypeError as exc:
        raise TypeError("kept indices must be an iterable of integers") from exc
    if not raw:
        raise ValueError("keep at least one dimension")
    checked: list[int] = []
    for item in raw:
        if isinstance(item, (bool, np.bool_)):
            raise TypeError("kept indices must be integers")
        try:
            coordinate = index(item)
        except TypeError as exc:
            raise TypeError("kept indices must be integers") from exc
        if not 0 <= coordinate < dim:
            raise ValueError("kept indices must be in [0, dim)")
        checked.append(coordinate)
    if len(set(checked)) != len(checked):
        raise ValueError("kept indices must be unique; use an explicit duplication transform to repeat coordinates")
    return np.asarray(checked, dtype=np.int64)


def gaussian_moments(
    value: Any,
    dim: int | None,
    *,
    diagonal: bool,
) -> tuple[np.ndarray | None, np.ndarray | None, float, int | None]:
    if not isinstance(value, tuple) or len(value) != 3:
        raise ValueError("Gaussian sufficient statistic must be a (sum, second_moment, count) tuple")
    count = weight(value[2], label="Gaussian sufficient-statistic count")
    if value[0] is None or value[1] is None:
        if value[0] is not None or value[1] is not None or count != 0.0:
            raise ValueError("empty Gaussian sufficient statistics require both moments to be None and count zero")
        return None, None, count, dim
    sum_x = vector(value[0], label="Gaussian first-moment statistic", dim=dim)
    inferred_dim = len(sum_x) if dim is None else dim
    if diagonal:
        sum_xx = vector(value[1], label="Gaussian diagonal second-moment statistic", dim=inferred_dim)
        if np.any(sum_xx < 0.0):
            raise ValueError("Gaussian diagonal second-moment statistic must be non-negative")
    else:
        sum_xx = matrix(
            value[1],
            label="Gaussian second-moment statistic",
            dim=inferred_dim,
        )
        scale = max(float(np.linalg.norm(sum_xx, ord=2)), 1.0)
        # sum_i w_i x_i x_i^T is symmetric by construction, so any asymmetry here is float
        # accumulation noise -- and float32/GPU EM produces exactly that. Requiring exact symmetry
        # rejected those legitimate sufficient statistics outright. Refuse asymmetry too large to be
        # noise (same relative tolerance as the PSD check below, which already concedes the point),
        # then symmetrize the remainder rather than failing on it.
        if np.max(np.abs(sum_xx - sum_xx.T), initial=0.0) > 1.0e-6 * scale:
            raise ValueError("Gaussian second-moment statistic must be symmetric")
        sum_xx = 0.5 * (sum_xx + sum_xx.T)
        if np.linalg.eigvalsh(sum_xx)[0] < -1.0e-6 * scale:
            raise ValueError("Gaussian second-moment statistic must be positive semidefinite")
    # A zero-count component is a starved one, which is a normal EM state and the whole reason the
    # mixture weight floor exists. Its moments are never read: the estimator sees count == 0 and
    # returns the floor defaults whatever they hold, so rejecting them fails closed on dead data and
    # turns a component the floor was built to revive into a hard crash. Their shape, finiteness,
    # symmetry, non-negativity and PSD-ness were all still checked above, unconditionally.
    return sum_x, sum_xx, count, inferred_dim


def student_t_moments(
    value: Any,
    dim: int | None,
) -> tuple[float, float, np.ndarray | None, np.ndarray | None, int | None]:
    if not isinstance(value, tuple) or len(value) != 4:
        raise ValueError("Student-t sufficient statistic must be a (count, latent_weight, sum, second_moment) tuple")
    count = weight(value[0], label="Student-t sufficient-statistic count")
    sum_u = weight(value[1], label="Student-t latent-weight total")
    if value[2] is None or value[3] is None:
        if value[2] is not None or value[3] is not None or count != 0.0 or sum_u != 0.0:
            raise ValueError(
                "empty Student-t sufficient statistics require both moments to be None and both totals zero"
            )
        return count, sum_u, None, None, dim
    sum_ux = vector(value[2], label="Student-t first-moment statistic", dim=dim)
    inferred_dim = len(sum_ux) if dim is None else dim
    sum_uxx = matrix(
        value[3],
        label="Student-t second-moment statistic",
        dim=inferred_dim,
        symmetric=True,
    )
    scale = max(float(np.linalg.norm(sum_uxx, ord=2)), 1.0)
    if np.linalg.eigvalsh(sum_uxx)[0] < -1.0e-6 * scale:
        raise ValueError("Student-t second-moment statistic must be positive semidefinite")
    if (count == 0.0) != (sum_u == 0.0):
        raise ValueError("Student-t count and latent-weight total must both be zero or both be positive")
    if sum_u == 0.0 and (np.any(sum_ux != 0.0) or np.any(sum_uxx != 0.0)):
        raise ValueError("zero Student-t latent weight requires zero weighted moments")
    return count, sum_u, sum_ux, sum_uxx, inferred_dim


# --------------------------------------------------------------------------------------------
# Shift-anchored moments for vector families whose M-step differences raw outer-product moments.
#
# The vector twin of ``mixle.stats.univariate.continuous._observation_contracts``. Four release
# waves fixed this defect one family at a time and each wave a family recorded as an accepted limit
# came back as a blocking finding, so the gate and the disclosure live here once rather than being
# transcribed into the next family that needs them.
# --------------------------------------------------------------------------------------------

# Same constant, same rationale, as every scalar and matrix sibling: the raw ``E[xx^T] - mu mu^T``
# form loses about ``eps * (mean/sd)^2`` relative accuracy, so a (mean/sd)^2 up to 4e6 (ratio ~2000)
# keeps it within ~1e-9 relative error and the historical single-pass statistics are bit-preserved
# there. Beyond it a shift-anchored track has to take over.
VECTOR_ANCHOR_CONDITION_RATIO = 4.0e6


def needs_vector_anchor(chunk_sum: np.ndarray, chunk_sum2_diag: np.ndarray, w_sum: float) -> bool:
    """Whether a chunk's weighted moments are too ill-conditioned for the raw covariance form.

    The per-coordinate version of the scalar gate: the offset that destroys a covariance is the same
    offset in every entry it touches and shows up first in that coordinate's own variance, so
    testing the diagonal tests the matrix. ``spread2`` computed here is itself the
    cancellation-prone estimate, but as a GATE it is reliable -- when cancellation has corrupted it,
    the corruption is bounded by ``eps * m^2``, which still leaves ``m*m`` orders of magnitude above
    ``VECTOR_ANCHOR_CONDITION_RATIO * spread2``. A non-positive computed spread in any coordinate
    activates the anchor outright (constant or near-constant data there).
    """
    if not w_sum > 0.0:
        return False
    m = np.asarray(chunk_sum, dtype=float) / w_sum
    spread2 = np.asarray(chunk_sum2_diag, dtype=float) / w_sum - m * m
    return bool(np.any(spread2 <= 0.0) or np.any(m * m > VECTOR_ANCHOR_CONDITION_RATIO * spread2))


def warn_uncorrectable_vector_moments(
    sum_x: np.ndarray | None,
    sum_xx_diag: np.ndarray | None,
    count: float,
    *,
    family: str,
) -> None:
    """Warn when raw-only vector statistics are too ill-conditioned for the covariance they imply.

    An anchored track fixes an accumulator's OWN accumulation. Statistics that arrive already
    reduced and without an anchor -- an engine/GPU kernel's stacked moments, a hand-built tuple, a
    legacy artifact -- cannot be corrected: the information cancellation destroyed is not in them
    any more. Naming it is the difference between an imprecise fit and a silently wrong one.

    Deliberately NOT a raise: these statistics are the declared exchange format and the raw M-step
    is what the library has always done with them.

    TWO regimes have to speak, and only the first one used to -- see
    :func:`mixle.stats.univariate.continuous._observation_contracts.warn_uncorrectable_raw_moments`
    for the full argument, which is the same one coordinate-wise. Partial loss (a coordinate's
    computed variance still positive but dominated by its squared mean) is gated by
    :data:`VECTOR_ANCHOR_CONDITION_RATIO`. TOTAL loss -- cancellation has eaten a coordinate's
    whole spread and the raw form computes a non-positive variance there -- was excluded because it
    is also what an ordinary degenerate or single-point component looks like, and that exclusion
    made the worst case the silent one. Both now warn; the total-loss message does not claim the
    data was ill-conditioned, because from raw moments alone that is not knowable. A single
    observation and all-zero coordinates stay quiet.
    """
    if count <= 0.0 or sum_x is None or sum_xx_diag is None:
        return
    mean = np.asarray(sum_x, dtype=float) / count
    variance = np.asarray(sum_xx_diag, dtype=float) / count - mean * mean
    ratio = np.where(variance > 0.0, mean * mean / np.where(variance > 0.0, variance, 1.0), 0.0)
    worst = int(np.argmax(ratio))
    if ratio[worst] > VECTOR_ANCHOR_CONDITION_RATIO:
        import warnings

        lost = min(100.0, 100.0 * float(np.log10(ratio[worst])) / 16.0)
        warnings.warn(
            "%s sufficient statistics arrived without shift-anchored moments and are too "
            "ill-conditioned for the raw E[xx^T] - mu mu^T scatter: coordinate %d has mean^2/variance "
            "%.3g, so the fitted scale matrix loses roughly %.0f%% of its significant digits to "
            "cancellation. Accumulate through this family's own accumulator (which anchors "
            "automatically), or subtract a constant origin from the data before fitting."
            % (family, worst, float(ratio[worst]), lost),
            RuntimeWarning,
            stacklevel=3,
        )
        return
    collapsed = np.flatnonzero((variance <= 0.0) & (mean != 0.0))
    if count <= 1.0 or collapsed.size == 0:
        return
    import warnings

    coordinate = int(collapsed[int(np.argmax(np.abs(mean[collapsed])))])
    warnings.warn(
        "%s sufficient statistics arrived without shift-anchored moments and imply a non-positive "
        "E[xx^T] - mu mu^T variance in coordinate %d at mean %.6g, so the fitted scale matrix falls "
        "onto this family's floor there. Raw moments at that magnitude cannot resolve a spread below "
        "about %.3g, so this is either a genuinely degenerate coordinate or one whose spread "
        "cancellation destroyed -- they are not distinguishable from these statistics. Accumulate "
        "through this family's own accumulator (which anchors automatically), or subtract a constant "
        "origin from the data before fitting."
        % (family, coordinate, float(mean[coordinate]), 1.5e-8 * abs(float(mean[coordinate]))),
        RuntimeWarning,
        stacklevel=3,
    )
