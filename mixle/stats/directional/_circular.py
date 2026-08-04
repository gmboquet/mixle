"""Shared validation contracts for circular distributions."""

import math
import operator
from typing import Any

import numpy as np

_TRIG_ATOL = 1.0e-8
_MOMENT_ATOL = 1.0e-8


def validated_angle(value: Any, name: str = "angle observation") -> float:
    """Return one finite real-valued angle."""
    if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0:
        raise TypeError("%s must be a real scalar" % name)
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError("%s must be a real scalar" % name) from exc
    if not np.isfinite(result):
        raise ValueError("%s must be finite" % name)
    return result


def validated_angles(value: Any, name: str = "angle observations") -> np.ndarray:
    """Return a finite one-dimensional angle batch."""
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("%s must be numeric" % name) from exc
    if result.ndim != 1:
        raise ValueError("%s must be a one-dimensional array" % name)
    if np.any(~np.isfinite(result)):
        raise ValueError("%s must be finite" % name)
    return result


def validated_trig(
    value: Any,
    name: str = "encoded circular observations",
) -> tuple[np.ndarray, np.ndarray]:
    """Return a finite same-shape ``(cos, sin)`` batch on the unit circle."""
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError("%s must be a two-item (cos, sin) tuple" % name)
    try:
        cosine = np.asarray(value[0], dtype=np.float64)
        sine = np.asarray(value[1], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("%s must be numeric" % name) from exc
    if cosine.ndim != 1 or sine.shape != cosine.shape:
        raise ValueError("%s must contain same-shape one-dimensional arrays" % name)
    if np.any(~np.isfinite(cosine)) or np.any(~np.isfinite(sine)):
        raise ValueError("%s must be finite" % name)
    if not np.allclose(
        cosine * cosine + sine * sine,
        1.0,
        rtol=0.0,
        atol=_TRIG_ATOL,
    ):
        raise ValueError("%s must satisfy cos^2 + sin^2 = 1" % name)
    return cosine, sine


def validated_weight(value: Any) -> float:
    """Return a finite non-negative observation weight."""
    if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0:
        raise TypeError("circular observation weight must be a real scalar")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError("circular observation weight must be a real scalar") from exc
    if not np.isfinite(result) or result < 0.0:
        raise ValueError("circular observation weight must be finite and non-negative")
    return result


def validated_weights(value: Any, rows: int) -> np.ndarray:
    """Return a finite non-negative weight vector matching ``rows``."""
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("circular observation weights must be numeric") from exc
    if result.shape != (rows,):
        raise ValueError("circular observation weights must have exact shape (%d,)" % rows)
    if np.any(~np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError("circular observation weights must be finite and non-negative")
    return result


def validated_sample_size(value: Any) -> int:
    """Return a non-negative integral sample size."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("sample size must be a non-negative integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError("sample size must be a non-negative integer") from exc
    if result < 0:
        raise ValueError("sample size must be non-negative")
    return result


def validated_circular_statistics(
    value: Any,
    *,
    count_index: int,
) -> tuple[float, float, float]:
    """Return canonical ``(count, sum_cos, sum_sin)`` feasible moments."""
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError("circular sufficient statistics must contain three items")
    if count_index == 0:
        raw_count, raw_cosine, raw_sine = value
    elif count_index == 2:
        raw_cosine, raw_sine, raw_count = value
    else:
        raise ValueError("circular statistic count index must be zero or two")
    count = validated_weight(raw_count)
    cosine = validated_angle(raw_cosine, "circular cosine sum")
    sine = validated_angle(raw_sine, "circular sine sum")
    tolerance = _MOMENT_ATOL * max(1.0, count)
    if count == 0.0:
        if cosine != 0.0 or sine != 0.0:
            raise ValueError("empty circular statistics must have zero moments")
    elif math.hypot(cosine, sine) > count + tolerance:
        raise ValueError("circular resultant cannot exceed observation weight")
    return count, cosine, sine


def validated_em_statistics(value: Any) -> tuple[float, float, float]:
    """Return ``(sum_x, sum_y, count)`` for projected-normal EM."""
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError("projected-normal sufficient statistics must contain three items")
    sum_x = validated_angle(value[0], "projected-normal x sum")
    sum_y = validated_angle(value[1], "projected-normal y sum")
    count = validated_weight(value[2])
    if count == 0.0 and (sum_x != 0.0 or sum_y != 0.0):
        raise ValueError("empty projected-normal statistics must have zero moments")
    return sum_x, sum_y, count


def symmetrized_scatter(scatter: np.ndarray, label: str) -> np.ndarray:
    """Accept a scatter matrix that is symmetric to rounding, and return it exactly symmetric.

    Directional scatter statistics are accumulated as sums of outer products ``x x^T``, which are
    symmetric in exact arithmetic. They are not always symmetric BIT-FOR-BIT: the (i, j) and (j, i)
    entries are independent floating-point reductions, and a vectorized or multi-threaded BLAS may
    sum them in different orders. The previous guard required ``np.array_equal(scatter, scatter.T)``,
    so it rejected accumulators the library itself had just produced -- keyed pooling across sites
    failed for Bingham, Kent, and Watson on one CI runner while passing on another, with no
    difference in the data.

    A one-ulp asymmetry is not a corrupt statistic, but it is also not something downstream should
    have to think about: the eigen-decompositions these families run assume symmetry. So the matrix
    is checked against a tolerance scaled to its own magnitude and then symmetrized, which both
    restores the exact invariant callers rely on and keeps a genuinely asymmetric matrix -- one that
    was never a scatter statistic -- refused.
    """
    if not scatter.size:
        return scatter
    scale = float(np.max(np.abs(scatter)))
    tolerance = 8.0 * float(np.finfo(np.float64).eps) * max(scale, 1.0)
    if float(np.max(np.abs(scatter - scatter.T))) > tolerance:
        raise ValueError(f"{label} scatter must be symmetric")
    return (scatter + scatter.T) * 0.5
