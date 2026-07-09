"""The precision spectrum's front door: route a computation to the lowest-cost accurate backend.

Ties the spectrum together -- native float64, double-double (:mod:`mixle.engines.extended`), and the MPFR
tail (:mod:`mixle.engines.highprec`) -- behind one call that reads the certified error bound
(:mod:`mixle.engines.error_tracing`) and escalates only as far as the accuracy budget demands. This is
'use logic to preserve numerical accuracy with minimal compute' as an actual API: a well-conditioned sum
stays in fast float64, a cancelling one steps up to vectorized double-double, and only a catastrophically
ill-conditioned one pays for arbitrary precision.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from mixle.engines.error_tracing import float64_sum_is_accurate, sum_error_bound
from mixle.engines.extended import DoubleDouble, dd_sum

_TINY = np.finfo(np.float64).tiny
_U_DD = 2.0**-106  # double-double unit roundoff


def accurate_sum(x: Any, target_rel_error: float = 1e-12) -> tuple[float, str]:
    """Sum ``x`` to ``target_rel_error`` relative accuracy using the lowest-cost sufficient backend.

    Returns ``(value, backend)`` where ``backend`` is ``"float64"``, ``"dd"`` (double-double), or
    ``"mpfr<bits>"``. Escalates only when the certified error bound says the lower-cost backend cannot meet
    the budget -- so the common well-conditioned case never leaves vectorized float64.
    """
    arr = np.asarray(x, dtype=np.float64).ravel()
    if arr.size == 0:
        return 0.0, "float64"

    if float64_sum_is_accurate(arr, target_rel_error):
        return float(np.sum(arr)), "float64"

    dd = dd_sum(arr)
    s_dd = abs(float(dd.to_float()))
    abs_sum = float(np.abs(arr).sum())
    cond = abs_sum / max(s_dd, _TINY)  # condition number of the summation
    if cond * _U_DD <= target_rel_error:
        return float(dd.to_float()), "dd"

    # arbitrary precision: enough mantissa bits to cover the conditioning and the target
    from mixle.engines.highprec import available, hp_sum

    if not available():  # pragma: no cover - gmpy2/mpmath both absent
        return float(dd.to_float()), "dd"  # best effort; certified bound was reported via sum_error_bound
    bits = max(128, int(math.ceil(math.log2(max(cond, 1.0)) - math.log2(target_rel_error))) + 16)
    bits = min(bits, 4096)  # cond can be enormous when the dd result underflows to ~0; cap the allocation
    return hp_sum(arr, bits), "mpfr%d" % bits


def sum_certificate(x: Any) -> dict[str, float]:
    """Report the certified float64 summation error and the condition number, without choosing a backend."""
    arr = np.asarray(x, dtype=np.float64).ravel()
    s = abs(float(np.sum(arr)))
    bound = sum_error_bound(arr)
    return {
        "float64_value": float(np.sum(arr)),
        "abs_error_bound": bound,
        "rel_error_bound": bound / max(s, _TINY),
        "condition_number": float(np.abs(arr).sum()) / max(s, _TINY),
    }


def cast(x: Any, precision: Any) -> Any:
    """Cast ``x`` onto the spectrum: a native dtype name, ``"dd"``/``"fp128"``, or an integer bit width.

    Returns a numpy array (native), a :class:`~mixle.engines.extended.DoubleDouble` (``dd``/``fp128``),
    or an MPFR object array (>= ~fp256 / explicit bit width).
    """
    if isinstance(precision, str) and precision in ("dd", "fp128"):
        return DoubleDouble.from_float(np.asarray(x, dtype=np.float64))
    if isinstance(precision, str) and precision.startswith("fp"):
        bits = int(precision[2:])
        if bits <= 64:
            return np.asarray(x, dtype="float%d" % bits) if bits in (16, 32, 64) else _native_round(x, bits)
        return _mpfr_cast(x, bits)
    if isinstance(precision, int):
        if precision in (16, 32, 64):
            return np.asarray(x, dtype="float%d" % precision)
        if precision <= 64:
            return _native_round(x, precision)
        if precision <= 128:
            return DoubleDouble.from_float(np.asarray(x, dtype=np.float64))
        return _mpfr_cast(x, precision)
    return np.asarray(x, dtype=np.dtype(precision))


def _native_round(x: Any, bits: int):
    from mixle.engines.formats import FloatFormat

    return FloatFormat.fp(bits).round_trip(x)


def _mpfr_cast(x: Any, bits: int):
    from mixle.engines.highprec import HighPrecisionFormat

    return HighPrecisionFormat(bits).quantize(x)
