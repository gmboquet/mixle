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
from dataclasses import dataclass
from typing import Any

import numpy as np

from mixle.engines.error_tracing import float64_sum_is_accurate, sum_error_bound
from mixle.engines.extended import DoubleDouble, dd_sum

_TINY = np.finfo(np.float64).tiny
_FLOAT64_MAX = np.finfo(np.float64).max
_U_DD = 2.0**-106  # double-double unit roundoff
_MIN_MPFR_BITS = 128
_MAX_MPFR_BITS = 4096  # a compute-cost ceiling on accurate_sum's own escalation, not a backend limit --
# see _condition_number: no representable finite float64 (cond, target_rel_error) pair legitimately
# needs more than ~2114 bits by this formula, so this cap only ever binds as a defensive backstop.


@dataclass
class SumResult:
    """Result of :func:`accurate_sum`: the summed value plus a certificate of the accuracy actually
    achieved -- so a caller can verify the target was met rather than trust the call implicitly.

    ``status`` is one of:

    * ``"ok"``: ``rel_error_bound`` is a certified bound ``<= target_rel_error`` -- ``backend`` actually
      met the caller's target.
    * ``"insufficient"``: no backend available in this process could certify the target -- double-double
      was proven insufficient and either arbitrary precision is unavailable at all, or even
      ``_MAX_MPFR_BITS`` is not enough. ``value``/``backend`` still carry the best effort found (the
      highest-precision backend actually tried), for diagnostics only -- callers that require the
      accuracy guarantee must treat this as a failure and not use ``value``.

    Attributes:
        value: the computed sum (best-effort, not certified, when ``status == "insufficient"``).
        backend: which representation computed ``value`` -- ``"float64"``, ``"dd"``, or ``"mpfr<bits>"``.
        rel_error_bound: certified upper bound on ``value``'s relative error (an analytic bound -- the
            same quantity :func:`sum_certificate` reports for the plain float64 sum, not a measurement).
        target_rel_error: the caller's requested target, echoed back for convenience.
        status: see above.
    """

    value: float
    backend: str
    rel_error_bound: float
    target_rel_error: float
    status: str = "ok"

    @property
    def met_target(self) -> bool:
        """``True`` only when ``rel_error_bound`` is certified to meet ``target_rel_error``."""
        return self.status == "ok"


def _condition_number(abs_sum: float, s_dd: float) -> float:
    """Return the summation's condition number ``sum|x_i| / |sum x_i|``, saturated at the largest finite
    ``float64`` instead of overflowing to ``inf``.

    Exact (or near-exact) cancellation sends ``s_dd`` toward 0 and this ratio toward the true mathematical
    +inf; in float64 arithmetic the division itself silently overflows past ~1.8e308 to ``inf`` well
    before that -- confirmed even for six unit-magnitude terms that cancel exactly (``abs_sum == 6.0``):
    once ``s_dd`` underflows below the smallest normal double, ``6.0 / s_dd`` already exceeds the largest
    finite double. An infinite condition number then poisons every downstream ``log2`` / bit-count
    computation in :func:`accurate_sum` -- ``math.ceil(inf)`` raises ``OverflowError`` rather than
    returning a value, surfacing as a crash that on its face looks unrelated to "this sum needs more
    precision than is available." Saturating here instead keeps ``cond`` a legitimate finite float for the
    rest of the pipeline, so that outcome is instead handled by the normal insufficient-precision path.
    """
    if s_dd <= _TINY:
        if abs_sum <= _TINY:
            return 0.0  # everything involved is ~0: no cancellation signal to speak of
        return float(_FLOAT64_MAX)
    with np.errstate(over="ignore"):  # overflow is handled by the clamp below, not left to leak out
        raw = abs_sum / s_dd
    return min(raw, float(_FLOAT64_MAX))


def accurate_sum(x: Any, target_rel_error: float = 1e-12) -> SumResult:
    """Sum ``x`` to ``target_rel_error`` relative accuracy using the lowest-cost sufficient backend.

    Returns a :class:`SumResult` certifying the accuracy actually achieved. Escalates from float64 to
    double-double to arbitrary precision only when the certified error bound says the lower-cost backend
    cannot meet the budget -- so the common well-conditioned case never leaves vectorized float64 -- and
    fails closed (``status="insufficient"``) rather than silently returning a value that does not meet
    ``target_rel_error`` when no backend available in this process can certify it.

    Raises ``ValueError`` if ``target_rel_error`` is not a positive, finite number.
    """
    if not (math.isfinite(target_rel_error) and target_rel_error > 0.0):
        raise ValueError("target_rel_error must be a positive, finite number; got %r." % (target_rel_error,))

    arr = np.asarray(x, dtype=np.float64).ravel()
    if arr.size == 0:
        return SumResult(0.0, "float64", 0.0, target_rel_error, "ok")

    if float64_sum_is_accurate(arr, target_rel_error):
        s0 = abs(float(np.sum(arr)))
        rel0 = sum_error_bound(arr) / max(s0, _TINY)
        return SumResult(float(np.sum(arr)), "float64", rel0, target_rel_error, "ok")

    dd = dd_sum(arr)
    s_dd = abs(float(dd.to_float()))
    abs_sum = float(np.abs(arr).sum())
    cond = _condition_number(abs_sum, s_dd)
    dd_rel_error = cond * _U_DD
    if dd_rel_error <= target_rel_error:
        return SumResult(float(dd.to_float()), "dd", dd_rel_error, target_rel_error, "ok")

    # Arbitrary precision: enough mantissa bits to cover the conditioning and the target.
    from mixle.engines.highprec import available, hp_sum

    if not available():  # gmpy2/mpmath both absent -- dd is the best available, and it was just proven
        # insufficient above: report that honestly instead of returning it as if it met the target.
        return SumResult(float(dd.to_float()), "dd", dd_rel_error, target_rel_error, "insufficient")

    # Clamp the *estimate* itself (not just its float->int conversion) at the cap, so a saturated
    # (formerly infinite) `cond` can never reach a conversion that could raise OverflowError.
    bits_needed = math.log2(max(cond, 1.0)) - math.log2(target_rel_error) + 16
    if bits_needed > _MAX_MPFR_BITS:
        bits = _MAX_MPFR_BITS
    else:
        bits = max(_MIN_MPFR_BITS, int(math.ceil(bits_needed)))
    value = hp_sum(arr, bits)
    # At very large `bits` (only reachable via a saturated `cond`, i.e. exact/near-exact cancellation)
    # this product can itself underflow to a flat 0.0 rather than the true tiny positive value -- but
    # never falsely: target_rel_error is itself a positive float64, so it is never smaller than the
    # smallest representable value this product could underflow *from* while still being genuinely
    # <= target. Underflow can therefore only land on cases that were already unambiguously "ok".
    mpfr_rel_error = cond * (2.0 ** -(bits + 1))
    status = "ok" if mpfr_rel_error <= target_rel_error else "insufficient"
    return SumResult(value, "mpfr%d" % bits, mpfr_rel_error, target_rel_error, status)


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
