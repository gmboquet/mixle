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
_MAX_EXACT_INT = 2**53  # above this an integer stops being exactly representable in float64
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
    * ``"overflow"``: the exact/high-precision result cannot be represented by this API's float64
      ``value`` field. No relative-error claim is issued.

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


def _native_exact_float64(x: Any) -> np.ndarray | None:
    """Return ``x`` as a flat ``float64`` array when its *native* dtype already carries every input bit
    exactly, else ``None`` (meaning: route through the evidence-preserving object path).

    :func:`accurate_sum` only needs one bit of information about its inputs -- can a ``float64``
    accumulator hold them without discarding source evidence? For an array that is already a native
    numeric dtype, the dtype answers that for every element at once, and the two value-dependent parts
    (integers past the 53-bit exact range, non-finite floats) are single vectorized reductions. Deciding
    it per element instead -- boxing each value into a Python object and re-deriving its type and width
    one at a time -- measured at ~79% of ``accurate_sum``'s runtime on plain ``float64`` input, putting
    the module's own fast path in the ~1000x-slower-than-float64 regime it exists to avoid. An
    ``object`` dtype (``Decimal``, strings, over-wide integers, backend-native scalars) still gets the
    per-element treatment, because there the element type genuinely varies.
    """
    try:
        arr = np.asarray(x)
    except (TypeError, ValueError, OverflowError):
        return None  # e.g. integers too large for any native dtype: keep them as objects
    if arr.dtype == object:
        return None
    kind = arr.dtype.kind
    if kind == "b":
        return arr.ravel().astype(np.float64)
    if kind in "iu":
        flat = arr.ravel()
        if flat.size and (int(flat.min()) < -_MAX_EXACT_INT or int(flat.max()) > _MAX_EXACT_INT):
            return None  # float64 would silently drop low bits; the high-precision path keeps them
        return flat.astype(np.float64)
    if kind == "f" and arr.dtype.itemsize <= 8:  # float16/32/64 widen to float64 exactly; not longdouble
        flat = arr.ravel().astype(np.float64, copy=False)
        if flat.size and not np.all(np.isfinite(flat)):
            return None  # non-finite: let the object path decide between "overflow" and a hard error
        return flat
    return None  # complex, datetime, strings, longdouble: not float64-exact numerics


def _preserved_evidence_sum(raw: np.ndarray, target_rel_error: float) -> SumResult:
    """Sum an object array whose elements are not uniformly float64-exact, keeping the source scalars
    (``Decimal``/string/large-integer/backend-native) intact until the selected accumulator."""
    if raw.size == 0:
        return SumResult(0.0, "float64", 0.0, target_rel_error, "ok")
    try:
        approx = np.array([float(value) for value in raw], dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("sum inputs must be finite numeric values") from None
    if np.any(np.isnan(approx)):
        raise ValueError("sum inputs must be finite numeric values")
    with np.errstate(over="ignore", invalid="ignore"):
        approx_sum = abs(float(np.sum(approx)))
        approx_abs_sum = float(np.abs(approx).sum())
    cond = (
        float(_FLOAT64_MAX)
        if not math.isfinite(approx_sum) or not math.isfinite(approx_abs_sum)
        else _condition_number(approx_abs_sum, approx_sum)
    )
    return _high_precision_sum(raw, cond, target_rel_error)


def _exact_infinite_sum(x: Any, target_rel_error: float) -> SumResult | None:
    """Return the exact sum when infinities alone decide it, else None.

    The log density of a zero-probability event is -inf, and adding finite terms to it is exactly
    -inf. That is not an overflow and no amount of extra precision improves on it, but routing it
    through the escalation path both paid for an arbitrary-precision accumulator and labelled an
    exact answer ``status="overflow"``. Only a one-signed infinity is resolvable this way: mixed
    signs are genuinely indeterminate, and a NaN input is an error, so both stay on the old path.
    """
    try:
        arr = np.asarray(x)
    except (TypeError, ValueError, OverflowError):
        return None
    if arr.dtype.kind != "f" or arr.dtype.itemsize > 8:  # object/int/longdouble keep their own paths
        return None
    flat = arr.ravel()
    if not flat.size or np.any(np.isnan(flat)):
        return None
    has_positive = bool(np.any(np.isposinf(flat)))
    has_negative = bool(np.any(np.isneginf(flat)))
    if has_positive == has_negative:  # neither (ordinary finite sum) or both (indeterminate)
        return None
    return SumResult(math.inf if has_positive else -math.inf, "float64", 0.0, target_rel_error, "ok")


def accurate_sum(x: Any, target_rel_error: float = 1e-12) -> SumResult:
    """Sum ``x`` to ``target_rel_error`` relative accuracy using the lowest-cost sufficient backend.

    Returns a :class:`SumResult` certifying the accuracy actually achieved. Escalates from float64 to
    double-double to arbitrary precision only when the certified error bound says the lower-cost backend
    cannot meet the budget -- so the common well-conditioned case never leaves vectorized float64 -- and
    fails closed (``status="insufficient"``) rather than silently returning a value that does not meet
    ``target_rel_error`` when no backend available in this process can certify it.

    Raises ``ValueError`` if ``target_rel_error`` is not a positive, finite number.
    """
    if isinstance(target_rel_error, (bool, np.bool_)) or not (
        math.isfinite(target_rel_error) and target_rel_error > 0.0
    ):
        raise ValueError("target_rel_error must be a positive, finite number; got %r." % (target_rel_error,))

    resolved = _exact_infinite_sum(x, target_rel_error)
    if resolved is not None:
        return resolved

    arr = _native_exact_float64(x)
    if arr is None:
        return _preserved_evidence_sum(np.asarray(x, dtype=object).ravel(), target_rel_error)
    if arr.size == 0:
        return SumResult(0.0, "float64", 0.0, target_rel_error, "ok")

    if float64_sum_is_accurate(arr, target_rel_error):
        s0_value = float(np.sum(arr))
        s0 = abs(s0_value)
        rel0 = sum_error_bound(arr) / max(s0, _TINY)
        return SumResult(s0_value, "float64", rel0, target_rel_error, "ok")

    with np.errstate(over="ignore", invalid="ignore"):
        abs_sum = float(np.abs(arr).sum())
    if not math.isfinite(abs_sum):
        return _high_precision_sum(arr, float(_FLOAT64_MAX), target_rel_error)

    with np.errstate(over="ignore", invalid="ignore"):
        dd = dd_sum(arr)
        dd_value = float(dd.to_float())
    if not math.isfinite(dd_value) or not np.all(np.isfinite(dd.hi)) or not np.all(np.isfinite(dd.lo)):
        return _high_precision_sum(arr, float(_FLOAT64_MAX), target_rel_error)
    s_dd = abs(dd_value)
    cond = _condition_number(abs_sum, s_dd)
    dd_rel_error = cond * _U_DD
    if dd_rel_error <= target_rel_error:
        return SumResult(dd_value, "dd", dd_rel_error, target_rel_error, "ok")
    result = _high_precision_sum(arr, cond, target_rel_error)
    if result.backend == "unavailable":
        return SumResult(dd_value, "dd", dd_rel_error, target_rel_error, "insufficient")
    return result


def _high_precision_sum(raw: np.ndarray, cond: float, target_rel_error: float) -> SumResult:
    """Accumulate preserved source scalars without a float64 conversion boundary."""
    from mixle.engines.highprec import available, hp_sum

    if not available():
        return SumResult(math.nan, "unavailable", math.inf, target_rel_error, "insufficient")

    bits_needed = math.log2(max(cond, 1.0)) - math.log2(target_rel_error) + 16
    if bits_needed > _MAX_MPFR_BITS:
        bits = _MAX_MPFR_BITS
    else:
        bits = max(_MIN_MPFR_BITS, int(math.ceil(bits_needed)))
    value = hp_sum(raw, bits)
    if not math.isfinite(value):
        return SumResult(value, "mpfr%d" % bits, math.inf, target_rel_error, "overflow")
    # At very large `bits` (only reachable via a saturated `cond`, i.e. exact/near-exact cancellation)
    # this product can itself underflow to a flat 0.0 rather than the true tiny positive value -- but
    # never falsely: target_rel_error is itself a positive float64, so it is never smaller than the
    # smallest representable value this product could underflow *from* while still being genuinely
    # <= target. Underflow can therefore only land on cases that were already unambiguously "ok".
    mpfr_rel_error = cond * (2.0 ** -(bits + 1))
    status = "ok" if mpfr_rel_error <= target_rel_error else "insufficient"
    return SumResult(value, "mpfr%d" % bits, mpfr_rel_error, target_rel_error, status)


def sum_certificate(x: Any) -> dict[str, float | str | bool]:
    """Report the certified float64 summation error and the condition number, without choosing a backend."""
    arr = np.asarray(x, dtype=np.float64).ravel()
    if arr.size and not np.all(np.isfinite(arr)):
        raise ValueError("sum certificates require finite inputs")
    with np.errstate(over="ignore", invalid="ignore"):
        value = float(np.sum(arr))
        s = abs(value)
        abs_sum = float(np.abs(arr).sum())
    bound = sum_error_bound(arr)
    if not math.isfinite(value) or not math.isfinite(bound) or not math.isfinite(abs_sum):
        return {
            "float64_value": value,
            "abs_error_bound": math.inf,
            "rel_error_bound": math.inf,
            "condition_number": math.inf,
            "status": "overflow_or_unbounded",
            "certified": False,
        }
    return {
        "float64_value": value,
        "abs_error_bound": bound,
        "rel_error_bound": bound / max(s, _TINY),
        "condition_number": abs_sum / max(s, _TINY),
        "status": "ok",
        "certified": True,
    }


_DD_MAX_BITS = 128  # "fp128" is DoubleDouble's own label (~106 mantissa bits + labelled overhead);
# mixle.engines.extended has no wider error-free-transform format yet (its module docstring names
# quad-double/fp256 as unimplemented future work), so this is where MPFR actually takes over today.
_NATIVE_DTYPE_BITS = (16, 32, 64)


def _precision_bits(precision: str | int) -> int:
    """Resolve a precision spelling to its canonical *total* bit width (sign + exponent + mantissa,
    IEEE-754-style -- matching :class:`~mixle.engines.formats.FloatFormat` and
    :class:`~mixle.engines.highprec.HighPrecisionFormat`'s own ``"fpN"`` naming).

    Accepts ``"dd"`` (an alias for ``"fp128"``), a ``"fp<bits>"`` string, or a bare integer bit count --
    the single source of truth every spelling normalizes through before :func:`cast` picks a
    representation, so ``"fp96"`` and the integer ``96`` always mean the identical width.

    Raises ``ValueError`` if the spelling does not resolve to a positive integer bit count.
    """
    if precision == "dd":
        bits = _DD_MAX_BITS
    elif isinstance(precision, str):  # "fp<bits>", e.g. "fp96"
        digits = precision[2:]
        try:
            bits = int(digits)
        except ValueError:
            raise ValueError(
                "precision string %r is not a valid 'fp<bits>' spelling (e.g. 'fp96')." % (precision,)
            ) from None
    else:
        bits = precision
    if isinstance(bits, (bool, np.bool_)) or not isinstance(bits, (int, np.integer)):
        raise ValueError("precision must resolve to an exact non-Boolean integer bit width")
    bits = int(bits)
    if bits < 1:
        raise ValueError("precision must resolve to a positive bit width; %r resolved to %d bits." % (precision, bits))
    return bits


def _cast_by_bits(x: Any, bits: int) -> Any:
    """Route a canonical total bit width to the one tiering policy every precision spelling shares:
    a native numpy float for ``bits in (16, 32, 64)``; a simulated low-bit float
    (:class:`~mixle.engines.formats.FloatFormat`) for any other ``bits <= 64``; a
    :class:`~mixle.engines.extended.DoubleDouble` for ``bits <= _DD_MAX_BITS``; an MPFR object array
    (:class:`~mixle.engines.highprec.HighPrecisionFormat`) beyond that.
    """
    if bits in _NATIVE_DTYPE_BITS:
        return np.asarray(x, dtype="float%d" % bits)
    if bits <= 64:
        return _native_round(x, bits)
    if bits <= _DD_MAX_BITS:
        return DoubleDouble.from_float(np.asarray(x, dtype=np.float64))
    return _mpfr_cast(x, bits)


def cast(x: Any, precision: Any) -> Any:
    """Cast ``x`` onto the spectrum: a native dtype name, ``"dd"``/``"fp<bits>"``, or an integer bit width.

    ``"dd"``, ``"fp128"``, and the integer ``128`` are the same request (:class:`DoubleDouble`); ``"fp96"``
    and the integer ``96`` are likewise the same request (both resolve to 96 total bits via
    :func:`_precision_bits`) and route to the identical representation via :func:`_cast_by_bits`,
    regardless of which spelling the caller used. A nonpositive bit width (e.g. ``"fp0"``, ``-5``) raises
    ``ValueError`` rather than reaching format construction. Anything else (e.g. ``"float64"``,
    ``np.float32``) is handed straight to :func:`numpy.dtype`.

    Returns a numpy array (native / simulated low-bit), a :class:`~mixle.engines.extended.DoubleDouble`,
    or an MPFR object array -- see :func:`_cast_by_bits` for the exact tiering.
    """
    if isinstance(precision, str) and (precision == "dd" or precision.startswith("fp")):
        return _cast_by_bits(x, _precision_bits(precision))
    if isinstance(precision, (int, np.integer)) and not isinstance(precision, (bool, np.bool_)):
        return _cast_by_bits(x, _precision_bits(precision))
    if isinstance(precision, (bool, np.bool_)):
        raise ValueError("precision must not be Boolean")
    if isinstance(precision, (float, np.floating)):
        raise ValueError("numeric precision widths must be exact integers")
    return np.asarray(x, dtype=np.dtype(precision))


def _native_round(x: Any, bits: int):
    from mixle.engines.formats import FloatFormat

    return FloatFormat.fp(bits).round_trip(x)


def _mpfr_cast(x: Any, bits: int):
    from mixle.engines.highprec import HighPrecisionFormat

    return HighPrecisionFormat(bits).quantize(x)
