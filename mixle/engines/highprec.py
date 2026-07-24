"""Arbitrary-precision tail of the spectrum: fp128, fp256, fp512, fp1024, ... fp(any bits).

The pure-numpy error-free-transform path (:mod:`mixle.engines.extended`) tops out near double-double
(fp128) / quad-double (fp256); beyond that the renormalization cost grows and MPFR becomes the practical
compute backend. This module is the correct backend for that tail, on gmpy2 (C-backed MPFR) with an mpmath
fallback. Cost note: gmpy2 is per-object, so array ops are an O(N) Python loop -- correct but not fast. For
fp <= 256 prefer the vectorized ``extended`` path.

So: spectrum coverage is complete (fp1..fp1024+), with the fast pure-numpy backends below fp256 and the
correct MPFR backend above it.

Both backends are optional extras (neither is a base dependency, per worklist P2.2): install the faster
one with ``pip install mixle[gmpy2]``, or the pure-Python fallback with ``pip install mixle[highprec]``.
With neither installed, :func:`available` returns False and any call requiring fp>256 raises a clear
``RuntimeError`` naming the ``extended`` fallback -- see :func:`_require`.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import numpy as np

try:  # gmpy2 (MPFR) is the preferred backend (pip install mixle[gmpy2]).
    import gmpy2

    _BACKEND = "gmpy2"
except ImportError:  # pragma: no cover - environment dependent
    try:  # mpmath is the pure-Python fallback (pip install mixle[highprec]).
        import mpmath

        _BACKEND = "mpmath"
    except ImportError:  # pragma: no cover
        _BACKEND = None


def available() -> bool:
    """True if an arbitrary-precision backend (gmpy2 or mpmath) is importable."""
    return _BACKEND is not None


def _require() -> None:
    if _BACKEND is None:  # pragma: no cover
        raise RuntimeError(
            "arbitrary precision (fp>256) needs gmpy2 or mpmath installed; the fast pure-numpy path "
            "(mixle.engines.extended) covers up to fp256."
        )


def _scalar_to_hp(v: Any, bits: int) -> Any:
    """Convert one scalar to a ``bits``-bit backend-native high-precision number.

    Dispatches on the *input's own type* so callers keep whatever precision they actually brought:
    strings and :class:`~decimal.Decimal` are parsed directly by the backend's own arbitrary-precision
    parser (exact decimal -> binary conversion, correctly rounded to ``bits`` bits -- no float64 stop
    along the way), Python/numpy integers hand off directly (already arbitrary precision, and correctly
    rounded by the backend if ``bits`` is narrower than the integer's own bit length), and an existing
    backend-native number (``gmpy2.mpfr`` / ``mpmath.mpf``) is re-rounded -- widened or narrowed --
    straight to ``bits`` without any intermediate stop.

    The one input that *does* go through ``float()`` is an ordinary Python/numpy float: it never had more
    than float64 precision to begin with, so the conversion preserves its exact dyadic value bit-for-bit
    (verified in the test suite against ``Decimal(float_value)``) rather than "fixing it up" to the
    nearest tidy decimal -- that fix-up is exactly what a naive ``Decimal(str(x))`` round-trip would do,
    and would be wrong: it changes the value being represented instead of just extending its precision.
    """
    if _BACKEND == "gmpy2":
        if isinstance(v, (gmpy2.mpfr, gmpy2.mpz)):
            return gmpy2.mpfr(v, precision=bits)
        if isinstance(v, (int, np.integer)):
            return gmpy2.mpfr(int(v), precision=bits)
        if isinstance(v, Decimal):
            return gmpy2.mpfr(str(v), precision=bits)  # exact string round-trip, no float64 detour
        if isinstance(v, str):
            return gmpy2.mpfr(v, precision=bits)
        if isinstance(v, (float, np.floating)):
            return gmpy2.mpfr(float(v), precision=bits)  # exact: float64 -> MPFR never loses bits
        raise TypeError(f"cannot convert {type(v).__name__!r} to a high-precision number")
    # pragma: no cover - mpmath fallback path
    with mpmath.workprec(bits):
        if isinstance(v, mpmath.mpf):
            return mpmath.mpf(v)
        if isinstance(v, (int, np.integer)):
            return mpmath.mpf(int(v))
        if isinstance(v, Decimal):
            return mpmath.mpf(str(v))
        if isinstance(v, str):
            return mpmath.mpf(v)
        if isinstance(v, (float, np.floating)):
            return mpmath.mpf(float(v))
        raise TypeError(f"cannot convert {type(v).__name__!r} to a high-precision number")


def hp_array(x: Any, bits: int) -> np.ndarray:
    """Convert values to an object array of ``bits``-bit arbitrary-precision numbers.

    Accepts float/int/str/:class:`~decimal.Decimal`/backend-native (MPFR) scalars, arrays, or nested
    sequences of them -- each element is converted straight into the high-precision backend via
    :func:`_scalar_to_hp`, without ever routing through a float64 intermediary first. That matters: a
    ``Decimal`` or numeric string can carry far more significant digits than float64's ~15-17, and an
    integer beyond ``2**53`` is not exactly representable as float64 either -- casting the whole input to
    float64 up front (the previous bug here) would throw that precision away before it ever reached the
    high-precision backend, no matter how many ``bits`` were then requested.
    """
    _require()
    arr = np.asarray(x, dtype=object)
    flat = [_scalar_to_hp(v, bits) for v in arr.ravel()]
    return np.array(flat, dtype=object).reshape(arr.shape)


def hp_to_float(obj: Any) -> np.ndarray:
    """Round an arbitrary-precision object array back to ``float64``."""
    flat = np.asarray(obj, dtype=object).ravel()
    return np.array([float(v) for v in flat], dtype=np.float64).reshape(np.asarray(obj).shape)


def hp_sum(x: Any, bits: int) -> float:
    """Sum values at ``bits`` mantissa precision (correct beyond what float64 / double-double give).

    Accepts the same input types as :func:`hp_array` (float/int/str/``Decimal``/backend-native scalars,
    arrays, or nested sequences) -- each element is converted via :func:`_scalar_to_hp`, without a float64
    detour, and the accumulation itself runs in ``bits``-bit backend arithmetic (``gmpy2`` context / MPFR
    adds, or ``mpmath.fsum`` under ``workprec``), not float64 -- so a catastrophic cancellation that would
    erase a value's distinguishing digits under naive float64 summation is preserved all the way through.

    O(N) per-object MPFR adds -- correct but not vectorized; for large N below fp256 prefer
    :func:`mixle.engines.extended.dd_sum`. Returns the float64-rounded result.
    """
    _require()
    flat = np.asarray(x, dtype=object).ravel()
    if _BACKEND == "gmpy2":
        with gmpy2.context(precision=bits):
            acc = gmpy2.mpfr(0, precision=bits)
            for v in flat:
                acc = acc + _scalar_to_hp(v, bits)
            return float(acc)
    with mpmath.workprec(bits):  # pragma: no cover - fallback path
        return float(mpmath.fsum(_scalar_to_hp(v, bits) for v in flat))


class HighPrecisionFormat:
    """An arbitrary ``bits``-mantissa float (fp128, fp256, fp512, fp1024, ...) -- MPFR-backed codec.

    Round-trips a float64 array losslessly (its 52 bits fit), and represents *more* than float64 when
    fed exact/high-precision values. ``max_rel_error == 2**-bits``.
    """

    def __init__(self, bits: int) -> None:
        if bits < 1:
            raise ValueError("bits must be >= 1")
        self.bits = int(bits)
        self.name = "fp%d" % (self.bits + 12)  # ~ exponent+sign overhead, for a readable label
        self.mantissa_bits = self.bits

    @property
    def max_rel_error(self) -> float:
        """Return the nominal relative error bound for the mantissa budget."""
        return 2.0 ** -(self.bits + 1)

    def quantize(self, x: Any) -> np.ndarray:
        """Encode values with the configured high-precision mantissa."""
        return hp_array(x, self.bits)

    def dequantize(self, q: Any) -> np.ndarray:
        """Decode high-precision values to float64."""
        return hp_to_float(q)

    def round_trip(self, x: Any) -> np.ndarray:
        """Quantize and decode values through the high-precision format."""
        return self.dequantize(self.quantize(x))
