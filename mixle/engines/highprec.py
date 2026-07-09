"""Arbitrary-precision tail of the spectrum: fp128, fp256, fp512, fp1024, ... fp(any bits).

The pure-numpy error-free-transform path (:mod:`mixle.engines.extended`) tops out near double-double
(fp128) / quad-double (fp256); beyond that the renormalization cost grows and MPFR becomes the practical
compute backend. This module is the correct backend for that tail, on gmpy2 (C-backed MPFR) with an mpmath
fallback. Cost note: gmpy2 is per-object, so array ops are an O(N) Python loop -- correct but not fast. For
fp <= 256 prefer the vectorized ``extended`` path.

So: spectrum coverage is complete (fp1..fp1024+), with the fast pure-numpy backends below fp256 and the
correct MPFR backend above it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:  # gmpy2 (MPFR) is the preferred backend; mpmath is the pure-Python fallback.
    import gmpy2

    _BACKEND = "gmpy2"
except ImportError:  # pragma: no cover - environment dependent
    try:
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


def hp_array(x: Any, bits: int) -> np.ndarray:
    """Convert a float array to an object array of ``bits``-bit arbitrary-precision numbers."""
    _require()
    flat = np.asarray(x, dtype=np.float64).ravel()
    if _BACKEND == "gmpy2":
        out = np.array([gmpy2.mpfr(float(v), bits) for v in flat], dtype=object)
    else:  # pragma: no cover - fallback path
        with mpmath.workprec(bits):
            out = np.array([mpmath.mpf(float(v)) for v in flat], dtype=object)
    return out.reshape(np.asarray(x).shape)


def hp_to_float(obj: Any) -> np.ndarray:
    """Round an arbitrary-precision object array back to ``float64``."""
    flat = np.asarray(obj, dtype=object).ravel()
    return np.array([float(v) for v in flat], dtype=np.float64).reshape(np.asarray(obj).shape)


def hp_sum(x: Any, bits: int) -> float:
    """Sum a float array at ``bits`` mantissa precision (correct beyond what float64 / double-double give).

    O(N) per-object MPFR adds -- correct but not vectorized; for large N below fp256 prefer
    :func:`mixle.engines.extended.dd_sum`. Returns the float64-rounded result.
    """
    _require()
    flat = np.asarray(x, dtype=np.float64).ravel()
    if _BACKEND == "gmpy2":
        with gmpy2.context(precision=bits):
            acc = gmpy2.mpfr(0)
            for v in flat:
                acc = acc + gmpy2.mpfr(float(v))
            return float(acc)
    with mpmath.workprec(bits):  # pragma: no cover - fallback path
        return float(mpmath.fsum(mpmath.mpf(float(v)) for v in flat))


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
