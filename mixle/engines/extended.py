"""Fast extended precision via error-free transformations -- the "big boy math" end of mixle's
numeric spectrum, with **no** mpmath/gmpy2 in the hot path.

mpmath and gmpy2 are per-object and non-vectorized (~1000x slower than ``float64`` at ~100-bit
precision); they belong in tests as a *correctness oracle*, never in compute. The fast way to exceed
``float64`` is **error-free transformations** (Dekker/Knuth ``TwoSum`` / ``TwoProd``): represent a number
as an unevaluated sum of ``float64`` components (``hi + lo``) and carry the rounding error explicitly.
Every operation is a handful of ``float64`` ops that **vectorize over numpy arrays**, so a double-double
(~106-bit mantissa, "fp128") costs ~5-25x a ``float64`` op -- versus ~1000x for mpmath at the same
precision.

This module provides the double-double primitives and the two reductions that matter for mixle's EM
hot paths -- an accurate sum and dot product -- where catastrophic cancellation (``E[x^2]-E[x]^2``,
log-sum-exp of near-equal terms) otherwise eats precision. Beyond double-double, quad-double / multi-limb
take over (a Cython/C job); this file is the part that needs only numpy.

Caveat: the Veltkamp split overflows for ``|x| > ~1e300``; mixle's log-densities are far from that.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Veltkamp splitting factor for IEEE double (53-bit significand): 2**ceil(53/2) + 1.
_SPLITTER = float(2**27 + 1)


def two_sum(a: Any, b: Any) -> tuple[Any, Any]:
    """Error-free transformation of a sum: returns ``(s, e)`` with ``a + b == s + e`` exactly.

    Knuth's TwoSum -- no assumption on the relative magnitudes of ``a`` and ``b``. Vectorized.
    """
    s = a + b
    bb = s - a
    e = (a - (s - bb)) + (b - bb)
    return s, e


def quick_two_sum(a: Any, b: Any) -> tuple[Any, Any]:
    """Error-free sum assuming ``|a| >= |b|`` (Dekker). One fewer op than :func:`two_sum`."""
    s = a + b
    e = b - (s - a)
    return s, e


def _split(a: Any) -> tuple[Any, Any]:
    """Veltkamp split: ``a == hi + lo`` with ``hi`` holding the top ~26 bits (exact, non-overlapping)."""
    c = _SPLITTER * a
    abig = c - a
    hi = c - abig
    lo = a - hi
    return hi, lo


def two_prod(a: Any, b: Any) -> tuple[Any, Any]:
    """Error-free transformation of a product: ``(p, e)`` with ``a * b == p + e`` exactly (Dekker).

    Uses the Veltkamp split because numpy exposes no fused-multiply-add. Vectorized.
    """
    p = a * b
    ahi, alo = _split(a)
    bhi, blo = _split(b)
    e = ((ahi * bhi - p) + ahi * blo + alo * bhi) + alo * blo
    return p, e


class DoubleDouble:
    """An (almost) ~106-bit float as two non-overlapping ``float64`` arrays ``hi + lo``.

    Scalars or numpy arrays; operations broadcast. The invariant is ``|lo| <= 0.5 * ulp(hi)``.
    """

    __slots__ = ("hi", "lo")

    def __init__(self, hi: Any, lo: Any = 0.0) -> None:
        self.hi = np.asarray(hi, dtype=np.float64)
        self.lo = np.asarray(lo, dtype=np.float64) + np.zeros_like(self.hi)

    @classmethod
    def from_float(cls, x: Any) -> DoubleDouble:
        """Create a double-double value from float64 data."""
        x = np.asarray(x, dtype=np.float64)
        return cls(x, np.zeros_like(x))

    def to_float(self) -> np.ndarray:
        """Collapse back to the nearest ``float64`` (``hi`` rounded with ``lo``)."""
        return self.hi + self.lo

    def __add__(self, other: DoubleDouble) -> DoubleDouble:
        # Dekker/HLB "sloppy" dd add: accurate to ~2**-104 relative, the standard fast variant.
        s, e = two_sum(self.hi, other.hi)
        e = e + (self.lo + other.lo)
        hi, lo = quick_two_sum(s, e)
        return DoubleDouble(hi, lo)

    def __sub__(self, other: DoubleDouble) -> DoubleDouble:
        return self + DoubleDouble(-other.hi, -other.lo)

    def __mul__(self, other: DoubleDouble) -> DoubleDouble:
        p1, p2 = two_prod(self.hi, other.hi)
        p2 = p2 + (self.hi * other.lo + self.lo * other.hi)
        hi, lo = quick_two_sum(p1, p2)
        return DoubleDouble(hi, lo)

    def __repr__(self) -> str:
        return "DoubleDouble(hi=%r, lo=%r)" % (self.hi, self.lo)


def dd_sum(x: Any) -> DoubleDouble:
    """Accurate sum of a ``float64`` array in double-double precision -- vectorized, no Python loop
    over elements.

    Pairwise tree reduction with an error-free :func:`two_sum` combine at every node: ``O(n)`` work in
    ``O(log n)`` vectorized passes, accumulating the rounding error into the ``lo`` component. The result
    is correct to ~106 bits even for catastrophically cancelling inputs that ``float64`` sums get wrong.
    """
    hi = np.asarray(x, dtype=np.float64).ravel().copy()
    if hi.size == 0:
        return DoubleDouble(0.0, 0.0)
    lo = np.zeros_like(hi)
    while hi.size > 1:
        if hi.size % 2 == 1:  # carry the odd tail element unchanged into the next level
            carry_hi, carry_lo = hi[-1:], lo[-1:]
            hi, lo = hi[:-1], lo[:-1]
        else:
            carry_hi = carry_lo = None
        a_hi, b_hi = hi[0::2], hi[1::2]
        a_lo, b_lo = lo[0::2], lo[1::2]
        s, e = two_sum(a_hi, b_hi)
        e = e + (a_lo + b_lo)
        hi, lo = quick_two_sum(s, e)
        if carry_hi is not None:
            hi = np.concatenate([hi, carry_hi])
            lo = np.concatenate([lo, carry_lo])
    return DoubleDouble(hi[0], lo[0])


try:  # optional compiled FMA kernel -- ~3x on the dot (numpy has no FMA, pays the Veltkamp split)
    from mixle.engines._dd_kernels import dd_dot_c as _dd_dot_c

    HAS_DD_KERNELS = True
except ImportError:  # pragma: no cover - package works fine without the compiled accelerator
    HAS_DD_KERNELS = False


def dd_dot(a: Any, b: Any) -> DoubleDouble:
    """Accurate dot product ``sum(a_i * b_i)`` in double-double precision.

    Uses the compiled hardware-FMA kernel when available (one ``fma`` per element, ~3x faster than the
    pure-numpy Veltkamp-split path and bit-for-bit identical); otherwise each product is split error-free
    by :func:`two_prod` and the products + errors are summed by :func:`dd_sum`. Defeats the cancellation
    that wrecks a naive ``float64`` dot.
    """
    a = np.ascontiguousarray(np.asarray(a, dtype=np.float64).ravel())
    b = np.ascontiguousarray(np.asarray(b, dtype=np.float64).ravel())
    if HAS_DD_KERNELS and a.size == b.size:
        hi, lo = _dd_dot_c(a, b)
        return DoubleDouble(np.float64(hi), np.float64(lo))
    p, e = two_prod(a, b)
    return dd_sum(np.concatenate([p, e]))
