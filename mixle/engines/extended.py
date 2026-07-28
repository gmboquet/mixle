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

The Veltkamp split is power-of-two scaled for large operands, so every finite representable product
either produces a finite error-free pair or fails closed when the exact result cannot be represented.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mixle.engines._optional_extension import load_optional_extension

# Veltkamp splitting factor for IEEE double (53-bit significand): 2**ceil(53/2) + 1.
_SPLITTER = float(2**27 + 1)
_SPLIT_SCALE = 2.0**-28
_SPLIT_THRESHOLD = np.finfo(np.float64).max / _SPLITTER


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
    a = np.asarray(a, dtype=np.float64)
    scale = np.where(np.abs(a) > _SPLIT_THRESHOLD, _SPLIT_SCALE, 1.0)
    scaled = a * scale
    c = _SPLITTER * scaled
    abig = c - scaled
    hi = (c - abig) / scale
    lo = (scaled - (c - abig)) / scale
    return hi, lo


def _product_and_residual(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Elementwise ``(p, e)`` with ``a * b == p + e`` exactly, for already-validated finite operands.

    Overflow fails closed here because an exact product past ``float64``'s range cannot be represented
    in *any* pair of doubles. Underflow deliberately is not decided here: whether a product that rounds
    to zero matters depends on what the caller does with it (see :func:`two_prod` versus :func:`dd_dot`).
    """
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        p = a * b
    if p.size and not np.all(np.isfinite(p)):
        raise OverflowError("double-double product overflowed float64")
    ahi, alo = _split(a)
    bhi, blo = _split(b)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        e = ((ahi * bhi - p) + ahi * blo + alo * bhi) + alo * blo
    if e.size and not np.all(np.isfinite(e)):
        raise ArithmeticError("double-double product transform produced a non-finite residual")
    return p, e


def two_prod(a: Any, b: Any) -> tuple[Any, Any]:
    """Error-free transformation of a product: ``(p, e)`` with ``a * b == p + e`` exactly (Dekker).

    Uses the Veltkamp split because numpy exposes no fused-multiply-add. Vectorized.

    A product of two non-zero operands that underflows all the way to zero fails closed: this
    transform's contract is that ``p + e`` reproduces ``a * b`` exactly, and ``(0, 0)`` does not. A
    *reduction* over many such products is a different question -- see :func:`dd_dot`.
    """
    a, b = np.broadcast_arrays(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))
    if (a.size and not np.all(np.isfinite(a))) or (b.size and not np.all(np.isfinite(b))):
        raise ValueError("two_prod requires finite inputs")
    p, e = _product_and_residual(a, b)
    if p.size and np.any((a != 0) & (b != 0) & (p == 0)):
        raise ArithmeticError("double-double product underflow cannot be represented")
    return p, e


def _resolve_nonfinite(
    hi_arr: np.ndarray, lo_arr: np.ndarray, s: np.ndarray, e: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Classify a normalization whose leading component came out non-finite, and either return the
    representable ``(hi, lo)`` or fail closed. Off the hot path: reached only when the single finiteness
    check in :meth:`DoubleDouble.__init__` has already tripped.

    An infinity that was *already* in the input is a legitimate exactly-representable ``float64`` value
    -- it is the log-density of a zero-probability event, the very thing this module's reductions exist
    to carry through log-sum-exp -- and it pairs with a zero rounding error. An infinity that appeared
    during normalization is an overflow of finite operands, and a NaN (including ``+inf`` paired with
    ``-inf``) is not a number at all; both of those still fail closed.
    """
    if np.any(np.isnan(s)):
        raise ValueError("DoubleDouble components must not be NaN, and +inf cannot be paired with -inf")
    infinite = np.isinf(s)
    if np.any(infinite & ~(np.isinf(hi_arr) | np.isinf(lo_arr))):
        raise OverflowError("DoubleDouble components cannot be normalized to a finite value")
    return np.array(s, dtype=np.float64), np.where(infinite, 0.0, e).astype(np.float64, copy=False)


class DoubleDouble:
    """An (almost) ~106-bit float as two non-overlapping ``float64`` arrays ``hi + lo``.

    Scalars or numpy arrays; operations broadcast. The invariant is ``|lo| <= 0.5 * ulp(hi)``. A
    component may be ``+inf`` or ``-inf`` -- exactly representable in ``float64``, carried with
    ``lo == 0`` -- but never NaN, and never an overflow of finite operands.
    """

    __slots__ = ("hi", "lo")

    def __init__(self, hi: Any, lo: Any = 0.0) -> None:
        # One finiteness pass, on the normalized leading component only. That single check subsumes the
        # four this used to run (two over the inputs, two over the outputs): ``s = hi + lo`` is
        # non-finite whenever either input is non-finite *or* the sum overflows, and TwoSum's residual
        # is provably representable whenever its inputs and its sum are all finite. Everything else is
        # deferred to _resolve_nonfinite, which only runs when that check trips -- this constructor is
        # on the hot path, called by every __add__ and __mul__.
        hi_arr = np.asarray(hi, dtype=np.float64)
        lo_arr = np.asarray(lo, dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            s, e = two_sum(hi_arr, lo_arr)
        # two_sum allocates both of its results, so they are already this object's own storage rather
        # than a view onto the caller's arrays; asarray here only re-wraps the 0-d scalar case, and no
        # defensive copy is needed to keep a later mutation of the caller's array out of this value.
        owned_hi = np.asarray(s, dtype=np.float64)
        owned_lo = np.asarray(e, dtype=np.float64)
        if owned_hi.size and not np.all(np.isfinite(owned_hi)):
            owned_hi, owned_lo = _resolve_nonfinite(hi_arr, lo_arr, owned_hi, owned_lo)
        owned_hi.setflags(write=False)
        owned_lo.setflags(write=False)
        self.hi = owned_hi
        self.lo = owned_lo

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


def _infinite_sum(x: np.ndarray) -> DoubleDouble:
    """The sum of an input containing an infinity -- or a hard failure when there is no such sum.

    ``-inf`` is the log-density of a zero-probability event, and this module exists for mixle's EM hot
    paths (log-sum-exp), which is precisely where it shows up; rejecting it rejected the inputs the
    reduction was written to serve. The sum of a set containing ``-inf`` and no ``+inf`` *is* ``-inf``:
    an exactly representable ``float64`` with a zero rounding error, not an error condition. NaN has no
    sum, and ``+inf`` together with ``-inf`` is genuinely indeterminate; both still fail closed.
    """
    if np.any(np.isnan(x)):
        raise ValueError("dd_sum requires inputs free of NaN")
    positive = bool(np.any(np.isposinf(x)))
    if positive and bool(np.any(np.isneginf(x))):
        raise ValueError("dd_sum cannot sum both +inf and -inf: the result is indeterminate")
    return DoubleDouble(np.float64(np.inf if positive else -np.inf), np.float64(0.0))


def dd_sum(x: Any) -> DoubleDouble:
    """Accurate sum of a ``float64`` array in double-double precision -- vectorized, no Python loop
    over elements.

    Pairwise tree reduction with an error-free :func:`two_sum` combine at every node: ``O(n)`` work in
    ``O(log n)`` vectorized passes, accumulating the rounding error into the ``lo`` component. The result
    is correct to ~106 bits even for catastrophically cancelling inputs that ``float64`` sums get wrong.

    An infinite term short-circuits to the IEEE answer (``-inf`` for a zero-probability log-density)
    instead of raising; NaN, and ``+inf`` mixed with ``-inf``, still fail closed -- see
    :func:`_infinite_sum`.
    """
    hi = np.asarray(x, dtype=np.float64).ravel()
    if hi.size == 0:
        return DoubleDouble(0.0, 0.0)
    if not np.all(np.isfinite(hi)):
        return _infinite_sum(hi)
    hi = hi.copy()
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


_DD_EXTENSION = load_optional_extension("mixle.engines._dd_kernels", ("dd_dot_c",))
HAS_DD_KERNELS = _DD_EXTENSION.available
DD_EXTENSION_DIAGNOSTIC = _DD_EXTENSION.diagnostic
if HAS_DD_KERNELS:  # pragma: no cover - depends on the optional local build
    (_dd_dot_c,) = _DD_EXTENSION.values


def _require_equal_length(a: np.ndarray, b: np.ndarray) -> None:
    """Raise if raveled ``a`` and ``b`` don't have the same flattened length.

    A dot product between mismatched-length vectors is mathematically undefined. Left unchecked, numpy's
    elementwise ops broadcast the length-1 side instead -- a *different*, unrequested operation that still
    returns some number, silently. :func:`dd_dot` calls this once, before choosing an implementation, so
    the compiled kernel and the pure-numpy fallback can never independently drift on this check.
    """
    if a.size != b.size:
        raise ValueError(
            "dd_dot requires vectors of equal length, got sizes %d and %d (shapes %s and %s); a dot "
            "product between mismatched lengths is undefined -- it is not the same operation as "
            "elementwise broadcasting." % (a.size, b.size, a.shape, b.shape)
        )


def dd_dot(a: Any, b: Any) -> DoubleDouble:
    """Accurate dot product ``sum(a_i * b_i)`` in double-double precision.

    Uses the compiled hardware-FMA kernel when available (one ``fma`` per element, ~3x faster than the
    pure-numpy Veltkamp-split path, and exactly as accurate -- each product's error is exact either way,
    though the two paths accumulate in different orders, sequential versus pairwise tree, so their
    results agree to double-double precision rather than bit for bit). Otherwise each product is split
    error-free by :func:`_product_and_residual` and the products + errors are summed by :func:`dd_sum`.
    Defeats the cancellation that wrecks a naive ``float64`` dot.

    Every check that decides whether an *input* is acceptable is made once, before dispatch, so the two
    implementations cannot drift apart on what they accept: equal length (see
    :func:`_require_equal_length`, ``ValueError``), finite operands (``ValueError``), and a result that
    is representable at all (``OverflowError``). A single product that underflows to zero is *not* one
    of them: it contributes less than one smallest-subnormal to the sum, so it is summed as the zero it
    rounds to rather than failing the whole reduction -- ``dd_dot([1e-200, 1.0], [1e-200, 1.0])`` is
    ``1.0``, which is the right answer, not an error. (:func:`two_prod` on its own still fails closed
    there, because a lone product has nothing to be negligible against.)
    """
    a = np.ascontiguousarray(np.asarray(a, dtype=np.float64).ravel())
    b = np.ascontiguousarray(np.asarray(b, dtype=np.float64).ravel())
    _require_equal_length(a, b)
    if (a.size and not np.all(np.isfinite(a))) or (b.size and not np.all(np.isfinite(b))):
        raise ValueError("dd_dot requires finite inputs")
    if HAS_DD_KERNELS:
        raw_hi, raw_lo = _dd_dot_c(a, b)
        hi, lo = np.float64(raw_hi), np.float64(raw_lo)
    else:
        p, e = _product_and_residual(a, b)
        total = dd_sum(np.concatenate([p, e]))
        hi, lo = total.hi, total.lo
    if not (np.all(np.isfinite(hi)) and np.all(np.isfinite(lo))):
        # The numpy path fails closed inside the error-free transform; the compiled kernel's
        # accumulator carries the same failure out as a non-finite (hi, lo). Same outcome either way.
        raise OverflowError("double-double dot product overflowed float64")
    return DoubleDouble(hi, lo)
