"""Sound error tracing via interval arithmetic.

An :class:`Interval` carries ``[lo, hi]`` enclosing a true value. Every operation rounds the bounds
*outward* (one ULP via :func:`numpy.nextafter`), so the enclosure provably contains the exact result
despite float64 round-off -- the interval *certifies* the numerical error rather than hoping it is small.
The width is a guaranteed error bound; a precision-allocation pass reads it to pick the lowest-cost format
that keeps the width under a target (pair with :func:`mixle.engines.formats.min_float_mantissa_bits`).

Interval arithmetic is sound but pessimistic because it ignores correlations
between operands. Affine arithmetic can tighten the bound when that extra
complexity is justified. This module provides the vectorized, dependency-free
core.

Soundness invariants, enforced at construction (:meth:`Interval.__init__`): endpoints are never NaN
(a NaN bound cannot certify anything) and ``lo <= hi`` always holds (``+/-inf`` endpoints are legal --
``[-inf, inf]`` and degenerate points like ``[inf, inf]`` both pass -- only NaN and reversed order are
rejected). Every operation is written to either produce a result that satisfies those invariants or to
raise, rather than let IEEE-754's indeterminate forms (``0 * inf``, ``inf - inf``) leak a ``NaN`` bound
out as if it were a valid enclosure. See :func:`_ivl_mul` for the ``0 * inf = 0`` extended-real rule
multiplication relies on.
"""

from __future__ import annotations

from typing import Any

import numpy as np

_NEG_INF = -np.inf
_POS_INF = np.inf


def _down(x: np.ndarray) -> np.ndarray:
    """Round each bound toward -inf by one ULP (sound lower bound)."""
    return np.nextafter(x, _NEG_INF)


def _up(x: np.ndarray) -> np.ndarray:
    """Round each bound toward +inf by one ULP (sound upper bound)."""
    return np.nextafter(x, _POS_INF)


def _ivl_mul(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Elementwise product, plus a mask of which entries are exact, under the extended-real
    interval-arithmetic convention for ``0 * inf``.

    IEEE-754 defines ``0 * (+/-inf) == NaN`` -- correct for flagging an indeterminate *limit*, but
    wrong here: a factor that is exactly zero denotes a degenerate, zero-width interval, so the
    enclosed quantity is provably zero regardless of how unbounded the other factor is. There is
    nothing indeterminate about it (unlike, say, ``inf - inf``): every pair of finite values drawn
    from ``{0} x (-inf, inf)`` multiplies to exactly ``0``, so the sound *and* tightest enclosure is
    ``0``, not "give up". Special-case that one pattern (either sign of zero, either operand order)
    and fall through to ordinary IEEE-754 multiplication for everything else -- finite*finite,
    finite*inf, and inf*inf are all already correct as-is.

    The second return value flags entries where *either* factor is a literal zero: IEEE-754
    multiplication by zero never rounds (``0 * finite`` is exact, and ``0 * inf`` is exact by the
    convention above), so these carry no rounding error for the caller to pad against -- unlike, say,
    two tiny nonzero finite factors that underflow *to* zero, which are not flagged and still need
    the pad, since their true product is nonzero even though it is not representable.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    zero_times_inf = ((x == 0.0) & np.isinf(y)) | ((y == 0.0) & np.isinf(x))
    with np.errstate(invalid="ignore"):  # the 0*inf corners computed here are overridden just below
        p = x * y
    p = np.where(zero_times_inf, 0.0, p)
    exact = (x == 0.0) | (y == 0.0)
    return p, exact


class Interval:
    """A guaranteed enclosure ``[lo, hi]`` of a value (scalar or numpy array), outward-rounded."""

    __slots__ = ("lo", "hi")

    def __init__(self, lo: Any, hi: Any) -> None:
        lo_arr = np.asarray(lo, dtype=np.float64)
        hi_arr = np.asarray(hi, dtype=np.float64) + np.zeros_like(lo_arr)
        lo_arr = lo_arr + np.zeros_like(hi_arr)
        # A NaN bound cannot certify anything: reject before it can be mistaken for a valid
        # enclosure by any downstream consumer (+/-inf remains legal -- only NaN is not a number).
        if np.any(np.isnan(lo_arr)) or np.any(np.isnan(hi_arr)):
            raise ValueError("Interval endpoints must not be NaN: lo=%r, hi=%r" % (lo_arr, hi_arr))
        # A reversed bound is not an enclosure of anything; +/-inf endpoints stay legal (e.g.
        # [-inf, inf], [0, inf], and the degenerate point [inf, inf] are all fine -- lo <= hi holds).
        if np.any(lo_arr > hi_arr):
            raise ValueError("Interval lower bound must not exceed the upper bound: lo=%r, hi=%r" % (lo_arr, hi_arr))
        self.lo = lo_arr
        self.hi = hi_arr

    @classmethod
    def exact(cls, x: Any) -> Interval:
        """A degenerate interval ``[x, x]`` for an exactly-represented value."""
        x = np.asarray(x, dtype=np.float64)
        return cls(x, x.copy())

    @classmethod
    def from_quantized(cls, original: Any, fmt: Any) -> Interval:
        """Enclose ``original`` using the quantized format's error bound.

        ``max_rel_error`` is a duck-typed, self-reported claim: nothing here enforces that an
        implementer's static bound actually holds for every value ``fmt`` might quantize, only that it
        is offered as one. A format that flushes underflowing values to exactly zero, or saturates on
        overflow, would still report a fixed ``max_rel_error`` while its true error at those magnitudes
        no longer scales with it -- the analytic pad below would then be too narrow, silently returning
        an interval that does not actually enclose ``original`` (:class:`~mixle.engines.formats.
        FloatFormat` avoids this today by staying mantissa-only with an unbounded exponent, so its
        ``max_rel_error`` genuinely is universal -- see that class's docstring -- but nothing here
        depends on every format that will ever be passed in honoring that same discipline). Guard
        against a format that breaks the assumption by cross-checking the analytic pad against the
        error actually measured on ``original`` -- already in hand, it is quantized right here -- and
        padding by whichever is WIDER. This needs nothing new from ``fmt`` beyond ``round_trip``, which
        every caller already requires, and is a no-op whenever ``max_rel_error`` genuinely does bound
        this data (the analytic pad is then always >= the measured one, by construction).
        """
        x = np.asarray(original, dtype=np.float64)
        q = np.asarray(fmt.round_trip(original), dtype=np.float64)
        rel = float(getattr(fmt, "max_rel_error", None) or 0.0)
        if rel <= 0.0:  # codecs without an analytic relative bound: use the measured absolute error
            d = float(fmt.measured_max_abs_error(original))
            return cls(_down(q - d), _up(q + d))
        # |original - q| <= rel * |original| -> a sound symmetric pad around q, PROVIDED max_rel_error
        # genuinely bounds this data; cross-check against the error observed on `x` itself and take
        # the wider (safer) of the two, so a format whose analytic claim doesn't hold here still yields
        # a sound enclosure instead of a silently too-narrow one (see docstring).
        analytic_d = np.abs(q) * (rel / (1.0 - rel))
        empirical_d = np.abs(q - x)
        d = np.maximum(analytic_d, empirical_d)
        return cls(_down(q - d), _up(q + d))

    def width(self) -> np.ndarray:
        """The guaranteed error bound: ``hi - lo`` (outward-rounded).

        A degenerate point exactly at ``+/-inf`` (``lo == hi == +/-inf``, legal per the constructor)
        has zero width by definition -- but IEEE-754 computes ``inf - inf == NaN`` for it, which
        would otherwise leak a NaN out of this function with no constructor downstream to catch it.
        Special-case that one pattern directly; every other case (including an ordinary, non-
        degenerate unbounded interval like ``[0, inf]``, whose width is correctly ``inf``) is
        unaffected.
        """
        degenerate_at_infinity = np.isinf(self.lo) & (self.lo == self.hi)
        with np.errstate(invalid="ignore"):  # the inf-inf case computed here is discarded just below
            d = _up(self.hi - self.lo)
        return np.where(degenerate_at_infinity, 0.0, d)

    def max_width(self) -> float:
        """Return the largest interval width."""
        return float(np.max(self.width())) if self.lo.size else 0.0

    def midpoint(self) -> np.ndarray:
        """Return interval midpoints.

        Same ``inf - inf`` gap as :meth:`width`: a degenerate point at ``+/-inf`` is its own
        midpoint by definition, computed directly rather than via the subtraction that would
        otherwise produce NaN.
        """
        degenerate_at_infinity = np.isinf(self.lo) & (self.lo == self.hi)
        with np.errstate(invalid="ignore"):  # the inf-inf case computed here is discarded just below
            m = self.lo + 0.5 * (self.hi - self.lo)
        return np.where(degenerate_at_infinity, self.lo, m)

    def contains(self, value: Any) -> np.ndarray:
        """Return a boolean mask for values inside the interval."""
        v = np.asarray(value, dtype=np.float64)
        return (self.lo <= v) & (v <= self.hi)

    def __add__(self, other: Interval) -> Interval:
        # Unlike multiplication's 0*inf, addition's only indeterminate form (inf + -inf) has no
        # single correct finite answer, and is only reachable at all when one operand is a
        # degenerate point exactly at +/-inf (an ordinary unbounded interval like [5, inf] has a
        # finite lo, so finite + -inf = -inf here, never NaN). When it is reached, the constructor
        # below rejects the resulting NaN bound and raises rather than silently certifying an
        # unsound result -- refusing to answer is sound, a fabricated finite bound would not be.
        return Interval(_down(self.lo + other.lo), _up(self.hi + other.hi))

    def __sub__(self, other: Interval) -> Interval:
        # Same inf - inf gap as __add__ above, same resolution: let the constructor's NaN check
        # turn it into a raised error instead of a silently unsound [nan, ...] bound.
        return Interval(_down(self.lo - other.hi), _up(self.hi - other.lo))

    def __mul__(self, other: Interval) -> Interval:
        # The product range is spanned by the four corner products. Each corner goes through
        # _ivl_mul rather than raw `*` so a 0*inf corner comes back exactly 0 (see _ivl_mul) instead
        # of NaN -- construction-time validation guarantees self/other never carry a NaN endpoint, so
        # 0*inf is the only way a corner could turn up NaN here.
        #
        # Each corner is outward-rounded on its own rather than after combining (equivalent for any
        # corner that does need rounding, since _down/_up are monotonic: rounding the min/max of a
        # set equals the min/max of the rounded set) -- except an exact corner (a literal-zero
        # factor, per _ivl_mul) skips the pad entirely, since there is no rounding error on it to
        # compensate for. Without this, e.g. [0,0]*[-inf,inf] would report a spurious 1-ULP-wide
        # [-5e-324, 5e-324] "certificate" around a product that is exactly, not approximately, 0.
        corners = [
            _ivl_mul(self.lo, other.lo),
            _ivl_mul(self.lo, other.hi),
            _ivl_mul(self.hi, other.lo),
            _ivl_mul(self.hi, other.hi),
        ]
        lo = np.min(np.stack([np.where(exact, p, _down(p)) for p, exact in corners]), axis=0)
        hi = np.max(np.stack([np.where(exact, p, _up(p)) for p, exact in corners]), axis=0)
        # np.min/np.max propagate NaN (unlike np.fmin/np.fmax, which silently discard it) -- so if a
        # corner is ever still NaN despite the above, it reaches the constructor below and raises
        # there instead of this function quietly handing back an unsound "certificate".
        return Interval(lo, hi)

    def __repr__(self) -> str:
        return "Interval(lo=%r, hi=%r)" % (self.lo, self.hi)


def sum_error_bound(x: Any) -> float:
    """Return a certified bound on the float64 error of ``sum(x)``.

    The standard a-priori bound ``|fl(sum) - sum| <= gamma_{n-1} * sum|x_i|`` with
    ``gamma_k = k*u / (1 - k*u)`` and ``u = 2**-53``. It is sound for any summation order. A bound that
    is large *relative to* ``|sum x|`` means the sum is ill-conditioned (cancellation) and warrants the
    double-double :func:`mixle.engines.extended.dd_sum`; a tight one means float64 already suffices and
    no extra compute is justified.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size < 2:
        return 0.0
    u = 2.0**-53
    k = x.size - 1
    gamma = (k * u) / (1.0 - k * u) if k * u < 1.0 else np.inf
    return float(gamma * np.abs(x).sum())


def sum_enclosure(x: Any) -> Interval:
    """Return an outward-rounded interval enclosing the true ``sum(x)``."""
    s = np.float64(np.sum(np.asarray(x, dtype=np.float64)))
    b = np.float64(sum_error_bound(x))
    return Interval(_down(s - b), _up(s + b))


def float64_sum_is_accurate(x: Any, target_rel_error: float = 1e-12) -> bool:
    """Return whether float64 summation is accurate to ``target_rel_error``.

    Reads the certified bound relative to the magnitude of the result -- precision allocation in one call.
    """
    s = abs(float(np.sum(np.asarray(x, dtype=np.float64))))
    bound = sum_error_bound(x)
    return bound <= target_rel_error * max(s, np.finfo(np.float64).tiny)
