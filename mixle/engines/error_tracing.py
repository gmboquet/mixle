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


class Interval:
    """A guaranteed enclosure ``[lo, hi]`` of a value (scalar or numpy array), outward-rounded."""

    __slots__ = ("lo", "hi")

    def __init__(self, lo: Any, hi: Any) -> None:
        self.lo = np.asarray(lo, dtype=np.float64)
        self.hi = np.asarray(hi, dtype=np.float64) + np.zeros_like(self.lo)
        self.lo = self.lo + np.zeros_like(self.hi)

    @classmethod
    def exact(cls, x: Any) -> Interval:
        """A degenerate interval ``[x, x]`` for an exactly-represented value."""
        x = np.asarray(x, dtype=np.float64)
        return cls(x, x.copy())

    @classmethod
    def from_quantized(cls, original: Any, fmt: Any) -> Interval:
        """Enclose ``original`` using the quantized format's error bound."""
        q = np.asarray(fmt.round_trip(original), dtype=np.float64)
        rel = float(getattr(fmt, "max_rel_error", None) or 0.0)
        if rel <= 0.0:  # codecs without an analytic relative bound: use the measured absolute error
            d = float(fmt.measured_max_abs_error(original))
            return cls(_down(q - d), _up(q + d))
        # |original - q| <= rel * |original| -> a sound symmetric pad around q
        d = np.abs(q) * (rel / (1.0 - rel))
        return cls(_down(q - d), _up(q + d))

    def width(self) -> np.ndarray:
        """The guaranteed error bound: ``hi - lo`` (outward-rounded)."""
        return _up(self.hi - self.lo)

    def max_width(self) -> float:
        """Return the largest interval width."""
        return float(np.max(self.width())) if self.lo.size else 0.0

    def midpoint(self) -> np.ndarray:
        """Return interval midpoints."""
        return self.lo + 0.5 * (self.hi - self.lo)

    def contains(self, value: Any) -> np.ndarray:
        """Return a boolean mask for values inside the interval."""
        v = np.asarray(value, dtype=np.float64)
        return (self.lo <= v) & (v <= self.hi)

    def __add__(self, other: Interval) -> Interval:
        return Interval(_down(self.lo + other.lo), _up(self.hi + other.hi))

    def __sub__(self, other: Interval) -> Interval:
        return Interval(_down(self.lo - other.hi), _up(self.hi - other.lo))

    def __mul__(self, other: Interval) -> Interval:
        # the product range is spanned by the four corner products
        p = np.stack([self.lo * other.lo, self.lo * other.hi, self.hi * other.lo, self.hi * other.hi])
        return Interval(_down(np.min(p, axis=0)), _up(np.max(p, axis=0)))

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
