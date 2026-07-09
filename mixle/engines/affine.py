"""Affine arithmetic -- tighter error tracing than intervals, and the precision-allocation dial.

An :class:`AffineForm` represents a quantity as a center plus a linear combination of shared *noise
symbols* ``x0 + sum_i x_i * eps_i`` with ``eps_i in [-1, 1]``. Unlike interval arithmetic, correlated
error *cancels*: ``(a + b) - a`` recovers ``b`` exactly because ``a``'s symbols subtract out, where an
interval would double the width. That tightness is what lets a precision-allocation pass avoid
over-spending bits.

The dial: evaluating an operation at dtype ``d`` injects a fresh roundoff symbol of radius
``u(d) * |result|`` (``u`` = unit roundoff). The affine radius at the root *is* the certified error
bound; a subtraction at a cancellation point makes it grow -- the escalation signal. Walk leaves->root
choosing the lowest-cost dtype whose injected radius keeps the root bound under target.

This is the tighter *estimate*; the fully IEEE-sound enclosure is :mod:`mixle.engines.error_tracing`
(interval, outward-rounded). The radius here is reported with one outward ULP of slop.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# unit roundoff (= 0.5 ULP, round-to-nearest) per dtype -- the precision dial's lookup.
UNIT_ROUNDOFF = {
    "float16": 2.0**-11,
    "bfloat16": 2.0**-8,
    "float32": 2.0**-24,
    "float64": 2.0**-53,
    "dd": 2.0**-106,  # double-double
    "qd": 2.0**-212,  # quad-double
}


def unit_roundoff(dtype: Any) -> float:
    """Unit roundoff for a dtype name / numpy dtype (``'dd'``/``'qd'`` for extended precision)."""
    if isinstance(dtype, str) and dtype in UNIT_ROUNDOFF:
        return UNIT_ROUNDOFF[dtype]
    name = np.dtype(dtype).name
    if name not in UNIT_ROUNDOFF:
        raise ValueError("no unit roundoff for dtype %r" % (dtype,))
    return UNIT_ROUNDOFF[name]


_counter = [0]


def _fresh() -> int:
    _counter[0] += 1
    return _counter[0]


class AffineForm:
    """``center + sum_i coeff_i * eps_i`` over shared noise symbols ``eps_i in [-1, 1]``."""

    __slots__ = ("center", "terms")

    def __init__(self, center: Any, terms: dict[int, np.ndarray] | None = None) -> None:
        self.center = np.asarray(center, dtype=np.float64)
        self.terms = terms if terms is not None else {}

    @classmethod
    def constant(cls, x: Any) -> AffineForm:
        """Create an affine form with no uncertainty terms."""
        return cls(np.asarray(x, dtype=np.float64), {})

    @classmethod
    def uncertain(cls, x: Any, radius: Any = 0.0) -> AffineForm:
        """An input known only to within +/- ``radius`` -- one fresh noise symbol of that radius."""
        f = cls(np.asarray(x, dtype=np.float64), {})
        r = np.abs(np.asarray(radius, dtype=np.float64))
        if np.any(r > 0):
            f.terms[_fresh()] = np.broadcast_to(r, f.center.shape).astype(np.float64).copy()
        return f

    def radius(self) -> np.ndarray:
        """Half-width = ``sum_i |coeff_i|`` (one outward ULP of slop for the f64 summation)."""
        if not self.terms:
            return np.zeros_like(self.center)
        total = np.zeros_like(self.center)
        for c in self.terms.values():
            total = total + np.abs(c)
        return np.nextafter(total, np.inf)

    def max_radius(self) -> float:
        """Return the largest interval half-width across entries."""
        r = self.radius()
        return float(np.max(r)) if r.size else 0.0

    def to_interval(self) -> Any:
        """Convert the affine form to an interval enclosure."""
        from mixle.engines.error_tracing import Interval

        r = self.radius()
        return Interval(self.center - r, self.center + r)

    def contains(self, value: Any) -> np.ndarray:
        """Return a boolean mask for values contained in the affine enclosure."""
        v = np.asarray(value, dtype=np.float64)
        r = self.radius()
        return (self.center - r <= v) & (v <= self.center + r)

    def _binary_terms(self, other: AffineForm, sign: float) -> dict[int, np.ndarray]:
        terms: dict[int, np.ndarray] = {k: v.copy() for k, v in self.terms.items()}
        for k, v in other.terms.items():
            terms[k] = terms[k] + sign * v if k in terms else sign * v
        return terms

    def __add__(self, other: AffineForm) -> AffineForm:
        return AffineForm(self.center + other.center, self._binary_terms(other, 1.0))

    def __sub__(self, other: AffineForm) -> AffineForm:
        return AffineForm(self.center - other.center, self._binary_terms(other, -1.0))

    def __mul__(self, other: AffineForm) -> AffineForm:
        center = self.center * other.center
        terms: dict[int, np.ndarray] = {}
        for k, v in self.terms.items():
            terms[k] = other.center * v
        for k, v in other.terms.items():
            terms[k] = terms.get(k, np.zeros_like(center)) + self.center * v
        # the second-order cross terms are lumped into one fresh symbol bounded by rad(self)*rad(other)
        nonlinear = self.radius() * other.radius()
        if np.any(nonlinear > 0):
            terms[_fresh()] = nonlinear
        return AffineForm(center, terms)

    def inject_roundoff(self, dtype: Any) -> AffineForm:
        """Add the roundoff a dtype-``dtype`` evaluation introduces: a fresh symbol of ``u*|center|``."""
        u = unit_roundoff(dtype)
        terms = {k: v.copy() for k, v in self.terms.items()}
        terms[_fresh()] = u * np.abs(self.center)
        return AffineForm(self.center, terms)


def allocate_precision(center_magnitude: float, op_count: int, target_abs_error: float) -> str:
    """Lowest-cost dtype whose accumulated roundoff over ``op_count`` ops keeps error under target.

    Each op injects ~``u(d) * |magnitude|``; ``op_count`` of them accumulate to ``op_count*u*|mag|``.
    Walk from lower to higher precision and return the first dtype that fits the budget.
    """
    for name in ("float16", "bfloat16", "float32", "float64", "dd", "qd"):
        if op_count * UNIT_ROUNDOFF[name] * abs(center_magnitude) <= target_abs_error:
            return name
    return "qd"
