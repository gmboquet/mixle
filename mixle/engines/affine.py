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

import math
from dataclasses import dataclass
from types import MappingProxyType
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
        owned_center = np.array(center, dtype=np.float64, copy=True)
        if owned_center.size and not np.all(np.isfinite(owned_center)):
            raise ValueError("affine center must contain only finite values")
        owned_center.setflags(write=False)
        owned_terms: dict[int, np.ndarray] = {}
        for symbol, coefficient in (terms or {}).items():
            if isinstance(symbol, (bool, np.bool_)) or not isinstance(symbol, (int, np.integer)) or symbol < 0:
                raise ValueError("affine noise-symbol identifiers must be nonnegative integers")
            try:
                owned = np.array(np.broadcast_to(coefficient, owned_center.shape), dtype=np.float64, copy=True)
            except ValueError:
                raise ValueError(
                    "affine coefficient for symbol %r is not broadcast-compatible with center shape %r"
                    % (symbol, owned_center.shape)
                ) from None
            if owned.size and not np.all(np.isfinite(owned)):
                raise ValueError("affine coefficients must contain only finite values")
            owned.setflags(write=False)
            owned_terms[int(symbol)] = owned
        self.center = owned_center
        self.terms = MappingProxyType(owned_terms)

    @classmethod
    def constant(cls, x: Any) -> AffineForm:
        """Create an affine form with no uncertainty terms."""
        return cls(np.asarray(x, dtype=np.float64), {})

    @classmethod
    def uncertain(cls, x: Any, radius: Any = 0.0) -> AffineForm:
        """An input known only to within +/- ``radius`` -- one fresh noise symbol of that radius."""
        center = np.asarray(x, dtype=np.float64)
        r = np.abs(np.asarray(radius, dtype=np.float64))
        if r.size and not np.all(np.isfinite(r)):
            raise ValueError("affine uncertainty radius must contain only finite values")
        terms = {}
        if np.any(r > 0):
            terms[_fresh()] = np.broadcast_to(r, center.shape)
        return cls(center, terms)

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


@dataclass(frozen=True)
class PrecisionResult:
    """Result of :func:`allocate_precision`: a dtype recommendation from an endpoint-magnitude
    heuristic. This is not an IEEE error certificate.

    ``status`` is one of:

    * ``"heuristic_fit"``: the endpoint estimate is no larger than the target.
    * ``"heuristic_insufficient"``: even the modeled ``qd`` estimate exceeds the target.

    Neither status certifies intermediate magnitude, conditioning, reduction order, underflow, or
    overflow. Use an operation graph with propagated intervals for a sound certificate.

    Attributes:
        dtype: the selected dtype name (best-effort, uncertified, when ``status == "insufficient"``).
        estimated_abs_error: ``op_count * unit_roundoff(dtype) * center_magnitude`` -- the estimated
            absolute roundoff ``dtype`` accumulates (an analytic estimate, not a measurement).
        target_abs_error: the caller's requested target, echoed back for convenience.
        status: see above; always explicitly heuristic.
    """

    dtype: str
    estimated_abs_error: float
    target_abs_error: float
    status: str = "heuristic_fit"

    @property
    def heuristic_met_target(self) -> bool:
        """Whether the explicitly non-certifying endpoint estimate fits the requested target."""
        return self.status == "heuristic_fit"

    @property
    def met_target(self) -> bool:
        """Always false: this result does not carry enough evidence to certify the target."""
        return False


def allocate_precision(center_magnitude: float, op_count: float, target_abs_error: float) -> PrecisionResult:
    """Suggest the lowest-cost dtype whose endpoint-magnitude error heuristic is under target.

    The estimate ``op_count*u*center_magnitude`` does not observe intermediate values or an operation
    graph and is therefore only a planning heuristic. It must never be used as an accuracy certificate.

    Raises ``ValueError`` unless ``center_magnitude`` and ``op_count`` are both finite and nonnegative and
    ``target_abs_error`` is finite and strictly positive. Before this validation existed, a negative
    ``op_count`` flipped the estimated error negative, which spuriously satisfies every dtype's budget
    comparison and misselects ``float16`` -- the least precise dtype -- regardless of how large the
    magnitude or how tight the target; a negative ``target_abs_error`` made every dtype's comparison fail
    and fell through to silently returning ``qd`` even though no dtype can certify a negative bound
    (impossible by definition); and a NaN/Inf value anywhere in the formula rode
    comparison-with-NaN-is-always-False (or Inf's unpredictable propagation) into whatever the fallthrough
    dtype happened to be, not a principled decision.
    """
    if not math.isfinite(center_magnitude) or center_magnitude < 0:
        raise ValueError("center_magnitude must be finite and nonnegative, got %r" % (center_magnitude,))
    if not math.isfinite(op_count) or op_count < 0:
        raise ValueError("op_count must be finite and nonnegative, got %r" % (op_count,))
    if not math.isfinite(target_abs_error) or target_abs_error <= 0:
        raise ValueError("target_abs_error must be finite and positive, got %r" % (target_abs_error,))

    for name in ("float16", "bfloat16", "float32", "float64", "dd", "qd"):
        estimated = op_count * UNIT_ROUNDOFF[name] * center_magnitude
        if estimated <= target_abs_error:
            return PrecisionResult(name, estimated, target_abs_error, "heuristic_fit")
    # qd is the highest-precision dtype this dial models, and its own check above already failed:
    # report that honestly (status="insufficient") instead of returning it as an uncertified last format.
    estimated = op_count * UNIT_ROUNDOFF["qd"] * center_magnitude
    return PrecisionResult("qd", estimated, target_abs_error, "heuristic_insufficient")
