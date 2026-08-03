"""Optional compensated (Kahan) accumulation with a running round-off estimate.

Precedent: :mod:`mixle.inference.precision_plan` already tracks a *validated* summed-LL error band
("~1e-6 relative") for the reduced-precision fused kernel -- a static, offline-verified bound picked
once per model/data pair. This module is the *dynamic, per-accumulator* counterpart: an OPT-IN
compensated-summation mode that carries its own running error estimate as the computation proceeds, so a
partition's numerics receipt travels with its sufficient statistics through ``combine()`` exactly like
the statistics themselves.

Design: the receipt an accumulator carries is ``(abs_total, n)`` -- the running sum of absolute
addend magnitudes and the running term count. ``n`` is exactly additive, while ``abs_total`` and its
partition merges are themselves floating-point reductions. The receipt therefore does not claim that
its fields compose exactly.

For ordinary recursive summation, :func:`error_bound` uses the conservative ``gamma_(n-1)`` factor
``k*eps/(1-k*eps)`` instead of its first-order approximation. For compensated summation, the commonly
used Higham expression ``(2*eps + n*eps**2) * sum(abs(x))`` remains an asymptotic diagnostic estimate,
not a certified upper bound: its omitted higher-order terms and the receipt's own floating-point
accumulation prevent certification. The compatibility names ``error_bound`` and ``bound`` are retained,
but their docstrings explicitly state this distinction.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import zip_longest
from operator import index
from typing import Any

import numpy as np

EPS = float(np.finfo(np.float64).eps)  # 2**-52 ~= 2.22e-16


def _validated_receipt_inputs(n: Any, abs_total: Any, compensated: Any) -> tuple[int, float, bool]:
    if isinstance(n, (bool, np.bool_)):
        raise TypeError("receipt term count must be an integer, not a boolean.")
    try:
        count = index(n)
    except TypeError as exc:
        raise TypeError("receipt term count must be an integer.") from exc
    if count < 0:
        raise ValueError("receipt term count must be non-negative.")
    try:
        magnitude = float(abs_total)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("receipt absolute total must be a finite non-negative scalar.") from exc
    if not np.isfinite(magnitude) or magnitude < 0.0:
        raise ValueError("receipt absolute total must be a finite non-negative scalar.")
    if not isinstance(compensated, (bool, np.bool_)):
        raise TypeError("receipt compensated mode must be a boolean.")
    return count, magnitude, bool(compensated)


def error_bound(n: int, abs_total: float, compensated: bool) -> float:
    """Return a round-off diagnostic for a finite float64 summation receipt.

    For plain recursive summation this is the standard ``gamma_(n-1)`` bound under the usual
    no-overflow/no-underflow model. For Kahan summation this is a first/second-order estimate and MUST
    NOT be treated as a certificate. The historical function name is retained for API compatibility.

    Args:
        n (int): Number of terms summed.
        abs_total (float): Running sum of the absolute value of each (weighted) addend.
        compensated (bool): Whether the sum was accumulated with Kahan compensation.

    Returns:
        float: A finite non-negative diagnostic, or ``inf`` when the classical gamma bound's
            precondition ``(n - 1) * eps < 1`` does not hold.
    """
    n, abs_total, compensated = _validated_receipt_inputs(n, abs_total, compensated)
    if n <= 1 or abs_total <= 0.0:
        return 0.0
    if compensated:
        return (2.0 * EPS + n * EPS * EPS) * abs_total
    scaled = (n - 1) * EPS
    if scaled >= 1.0:
        return float("inf")
    return (scaled / (1.0 - scaled)) * abs_total


@dataclass
class CompensatedAccumulator:
    """A running (optionally Kahan-compensated) sum plus a validated numerical receipt.

    ``total`` is the running sum (Kahan-corrected when ``compensated=True``, plain float64
    accumulation otherwise). ``abs_total`` and ``n`` compose under :meth:`combine`, subject to
    float64 rounding of ``abs_total`` itself.
    """

    total: float = 0.0
    compensation: float = 0.0
    abs_total: float = 0.0
    n: int = 0
    compensated: bool = True

    def __post_init__(self) -> None:
        self.n, self.abs_total, self.compensated = _validated_receipt_inputs(self.n, self.abs_total, self.compensated)
        for label in ("total", "compensation"):
            try:
                value = float(getattr(self, label))
            except (TypeError, ValueError, OverflowError) as exc:
                raise TypeError(f"accumulator {label} must be a finite scalar.") from exc
            if not np.isfinite(value):
                raise ValueError(f"accumulator {label} must be a finite scalar.")
            setattr(self, label, value)
        if not self.compensated and self.compensation != 0.0:
            raise ValueError("an uncompensated accumulator cannot carry a compensation term.")
        if self.n == 0 and (self.abs_total != 0.0 or self.total != 0.0 or self.compensation != 0.0):
            raise ValueError("an empty accumulator must have zero total, compensation, and absolute total.")
        if self.abs_total == 0.0 and (self.total != 0.0 or self.compensation != 0.0):
            raise ValueError("a zero-magnitude receipt cannot carry a nonzero total or compensation.")

    def add(self, x: float, weight: float = 1.0) -> CompensatedAccumulator:
        """Fold one (weighted) addend into the running sum."""
        try:
            value, scale = float(x), float(weight)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError("accumulator values and weights must be finite scalars.") from exc
        if not np.isfinite(value) or not np.isfinite(scale):
            raise ValueError("accumulator values and weights must be finite scalars.")
        term = value * scale
        if not np.isfinite(term):
            raise ValueError("weighted accumulator term must be finite.")
        if self.compensated:
            y = term - self.compensation
            t = self.total + y
            compensation = (t - self.total) - y
        else:
            t = self.total + term
            compensation = 0.0
        abs_total = self.abs_total + abs(term)
        if not np.isfinite(t) or not np.isfinite(compensation) or not np.isfinite(abs_total):
            raise ValueError("accumulator update overflowed float64.")
        self.total = t
        self.compensation = compensation
        self.abs_total = abs_total
        self.n += 1
        return self

    def combine(self, other: CompensatedAccumulator) -> CompensatedAccumulator:
        """Merge another partition's running sum + receipt into this one, in place.

        Both accumulators must use the same numerical mode. The other accumulator's
        compensation-corrected estimate is folded in as one addend, while receipt counts and
        magnitudes are merged separately.
        """
        if not isinstance(other, CompensatedAccumulator):
            raise TypeError("can only combine another CompensatedAccumulator.")
        if self.compensated is not other.compensated:
            raise ValueError("cannot combine compensated and uncompensated accumulator receipts.")
        value_to_add = other.total - other.compensation
        if self.compensated:
            y = value_to_add - self.compensation
            t = self.total + y
            compensation = (t - self.total) - y
        else:
            t = self.total + value_to_add
            compensation = 0.0
        abs_total = self.abs_total + other.abs_total
        count = self.n + other.n
        if not np.isfinite(value_to_add) or not np.isfinite(t) or not np.isfinite(compensation):
            raise ValueError("accumulator combination overflowed float64.")
        if not np.isfinite(abs_total):
            raise ValueError("receipt combination overflowed float64.")
        self.total = t
        self.compensation = compensation
        self.abs_total = abs_total
        self.n = count
        return self

    def bound(self) -> float:
        """Return the historical round-off diagnostic; Kahan mode is not a certified bound."""
        return error_bound(self.n, self.abs_total, self.compensated)


def kahan_reduce(values: Iterable[float], weights: Iterable[float] | None = None) -> CompensatedAccumulator:
    """Reduce a sequence (optionally weighted) to a fresh :class:`CompensatedAccumulator`.

    A convenience one-shot reducer, mainly used by tests and small ad hoc reductions; accumulators
    that need this incrementally (across ``update``/``seq_update`` calls) should hold their own
    :class:`CompensatedAccumulator` instance instead of re-reducing from scratch each time.
    """
    acc = CompensatedAccumulator(compensated=True)
    if weights is None:
        for v in values:
            acc.add(v)
    else:
        missing = object()
        for v, w in zip_longest(values, weights, fillvalue=missing):
            if v is missing or w is missing:
                raise ValueError("values and weights must have equal cardinality.")
            acc.add(v, w)
    return acc


@dataclass(frozen=True)
class ConditioningReceipt:
    """A real, computed numerical-conditioning diagnostic for a (multivariate) fit.

    Captures the covariance eigenvalue spectrum an ``estimate()`` call saw, so a caller can tell a
    healthy fit from one balanced on a near-degenerate direction without recomputing the eigenspectrum
    itself.

    Attributes:
        eigenvalues (np.ndarray): Eigenvalues of the (raw, pre-regularization) covariance, ascending.
        condition_number (float): ``max_eigenvalue / min_eigenvalue`` (``inf`` if the smallest
            eigenvalue is <= 0, i.e. the raw covariance is singular / numerically indefinite).
        near_degenerate (bool): True when the smallest-to-largest eigenvalue ratio falls below
            ``degenerate_ratio_threshold`` (or the smallest eigenvalue is non-positive).
        degenerate_ratio_threshold (float): The ratio threshold used to set ``near_degenerate``.
        symmetry_error (float): Maximum absolute difference between opposite covariance entries
            before the within-tolerance numerical projection to symmetry.
        symmetry_tolerance (float): Maximum accepted symmetry error for this input.
    """

    eigenvalues: np.ndarray
    condition_number: float
    near_degenerate: bool
    degenerate_ratio_threshold: float
    symmetry_error: float
    symmetry_tolerance: float

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly view of this receipt."""
        return {
            "eigenvalues": [float(v) for v in self.eigenvalues],
            "condition_number": float(self.condition_number),
            "near_degenerate": bool(self.near_degenerate),
            "degenerate_ratio_threshold": float(self.degenerate_ratio_threshold),
            "symmetry_error": float(self.symmetry_error),
            "symmetry_tolerance": float(self.symmetry_tolerance),
        }


def conditioning_receipt(covar: np.ndarray, degenerate_ratio: float = 1.0e-6) -> ConditioningReceipt:
    """Compute a :class:`ConditioningReceipt` from a (symmetric) covariance matrix.

    ``degenerate_ratio`` is the smallest/largest eigenvalue ratio below which the covariance is
    flagged near-degenerate (i.e. it has a direction of near-zero variance relative to the dominant
    one) -- the "near-degenerate variance flag" the roadmap calls for, generalized from a single
    variance to the eigenvalue spectrum for the multivariate case.
    """
    if isinstance(degenerate_ratio, (bool, np.bool_)):
        raise TypeError("degenerate_ratio must be a finite scalar in (0, 1], not a boolean.")
    try:
        threshold = float(degenerate_ratio)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("degenerate_ratio must be a finite scalar in (0, 1].") from exc
    if not np.isfinite(threshold) or not 0.0 < threshold <= 1.0:
        raise ValueError("degenerate_ratio must be a finite scalar in (0, 1].")
    try:
        raw_covar = np.asarray(covar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("covariance must be a finite, non-empty square matrix.") from exc
    if np.iscomplexobj(raw_covar):
        raise TypeError("covariance must be real-valued.")
    try:
        covar = np.asarray(raw_covar, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("covariance must be a finite, non-empty square matrix.") from exc
    if covar.ndim != 2 or covar.shape[0] == 0 or covar.shape[0] != covar.shape[1]:
        raise ValueError("covariance must be a finite, non-empty square matrix.")
    if not np.isfinite(covar).all():
        raise ValueError("covariance must contain only finite values.")
    symmetry_error = float(np.max(np.abs(covar - covar.T)))
    symmetry_tolerance = 64.0 * EPS * max(1.0, float(np.max(np.abs(covar))))
    if symmetry_error > symmetry_tolerance:
        raise ValueError(
            f"covariance must be symmetric within {symmetry_tolerance:.6g}; maximum asymmetry is {symmetry_error:.6g}."
        )
    # Project only an already-certified round-off-sized asymmetry, retaining its measured size.
    symmetric_covar = 0.5 * (covar + covar.T)
    eigvals = np.linalg.eigvalsh(symmetric_covar)
    max_eig = float(np.max(eigvals))
    min_eig = float(np.min(eigvals))
    if min_eig <= 0.0:
        condition_number = float("inf")
        near_degenerate = True
    else:
        condition_number = max_eig / min_eig if max_eig > 0.0 else float("inf")
        near_degenerate = (min_eig / max_eig) < threshold if max_eig > 0.0 else True
    return ConditioningReceipt(
        eigenvalues=eigvals,
        condition_number=condition_number,
        near_degenerate=near_degenerate,
        degenerate_ratio_threshold=threshold,
        symmetry_error=symmetry_error,
        symmetry_tolerance=symmetry_tolerance,
    )
