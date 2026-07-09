"""Optional compensated (Kahan) accumulation, carrying a running numerics-error bound.

Precedent: :mod:`mixle.inference.precision_plan` already tracks a *validated* summed-LL error band
("~1e-6 relative") for the reduced-precision fused kernel -- a static, offline-verified bound picked
once per model/data pair. This module is the *dynamic, per-accumulator* counterpart: an OPT-IN
compensated-summation mode that carries its own running error bound as the computation proceeds, so a
partition's numerics receipt travels with its sufficient statistics through ``combine()`` exactly like
the statistics themselves.

Design: the receipt an accumulator carries is ``(abs_total, n)`` -- the running sum of absolute
addend magnitudes and the running term count. Both are *exactly* additive under ``combine()`` (no
approximation: ``abs_total_combined = abs_total_a + abs_total_b``, ``n_combined = n_a + n_b``, by
definition), so they satisfy the roadmap's "bounds ADD through combine() like the stats" literally --
they are themselves additive sufficient statistics. The error BOUND is then the pure function
:func:`error_bound` of those two additive quantities. This is deliberately the SAFE direction: for the
naive bound, recomputing from the merged ``(n, abs_total)`` is always >= summing the two children's
bounds separately (since ``(n_a + n_b - 1) >= (n_a - 1) + (n_b - 1)`` whenever ``n_a, n_b >= 1``), so
composing through the additive receipt never under-reports the merged error.

Bound formulas (Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., secs 4.2 / 8.1):
  - naive summation of n terms: ``|error| <= (n-1) * eps * sum(|x_i|)``           (eq. 8.13 style)
  - Kahan compensated summation: ``|error| <= (2*eps + n*eps**2) * sum(|x_i|)``   (eq. 8.15 style)
where ``eps`` is the float64 machine epsilon (``2**-52``). Both are asymptotic first/second-order
bounds under the idealized no-overflow/no-underflow model; they are verified as non-violated upper
bounds (against a high-precision ``math.fsum`` reference) in
``mixle/tests/numerics_error_receipts_test.py``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

EPS = float(np.finfo(np.float64).eps)  # 2**-52 ~= 2.22e-16


def error_bound(n: int, abs_total: float, compensated: bool) -> float:
    """Return a valid upper bound on the float64 summation round-off error.

    Args:
        n (int): Number of terms summed.
        abs_total (float): Running sum of the absolute value of each (weighted) addend.
        compensated (bool): Whether the sum was accumulated with Kahan compensation.

    Returns:
        float: Non-negative upper bound on ``|computed_sum - true_sum|``.
    """
    if n <= 1 or abs_total <= 0.0:
        return 0.0
    if compensated:
        return (2.0 * EPS + n * EPS * EPS) * abs_total
    return (n - 1) * EPS * abs_total


@dataclass
class CompensatedAccumulator:
    """A running (optionally Kahan-compensated) sum plus the additive receipt its bound derives from.

    ``total`` is the running sum (Kahan-corrected when ``compensated=True``, plain float64
    accumulation otherwise). ``abs_total`` and ``n`` are the additive receipt: they compose exactly
    under :meth:`combine`, and :meth:`bound` derives the error bound from them via
    :func:`error_bound`.
    """

    total: float = 0.0
    compensation: float = 0.0
    abs_total: float = 0.0
    n: int = 0
    compensated: bool = True

    def add(self, x: float, weight: float = 1.0) -> CompensatedAccumulator:
        """Fold one (weighted) addend into the running sum."""
        term = float(x) * float(weight)
        if self.compensated:
            y = term - self.compensation
            t = self.total + y
            self.compensation = (t - self.total) - y
            self.total = t
        else:
            self.total += term
        self.abs_total += abs(term)
        self.n += 1
        return self

    def combine(self, other: CompensatedAccumulator) -> CompensatedAccumulator:
        """Merge another partition's running sum + receipt into this one, in place.

        The other accumulator's compensation-corrected estimate (``other.total -
        other.compensation``) is folded in as a single addend via the same (compensated or plain)
        addition rule this accumulator uses; ``abs_total``/``n`` -- the receipt -- add exactly, the
        same way the underlying sufficient statistics (``sum``, ``count``, ...) do.
        """
        value_to_add = other.total - other.compensation
        if self.compensated:
            y = value_to_add - self.compensation
            t = self.total + y
            self.compensation = (t - self.total) - y
            self.total = t
        else:
            self.total += value_to_add
        self.abs_total += other.abs_total
        self.n += other.n
        return self

    def bound(self) -> float:
        """Return the current error-bound receipt (see :func:`error_bound`)."""
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
        for v, w in zip(values, weights):
            acc.add(v, w)
    return acc


@dataclass
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
    """

    eigenvalues: np.ndarray
    condition_number: float
    near_degenerate: bool
    degenerate_ratio_threshold: float

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly view of this receipt."""
        return {
            "eigenvalues": [float(v) for v in self.eigenvalues],
            "condition_number": float(self.condition_number),
            "near_degenerate": bool(self.near_degenerate),
            "degenerate_ratio_threshold": float(self.degenerate_ratio_threshold),
        }


def conditioning_receipt(covar: np.ndarray, degenerate_ratio: float = 1.0e-6) -> ConditioningReceipt:
    """Compute a :class:`ConditioningReceipt` from a (symmetric) covariance matrix.

    ``degenerate_ratio`` is the smallest/largest eigenvalue ratio below which the covariance is
    flagged near-degenerate (i.e. it has a direction of near-zero variance relative to the dominant
    one) -- the "near-degenerate variance flag" the roadmap calls for, generalized from a single
    variance to the eigenvalue spectrum for the multivariate case.
    """
    covar = np.asarray(covar, dtype=np.float64)
    covar = 0.5 * (covar + covar.T)  # symmetrize away accumulation round-off before eigh
    eigvals = np.linalg.eigvalsh(covar)
    max_eig = float(np.max(eigvals))
    min_eig = float(np.min(eigvals))
    if min_eig <= 0.0:
        condition_number = float("inf")
        near_degenerate = True
    else:
        condition_number = max_eig / min_eig if max_eig > 0.0 else float("inf")
        near_degenerate = (min_eig / max_eig) < degenerate_ratio if max_eig > 0.0 else True
    return ConditioningReceipt(
        eigenvalues=eigvals,
        condition_number=condition_number,
        near_degenerate=near_degenerate,
        degenerate_ratio_threshold=degenerate_ratio,
    )
