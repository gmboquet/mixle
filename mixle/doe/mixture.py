"""Mixture experiment designs on the simplex.

In a mixture experiment the factors are component *proportions* that must be non-negative and sum to
one, so the design space is the ``(q-1)``-simplex rather than a box. These generators return a
``(n, q)`` array whose rows are valid blends (each row sums to 1). Use them with a mixture (Scheffe)
polynomial when modelling the response. ``lower``/``upper`` bounds map the canonical simplex design
onto *pseudo-components* for constrained mixtures.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

import numpy as np


def _compositions(total: int, slots: int):
    """Yield every way to write ``total`` as an ordered sum of ``slots`` non-negative integers."""
    if slots == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _compositions(total - first, slots - 1):
            yield (first, *rest)


def simplex_lattice(q: int, m: int) -> np.ndarray:
    """``{q, m}`` simplex-lattice design for ``q`` mixture components.

    Each component takes one of the ``m + 1`` evenly spaced proportions ``0, 1/m, ..., 1`` and the
    proportions in a blend sum to one. The design is exactly the set of such blends -- ``C(q+m-1, m)``
    points -- and supports a degree-``m`` Scheffe mixture polynomial.

    Returns a ``(C(q+m-1, m), q)`` array of blends (rows sum to 1).
    """
    if q < 2:
        raise ValueError("q (number of components) must be >= 2.")
    if m < 1:
        raise ValueError("m (lattice degree) must be >= 1.")
    pts = np.array(list(_compositions(m, q)), dtype=np.float64)
    return pts / m


def simplex_centroid(q: int) -> np.ndarray:
    """Simplex-centroid design for ``q`` mixture components.

    Runs the centroid of every non-empty subset of components: the ``q`` pure components, the
    ``C(q,2)`` binary 1/2:1/2 blends, ... up to the overall centroid (all components at ``1/q``) --
    ``2**q - 1`` blends in total. Supports the special cubic mixture model.

    Returns a ``(2**q - 1, q)`` array of blends (rows sum to 1).
    """
    if q < 2:
        raise ValueError("q (number of components) must be >= 2.")
    rows = []
    for r in range(1, q + 1):
        for subset in combinations(range(q), r):
            row = np.zeros(q, dtype=np.float64)
            row[list(subset)] = 1.0 / r
            rows.append(row)
    return np.array(rows, dtype=np.float64)


_SIMPLEX_ATOL = 1e-9  # absolute tolerance for "non-negative" / "rows sum to one" checks (MXR-080-0180)


def to_pseudocomponents(blends: np.ndarray, lower: Sequence[float]) -> np.ndarray:
    """Map canonical simplex blends onto pseudo-components with per-component lower bounds.

    With lower bounds ``l_i`` (finite, non-negative, summing to ``< 1``), a constrained mixture's
    feasible region is itself a smaller simplex; this maps a canonical blend ``x`` -- itself a finite
    ``(n, q)`` array of genuine simplex points (non-negative, rows summing to 1) -- onto the real
    proportions ``a_i = l_i + (1 - sum l) * x_i`` (the standard L-pseudocomponent transform), so any
    simplex design above can be run inside the constrained region. Rows still sum to 1.

    Raises:
        ValueError: ``lower`` is not a finite, non-negative, 1-D vector summing to less than 1 (i.e. one
            that itself admits at least one feasible simplex point), or ``blends`` is not a finite 2-D
            array of shape ``(n, len(lower))`` whose rows are genuine canonical simplex points
            (non-negative, summing to 1). Invalid input is rejected outright instead of silently mapped
            to an out-of-simplex result (MXR-080-0180): e.g. blend ``[2, -1]`` with lower bounds
            ``[0.1, 0.1]`` used to silently return ``[1.7, -0.7]``, a negative "proportion".
    """
    low = np.asarray(lower, dtype=np.float64)
    if low.ndim != 1:
        raise ValueError(f"lower must be one-dimensional, got {low.ndim} dimensions.")
    if not np.all(np.isfinite(low)):
        raise ValueError(f"lower bounds must be finite, got {lower!r}.")
    if np.any(low < 0.0):
        raise ValueError(f"lower bounds must be non-negative, got {lower!r}.")
    total = float(low.sum())
    if total >= 1.0:
        raise ValueError(f"lower bounds must sum to less than 1, got sum={total!r}.")

    x = np.asarray(blends, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != low.shape[0]:
        raise ValueError(
            f"blends must be a 2-D (n, q) array with q == len(lower) ({low.shape[0]}); got shape {x.shape}."
        )
    if not np.all(np.isfinite(x)):
        raise ValueError("blends must be finite (no inf or nan).")
    if np.any(x < -_SIMPLEX_ATOL):
        raise ValueError(
            f"blends must be non-negative (canonical simplex points), got a minimum of {float(x.min())!r}."
        )
    row_sums = x.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=_SIMPLEX_ATOL):
        bad_rows = np.flatnonzero(~np.isclose(row_sums, 1.0, atol=_SIMPLEX_ATOL))
        raise ValueError(
            "blends must be canonical simplex points (each row summing to 1); row(s) "
            f"{bad_rows.tolist()} sum to {row_sums[bad_rows].tolist()}."
        )
    return low + (1.0 - total) * x


__all__ = ["simplex_lattice", "simplex_centroid", "to_pseudocomponents"]
