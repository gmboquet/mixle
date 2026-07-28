"""Blending: minimum-cost mixture of several sources hitting a target attribute window.

A blend draws quantity ``w_s`` from each of several sources (stockpiles, batches, parcels), each
with its own per-attribute grade and unit cost, to hit a target *window* on the combined material's
attributes -- ``spec_lo[e] <= (sum_s w_s grades[s,e]) / sum_s w_s <= spec_hi[e]`` for every attribute
``e`` -- at minimum blending cost. Fixing the total blended quantity to the requested ``demand``
linearizes the otherwise-fractional ratio constraint into a pair of linear inequalities, so the whole
problem is a linear (or, with a minimum-parcel floor per source, mixed-integer) program solved by
:func:`mixle.relations.branch_and_bound_milp`. When the spec window cannot be met by any blend of the
given sources, :func:`mixle.relations.irreducible_infeasible_subset` names the conflicting
constraint rows -- which attribute window, high or low, is unmeetable -- rather than just reporting
"infeasible".

This is the classic blending problem of operations research: fuel/petroleum blending (octane rating),
feed formulation (protein/nutrient content), alloy composition, and ore blending & grade control
(the worked example below) all reduce to the identical LP/MILP -- sources, a per-attribute grade
matrix, and a target window, nothing domain-specific in the construction itself.

    >>> import numpy as np
    >>> grades = np.array([[0.50], [0.55], [0.65], [0.70]])  # Fe fraction per stockpile
    >>> costs = np.array([10.0, 12.0, 15.0, 18.0])
    >>> avail = np.array([1000.0, 1000.0, 1000.0, 1000.0])
    >>> cost, tonnage = blend_to_spec(grades, costs, [0.58], [0.62], avail, 1000.0)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mixle.relations import branch_and_bound_milp, irreducible_infeasible_subset

__all__ = ["blend_to_spec", "blend_feasibility"]

# Relative slack allowed when re-checking a solver's answer against the physical blend it claims.
_VERIFY_RTOL = 1.0e-6


def _finite(name: str, value: np.ndarray) -> np.ndarray:
    if not np.isfinite(value).all():
        raise ValueError(f"blend {name} must be finite")
    return value


def _validate_blend_geometry(
    grades: np.ndarray, spec_lo: np.ndarray, spec_hi: np.ndarray, avail: np.ndarray, demand: float
) -> float:
    """Check the physical geometry of a blend before any solver sees it.

    ``demand`` in particular must be finite and strictly positive. The linearized window constraints
    multiply every grade deviation by source tonnage, so at ``demand == 0`` every grade row reads
    ``0 <= 0`` and is vacuously satisfied: the solver happily certified a zero-tonnage "blend" of
    0.1-0.2 grade sources against a 0.9-1.0 window at cost zero, even though the blended grade it is
    certified against -- a ratio whose denominator is the total tonnage -- is undefined there.
    """
    if grades.ndim != 2 or grades.shape[0] == 0 or grades.shape[1] == 0:
        raise ValueError(f"blend grades must be a non-empty (n_sources, n_elements) matrix, got shape {grades.shape}")
    n_sources, n_elements = grades.shape
    _finite("grades", grades)
    for name, window in (("spec_lo", spec_lo), ("spec_hi", spec_hi)):
        if window.shape != (n_elements,):
            raise ValueError(f"blend {name} must have one entry per element ({n_elements}), got shape {window.shape}")
        _finite(name, window)
    if np.any(spec_lo > spec_hi):
        raise ValueError("blend spec window is reversed: every spec_lo must be <= its spec_hi")
    if avail.shape != (n_sources,):
        raise ValueError(f"blend avail must have one entry per source ({n_sources}), got shape {avail.shape}")
    _finite("avail", avail)
    if np.any(avail < 0.0):
        raise ValueError("blend avail must be non-negative: a source cannot hold negative material")
    demand = float(demand)
    if not np.isfinite(demand) or demand <= 0.0:
        raise ValueError(
            f"blend demand must be finite and strictly positive, got {demand!r}; the blended grade is a "
            "ratio over total tonnage and is undefined for a zero-tonnage blend, so the spec window "
            "cannot be certified there."
        )
    return demand


def _verify_blend(
    tonnage: np.ndarray, grades: np.ndarray, spec_lo: np.ndarray, spec_hi: np.ndarray, demand: float
) -> None:
    """Recompute the returned blend's actual tonnage and grade and refuse to certify a violation."""
    total = float(tonnage.sum())
    if not np.isclose(total, demand, rtol=_VERIFY_RTOL, atol=_VERIFY_RTOL * max(1.0, demand)):
        raise ValueError(f"blend_to_spec: returned blend totals {total}, not the requested demand {demand}")
    blended = (tonnage @ grades) / total
    tol = _VERIFY_RTOL * np.maximum(1.0, np.maximum(np.abs(spec_lo), np.abs(spec_hi)))
    outside = np.nonzero((blended < spec_lo - tol) | (blended > spec_hi + tol))[0]
    if outside.size:
        raise ValueError(
            f"blend_to_spec: returned blend's actual grade is outside the spec window for element(s) "
            f"{outside.tolist()}: blended={blended[outside].tolist()}, "
            f"window=[{spec_lo[outside].tolist()}, {spec_hi[outside].tolist()}]"
        )


def _spec_constraints(
    grades: np.ndarray, spec_lo: np.ndarray, spec_hi: np.ndarray, avail: np.ndarray, demand: float
) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float]]]:
    """Build the blend-to-spec LP's ``a_ub @ w <= b_ub`` rows and per-source bounds.

    Per element ``e``: the linearized head-grade window ``sum_s w_s (grades[s,e]-spec_lo[e]) >= 0``
    and ``sum_s w_s (grades[s,e]-spec_hi[e]) <= 0`` (two rows). Then the total-tonnage equality
    ``sum_s w_s = demand`` as a paired ``<=``/``>=`` inequality (two more rows). Bounds are
    ``0 <= w_s <= avail[s]``. Row order is fixed so callers can map an IIS row index back to
    "element e's lower window" / "element e's upper window" / "total tonnage".
    """
    n_sources, n_elements = grades.shape
    rows: list[np.ndarray] = []
    rhs: list[float] = []
    for e in range(n_elements):
        rows.append(-(grades[:, e] - spec_lo[e]))  # -> sum_s w_s (grade - lo) >= 0
        rhs.append(0.0)
        rows.append(grades[:, e] - spec_hi[e])  # -> sum_s w_s (grade - hi) <= 0
        rhs.append(0.0)
    rows.append(np.ones(n_sources))  # sum_s w_s <= demand
    rhs.append(float(demand))
    rows.append(-np.ones(n_sources))  # -sum_s w_s <= -demand  (together: sum_s w_s == demand)
    rhs.append(float(-demand))
    a_ub = np.array(rows, dtype=np.float64)
    b_ub = np.array(rhs, dtype=np.float64)
    bounds = [(0.0, float(avail[s])) for s in range(n_sources)]
    return a_ub, b_ub, bounds


def _min_parcel_gates(
    a_ub: np.ndarray, b_ub: np.ndarray, bounds: list[tuple[float, float]], avail: np.ndarray, min_parcel: float
) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float]], list[int]]:
    """Extend the LP with binary draw gates ``z_s`` for a minimum-parcel (discrete stockpile draw) floor.

    The big-M indicator pattern from :func:`mixle.relations.cardinality_constrained_milp`: appends one
    binary ``z_s`` per source and the rows ``w_s <= avail_s z_s`` / ``w_s >= min_parcel z_s``, so
    ``z_s = 0`` forces ``w_s = 0`` and ``z_s = 1`` forces ``w_s`` into ``[min_parcel, avail_s]`` --
    either no draw at all or a discrete draw above the floor, never an arbitrarily small sliver.
    """
    n_sources = len(bounds)
    n_rows = a_ub.shape[0]
    a_padded = np.hstack([a_ub, np.zeros((n_rows, n_sources))])
    gate_rows = np.zeros((2 * n_sources, 2 * n_sources))
    gate_rhs = np.zeros(2 * n_sources)
    for s in range(n_sources):
        gate_rows[2 * s, s] = 1.0  # w_s - avail_s z_s <= 0
        gate_rows[2 * s, n_sources + s] = -float(avail[s])
        gate_rows[2 * s + 1, s] = -1.0  # -w_s + min_parcel z_s <= 0
        gate_rows[2 * s + 1, n_sources + s] = float(min_parcel)
    a_ext = np.vstack([a_padded, gate_rows])
    b_ext = np.concatenate([b_ub, gate_rhs])
    bounds_ext = [*bounds, *([(0.0, 1.0)] * n_sources)]
    integer_idx = list(range(n_sources, 2 * n_sources))
    return a_ext, b_ext, bounds_ext, integer_idx


def blend_to_spec(
    grades: Any,
    costs: Any,
    spec_lo: Any,
    spec_hi: Any,
    avail: Any,
    demand: float,
    *,
    min_parcel: float | None = None,
) -> tuple[float, np.ndarray]:
    """Minimum-cost blend hitting a target attribute window, drawing quantity from several sources.

    ``grades`` is ``(n_sources, n_elements)``, ``costs``/``avail`` are length ``n_sources``,
    ``spec_lo``/``spec_hi`` are length ``n_elements``. The blend must total ``demand`` units and, for
    every attribute ``e``, ``spec_lo[e] <= blended_grade[e] <= spec_hi[e]``. Solved as an LP by default
    (``branch_and_bound_milp`` with no integer variables -> the HiGHS relaxation); pass
    ``min_parcel`` to additionally require that any source actually drawn from contributes at least
    that many units (a discrete parcel-draw floor), which turns it into a MILP.

    Returns ``(min blend cost, per-source tonnage)``. Raises :class:`ValueError` -- naming the
    conflicting constraint rows from :func:`mixle.relations.irreducible_infeasible_subset` -- when no
    blend of the given sources can meet the spec window at the requested tonnage.
    """
    grades = np.asarray(grades, dtype=np.float64)
    costs = np.asarray(costs, dtype=np.float64)
    spec_lo = np.asarray(spec_lo, dtype=np.float64)
    spec_hi = np.asarray(spec_hi, dtype=np.float64)
    avail = np.asarray(avail, dtype=np.float64)
    demand = _validate_blend_geometry(grades, spec_lo, spec_hi, avail, demand)
    n_sources = grades.shape[0]
    if costs.shape != (n_sources,):
        raise ValueError(f"blend costs must have one entry per source ({n_sources}), got shape {costs.shape}")
    _finite("costs", costs)
    if min_parcel is not None:
        min_parcel = float(min_parcel)
        if not np.isfinite(min_parcel) or min_parcel < 0.0:
            raise ValueError(f"blend min_parcel must be finite and non-negative, got {min_parcel!r}")

    a_ub, b_ub, bounds = _spec_constraints(grades, spec_lo, spec_hi, avail, demand)
    if min_parcel is None:
        res = branch_and_bound_milp(costs, a_ub, b_ub, integer=[], bounds=bounds, sense="min")
    else:
        a_ext, b_ext, bounds_ext, integer_idx = _min_parcel_gates(a_ub, b_ub, bounds, avail, min_parcel)
        c_ext = np.concatenate([costs, np.zeros(n_sources)])
        res = branch_and_bound_milp(c_ext, a_ext, b_ext, integer=integer_idx, bounds=bounds_ext, sense="min")

    if res is None:
        if min_parcel is None:
            iis = irreducible_infeasible_subset(a_ub, b_ub, bounds)
            row_desc = "element lower/upper windows, then total-tonnage"
        else:
            # The MILP actually solved (and found infeasible) is the min-parcel-gated system
            # (a_ext/b_ext/bounds_ext), not the base one -- diagnosing against a_ub/b_ub/bounds would
            # name conflicts in a DIFFERENT, ungated system that may itself be perfectly feasible,
            # producing a self-contradictory "no conflicting rows" result.
            iis = irreducible_infeasible_subset(a_ext, b_ext, bounds_ext)
            row_desc = "element lower/upper windows, then total-tonnage, then each source's min-parcel draw gate"
        raise ValueError(
            f"blend_to_spec: no blend of these sources meets the spec window at demand={demand}; "
            f"infeasible constraint rows ({row_desc}): {iis}"
        )
    value, x = res
    tonnage = np.asarray(x[:n_sources], dtype=np.float64)
    _verify_blend(tonnage, grades, spec_lo, spec_hi, demand)
    return float(value), tonnage


def blend_feasibility(grades: Any, spec_lo: Any, spec_hi: Any, avail: Any, demand: float) -> list[int] | None:
    """Whether some blend of the sources can meet the spec window at ``demand`` units, cost aside.

    Returns ``None`` when feasible. Otherwise returns the row indices of an irreducible infeasible
    subset (:func:`mixle.relations.irreducible_infeasible_subset`) of the linearized spec-window /
    total-quantity system -- the minimal set of conflicting constraints, i.e. which attribute's window
    (and whether its floor or ceiling) cannot be reached given the sources on hand.
    """
    grades = np.asarray(grades, dtype=np.float64)
    spec_lo = np.asarray(spec_lo, dtype=np.float64)
    spec_hi = np.asarray(spec_hi, dtype=np.float64)
    avail = np.asarray(avail, dtype=np.float64)
    demand = _validate_blend_geometry(grades, spec_lo, spec_hi, avail, demand)
    a_ub, b_ub, bounds = _spec_constraints(grades, spec_lo, spec_hi, avail, demand)
    return irreducible_infeasible_subset(a_ub, b_ub, bounds)
