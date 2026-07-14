"""Ore blending & grade control (H2): blend-to-spec LP/MILP + IIS feasibility diagnostics."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from mixle.blending import blend_feasibility, blend_to_spec


def _reference_blend_cost(grades, costs, spec_lo, spec_hi, avail, demand):
    """Hand-built `linprog` reference for the single-element blend-to-spec LP.

    Independent of `mixle.blending`'s constraint assembly: minimizes `costs @ x` subject to the
    linearized head-grade window, the total-tonnage equality, and per-source availability bounds.
    """
    grades = np.asarray(grades, dtype=np.float64)
    costs = np.asarray(costs, dtype=np.float64)
    s = grades.shape[0]
    a_ub = np.array(
        [
            -(grades[:, 0] - spec_lo[0]),  # sum_s x_s (grade_s - lo) >= 0
            grades[:, 0] - spec_hi[0],  # sum_s x_s (grade_s - hi) <= 0
        ]
    )
    b_ub = np.array([0.0, 0.0])
    a_eq = np.ones((1, s))
    b_eq = np.array([demand])
    bounds = [(0.0, float(a)) for a in avail]
    res = linprog(costs, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    assert res.success
    return float(res.fun), res.x


def test_blend_to_spec_matches_reference_lp():
    # 4 stockpiles, single element (Fe fraction). Cheap sources are low-grade, expensive ones high-grade,
    # so the min-cost blend must trade off grade against cost while sitting inside [0.58, 0.62].
    grades = np.array([[0.50], [0.55], [0.65], [0.70]])
    costs = np.array([10.0, 12.0, 15.0, 18.0])
    avail = np.array([1000.0, 1000.0, 1000.0, 1000.0])
    spec_lo = np.array([0.58])
    spec_hi = np.array([0.62])
    demand = 1000.0

    ref_cost, ref_x = _reference_blend_cost(grades, costs, spec_lo, spec_hi, avail, demand)
    cost, tonnage = blend_to_spec(grades, costs, spec_lo, spec_hi, avail, demand)

    assert np.isclose(cost, ref_cost, atol=1e-6, rtol=1e-6)
    assert np.isclose(cost, np.dot(costs, ref_x), atol=1e-6, rtol=1e-6)
    assert tonnage.shape == (4,)
    assert np.isclose(tonnage.sum(), demand, atol=1e-6)
    assert np.all(tonnage >= -1e-9)
    assert np.all(tonnage <= avail + 1e-9)
    blended_grade = np.dot(tonnage, grades[:, 0]) / tonnage.sum()
    assert spec_lo[0] - 1e-6 <= blended_grade <= spec_hi[0] + 1e-6


def test_blend_feasibility_infeasible_returns_iis():
    # Every source is below the required floor -- no blend can reach spec_lo, regardless of weights.
    grades = np.array([[0.40], [0.45], [0.48], [0.50]])
    avail = np.array([1000.0, 1000.0, 1000.0, 1000.0])
    spec_lo = np.array([0.58])
    spec_hi = np.array([0.62])
    demand = 1000.0

    iis = blend_feasibility(grades, spec_lo, spec_hi, avail, demand)

    assert iis is not None
    assert len(iis) > 0
    assert all(isinstance(i, (int, np.integer)) for i in iis)


def test_blend_feasibility_feasible_returns_none():
    grades = np.array([[0.50], [0.55], [0.65], [0.70]])
    avail = np.array([1000.0, 1000.0, 1000.0, 1000.0])
    spec_lo = np.array([0.58])
    spec_hi = np.array([0.62])
    demand = 1000.0

    assert blend_feasibility(grades, spec_lo, spec_hi, avail, demand) is None


def test_blend_to_spec_min_parcel_gates_small_draws():
    # With a minimum-parcel gate, any source actually drawn from must contribute at least `min_parcel`
    # tons -- either 0 or a discrete draw above the floor, never a small continuous sliver.
    grades = np.array([[0.50], [0.55], [0.65], [0.70]])
    costs = np.array([10.0, 12.0, 15.0, 18.0])
    avail = np.array([1000.0, 1000.0, 1000.0, 1000.0])
    spec_lo = np.array([0.58])
    spec_hi = np.array([0.62])
    demand = 1000.0

    cost, tonnage = blend_to_spec(grades, costs, spec_lo, spec_hi, avail, demand, min_parcel=50.0)

    assert np.isclose(tonnage.sum(), demand, atol=1e-6)
    for w in tonnage:
        assert w <= 1e-6 or w >= 50.0 - 1e-6
    blended_grade = np.dot(tonnage, grades[:, 0]) / tonnage.sum()
    assert spec_lo[0] - 1e-6 <= blended_grade <= spec_hi[0] + 1e-6
    assert cost >= np.dot(costs, tonnage) - 1e-6


def test_blend_to_spec_infeasible_raises_with_iis_context():
    grades = np.array([[0.40], [0.45], [0.48], [0.50]])
    costs = np.array([10.0, 12.0, 15.0, 18.0])
    avail = np.array([1000.0, 1000.0, 1000.0, 1000.0])
    spec_lo = np.array([0.58])
    spec_hi = np.array([0.62])
    demand = 1000.0

    try:
        blend_to_spec(grades, costs, spec_lo, spec_hi, avail, demand)
    except ValueError as exc:
        assert "infeasible" in str(exc).lower()
    else:
        raise AssertionError("expected blend_to_spec to raise on an unmeetable spec window")
