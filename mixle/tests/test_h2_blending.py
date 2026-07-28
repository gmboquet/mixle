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


def test_blend_to_spec_min_parcel_infeasibility_diagnoses_the_gated_system():
    # min_parcel=90 exceeds every source's own availability (80 each), so no source can ever be
    # gated "on" (z_s=1 needs w_s in [90, 80], empty) -- the only min-parcel-respecting solution is
    # every source at 0, which can never reach demand=100. The BASE (ungated) system alone is
    # trivially feasible here (a wide-open spec window, plenty of combined tonnage available) --
    # the diagnostic must run on the min-parcel-gated system that was actually solved, not the base
    # one, or it reports "no conflicting rows" for a problem that is genuinely infeasible.
    grades = np.array([[0.5], [0.7]])
    costs = np.array([10.0, 12.0])
    avail = np.array([80.0, 80.0])
    spec_lo = np.array([0.0])
    spec_hi = np.array([1.0])
    demand = 100.0

    try:
        blend_to_spec(grades, costs, spec_lo, spec_hi, avail, demand, min_parcel=90.0)
    except ValueError as exc:
        message = str(exc)
        assert "infeasible" in message.lower()
        assert "none" not in message.lower()  # must name real conflicting rows, not report iis=None
    else:
        raise AssertionError("expected blend_to_spec to raise: no source can meet a min_parcel it can't hold")


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


def test_blend_to_spec_rejects_a_zero_demand_blend():
    # MXR-080-1674: the linearized window rows multiply every grade deviation by source tonnage, so at
    # demand=0 every grade row reads 0 <= 0 and is vacuously satisfied. blend_to_spec returned cost 0
    # and two zero tonnages for 0.1-0.2 grade sources against a 0.9-1.0 window -- certifying a blend
    # whose documented grade is a ratio with a zero denominator.
    grades = np.array([[0.1], [0.2]])
    costs = np.array([1.0, 2.0])
    avail = np.array([100.0, 100.0])

    for demand in (0.0, -5.0, float("nan"), float("inf")):
        try:
            blend_to_spec(grades, costs, [0.9], [1.0], avail, demand)
        except ValueError as exc:
            assert "demand" in str(exc).lower()
        else:
            raise AssertionError(f"expected blend_to_spec to reject demand={demand}")

    try:
        blend_feasibility(grades, [0.9], [1.0], avail, 0.0)
    except ValueError as exc:
        assert "demand" in str(exc).lower()
    else:
        raise AssertionError("expected blend_feasibility to reject demand=0")


def test_blend_to_spec_validates_physical_geometry():
    grades = np.array([[0.5], [0.7]])
    costs = np.array([10.0, 12.0])
    avail = np.array([80.0, 80.0])

    bad_cases = [
        (np.array([[np.nan], [0.7]]), costs, [0.5], [0.7], avail),  # non-finite grade
        (grades, costs, [0.7], [0.5], avail),  # reversed spec window
        (grades, costs, [0.5, 0.6], [0.7, 0.8], avail),  # window length != n_elements
        (grades, costs, [0.5], [0.7], np.array([80.0, -1.0])),  # negative availability
        (grades, np.array([10.0]), [0.5], [0.7], avail),  # costs length != n_sources
        (np.empty((0, 1)), np.array([]), [0.5], [0.7], np.array([])),  # no sources
    ]
    for g, c, lo, hi, av in bad_cases:
        try:
            blend_to_spec(g, c, lo, hi, av, 100.0)
        except ValueError:
            pass
        else:
            raise AssertionError("expected blend_to_spec to reject malformed blend geometry")


def test_blend_to_spec_verifies_the_returned_blend_it_certifies():
    # the happy path must still pass its own post-solve re-check of tonnage and blended grade
    grades = np.array([[0.50], [0.55], [0.65], [0.70]])
    costs = np.array([10.0, 12.0, 15.0, 18.0])
    avail = np.array([1000.0, 1000.0, 1000.0, 1000.0])

    cost, tonnage = blend_to_spec(grades, costs, [0.58], [0.62], avail, 1000.0)

    assert np.isclose(tonnage.sum(), 1000.0, atol=1e-6)
    blended = float(np.dot(tonnage, grades[:, 0]) / tonnage.sum())
    assert 0.58 - 1e-6 <= blended <= 0.62 + 1e-6
    assert cost > 0.0
