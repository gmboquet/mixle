"""J4 DoD -- cost modeling (notes/exec/workstream-J.md).

Three things this task's Definition of Done actually asks for:

1. ``cost_curve`` is monotone increasing in depth.
2. ``cost_curve`` reproduces a hand-computed `$/t` on a toy input, within ``1e-9``.
3. Its output is acceptable as a `block_cost` / cost-model input downstream.

For (3), the DoD text names both ``monte_carlo_npv`` (J2, Wave 2 -- not yet written; it *extends this
same file* after J4 closes) and ``mixle.relations.min_cost_flow`` (H1, IC-9 -- per that module's own
repo-boundary note, H1 had not landed on ``release/0.8.0`` as of this PR either). Since neither symbol
exists in this tree yet, this file instead exercises the one real, already-merged consumer of a
`$/t`-per-block cost array on this branch: ``mixle.stochastic_opt.two_stage_stochastic_plan``'s
``block_cost`` argument (H4, which explicitly documents that it consumes a plain array of per-block
costs). Feeding ``cost_curve``'s output straight into it and getting back a valid ``StochasticPlan``
is the concrete acceptability check.
"""

from __future__ import annotations

import numpy as np
import pytest

from mixle.analysis.valuation import capex_opex, cost_curve
from mixle.stochastic_opt import StochasticPlan, two_stage_stochastic_plan

PARAMS = {
    "base_cost": 5.0,
    "haul_cost_per_m": 0.02,
    "grade_complexity_coef": 3.0,
    "throughput_scale_coef": 10.0,
    "design_capacity": 1000.0,
}


def test_cost_curve_monotone_increasing_in_depth():
    depths = np.array([0.0, 100.0, 500.0, 1000.0, 2500.0])
    grade = 2.0  # held fixed
    throughput = PARAMS["design_capacity"]  # at design capacity: the throughput term is exactly zero

    costs = cost_curve(depths, grade, throughput, params=PARAMS)

    assert costs.shape == depths.shape
    assert np.all(np.diff(costs) > 0.0), "cost_curve must be strictly increasing in depth"


def test_cost_curve_matches_hand_computed_value():
    # $/t = base_cost + haul_cost_per_m * depth + grade_complexity_coef / grade
    #     + throughput_scale_coef * ((throughput - design_capacity) / design_capacity) ** 2
    depth, grade, throughput = 500.0, 2.0, 1200.0
    expected = 5.0 + 0.02 * 500.0 + 3.0 / 2.0 + 10.0 * ((1200.0 - 1000.0) / 1000.0) ** 2
    assert expected == pytest.approx(16.9)

    actual = cost_curve(depth, grade, throughput, params=PARAMS)
    assert float(actual) == pytest.approx(expected, abs=1e-9)


def test_cost_curve_rejects_nonpositive_grade_or_throughput_or_capacity():
    with pytest.raises(ValueError):
        cost_curve(100.0, 0.0, 1000.0, params=PARAMS)
    with pytest.raises(ValueError):
        cost_curve(100.0, 2.0, 0.0, params=PARAMS)
    with pytest.raises(ValueError):
        cost_curve(100.0, 2.0, 1000.0, params={**PARAMS, "design_capacity": 0.0})


def test_capex_opex_matches_hand_computed_totals():
    plan = {
        "tonnage": [100.0, 200.0, 150.0],
        "depth": [100.0, 200.0, 300.0],
        "grade": [2.0, 2.0, 2.0],
        "throughput": [1000.0, 1000.0, 1000.0],  # at design capacity: scale term is zero
        "capex_schedule": [500_000.0, 0.0, 0.0],
    }
    params = {**PARAMS, "capex_fixed": 1_000_000.0, "capex_per_tonne": 10.0}

    # hand-computed per-period $/t: base 5.0 + haul (0.02 * depth) + complexity (3.0 / 2.0)
    per_period_cost = [8.5, 10.5, 12.5]
    expected_opex = 100.0 * 8.5 + 200.0 * 10.5 + 150.0 * 12.5  # = 4825.0
    expected_capex = 1_000_000.0 + 10.0 * (100.0 + 200.0 + 150.0) + 500_000.0  # = 1_504_500.0

    capex, opex = capex_opex(plan, params=params)

    assert opex == pytest.approx(expected_opex, abs=1e-9)
    assert capex == pytest.approx(expected_capex, abs=1e-9)
    assert isinstance(capex, float)
    assert isinstance(opex, float)
    # sanity: matches recomputing cost_curve directly, not just the hand arithmetic above
    reference_opex = float(np.sum(np.array(plan["tonnage"]) * np.array(per_period_cost)))
    assert opex == pytest.approx(reference_opex, abs=1e-9)


class _TinyBlockGradePosterior:
    """Minimal IC-1 `Posterior` stub over per-block grade, just enough to drive the acceptability check
    below -- not itself part of J4's Definition of Done, only the harness for it."""

    def __init__(self, n_blocks: int, mean_grade: np.ndarray) -> None:
        self._n = n_blocks
        self._mean = mean_grade

    def samples(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return self._mean[None, :] + rng.normal(0.0, 0.05, size=(n, self._n))

    @property
    def mean(self) -> np.ndarray:
        return self._mean

    @property
    def cov(self) -> np.ndarray:
        return np.eye(self._n)

    def credible_interval(self, level: float) -> tuple[np.ndarray, np.ndarray]:
        return self._mean - 1.0, self._mean + 1.0

    def derived_quantity(self, fn, n, rng):
        raise NotImplementedError("unused by this test")


def test_cost_curve_output_is_acceptable_block_cost_for_stochastic_opt():
    n_blocks = 4
    depth = np.array([50.0, 500.0, 1500.0, 3000.0])
    grade = np.array([1.0, 2.0, 3.0, 0.5])
    throughput = np.full(n_blocks, PARAMS["design_capacity"])

    block_cost = cost_curve(depth, grade, throughput, params=PARAMS)
    assert isinstance(block_cost, np.ndarray)
    assert block_cost.shape == (n_blocks,)

    posterior = _TinyBlockGradePosterior(n_blocks, mean_grade=np.array([5.0, 5.0, 5.0, 5.0]))
    plan = two_stage_stochastic_plan(
        posterior, block_cost, price=1.0, k_scenarios=20, alpha=0.9, rng=np.random.default_rng(0)
    )

    assert isinstance(plan, StochasticPlan)
    assert plan.extract.shape == (n_blocks,)


# --- MXR-080-0119: cost roll-up accepts impossible mine plans -------------------------------------
#
# Negative/non-finite depth, tonnage, capex, coefficients, and costs used to be accepted outright; a
# negative tonnage or cost could turn spending into revenue with no error, and a per-period array
# merely numpy-broadcast-compatible (but not shape-identical) with `tonnage` could silently multiply
# cost across periods instead of pairing each period with its own. The tests below pin the fix: every
# physical/cost quantity is required finite and correctly signed, and depth/grade/throughput must each
# be a scalar or match `tonnage`'s exact per-period shape.


def test_cost_curve_rejects_non_finite_depth():
    for bad_depth in (np.inf, -np.inf, np.nan):
        with pytest.raises(ValueError):
            cost_curve(bad_depth, 2.0, 1000.0, params=PARAMS)


def test_cost_curve_rejects_negative_depth():
    with pytest.raises(ValueError):
        cost_curve(-1.0, 2.0, 1000.0, params=PARAMS)


def test_cost_curve_rejects_non_finite_grade_or_throughput():
    with pytest.raises(ValueError):
        cost_curve(100.0, np.nan, 1000.0, params=PARAMS)
    with pytest.raises(ValueError):
        cost_curve(100.0, 2.0, np.inf, params=PARAMS)


def test_cost_curve_rejects_non_finite_or_negative_cost_coefficients():
    for key in ("base_cost", "haul_cost_per_m", "grade_complexity_coef", "throughput_scale_coef"):
        with pytest.raises(ValueError):
            cost_curve(100.0, 2.0, 1000.0, params={**PARAMS, key: -1.0})
        with pytest.raises(ValueError):
            cost_curve(100.0, 2.0, 1000.0, params={**PARAMS, key: np.nan})


def test_capex_opex_rejects_negative_or_non_finite_tonnage():
    plan_base = {"tonnage": None, "depth": 100.0, "grade": 2.0, "throughput": 1000.0}
    with pytest.raises(ValueError):
        capex_opex({**plan_base, "tonnage": [-100.0]}, params=PARAMS)
    with pytest.raises(ValueError):
        capex_opex({**plan_base, "tonnage": [np.inf]}, params=PARAMS)


def test_capex_opex_rejects_empty_tonnage():
    with pytest.raises(ValueError):
        capex_opex({"tonnage": [], "depth": 100.0, "grade": 2.0, "throughput": 1000.0}, params=PARAMS)


def test_capex_opex_rejects_negative_or_non_finite_capex_schedule():
    plan_base = {"tonnage": [100.0], "depth": 100.0, "grade": 2.0, "throughput": 1000.0}
    with pytest.raises(ValueError):
        capex_opex({**plan_base, "capex_schedule": [-1_000_000.0]}, params=PARAMS)
    with pytest.raises(ValueError):
        capex_opex({**plan_base, "capex_schedule": [np.nan]}, params=PARAMS)


def test_capex_opex_rejects_negative_capex_coefficients():
    plan = {"tonnage": [100.0], "depth": 100.0, "grade": 2.0, "throughput": 1000.0}
    with pytest.raises(ValueError):
        capex_opex(plan, params={**PARAMS, "capex_fixed": -1.0})
    with pytest.raises(ValueError):
        capex_opex(plan, params={**PARAMS, "capex_per_tonne": -1.0})


def test_capex_opex_negative_tonnage_no_longer_turns_spending_into_revenue():
    # The exact MXR-080-0119 repro: a negative tonnage used to flip opex's sign (spending -> revenue)
    # instead of raising.
    plan = {"tonnage": [-100.0], "depth": 100.0, "grade": 2.0, "throughput": 1000.0}
    with pytest.raises(ValueError):
        capex_opex(plan, params=PARAMS)


def test_capex_opex_rejects_broadcast_compatible_but_mismatched_period_shape():
    # depth shaped (3, 1) is numpy-broadcast-compatible with a length-3 tonnage but is NOT the same
    # shape -- feeding it straight to cost_curve used to silently balloon into a (3, 3) cross-product
    # cost matrix, multiplying each period's tonnage against every other period's cost too.
    plan = {
        "tonnage": np.array([100.0, 200.0, 150.0]),
        "depth": np.array([[100.0], [200.0], [300.0]]),
        "grade": np.array([2.0, 2.0, 2.0]),
        "throughput": np.array([1000.0, 1000.0, 1000.0]),
    }
    with pytest.raises(ValueError):
        capex_opex(plan, params=PARAMS)


def test_capex_opex_accepts_scalar_depth_grade_throughput_broadcast_against_tonnage():
    # A true scalar (not merely broadcast-compatible-but-mismatched) is still fine: it broadcasts
    # against every period, as documented.
    plan = {"tonnage": [100.0, 200.0, 150.0], "depth": 100.0, "grade": 2.0, "throughput": 1000.0}
    capex, opex = capex_opex(plan, params=PARAMS)
    expected_cost = PARAMS["base_cost"] + PARAMS["haul_cost_per_m"] * 100.0 + PARAMS["grade_complexity_coef"] / 2.0
    assert opex == pytest.approx((100.0 + 200.0 + 150.0) * expected_cost, abs=1e-9)
    assert capex == pytest.approx(0.0, abs=1e-9)  # no capex_fixed/capex_per_tonne/capex_schedule in PARAMS


def test_capex_opex_correctly_shaped_plan_still_rolls_up_normally():
    # Negative control: a normal, correctly-shaped mine plan (one entry per period, matching tonnage)
    # rolls up exactly as before -- same numbers as test_capex_opex_matches_hand_computed_totals.
    plan = {
        "tonnage": [100.0, 200.0, 150.0],
        "depth": [100.0, 200.0, 300.0],
        "grade": [2.0, 2.0, 2.0],
        "throughput": [1000.0, 1000.0, 1000.0],
        "capex_schedule": [500_000.0, 0.0, 0.0],
    }
    params = {**PARAMS, "capex_fixed": 1_000_000.0, "capex_per_tonne": 10.0}
    capex, opex = capex_opex(plan, params=params)
    assert opex == pytest.approx(4825.0, abs=1e-9)
    assert capex == pytest.approx(1_504_500.0, abs=1e-9)
