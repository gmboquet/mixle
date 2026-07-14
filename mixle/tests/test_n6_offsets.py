"""N6: reclamation ecology & biodiversity offsets.

J6 ("register the priced term into ``risk_adjusted_plan(..., liabilities=...)``") has not landed on this
branch yet -- ``mixle.stochastic_opt.risk_adjusted_plan`` / ``mixle.analysis.objective.priced_liabilities``
do not exist here (see the N6 PR body Notes). This test therefore exercises the real, already-landed
``mixle.relations.branch_and_bound_milp`` (the same solver both ``two_stage_stochastic_plan`` and J6's
forthcoming ``risk_adjusted_plan`` are built on) directly, netting a per-block habitat-offset liability rate
out of expected profit before block selection -- exactly the "total liability subtracted per block before
optimization" pattern J6's ``priced_liabilities``/``risk_adjusted_plan`` use. Once J6 lands, the same
``habitat_offset_liability``/``no_net_loss_constraint`` outputs plug directly into its
``liabilities["total"]``/``constraints["caps"]`` without any change to this module.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from mixle.analysis.biodiversity import habitat_offset_liability, no_net_loss_constraint
from mixle.relations import branch_and_bound_milp

N_BLOCKS = 8
# Blocks 0-3 sit on prime (high-suitability) habitat; blocks 4-7 are low-suitability. Prime blocks are
# given a slightly higher baseline profit than non-prime blocks so that, with NO offset liability, the
# unconstrained-by-habitat optimum favors extracting them -- the contrast the DoD requires.
_SUITABILITY = np.array([0.9, 0.85, 0.8, 0.75, 0.1, 0.12, 0.08, 0.15])
_AREA = np.ones(N_BLOCKS)
_PROFIT = np.array([12.0, 11.5, 11.0, 10.5, 10.0, 9.6, 9.2, 8.8])  # prime blocks pay a bit more
_CARDINALITY = 4  # exactly-4-of-8 block-selection budget (forces trade-offs)
_HIGH_SUIT_IDX = np.array([0, 1, 2, 3])


def _habitat(mean: np.ndarray, area: np.ndarray) -> SimpleNamespace:
    """A minimal IC-1-shaped stand-in exposing only what N6 reads: ``.mean`` and ``.cell_area``."""
    return SimpleNamespace(mean=mean, cell_area=area)


def _select_blocks(profit: np.ndarray, liability_rate: np.ndarray, cardinality: int) -> np.ndarray:
    """Choose exactly ``cardinality`` blocks maximizing ``(profit - liability_rate) @ x`` via the real
    branch-and-bound MILP solver -- a minimal stand-in for J6's not-yet-landed ``risk_adjusted_plan``,
    which nets ``liabilities["total"]`` out of per-block profit before solving the same kind of MILP."""
    n = profit.size
    objective = profit - liability_rate
    a_ub = np.ones((1, n))
    b_ub = np.array([float(cardinality)])
    solved = branch_and_bound_milp(objective, a_ub, b_ub, integer=list(range(n)), bounds=[(0.0, 1.0)] * n, sense="max")
    assert solved is not None
    _, x = solved
    return np.round(x).astype(bool)


def test_offset_liability_reshapes_plan_toward_net_neutral_vs_baseline():
    habitat = _habitat(_SUITABILITY, _AREA)
    offset_ratio, unit_offset_cost = 2.0, 5.0

    baseline_plan = _select_blocks(_PROFIT, np.zeros(N_BLOCKS), _CARDINALITY)
    # sanity: with no habitat cost, the optimizer picks purely by profit, i.e. the 4 highest-profit
    # blocks, which are exactly the 4 prime-habitat ones.
    assert baseline_plan.sum() == _CARDINALITY
    assert np.array_equal(np.flatnonzero(baseline_plan), _HIGH_SUIT_IDX)

    liability_rate = offset_ratio * unit_offset_cost * _SUITABILITY * _AREA
    offset_plan = _select_blocks(_PROFIT, liability_rate, _CARDINALITY)
    assert offset_plan.sum() == _CARDINALITY  # same budget/cardinality -- a like-for-like comparison

    baseline_liability = habitat_offset_liability(
        baseline_plan, habitat, offset_ratio=offset_ratio, unit_offset_cost=unit_offset_cost
    )
    offset_liability = habitat_offset_liability(
        offset_plan, habitat, offset_ratio=offset_ratio, unit_offset_cost=unit_offset_cost
    )

    # the core DoD assertion: introducing the offset requirement drops/derates high-suitability-loss
    # blocks relative to the no-offset baseline plan -- fewer prime-habitat blocks are chosen, and the
    # resulting plan's own priced habitat liability is strictly lower than charging the baseline plan the
    # same rate.
    assert int(offset_plan[_HIGH_SUIT_IDX].sum()) < int(baseline_plan[_HIGH_SUIT_IDX].sum())
    assert offset_liability < baseline_liability
    assert offset_liability >= 0.0


def test_habitat_offset_liability_matches_closed_form_and_scales():
    habitat = _habitat(_SUITABILITY, _AREA)
    footprint = np.array([True, False, True, False, False, False, False, False])
    expected = 1.5 * (_SUITABILITY[0] * _AREA[0] + _SUITABILITY[2] * _AREA[2]) * 4.0
    got = habitat_offset_liability(footprint, habitat, offset_ratio=1.5, unit_offset_cost=4.0)
    assert got == pytest.approx(expected)

    # zero ratio/cost collapses to no liability; liability scales linearly in each parameter
    assert habitat_offset_liability(footprint, habitat, offset_ratio=0.0, unit_offset_cost=4.0) == 0.0
    assert habitat_offset_liability(footprint, habitat, offset_ratio=1.5, unit_offset_cost=0.0) == 0.0
    doubled = habitat_offset_liability(footprint, habitat, offset_ratio=3.0, unit_offset_cost=4.0)
    assert doubled == pytest.approx(2.0 * expected)


def test_habitat_offset_liability_rejects_shape_mismatch():
    habitat = _habitat(_SUITABILITY, _AREA)
    with pytest.raises(ValueError):
        habitat_offset_liability(np.ones(3, dtype=bool), habitat, offset_ratio=1.0, unit_offset_cost=1.0)


def test_no_net_loss_constraint_payload_and_row_semantics():
    habitat = _habitat(_SUITABILITY, _AREA)
    footprint = np.array([True, True, False, False, False, False, False, False])
    offset_ratio = 2.0

    payload = no_net_loss_constraint(footprint, habitat, offset_ratio=offset_ratio)

    expected_lost = float(_SUITABILITY[0] * _AREA[0] + _SUITABILITY[1] * _AREA[1])
    assert payload["lost_equivalents"] == pytest.approx(expected_lost)
    assert payload["required_offset"] == pytest.approx(offset_ratio * expected_lost)
    assert payload["per_cell_lost_equivalents"].shape == (N_BLOCKS,)
    assert payload["sense"] == ">="
    assert payload["variable"] == "offsets_created"

    # coeffs @ [offsets_created] <= bound must encode offsets_created >= required_offset
    coeffs, bound, required = payload["coeffs"], payload["bound"], payload["required_offset"]
    feasible_offsets = required + 1e-6  # meets the requirement
    infeasible_offsets = max(required - 1e-6, 0.0)  # falls short, unless required is ~0
    assert bool(coeffs @ np.array([feasible_offsets]) <= bound + 1e-9)
    if required > 1e-6:
        assert not bool(coeffs @ np.array([infeasible_offsets]) <= bound + 1e-9)


def test_no_net_loss_constraint_zero_footprint_requires_nothing():
    habitat = _habitat(_SUITABILITY, _AREA)
    payload = no_net_loss_constraint(np.zeros(N_BLOCKS, dtype=bool), habitat, offset_ratio=2.0)
    assert payload["lost_equivalents"] == 0.0
    assert payload["required_offset"] == 0.0
