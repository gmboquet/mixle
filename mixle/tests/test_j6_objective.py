"""J6 DoD -- economic objective integration, the grand synthesis (notes/exec/workstream-J.md).

Six synthetic blocks, all with identical grade/cost/price so the *only* thing that can move the optimal
plan is a priced liability or a hard constraint -- isolating each effect cleanly:

  * Blocks 0-1 are "high-emission": raising ``carbon_price`` far enough must remove exactly them from
    the optimal plan (the other four stay, since their emissions -- and hence their carbon liability --
    are negligible by comparison).
  * A no-mine polygon (a G9-style hard constraint) enclosing blocks 2-3 must remove exactly those two
    blocks, regardless of the (zero, in this test) carbon price.

Both scenarios are asserted against the zero-liability/zero-constraint baseline plan, which -- with
every block identically profitable -- extracts all six: proving grade/cost/carbon/enviro terms all trade
against each other on the one `risk_adjusted_plan` objective, per the task's Algorithm and DoD text.
"""

from __future__ import annotations

import numpy as np
import pytest

from mixle.analysis.objective import hard_constraints, priced_liabilities
from mixle.reason.posterior_protocol import Posterior
from mixle.stochastic_opt import StochasticPlan, risk_adjusted_plan

N_BLOCKS = 6
PRICE = 10.0
GRADE = 1.0  # identical for every block
COST = 2.0  # identical for every block -> raw per-block profit = PRICE * GRADE - COST = 8.0 everywhere
RAW_PROFIT = PRICE * GRADE - COST

HIGH_EMISSION_IDX = np.array([0, 1])
LOW_EMISSION_IDX = np.array([2, 3, 4, 5])
EMISSIONS = np.array([5.0, 5.0, 0.2, 0.2, 0.2, 0.2])
EXPOSURE = np.zeros(N_BLOCKS)  # health_cost is a no-op in this test; isolates the carbon effect

NO_MINE_IDX = np.array([2, 3])


class _FlatGradePosterior:
    """A minimal IC-1 `Posterior`: every block's grade is ``GRADE`` plus tiny iid scenario noise.

    The noise is small enough that the CVaR term tracks the mean profit closely and does not itself
    drive any block's inclusion/exclusion -- this DoD isolates the liability/constraint wiring, not
    H4's own grade-uncertainty risk aversion (that is H4's DoD, not J6's).
    """

    def samples(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return np.clip(GRADE + rng.normal(0.0, 0.01, size=(n, N_BLOCKS)), 0.0, None)

    @property
    def mean(self) -> np.ndarray:
        return np.full(N_BLOCKS, GRADE)

    @property
    def cov(self) -> np.ndarray:
        return np.eye(N_BLOCKS) * 0.01**2

    def credible_interval(self, level: float) -> tuple[np.ndarray, np.ndarray]:
        return self.mean - 0.1, self.mean + 0.1

    def derived_quantity(self, fn, n, rng):
        s = fn(self.samples(n, rng))

        class _DQ:
            samples = s
            prior_dominated = False

            def credible_interval(self, level):
                a = (1.0 - level) / 2.0
                return np.quantile(self.samples, a, axis=0), np.quantile(self.samples, 1 - a, axis=0)

        return _DQ()


def _plan(emissions: np.ndarray) -> dict:
    return {"grade": np.full(N_BLOCKS, GRADE), "exposure": EXPOSURE, "emissions": emissions}


def _no_cost(_arr: np.ndarray) -> np.ndarray:
    return np.zeros(N_BLOCKS)


def test_posterior_stub_conforms_to_ic1():
    assert isinstance(_FlatGradePosterior(), Posterior)


def test_priced_liabilities_shape_and_additivity():
    liabilities = priced_liabilities(
        _plan(EMISSIONS), carbon_price=3.0, health_cost=_no_cost, remediation_cost=_no_cost
    )
    assert liabilities["carbon"].shape == (N_BLOCKS,)
    assert np.allclose(liabilities["carbon"], 3.0 * EMISSIONS)
    assert np.allclose(liabilities["remediation"], 0.0)
    assert np.allclose(liabilities["health"], 0.0)
    assert np.allclose(liabilities["total"], liabilities["remediation"] + liabilities["health"] + liabilities["carbon"])
    assert liabilities["carbon_total"] == pytest.approx(float((3.0 * EMISSIONS).sum()))


def test_baseline_plan_extracts_every_block():
    posterior = _FlatGradePosterior()
    cost = np.full(N_BLOCKS, COST)
    baseline_liabilities = priced_liabilities(
        _plan(EMISSIONS), carbon_price=0.0, health_cost=_no_cost, remediation_cost=_no_cost
    )

    baseline_plan = risk_adjusted_plan(
        posterior,
        cost,
        PRICE,
        baseline_liabilities,
        {},
        k_scenarios=50,
        alpha=0.9,
        rng=np.random.default_rng(0),
    )
    assert isinstance(baseline_plan, StochasticPlan)
    assert baseline_plan.extract.shape == (N_BLOCKS,)
    assert baseline_plan.extract.dtype == np.bool_
    assert bool(baseline_plan.extract.all())


def test_raising_carbon_price_removes_exactly_the_high_emission_blocks():
    posterior = _FlatGradePosterior()
    cost = np.full(N_BLOCKS, COST)

    baseline_liabilities = priced_liabilities(
        _plan(EMISSIONS), carbon_price=0.0, health_cost=_no_cost, remediation_cost=_no_cost
    )
    baseline_plan = risk_adjusted_plan(
        posterior, cost, PRICE, baseline_liabilities, {}, k_scenarios=50, alpha=0.9, rng=np.random.default_rng(0)
    )
    assert bool(baseline_plan.extract[HIGH_EMISSION_IDX].all())

    # carbon_price = 3.0: high-emission blocks' liability (3.0 * 5.0 = 15.0) exceeds their raw profit
    # (8.0), while low-emission blocks' liability (3.0 * 0.2 = 0.6) does not -- so only the high-emission
    # blocks should flip from extracted to excluded.
    high_carbon_liabilities = priced_liabilities(
        _plan(EMISSIONS), carbon_price=3.0, health_cost=_no_cost, remediation_cost=_no_cost
    )
    assert RAW_PROFIT - float(high_carbon_liabilities["carbon"][0]) < 0.0
    assert RAW_PROFIT - float(high_carbon_liabilities["carbon"][2]) > 0.0

    high_carbon_plan = risk_adjusted_plan(
        posterior,
        cost,
        PRICE,
        high_carbon_liabilities,
        {},
        k_scenarios=50,
        alpha=0.9,
        rng=np.random.default_rng(0),
    )
    assert not bool(high_carbon_plan.extract[HIGH_EMISSION_IDX].any())
    assert bool(high_carbon_plan.extract[LOW_EMISSION_IDX].all())
    assert high_carbon_plan.expected_value < baseline_plan.expected_value


def test_no_mine_polygon_removes_exactly_its_enclosed_blocks():
    posterior = _FlatGradePosterior()
    cost = np.full(N_BLOCKS, COST)
    liabilities = priced_liabilities(
        _plan(EMISSIONS), carbon_price=0.0, health_cost=_no_cost, remediation_cost=_no_cost
    )

    baseline_plan = risk_adjusted_plan(
        posterior, cost, PRICE, liabilities, {}, k_scenarios=50, alpha=0.9, rng=np.random.default_rng(0)
    )
    assert bool(baseline_plan.extract[NO_MINE_IDX].all())

    no_mine_mask = np.zeros(N_BLOCKS, dtype=bool)
    no_mine_mask[NO_MINE_IDX] = True
    constraints = hard_constraints(no_mine_mask=no_mine_mask)

    no_mine_plan = risk_adjusted_plan(
        posterior, cost, PRICE, liabilities, constraints, k_scenarios=50, alpha=0.9, rng=np.random.default_rng(0)
    )
    assert not bool(no_mine_plan.extract[NO_MINE_IDX].any())
    other_idx = np.setdiff1d(np.arange(N_BLOCKS), NO_MINE_IDX)
    assert bool(no_mine_plan.extract[other_idx].all())


def test_exposure_cap_constrains_selection():
    """A K6-style exposure cap (via `hard_constraints(caps=...)`) forces fewer blocks than the
    unconstrained baseline when every block contributes equally to the capped resource."""
    posterior = _FlatGradePosterior()
    cost = np.full(N_BLOCKS, COST)
    liabilities = priced_liabilities(
        _plan(EMISSIONS), carbon_price=0.0, health_cost=_no_cost, remediation_cost=_no_cost
    )

    # sum(x) <= 3 (a per-block-equal "exposure unit" budget) with 6 identically profitable blocks --
    # exactly 3 must be selected instead of all 6.
    constraints = hard_constraints(caps=[{"coeffs": np.ones(N_BLOCKS), "bound": 3.0}])

    capped_plan = risk_adjusted_plan(
        posterior, cost, PRICE, liabilities, constraints, k_scenarios=50, alpha=0.9, rng=np.random.default_rng(0)
    )
    assert int(capped_plan.extract.sum()) == 3


def test_hard_constraints_rejects_bad_cap_sense():
    with pytest.raises(ValueError):
        hard_constraints(caps=[{"coeffs": np.ones(N_BLOCKS), "bound": 1.0, "sense": "=="}])
