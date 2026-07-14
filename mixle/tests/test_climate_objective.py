"""L6 DoD -- climate objective + risk into J/H (notes/exec/workstream-L.md).

Two operating options with otherwise-identical (and identically profitable) ore grade: a "clean"
option with a small emissions footprint and ample water, and a "dirty" option with a large emissions
footprint whose water budget already ran dry (``shortfall_m3 > 0``). :func:`climate_terms` turns each
option's `Footprint` + water budget into a priced carbon cost and a hard water-feasibility flag; folding
those into H4's `two_stage_stochastic_plan` per-option cost reshapes the optimal plan: the dirty/
water-short option, extracted in the no-climate baseline, is dropped once a carbon price and a binding
water limit are introduced, while the clean option stays in.

``water`` here is a plain object exposing ``.shortfall_m3``, ``.storage`` (the per-step trajectory), and
``.provenance`` -- exactly what an L2 `WaterBudget` duck-types against (L2 had not landed on this branch
as of this PR; see `mixle/analysis/emissions.py`'s module docstring for the repo-boundary note).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from mixle.analysis.emissions import Footprint, climate_terms
from mixle.reason.posterior_protocol import Posterior
from mixle.stochastic_opt import two_stage_stochastic_plan

PRICE = 1.0
GRADE_MEAN = 5.0
GRADE_NOISE = 0.01
BASE_COST = 1.0
CARBON_PRICE = 2.0
WATER_LIMIT_M3 = 1_000.0
INFEASIBLE_PENALTY = 1.0e6  # H4-side derate applied when climate_terms flags a hard water infeasibility


class _TwoOptionGradePosterior:
    """A minimal IC-1 `Posterior` over two options' ore grade, both clearly profitable at baseline."""

    def samples(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return GRADE_MEAN + rng.normal(0.0, GRADE_NOISE, size=(n, 2))

    @property
    def mean(self) -> np.ndarray:
        return np.full(2, GRADE_MEAN)

    @property
    def cov(self) -> np.ndarray:
        return np.eye(2) * GRADE_NOISE**2

    def credible_interval(self, level: float) -> tuple[np.ndarray, np.ndarray]:
        return self.mean - 1.0, self.mean + 1.0

    def derived_quantity(self, fn, n, rng):
        s = fn(self.samples(n, rng))

        class _DQ:
            samples = s
            prior_dominated = False

            def credible_interval(self, level):
                a = (1.0 - level) / 2.0
                return np.quantile(self.samples, a, axis=0), np.quantile(self.samples, 1 - a, axis=0)

        return _DQ()


def _clean_option():
    footprint = Footprint(scope1=0.05, scope2=0.05, scope3=0.0, total=0.1)
    water = SimpleNamespace(
        shortfall_m3=0.0,
        storage=np.full(12, 100.0),
        provenance={"demand_m3": 500.0},
    )
    return footprint, water


def _dirty_water_short_option():
    footprint = Footprint(scope1=5.0, scope2=3.0, scope3=2.0, total=10.0)
    water = SimpleNamespace(
        shortfall_m3=250.0,
        storage=np.array([50.0, 20.0, 0.0, 0.0, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]),
        provenance={"demand_m3": 5_000.0},
    )
    return footprint, water


def _climate_adjusted_cost(base_cost: float, footprint: Footprint, water) -> float:
    terms = climate_terms(footprint, water, carbon_price=CARBON_PRICE, water_limit_m3=WATER_LIMIT_M3)
    penalty = 0.0 if terms["water_feasible"] else INFEASIBLE_PENALTY
    return base_cost + terms["carbon_cost"] + penalty


def test_carbon_and_water_reshape_plan():
    posterior = _TwoOptionGradePosterior()
    clean_footprint, clean_water = _clean_option()
    dirty_footprint, dirty_water = _dirty_water_short_option()

    baseline_cost = np.array([BASE_COST, BASE_COST])
    baseline_plan = two_stage_stochastic_plan(
        posterior, baseline_cost, PRICE, k_scenarios=50, alpha=0.9, rng=np.random.default_rng(0)
    )
    # No-climate baseline: both options are (near-identically) profitable and both get extracted.
    assert bool(baseline_plan.extract[0]) is True
    assert bool(baseline_plan.extract[1]) is True

    climate_cost = np.array(
        [
            _climate_adjusted_cost(BASE_COST, clean_footprint, clean_water),
            _climate_adjusted_cost(BASE_COST, dirty_footprint, dirty_water),
        ]
    )
    climate_plan = two_stage_stochastic_plan(
        posterior, climate_cost, PRICE, k_scenarios=50, alpha=0.9, rng=np.random.default_rng(0)
    )

    # With a carbon price plus a binding water limit, the clean option is barely touched (small carbon
    # adder) and stays in, while the high-carbon, water-short option is derated to the point of exclusion.
    assert bool(climate_plan.extract[0]) is True
    assert bool(climate_plan.extract[1]) is False


def test_climate_terms_prices_carbon_and_flags_water_infeasibility():
    clean_footprint, clean_water = _clean_option()
    dirty_footprint, dirty_water = _dirty_water_short_option()

    clean = climate_terms(clean_footprint, clean_water, carbon_price=CARBON_PRICE, water_limit_m3=WATER_LIMIT_M3)
    dirty = climate_terms(dirty_footprint, dirty_water, carbon_price=CARBON_PRICE, water_limit_m3=WATER_LIMIT_M3)

    assert clean["carbon_cost"] == CARBON_PRICE * clean_footprint.total
    assert dirty["carbon_cost"] == CARBON_PRICE * dirty_footprint.total
    assert dirty["carbon_cost"] > clean["carbon_cost"]

    assert clean["water_feasible"] is True
    assert clean["water_binding"] is False
    assert dirty["water_feasible"] is False
    assert dirty["water_binding"] is True

    # shortfall_prob reads the water budget's own step trajectory: 3 of 12 dirty steps sit at zero.
    assert clean["shortfall_prob"] == 0.0
    assert dirty["shortfall_prob"] == pytest.approx(3.0 / 12.0)


def test_climate_terms_water_limit_alone_can_bind():
    footprint = Footprint(scope1=0.1, scope2=0.0, scope3=0.0, total=0.1)
    water = SimpleNamespace(shortfall_m3=0.0, storage=np.full(4, 10.0), provenance={"demand_m3": 2_000.0})

    terms = climate_terms(footprint, water, carbon_price=1.0, water_limit_m3=1_000.0)
    assert terms["water_binding"] is True
    assert terms["water_feasible"] is False


def test_climate_terms_none_water_is_permissive():
    footprint = Footprint(scope1=1.0, scope2=1.0, scope3=1.0, total=3.0)
    terms = climate_terms(footprint, None, carbon_price=5.0)
    assert terms == {
        "carbon_cost": 15.0,
        "water_feasible": True,
        "water_binding": False,
        "shortfall_prob": 0.0,
    }


def test_posterior_stub_conforms_to_ic1():
    assert isinstance(_TwoOptionGradePosterior(), Posterior)
