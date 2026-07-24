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

    # shortfall_time_fraction reads the water budget's own step trajectory: 3 of 12 dirty steps sit at
    # zero. It is a single-trajectory duration statistic, not a probability (MXR-080-0087).
    assert clean["shortfall_time_fraction"] == 0.0
    assert dirty["shortfall_time_fraction"] == pytest.approx(3.0 / 12.0)


def test_climate_terms_water_limit_alone_can_bind():
    footprint = Footprint(scope1=0.1, scope2=0.0, scope3=0.0, total=0.1)
    water = SimpleNamespace(shortfall_m3=0.0, storage=np.full(4, 10.0), provenance={"demand_m3": 2_000.0})

    terms = climate_terms(footprint, water, carbon_price=1.0, water_limit_m3=1_000.0)
    assert terms["water_binding"] is True
    assert terms["water_feasible"] is False


def test_climate_terms_none_water_is_unknown_not_permissive():
    # Regression for MXR-080-0087: an absent water budget used to default to "feasible" (a silent,
    # permissive pass). It is now an explicit unknown/not-evaluated state -- carbon_cost is still a
    # definite number (it never depended on water evidence), but the water terms are all None rather
    # than a fabricated "known feasible".
    footprint = Footprint(scope1=1.0, scope2=1.0, scope3=1.0, total=3.0)
    terms = climate_terms(footprint, None, carbon_price=5.0)
    assert terms == {
        "carbon_cost": 15.0,
        "water_feasible": None,
        "water_binding": None,
        "shortfall_time_fraction": None,
    }


def test_non_finite_shortfall_m3_is_unknown_not_feasible():
    # Regression for MXR-080-0087: `shortfall_m3 > 0.0` is False for NaN (NaN-comparison-is-False), so
    # an invalid/unknown shortfall reading used to silently mark the option feasible. It must now come
    # back as an explicit unknown rather than a permissive pass.
    footprint = Footprint(scope1=1.0, scope2=0.0, scope3=0.0, total=1.0)
    water = SimpleNamespace(shortfall_m3=float("nan"))
    terms = climate_terms(footprint, water, carbon_price=1.0)
    assert terms["water_feasible"] is None
    assert terms["water_binding"] is None
    assert terms["shortfall_time_fraction"] is None


def test_non_finite_water_limit_is_unknown_not_feasible():
    # A non-finite water_limit_m3 makes the demand-vs-limit check unevaluable; it must not silently
    # fall through to "not binding".
    footprint = Footprint(scope1=1.0, scope2=0.0, scope3=0.0, total=1.0)
    water = SimpleNamespace(shortfall_m3=0.0, demand_m3=5_000.0)
    terms = climate_terms(footprint, water, carbon_price=1.0, water_limit_m3=float("nan"))
    assert terms["water_feasible"] is None
    assert terms["water_binding"] is None


def test_non_finite_demand_is_unknown_not_feasible():
    footprint = Footprint(scope1=1.0, scope2=0.0, scope3=0.0, total=1.0)
    water = SimpleNamespace(shortfall_m3=0.0, demand_m3=float("nan"))
    terms = climate_terms(footprint, water, carbon_price=1.0, water_limit_m3=1_000.0)
    assert terms["water_feasible"] is None
    assert terms["water_binding"] is None


def test_confirmed_binding_survives_an_unknown_from_the_other_check():
    # Three-valued OR: a confirmed shortfall (True) must not be erased by an unrelated unevaluable
    # water-limit check (Unknown) -- True dominates Unknown dominates False.
    footprint = Footprint(scope1=1.0, scope2=0.0, scope3=0.0, total=1.0)
    water = SimpleNamespace(shortfall_m3=250.0)  # confirmed binding
    terms = climate_terms(footprint, water, carbon_price=1.0, water_limit_m3=float("nan"))  # unknown check
    assert terms["water_binding"] is True
    assert terms["water_feasible"] is False


def test_non_finite_storage_entries_make_shortfall_time_fraction_unknown():
    # A storage trajectory with non-finite entries can't yield a reliable duration statistic -- NaN
    # entries must not silently compare False against the zero threshold and dilute the fraction.
    footprint = Footprint(scope1=1.0, scope2=0.0, scope3=0.0, total=1.0)
    water = SimpleNamespace(shortfall_m3=1.0, storage=np.array([0.0, 0.0, float("nan"), 5.0]))
    terms = climate_terms(footprint, water, carbon_price=1.0)
    assert terms["shortfall_time_fraction"] is None


def test_shortfall_prob_key_no_longer_exists():
    # The old key name claimed a probability it never computed; confirm the rename actually took.
    footprint = Footprint(scope1=1.0, scope2=0.0, scope3=0.0, total=1.0)
    water = SimpleNamespace(shortfall_m3=0.0, storage=np.full(4, 10.0))
    terms = climate_terms(footprint, water, carbon_price=1.0)
    assert "shortfall_prob" not in terms
    assert "shortfall_time_fraction" in terms


def test_shortfall_probability_over_ensemble_matches_known_fraction():
    # A genuine cross-realization probability, given an explicit ensemble: 2 of 4 independent
    # trajectories hit a binding zero storage at some point -> analytically expected probability 0.5.
    from mixle.analysis.emissions import _shortfall_probability_over_ensemble

    ensemble = np.array(
        [
            [10.0, 5.0, 0.0, 3.0],  # hits shortfall
            [10.0, 8.0, 6.0, 4.0],  # never hits shortfall
            [1.0, 0.0, 0.0, 0.0],  # hits shortfall
            [5.0, 5.0, 5.0, 5.0],  # never hits shortfall
        ]
    )
    assert _shortfall_probability_over_ensemble(ensemble) == pytest.approx(0.5)


def test_shortfall_probability_over_ensemble_rejects_a_single_trajectory():
    # A single 1-D trajectory is a duration statistic (_shortfall_time_fraction), not an ensemble.
    from mixle.analysis.emissions import _shortfall_probability_over_ensemble

    with pytest.raises(ValueError, match="2-D"):
        _shortfall_probability_over_ensemble(np.array([1.0, 2.0, 0.0, 3.0]))


def test_posterior_stub_conforms_to_ic1():
    assert isinstance(_TwoOptionGradePosterior(), Posterior)
