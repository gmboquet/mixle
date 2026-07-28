"""L1 DoD -- emissions / carbon accounting (notes/exec/workstream-L.md).

A synthetic production activity schedule (diesel combustion, purchased grid electricity, blasting
explosives, downstream haulage) run through `emissions_footprint` against hand-computed Scope 1/2/3
GHG-Protocol totals, with a Monte-Carlo 90% CI when factor uncertainties are supplied and a
content-addressed activity hash (IC-2 hashing convention) for provenance.
"""

from __future__ import annotations

import numpy as np
import pytest

from mixle.analysis.emissions import EmissionFactors, Footprint, climate_terms, emissions_footprint

ACTIVITY = {
    "diesel_L": 5_000.0,
    "grid_kWh": 20_000.0,
    "explosives_kg": 800.0,
    "transport_t_km": 15_000.0,
}

# Scope 1: direct combustion (diesel) + blasting (explosives).
# Scope 2: purchased grid electricity.
# Scope 3: downstream haulage.
FACTORS = EmissionFactors(
    scope1={"diesel_L": 2.68, "explosives_kg": 0.20},
    scope2={"grid_kWh": 0.42},
    scope3={"transport_t_km": 0.12},
    sigma={"diesel_L": 0.05, "grid_kWh": 0.01, "explosives_kg": 0.02, "transport_t_km": 0.01},
)

# Hand-computed reference (kg CO2e).
REF_SCOPE1 = 5_000.0 * 2.68 + 800.0 * 0.20  # 13560.0
REF_SCOPE2 = 20_000.0 * 0.42  # 8400.0
REF_SCOPE3 = 15_000.0 * 0.12  # 1800.0
REF_TOTAL = REF_SCOPE1 + REF_SCOPE2 + REF_SCOPE3  # 23760.0


def test_schedule_yields_scoped_footprint():
    fp = emissions_footprint(ACTIVITY, FACTORS)

    assert isinstance(fp, Footprint)
    assert fp.scope1 == pytest.approx(REF_SCOPE1)
    assert fp.scope2 == pytest.approx(REF_SCOPE2)
    assert fp.scope3 == pytest.approx(REF_SCOPE3)
    assert fp.total == pytest.approx(REF_TOTAL)
    assert fp.ci is None  # n=0 by default: no sampling requested

    ah = fp.provenance["activity_content_hash"]
    assert isinstance(ah, str) and len(ah) == 64
    assert all(c in "0123456789abcdef" for c in ah)
    assert fp.provenance["scopes"] == (1, 2, 3)

    # Deterministic: the same activity numbers always hash the same.
    fp2 = emissions_footprint(dict(ACTIVITY), FACTORS)
    assert fp2.provenance["activity_content_hash"] == ah

    # A different activity schedule hashes differently.
    other = dict(ACTIVITY, diesel_L=5_001.0)
    fp3 = emissions_footprint(other, FACTORS)
    assert fp3.provenance["activity_content_hash"] != ah


def test_ci_present_when_sigma_and_n_given():
    fp = emissions_footprint(ACTIVITY, FACTORS, n=5_000, rng=np.random.default_rng(0))
    assert fp.ci is not None
    lo, hi = fp.ci
    assert lo < REF_TOTAL < hi
    # 90% CI should be reasonably tight around the reference mean for these small factor sigmas.
    assert (hi - lo) < 0.5 * REF_TOTAL


def test_no_ci_without_sigma():
    factors_no_sigma = EmissionFactors(scope1=FACTORS.scope1, scope2=FACTORS.scope2, scope3=FACTORS.scope3)
    fp = emissions_footprint(ACTIVITY, factors_no_sigma, n=1_000, rng=np.random.default_rng(0))
    assert fp.ci is None


@pytest.mark.parametrize("n", [-1, 1.5, True, np.bool_(False)])
def test_draw_count_must_be_an_exact_nonnegative_integer(n):
    with pytest.raises(ValueError, match="n must"):
        emissions_footprint(ACTIVITY, FACTORS, n=n)


def test_zero_draws_explicitly_selects_point_estimate():
    fp = emissions_footprint(ACTIVITY, FACTORS, n=0)
    assert fp.total == pytest.approx(REF_TOTAL)
    assert fp.ci is None


def test_scopes_subset_excludes_unrequested_scope():
    fp = emissions_footprint(ACTIVITY, FACTORS, scopes=(1, 2))
    assert fp.scope1 == pytest.approx(REF_SCOPE1)
    assert fp.scope2 == pytest.approx(REF_SCOPE2)
    assert fp.scope3 == 0.0
    assert fp.total == pytest.approx(REF_SCOPE1 + REF_SCOPE2)


def test_invalid_scope_raises():
    with pytest.raises(ValueError):
        emissions_footprint(ACTIVITY, FACTORS, scopes=(1, 4))


def test_duplicate_scope_raises():
    # Regression for MXR-080-0085: a duplicate scope used to be counted once in the dict-keyed point
    # total but once per occurrence in the Monte-Carlo loop (which iterates the raw `scopes` tuple),
    # so the point estimate and the credible interval silently described two different footprints.
    # A duplicate is now rejected outright rather than being aggregated (silently or otherwise).
    with pytest.raises(ValueError, match="duplicate"):
        emissions_footprint(ACTIVITY, FACTORS, scopes=(1, 1, 2))
    # Rejected eagerly regardless of whether a Monte-Carlo interval is even requested.
    with pytest.raises(ValueError, match="duplicate"):
        emissions_footprint(ACTIVITY, FACTORS, scopes=(2, 3, 3), n=0)


def test_point_total_and_ci_describe_the_same_footprint():
    # Negative control for MXR-080-0085: with a unique (if reordered) scope set, the Monte-Carlo CI
    # must bracket the exact same point total that is reported alongside it -- the two paths must
    # describe one footprint, not two.
    fp = emissions_footprint(ACTIVITY, FACTORS, scopes=(3, 1, 2), n=20_000, rng=np.random.default_rng(0))
    assert fp.total == pytest.approx(REF_TOTAL)
    lo, hi = fp.ci
    assert lo <= fp.total <= hi


def test_negative_sigma_raises():
    # Regression for MXR-080-0085: a negative sigma used to silently take the `std > 0` else-branch
    # (an exactly-known, zero-uncertainty factor) instead of being rejected as a physically
    # meaningless "negative standard deviation".
    bad_factors = EmissionFactors(
        scope1=FACTORS.scope1, scope2=FACTORS.scope2, scope3=FACTORS.scope3, sigma={"diesel_L": -0.05}
    )
    with pytest.raises(ValueError, match="sigma"):
        emissions_footprint(ACTIVITY, bad_factors, n=100)


def test_non_finite_sigma_raises():
    bad_factors = EmissionFactors(
        scope1=FACTORS.scope1, scope2=FACTORS.scope2, scope3=FACTORS.scope3, sigma={"diesel_L": float("nan")}
    )
    with pytest.raises(ValueError, match="sigma"):
        emissions_footprint(ACTIVITY, bad_factors, n=100)


def test_negative_activity_raises():
    bad_activity = dict(ACTIVITY, diesel_L=-1.0)
    with pytest.raises(ValueError, match="activity"):
        emissions_footprint(bad_activity, FACTORS)


def test_non_finite_activity_raises():
    for bad_value in (float("nan"), float("inf")):
        bad_activity = dict(ACTIVITY, diesel_L=bad_value)
        with pytest.raises(ValueError, match="activity"):
            emissions_footprint(bad_activity, FACTORS)


def test_non_finite_factor_raises():
    bad_factors = EmissionFactors(scope1={"diesel_L": float("inf")}, scope2=FACTORS.scope2, scope3=FACTORS.scope3)
    with pytest.raises(ValueError, match="scope1"):
        emissions_footprint(ACTIVITY, bad_factors)


def test_negative_factor_is_allowed():
    # Sign is NOT constrained for emission factors themselves (unlike activity/sigma): a documented
    # carbon-negative feedstock factor is a legitimate finite negative number.
    factors_with_credit = EmissionFactors(scope1={"biochar_kg": -1.5}, scope2={}, scope3={})
    fp = emissions_footprint({"biochar_kg": 10.0}, factors_with_credit)
    assert fp.total == pytest.approx(-15.0)


def test_climate_terms_type_hints_are_resolvable():
    # climate_terms's `water` parameter used to be a bare forward-reference string naming an import
    # that never happened, so any runtime introspection (typing.get_type_hints, some doc generators,
    # dataclass-style tooling) raised NameError. It must resolve now that WaterBudget is a real name
    # in this module's namespace.
    import typing

    from mixle.analysis.emissions import WaterBudget, climate_terms

    hints = typing.get_type_hints(climate_terms)
    assert hints["water"] == (WaterBudget | None)


def test_climate_terms_still_duck_types_an_unrelated_water_object():
    class ThirdPartyWaterBudget:
        shortfall_m3 = 0.0
        storage = [1.0, 2.0, 0.0]

    fp = emissions_footprint(ACTIVITY, FACTORS)
    result = climate_terms(fp, ThirdPartyWaterBudget(), carbon_price=10.0)
    assert result["water_feasible"] is True
    assert result["shortfall_time_fraction"] == pytest.approx(1.0 / 3.0)


@pytest.mark.parametrize("price", [-1.0, float("nan"), float("inf"), True, np.array([1.0])])
def test_climate_terms_rejects_invalid_carbon_price(price):
    with pytest.raises(ValueError, match="carbon_price"):
        climate_terms(emissions_footprint(ACTIVITY, FACTORS), None, carbon_price=price)


def test_climate_terms_rejects_invalid_or_overflowing_cost():
    with pytest.raises(ValueError, match="footprint.total"):
        climate_terms(Footprint(0.0, 0.0, 0.0, float("nan")), None, carbon_price=1.0)
    with pytest.raises(ValueError, match="overflow"):
        climate_terms(Footprint(1e308, 0.0, 0.0, 1e308), None, carbon_price=1e308)


class _Water:
    """Minimal `WaterBudget`-shaped stand-in with no `storage` trajectory (summary-only caller)."""

    def __init__(self, shortfall_m3=0.0, demand_m3=None):
        self.shortfall_m3 = shortfall_m3
        if demand_m3 is not None:
            self.demand_m3 = demand_m3


def test_provenance_distinguishes_two_different_factor_models():
    # Regression for MXR-080-1583: provenance recorded only the activity hash plus the constant
    # string "caller_supplied", so the same schedule priced against two entirely different factor
    # models -- e.g. a supplier-specific table vs a national average -- produced byte-identical
    # provenance while reporting totals that differ by a factor of several.
    cheap = EmissionFactors(scope1={"a": 2.0}, scope2={}, scope3={})
    dear = EmissionFactors(scope1={"a": 9.0}, scope2={}, scope3={})

    fp_cheap = emissions_footprint({"a": 3.0}, cheap)
    fp_dear = emissions_footprint({"a": 3.0}, dear)

    assert fp_cheap.total == pytest.approx(6.0)
    assert fp_dear.total == pytest.approx(27.0)
    assert fp_cheap.provenance["activity_content_hash"] == fp_dear.provenance["activity_content_hash"]
    assert fp_cheap.provenance["factor_content_hash"] != fp_dear.provenance["factor_content_hash"]

    # The factor hash is a well-formed content digest and is stable for an equal factor model.
    fh = fp_cheap.provenance["factor_content_hash"]
    assert isinstance(fh, str) and len(fh) == 64 and all(c in "0123456789abcdef" for c in fh)
    assert (
        emissions_footprint({"a": 3.0}, EmissionFactors(scope1={"a": 2.0}, scope2={}, scope3={})).provenance[
            "factor_content_hash"
        ]
        == fh
    )


def test_provenance_records_sigma_and_sampling_configuration():
    # sigma sets the credible interval, so a change to it changes the reported footprint even when
    # the point total is identical -- it must move the factor hash. The Monte-Carlo configuration
    # behind `ci` is recorded alongside it.
    exact = EmissionFactors(scope1={"a": 2.0}, scope2={}, scope3={})
    uncertain = EmissionFactors(scope1={"a": 2.0}, scope2={}, scope3={}, sigma={"a": 0.1})

    fp_exact = emissions_footprint({"a": 3.0}, exact)
    fp_uncertain = emissions_footprint({"a": 3.0}, uncertain, n=256, rng=np.random.default_rng(0))

    assert fp_exact.total == pytest.approx(fp_uncertain.total)
    assert fp_exact.provenance["factor_content_hash"] != fp_uncertain.provenance["factor_content_hash"]
    assert fp_exact.provenance["draws"] == 0
    assert fp_exact.provenance["uncertainty_propagated"] is False
    assert fp_uncertain.provenance["draws"] == 256
    assert fp_uncertain.provenance["uncertainty_propagated"] is True
    assert fp_exact.provenance["factor_keys"] == ("a",)


@pytest.mark.parametrize("bad", [-5.0, -1e-9])
def test_negative_water_volumes_are_unknown_not_feasible(bad):
    # Regression for MXR-080-1584: shortfall, demand and water_limit_m3 are physical volumes in m3
    # and cannot be negative. A negative one used to be treated as a perfectly good finite number:
    # a negative shortfall compared false and reported a definite "feasible", while a negative
    # demand or limit manufactured a binding violation out of nothing. Malformed water evidence must
    # stay UNKNOWN, never become a definite answer in either direction.
    fp = emissions_footprint(ACTIVITY, FACTORS)

    negative_shortfall = climate_terms(fp, _Water(shortfall_m3=bad), carbon_price=1.0)
    assert negative_shortfall["water_binding"] is None
    assert negative_shortfall["water_feasible"] is None
    assert negative_shortfall["shortfall_time_fraction"] is None

    negative_demand = climate_terms(fp, _Water(demand_m3=bad), carbon_price=1.0, water_limit_m3=100.0)
    assert negative_demand["water_binding"] is None
    assert negative_demand["water_feasible"] is None

    negative_limit = climate_terms(fp, _Water(demand_m3=10.0), carbon_price=1.0, water_limit_m3=bad)
    assert negative_limit["water_binding"] is None
    assert negative_limit["water_feasible"] is None


def test_valid_water_volumes_still_give_definite_answers():
    # Negative control for MXR-080-1584: the nonnegativity guard must not swallow legitimate
    # evidence -- zero is a valid volume and the ordinary binding/non-binding verdicts stand.
    fp = emissions_footprint(ACTIVITY, FACTORS)

    ok = climate_terms(fp, _Water(shortfall_m3=0.0, demand_m3=10.0), carbon_price=1.0, water_limit_m3=100.0)
    assert ok["water_binding"] is False
    assert ok["water_feasible"] is True
    assert ok["shortfall_time_fraction"] == 0.0

    short = climate_terms(fp, _Water(shortfall_m3=5.0), carbon_price=1.0)
    assert short["water_binding"] is True
    assert short["water_feasible"] is False
    assert short["shortfall_time_fraction"] == 1.0

    over = climate_terms(fp, _Water(shortfall_m3=0.0, demand_m3=250.0), carbon_price=1.0, water_limit_m3=100.0)
    assert over["water_binding"] is True
    assert over["water_feasible"] is False
