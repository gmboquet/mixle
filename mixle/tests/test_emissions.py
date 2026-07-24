"""L1 DoD -- emissions / carbon accounting (notes/exec/workstream-L.md).

A synthetic production activity schedule (diesel combustion, purchased grid electricity, blasting
explosives, downstream haulage) run through `emissions_footprint` against hand-computed Scope 1/2/3
GHG-Protocol totals, with a Monte-Carlo 90% CI when factor uncertainties are supplied and a
content-addressed activity hash (IC-2 hashing convention) for provenance.
"""

from __future__ import annotations

import numpy as np
import pytest

from mixle.analysis.emissions import EmissionFactors, Footprint, emissions_footprint

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
    from mixle.analysis.emissions import climate_terms

    class ThirdPartyWaterBudget:
        shortfall_m3 = 0.0
        storage = [1.0, 2.0, 0.0]

    fp = emissions_footprint(ACTIVITY, FACTORS)
    result = climate_terms(fp, ThirdPartyWaterBudget(), carbon_price=10.0)
    assert result["water_feasible"] is True
    assert result["shortfall_prob"] == pytest.approx(1.0 / 3.0)
