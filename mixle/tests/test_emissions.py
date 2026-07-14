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
