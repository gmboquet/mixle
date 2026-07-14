"""J5 -- risk / tail metrics DoD: VaR/CVaR ordering + hand-computed reference, stress ranking."""

from __future__ import annotations

import numpy as np
import pytest

from mixle.inference.risk import conditional_value_at_risk, stress_rank, value_at_risk

# A fixed, hand-typed sample array standing in for a J2 NPV Monte-Carlo distribution -- no rng
# involved, so the reference values below can be checked by eye against the sorted array.
FIXED_SAMPLES = np.array(
    [
        120.0, 95.0, 110.0, 88.0, 130.0, 102.0, 76.0, 140.0, 99.0, 84.0,
        115.0, 60.0, 105.0, 92.0, 125.0, -40.0, 108.0, 70.0, 118.0, -10.0,
    ]
)  # fmt: skip


def test_var_matches_hand_computed_quantile():
    # 1 - alpha = 0.2 -> the 20th percentile of 20 sorted values. numpy's default (linear)
    # interpolation at position 0.2*(20-1) = 3.8 sits between the 4th and 5th smallest values.
    alpha = 0.8
    sorted_x = np.sort(FIXED_SAMPLES)
    lo, hi = sorted_x[3], sorted_x[4]
    expected_quantile = lo + 0.8 * (hi - lo)
    expected_var = -expected_quantile

    var = value_at_risk(FIXED_SAMPLES, alpha=alpha)

    assert var == pytest.approx(expected_var)


def test_cvar_matches_hand_computed_tail_mean_and_dominates_var():
    # 20 samples -> the alpha=0.8 tail has only ~4 points, which is exactly the "sparse" case the
    # module refines with a GPD fit (covered separately below). min_tail=1 pins this test to the
    # raw formula (bullet 2 of the algorithm) so it checks the plain tail-mean arithmetic.
    alpha = 0.8
    var = value_at_risk(FIXED_SAMPLES, alpha=alpha)
    expected_tail = FIXED_SAMPLES[FIXED_SAMPLES <= -var]
    expected_cvar = -expected_tail.mean()

    cvar = conditional_value_at_risk(FIXED_SAMPLES, alpha=alpha, min_tail=1)

    assert cvar == pytest.approx(expected_cvar)
    assert cvar >= var  # CVaR is always at least as conservative as VaR


def test_var_cvar_on_large_j2_like_npv_distribution():
    # A larger, more realistic NPV-shaped distribution (asymmetric downside from a lognormal-cost /
    # normal-price mix), independently computed with raw numpy rather than the module under test.
    rng = np.random.default_rng(0)
    price_upside = rng.normal(loc=150.0, scale=40.0, size=8000)
    cost_shock = rng.lognormal(mean=3.0, sigma=0.6, size=8000)
    npv = price_upside - cost_shock

    alpha = 0.95
    expected_var = float(-np.quantile(npv, 1 - alpha))
    expected_cvar = float(-npv[npv <= -expected_var].mean())

    var = value_at_risk(npv, alpha=alpha)
    cvar = conditional_value_at_risk(npv, alpha=alpha)

    assert var == pytest.approx(expected_var)
    assert cvar == pytest.approx(expected_cvar)
    assert cvar >= var


def test_cvar_sparse_tail_refinement_stays_finite_and_conservative():
    # Small sample -> tiny tail -> triggers the GPD refinement path. We don't hand-duplicate the
    # GPD fit here; we only assert the invariants the refinement must preserve.
    rng = np.random.default_rng(1)
    small = rng.normal(loc=50.0, scale=15.0, size=40)

    alpha = 0.95
    var = value_at_risk(small, alpha=alpha)
    cvar = conditional_value_at_risk(small, alpha=alpha)

    assert np.isfinite(var)
    assert np.isfinite(cvar)
    assert cvar >= var


def test_value_at_risk_rejects_invalid_alpha():
    with pytest.raises(ValueError):
        value_at_risk(FIXED_SAMPLES, alpha=1.0)
    with pytest.raises(ValueError):
        value_at_risk(FIXED_SAMPLES, alpha=0.0)


def test_stress_rank_orders_worst_scenario_first():
    scenarios = {
        "baseline": 100.0,
        "low_grade": 40.0,
        "price_crash": -25.0,
        "carbon_spike": 60.0,
    }

    ranked = stress_rank(scenarios)

    assert [name for name, _ in ranked] == ["price_crash", "low_grade", "carbon_spike", "baseline"]
    # losses are -value
    assert dict(ranked)["price_crash"] == pytest.approx(25.0)
    assert dict(ranked)["baseline"] == pytest.approx(-100.0)


def test_stress_rank_accepts_sample_arrays_per_scenario():
    scenarios = {
        "baseline": np.array([100.0, 110.0, 90.0]),
        "price_crash": np.array([-30.0, -20.0, -10.0]),
    }

    ranked = stress_rank(scenarios)

    assert ranked[0][0] == "price_crash"
    assert ranked[0][1] == pytest.approx(20.0)


def test_stress_rank_rejects_empty_scenarios():
    with pytest.raises(ValueError):
        stress_rank({})
