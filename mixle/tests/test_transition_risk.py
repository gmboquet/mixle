"""L3 DoD -- transition risk (notes/exec/workstream-L.md).

A high carbon-price scenario should re-rank strictly below a low carbon-price scenario on
carbon-adjusted mean NPV once :func:`transition_risk` prices L1's :class:`Footprint` against a set of
carbon-price/policy scenario paths and subtracts the result from J2-shaped ``npv_samples``; the gap
between the two scenarios' carbon-adjusted mean NPV should scale linearly with ``footprint.total``
(a bigger footprint pays more carbon cost under the same price paths). The returned object also has to
satisfy the IC-1 ``DerivedQuantity`` shape (``samples``, ``prior_dominated``, ``credible_interval``) so
the re-ranking stays uncertainty-aware rather than collapsing straight to a point estimate.
"""

from __future__ import annotations

import numpy as np
import pytest

from mixle.analysis.emissions import Footprint, transition_risk


def _footprint(total: float) -> Footprint:
    return Footprint(
        scope1=total,
        scope2=0.0,
        scope3=0.0,
        total=total,
        ci=None,
        provenance={"factor_source": "test", "activity_content_hash": "0" * 64, "scopes": (1, 2, 3)},
    )


LOW_IDX, HIGH_IDX = 0, 1


def _price_paths() -> np.ndarray:
    # Two flat 10-period carbon-price scenarios ($/tCO2e): a low-price and a high-price policy path.
    low = np.full(10, 20.0)
    high = np.full(10, 120.0)
    return np.stack([low, high])


def test_carbon_price_reranks_value():
    rng = np.random.default_rng(0)
    npv_samples = rng.normal(loc=1_000_000.0, scale=50_000.0, size=5_000)
    carbon_price_paths = _price_paths()
    discount = 1.0 / 1.08 ** np.arange(10)

    footprint = _footprint(1_000.0)  # tCO2e
    result = transition_risk(footprint, carbon_price_paths, npv_samples=npv_samples, discount=discount)

    # IC-1 DerivedQuantity shape.
    assert isinstance(result.samples, np.ndarray)
    assert result.samples.shape == (5_000, 2)
    assert result.prior_dominated is False
    lo, hi = result.credible_interval(0.9)
    assert lo.shape == (2,) and hi.shape == (2,)
    assert np.all(lo <= hi)

    # High carbon price ranks strictly below low carbon price on mean carbon-adjusted NPV.
    assert result.scenario_mean[HIGH_IDX] < result.scenario_mean[LOW_IDX]
    assert result.ranking.index(LOW_IDX) < result.ranking.index(HIGH_IDX)

    gap_small = result.scenario_mean[LOW_IDX] - result.scenario_mean[HIGH_IDX]
    assert gap_small > 0.0

    # The gap scales with footprint.total: a 5x bigger footprint pays 5x more carbon cost under the
    # same price paths, so the low-vs-high scenario gap grows by the same factor.
    bigger_footprint = _footprint(5_000.0)
    result2 = transition_risk(bigger_footprint, carbon_price_paths, npv_samples=npv_samples, discount=discount)
    gap_big = result2.scenario_mean[LOW_IDX] - result2.scenario_mean[HIGH_IDX]
    assert gap_big == pytest.approx(5.0 * gap_small)


def test_zero_footprint_is_carbon_price_invariant():
    rng = np.random.default_rng(1)
    npv_samples = rng.normal(loc=500_000.0, scale=10_000.0, size=2_000)
    carbon_price_paths = _price_paths()

    result = transition_risk(_footprint(0.0), carbon_price_paths, npv_samples=npv_samples)
    assert result.scenario_mean[LOW_IDX] == pytest.approx(result.scenario_mean[HIGH_IDX])
    assert result.scenario_mean[LOW_IDX] == pytest.approx(float(np.mean(npv_samples)))


def test_undiscounted_matches_explicit_flat_discount():
    rng = np.random.default_rng(2)
    npv_samples = rng.normal(loc=200_000.0, scale=5_000.0, size=1_000)
    carbon_price_paths = _price_paths()

    no_discount = transition_risk(_footprint(500.0), carbon_price_paths, npv_samples=npv_samples)
    flat_discount = transition_risk(
        _footprint(500.0), carbon_price_paths, npv_samples=npv_samples, discount=np.ones(10)
    )
    assert no_discount.scenario_mean == pytest.approx(flat_discount.scenario_mean)


# --- MXR-080-0086: empty/malformed scenarios and NPV samples must be rejected, not silently ranked. ---


def test_empty_npv_samples_raises():
    # Regression: construction used to succeed with a NaN scenario_mean/ranking and only fail later,
    # when a caller asked for credible_interval().
    with pytest.raises(ValueError, match="npv_samples"):
        transition_risk(_footprint(100.0), _price_paths(), npv_samples=np.array([]))


def test_empty_scenario_set_raises():
    with pytest.raises(ValueError, match="scenario"):
        transition_risk(_footprint(100.0), np.array([]), npv_samples=np.array([1.0, 2.0, 3.0]))


def test_empty_period_axis_raises():
    with pytest.raises(ValueError, match="period"):
        transition_risk(_footprint(100.0), np.zeros((3, 0)), npv_samples=np.array([1.0, 2.0, 3.0]))


def test_negative_or_non_finite_prices_raise():
    for bad_prices in (np.array([-50.0, 20.0]), np.array([np.nan, 20.0]), np.array([np.inf, 20.0])):
        with pytest.raises(ValueError, match="carbon_price_paths"):
            transition_risk(_footprint(100.0), bad_prices, npv_samples=np.array([1.0, 2.0, 3.0]))


def test_negative_or_non_finite_discount_raises():
    carbon_price_paths = np.array([20.0, 120.0])  # (k,) -> 1 period each
    for bad_discount in (np.array([-1.0]), np.array([np.nan]), np.array([np.inf])):
        with pytest.raises(ValueError, match="discount"):
            transition_risk(
                _footprint(100.0), carbon_price_paths, npv_samples=np.array([1.0, 2.0, 3.0]), discount=bad_discount
            )


def test_non_finite_footprint_total_raises():
    for bad_total in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="footprint.total"):
            transition_risk(_footprint(bad_total), _price_paths(), npv_samples=np.array([1.0, 2.0, 3.0]))


def test_multi_dimensional_npv_samples_raises_instead_of_flattening():
    # Regression: `npv_samples` used to be coerced with `.reshape(-1)`, so a meaningfully-shaped
    # (n_scenarios, n_draws) matrix was silently flattened into one long, structure-free set of
    # draws instead of being rejected. The documented contract is a single 1-D (n,) baseline
    # distribution shared across every scenario, not one draw set per scenario.
    rng = np.random.default_rng(3)
    npv_2d = rng.normal(loc=1_000_000.0, scale=50_000.0, size=(2, 5_000))  # (n_scenarios, n_draws)
    with pytest.raises(ValueError, match="npv_samples"):
        transition_risk(_footprint(1_000.0), _price_paths(), npv_samples=npv_2d)


def test_well_shaped_multi_scenario_input_still_ranks_correctly():
    # Negative control for MXR-080-0086: a properly-shaped 1-D npv_samples array against a
    # nonempty, finite, nonnegative scenario matrix must still construct and rank normally.
    rng = np.random.default_rng(4)
    npv_samples = rng.normal(loc=1_000_000.0, scale=50_000.0, size=5_000)
    result = transition_risk(_footprint(1_000.0), _price_paths(), npv_samples=npv_samples)
    assert result.samples.shape == (5_000, 2)
    assert result.scenario_mean[HIGH_IDX] < result.scenario_mean[LOW_IDX]
    lo, hi = result.credible_interval(0.9)
    assert np.all(lo <= hi)
