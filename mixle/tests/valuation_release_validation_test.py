"""Focused release-contract probes for extraction costs and discounted valuation."""

from __future__ import annotations

import numpy as np
import pytest

from mixle.analysis.valuation import NPVDistribution, capex_opex, cost_curve, monte_carlo_npv


class _FixedPosterior:
    def __init__(self, values) -> None:
        self.values = np.asarray(values, dtype=float)

    def samples(self, n, rng):
        return np.broadcast_to(self.values, (n, self.values.size)).copy()


def _npv(
    posterior=None,
    *,
    prices=None,
    cost_model=lambda _period: 0.0,
    schedule=None,
    discount_rate=0.0,
    n=2,
):
    return monte_carlo_npv(
        _FixedPosterior([1.0]) if posterior is None else posterior,
        np.ones((n, 1)) if prices is None else prices,
        cost_model,
        {"tonnage": np.ones(1)} if schedule is None else schedule,
        discount_rate=discount_rate,
        n=n,
        rng=np.random.default_rng(0),
    )


@pytest.mark.parametrize(
    "call",
    [
        lambda: cost_curve(
            1e308,
            1.0,
            1.0,
            params={"haul_cost_per_m": 1e308},
        ),
        lambda: cost_curve(
            1.0,
            1.0,
            1e308,
            params={"throughput_scale_coef": 1e308, "design_capacity": 1.0},
        ),
        lambda: capex_opex(
            {"tonnage": [1e308], "depth": 1.0, "grade": 1.0, "throughput": 1.0},
            params={"base_cost": 1e308},
        ),
        lambda: capex_opex(
            {"tonnage": [1e308], "depth": 1.0, "grade": 1.0, "throughput": 1.0},
            params={"capex_per_tonne": 1e308},
        ),
    ],
)
def test_cost_and_capital_arithmetic_must_remain_representable(call):
    with pytest.raises(ValueError, match="representable"):
        call()


@pytest.mark.parametrize(
    "call",
    [
        lambda: cost_curve(1.0, 1.0, 1.0, params={"base_cots": 100.0}),
        lambda: capex_opex(
            {"tonnage": [1.0], "depth": 1.0, "grade": 1.0, "throughput": 1.0},
            params={"capex_fiixed": 100.0},
        ),
    ],
)
def test_public_valuation_entry_points_reject_unknown_parameters(call):
    with pytest.raises(ValueError, match="unknown parameter"):
        call()


def test_capital_schedule_has_one_value_per_operating_period():
    with pytest.raises(ValueError, match="same one-dimensional period shape"):
        capex_opex(
            {
                "tonnage": [1.0, 2.0],
                "depth": 1.0,
                "grade": 1.0,
                "throughput": 1.0,
                "capex_schedule": [[1.0, 2.0], [3.0, 4.0]],
            },
            params={},
        )


def test_keyword_only_cost_configuration_is_not_misclassified_as_tonnage():
    def cost(period, *, adjustment=0.0):
        return period + adjustment

    result = _npv(cost_model=cost)
    assert np.isfinite(result.samples).all()


def test_ambiguous_optional_tonnage_cost_signature_is_rejected():
    def cost(period, tonnage=0.0):
        return period + tonnage

    with pytest.raises(TypeError, match="ambiguous"):
        _npv(cost_model=cost)


def test_dcf_rejects_revenue_overflow_from_finite_inputs():
    with pytest.raises(ValueError, match="period revenue"):
        _npv(
            posterior=_FixedPosterior([1e308]),
            prices=np.full((2, 1), 1e308),
        )


def test_dcf_rejects_unrepresentable_discount_factors():
    n_periods = 200
    with pytest.raises(ValueError, match="discount factors"):
        _npv(
            prices=np.ones((2, n_periods)),
            schedule={"tonnage": np.ones(n_periods)},
            discount_rate=-0.999999,
        )


def test_dcf_rejects_unrepresentable_sensitivity_variance():
    with pytest.raises(ValueError, match="total NPV variance|Sobol"):
        _npv(prices=np.array([[1e308], [5e307]]))


@pytest.mark.parametrize(
    "schedule",
    [
        {"tonnage": np.ones((2, 2))},
        {"tonnage": np.ones(2), "capex": np.ones((2, 1))},
    ],
)
def test_dcf_requires_one_dimensional_period_schedules(schedule):
    with pytest.raises(ValueError, match="one-dimensional|same shape"):
        _npv(prices=np.ones((2, 2)), schedule=schedule)


def test_npv_distribution_owns_immutable_consistent_evidence():
    samples = np.array([1.0, 2.0, 3.0])
    sensitivity = {"grade": 0.4}
    result = NPVDistribution(samples, 2.0, 1.2, 2.0, 2.8, sensitivity)
    samples[:] = 100.0
    sensitivity["grade"] = 0.9
    np.testing.assert_array_equal(result.samples, [1.0, 2.0, 3.0])
    assert result.sensitivity["grade"] == 0.4
    with pytest.raises(ValueError):
        result.samples[0] = 9.0
    with pytest.raises(TypeError):
        result.sensitivity["grade"] = 0.9


def test_npv_distribution_rejects_summaries_inconsistent_with_samples():
    with pytest.raises(ValueError, match="summaries must agree"):
        NPVDistribution(np.array([1.0, 2.0, 3.0]), 99.0, 1.2, 2.0, 2.8, {})
