"""H6 DoD: forecast_demand's calibrated coverage, and route_distribution vs a min_cost_flow reference."""

from __future__ import annotations

import numpy as np

from mixle.distribution import forecast_demand, route_distribution
from mixle.inference.forecast import Forecast
from mixle.relations import min_cost_flow


def _simulate_regime_series(n: int, seed: int) -> np.ndarray:
    """A synthetic two-regime demand series (low/high mean, sticky Markov switching + Gaussian noise)."""
    rng = np.random.RandomState(seed)
    means = (40.0, 60.0)
    trans = np.array([[0.9, 0.1], [0.15, 0.85]])
    state = 0
    out = np.empty(n)
    for t in range(n):
        out[t] = rng.normal(means[state], 4.0)
        state = rng.choice(2, p=trans[state])
    return out


def test_forecast_demand_held_out_coverage_matches_nominal_level():
    level = 0.9
    horizon = 8
    n_history = 70
    n_trials = 25

    hits = 0
    total = 0
    for trial in range(n_trials):
        series = _simulate_regime_series(n_history + horizon, seed=1000 + trial)
        history, future = series[:n_history], series[n_history : n_history + horizon]

        f = forecast_demand(history, horizon, level=level, seed=trial)
        assert f.mean.shape == (horizon,)
        assert np.all(f.hi >= f.lo)

        hits += int(np.sum((future >= f.lo) & (future <= f.hi)))
        total += horizon

    coverage = hits / total
    assert abs(coverage - level) <= 0.05, f"coverage {coverage} not within 0.05 of level {level}"


def test_route_distribution_cost_matches_min_cost_flow_reference():
    # 2 plants feeding 3 distribution hubs, each hub's forecast demand net against its own supply.
    rng = np.random.RandomState(0)
    history = 50.0 + 10.0 * np.sin(2 * np.pi * np.arange(60) / 12.0) + rng.normal(0.0, 2.0, size=60)
    demand_forecast = forecast_demand(history.tolist(), horizon=3, level=0.9, seed=0)

    # supply_nodes = forecast mean + a zero-sum surplus/deficit pattern, so net supply is exactly
    # routable (min_cost_flow requires supply to sum to zero) regardless of the forecast's own mean.
    demand_mean = np.asarray(demand_forecast.mean)
    surplus_deficit = np.array([5.0, -8.0, 3.0])
    supply_nodes = demand_mean + surplus_deficit
    n = supply_nodes.shape[0]
    cap = np.full((n, n), 100.0)
    np.fill_diagonal(cap, 0.0)
    cost = np.array(
        [
            [0.0, 2.0, 5.0],
            [3.0, 0.0, 1.0],
            [4.0, 2.0, 0.0],
        ]
    )

    result = route_distribution(supply_nodes, demand_forecast, cost, cap)

    reference_supply = supply_nodes - np.asarray(demand_forecast.mean)
    reference = min_cost_flow(cap, cost, reference_supply)

    assert result.value == reference.value
    np.testing.assert_allclose(result.flow, reference.flow)


def test_route_distribution_rejects_misaligned_shapes():
    f = Forecast(
        mean=np.array([1.0, 2.0]),
        lo=np.array([0.0, 1.0]),
        hi=np.array([2.0, 3.0]),
        level=0.9,
        state_probs=np.zeros((2, 2)),
    )
    try:
        route_distribution(np.array([1.0, 2.0, 3.0]), f, np.zeros((3, 3)), np.zeros((3, 3)))
        raise AssertionError("expected a ValueError for misaligned supply_nodes/demand shapes")
    except ValueError:
        pass
