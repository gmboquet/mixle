"""H6 DoD: forecast_demand's calibrated coverage, and route_distribution vs a min_cost_flow reference.

Renamed from mixle.distribution to mixle.fulfillment -- "distribution" collided with mixle.dist (the
object-namespace alias for mixle.stats): same word, unrelated meanings (probability distribution vs.
supply-chain distribution/routing). The H6 label is a stable worklist identifier and stays put; only
the module name changed.
"""

from __future__ import annotations

import numpy as np

from mixle.fulfillment import forecast_demand, route_distribution
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


def _hub_network(n=3, capacity=100.0):
    cap = np.full((n, n), capacity)
    np.fill_diagonal(cap, 0.0)
    cost = np.array(
        [
            [0.0, 2.0, 5.0],
            [3.0, 0.0, 1.0],
            [4.0, 2.0, 0.0],
        ]
    )
    return cap, cost


def _forecast(mean, hi=None):
    mean = np.asarray(mean, dtype=float)
    hi = mean if hi is None else np.asarray(hi, dtype=float)
    return Forecast(mean=mean, lo=mean, hi=hi, level=0.9, state_probs=np.zeros((len(mean), 2)))


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
    cap, cost = _hub_network(supply_nodes.shape[0])

    result = route_distribution(supply_nodes, demand_forecast, cost, cap, demand_nodes=[0, 1, 2], risk="expected_value")

    reference_supply = supply_nodes - np.asarray(demand_forecast.mean)
    reference = min_cost_flow(cap, cost, reference_supply)

    assert result.value == reference.value
    np.testing.assert_allclose(result.flow, reference.flow)


def test_route_distribution_rejects_misaligned_shapes():
    f = _forecast([1.0, 2.0])
    try:
        route_distribution(
            np.array([1.0, 2.0, 3.0]),
            f,
            np.zeros((3, 3)),
            np.zeros((3, 3)),
            demand_nodes=[0, 1, 2],
            risk="expected_value",
        )
        raise AssertionError("expected a ValueError for misaligned demand_nodes/forecast lengths")
    except ValueError:
        pass


def test_forecast_coordinates_must_be_mapped_onto_network_nodes_explicitly():
    """MXR-080-1698: forecast_demand returns one value per future TIME STEP, but route_distribution
    required that vector to align one-to-one with supply_nodes by position and used its length as the
    dimension of a spatial graph -- so a horizon of four silently became four plant/depot/customer
    nodes and the solver returned a valid flow for a network nobody described."""
    cap, cost = _hub_network()
    supply_nodes = np.array([10.0, 0.0, 0.0])
    horizon_forecast = _forecast([4.0, 6.0])  # two TIME STEPS, not two nodes

    # the mapping is required: it cannot be inferred from a temporal forecast
    try:
        route_distribution(supply_nodes, horizon_forecast, cost, cap, risk="expected_value")
        raise AssertionError("expected demand_nodes to be required")
    except TypeError:
        pass

    # and it is validated against the real network, not assumed to be positional
    for bad in ([0, 5], [0, 0], [0, 1, 2]):
        try:
            route_distribution(supply_nodes, horizon_forecast, cost, cap, demand_nodes=bad, risk="expected_value")
            raise AssertionError(f"expected a ValueError for demand_nodes={bad}")
        except ValueError:
            pass

    # a stated mapping puts each coordinate's demand on the node the caller named, not on node i
    routed = route_distribution(supply_nodes, horizon_forecast, cost, cap, demand_nodes=[2, 1], risk="expected_value")
    reference = min_cost_flow(cap, cost, supply_nodes - np.array([0.0, 6.0, 4.0]))
    assert routed.value == reference.value


def test_dispatch_requires_an_explicit_risk_policy_and_can_use_the_band():
    """MXR-080-1699: dispatch read only Forecast.mean, so two forecasts with the same mean [0, 10] but
    upper bounds [0, 10] and [1000, 1000] produced the identical cost-10 flow -- the second can
    represent catastrophic unmet-demand risk, yet no service level, shortage penalty or band reached
    the optimizer."""
    cap, cost = _hub_network(capacity=5000.0)
    tight = _forecast([0.0, 10.0], hi=[0.0, 10.0])
    wide = _forecast([0.0, 10.0], hi=[1000.0, 1000.0])

    def routed(forecast, risk):
        # source node 0 carries exactly what this policy dispatches, so min_cost_flow stays balanced
        quantity = forecast.mean if risk == "expected_value" else forecast.hi
        supply_nodes = np.array([float(np.sum(quantity)), 0.0, 0.0])
        return route_distribution(supply_nodes, forecast, cost, cap, demand_nodes=[1, 2], risk=risk)

    try:
        route_distribution(np.array([10.0, 0.0, -10.0]), tight, cost, cap, demand_nodes=[1, 2])
        raise AssertionError("expected risk to be required")
    except TypeError:
        pass
    try:
        route_distribution(np.array([10.0, 0.0, 0.0]), tight, cost, cap, demand_nodes=[1, 2], risk="whatever")
        raise AssertionError("expected a ValueError for an unknown risk policy")
    except ValueError:
        pass

    # an expected-value dispatch is labelled as one and, being the point forecast, is band-blind
    assert routed(tight, "expected_value").value == routed(wide, "expected_value").value

    # covering the band is a different dispatch: the uncertainty now reaches the optimizer
    band_tight = routed(tight, "cover_band")
    band_wide = routed(wide, "cover_band")
    assert band_tight.value == routed(tight, "expected_value").value  # a zero-width band costs the same
    assert band_wide.value > band_tight.value  # a catastrophic band does not


def test_forecast_demand_does_not_advertise_conformal_coverage_it_cannot_establish():
    # MXR-080-1679: the band was described as retaining nominal split-conformal coverage, but its
    # calibration cases are successive horizons from one held-out tail -- ordered, dependent and
    # generally non-exchangeable -- and one constant half-width is transferred to every horizon of a
    # different model refit on the full series. The returned receipt has to say so.
    history = _simulate_regime_series(70, seed=7)
    f = forecast_demand(history, horizon=4, level=0.9, seed=0)

    assert isinstance(f, Forecast)  # still a drop-in Forecast for route_distribution
    receipt = f.calibration
    assert receipt is not None
    assert receipt.guarantee == "empirical"
    assert receipt.calibration_origins == 1
    assert receipt.horizon_specific is False
    assert receipt.n_calibration >= 4
    assert receipt.half_width > 0.0
    assert any("exchangeable" in item for item in receipt.assumptions)
