"""Digital-twin simulation of the pipeline (H8): period-stepped re-solve + scenario intervention."""

from __future__ import annotations

import numpy as np

from mixle.pipeline_twin import _SLACK_CAPACITY, _SLACK_COST, PipelineTwin, build_twin

# Nodes: 0 mine0, 1 mine1, 2 plant0, 3 plant1, 4 customer0, 5 customer1.
N = 6


def _network():
    """A 2-mine/2-plant/2-customer network. Plant0's line to customer0 has a generous nameplate
    arc capacity (100) but H5 slurry-transport physics binds the real throughput to 12 -- well
    below customer0's demand of 18 -- so it saturates every period absent an inter-plant transfer.
    Plant1 already has an (idle, in the base case) arc to customer0 that a transfer arc can feed."""
    cap = np.zeros((N, N))
    cost = np.zeros((N, N))

    def arc(u, v, c, w):
        cap[u, v] = c
        cost[u, v] = w

    arc(0, 2, 100.0, 0.1)  # mine0 -> plant0
    arc(1, 3, 100.0, 0.1)  # mine1 -> plant1
    arc(2, 4, 100.0, 5.0)  # plant0 -> customer0 (nameplate; H5 transport caps the real throughput)
    arc(3, 5, 100.0, 1.0)  # plant1 -> customer1
    arc(3, 4, 100.0, 1.0)  # plant1 -> customer0 (present but idle without a transfer feeding plant1)

    supply = np.zeros(N)
    supply[0] = 18.0
    supply[1] = 10.0
    supply[4] = -18.0
    supply[5] = -10.0

    network = {
        "cap": cap,
        "cost": cost,
        "supply": supply,
        "supply_nodes": [0, 1],
        "demand_nodes": [4, 5],
    }
    transport = {(2, 4): 12.0}  # H5-derived slurry-line throughput ceiling, plain-array input
    return network, transport


def test_twin_reproduces_and_relieves_a_saturated_plant_arc():
    network, transport = _network()
    twin = build_twin(network, transport, seed=0)

    base = twin.run(6)
    assert base["bottleneck_arc"] == (2, 4)
    assert np.all(base["bottleneck_utilization"] > 0.999)  # arc (2,4) runs pinned at its cap every period
    assert base["queue"][-1] > 0.0  # unmet customer0 demand accumulates as backlog

    relieved = twin.scenario("inter_plant_transfer", {"add_arc": {(2, 3): (20.0, 0.3)}}).run(
        6, scenario="inter_plant_transfer"
    )

    assert relieved["throughput"][-1] > base["throughput"][-1]
    assert relieved["queue"][-1] < base["queue"][-1]

    base_util_24 = base["utilization"][:, 2, 4]
    relieved_util_24 = relieved["utilization"][:, 2, 4]
    assert np.mean(relieved_util_24) < np.mean(base_util_24) - 1.0e-6


def test_build_twin_returns_pipeline_twin_and_scenario_is_immutable():
    network, transport = _network()
    twin = build_twin(network, transport, seed=1)
    assert isinstance(twin, PipelineTwin)

    out = twin.run(1)
    assert {"throughput", "queue", "utilization", "bottleneck_arc", "bottleneck_utilization"} <= set(out)

    twin.scenario("plant_down", {"zero_capacity_node": [3]})
    # the base twin (no scenario requested) is unaffected by having registered one
    still_base = twin.run(1)
    assert still_base["bottleneck_arc"] == out["bottleneck_arc"]


def test_unregistered_scenario_name_raises():
    network, transport = _network()
    twin = build_twin(network, transport, seed=0)
    try:
        twin.run(1, scenario="nonexistent")
        raise AssertionError("expected KeyError for an unregistered scenario name")
    except KeyError:
        pass


def test_slack_is_never_preferred_over_a_costlier_real_route():
    """A single real arc whose unit cost exceeds the fixed slack round-trip (2 * old _SLACK_COST =
    20_000) used to lose to slack outright: the solver would route entirely through the universal
    slack node instead of the real, demand-satisfying arc, because a fixed _SLACK_COST cannot
    dominate an arbitrarily expensive real network. Slack must only ever carry what the real network
    genuinely could not -- a real route that CAN satisfy demand must always win, however expensive."""
    n = 2
    cap = np.zeros((n, n))
    cost = np.zeros((n, n))
    cap[0, 1] = 1.0
    cost[0, 1] = 2.0 * _SLACK_COST + 10_000.0  # strictly above the old fixed slack round-trip cost
    supply = np.zeros(n)
    supply[0] = 1.0
    supply[1] = -1.0

    network = {
        "cap": cap,
        "cost": cost,
        "supply": supply,
        "supply_nodes": [0],
        "demand_nodes": [1],
    }
    twin = build_twin(network, {}, seed=0)
    out = twin.run(1)

    assert out["throughput"][0] == 1.0  # the real arc satisfied the one unit of demand
    assert out["queue"][-1] == 0.0  # nothing went unmet
    assert out["bottleneck_arc"] == (0, 1)  # the real arc, not slack, carried the flow
    assert out["bottleneck_utilization"][0] == 1.0


def test_slack_absorbs_supply_beyond_the_default_slack_capacity():
    """A disconnected network (no real arcs at all) whose supply exceeds the fixed old
    _SLACK_CAPACITY (1_000_000) used to raise ValueError: infeasible -- the very slack mechanism
    meant to keep every period feasible became itself the infeasibility. Slack capacity must scale
    with the problem so it can always absorb what the real network cannot carry."""
    n = 2
    cap = np.zeros((n, n))  # fully disconnected: no real arcs at all
    cost = np.zeros((n, n))
    supply = np.zeros(n)
    supply[0] = _SLACK_CAPACITY + 1.0
    supply[1] = -(_SLACK_CAPACITY + 1.0)

    network = {
        "cap": cap,
        "cost": cost,
        "supply": supply,
        "supply_nodes": [0],
        "demand_nodes": [1],
    }
    twin = build_twin(network, {}, seed=0)
    out = twin.run(1)  # must not raise

    assert out["throughput"][0] == 0.0  # no real arc exists to deliver anything
    assert out["queue"][-1] == _SLACK_CAPACITY + 1.0  # the whole demand is unmet, absorbed by slack
    assert out["bottleneck_arc"] is None  # no real arcs at all, so no bottleneck among them


def test_run_rejects_non_positive_period_counts():
    """run(0) used to step zero periods and then unconditionally index bottleneck_arcs[-1] into an
    empty list, raising IndexError instead of a clear error; negative counts hit the same bug. A
    period count with no periods to run has no meaningful per-period diagnostics, so it must raise up
    front -- consistent with this class's other invalid-input handling (an unregistered scenario name
    or intervention kind also raises rather than returning a degraded result)."""
    network, transport = _network()
    twin = build_twin(network, transport, seed=0)
    for bad_n in (0, -1, -5):
        try:
            twin.run(bad_n)
            raise AssertionError(f"expected ValueError for n_periods={bad_n}")
        except ValueError:
            pass
