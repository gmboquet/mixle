"""Digital-twin simulation of the pipeline (H8): period-stepped re-solve + scenario intervention."""

from __future__ import annotations

import numpy as np

from mixle.pipeline_twin import PipelineTwin, build_twin

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
