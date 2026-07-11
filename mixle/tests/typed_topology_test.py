"""Measured topology and structured placement tests."""

import json

import pytest

from mixle.experimental.typed_runtime import (
    ClusterTopology,
    LinkProfile,
    PlacementScope,
    TopologyDevice,
    TransferSample,
    calibrate_links,
    compile_update_graph,
    plan_structured_placement,
)
from mixle.stats import GaussianDistribution, GaussianEstimator, MixtureDistribution, MixtureEstimator
from mixle.utils.parallel.planner import DeviceSpec

pytestmark = [pytest.mark.experimental, pytest.mark.fast]


def _device(device_id, island, throughput):
    return TopologyDevice(
        device_id,
        host="host-%s" % device_id,
        island=island,
        provider="provider",
        region="region",
        spec=DeviceSpec(device_id, kind="gpu", memory_bytes=8_000_000_000, throughput=throughput),
    )


def _topology():
    devices = (
        _device("a0", "island-a", 4.0),
        _device("a1", "island-a", 4.0),
        _device("b0", "island-b", 2.0),
    )
    links = (
        LinkProfile("a0", "a1", 1.0e-6, 100.0e9),
        LinkProfile("a1", "a0", 1.0e-6, 100.0e9),
        LinkProfile("a0", "b0", 0.05, 100.0e6),
        LinkProfile("b0", "a0", 0.05, 100.0e6),
        LinkProfile("a1", "b0", 0.05, 100.0e6),
        LinkProfile("b0", "a1", 0.05, 100.0e6),
    )
    return ClusterTopology(devices, links)


def test_link_calibration_recovers_startup_and_bandwidth():
    samples = (
        TransferSample("a", "b", 1_000_000, 0.02),
        TransferSample("a", "b", 2_000_000, 0.03),
        TransferSample("a", "b", 3_000_000, 0.04),
    )
    profile = calibrate_links(samples)[0]
    assert profile.latency_seconds == pytest.approx(0.01)
    assert profile.bandwidth_bytes_per_second == pytest.approx(100.0e6)
    assert profile.sample_count == 3
    assert profile.provenance == "measured-linear-fit"


def test_shortest_transfer_route_can_use_an_intermediate_device():
    devices = (_device("x", "i", 1.0), _device("y", "i", 1.0), _device("z", "j", 1.0))
    topology = ClusterTopology(
        devices,
        (
            LinkProfile("x", "z", 1.0, 1.0e9),
            LinkProfile("x", "y", 0.1, 1.0e9),
            LinkProfile("y", "z", 0.1, 1.0e9),
        ),
    )
    estimate = topology.transfer("x", "z", 1_000)
    assert estimate.path == ("x", "y", "z")
    assert estimate.seconds < 0.3

    unreachable = topology.transfer("z", "x", 1_000)
    assert unreachable.path == ()
    assert unreachable.seconds == float("inf")


def test_exact_component_shards_stay_in_fast_island_and_cross_islands_are_replicas():
    model = MixtureDistribution(
        [GaussianDistribution(float(index), 1.0) for index in range(4)],
        [0.25] * 4,
    )
    estimator = MixtureEstimator([GaussianEstimator() for _ in range(4)])
    graph = compile_update_graph(model, estimator)
    plan = plan_structured_placement(graph, _topology(), n_data=10)
    root = plan.placement(graph.root_node)

    assert plan.primary_island == "island-a"
    assert root.scope is PlacementScope.INTRA_ISLAND_SYNCHRONOUS
    assert root.replica_islands == ("island-b",)
    assert len(root.shards) == 2
    assert {shard.device_id for shard in root.shards} == {"a0", "a1"}
    assert {shard.axis for shard in root.shards} == {"component"}
    assert all(shard.exact for shard in root.shards)
    assert sum(shard.stop - shard.start for shard in root.shards) == 4
    assert all(shard.island == "island-a" for placement in plan.placements for shard in placement.shards)
    json.dumps(plan.as_dict(), allow_nan=False)
    json.dumps(_topology().as_dict(), allow_nan=False)
