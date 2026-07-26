"""Bounded graph partition, LRU, prefetch, pinning, and version checks."""

import json
from dataclasses import replace

import pytest

from mixle.experimental.typed_runtime import (
    ContextEdge,
    ContextEdgeKind,
    ContextGraph,
    ContextNode,
    ContextNodeKind,
    EvidenceStatus,
    GraphMemoryCache,
    GraphPartitionPlan,
    partition_context_graph,
)

pytestmark = [pytest.mark.experimental, pytest.mark.fast]


def _graph(count=8):
    graph = ContextGraph()
    for index in range(count):
        graph.add_node(ContextNode("n%d" % index, ContextNodeKind.MEMORY, "node %d" % index, 10))
    for index in range(count - 1):
        graph.add_edge(
            ContextEdge(
                "e%d" % index,
                "n%d" % index,
                "n%d" % (index + 1),
                ContextEdgeKind.SEMANTIC,
            )
        )
    return graph


def test_partition_plan_owns_every_node_once_and_marks_cross_partition_edges():
    graph = _graph()
    plan = partition_context_graph(graph, maximum_tokens=30, maximum_nodes=3)
    owned = [node_id for partition in plan.partitions for node_id in partition.node_ids]
    assert sorted(owned) == sorted(graph.nodes)
    assert len(owned) == len(set(owned))
    assert all(partition.token_count <= 30 and len(partition.node_ids) <= 3 for partition in plan.partitions)
    assert plan.boundary_edge_ids
    assert all(plan.owner(node_id).startswith("partition-") for node_id in graph.nodes)
    json.dumps(plan.as_dict(), allow_nan=False)


def test_lru_prefetch_remains_token_bounded_and_reports_eviction():
    graph = _graph()
    plan = partition_context_graph(graph, maximum_tokens=30, maximum_nodes=3)
    cache = GraphMemoryCache(maximum_tokens=50, maximum_partitions=2)
    first, second = plan.partitions[:2]
    receipt = cache.prefetch(plan, graph, (first.partition_id, second.partition_id))
    assert receipt.loaded == (first.partition_id, second.partition_id)
    assert receipt.evicted == (first.partition_id,)
    assert cache.resident_tokens <= 50
    assert cache.get(first.partition_id, graph) is None
    assert cache.get(second.partition_id, graph) is not None
    json.dumps(cache.as_dict(), allow_nan=False)


def test_lru_access_changes_victim_order():
    graph = _graph(count=7)
    plan = partition_context_graph(graph, maximum_tokens=20, maximum_nodes=2)
    cache = GraphMemoryCache(maximum_tokens=40, maximum_partitions=2)
    first, second, third = plan.partitions[:3]
    cache.put(first, graph)
    cache.put(second, graph)
    assert cache.get(first.partition_id, graph) is not None
    evicted = cache.put(third, graph)
    assert evicted == (second.partition_id,)
    assert cache.get(first.partition_id, graph) is not None


def test_node_change_invalidates_only_owning_partition_not_whole_graph_version():
    graph = _graph()
    plan = partition_context_graph(graph, maximum_tokens=30, maximum_nodes=3)
    first, second = plan.partitions[:2]
    cache = GraphMemoryCache(maximum_tokens=100, maximum_partitions=4)
    cache.put(first, graph)
    cache.put(second, graph)
    changed_node = first.node_ids[0]
    graph.verify(changed_node, EvidenceStatus.INCONCLUSIVE, confidence=0.2)

    assert cache.get(first.partition_id, graph) is None
    assert cache.get(second.partition_id, graph) is not None


def test_pinned_partitions_cannot_silently_break_cache_bound():
    graph = _graph()
    plan = partition_context_graph(graph, maximum_tokens=30, maximum_nodes=3)
    first, second = plan.partitions[:2]
    cache = GraphMemoryCache(maximum_tokens=50, maximum_partitions=2)
    cache.put(first, graph, pinned=True)
    with pytest.raises(MemoryError, match="pinned"):
        cache.put(second, graph, pinned=True)
    assert cache.resident_tokens == first.token_count
    assert cache.get(first.partition_id, graph) is not None


def test_cache_recomputes_partition_tokens_instead_of_trusting_descriptor():
    graph = _graph(count=2)
    partition = partition_context_graph(graph, maximum_tokens=20, maximum_nodes=2).partitions[0]
    forged = replace(partition, token_count=0)
    cache = GraphMemoryCache(maximum_tokens=5, maximum_partitions=1)

    with pytest.raises(ValueError, match="declared 0, measured 20"):
        cache.put(forged, graph)
    assert cache.resident_tokens == 0


def test_boundary_edge_identity_invalidates_each_incident_partition():
    graph = _graph(count=4)
    plan = partition_context_graph(graph, maximum_tokens=20, maximum_nodes=2)
    first, second = plan.partitions
    cache = GraphMemoryCache(maximum_tokens=100, maximum_partitions=4)
    cache.put(first, graph)
    cache.put(second, graph)
    graph.add_edge(
        ContextEdge(
            "new-boundary",
            first.node_ids[0],
            second.node_ids[0],
            ContextEdgeKind.SEMANTIC,
        )
    )

    assert cache.get(first.partition_id, graph) is None
    assert cache.get(second.partition_id, graph) is None


def test_multi_partition_prefetch_is_atomic_on_late_validation_failure():
    graph = _graph(count=4)
    plan = partition_context_graph(graph, maximum_tokens=20, maximum_nodes=2)
    first, second = plan.partitions
    invalid_second = replace(second, token_count=0)
    invalid_plan = GraphPartitionPlan(
        (first, invalid_second),
        plan.boundary_edge_ids,
    )
    cache = GraphMemoryCache(maximum_tokens=100, maximum_partitions=4)

    with pytest.raises(ValueError, match="token_count"):
        cache.prefetch(
            invalid_plan,
            graph,
            (first.partition_id, second.partition_id),
        )
    assert cache.as_dict()["entries"] == []
