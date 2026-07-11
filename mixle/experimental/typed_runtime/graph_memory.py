"""Partitioned, version-checked, bounded cache for revisitable context graphs."""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Any

from mixle.experimental.typed_runtime.context_ir import ContextGraph
from mixle.experimental.typed_runtime.proposal import payload_fingerprint


@dataclass(frozen=True)
class GraphPartition:
    """Bounded graph region and its source-locality metadata."""

    partition_id: str
    node_ids: tuple[str, ...]
    token_count: int
    source_ids: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible partition descriptor."""

        return {
            "partition_id": self.partition_id,
            "node_ids": list(self.node_ids),
            "token_count": self.token_count,
            "source_ids": list(self.source_ids),
        }


@dataclass(frozen=True)
class GraphPartitionPlan:
    """Complete node ownership and cross-partition edge list."""

    partitions: tuple[GraphPartition, ...]
    boundary_edge_ids: tuple[str, ...]

    def partition(self, partition_id: str) -> GraphPartition:
        """Return a partition by id."""

        return next(row for row in self.partitions if row.partition_id == partition_id)

    def owner(self, node_id: str) -> str:
        """Return the unique partition that owns a node."""

        return next(row.partition_id for row in self.partitions if node_id in row.node_ids)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible partition plan."""

        return {
            "partitions": [partition.as_dict() for partition in self.partitions],
            "boundary_edge_ids": list(self.boundary_edge_ids),
        }


def partition_context_graph(
    graph: ContextGraph,
    *,
    maximum_tokens: int,
    maximum_nodes: int,
) -> GraphPartitionPlan:
    """Greedily partition connected/source-local regions under hard bounds."""

    if maximum_tokens <= 0 or maximum_nodes <= 0:
        raise ValueError("graph partition token and node limits must be positive.")
    oversized = [node.node_id for node in graph.nodes.values() if node.token_count > maximum_tokens]
    if oversized:
        raise ValueError("context nodes exceed maximum partition tokens: %s" % ", ".join(sorted(oversized)))
    adjacency = {node_id: set() for node_id in graph.nodes}
    for edge in graph.edges.values():
        adjacency[edge.source_node].add(edge.target_node)
        adjacency[edge.target_node].add(edge.source_node)

    unassigned = set(graph.nodes)
    partitions = []
    while unassigned:
        seed = min(unassigned)
        queue = deque([seed])
        selected: list[str] = []
        tokens = 0
        while queue:
            node_id = queue.popleft()
            if node_id not in unassigned:
                continue
            node = graph.nodes[node_id]
            if len(selected) >= maximum_nodes or tokens + node.token_count > maximum_tokens:
                continue
            selected.append(node_id)
            tokens += node.token_count
            unassigned.remove(node_id)
            source_ids = {item.source_id for item in node.provenance}
            neighbors = sorted(
                adjacency[node_id] & unassigned,
                key=lambda neighbor: (
                    not bool(source_ids & {item.source_id for item in graph.nodes[neighbor].provenance}),
                    neighbor,
                ),
            )
            queue.extend(neighbors)
        if not selected:
            raise RuntimeError("graph partitioner made no progress.")
        all_sources = tuple(
            sorted({item.source_id for node_id in selected for item in graph.nodes[node_id].provenance})
        )
        partitions.append(
            GraphPartition(
                "partition-%05d" % len(partitions),
                tuple(sorted(selected)),
                tokens,
                all_sources,
            )
        )

    owner = {node_id: partition.partition_id for partition in partitions for node_id in partition.node_ids}
    boundary = tuple(
        sorted(edge.edge_id for edge in graph.edges.values() if owner[edge.source_node] != owner[edge.target_node])
    )
    return GraphPartitionPlan(tuple(partitions), boundary)


@dataclass(frozen=True)
class CachedGraphPartition:
    """Partition content fingerprint captured at cache insertion."""

    partition: GraphPartition
    content_hash: str
    graph_version: int
    pinned: bool = False


@dataclass(frozen=True)
class GraphPrefetchReceipt:
    """Requested, loaded, and evicted partition ids for one prefetch."""

    requested: tuple[str, ...]
    loaded: tuple[str, ...]
    evicted: tuple[str, ...]
    resident_tokens: int

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible prefetch receipt."""

        return {
            "requested": list(self.requested),
            "loaded": list(self.loaded),
            "evicted": list(self.evicted),
            "resident_tokens": self.resident_tokens,
        }


def _partition_hash(partition: GraphPartition, graph: ContextGraph) -> str:
    node_hashes = tuple((node_id, graph.nodes[node_id].content_hash) for node_id in partition.node_ids)
    edge_rows = tuple(
        sorted(
            (edge.edge_id, edge.as_dict())
            for edge in graph.edges.values()
            if edge.source_node in partition.node_ids and edge.target_node in partition.node_ids
        )
    )
    return payload_fingerprint((partition.as_dict(), node_hashes, edge_rows))


class GraphMemoryCache:
    """LRU partition cache bounded by both tokens and partition count."""

    def __init__(self, *, maximum_tokens: int, maximum_partitions: int) -> None:
        if maximum_tokens <= 0 or maximum_partitions <= 0:
            raise ValueError("graph memory cache limits must be positive.")
        self.maximum_tokens = maximum_tokens
        self.maximum_partitions = maximum_partitions
        self._entries: OrderedDict[str, CachedGraphPartition] = OrderedDict()

    @property
    def resident_tokens(self) -> int:
        """Total tokens in resident partitions."""

        return sum(entry.partition.token_count for entry in self._entries.values())

    def get(self, partition_id: str, graph: ContextGraph) -> CachedGraphPartition | None:
        """Return a current partition, dropping stale content fingerprints."""

        entry = self._entries.get(partition_id)
        if entry is None:
            return None
        if any(node_id not in graph.nodes for node_id in entry.partition.node_ids):
            del self._entries[partition_id]
            return None
        if _partition_hash(entry.partition, graph) != entry.content_hash:
            del self._entries[partition_id]
            return None
        self._entries.move_to_end(partition_id)
        return entry

    def _evict(self) -> tuple[str, ...]:
        evicted = []
        while len(self._entries) > self.maximum_partitions or self.resident_tokens > self.maximum_tokens:
            victim = next((key for key, entry in self._entries.items() if not entry.pinned), None)
            if victim is None:
                raise MemoryError("pinned graph partitions exceed cache bounds.")
            del self._entries[victim]
            evicted.append(victim)
        return tuple(evicted)

    def put(self, partition: GraphPartition, graph: ContextGraph, *, pinned: bool = False) -> tuple[str, ...]:
        """Insert/refresh one partition and return evicted ids."""

        if partition.token_count > self.maximum_tokens:
            raise MemoryError("graph partition is larger than the entire cache token budget.")
        if any(node_id not in graph.nodes for node_id in partition.node_ids):
            raise KeyError("graph partition refers to missing nodes.")
        previous = self._entries.copy()
        entry = CachedGraphPartition(partition, _partition_hash(partition, graph), graph.version, pinned)
        self._entries[partition.partition_id] = entry
        self._entries.move_to_end(partition.partition_id)
        try:
            return self._evict()
        except MemoryError:
            self._entries = previous
            raise

    def prefetch(
        self,
        plan: GraphPartitionPlan,
        graph: ContextGraph,
        partition_ids: tuple[str, ...],
    ) -> GraphPrefetchReceipt:
        """Load requested partitions in order under LRU bounds."""

        loaded = []
        evicted = []
        for partition_id in partition_ids:
            partition = plan.partition(partition_id)
            if self.get(partition_id, graph) is None:
                evicted.extend(self.put(partition, graph))
                loaded.append(partition_id)
        return GraphPrefetchReceipt(partition_ids, tuple(loaded), tuple(evicted), self.resident_tokens)

    def as_dict(self) -> dict[str, Any]:
        """Return resident cache metadata in LRU order."""

        return {
            "maximum_tokens": self.maximum_tokens,
            "maximum_partitions": self.maximum_partitions,
            "resident_tokens": self.resident_tokens,
            "entries": [
                {
                    "partition_id": entry.partition.partition_id,
                    "token_count": entry.partition.token_count,
                    "content_hash": entry.content_hash,
                    "graph_version": entry.graph_version,
                    "pinned": entry.pinned,
                }
                for entry in self._entries.values()
            ],
        }


__all__ = [
    "CachedGraphPartition",
    "GraphMemoryCache",
    "GraphPartition",
    "GraphPartitionPlan",
    "GraphPrefetchReceipt",
    "partition_context_graph",
]
