"""Partitioned, version-checked, bounded cache for revisitable context graphs."""

from __future__ import annotations

import threading
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

    def __post_init__(self) -> None:
        if not self.partition_id or not self.node_ids:
            raise ValueError("graph partition id and node ownership must be non-empty.")
        if len(set(self.node_ids)) != len(self.node_ids) or any(not node_id for node_id in self.node_ids):
            raise ValueError("graph partition node ids must be unique and non-empty.")
        if isinstance(self.token_count, bool) or not isinstance(self.token_count, int) or self.token_count < 0:
            raise ValueError("graph partition token_count must be a non-negative integer.")
        if len(set(self.source_ids)) != len(self.source_ids) or any(not source_id for source_id in self.source_ids):
            raise ValueError("graph partition source ids must be unique and non-empty.")
        object.__setattr__(self, "node_ids", tuple(sorted(self.node_ids)))
        object.__setattr__(self, "source_ids", tuple(sorted(self.source_ids)))

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

    def __post_init__(self) -> None:
        partition_ids = [partition.partition_id for partition in self.partitions]
        if len(partition_ids) != len(set(partition_ids)):
            raise ValueError("graph partition ids must be unique.")
        node_ids = [node_id for partition in self.partitions for node_id in partition.node_ids]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("graph partition ownership must be unique.")
        if len(self.boundary_edge_ids) != len(set(self.boundary_edge_ids)):
            raise ValueError("graph boundary edge ids must be unique.")

    def partition(self, partition_id: str) -> GraphPartition:
        """Return a partition by id."""

        for row in self.partitions:
            if row.partition_id == partition_id:
                return row
        raise KeyError(partition_id)

    def owner(self, node_id: str) -> str:
        """Return the unique partition that owns a node."""

        for row in self.partitions:
            if node_id in row.node_ids:
                return row.partition_id
        raise KeyError(node_id)

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
    measured_token_count: int
    boundary_edge_ids: tuple[str, ...]
    pinned: bool = False


@dataclass(frozen=True)
class GraphPrefetchReceipt:
    """Requested, loaded, and evicted partition ids for one prefetch."""

    requested: tuple[str, ...]
    loaded: tuple[str, ...]
    evicted: tuple[str, ...]
    resident_tokens: int

    def __post_init__(self) -> None:
        """Bind the receipt to a prefetch that could actually have happened (MXR-080-0643).

        The counts here are read as memory evidence, so an unchecked receipt is a forgeable one:
        ``GraphPrefetchReceipt(requested=("a",), loaded=("a", "b", "c"), evicted=(),
        resident_tokens=999999)`` used to construct, claiming three loads from a one-partition
        request and an arbitrary residency.

        ``GraphMemoryCache.prefetch`` already satisfies all of this -- it appends to ``loaded`` only
        from the requested ids and measures ``resident_tokens`` -- so this rejects nothing the cache
        produces. ``loaded`` and ``evicted`` are deliberately allowed to intersect: a partition
        loaded early in one prefetch can be evicted by LRU later in that same prefetch.
        """
        unknown = [item for item in self.loaded if item not in set(self.requested)]
        if unknown:
            raise ValueError(
                f"prefetch receipt loaded partitions that were never requested: {sorted(unknown)}. "
                "A prefetch can only load what it was asked for."
            )
        if len(set(self.loaded)) != len(self.loaded):
            raise ValueError("prefetch receipt loaded the same partition twice in one prefetch.")
        if isinstance(self.resident_tokens, bool) or not isinstance(self.resident_tokens, int):
            raise ValueError(
                f"prefetch receipt resident_tokens must be an exact integer; got {self.resident_tokens!r}."
            )
        if self.resident_tokens < 0:
            raise ValueError(f"prefetch receipt resident_tokens must be non-negative; got {self.resident_tokens}.")

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible prefetch receipt."""

        return {
            "requested": list(self.requested),
            "loaded": list(self.loaded),
            "evicted": list(self.evicted),
            "resident_tokens": self.resident_tokens,
        }


def _canonical_partition(
    partition: GraphPartition,
    graph: ContextGraph,
) -> tuple[int, tuple[str, ...], tuple[str, ...]]:
    missing = sorted(set(partition.node_ids) - set(graph.nodes))
    if missing:
        raise KeyError("graph partition refers to missing nodes: %s" % ", ".join(missing))
    measured_tokens = sum(graph.nodes[node_id].token_count for node_id in partition.node_ids)
    source_ids = tuple(
        sorted(
            {provenance.source_id for node_id in partition.node_ids for provenance in graph.nodes[node_id].provenance}
        )
    )
    if partition.token_count != measured_tokens:
        raise ValueError(
            "graph partition token_count does not match canonical node content: declared %d, measured %d."
            % (partition.token_count, measured_tokens)
        )
    if partition.source_ids != source_ids:
        raise ValueError("graph partition source_ids do not match canonical node provenance.")
    owned = set(partition.node_ids)
    boundary_edge_ids = tuple(
        sorted(
            edge.edge_id for edge in graph.edges.values() if (edge.source_node in owned) != (edge.target_node in owned)
        )
    )
    return measured_tokens, source_ids, boundary_edge_ids


def _partition_hash(partition: GraphPartition, graph: ContextGraph) -> str:
    measured_tokens, source_ids, boundary_edge_ids = _canonical_partition(partition, graph)
    node_hashes = tuple((node_id, graph.nodes[node_id].content_hash) for node_id in partition.node_ids)
    edge_rows = tuple(
        sorted(
            (edge.edge_id, edge.as_dict())
            for edge in graph.edges.values()
            if edge.source_node in partition.node_ids or edge.target_node in partition.node_ids
        )
    )
    canonical = (
        partition.partition_id,
        partition.node_ids,
        measured_tokens,
        source_ids,
        boundary_edge_ids,
    )
    return payload_fingerprint((canonical, node_hashes, edge_rows))


class GraphMemoryCache:
    """LRU partition cache bounded by both tokens and partition count."""

    def __init__(self, *, maximum_tokens: int, maximum_partitions: int) -> None:
        if maximum_tokens <= 0 or maximum_partitions <= 0:
            raise ValueError("graph memory cache limits must be positive.")
        self.maximum_tokens = maximum_tokens
        self.maximum_partitions = maximum_partitions
        self._entries: OrderedDict[str, CachedGraphPartition] = OrderedDict()
        self._lock = threading.RLock()

    @property
    def resident_tokens(self) -> int:
        """Total tokens in resident partitions."""

        with self._lock:
            return self._resident_tokens(self._entries)

    @staticmethod
    def _resident_tokens(entries: OrderedDict[str, CachedGraphPartition]) -> int:
        return sum(entry.measured_token_count for entry in entries.values())

    @staticmethod
    def _entry(
        partition: GraphPartition,
        graph: ContextGraph,
        *,
        pinned: bool,
    ) -> CachedGraphPartition:
        measured_tokens, _, boundary_edge_ids = _canonical_partition(partition, graph)
        return CachedGraphPartition(
            partition,
            _partition_hash(partition, graph),
            graph.version,
            measured_tokens,
            boundary_edge_ids,
            pinned,
        )

    @staticmethod
    def _get_from(
        entries: OrderedDict[str, CachedGraphPartition],
        partition_id: str,
        graph: ContextGraph,
    ) -> CachedGraphPartition | None:
        entry = entries.get(partition_id)
        if entry is None:
            return None
        try:
            current_hash = _partition_hash(entry.partition, graph)
        except (KeyError, ValueError):
            del entries[partition_id]
            return None
        if current_hash != entry.content_hash:
            del entries[partition_id]
            return None
        entries.move_to_end(partition_id)
        return entry

    def get(self, partition_id: str, graph: ContextGraph) -> CachedGraphPartition | None:
        """Return a current partition, dropping stale content fingerprints."""

        with self._lock, graph.transaction():
            return self._get_from(self._entries, partition_id, graph)

    def _evict(self, entries: OrderedDict[str, CachedGraphPartition]) -> tuple[str, ...]:
        evicted = []
        while len(entries) > self.maximum_partitions or self._resident_tokens(entries) > self.maximum_tokens:
            victim = next((key for key, entry in entries.items() if not entry.pinned), None)
            if victim is None:
                raise MemoryError("pinned graph partitions exceed cache bounds.")
            del entries[victim]
            evicted.append(victim)
        return tuple(evicted)

    def put(self, partition: GraphPartition, graph: ContextGraph, *, pinned: bool = False) -> tuple[str, ...]:
        """Insert/refresh one partition and return evicted ids."""

        with self._lock, graph.transaction():
            entry = self._entry(partition, graph, pinned=pinned)
            if entry.measured_token_count > self.maximum_tokens:
                raise MemoryError("graph partition is larger than the entire cache token budget.")
            staged = self._entries.copy()
            staged[partition.partition_id] = entry
            staged.move_to_end(partition.partition_id)
            evicted = self._evict(staged)
            self._entries = staged
            return evicted

    def prefetch(
        self,
        plan: GraphPartitionPlan,
        graph: ContextGraph,
        partition_ids: tuple[str, ...],
    ) -> GraphPrefetchReceipt:
        """Load requested partitions in order under LRU bounds."""

        if len(partition_ids) != len(set(partition_ids)):
            raise ValueError("graph prefetch partition ids must be unique.")
        with self._lock, graph.transaction():
            owned = {node_id for partition in plan.partitions for node_id in partition.node_ids}
            if owned != set(graph.nodes):
                raise ValueError("graph partition plan must own every current graph node exactly once.")
            owner = {node_id: partition.partition_id for partition in plan.partitions for node_id in partition.node_ids}
            boundary = tuple(
                sorted(
                    edge.edge_id for edge in graph.edges.values() if owner[edge.source_node] != owner[edge.target_node]
                )
            )
            if boundary != plan.boundary_edge_ids:
                raise ValueError("graph partition plan boundary edges do not match the current graph.")

            staged = self._entries.copy()
            loaded = []
            evicted = []
            for partition_id in partition_ids:
                partition = plan.partition(partition_id)
                if self._get_from(staged, partition_id, graph) is None:
                    entry = self._entry(partition, graph, pinned=False)
                    if entry.measured_token_count > self.maximum_tokens:
                        raise MemoryError("graph partition is larger than the entire cache token budget.")
                    staged[partition.partition_id] = entry
                    staged.move_to_end(partition.partition_id)
                    evicted.extend(self._evict(staged))
                    loaded.append(partition_id)
            self._entries = staged
            return GraphPrefetchReceipt(
                partition_ids,
                tuple(loaded),
                tuple(evicted),
                self._resident_tokens(staged),
            )

    def as_dict(self) -> dict[str, Any]:
        """Return resident cache metadata in LRU order."""

        with self._lock:
            return {
                "maximum_tokens": self.maximum_tokens,
                "maximum_partitions": self.maximum_partitions,
                "resident_tokens": self._resident_tokens(self._entries),
                "entries": [
                    {
                        "partition_id": entry.partition.partition_id,
                        "declared_token_count": entry.partition.token_count,
                        "measured_token_count": entry.measured_token_count,
                        "boundary_edge_ids": list(entry.boundary_edge_ids),
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
