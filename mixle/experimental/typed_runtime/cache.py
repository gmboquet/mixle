"""Versioned local artifact cache driven by update-graph invalidation."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from threading import RLock
from types import MappingProxyType
from typing import Any

from mixle.experimental.typed_runtime.contracts import ArtifactKind
from mixle.experimental.typed_runtime.graph import UpdateGraph


@dataclass(frozen=True)
class CachedArtifact:
    """One cached value tied to the producing node's generation."""

    node_id: str
    artifact: ArtifactKind
    generation: int
    value: Any


@dataclass(frozen=True)
class InvalidationReceipt:
    """Exact cache entries and node generations invalidated by a write."""

    source_nodes: tuple[str, ...]
    written_artifact: ArtifactKind
    invalidated_nodes: tuple[str, ...]
    removed_entries: tuple[tuple[str, ArtifactKind], ...]
    generations: dict[str, int]

    def __post_init__(self) -> None:
        """Make the receipt a record of an invalidation rather than an annotation of one.

        Nothing here was checked (MXR-080-1874). ``written_artifact="not-an-artifact"`` -- the string,
        not the enum -- constructed and then crashed in ``as_dict()`` where ``.value`` does not exist;
        the node lists accepted a bare string and iterated it as characters; ``generations`` was a
        caller-owned dict stored by reference on a frozen dataclass and could be rewritten after the
        invalidation was recorded; and a removed entry could name a node the receipt never says was
        invalidated. ``VersionedArtifactCache.invalidate`` satisfies all of it.
        """
        if not isinstance(self.written_artifact, ArtifactKind):
            raise TypeError(
                f"invalidation receipt written_artifact must be an ArtifactKind, got "
                f"{type(self.written_artifact).__name__} ({self.written_artifact!r}); a string has no .value."
            )
        for name in ("source_nodes", "invalidated_nodes"):
            ids = getattr(self, name)
            if isinstance(ids, (str, bytes)) or not isinstance(ids, (tuple, list)):
                raise TypeError(
                    f"invalidation receipt {name} must be a sequence of node ids, got {type(ids).__name__}: "
                    "a string would iterate as its characters."
                )
            object.__setattr__(self, name, tuple(ids))
            blank = [item for item in getattr(self, name) if not isinstance(item, str) or not item.strip()]
            if blank:
                raise ValueError(f"invalidation receipt {name} contains a node id that names nothing: {blank!r}.")
        entries = tuple(self.removed_entries)
        for entry in entries:
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise TypeError(f"invalidation receipt removed_entries must hold (node_id, artifact) pairs: {entry!r}.")
            node_id, artifact = entry
            if not isinstance(node_id, str) or not node_id.strip():
                raise ValueError(f"invalidation receipt removed entry names no node: {entry!r}.")
            if not isinstance(artifact, ArtifactKind):
                raise TypeError(f"invalidation receipt removed entry artifact must be an ArtifactKind: {entry!r}.")
        object.__setattr__(self, "removed_entries", entries)
        # An entry can only be dropped from a node this receipt says was invalidated; a removal
        # outside that set is a claim about a cache state the invalidation never reached.
        outside = sorted({node_id for node_id, _artifact in entries} - set(self.invalidated_nodes))
        if outside:
            raise ValueError(
                f"invalidation receipt removed cache entries for {outside}, which its own "
                "invalidated_nodes does not contain."
            )
        if not isinstance(self.generations, Mapping):
            raise TypeError(
                f"invalidation receipt generations must be a mapping, got {type(self.generations).__name__}."
            )
        generations = dict(self.generations)
        for node_id, generation in generations.items():
            if not isinstance(node_id, str) or not node_id.strip():
                raise ValueError(f"invalidation receipt generations names no node: {node_id!r}.")
            if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
                raise ValueError(
                    f"invalidation receipt generation for {node_id!r} must be a non-negative integer, "
                    f"got {generation!r}."
                )
        object.__setattr__(self, "generations", MappingProxyType(generations))

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible receipt without cached values."""

        return {
            "source_nodes": list(self.source_nodes),
            "written_artifact": self.written_artifact.value,
            "invalidated_nodes": list(self.invalidated_nodes),
            "removed_entries": [
                {"node_id": node_id, "artifact": artifact.value} for node_id, artifact in self.removed_entries
            ],
            "generations": dict(self.generations),
        }


class VersionedArtifactCache:
    """Thread-safe local cache with graph-derived transitive invalidation.

    A node generation advances even when no entry currently exists. A late
    worker can therefore compare its captured generation to the current one and
    detect that its result was computed before an intervening update.
    """

    def __init__(self, graph: UpdateGraph) -> None:
        self.graph = graph
        self._generations = {node.node_id: 0 for node in graph.nodes}
        self._entries: dict[tuple[str, ArtifactKind], CachedArtifact] = {}
        self._lock = RLock()

    def generation(self, node_id: str) -> int:
        """Return the current generation for a node."""

        self.graph.node(node_id)
        with self._lock:
            return self._generations[node_id]

    def put(self, node_id: str, artifact: ArtifactKind, value: Any) -> CachedArtifact:
        """Store a value at the node's current generation."""

        self.graph.node(node_id)
        with self._lock:
            entry = CachedArtifact(node_id, artifact, self._generations[node_id], value)
            self._entries[(node_id, artifact)] = entry
            return entry

    def get(self, node_id: str, artifact: ArtifactKind) -> Any:
        """Return a current value or raise ``KeyError`` for absent/stale data."""

        self.graph.node(node_id)
        key = (node_id, artifact)
        with self._lock:
            entry = self._entries[key]
            if entry.generation != self._generations[node_id]:
                del self._entries[key]
                raise KeyError(key)
            return entry.value

    def contains(self, node_id: str, artifact: ArtifactKind) -> bool:
        """Whether a current cache entry exists."""

        try:
            self.get(node_id, artifact)
        except KeyError:
            return False
        return True

    def invalidate(
        self,
        source_node: str,
        written_artifact: ArtifactKind = ArtifactKind.PARAMETERS,
    ) -> InvalidationReceipt:
        """Invalidate one source and every transitively dependent node."""

        return self.invalidate_many((source_node,), written_artifact)

    def invalidate_many(
        self,
        source_nodes: Collection[str],
        written_artifact: ArtifactKind = ArtifactKind.PARAMETERS,
    ) -> InvalidationReceipt:
        """Atomically invalidate the union of several dependency closures."""

        sources = tuple(dict.fromkeys(source_nodes))
        if not sources:
            raise ValueError("invalidate_many requires at least one source node.")
        closures = {node_id for source in sources for node_id in self.graph.invalidated_by(source)}
        invalidated = tuple(node_id for node_id in self.graph.topological_order() if node_id in closures)
        invalidated_set = set(invalidated)
        with self._lock:
            for node_id in invalidated:
                self._generations[node_id] += 1
            removed = tuple(
                sorted(
                    (key for key in self._entries if key[0] in invalidated_set),
                    key=lambda key: (key[0], key[1].value),
                )
            )
            for key in removed:
                del self._entries[key]
            generations = {node_id: self._generations[node_id] for node_id in invalidated}
        return InvalidationReceipt(sources, written_artifact, invalidated, removed, generations)

    def clear(self) -> None:
        """Invalidate all nodes and remove all entries."""

        with self._lock:
            for node_id in self._generations:
                self._generations[node_id] += 1
            self._entries.clear()

    def as_dict(self) -> dict[str, Any]:
        """Return cache metadata without serializing runtime values."""

        with self._lock:
            return {
                "generations": dict(self._generations),
                "entries": [
                    {
                        "node_id": entry.node_id,
                        "artifact": entry.artifact.value,
                        "generation": entry.generation,
                    }
                    for entry in sorted(
                        self._entries.values(),
                        key=lambda entry: (entry.node_id, entry.artifact.value),
                    )
                ],
            }


__all__ = ["CachedArtifact", "InvalidationReceipt", "VersionedArtifactCache"]
