"""Measured topology and typed structured-shard placement."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from mixle.experimental.typed_runtime.contracts import ConsistencyRequirement, MergeLaw, UpdateKind
from mixle.experimental.typed_runtime.graph import UpdateGraph
from mixle.stats.compute.decomposition import DecompAxis, ReductionOp
from mixle.utils.parallel.model_decomposition import decompose_model
from mixle.utils.parallel.planner import DeviceSpec, Resources


@dataclass(frozen=True)
class TopologyDevice:
    """A compute device with failure, network, provider, and storage locality."""

    device_id: str
    host: str
    island: str
    provider: str
    region: str
    spec: DeviceSpec
    storage_region: str | None = None

    def __post_init__(self) -> None:
        values = (self.device_id, self.host, self.island, self.provider, self.region)
        if any(not value for value in values):
            raise ValueError("topology device identity and locality fields must be non-empty.")
        if self.spec.name != self.device_id:
            raise ValueError("DeviceSpec.name must equal topology device_id.")

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible device declaration."""

        return {
            "device_id": self.device_id,
            "host": self.host,
            "island": self.island,
            "provider": self.provider,
            "region": self.region,
            "storage_region": self.storage_region,
            "spec": self.spec.to_dict(),
        }


@dataclass(frozen=True)
class TransferSample:
    """One observed point-to-point transfer."""

    source: str
    target: str
    bytes_transferred: int
    elapsed_seconds: float

    def __post_init__(self) -> None:
        if not self.source or not self.target or self.source == self.target:
            raise ValueError("transfer samples require distinct non-empty endpoints.")
        if self.bytes_transferred <= 0 or self.elapsed_seconds <= 0.0:
            raise ValueError("transfer sample bytes and time must be positive.")


@dataclass(frozen=True)
class LinkProfile:
    """Directed link startup latency and sustained bandwidth."""

    source: str
    target: str
    latency_seconds: float
    bandwidth_bytes_per_second: float
    contention_multiplier: float = 1.0
    sample_count: int = 0
    provenance: str = "declared"

    def __post_init__(self) -> None:
        if not self.source or not self.target or self.source == self.target:
            raise ValueError("link profiles require distinct non-empty endpoints.")
        if self.latency_seconds < 0.0 or self.bandwidth_bytes_per_second <= 0.0:
            raise ValueError("link latency must be non-negative and bandwidth positive.")
        if self.contention_multiplier < 1.0 or self.sample_count < 0:
            raise ValueError("link contention must be >= 1 and sample_count non-negative.")

    def transfer_seconds(self, nbytes: int) -> float:
        """Estimated one-way time including startup and measured contention."""

        if nbytes < 0:
            raise ValueError("transfer bytes must be non-negative.")
        return self.contention_multiplier * (self.latency_seconds + nbytes / self.bandwidth_bytes_per_second)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible link profile."""

        return {
            "source": self.source,
            "target": self.target,
            "latency_seconds": self.latency_seconds,
            "bandwidth_bytes_per_second": self.bandwidth_bytes_per_second,
            "contention_multiplier": self.contention_multiplier,
            "sample_count": self.sample_count,
            "provenance": self.provenance,
        }


def calibrate_links(samples: tuple[TransferSample, ...]) -> tuple[LinkProfile, ...]:
    """Fit ``time = latency + bytes / bandwidth`` independently per direction."""

    grouped: dict[tuple[str, str], list[TransferSample]] = {}
    for sample in samples:
        grouped.setdefault((sample.source, sample.target), []).append(sample)
    profiles = []
    for (source, target), rows in sorted(grouped.items()):
        sizes = np.asarray([row.bytes_transferred for row in rows], dtype=np.float64)
        times = np.asarray([row.elapsed_seconds for row in rows], dtype=np.float64)
        if len(rows) >= 2 and np.ptp(sizes) > 0.0:
            slope, intercept = np.polyfit(sizes, times, 1)
            slope = max(float(slope), np.finfo(np.float64).tiny)
            latency = max(float(intercept), 0.0)
            bandwidth = 1.0 / slope
        else:
            latency = 0.0
            bandwidth = float(sizes[0] / times[0])
        profiles.append(
            LinkProfile(
                source,
                target,
                latency,
                bandwidth,
                sample_count=len(rows),
                provenance="measured-linear-fit",
            )
        )
    return tuple(profiles)


@dataclass(frozen=True)
class TransferEstimate:
    """Best measured route and transfer time between two devices."""

    source: str
    target: str
    nbytes: int
    seconds: float
    path: tuple[str, ...]


@dataclass(frozen=True)
class ClusterTopology:
    """Directed measured network over topology-aware compute devices."""

    devices: tuple[TopologyDevice, ...]
    links: tuple[LinkProfile, ...]

    def __post_init__(self) -> None:
        identifiers = {device.device_id for device in self.devices}
        if not identifiers or len(identifiers) != len(self.devices):
            raise ValueError("topology device ids must be non-empty and unique.")
        if any(link.source not in identifiers or link.target not in identifiers for link in self.links):
            raise ValueError("topology links must refer to declared devices.")
        pairs = {(link.source, link.target) for link in self.links}
        if len(pairs) != len(self.links):
            raise ValueError("topology can contain only one profile per directed link.")

    @property
    def islands(self) -> tuple[str, ...]:
        """Sorted failure/network islands."""

        return tuple(sorted({device.island for device in self.devices}))

    def devices_in(self, island: str) -> tuple[TopologyDevice, ...]:
        """Devices in one island, ordered by id."""

        return tuple(
            sorted((device for device in self.devices if device.island == island), key=lambda row: row.device_id)
        )

    def resources(self, island: str | None = None) -> Resources:
        """Adapt topology devices to the stable placement planner's resource type."""

        rows = self.devices if island is None else self.devices_in(island)
        if not rows:
            raise KeyError(island)
        return Resources(tuple(device.spec for device in rows))

    def best_island(self) -> str:
        """Island with the largest declared aggregate throughput."""

        return max(
            self.islands,
            key=lambda island: (
                sum(device.spec.throughput for device in self.devices_in(island)),
                len(self.devices_in(island)),
                island,
            ),
        )

    def transfer(self, source: str, target: str, nbytes: int) -> TransferEstimate:
        """Shortest transfer-time route under current link profiles."""

        identifiers = {device.device_id for device in self.devices}
        if source not in identifiers or target not in identifiers:
            raise KeyError(source if source not in identifiers else target)
        if nbytes < 0:
            raise ValueError("transfer bytes must be non-negative.")
        if source == target:
            return TransferEstimate(source, target, nbytes, 0.0, (source,))
        outgoing: dict[str, list[LinkProfile]] = {device_id: [] for device_id in identifiers}
        for link in self.links:
            outgoing[link.source].append(link)
        queue = [(0.0, source, (source,))]
        best = {source: 0.0}
        while queue:
            seconds, current, path = heapq.heappop(queue)
            if current == target:
                return TransferEstimate(source, target, nbytes, seconds, path)
            if seconds > best.get(current, math.inf):
                continue
            for link in outgoing[current]:
                candidate = seconds + link.transfer_seconds(nbytes)
                if candidate < best.get(link.target, math.inf):
                    best[link.target] = candidate
                    heapq.heappush(queue, (candidate, link.target, path + (link.target,)))
        return TransferEstimate(source, target, nbytes, math.inf, ())

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible topology receipt."""

        return {
            "devices": [device.as_dict() for device in self.devices],
            "links": [link.as_dict() for link in self.links],
            "islands": list(self.islands),
        }


class PlacementScope(StrEnum):
    """Communication scope permitted for one typed node."""

    LOCAL = "local"
    INTRA_ISLAND_SYNCHRONOUS = "intra_island_synchronous"
    CROSS_ISLAND_PROPOSAL = "cross_island_proposal"


@dataclass(frozen=True)
class StructuredShard:
    """One contiguous typed model-axis cut."""

    shard_id: str
    node_id: str
    island: str
    device_id: str
    axis: str
    start: int
    stop: int
    reduction: str
    exact: bool

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible shard placement."""

        return {
            "shard_id": self.shard_id,
            "node_id": self.node_id,
            "island": self.island,
            "device_id": self.device_id,
            "axis": self.axis,
            "start": self.start,
            "stop": self.stop,
            "reduction": self.reduction,
            "exact": self.exact,
        }


@dataclass(frozen=True)
class NodePlacement:
    """Placement and cross-island replication policy for one typed node."""

    node_id: str
    scope: PlacementScope
    primary_island: str
    replica_islands: tuple[str, ...]
    shards: tuple[StructuredShard, ...]
    rationale: str

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible node placement."""

        return {
            "node_id": self.node_id,
            "scope": self.scope.value,
            "primary_island": self.primary_island,
            "replica_islands": list(self.replica_islands),
            "shards": [shard.as_dict() for shard in self.shards],
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class StructuredPlacementPlan:
    """Typed placement over one calibrated cluster topology."""

    primary_island: str
    placements: tuple[NodePlacement, ...]

    def placement(self, node_id: str) -> NodePlacement:
        """Return placement metadata for one node."""

        return next(row for row in self.placements if row.node_id == node_id)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible plan."""

        return {
            "primary_island": self.primary_island,
            "placements": [placement.as_dict() for placement in self.placements],
        }


def plan_structured_placement(
    graph: UpdateGraph,
    topology: ClusterTopology,
    *,
    n_data: int | None = None,
) -> StructuredPlacementPlan:
    """Place exact structured cuts inside one island and proposals across islands."""

    primary = topology.best_island()
    resources = topology.resources(primary)
    replicas = tuple(island for island in topology.islands if island != primary)
    placements = []
    for node_id in graph.topological_order():
        node = graph.node(node_id)
        contract = node.contract
        if contract.update_kind is UpdateKind.FROZEN:
            scope = PlacementScope.LOCAL
        elif contract.consistency is ConsistencyRequirement.LOCAL_ONLY:
            scope = PlacementScope.LOCAL
        elif len(resources.devices) > 1:
            scope = PlacementScope.INTRA_ISLAND_SYNCHRONOUS
        else:
            scope = PlacementScope.LOCAL

        shards: tuple[StructuredShard, ...] = ()
        rationale = "atomic typed node on primary island"
        if contract.decomposition_axes:
            if contract.merge_law in (MergeLaw.NON_MERGEABLE, MergeLaw.REPLICATED):
                raise ValueError("node %s declares sharding without a merge law." % node_id)
            decomposition = decompose_model(
                node.model,
                resources,
                n_data=n_data,
                prefer_data_parallel=False,
            )
            if decomposition.axis is not DecompAxis.NONE:
                shards = tuple(
                    StructuredShard(
                        shard_id="%s:%d" % (node_id, index),
                        node_id=node_id,
                        island=primary,
                        device_id=cut.device.name,
                        axis=decomposition.axis.value,
                        start=cut.start,
                        stop=cut.stop,
                        reduction=cut.reduction.value,
                        exact=contract.exact,
                    )
                    for index, cut in enumerate(decomposition.cuts)
                )
                rationale = decomposition.rationale
        if not shards:
            device = topology.devices_in(primary)[0]
            shards = (
                StructuredShard(
                    "%s:0" % node_id,
                    node_id,
                    primary,
                    device.device_id,
                    "none",
                    0,
                    1,
                    ReductionOp.REPLICATE.value,
                    contract.exact,
                ),
            )
        placements.append(
            NodePlacement(
                node_id,
                scope,
                primary,
                replicas if contract.consistency is not ConsistencyRequirement.LOCAL_ONLY else (),
                shards,
                rationale,
            )
        )
    return StructuredPlacementPlan(primary, tuple(placements))


__all__ = [
    "ClusterTopology",
    "LinkProfile",
    "NodePlacement",
    "PlacementScope",
    "StructuredPlacementPlan",
    "StructuredShard",
    "TopologyDevice",
    "TransferEstimate",
    "TransferSample",
    "calibrate_links",
    "plan_structured_placement",
]
