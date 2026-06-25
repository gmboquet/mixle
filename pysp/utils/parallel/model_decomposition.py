"""Structural model-decomposition planner (component C2 of the model-parallel design).

Turns *any* pysparkplug model into a model-parallel placement by walking the opt-in decomposition
contract (:mod:`pysp.stats.compute.decomposition`) instead of a blind reflective ``__dict__`` walk:

  * :func:`size_model_tree` -- a structural byte sizing of the model tree (own params per node, recursing
    into children via the declared axis), so shared subtrees are not double-counted and a node's own
    parameter footprint (e.g. an HMM's dense ``S*S`` transition block) is visible rather than silently
    zeroed. Replaces the reflective ``estimate_model_nbytes`` for sizing decisions.
  * :func:`decompose_model` -- choose the cut (data vs model vs none) for the root shardable node and
    bin-pack the component / factor units across the device budget, carrying the per-cut reduction. This
    is the general form of, and a real consumer-shaped output for, ``planner.model_sharding_plan`` (which
    only ever understood mixture components and had no consumer).

It works for every family: nodes that do not opt into the contract report ``Decomposition.atomic()`` and
are simply replicated (data-parallel, already optimal). See ``~/codex/notes/model-parallel-design.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pysp.stats.compute.decomposition import DecompAxis, Decomposition, ReductionOp, decomposition_for
from pysp.utils.parallel.planner import DeviceSpec, Resources


# --- structural sizing ---------------------------------------------------------------------------
def _own_param_bytes(node: Any) -> int:
    """Bytes of a node's OWN numeric parameters (its ndarray attributes), excluding child subtrees.

    Children are distribution objects (lists/tuples/dicts of them), not ndarrays, so they are skipped
    here and sized by recursion -- this avoids the double-counting the reflective walk suffers from.
    """
    total = 0
    for v in getattr(node, "__dict__", {}).values():
        if isinstance(v, np.ndarray):
            total += int(v.nbytes)
    return total


def shard_children(node: Any, dc: Decomposition | None = None) -> tuple[Any, ...]:
    """Return the actual child distributions a node splits along its declared axis (else ``()``)."""
    dc = dc if dc is not None else decomposition_for(node)
    if dc.axis is DecompAxis.COMPONENT:
        return tuple(getattr(node, "components", ()) or ())
    if dc.axis is DecompAxis.FACTOR:
        return tuple(getattr(node, "dists", ()) or ())
    if dc.axis is DecompAxis.TOPIC:
        return tuple(getattr(node, "topics", ()) or ())
    if dc.axis is DecompAxis.SEQUENCE:
        base = getattr(node, "dist", None)
        return (base,) if base is not None else ()
    return ()


@dataclass(frozen=True)
class NodeSize:
    """Structural size of one model-tree node."""

    path: str
    axis: DecompAxis
    num_units: int
    own_param_bytes: int  # this node's own parameter footprint (NOT its children)
    subtree_param_bytes: int  # own + all descendants (the replicated cost of holding this subtree)
    children: tuple[NodeSize, ...] = ()


def size_model_tree(model: Any, _path: str = "", _seen: set[int] | None = None) -> NodeSize:
    """Recursively size the model tree via the decomposition contract (shared subtrees counted once)."""
    seen = _seen if _seen is not None else set()
    dc = decomposition_for(model)
    children = shard_children(model, dc)
    child_sizes: list[NodeSize] = []
    subtree = _own_param_bytes(model)
    for i, child in enumerate(children):
        if child is None or id(child) in seen:
            continue
        seen.add(id(child))
        role = dc.child_roles[i] if i < len(dc.child_roles) else f"{dc.axis.value}_{i}"
        cs = size_model_tree(child, f"{_path}/{role}".lstrip("/"), seen)
        child_sizes.append(cs)
        subtree += cs.subtree_param_bytes
    return NodeSize(
        path=_path,
        axis=dc.axis,
        num_units=dc.num_units,
        own_param_bytes=_own_param_bytes(model),
        subtree_param_bytes=subtree,
        children=tuple(child_sizes),
    )


# --- decomposition / placement -------------------------------------------------------------------
@dataclass(frozen=True)
class ModelCut:
    """One model-parallel cut: a contiguous unit range placed on one device, with its reduction."""

    device: DeviceSpec
    start: int
    stop: int
    reduction: ReductionOp


@dataclass(frozen=True)
class ModelDecomposition:
    """The chosen decomposition of a model onto a device budget."""

    axis: DecompAxis
    reduction: ReductionOp
    num_units: int
    cuts: tuple[ModelCut, ...]
    rationale: str
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_model_parallel(self) -> bool:
        return self.axis is not DecompAxis.NONE and len(self.cuts) > 1


def _unit_partition(num_units: int, devices: tuple[DeviceSpec, ...], min_per_shard: int) -> list[int]:
    """Throughput-weighted contiguous partition of ``num_units`` across ``devices`` (counts per device)."""
    max_shards = max(1, min(len(devices), num_units // max(1, min_per_shard)))
    w = np.asarray([max(d.throughput, 1e-9) for d in devices[:max_shards]], dtype=float)
    w /= w.sum()
    counts = np.maximum(np.floor(w * num_units).astype(int), min_per_shard)
    while counts.sum() > num_units:  # trim the largest until exact
        i = int(np.argmax(counts))
        if counts[i] <= min_per_shard:
            break
        counts[i] -= 1
    while counts.sum() < num_units:  # grow the most under-allocated
        counts[int(np.argmax(w - counts / float(num_units)))] += 1
    return [int(c) for c in counts]


def decompose_model(
    model: Any,
    resources: Resources,
    *,
    n_data: int | None = None,
    min_components_per_shard: int = 1,
    prefer_data_parallel: bool = True,
) -> ModelDecomposition:
    """Decide how to place ``model`` across ``resources`` for model parallelism.

    The root node's decomposition descriptor drives the choice. A node that does not opt into the
    contract (or whose subtree fits replicated and is best served by data parallelism) yields an
    ``axis="none"`` plan -- replicate the model, shard the data, which is already optimal for large N.
    Otherwise the component / factor units are bin-packed across devices, carrying the per-cut reduction.
    """
    devices = tuple(resources.devices)
    dc = decomposition_for(model)
    if not dc.is_shardable or len(devices) < 2:
        return ModelDecomposition(
            DecompAxis.NONE,
            ReductionOp.REPLICATE,
            dc.num_units,
            (),
            rationale="not shardable or single device -> replicate model, data-parallel",
        )

    # Refuse model-parallelism when data-parallelism clearly wins: the whole model fits on one device and
    # there are not many more units than devices to gain from splitting (data-parallel has no model-axis
    # communication). The planner stays advisory; the executor may still force model-parallel.
    sized = size_model_tree(model)
    mem = min((d.memory_bytes or 0) for d in devices)
    fits_replicated = mem == 0 or sized.subtree_param_bytes <= mem
    if (
        prefer_data_parallel
        and fits_replicated
        and dc.num_units < 2 * len(devices)
        and (n_data is None or n_data >= 8 * len(devices))
    ):
        return ModelDecomposition(
            DecompAxis.NONE,
            ReductionOp.REPLICATE,
            dc.num_units,
            (),
            rationale="model fits replicated and N dominates -> data-parallel preferred over model-parallel",
        )

    counts = _unit_partition(dc.num_units, devices, min_components_per_shard)
    cuts: list[ModelCut] = []
    start = 0
    for device, c in zip(devices, counts):
        if c <= 0:
            continue
        cuts.append(ModelCut(device=device, start=start, stop=start + c, reduction=dc.reduction))
        start += c
    return ModelDecomposition(
        axis=dc.axis,
        reduction=dc.reduction,
        num_units=dc.num_units,
        cuts=tuple(cuts),
        rationale=f"model-parallel along {dc.axis.value}: {dc.num_units} units over {len(cuts)} device(s)",
        extra={"engine_axis": dc.engine_axis, "key_pooling": dc.key_pooling, "exact": dc.exact},
    )


__all__ = [
    "NodeSize",
    "size_model_tree",
    "shard_children",
    "ModelCut",
    "ModelDecomposition",
    "decompose_model",
]
