"""Structural model-decomposition planner for model-parallel execution.

Turns *any* mixle model into a model-parallel placement by walking the opt-in decomposition
contract (:mod:`mixle.stats.compute.decomposition`) instead of a blind reflective ``__dict__`` walk:

  * :func:`size_model_tree` -- a structural byte sizing of the model tree (own params per node, recursing
    into children via the declared axis), so shared subtrees are not double-counted and a node's own
    parameter footprint (e.g. an HMM's dense ``S*S`` transition block) is visible rather than silently
    zeroed. Replaces the reflective ``estimate_model_nbytes`` for sizing decisions.
  * :func:`decompose_model` -- choose the cut (data vs model vs none) for the root shardable node and
    bin-pack the component / factor units across the device budget, carrying the per-cut reduction. This
    is the general form of, and a real consumer-shaped output for, ``planner.model_sharding_plan`` (which
    only ever understood mixture components and had no consumer).

It works for every family: nodes that do not opt into the contract report ``Decomposition.atomic()`` and
are simply replicated, preserving the ordinary data-parallel path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mixle.stats.compute.decomposition import DecompAxis, Decomposition, ReductionOp, decomposition_for
from mixle.utils.parallel.planner import DeviceSpec, Resources


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


def _own_work(node: Any) -> float:
    """A relative per-observation COMPUTE weight for a node's own emission (a FLOP proxy).

    Scoring an exponential-family emission costs ~O(#scalar parameters) per observation (the dot product
    / quadratic form), so we count the node's own numeric scalars: ``ndarray.size`` for array params
    (MVGaussian's D*D covariance, a Categorical's V probabilities) plus one per float/int scalar
    (Gaussian's mu, sigma2 -- stored as plain floats, not arrays). Children are excluded (summed by
    recursion). A base of 1.0 keeps every leaf non-zero. This is the cost that balances the work split;
    swap in :func:`mixle.utils.parallel.planner.calibrate_resources` timings when an exact model is needed.
    """
    w = 1.0
    for v in getattr(node, "__dict__", {}).values():
        if isinstance(v, np.ndarray):
            w += float(v.size)
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            w += 1.0
    return w


def cost_children(model: Any) -> tuple[Any, ...]:
    """ALL child distributions of a node (for COST), not just the shardable ones.

    ``shard_children`` returns only the axis a node can be *split* along; for *costing* we need every
    nested distribution's compute counted -- e.g. an HMM is atomic (not shardable here) yet its ``S``
    emission distributions and ``len_dist`` are real work, and a heavy leaf buried under a non-shardable
    wrapper still costs FLOPs. Discovery is reflective (any ``SequenceEncodableProbabilityDistribution``
    held directly or inside a list/tuple/dict), so it works for *any* model without per-family wiring.
    """
    from mixle.stats.compute.pdist import SequenceEncodableProbabilityDistribution as _Dist

    out: list[Any] = []
    for v in getattr(model, "__dict__", {}).values():
        if isinstance(v, _Dist):
            out.append(v)
        elif isinstance(v, (list, tuple)):
            out.extend(x for x in v if isinstance(x, _Dist))
        elif isinstance(v, dict):
            out.extend(x for x in v.values() if isinstance(x, _Dist))
    return tuple(out)


def subtree_work(model: Any, _seen: dict[int, bool] | None = None) -> float:
    """Total compute weight of a model subtree -- own emission cost plus ALL descendants (counted once).

    Recurses over :func:`cost_children` (every nested distribution), so a unit's cost includes heavy
    subtrees the executor can't split (a nested HMM, a GP leaf), making the balance explicit about where the
    FLOPs actually are -- not just where the model happens to be shardable.
    """
    seen = _seen if _seen is not None else {}
    if id(model) in seen:
        return 0.0  # shared subtree: counted once (mirrors the byte-sizing policy)
    seen[id(model)] = True
    total = _own_work(model)
    for child in cost_children(model):
        total += subtree_work(child, seen)
    return total


def compute_cost(model: Any, _seen: dict[int, bool] | None = None) -> tuple[float, int]:
    """``(flops_per_observation_proxy, bytes)`` for the whole model -- compute load and memory footprint,
    the two resources the balancer trades off (compute is the load, memory is the constraint)."""
    seen = _seen if _seen is not None else {}
    if id(model) in seen:
        return 0.0, 0
    seen[id(model)] = True
    flops = _own_work(model)
    total_bytes = _own_param_bytes(model)
    for child in cost_children(model):
        cf, cb = compute_cost(child, seen)
        flops += cf
        total_bytes += cb
    return flops, total_bytes


def shard_children(node: Any, dc: Decomposition | None = None) -> tuple[Any, ...]:
    """Return the actual child distributions a node splits along its declared axis (else ``()``)."""
    dc = dc if dc is not None else decomposition_for(node)
    if dc.axis is DecompAxis.COMPONENT:
        return tuple(getattr(node, "components", ()) or ())
    if dc.axis is DecompAxis.FACTOR:
        return tuple(getattr(node, "dists", ()) or ())
    if dc.axis is DecompAxis.TOPIC:
        return tuple(getattr(node, "topics", ()) or ())
    if dc.axis is DecompAxis.STATE:
        return tuple(getattr(node, "topics", ()) or getattr(node, "components", ()) or ())
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
    subtree_work: float = 1.0  # own + all descendants compute weight (the work this subtree's E-step costs)
    children: tuple[NodeSize, ...] = ()


def size_model_tree(model: Any, _path: str = "", _seen: set[int] | None = None) -> NodeSize:
    """Recursively size the model tree via the decomposition contract (shared subtrees counted once)."""
    seen = _seen if _seen is not None else set()
    dc = decomposition_for(model)
    children = shard_children(model, dc)
    child_sizes: list[NodeSize] = []
    subtree = _own_param_bytes(model)
    work = _own_work(model)
    for i, child in enumerate(children):
        if child is None or id(child) in seen:
            continue
        seen.add(id(child))
        role = dc.child_roles[i] if i < len(dc.child_roles) else f"{dc.axis.value}_{i}"
        cs = size_model_tree(child, f"{_path}/{role}".lstrip("/"), seen)
        child_sizes.append(cs)
        subtree += cs.subtree_param_bytes
        work += cs.subtree_work
    return NodeSize(
        path=_path,
        axis=dc.axis,
        num_units=dc.num_units,
        own_param_bytes=_own_param_bytes(model),
        subtree_param_bytes=subtree,
        subtree_work=work,
        children=tuple(child_sizes),
    )


# --- whole-tree axis enumeration (every shardable axis, anywhere in the tree, with its work) ------
@dataclass(frozen=True)
class AxisCandidate:
    """One shardable axis somewhere in the model tree, with the compute weight of each of its units."""

    path: str
    axis: DecompAxis
    reduction: ReductionOp
    num_units: int
    unit_works: tuple[float, ...]  # subtree compute weight of each unit -> drives the cost-balanced split

    @property
    def total_work(self) -> float:
        """Total compute-weight proxy across all units in this candidate axis."""
        return float(sum(self.unit_works))


def tree_axes(model: Any) -> list[AxisCandidate]:
    """Enumerate every shardable axis anywhere in the tree (not just the root), each with per-unit work."""
    out: list[AxisCandidate] = []
    seen: set[int] = set()

    def walk(node: Any, path: str) -> None:
        if id(node) in seen:
            return
        seen.add(id(node))
        dc = decomposition_for(node)
        kids = shard_children(node, dc)
        if dc.is_shardable and len(kids) == dc.num_units and dc.num_units >= 2:
            out.append(
                AxisCandidate(path or "/", dc.axis, dc.reduction, dc.num_units, tuple(subtree_work(k) for k in kids))
            )
        for i, child in enumerate(kids):
            if child is not None:
                role = dc.child_roles[i] if i < len(dc.child_roles) else f"{dc.axis.value}_{i}"
                walk(child, f"{path}/{role}".lstrip("/"))

    walk(model, "")
    return out


def best_parallel_axis(model: Any, max_workers: int | None = None) -> AxisCandidate | None:
    """Pick the axis whose parallelization removes the most serial wall-time (heaviest, parallelizable).

    Benefit of cutting an axis with P available workers is ``total_work * (1 - 1/min(P, num_units))`` --
    favouring the axis that carries the most work AND has enough units to keep the workers busy. This
    looks at the WHOLE tree, so a heavy mixture nested inside a thin composite is found (the root-only
    planner missed it). Returns ``None`` when nothing in the tree is worth splitting.
    """
    candidates = tree_axes(model)
    if not candidates:
        return None

    def benefit(c: AxisCandidate) -> float:
        # serial-time removed by cutting this axis with P workers under greedy scheduling: the parallel
        # time is bounded BELOW by the heaviest single unit (a fat unit can't be split here), so an
        # imbalanced axis -- e.g. a composite whose two factors are [heavy mixture, light leaf] -- scores
        # near zero even though its total_work (which includes the child) is large. This stops a thin
        # parent from masking the real, balanced axis nested inside it.
        p = c.num_units if max_workers is None else min(max_workers, c.num_units)
        parallel_time = max(max(c.unit_works), c.total_work / max(1, p))
        return c.total_work - parallel_time

    return max(candidates, key=benefit)


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
        """Whether this decomposition uses more than one model shard."""
        return self.axis is not DecompAxis.NONE and len(self.cuts) > 1


def _cost_partition(unit_works: tuple[float, ...], devices: tuple[DeviceSpec, ...], min_per_shard: int) -> list[int]:
    """Contiguous partition of units across devices that BALANCES WORK (not count), throughput-weighted.

    Device ``s`` should carry a fraction of the total work proportional to its throughput; we walk the
    units in order and close a shard once its cumulative work crosses that device's capacity quantile, so
    a fast device gets more (or heavier) units. Empty shards are dropped. Counts -- not works -- are
    returned because the cut ranges are contiguous unit indices.
    """
    n = len(unit_works)
    max_shards = max(1, min(len(devices), n // max(1, min_per_shard)))
    tput = np.asarray([max(d.throughput, 1e-9) for d in devices[:max_shards]], dtype=float)
    capacity = np.cumsum(tput / tput.sum())  # cumulative throughput fraction per device
    cum = np.cumsum(np.asarray(unit_works, dtype=float))
    total = float(cum[-1]) if cum[-1] > 0 else 1.0
    counts = [0] * max_shards
    s = 0
    for i in range(n):
        while s < max_shards - 1 and cum[i] / total > capacity[s] + 1e-12:
            s += 1
        counts[s] += 1
    return [c for c in counts if c > 0]


def decompose_model(
    model: Any,
    resources: Resources,
    *,
    n_data: int | None = None,
    min_components_per_shard: int = 1,
    prefer_data_parallel: bool = True,
) -> ModelDecomposition:
    """Decide how to place ``model`` across ``resources`` for model parallelism.

    The WHOLE tree is searched for the heaviest worth-splitting axis (:func:`best_parallel_axis`), so a
    big mixture nested inside a thin composite is found -- not just the root. A model that does not opt
    into the contract, or whose subtree fits replicated and is best served by data parallelism, yields an
    ``axis="none"`` plan (replicate the model, shard the data -- already optimal for large N). Otherwise
    that axis's units are bin-packed across devices BALANCED BY COMPUTE COST, carrying the per-cut
    reduction.
    """
    devices = tuple(resources.devices)
    best = best_parallel_axis(model, max_workers=len(devices))
    if best is None or len(devices) < 2:
        return ModelDecomposition(
            DecompAxis.NONE,
            ReductionOp.REPLICATE,
            decomposition_for(model).num_units,
            (),
            rationale="no worthwhile shardable axis or single device -> replicate model, data-parallel",
        )

    # Refuse model-parallelism when data-parallelism clearly wins: the whole model fits on one device and
    # the chosen axis does not have many more units than devices to gain from splitting (data-parallel has
    # no model-axis communication). The planner stays advisory; the executor may still force model-parallel.
    sized = size_model_tree(model)
    mem = min((d.memory_bytes or 0) for d in devices)
    fits_replicated = mem == 0 or sized.subtree_param_bytes <= mem
    if (
        prefer_data_parallel
        and fits_replicated
        and best.num_units < 2 * len(devices)
        and (n_data is None or n_data >= 8 * len(devices))
    ):
        return ModelDecomposition(
            DecompAxis.NONE,
            ReductionOp.REPLICATE,
            best.num_units,
            (),
            rationale="model fits replicated and N dominates -> data-parallel preferred over model-parallel",
        )

    counts = _cost_partition(best.unit_works, devices, min_components_per_shard)
    cuts: list[ModelCut] = []
    start = 0
    for device, c in zip(devices, counts):
        if c <= 0:
            continue
        cuts.append(ModelCut(device=device, start=start, stop=start + c, reduction=best.reduction))
        start += c
    where = f" at {best.path}" if best.path not in ("", "/") else ""
    return ModelDecomposition(
        axis=best.axis,
        reduction=best.reduction,
        num_units=best.num_units,
        cuts=tuple(cuts),
        rationale=f"model-parallel along {best.axis.value}{where}: {best.num_units} units over {len(cuts)} device(s)",
        extra={"path": best.path, "total_work": best.total_work},
    )


__all__ = [
    "NodeSize",
    "size_model_tree",
    "shard_children",
    "cost_children",
    "subtree_work",
    "compute_cost",
    "AxisCandidate",
    "tree_axes",
    "best_parallel_axis",
    "ModelCut",
    "ModelDecomposition",
    "decompose_model",
]
