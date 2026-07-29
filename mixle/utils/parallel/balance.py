"""Automatic compute / memory / load-balancing planner for EM estimation of *any* mixle model.

The currency is **FLOPs per iteration**, with **memory as a hard constraint** -- memory decides whether
the model *fits*, compute decides how long the iteration *takes*, and the planner balances compute across
the cluster subject to memory.

An EM iteration is a fixed amount of work ``W = N * C`` FLOPs (``C`` = per-observation model cost from
:func:`compute_cost`). A worker grid of ``D`` data-parallel replicas x ``M`` model-shards gives every
worker ``W / (D*M)`` FLOPs, so the iteration time is ``max_worker_FLOPs / throughput + coupling``. To
balance the load we therefore:

  * prefer **data parallelism** (``D``) -- it has no cross-worker coupling (a data point's whole model
    lives on one worker) and balances trivially by equal row counts;
  * use **model parallelism** (``M``) only as forced -- by *memory* (the model does not fit: ``M >=
    ceil(bytes/mem)``) or by *compute concurrency* (too few data points to fill the cluster: with ``N``
    points only ``N`` data-replicas exist, so the rest of the cluster can only be used by splitting the
    model);
  * **balance the model split by FLOPs**, not bytes -- a memory-light but compute-heavy leaf (a GP, a big
    quadratic form) must not become the straggler everyone waits on.

This covers the whole spectrum the same way: a compact model on lots of data -> ``M=1, D=P`` (data-parallel);
a model too big for one worker -> ``M`` from memory, ``D`` fills the rest; a huge model on a *single*
observation (``N=1``) -> ``D=1, M=`` as many model shards as the model exposes; and an unbalanced
heterogeneous nest -> the FLOP cost model finds where the work is and the split equalizes it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mixle.stats.compute.decomposition import DecompAxis
from mixle.utils.parallel.model_decomposition import (
    ModelCut,
    best_parallel_axis,
    compute_cost,
)
from mixle.utils.parallel.planner import Resources


@dataclass(frozen=True)
class BalancePlan:
    """A worker-grid assignment for one EM estimation, balanced by compute under a memory constraint."""

    data_parallel: int  # D: independent data-shard replicas (no model-axis coupling)
    model_parallel: int  # M: model shards per replica (>=1; >1 only when memory or concurrency forces it)
    workers_used: int  # D * M
    workers_total: int  # P
    axis: DecompAxis  # the model axis split when M > 1 (NONE when pure data-parallel)
    model_cuts: tuple[ModelCut, ...]  # FLOP-balanced contiguous unit ranges of that axis
    model_flops: float  # per-observation compute proxy C
    model_bytes: int  # replicated model footprint
    per_worker_flops: float  # predicted busiest-worker FLOPs/iteration (the balance metric)
    fits: bool | None  # true/false when evidenced; None when any assigned device memory is unknown
    rationale: str
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_model_parallel(self) -> bool:
        """Whether the selected worker grid includes model parallelism."""
        return self.model_parallel > 1

    @property
    def workers_idle(self) -> int:
        """Number of available workers left unused by the selected grid."""
        return max(0, self.workers_total - self.workers_used)


def _balance_units(unit_works: tuple[float, ...], shards: int) -> list[int]:
    """Contiguous partition of units into ``shards`` groups with near-equal total work (counts per group)."""
    n = len(unit_works)
    if n == 0:
        raise ValueError("unit_works must not be empty.")
    works = np.asarray(unit_works, dtype=np.float64)
    if not np.all(np.isfinite(works)) or np.any(works < 0.0):
        raise ValueError("unit_works must contain finite non-negative values.")
    if isinstance(shards, bool) or not isinstance(shards, (int, np.integer)) or not 1 <= shards <= n:
        raise ValueError("shards must be an exact integer between one and the number of units.")
    cum = np.cumsum(np.asarray(unit_works, dtype=float))
    total = float(cum[-1]) if cum[-1] > 0 else 1.0
    boundaries = [0]
    for shard in range(1, shards):
        minimum = boundaries[-1] + 1
        maximum = n - (shards - shard)
        candidates = np.arange(minimum, maximum + 1)
        target = total * shard / shards
        boundary = int(candidates[np.argmin(np.abs(cum[candidates - 1] - target))])
        boundaries.append(boundary)
    boundaries.append(n)
    return [b - a for a, b in zip(boundaries[:-1], boundaries[1:])]


def balance_plan(model: Any, resources: Resources, *, n_data: int) -> BalancePlan:
    """Choose the ``(D data-parallel) x (M model-parallel)`` worker grid that balances compute under memory.

    Searches ``M`` from the memory-required minimum up to the model's splittable units, picking the grid
    that keeps the most workers busy (ties broken toward *smaller* ``M`` -- less coupling). Works for any
    model: a model with no splittable axis simply gets ``M=1`` (data-parallel / single worker)."""
    devices = tuple(resources.devices)
    p = len(devices)
    if isinstance(n_data, bool) or not isinstance(n_data, (int, np.integer)) or n_data <= 0:
        raise ValueError("n_data must be an exact positive integer.")
    n_data = int(n_data)
    flops, model_bytes = compute_cost(model)
    if not np.isfinite(flops) or flops < 0.0:
        raise ValueError("model FLOP cost must be finite and non-negative.")
    if isinstance(model_bytes, bool) or not isinstance(model_bytes, (int, np.integer)) or model_bytes < 0:
        raise ValueError("model byte cost must be an exact non-negative integer.")
    model_bytes = int(model_bytes)

    best = best_parallel_axis(model, p)
    max_units = best.num_units if best is not None else 1
    if best is not None:
        unit_works = np.asarray(best.unit_works, dtype=np.float64)
        if unit_works.shape != (best.num_units,) or not np.all(np.isfinite(unit_works)) or np.any(unit_works < 0.0):
            raise ValueError("parallel-axis unit work must be finite, non-negative, and schema-aligned.")
        if len(best.unit_bytes) != best.num_units or any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0
            for value in best.unit_bytes
        ):
            raise ValueError("parallel-axis unit bytes must be exact non-negative integers and schema-aligned.")

    known_memory = [device.memory_bytes for device in devices if device.memory_bytes is not None]
    mem = min(known_memory) if known_memory else None
    m_mem = math.ceil(model_bytes / mem) if mem is not None and model_bytes > mem else 1

    # The memory floor is a HARD lower bound on the model split: M must be at least m_mem (so each shard's
    # ~bytes/M fits a worker), capped by the units the model actually exposes. Search M up from there and
    # fill the rest of the cluster with data replicas; keep the most-utilizing grid, ties toward smaller M
    # (less coupling). If the axis cannot supply m_mem shards the model does not fit -- reported, not hidden.
    m_lo = max(1, min(m_mem, max_units, p))  # cannot use more model shards than workers (floor is bytes/P)
    best_grid = (m_lo, max(1, min(n_data, p // m_lo)))
    best_util = best_grid[0] * best_grid[1]
    for m in range(m_lo, min(max_units, p) + 1):
        d = min(n_data, p // m)
        if d < 1:
            continue
        util = m * d
        if util > best_util or (util == best_util and m < best_grid[0]):
            best_util, best_grid = util, (m, d)
    m, d = best_grid
    workers_used = m * d

    cuts: tuple[ModelCut, ...] = ()
    cut_works = [float(flops)]
    cut_bytes = [model_bytes]
    axis = DecompAxis.NONE
    if m > 1 and best is not None:
        counts = _balance_units(best.unit_works, m)
        fixed_flops = max(0.0, float(flops) - float(sum(best.unit_works)))
        fixed_bytes = max(0, model_bytes - int(sum(best.unit_bytes)))
        axis = best.axis
        out: list[ModelCut] = []
        cut_works = []
        cut_bytes = []
        start = 0
        for dev, c in zip(devices, counts):
            out.append(ModelCut(device=dev, start=start, stop=start + c, reduction=best.reduction))
            cut_works.append(fixed_flops + float(sum(best.unit_works[start : start + c])))
            cut_bytes.append(fixed_bytes + int(sum(best.unit_bytes[start : start + c])))
            start += c
        cuts = tuple(out)
    rows_per_replica = math.ceil(n_data / d)
    per_worker = rows_per_replica * max(cut_works)

    fits: bool | None = True
    for worker_position, device in enumerate(devices[:workers_used]):
        required = cut_bytes[worker_position % m]
        if device.memory_bytes is None:
            if fits is True:
                fits = None
        elif required > device.memory_bytes:
            fits = False
            break

    if workers_used < p and m == 1 and max_units <= 1 and n_data < p:
        # the explicit corner: too few observations to data-parallel AND the model exposes no axis to split
        # (e.g. a single dense HMM). Naive model-parallelism can't help -- this needs a STRUCTURED
        # decomposition (sparse/banded/Kronecker transitions, or a Composite/Mixture of sub-models).
        why = (
            f"single-worker ({workers_used}/{p} used): N={n_data} too small to data-parallel and the model "
            f"exposes no splittable axis (atomic). Model-parallelism needs a structured decomposition."
        )
    elif m == 1:
        why = f"data-parallel: model fits and N={n_data} fills {d}/{p} workers (no model-axis coupling)"
    elif d == 1:
        why = f"model-parallel x{m}: N={n_data} too small to data-parallel, split the model across {m} workers"
    else:
        why = f"data x model grid {d}x{m}={workers_used}/{p}: model split {m}-way (memory/concurrency), data {d}-way"
    if fits is False:
        why += f"  [WARNING: model needs {m_mem} shards for memory but axis offers only {max_units}]"
    elif fits is None:
        why += "  [MEMORY FIT UNKNOWN: at least one assigned device has no memory evidence]"

    return BalancePlan(
        data_parallel=d,
        model_parallel=m,
        workers_used=workers_used,
        workers_total=p,
        axis=axis,
        model_cuts=cuts,
        model_flops=flops,
        model_bytes=model_bytes,
        per_worker_flops=per_worker,
        fits=fits,
        rationale=why,
        extra={
            "max_units": max_units,
            "m_mem": m_mem,
            "best_axis": None if best is None else best.path,
            "rows_per_data_replica": rows_per_replica,
            "per_cut_flops": tuple(cut_works),
            "per_cut_bytes": tuple(cut_bytes),
            "fit_status": "unknown" if fits is None else ("fits" if fits else "does_not_fit"),
        },
    )


def auto_balanced_estimator(
    estimator: Any, model: Any, resources: Any = None, *, n_data: int
) -> tuple[Any, BalancePlan]:
    """Return an estimator only when :func:`balance_plan` needs no model placement.

    Data-parallel plans return the original estimator. Plans that require model
    storage to be sharded raise ``UnrealizedModelPlacementError`` because the
    in-process component-thread runtime does not execute their cuts.
    """
    from mixle.utils.parallel.model_parallel import UnrealizedModelPlacementError
    from mixle.utils.parallel.planner import Resources

    resources = Resources.local() if resources is None else resources
    plan = balance_plan(model, resources, n_data=n_data)
    if plan.is_model_parallel:
        raise UnrealizedModelPlacementError(plan)
    return estimator, plan


__all__ = ["BalancePlan", "balance_plan", "auto_balanced_estimator"]
