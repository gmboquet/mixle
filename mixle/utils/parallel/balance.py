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
    fits: bool  # whether the model shards actually fit in per-worker memory
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
    shards = max(1, min(shards, n))
    cum = np.cumsum(np.asarray(unit_works, dtype=float))
    total = float(cum[-1]) if cum[-1] > 0 else 1.0
    counts = [0] * shards
    s = 0
    for i in range(n):
        # close shard s once its cumulative work crosses its equal-share quantile
        while s < shards - 1 and cum[i] / total > (s + 1) / shards + 1e-12:
            s += 1
        counts[s] += 1
    return [c for c in counts if c > 0]


def balance_plan(model: Any, resources: Resources, *, n_data: int) -> BalancePlan:
    """Choose the ``(D data-parallel) x (M model-parallel)`` worker grid that balances compute under memory.

    Searches ``M`` from the memory-required minimum up to the model's splittable units, picking the grid
    that keeps the most workers busy (ties broken toward *smaller* ``M`` -- less coupling). Works for any
    model: a model with no splittable axis simply gets ``M=1`` (data-parallel / single worker)."""
    devices = tuple(resources.devices)
    p = len(devices)
    n_data = max(1, int(n_data))
    flops, model_bytes = compute_cost(model)

    best = best_parallel_axis(model, p)
    max_units = best.num_units if best is not None else 1

    mem = min((d.memory_bytes or 0) for d in devices)
    m_mem = math.ceil(model_bytes / mem) if mem and model_bytes > mem else 1

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
    fits = mem == 0 or (model_bytes / m) <= mem

    total_flops = flops * n_data
    per_worker = total_flops / max(1, workers_used)
    if m > 1 and best is not None:  # a fat indivisible unit floors the busiest worker (Amdahl)
        per_worker = max(per_worker, n_data * (max(best.unit_works) if best.unit_works else flops))

    cuts: tuple[ModelCut, ...] = ()
    axis = DecompAxis.NONE
    if m > 1 and best is not None:
        counts = _balance_units(best.unit_works, m)
        axis = best.axis
        out: list[ModelCut] = []
        start = 0
        for dev, c in zip(devices, counts):
            out.append(ModelCut(device=dev, start=start, stop=start + c, reduction=best.reduction))
            start += c
        cuts = tuple(out)

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
    if not fits:
        why += f"  [WARNING: model needs {m_mem} shards for memory but axis offers only {max_units}]"

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
        extra={"max_units": max_units, "m_mem": m_mem, "best_axis": None if best is None else best.path},
    )


def auto_balanced_estimator(
    estimator: Any, model: Any, resources: Any = None, *, n_data: int
) -> tuple[Any, BalancePlan]:
    """Realize :func:`balance_plan` -- return ``(estimator, plan)`` ready to drive ``optimize``.

    When the plan is model-parallel the estimator is wrapped in :class:`ModelParallelEstimator` sized to
    ``plan.model_parallel`` (the model axis distributes inside each worker / across the model shards);
    otherwise the plain estimator is returned (pure data-parallel). The *data* degree ``plan.data_parallel``
    is realized by the data backend you pass to ``optimize`` (``"local"|"mp"|"spark"|"mpi"``), so
    ``optimize(data, returned_estimator, backend=...)`` runs the full ``D x M`` grid. ``resources`` defaults
    to the local CPU slots; pass ``Resources.from_spark(sc)`` / ``from_mpi()`` to plan for a real cluster."""
    from mixle.utils.parallel.model_parallel import ModelParallelEstimator
    from mixle.utils.parallel.planner import Resources

    resources = Resources.local() if resources is None else resources
    plan = balance_plan(model, resources, n_data=n_data)
    if plan.is_model_parallel:
        return ModelParallelEstimator(estimator, num_workers=plan.model_parallel), plan
    return estimator, plan


__all__ = ["BalancePlan", "balance_plan", "auto_balanced_estimator"]
