"""Precision-aware planning for distributed EM across HETEROGENEOUS compute.

A 1000-worker pool is rarely uniform: some workers have GPU tensor cores (fast at fp8/bf16), others are
CPU-only (fp32, or vectorized double-double for accuracy). This module decides, per worker, (a) how many
rows of the E-step to give it -- balanced by its throughput -- and (b) which precision band to run, the
fastest one its hardware supports that still meets the accuracy budget. It also sizes the k-way tree
reduce depth so the fixed-size sufficient-statistic payloads fold in ``O(log W)`` rather than a single-root
fan-in.

This is the planning layer (pure-Python, testable); the actual Spark ``treeReduce`` / MPI ``comm.reduce``
dispatch that consumes the plan lives in mixle.inference and needs a real cluster to exercise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from mixle.engines.affine import UNIT_ROUNDOFF

# Rough relative throughput multiplier per (device, precision) -- lower precision is faster on a GPU,
# while on a CPU sub-f32 is a slowdown (no native arithmetic) and double-double costs ~15x.
_THROUGHPUT = {
    ("gpu", "fp8"): 4.0,
    ("gpu", "bfloat16"): 2.5,
    ("gpu", "float16"): 2.5,
    ("gpu", "float32"): 1.5,
    ("gpu", "float64"): 1.0,
    ("cpu", "float32"): 1.4,
    ("cpu", "float64"): 1.0,
    ("cpu", "dd"): 1.0 / 15.0,
}


@dataclass(frozen=True)
class Worker:
    """A compute worker: its device, the precisions it can run (any order), and a base throughput."""

    name: str
    device: str  # "cpu" or "gpu"
    precisions: tuple[str, ...]
    base_throughput: float = 1.0


@dataclass(frozen=True)
class WorkerAssignment:
    name: str
    rows: int
    precision: str
    effective_throughput: float


@dataclass(frozen=True)
class HeterogeneousPlan:
    assignments: tuple[WorkerAssignment, ...]
    reduce_depth: int

    def total_rows(self) -> int:
        return sum(a.rows for a in self.assignments)


def _meets_budget(precision: str, op_count: int, magnitude: float, target_rel_error: float | None) -> bool:
    if target_rel_error is None:
        return True
    u = UNIT_ROUNDOFF.get(precision)
    if u is None:
        return False
    return op_count * u <= target_rel_error  # roundoff accumulates ~op_count * u (relative)


def _best_precision(worker: Worker, allowed: tuple[str, ...], op_count: int, target_rel_error: float | None) -> str:
    """The highest-throughput precision the worker supports that is allowed and meets the budget."""
    candidates = [p for p in worker.precisions if p in allowed and _meets_budget(p, op_count, 1.0, target_rel_error)]
    if not candidates:
        # fall back to the worker's most accurate supported+allowed precision (ignore the budget but be honest)
        candidates = [p for p in worker.precisions if p in allowed] or list(worker.precisions)
    return max(candidates, key=lambda p: _THROUGHPUT.get((worker.device, p), 0.5))


def plan_heterogeneous(
    workers: list[Worker],
    n_rows: int,
    allowed_precisions: tuple[str, ...] = ("fp8", "bfloat16", "float16", "float32", "float64", "dd"),
    target_rel_error: float | None = None,
    op_count: int = 1000,
) -> HeterogeneousPlan:
    """Assign rows + a precision band to each worker, balanced by precision-adjusted throughput.

    Each worker runs the fastest precision its hardware supports that stays within ``target_rel_error``
    (``None`` = no accuracy constraint); rows are split proportionally to the resulting throughput so all
    workers finish together. ``reduce_depth`` is the k-way tree depth for folding the sufficient-statistic
    payloads (``~ceil(log2(W)/2)``), avoiding the single-root fan-in.
    """
    if not workers:
        raise ValueError("need at least one worker")
    chosen = []
    for w in workers:
        p = _best_precision(w, allowed_precisions, op_count, target_rel_error)
        eff = w.base_throughput * _THROUGHPUT.get((w.device, p), 0.5)
        chosen.append((w, p, eff))

    total_eff = sum(eff for _, _, eff in chosen)
    assignments: list[WorkerAssignment] = []
    assigned = 0
    for i, (w, p, eff) in enumerate(chosen):
        if i == len(chosen) - 1:
            rows = n_rows - assigned  # last worker takes the remainder (exact total)
        else:
            rows = int(round(n_rows * eff / total_eff))
            rows = min(rows, n_rows - assigned)
        assigned += rows
        assignments.append(WorkerAssignment(w.name, rows, p, eff))

    depth = max(1, math.ceil(math.log2(max(2, len(workers))) / 2))
    return HeterogeneousPlan(tuple(assignments), depth)
