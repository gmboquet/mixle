"""Precision-aware planning for distributed EM across heterogeneous compute.

Large worker pools are rarely uniform: some workers may have GPU tensor cores,
while others are CPU-only or accuracy-oriented. This module chooses, per
worker, how many E-step rows to assign and which precision band to run. The
selected precision is the fastest supported band that still satisfies the
requested error budget. The plan also sizes the k-way reduction depth so
fixed-size sufficient-statistic payloads fold in ``O(log W)`` instead of a
single-root fan-in.

This module is the pure-Python planning layer. Spark, MPI, or other distributed
dispatchers consume the returned plan from the inference layer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from mixle.engines.affine import UNIT_ROUNDOFF

# Relative throughput multipliers per ``(device, precision)`` for planning.
# Lower precision is faster on GPUs; sub-float32 arithmetic is slower on CPUs
# without native support, and double-double arithmetic is much more expensive.
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
    """One worker's row allocation, precision, and effective throughput."""

    name: str
    rows: int
    precision: str
    effective_throughput: float


@dataclass(frozen=True)
class HeterogeneousPlan:
    """Assignments and reduction depth for heterogeneous execution."""

    assignments: tuple[WorkerAssignment, ...]
    reduce_depth: int

    def total_rows(self) -> int:
        """Return total rows assigned across workers."""
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
        # Fall back to the worker's most accurate supported and allowed precision.
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
