"""MPI distributed EM transport: ``comm.reduce`` (a real tree fold) over the verified combine-tree.

The MPI fabric over the same sharded-E-step + tree-reduce algorithm: each rank scores its shard to a
fixed-size ``(count, sufficient-stat)`` payload, and ``comm.reduce(payload, op=combine, root=0)`` folds
them -- mpi4py's lowercase object-mode ``reduce`` performs a genuine reduction tree (``O(log W)``), not a
gather-to-root. The combine operates on freshly-deserialized payloads (MPI ships them between ranks), so
the in-place ``combine()`` is safe. ``combine`` is associative + commutative, as MPI reduce requires.

Run under ``mpirun -n W python your_script.py``; each rank slices its contiguous portion of the data.
"""

from __future__ import annotations

from typing import Any

from mixle.inference.heterogeneous_executor import _shard_estep


def _rank_shard(data: Any, rank: int, size: int) -> Any:
    n = len(data)
    lo = rank * n // size
    hi = (rank + 1) * n // size
    return data[lo:hi]


def mpi_em_step(comm: Any, estimator: Any, model: Any, data: Any) -> Any:
    """One EM step across MPI ranks: each scores its shard, ``comm.reduce`` folds, rank 0 estimates.

    Returns the new model on rank 0 and ``None`` on the others (use :func:`mpi_fit` for the broadcast loop).
    """
    size = comm.Get_size()
    rank = comm.Get_rank()
    shard = _rank_shard(data, rank, size)
    local = _shard_estep(estimator, model, shard) if len(shard) else (0, None)
    factory = estimator.accumulator_factory()

    def combine(a: tuple[int, Any], b: tuple[int, Any]) -> tuple[int, Any]:
        if a[1] is None:
            return b
        if b[1] is None:
            return a
        acc = factory.make().from_value(a[1])
        acc.combine(b[1])
        return a[0] + b[0], acc.value()

    folded = comm.reduce(local, op=combine, root=0)
    if rank == 0:
        return estimator.estimate(float(folded[0]), folded[1])
    return None


def mpi_fit(comm: Any, model: Any, data: Any, max_its: int = 10) -> Any:
    """Run ``max_its`` EM iterations across MPI ranks; the new model is broadcast each iteration.

    All ranks return the same fitted model.
    """
    estimator = model.estimator()
    current = model
    for _ in range(max_its):
        new = mpi_em_step(comm, estimator, current, data)  # model on rank 0, None elsewhere
        current = comm.bcast(new, root=0)  # share the new estimate with every rank
    return current
