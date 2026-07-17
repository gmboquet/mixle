"""mpi4py backend for distributed estimation (SPMD).

Every MPI rank runs the same script; rank 0 plays the Spark driver. Each rank
holds a shard of the raw data, encodes it locally once, and keeps the encoded
chunks resident (process persistence is free under MPI). Per EM iteration the
model is broadcast from the root for bitwise consistency, every rank
accumulates sufficient statistics over its shard, and the per-rank
``(count, accumulator.value())`` payloads fold with ``comm.reduce`` -- mpi4py's
lowercase, pickle-based object-mode reduce, which performs a genuine ``O(log W)``
binary reduction tree rather than a gather-to-root: each internal tree node
combines two already-deserialized payloads via a fresh accumulator's
``combine()``, so no single rank ever folds more than ``O(log W)`` payloads.
This is the same technique the Spark transport uses via ``RDD.treeReduce`` and
the local executor uses via its in-process combine tree (see
:mod:`mixle.inference.spark_executor` and :mod:`mixle.inference.heterogeneous_executor`).
The root then applies ``key_merge``/``key_replace`` once globally, runs the
M-step, and broadcasts the new model - so every rank returns an identical
distribution and the surrounding ``optimize`` loop stays in lockstep across
ranks with no further coordination.

Usage (run with ``mpiexec -n 4 python script.py``)::

    from mixle.utils.parallel.mpi import MPIEncodedData, mpi_out
    from mixle.inference.estimation import optimize

    data = load_data()                       # every rank loads (or root_only=True)
    enc = MPIEncodedData(data, estimator=est)
    model = optimize(None, est, enc_data=enc, max_its=50, out=mpi_out())

``mpi_out()`` returns ``sys.stdout`` on the root and a throwaway buffer on
other ranks so iteration logging is printed once. Log-density sums are
``allreduce``-d, so the convergence test in ``optimize`` agrees on all ranks.

mpi4py is an optional dependency (``pip install mixle[mpi]``); importing
this module without it raises ImportError.

This is mixle's one MPI EM transport: an :class:`~mixle.utils.parallel.planner.EncodedDataHandle` that
plugs directly into ``optimize``, inheriting its convergence loop and logging. (A second, standalone
``comm.reduce``-based transport lived at ``mixle.inference.mpi_executor`` pre-0.8.0; it was folded in
here -- its ``O(log W)`` tree-reduce technique is what ``_fold_and_share``/``_fold_value_and_share``
now use, in place of the old gather-to-root loop -- and removed as a redundant second entry point.)
"""

import io
import pickle
import sys
from collections.abc import Sequence
from typing import Any, Optional

import numpy as np

from mixle.utils.optional_deps import HAS_MPI4PY, MPI, require
from mixle.utils.parallel.planner import EncodedDataHandle

__all__ = ["MPIEncodedData", "mpi_out"]

_PROTO = pickle.HIGHEST_PROTOCOL


def mpi_out(comm: Optional["MPI.Comm"] = None, root: int = 0):
    """sys.stdout on the root rank, a discarded buffer elsewhere."""
    if not HAS_MPI4PY:
        require("mpi4py", "mpi")
    comm = MPI.COMM_WORLD if comm is None else comm
    return sys.stdout if comm.Get_rank() == root else io.StringIO()


class MPIEncodedData(EncodedDataHandle):
    """Encoded-data handle sharded across MPI ranks (SPMD).

    Drop-in for the ``enc_data`` argument of ``optimize``/``best_of``/
    ``seq_estimate``/``seq_initialize``/``seq_log_density_sum``; construct it
    and call those functions identically on every rank.

    Args:
        data (Sequence): Raw observations. With ``root_only=False`` (default)
            every rank passes the SAME full dataset and keeps the round-robin
            shard ``data[rank::size]``. With ``root_only=True`` only the root
            needs real data (other ranks may pass None); shards are scattered.
        estimator (Optional[ParameterEstimator]): Used to build the encoder
            when ``encoder`` is not given.
        encoder (Optional[DataSequenceEncoder]): Explicit encoder.
        sub_chunks (int): Encoded sub-chunks per rank (bounds peak memory of
            the vectorized update).
        comm (Optional[MPI.Comm]): Communicator (default COMM_WORLD).
        root (int): Driver rank for the combine/M-step.
    """

    def __init__(
        self,
        data: Sequence[Any] | None,
        estimator=None,
        encoder=None,
        sub_chunks: int = 1,
        comm: Optional["MPI.Comm"] = None,
        root: int = 0,
        root_only: bool = False,
    ):
        if not HAS_MPI4PY:
            require("mpi4py", "mpi")
        self.comm = MPI.COMM_WORLD if comm is None else comm
        self.root = root
        self.rank = self.comm.Get_rank()
        self.world = self.comm.Get_size()

        if encoder is None:
            if estimator is None:
                raise ValueError("MPIEncodedData requires an estimator or an explicit encoder.")
            encoder = estimator.accumulator_factory().make().acc_to_encoder()

        if root_only:
            if self.rank == self.root:
                if data is None:
                    raise ValueError("root_only=True requires data on the root rank.")
                shards = [[data[j] for j in range(i, len(data), self.world)] for i in range(self.world)]
            else:
                shards = None
            shard = self.comm.scatter(shards, root=self.root)
        else:
            if data is None:
                raise ValueError("every rank must pass data when root_only=False.")
            shard = [data[j] for j in range(self.rank, len(data), self.world)]

        n = len(shard)
        k = max(1, min(int(sub_chunks), n)) if n else 1
        self._enc_chunks = []
        for i in range(k):
            part = [shard[j] for j in range(i, n, k)]
            if part:
                self._enc_chunks.append((len(part), encoder.seq_encode(part)))

        self.size = self.comm.allreduce(float(n), op=MPI.SUM)

    # -- local accumulation --------------------------------------------------

    def _local_update(self, estimator, model) -> tuple[float, Any]:
        accumulator = estimator.accumulator_factory().make()
        count = 0.0
        for sz, x in self._enc_chunks:
            count += sz
            accumulator.seq_update(x, np.ones(sz), model)
        return count, accumulator.value()

    def _tree_combine(self, estimator):
        """A pairwise ``(count, value)`` combiner for ``comm.reduce``: fresh accumulator per tree node,
        seeded via ``from_value`` and folded via ``combine`` -- the same shape used by
        :func:`mixle.inference.heterogeneous_executor.tree_reduce_values` and the Spark transport, so a
        shared-reference ``value()`` is never mutated in place (the HMM-stat aliasing hazard)."""
        factory = estimator.accumulator_factory()

        def combine(a: tuple[float, Any], b: tuple[float, Any]) -> tuple[float, Any]:
            acc = factory.make().from_value(a[1])
            acc.combine(b[1])
            return a[0] + b[0], acc.value()

        return combine

    def _fold_and_share(self, estimator, local: tuple[float, Any]):
        """Fold per-rank stats with an O(log W) reduction tree, M-step + broadcast the model from the root."""
        folded = self.comm.reduce(local, op=self._tree_combine(estimator), root=self.root)
        if self.rank == self.root:
            nobs, value = folded
            accumulator = estimator.accumulator_factory().make().from_value(value)
            stats_dict = dict()
            accumulator.key_merge(stats_dict)
            accumulator.key_replace(stats_dict)
            model_b = pickle.dumps(estimator.estimate(nobs, accumulator.value()), protocol=_PROTO)
        else:
            model_b = None
        return pickle.loads(self.comm.bcast(model_b, root=self.root))

    def _fold_value_and_share(self, estimator, local: tuple[float, Any]) -> tuple[float, Any]:
        """Fold per-rank stats with an O(log W) reduction tree, key-tie on root, broadcast the folded value."""
        folded = self.comm.reduce(local, op=self._tree_combine(estimator), root=self.root)
        if self.rank == self.root:
            nobs, value = folded
            accumulator = estimator.accumulator_factory().make().from_value(value)
            stats_dict = dict()
            accumulator.key_merge(stats_dict)
            accumulator.key_replace(stats_dict)
            payload_b = pickle.dumps((nobs, accumulator.value()), protocol=_PROTO)
        else:
            payload_b = None
        return pickle.loads(self.comm.bcast(payload_b, root=self.root))

    # -- protocol recognized by mixle.stats dispatch -------------------------

    def pysp_seq_estimate(self, estimator, prev_estimate):
        """One distributed EM step; every rank returns the identical model."""
        # broadcast the root's model so all ranks accumulate against the same
        # floating-point parameters even if a caller diverged
        model = pickle.loads(self.comm.bcast(pickle.dumps(prev_estimate, protocol=_PROTO), root=self.root))
        return self._fold_and_share(estimator, self._local_update(estimator, model))

    def pysp_seq_initialize(self, estimator, rng: np.random.RandomState, p: float):
        """Distributed randomized initialization; identical model on all ranks."""
        if self.rank == self.root:
            seeds = [int(s) for s in rng.randint(2**31, size=self.world)]
        else:
            seeds = None
        seed = self.comm.scatter(seeds, root=self.root)
        rng_loc = np.random.RandomState(seed)
        rng_w = np.random.RandomState(seed=rng_loc.randint(2**31))

        accumulator = estimator.accumulator_factory().make()
        count = 0.0
        for sz, x in self._enc_chunks:
            w = np.zeros(sz, dtype=float)
            w[rng_w.rand(sz) <= p] = 1.0
            count += np.sum(w)
            accumulator.seq_initialize(x, w, rng_loc)
        return self._fold_and_share(estimator, (count, accumulator.value()))

    def pysp_seq_log_density_sum(self, estimate) -> tuple[float, float]:
        """Allreduced (count, log-density sum) - identical on every rank."""
        cnt, ll = 0.0, 0.0
        for sz, x in self._enc_chunks:
            cnt += sz
            ll += estimate.seq_log_density(x).sum()
        return (self.comm.allreduce(cnt, op=MPI.SUM), self.comm.allreduce(ll, op=MPI.SUM))

    def pysp_stream_accumulate(self, estimator, model) -> tuple[float, Any]:
        """Globally folded batch sufficient statistics for streaming EM."""
        model = pickle.loads(self.comm.bcast(pickle.dumps(model, protocol=_PROTO), root=self.root))
        return self._fold_value_and_share(estimator, self._local_update(estimator, model))

    def __len__(self) -> int:
        return int(self.size)
