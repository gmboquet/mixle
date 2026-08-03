"""Resilient multiprocessing EM backend: retry + rank blacklisting + mid-fit checkpointing (K4).

Status audit this module leans on (see the roadmap): fits are deterministic given seed
(#115) and sufficient statistics are ADDITIVE (``combine()`` folds any partition of the
data, in any grouping, into the same total). Those two facts together make worker-failure
recovery *exact*, not approximate:

    1. **Checkpointing is trivial and exact.** An accumulator's ``value()`` payload IS the
       sufficient statistic, not opaque optimizer state, so serializing it mid-fold and
       restoring it later via ``from_value()`` reconstructs the identical accumulator.
       See :func:`checkpointed_fold`.
    2. **Only the failed shard needs to be redone.** If a worker dies mid-E-step, its
       surviving peers' already-computed ``(count, accumulator.value())`` payloads are
       trusted as-is; only the dead worker's shard is recomputed -- on a surviving worker,
       from the SAME raw shard bytes the driver still holds (the "elastic re-partition").
    3. **Recovery is bit-identical, not just close.** ``seq_update`` (the E-step) is a pure,
       deterministic function of (encoded data, weights, model) -- no RNG is involved -- so
       recomputing a shard on a different physical worker produces byte-identical floats to
       the original owner computing it. The one place determinism could quietly break is
       fold ORDER: floating-point summation is not associative, so this module always folds
       per-shard payloads back together in canonical shard-id order (matching what a
       failure-free run would have done), never in "whichever worker replied first" order.
    4. **Retry + rank blacklisting.** A worker that dies is retried by respawning a fresh
       process for the same rank and re-registering its shard (a transient hiccup does not
       cost that rank its place). A rank that fails repeatedly (``failures >= max_retries``)
       is blacklisted for the rest of the fit: it is never respawned again and its shard is
       migrated permanently onto a surviving worker.

This is the EM-side sibling of F2 (see the roadmap's checkpoint/resume line for the
model-parallel path); this module is the mp-backend line for ordinary (non-model-parallel)
distributed EM.
"""

from __future__ import annotations

import math
import multiprocessing as mp
import os
import pickle
import time
from collections.abc import Callable, Sequence
from numbers import Integral, Real
from typing import Any

import numpy as np

from mixle.utils.parallel.planner import EncodedDataHandle
from mixle.utils.vector import validate_initialization_probability, validated_initialized_observations

__all__ = ["ResilientMPEncodedData", "checkpointed_fold"]

_PROTO = pickle.HIGHEST_PROTOCOL

# Exceptions that mean "the worker on the other end of this pipe is gone."
_WORKER_DEAD_ERRORS = (EOFError, OSError, BrokenPipeError, ConnectionResetError)


def checkpointed_fold(
    estimator: Any, payloads: Sequence[bytes], checkpoint_after: int | None = None
) -> tuple[float, Any]:
    """Fold pickled ``(count, accumulator.value())`` payloads into one sufficient statistic.

    This is the additive fold every backend in this repo performs (see
    ``MPEncodedData._fold_stats`` / ``MPIEncodedData._fold_and_share``), pulled out standalone
    so a checkpoint can be taken mid-fold: pass ``checkpoint_after=k`` to, immediately after
    combining payload index ``k``, serialize the running accumulator via ``value()``, DISCARD
    the in-memory accumulator object entirely, and rebuild a fresh one from that serialized
    value via ``from_value()`` before continuing. Because ``value()``/``from_value()`` is an
    exact round-trip of the accumulator's own state (not lossy optimizer state), the returned
    ``(nobs, value)`` is identical whether or not a checkpoint was taken partway through --
    that identity is the mid-fit-checkpointing acceptance criterion for K4.
    """
    accumulator = estimator.accumulator_factory().make()
    nobs = 0.0
    for i, raw in enumerate(payloads):
        count, stats = pickle.loads(raw)  # nosec B301 # IPC: a (count, stats) payload one of this run's own workers pickled and returned over its pipe
        nobs += count
        accumulator.combine(stats)
        if checkpoint_after is not None and i == checkpoint_after:
            checkpoint_bytes = pickle.dumps((nobs, accumulator.value()), protocol=_PROTO)
            del accumulator  # simulate a real crash/restart: no reference to the live object survives
            nobs, restored_value = pickle.loads(checkpoint_bytes)  # nosec B301 # round-trip of the bytes pickled two lines above in this same function -- the simulated crash/restart deliberately rebuilds the accumulator from its own serialized value
            accumulator = estimator.accumulator_factory().make()
            accumulator.from_value(restored_value)
    stats_dict: dict[str, Any] = {}
    accumulator.key_merge(stats_dict)
    accumulator.key_replace(stats_dict)
    return nobs, accumulator.value()


def _canonical_shard_payloads(groups: Sequence[tuple[int, bytes]], expected_shards: set[int]) -> list[bytes]:
    """Validate exact logical-shard coverage and return payloads in canonical shard order."""
    by_shard: dict[int, bytes] = {}
    for shard_id, payload in groups:
        if isinstance(shard_id, bool) or not isinstance(shard_id, (int, np.integer)):
            raise RuntimeError(f"worker returned invalid shard identity {shard_id!r}")
        shard_id = int(shard_id)
        if shard_id not in expected_shards:
            raise RuntimeError(f"worker returned unknown shard {shard_id}")
        if shard_id in by_shard:
            raise RuntimeError(f"worker returned duplicate payload for shard {shard_id}")
        if not isinstance(payload, bytes):
            raise RuntimeError(f"worker returned a non-bytes payload for shard {shard_id}")
        by_shard[shard_id] = payload
    missing = expected_shards.difference(by_shard)
    if missing:
        raise RuntimeError(f"workers returned no payload for shards {sorted(missing)}")
    return [by_shard[shard_id] for shard_id in sorted(expected_shards)]


def _encode_shard(encoder: Any, shard_b: bytes, sub_chunks: int) -> list[tuple[int, Any]]:
    shard = pickle.loads(shard_b)  # nosec B301 # IPC: the raw shard the driver pickled into an add_shard/update_shard command on this worker's pipe
    n = len(shard)
    k = max(1, min(int(sub_chunks), n)) if n else 1
    chunks: list[tuple[int, Any]] = []
    for i in range(k):
        part = [shard[j] for j in range(i, n, k)]
        if part:
            chunks.append((len(part), encoder.seq_encode(part)))
    return chunks


def _worker_main(conn) -> None:
    """Resilient worker loop.

    Holds a dict of ``shard_id -> encoded chunks`` (its resident set, which can grow via
    ``add_shard`` when the driver migrates a dead rank's shard onto it) plus the encoder
    (loaded once via ``load_encoder``, independent of any particular shard, so this process
    can also encode an unfamiliar shard on demand for one-off recovery via ``update_shard``).

    ``update``/``update_shard`` send a ``"started"`` acknowledgement and then BLOCK waiting for
    an explicit ``"go"`` from the driver before doing any actual accumulation work. This
    handshake -- not just the ack -- is what makes chaos injection deterministic: a "started"
    send alone does not stop the worker from racing ahead and finishing its (possibly tiny)
    shard before a driver-side kill signal is even delivered. Blocking on "go" pins the worker
    at a known rendezvous point until the driver either kills it or waves it through, so a
    test's kill is guaranteed to land before any accumulation begins.
    """
    encoder: Any = None
    resident: dict[int, list[tuple[int, Any]]] = {}

    def _fold_resident(estimator: Any, model: Any) -> list[tuple[int, bytes]]:
        payloads: list[tuple[int, bytes]] = []
        for sid in sorted(resident):
            accumulator = estimator.accumulator_factory().make()
            count = 0.0
            for sz, x in resident[sid]:
                count += sz
                accumulator.seq_update(x, np.ones(sz), model)
            payloads.append((sid, pickle.dumps((count, accumulator.value()), protocol=_PROTO)))
        return payloads

    def _initialize_shard(estimator: Any, chunks: list[tuple[int, Any]], p: float, seed: int) -> bytes:
        accumulator = estimator.accumulator_factory().make()
        rng_loc = np.random.RandomState(seed)
        rng_w = np.random.RandomState(seed=rng_loc.randint(2**31))
        count = 0.0
        for sz, x in chunks:
            weights = np.zeros(sz, dtype=float)
            weights[rng_w.rand(sz) <= p] = 1.0
            count += np.sum(weights)
            accumulator.seq_initialize(x, weights, rng_loc)
        return pickle.dumps((count, accumulator.value()), protocol=_PROTO)

    def _score_shard(model: Any, chunks: list[tuple[int, Any]]) -> tuple[float, float]:
        count, log_density = 0.0, 0.0
        for size, encoded in chunks:
            count += size
            log_density += model.seq_log_density(encoded).sum()
        return count, log_density

    while True:
        msg = conn.recv()
        cmd = msg[0]
        try:
            if cmd == "load_encoder":
                _, encoder_b = msg
                encoder = pickle.loads(encoder_b)  # nosec B301 # IPC: a field of the command tuple this worker just took off its own mp.Pipe, which Connection.recv already unpickled; the only writer is the parent that spawned it
                conn.send(("ok", None))

            elif cmd == "add_shard":
                _, shard_id, shard_b, sub_chunks = msg
                resident[shard_id] = _encode_shard(encoder, shard_b, sub_chunks)
                conn.send(("ok", sum(sz for sz, _ in resident[shard_id])))

            elif cmd == "remove_shard":
                _, shard_id = msg
                resident.pop(shard_id, None)
                conn.send(("ok", shard_id))

            elif cmd == "update":
                _, estimator_b, model_b = msg
                conn.send(("started", os.getpid()))
                conn.recv()  # block for the driver's "go" -- the deterministic kill rendezvous
                estimator = pickle.loads(estimator_b)  # nosec B301 # IPC: a field of the command tuple this worker just took off its own mp.Pipe, which Connection.recv already unpickled; the only writer is the parent that spawned it
                model = pickle.loads(model_b)  # nosec B301 # IPC: a field of the command tuple this worker just took off its own mp.Pipe, which Connection.recv already unpickled; the only writer is the parent that spawned it
                conn.send(("ok", _fold_resident(estimator, model)))

            elif cmd == "update_shard":
                _, estimator_b, model_b, shard_id, shard_b, sub_chunks = msg
                conn.send(("started", os.getpid()))
                conn.recv()  # "go" rendezvous (see "update")
                estimator = pickle.loads(estimator_b)  # nosec B301 # IPC: a field of the command tuple this worker just took off its own mp.Pipe, which Connection.recv already unpickled; the only writer is the parent that spawned it
                model = pickle.loads(model_b)  # nosec B301 # IPC: a field of the command tuple this worker just took off its own mp.Pipe, which Connection.recv already unpickled; the only writer is the parent that spawned it
                chunks = _encode_shard(encoder, shard_b, sub_chunks)
                accumulator = estimator.accumulator_factory().make()
                count = 0.0
                for sz, x in chunks:
                    count += sz
                    accumulator.seq_update(x, np.ones(sz), model)
                conn.send(("ok", pickle.dumps((count, accumulator.value()), protocol=_PROTO)))

            elif cmd == "init":
                _, estimator_b, p, seeds_by_shard = msg
                estimator = pickle.loads(estimator_b)  # nosec B301 # IPC: a field of the command tuple this worker just took off its own mp.Pipe, which Connection.recv already unpickled; the only writer is the parent that spawned it
                payloads = [
                    (sid, _initialize_shard(estimator, resident[sid], p, int(seeds_by_shard[sid])))
                    for sid in sorted(resident)
                ]
                conn.send(("ok", payloads))

            elif cmd == "init_shard":
                _, estimator_b, p, seed, shard_id, shard_b, sub_chunks = msg
                estimator = pickle.loads(estimator_b)  # nosec B301 # IPC: a field of the command tuple this worker just took off its own mp.Pipe, which Connection.recv already unpickled; the only writer is the parent that spawned it
                chunks = _encode_shard(encoder, shard_b, sub_chunks)
                conn.send(("ok", (shard_id, _initialize_shard(estimator, chunks, p, int(seed)))))

            elif cmd == "llsum":
                _, model_b = msg
                model = pickle.loads(model_b)  # nosec B301 # IPC: a field of the command tuple this worker just took off its own mp.Pipe, which Connection.recv already unpickled; the only writer is the parent that spawned it
                conn.send(("ok", [(sid, *_score_shard(model, resident[sid])) for sid in sorted(resident)]))

            elif cmd == "llsum_shard":
                _, model_b, shard_id, shard_b, sub_chunks = msg
                model = pickle.loads(model_b)  # nosec B301 # IPC: a field of the command tuple this worker just took off its own mp.Pipe, which Connection.recv already unpickled; the only writer is the parent that spawned it
                chunks = _encode_shard(encoder, shard_b, sub_chunks)
                conn.send(("ok", (shard_id, *_score_shard(model, chunks))))

            elif cmd == "stop":
                conn.send(("ok", None))
                return

            else:
                conn.send(("err", "unknown command %r" % (cmd,)))

        except BaseException as e:  # surface worker failures on the driver  # noqa: BLE001
            import traceback

            try:
                conn.send(("err", "%s\n%s" % (e, traceback.format_exc())))
            except (BrokenPipeError, OSError):
                return


class ResilientMPEncodedData(EncodedDataHandle):
    """``MPEncodedData`` with retry, rank blacklisting, and exact chaos-tolerant recovery.

    Drop-in for the ``enc_data`` argument of ``optimize``/``best_of``/``seq_estimate``/
    ``seq_initialize``/``seq_log_density_sum``, exactly like :class:`MPEncodedData
    <mixle.utils.parallel.multiprocessing.MPEncodedData>`. Data is split round-robin into
    ``num_workers`` SHARDS (a fixed id space, ``0..num_workers-1``, that outlives any one
    worker process); each shard is initially resident on the worker of the same id, but the
    driver also keeps the shard's raw (pre-encode) bytes so a shard can be recomputed
    elsewhere, or migrated permanently, if its worker dies.

    Args:
        data (Sequence): Raw observations. Must be an in-memory sequence.
        estimator (Optional[ParameterEstimator]): Used to build the encoder when ``encoder``
            is not given.
        encoder (Optional[DataSequenceEncoder]): Explicit encoder; overrides ``estimator``.
        num_workers (Optional[int]): Worker process count (default: CPU count, capped at the
            number of observations).
        sub_chunks (int): Encoded sub-chunks per shard (bounds peak memory of the vectorized
            update inside each worker); also carried along to ad hoc shard recovery so a
            recomputed shard's encode/accumulate split -- and therefore its floating-point
            summation order -- matches what the shard's original owner would have done.
        max_retries (int): A rank is blacklisted once its cumulative failure count reaches
            this threshold; below it, a dead rank is respawned and keeps its place.
        worker_timeout_s (float): Maximum time to wait for one distributed-operation phase
            before a nonresponsive worker is retired and its logical shards are recovered.

    Testing hook:
        :meth:`arm_kill` registers a one-shot callback invoked, for every worker, right after
        that worker acknowledges it has started an ``update`` command and while it is still
        blocked waiting for the driver's "go" -- strictly before any accumulation happens --
        the deterministic rendezvous a chaos test uses to kill a real OS process mid-E-step
        with no timing race.
    """

    def __init__(
        self,
        data: Sequence[Any],
        estimator: Any | None = None,
        encoder: Any | None = None,
        num_workers: int | None = None,
        sub_chunks: int = 1,
        max_retries: int = 2,
        worker_timeout_s: float = 30.0,
    ) -> None:
        if encoder is None:
            if estimator is None:
                raise ValueError("ResilientMPEncodedData requires an estimator or an explicit encoder.")
            encoder = estimator.accumulator_factory().make().acc_to_encoder()

        n = len(data)
        if n == 0:
            raise ValueError("ResilientMPEncodedData requires non-empty data.")
        if num_workers is None:
            num_workers = mp.cpu_count()
        if isinstance(num_workers, bool) or not isinstance(num_workers, Integral):
            raise TypeError("num_workers must be an integer")
        num_workers = max(1, min(int(num_workers), n))
        if isinstance(sub_chunks, bool) or not isinstance(sub_chunks, Integral) or sub_chunks < 1:
            raise ValueError("sub_chunks must be a positive integer")
        if isinstance(max_retries, bool) or not isinstance(max_retries, Integral) or max_retries < 1:
            raise ValueError("max_retries must be at least 1.")
        if (
            isinstance(worker_timeout_s, bool)
            or not isinstance(worker_timeout_s, Real)
            or not math.isfinite(worker_timeout_s)
            or worker_timeout_s <= 0
        ):
            raise ValueError("worker_timeout_s must be finite and positive")

        self.num_workers = num_workers
        self.sub_chunks = int(sub_chunks)
        self.max_retries = int(max_retries)
        self.worker_timeout_s = float(worker_timeout_s)
        self._ctx = mp.get_context("spawn")
        self._encoder_b = pickle.dumps(encoder, protocol=_PROTO)

        # The final driver-side fold of per-shard payloads (see _resilient_update_round below)
        # is always exactly checkpointed_fold -- UNLESS a subclass swaps this hook out. K5's
        # AuditedMPEncodedData replaces it with a fold that wraps every combine() call with a
        # NaN/Inf watchdog; this class's own retry/blacklist/elastic-repartition logic (the
        # thing K5 reuses rather than reimplements) is otherwise untouched.
        self._fold_fn: Callable[[Any, Sequence[bytes]], tuple[float, Any]] = checkpointed_fold

        self._shard_raw: dict[int, bytes] = {}
        self._worker_shards: dict[int, set[int]] = {}
        self._conns: dict[int, Any] = {}
        self._procs: dict[int, Any] = {}
        self._failures: dict[int, int] = {i: 0 for i in range(num_workers)}
        self._blacklist: set[int] = set()
        self._kill_hook: Callable[[int, Any], None] | None = None

        # instrumentation from the most recent pysp_seq_estimate/pysp_stream_accumulate round,
        # exposed for tests (and operators) to verify recovery only redid what it had to.
        self.last_round_reused_shards: set[int] = set()
        self.last_round_recomputed_shards: set[int] = set()
        self.last_round_failed_workers: set[int] = set()
        self.last_round_blacklisted_workers: set[int] = set()

        self.size = 0
        try:
            for i in range(num_workers):
                shard = [data[j] for j in range(i, n, num_workers)]
                self._shard_raw[i] = pickle.dumps(shard, protocol=_PROTO)
                self.size += len(shard)
            for i in range(num_workers):
                self._spawn_worker(i, [i])
        except BaseException:
            self.close()
            raise

    # -- worker process lifecycle --------------------------------------------

    def _send_raw(self, worker_id: int, msg: tuple) -> None:
        self._conns[worker_id].send(msg)

    def _deadline(self) -> float:
        return time.monotonic() + self.worker_timeout_s

    def _recv_raw(self, worker_id: int, *, deadline: float | None = None) -> tuple[str, Any]:
        deadline = self._deadline() if deadline is None else deadline
        remaining = max(0.0, deadline - time.monotonic())
        conn = self._conns[worker_id]
        if not conn.poll(remaining):
            proc = self._procs.get(worker_id)
            if proc is not None and not proc.is_alive():
                raise EOFError(f"worker {worker_id} exited without replying")
            raise TimeoutError(f"worker {worker_id} did not reply within {self.worker_timeout_s:g} seconds")
        status, payload = conn.recv()
        if status == "err":
            raise RuntimeError("worker %d failed:\n%s" % (worker_id, payload))
        if status not in {"ok", "started"}:
            raise RuntimeError(f"worker {worker_id} returned invalid status {status!r}")
        return status, payload

    def _spawn_worker(self, worker_id: int, initial_shards: list[int]) -> None:
        parent, child = self._ctx.Pipe()
        proc = self._ctx.Process(target=_worker_main, args=(child,), daemon=True)
        proc.start()
        child.close()
        self._conns[worker_id] = parent
        self._procs[worker_id] = proc
        try:
            self._send_raw(worker_id, ("load_encoder", self._encoder_b))
            self._recv_raw(worker_id)
            for shard_id in initial_shards:
                self._send_raw(worker_id, ("add_shard", shard_id, self._shard_raw[shard_id], self.sub_chunks))
                self._recv_raw(worker_id)
        except BaseException:
            self._retire_worker(worker_id)
            raise
        self._worker_shards[worker_id] = set(initial_shards)

    def _retire_worker(self, worker_id: int) -> None:
        conn = self._conns.pop(worker_id, None)
        proc = self._procs.pop(worker_id, None)
        if conn is not None:
            try:
                conn.close()
            except OSError:
                pass
        if proc is not None:
            proc.join(timeout=min(5.0, self.worker_timeout_s))
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=min(5.0, self.worker_timeout_s))

    def _respawn_worker(self, worker_id: int, shard_ids: set[int]) -> None:
        """Retry: bring rank ``worker_id`` back with the SAME shard assignment."""
        self._spawn_worker(worker_id, sorted(shard_ids))

    def _migrate_shard_permanently(self, shard_id: int) -> None:
        """Elastic re-partition: move a shard onto a surviving worker for good."""
        candidates = sorted(w for w in self._conns if w not in self._blacklist)
        if not candidates:
            raise RuntimeError("no surviving worker available to migrate shard %d onto." % shard_id)
        target = candidates[0]
        self._send_raw(target, ("add_shard", shard_id, self._shard_raw[shard_id], self.sub_chunks))
        self._recv_raw(target)
        self._worker_shards.setdefault(target, set()).add(shard_id)

    def _retire_failed_workers(self, failed: set[int]) -> dict[int, set[int]]:
        assignments: dict[int, set[int]] = {}
        for worker_id in sorted(failed):
            self._failures[worker_id] = self._failures.get(worker_id, 0) + 1
            assignments[worker_id] = self._worker_shards.pop(worker_id, set())
            self._retire_worker(worker_id)
        return assignments

    def _restore_failed_workers(self, assignments: dict[int, set[int]]) -> set[int]:
        blacklisted_now: set[int] = set()
        retryable: list[int] = []
        for worker_id in sorted(assignments):
            if self._failures[worker_id] >= self.max_retries:
                self._blacklist.add(worker_id)
                blacklisted_now.add(worker_id)
            else:
                retryable.append(worker_id)
        for worker_id in retryable:
            if worker_id not in self._conns:
                self._respawn_worker(worker_id, assignments[worker_id])
        for worker_id in sorted(blacklisted_now):
            for shard_id in assignments[worker_id]:
                self._migrate_shard_permanently(shard_id)
        return blacklisted_now

    def _request_recovery(
        self,
        message: tuple,
        shard_id: int,
        failed_assignments: dict[int, set[int]],
        *,
        rendezvous: bool = False,
    ) -> Any:
        """Execute one shard-local recovery request, retiring any target that fails."""
        attempted: set[int] = set()
        while True:
            candidates = sorted(
                worker_id
                for worker_id in self._conns
                if worker_id not in self._blacklist and worker_id not in attempted
            )
            if not candidates:
                retryable = sorted(
                    worker_id
                    for worker_id in failed_assignments
                    if worker_id not in attempted
                    and worker_id not in self._blacklist
                    and self._failures[worker_id] < self.max_retries
                )
                if not retryable:
                    raise RuntimeError("no surviving worker available to recover shard %d." % shard_id)
                worker_id = retryable[0]
                self._respawn_worker(worker_id, failed_assignments[worker_id])
                continue
            target = candidates[0]
            attempted.add(target)
            try:
                self._send_raw(target, message)
                if rendezvous:
                    self._recv_raw(target)
                    self._send_raw(target, ("go",))
                _, payload = self._recv_raw(target)
                return payload
            except _WORKER_DEAD_ERRORS + (RuntimeError, TimeoutError):
                failed_assignments.update(self._retire_failed_workers({target}))

    def _recover_shard(
        self,
        estimator_b: bytes,
        model_b: bytes,
        shard_id: int,
        failed_assignments: dict[int, set[int]],
    ) -> bytes:
        """Recompute one shard's E-step contribution on a surviving worker, ad hoc."""
        return self._request_recovery(
            ("update_shard", estimator_b, model_b, shard_id, self._shard_raw[shard_id], self.sub_chunks),
            shard_id,
            failed_assignments,
            rendezvous=True,
        )

    # -- resilient E-step / streaming-accumulate round -----------------------

    def arm_kill(self, hook: Callable[[int, Any], None]) -> None:
        """Register a one-shot hook fired for each worker right after its "started" ack, while
        that worker is still blocked waiting for the driver's "go" (see ``_worker_main``).

        ``hook(worker_id, proc)`` may kill ``proc`` (e.g. ``proc.kill(); proc.join()``) to
        simulate a real worker death mid-E-step, with no timing race: the worker cannot have
        started accumulating yet. It is consumed (cleared) the moment the next
        ``pysp_seq_estimate``/``pysp_stream_accumulate`` call begins, so it fires for exactly
        one round.
        """
        self._kill_hook = hook

    def _resilient_update_round(
        self, estimator: Any, model: Any, kill_hook: Callable[[int, Any], None] | None
    ) -> tuple[float, Any]:
        estimator_b = pickle.dumps(estimator, protocol=_PROTO)
        model_b = pickle.dumps(model, protocol=_PROTO)

        live_workers = sorted(w for w in self._conns if w not in self._blacklist)
        if not live_workers:
            raise RuntimeError("ResilientMPEncodedData has no live workers left.")

        failed: set[int] = set()
        for w in live_workers:
            try:
                self._send_raw(w, ("update", estimator_b, model_b))
            except _WORKER_DEAD_ERRORS:
                failed.add(w)

        # Phase 1: wait for every worker to ack it has started, then let the kill hook look at
        # it. Each worker BLOCKS after its ack waiting for an explicit "go" (see _worker_main),
        # so this is a real rendezvous, not a race: a kill issued here is guaranteed to land
        # before that worker does any accumulation.
        started: list[int] = []
        ack_deadline = self._deadline()
        for w in live_workers:
            if w in failed:
                continue
            try:
                self._recv_raw(w, deadline=ack_deadline)
            except _WORKER_DEAD_ERRORS + (RuntimeError, TimeoutError):
                failed.add(w)
                continue
            started.append(w)

        for w in started:
            if kill_hook is not None:
                kill_hook(w, self._procs.get(w))
            proc = self._procs.get(w)
            if proc is not None and not proc.is_alive():
                failed.add(w)
                continue
            try:
                self._send_raw(w, ("go",))
            except _WORKER_DEAD_ERRORS:
                failed.add(w)

        # Phase 2: collect final results from whichever workers are still alive.
        worker_payload: dict[int, list[tuple[int, bytes]]] = {}
        result_deadline = self._deadline()
        for w in started:
            if w in failed:
                continue
            try:
                _, payload = self._recv_raw(w, deadline=result_deadline)
            except _WORKER_DEAD_ERRORS + (RuntimeError, TimeoutError):
                failed.add(w)
                continue
            worker_payload[w] = payload

        groups = [shard_payload for payloads in worker_payload.values() for shard_payload in payloads]
        reused_shards = {shard_id for shard_id, _ in groups}

        recomputed_shards: set[int] = set()
        # Retire the complete failed set before choosing any recovery target.
        failed_assignments = self._retire_failed_workers(failed)
        recovery_shards = sorted(shard_id for shard_ids in failed_assignments.values() for shard_id in shard_ids)

        for shard_id in recovery_shards:
            payload = self._recover_shard(estimator_b, model_b, shard_id, failed_assignments)
            groups.append((shard_id, payload))
            recomputed_shards.add(shard_id)

        blacklisted_now = self._restore_failed_workers(failed_assignments)
        failed = set(failed_assignments)

        payloads = _canonical_shard_payloads(groups, set(self._shard_raw))

        self.last_round_reused_shards = reused_shards
        self.last_round_recomputed_shards = recomputed_shards
        self.last_round_failed_workers = set(failed)
        self.last_round_blacklisted_workers = blacklisted_now

        return self._fold_fn(estimator, payloads)

    # -- protocol recognized by mixle.stats dispatch -------------------------

    def pysp_seq_estimate(self, estimator: Any, prev_estimate: Any) -> Any:
        """One distributed EM step, tolerant of a worker dying mid-accumulation."""
        kill_hook, self._kill_hook = self._kill_hook, None
        nobs, value = self._resilient_update_round(estimator, prev_estimate, kill_hook)
        return estimator.estimate(nobs, value)

    def pysp_seq_initialize(self, estimator: Any, rng: np.random.RandomState, p: float) -> Any:
        """Distributed randomized initialization; seeds are anchored to shard id, not worker
        identity, so a shard reassigned to a different worker still uses its own fixed seed."""
        p = validate_initialization_probability(p)
        estimator_b = pickle.dumps(estimator, protocol=_PROTO)
        seeds = rng.randint(2**31, size=self.num_workers)
        seeds_by_shard = {sid: int(seeds[sid]) for sid in range(self.num_workers)}
        live_workers = sorted(w for w in self._conns if w not in self._blacklist)
        failed: set[int] = set()
        for w in live_workers:
            my_seeds = {sid: seeds_by_shard[sid] for sid in self._worker_shards.get(w, set())}
            try:
                self._send_raw(w, ("init", estimator_b, float(p), my_seeds))
            except _WORKER_DEAD_ERRORS:
                failed.add(w)
        groups: list[tuple[int, bytes]] = []
        deadline = self._deadline()
        for worker_id in live_workers:
            if worker_id in failed:
                continue
            try:
                _, worker_groups = self._recv_raw(worker_id, deadline=deadline)
                groups.extend(worker_groups)
            except _WORKER_DEAD_ERRORS + (RuntimeError, TimeoutError):
                failed.add(worker_id)
        failed_assignments = self._retire_failed_workers(failed)
        recovery_shards = sorted(shard_id for shard_ids in failed_assignments.values() for shard_id in shard_ids)
        for shard_id in recovery_shards:
            payload = self._request_recovery(
                (
                    "init_shard",
                    estimator_b,
                    float(p),
                    seeds_by_shard[shard_id],
                    shard_id,
                    self._shard_raw[shard_id],
                    self.sub_chunks,
                ),
                shard_id,
                failed_assignments,
            )
            recovered_shard, recovered_payload = payload
            if recovered_shard != shard_id:
                raise RuntimeError(f"worker recovered shard {recovered_shard!r} instead of {shard_id}")
            groups.append((shard_id, recovered_payload))
        self._restore_failed_workers(failed_assignments)
        payloads = _canonical_shard_payloads(groups, set(self._shard_raw))
        nobs, value = checkpointed_fold(estimator, payloads)
        return estimator.estimate(validated_initialized_observations(nobs), value)

    def pysp_seq_log_density_sum(self, estimate: Any) -> tuple[float, float]:
        """Total observation count and summed log density across all live workers."""
        model_b = pickle.dumps(estimate, protocol=_PROTO)
        live_workers = sorted(w for w in self._conns if w not in self._blacklist)
        failed: set[int] = set()
        for w in live_workers:
            try:
                self._send_raw(w, ("llsum", model_b))
            except _WORKER_DEAD_ERRORS:
                failed.add(w)
        shard_scores: list[tuple[int, float, float]] = []
        deadline = self._deadline()
        for worker_id in live_workers:
            if worker_id in failed:
                continue
            try:
                _, scores = self._recv_raw(worker_id, deadline=deadline)
                shard_scores.extend(scores)
            except _WORKER_DEAD_ERRORS + (RuntimeError, TimeoutError):
                failed.add(worker_id)
        failed_assignments = self._retire_failed_workers(failed)
        recovery_shards = sorted(shard_id for shard_ids in failed_assignments.values() for shard_id in shard_ids)
        for shard_id in recovery_shards:
            recovered = self._request_recovery(
                ("llsum_shard", model_b, shard_id, self._shard_raw[shard_id], self.sub_chunks),
                shard_id,
                failed_assignments,
            )
            if recovered[0] != shard_id:
                raise RuntimeError(f"worker recovered shard {recovered[0]!r} instead of {shard_id}")
            shard_scores.append(recovered)
        self._restore_failed_workers(failed_assignments)

        by_shard: dict[int, tuple[float, float]] = {}
        for shard_id, count, score in shard_scores:
            if shard_id not in self._shard_raw or shard_id in by_shard:
                raise RuntimeError(f"invalid or duplicate log-density result for shard {shard_id!r}")
            by_shard[shard_id] = (count, score)
        missing = set(self._shard_raw).difference(by_shard)
        if missing:
            raise RuntimeError(f"workers returned no log-density result for shards {sorted(missing)}")
        count, score = 0.0, 0.0
        for shard_id in sorted(by_shard):
            shard_count, shard_score = by_shard[shard_id]
            count += shard_count
            score += shard_score
        return count, score

    def pysp_stream_accumulate(self, estimator: Any, model: Any) -> tuple[float, Any]:
        """Return globally folded batch sufficient statistics for streaming EM, chaos-tolerant."""
        kill_hook, self._kill_hook = self._kill_hook, None
        return self._resilient_update_round(estimator, model, kill_hook)

    # -- lifecycle -------------------------------------------------------------

    def close(self) -> None:
        """Shut the worker pool down. Idempotent."""
        for w in list(self._conns):
            try:
                self._send_raw(w, ("stop",))
            except (BrokenPipeError, OSError):
                pass
        deadline = self._deadline()
        for w in list(self._conns):
            try:
                self._recv_raw(w, deadline=deadline)
            except (EOFError, OSError, RuntimeError, TimeoutError):
                pass
            conn = self._conns.pop(w, None)
            if conn is not None:
                conn.close()
        for w in list(self._procs):
            proc = self._procs.pop(w, None)
            if proc is not None:
                proc.join(timeout=min(5.0, self.worker_timeout_s))
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=min(5.0, self.worker_timeout_s))

    def __len__(self) -> int:
        return int(self.size)

    def __enter__(self) -> ResilientMPEncodedData:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self):
        try:
            if self._conns:
                self.close()
        except Exception:  # noqa: BLE001
            pass
