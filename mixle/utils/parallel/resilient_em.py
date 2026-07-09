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

import multiprocessing as mp
import os
import pickle
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from mixle.utils.parallel.planner import EncodedDataHandle

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
        count, stats = pickle.loads(raw)
        nobs += count
        accumulator.combine(stats)
        if checkpoint_after is not None and i == checkpoint_after:
            checkpoint_bytes = pickle.dumps((nobs, accumulator.value()), protocol=_PROTO)
            del accumulator  # simulate a real crash/restart: no reference to the live object survives
            nobs, restored_value = pickle.loads(checkpoint_bytes)
            accumulator = estimator.accumulator_factory().make()
            accumulator.from_value(restored_value)
    stats_dict: dict[str, Any] = {}
    accumulator.key_merge(stats_dict)
    accumulator.key_replace(stats_dict)
    return nobs, accumulator.value()


def _encode_shard(encoder: Any, shard_b: bytes, sub_chunks: int) -> list[tuple[int, Any]]:
    shard = pickle.loads(shard_b)
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

    def _fold_resident(estimator: Any, model: Any) -> tuple[float, Any]:
        accumulator = estimator.accumulator_factory().make()
        count = 0.0
        for sid in sorted(resident):
            for sz, x in resident[sid]:
                count += sz
                accumulator.seq_update(x, np.ones(sz), model)
        return count, accumulator.value()

    while True:
        msg = conn.recv()
        cmd = msg[0]
        try:
            if cmd == "load_encoder":
                _, encoder_b = msg
                encoder = pickle.loads(encoder_b)
                conn.send(("ok", None))

            elif cmd == "add_shard":
                _, shard_id, shard_b, sub_chunks = msg
                resident[shard_id] = _encode_shard(encoder, shard_b, sub_chunks)
                conn.send(("ok", sum(sz for sz, _ in resident[shard_id])))

            elif cmd == "update":
                _, estimator_b, model_b = msg
                conn.send(("started", os.getpid()))
                conn.recv()  # block for the driver's "go" -- the deterministic kill rendezvous
                estimator = pickle.loads(estimator_b)
                model = pickle.loads(model_b)
                count, value = _fold_resident(estimator, model)
                conn.send(("ok", pickle.dumps((count, value), protocol=_PROTO)))

            elif cmd == "update_shard":
                _, estimator_b, model_b, shard_id, shard_b, sub_chunks = msg
                conn.send(("started", os.getpid()))
                conn.recv()  # "go" rendezvous (see "update")
                estimator = pickle.loads(estimator_b)
                model = pickle.loads(model_b)
                chunks = _encode_shard(encoder, shard_b, sub_chunks)
                accumulator = estimator.accumulator_factory().make()
                count = 0.0
                for sz, x in chunks:
                    count += sz
                    accumulator.seq_update(x, np.ones(sz), model)
                conn.send(("ok", pickle.dumps((count, accumulator.value()), protocol=_PROTO)))

            elif cmd == "init":
                _, estimator_b, p, seeds_by_shard = msg
                estimator = pickle.loads(estimator_b)
                accumulator = estimator.accumulator_factory().make()
                count = 0.0
                for sid in sorted(resident):
                    rng_loc = np.random.RandomState(int(seeds_by_shard[sid]))
                    rng_w = np.random.RandomState(seed=rng_loc.randint(2**31))
                    for sz, x in resident[sid]:
                        w = np.zeros(sz, dtype=float)
                        w[rng_w.rand(sz) <= p] = 1.0
                        count += np.sum(w)
                        accumulator.seq_initialize(x, w, rng_loc)
                conn.send(("ok", pickle.dumps((count, accumulator.value()), protocol=_PROTO)))

            elif cmd == "llsum":
                _, model_b = msg
                model = pickle.loads(model_b)
                cnt, ll = 0.0, 0.0
                for sid in sorted(resident):
                    for sz, x in resident[sid]:
                        cnt += sz
                        ll += model.seq_log_density(x).sum()
                conn.send(("ok", (cnt, ll)))

            elif cmd == "stop":
                conn.send(("ok", None))
                return

            else:
                conn.send(("err", "unknown command %r" % (cmd,)))

        except BaseException as e:  # surface worker failures on the driver
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
        num_workers = max(1, min(int(num_workers), n))
        if max_retries < 1:
            raise ValueError("max_retries must be at least 1.")

        self.num_workers = num_workers
        self.sub_chunks = int(sub_chunks)
        self.max_retries = int(max_retries)
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

    def _recv_raw(self, worker_id: int) -> tuple[str, Any]:
        status, payload = self._conns[worker_id].recv()
        if status == "err":
            raise RuntimeError("worker %d failed:\n%s" % (worker_id, payload))
        return status, payload

    def _spawn_worker(self, worker_id: int, initial_shards: list[int]) -> None:
        parent, child = self._ctx.Pipe()
        proc = self._ctx.Process(target=_worker_main, args=(child,), daemon=True)
        proc.start()
        child.close()
        self._conns[worker_id] = parent
        self._procs[worker_id] = proc
        self._send_raw(worker_id, ("load_encoder", self._encoder_b))
        self._recv_raw(worker_id)
        for shard_id in initial_shards:
            self._send_raw(worker_id, ("add_shard", shard_id, self._shard_raw[shard_id], self.sub_chunks))
            self._recv_raw(worker_id)
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
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)

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

    def _recover_shard(self, estimator_b: bytes, model_b: bytes, shard_id: int) -> bytes:
        """Recompute one shard's E-step contribution on a surviving worker, ad hoc."""
        candidates = sorted(w for w in self._conns if w not in self._blacklist)
        if not candidates:
            raise RuntimeError("no surviving worker available to recover shard %d." % shard_id)
        target = candidates[0]
        self._send_raw(
            target, ("update_shard", estimator_b, model_b, shard_id, self._shard_raw[shard_id], self.sub_chunks)
        )
        self._recv_raw(target)  # "started" ack
        self._send_raw(target, ("go",))
        _, payload = self._recv_raw(target)
        return payload

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

        for w in live_workers:
            self._send_raw(w, ("update", estimator_b, model_b))

        # Phase 1: wait for every worker to ack it has started, then let the kill hook look at
        # it. Each worker BLOCKS after its ack waiting for an explicit "go" (see _worker_main),
        # so this is a real rendezvous, not a race: a kill issued here is guaranteed to land
        # before that worker does any accumulation.
        failed: set[int] = set()
        started: list[int] = []
        for w in live_workers:
            try:
                self._recv_raw(w)
            except _WORKER_DEAD_ERRORS + (RuntimeError,):
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
        worker_payload: dict[int, bytes] = {}
        for w in started:
            if w in failed:
                continue
            try:
                _, payload = self._recv_raw(w)
            except _WORKER_DEAD_ERRORS + (RuntimeError,):
                failed.add(w)
                continue
            worker_payload[w] = payload

        reused_shards: set[int] = set()
        for w in worker_payload:
            reused_shards |= self._worker_shards.get(w, set())

        # (sort_key, payload) pairs; sort_key anchors each group to its lowest shard id so the
        # final fold order matches the canonical 0..num_workers-1 shard order a failure-free
        # run would have used, regardless of which worker (or recovery path) produced it.
        groups: list[tuple[int, bytes]] = [
            (min(self._worker_shards.get(w, {w})), payload) for w, payload in worker_payload.items()
        ]

        recomputed_shards: set[int] = set()
        blacklisted_now: set[int] = set()
        for w in sorted(failed):
            self._failures[w] = self._failures.get(w, 0) + 1
            shard_ids = self._worker_shards.pop(w, set())
            self._retire_worker(w)

            for sid in shard_ids:
                payload = self._recover_shard(estimator_b, model_b, sid)
                groups.append((sid, payload))
                recomputed_shards.add(sid)

            if self._failures[w] >= self.max_retries:
                self._blacklist.add(w)
                blacklisted_now.add(w)
                for sid in shard_ids:
                    self._migrate_shard_permanently(sid)
            else:
                self._respawn_worker(w, shard_ids)

        groups.sort(key=lambda g: g[0])
        payloads = [payload for _, payload in groups]

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
        estimator_b = pickle.dumps(estimator, protocol=_PROTO)
        seeds = rng.randint(2**31, size=self.num_workers)
        seeds_by_shard = {sid: int(seeds[sid]) for sid in range(self.num_workers)}
        live_workers = sorted(w for w in self._conns if w not in self._blacklist)
        for w in live_workers:
            my_seeds = {sid: seeds_by_shard[sid] for sid in self._worker_shards.get(w, set())}
            self._send_raw(w, ("init", estimator_b, float(p), my_seeds))
        payloads = [self._recv_raw(w)[1] for w in live_workers]
        nobs, value = checkpointed_fold(estimator, payloads)
        return estimator.estimate(nobs, value)

    def pysp_seq_log_density_sum(self, estimate: Any) -> tuple[float, float]:
        """Total observation count and summed log density across all live workers."""
        model_b = pickle.dumps(estimate, protocol=_PROTO)
        live_workers = sorted(w for w in self._conns if w not in self._blacklist)
        for w in live_workers:
            self._send_raw(w, ("llsum", model_b))
        cnt, ll = 0.0, 0.0
        for w in live_workers:
            _, (c, s) = self._recv_raw(w)
            cnt += c
            ll += s
        return cnt, ll

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
        for w in list(self._conns):
            conn = self._conns.pop(w, None)
            try:
                if conn is not None:
                    conn.recv()
            except (EOFError, OSError):
                pass
            finally:
                if conn is not None:
                    conn.close()
        for w in list(self._procs):
            proc = self._procs.pop(w, None)
            if proc is not None:
                proc.join(timeout=5)
                if proc.is_alive():
                    proc.terminate()

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
        except Exception:
            pass
