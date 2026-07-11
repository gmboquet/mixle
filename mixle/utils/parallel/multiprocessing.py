"""Multiprocessing backend for distributed estimation.

Mirrors the Spark estimation path with a pool of persistent local worker
processes: the driver builds one encoder and ships each worker a shard of the
raw data once; every worker encodes its own shard and keeps the encoded chunks
resident for the lifetime of the handle. Per EM iteration only the (small)
pickled model crosses the process boundary outward and only the per-worker
``(count, accumulator.value())`` sufficient-statistic payloads return; the
driver folds them with ``combine()``, applies ``key_merge``/``key_replace``
once globally (parameter tying must happen after the full combine), and runs
the M-step.

Usage - the handle plugs into the ordinary estimation entry points::

    from mixle.utils.parallel.multiprocessing import MPEncodedData
    from mixle.inference.estimation import optimize

    with MPEncodedData(data, estimator=est, num_workers=8) as enc:
        model = optimize(None, est, enc_data=enc, max_its=50)

``optimize``/``best_of`` need no changes: ``mixle.inference.seq_estimate``,
``seq_initialize`` and ``mixle.stats.seq_log_density_sum`` recognize the handle and
delegate to it. Validation data can stay locally encoded (a plain
``seq_encode`` result) alongside an ``MPEncodedData`` training handle.

Notes:
    - Worker processes are started with the ``spawn`` method for cross-platform
      consistency; each worker imports ``mixle.stats`` on first use (fast when
      the numba cache is warm).
    - The M-step receives the true observation count (as the Spark path does),
      not ``None`` (as the in-process path does).
    - Runtime kernel/engine objects are not picklable; ship plain
      distributions/estimators (the handle does this for you).
"""

import multiprocessing as mp
import pickle
from collections.abc import Sequence
from typing import Any

import numpy as np

from mixle.utils.parallel.planner import EncodedDataHandle

__all__ = ["MPEncodedData"]

_PROTO = pickle.HIGHEST_PROTOCOL


def _worker_main(conn) -> None:
    """Worker loop: encode the assigned shard once, then serve per-iteration
    accumulate/score commands against the resident encoded chunks."""
    enc_chunks: list[tuple[int, Any]] = []

    while True:
        msg = conn.recv()
        cmd = msg[0]

        try:
            if cmd == "setup":
                _, encoder_b, shard_b, sub_chunks = msg
                encoder = pickle.loads(encoder_b)
                shard = pickle.loads(shard_b)
                n = len(shard)
                k = max(1, min(int(sub_chunks), n)) if n else 1
                enc_chunks = []
                for i in range(k):
                    part = [shard[j] for j in range(i, n, k)]
                    if part:
                        enc_chunks.append((len(part), encoder.seq_encode(part)))
                conn.send(("ok", n))

            elif cmd == "update":
                _, estimator_b, model_b = msg
                estimator = pickle.loads(estimator_b)
                model = pickle.loads(model_b)
                accumulator = estimator.accumulator_factory().make()
                count = 0.0
                for sz, x in enc_chunks:
                    count += sz
                    accumulator.seq_update(x, np.ones(sz), model)
                conn.send(("ok", pickle.dumps((count, accumulator.value()), protocol=_PROTO)))

            elif cmd == "init":
                _, estimator_b, seed, p = msg
                estimator = pickle.loads(estimator_b)
                accumulator = estimator.accumulator_factory().make()
                rng_loc = np.random.RandomState(seed)
                rng_w = np.random.RandomState(seed=rng_loc.randint(2**31))
                count = 0.0
                for sz, x in enc_chunks:
                    w = np.zeros(sz, dtype=float)
                    w[rng_w.rand(sz) <= p] = 1.0
                    count += np.sum(w)
                    accumulator.seq_initialize(x, w, rng_loc)
                conn.send(("ok", pickle.dumps((count, accumulator.value()), protocol=_PROTO)))

            elif cmd == "llsum":
                _, model_b = msg
                model = pickle.loads(model_b)
                cnt, ll = 0.0, 0.0
                for sz, x in enc_chunks:
                    cnt += sz
                    ll += model.seq_log_density(x).sum()
                conn.send(("ok", (cnt, ll)))

            elif cmd == "stop":
                conn.send(("ok", None))
                return

            else:
                conn.send(("err", "unknown command %r" % (cmd,)))

        except BaseException as e:  # surface worker failures on the driver  # noqa: BLE001
            import traceback

            conn.send(("err", "%s\n%s" % (e, traceback.format_exc())))


class MPEncodedData(EncodedDataHandle):
    """Encoded-data handle sharded across persistent local worker processes.

    Drop-in for the ``enc_data`` argument of ``optimize``/``best_of``/
    ``seq_estimate``/``seq_initialize``/``seq_log_density_sum``. The raw data
    is split round-robin into ``num_workers`` shards; each worker encodes its
    shard once and keeps it resident across EM iterations.

    Args:
        data (Sequence): Raw observations (anything the model's encoder
            accepts). Must be an in-memory sequence.
        estimator (Optional[ParameterEstimator]): Used to build the encoder
            when ``encoder`` is not given.
        encoder (Optional[DataSequenceEncoder]): Explicit encoder; overrides
            ``estimator``.
        num_workers (Optional[int]): Worker process count (default: CPU count,
            capped at the number of observations).
        sub_chunks (int): Encoded sub-chunks per worker (bounds peak memory of
            the vectorized update inside each worker).
    """

    def __init__(
        self, data: Sequence[Any], estimator=None, encoder=None, num_workers: int | None = None, sub_chunks: int = 1
    ):
        if encoder is None:
            if estimator is None:
                raise ValueError("MPEncodedData requires an estimator or an explicit encoder.")
            encoder = estimator.accumulator_factory().make().acc_to_encoder()

        n = len(data)
        if n == 0:
            raise ValueError("MPEncodedData requires non-empty data.")
        if num_workers is None:
            num_workers = mp.cpu_count()
        num_workers = max(1, min(int(num_workers), n))

        self.num_workers = num_workers
        self._ctx = mp.get_context("spawn")
        self._conns = []
        self._procs = []

        encoder_b = pickle.dumps(encoder, protocol=_PROTO)
        try:
            for i in range(num_workers):
                parent, child = self._ctx.Pipe()
                proc = self._ctx.Process(target=_worker_main, args=(child,), daemon=True)
                proc.start()
                child.close()
                self._conns.append(parent)
                self._procs.append(proc)

            self.size = 0
            for i, conn in enumerate(self._conns):
                shard = [data[j] for j in range(i, n, num_workers)]
                conn.send(("setup", encoder_b, pickle.dumps(shard, protocol=_PROTO), sub_chunks))
            for conn in self._conns:
                self.size += self._recv(conn)
        except BaseException:
            self.close()
            raise

    # -- driver-side helpers ------------------------------------------------

    @staticmethod
    def _recv(conn):
        status, payload = conn.recv()
        if status != "ok":
            raise RuntimeError("parallel worker failed:\n%s" % payload)
        return payload

    def _broadcast_collect(self, msg) -> list[Any]:
        for conn in self._conns:
            conn.send(msg)
        return [self._recv(conn) for conn in self._conns]

    def _fold_stats(self, estimator, payloads) -> tuple[float, Any]:
        accumulator = estimator.accumulator_factory().make()
        nobs = 0.0
        for raw in payloads:
            count, stats = pickle.loads(raw)
            nobs += count
            accumulator.combine(stats)
        stats_dict = dict()
        accumulator.key_merge(stats_dict)
        accumulator.key_replace(stats_dict)
        return nobs, accumulator.value()

    # -- protocol recognized by mixle.stats dispatch -------------------------

    def pysp_seq_estimate(self, estimator, prev_estimate):
        """One distributed EM step: returns the re-estimated distribution."""
        payloads = self._broadcast_collect(
            (
                "update",
                pickle.dumps(estimator, protocol=_PROTO),
                pickle.dumps(prev_estimate, protocol=_PROTO),
            )
        )
        nobs, value = self._fold_stats(estimator, payloads)
        return estimator.estimate(nobs, value)

    def pysp_seq_initialize(self, estimator, rng: np.random.RandomState, p: float):
        """Distributed randomized initialization (mirrors seq_initialize)."""
        seeds = rng.randint(2**31, size=self.num_workers)
        estimator_b = pickle.dumps(estimator, protocol=_PROTO)
        for conn, seed in zip(self._conns, seeds):
            conn.send(("init", estimator_b, int(seed), float(p)))
        payloads = [self._recv(conn) for conn in self._conns]
        nobs, value = self._fold_stats(estimator, payloads)
        return estimator.estimate(nobs, value)

    def pysp_seq_log_density_sum(self, estimate) -> tuple[float, float]:
        """Total observation count and summed log density across all workers."""
        results = self._broadcast_collect(("llsum", pickle.dumps(estimate, protocol=_PROTO)))
        cnt = sum(r[0] for r in results)
        ll = sum(r[1] for r in results)
        return cnt, ll

    def pysp_stream_accumulate(self, estimator, model) -> tuple[float, Any]:
        """Return globally folded batch sufficient statistics for streaming EM."""
        payloads = self._broadcast_collect(
            (
                "update",
                pickle.dumps(estimator, protocol=_PROTO),
                pickle.dumps(model, protocol=_PROTO),
            )
        )
        return self._fold_stats(estimator, payloads)

    # -- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        """Shut the worker pool down. Idempotent."""
        for conn in self._conns:
            try:
                conn.send(("stop",))
            except (BrokenPipeError, OSError):
                pass
        for conn in self._conns:
            try:
                conn.recv()
            except (EOFError, OSError):
                pass
            conn.close()
        for proc in self._procs:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
        self._conns, self._procs = [], []

    def __len__(self) -> int:
        return int(self.size)

    def __enter__(self) -> "MPEncodedData":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self):
        try:
            if self._conns:
                self.close()
        except Exception:  # noqa: BLE001
            pass
