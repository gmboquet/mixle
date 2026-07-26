"""Fault-tolerant gradient training: async DCP snapshots, loader-state capture, elastic restart,
resume-with-receipts (roadmap F2).

This is the gradient-training-side analogue of :mod:`mixle.utils.parallel.resilient_em` (K4): that module
makes the accumulator-combining EM path tolerant of a dying worker (retry, blacklist, elastic
re-partition, deterministic rendezvous-based chaos test); this module carries the same PATTERN --
"detect a failure, don't restart the whole job, resume from a checkpoint" -- to the gradient-training path,
where the parallelism is data-parallel gradient averaging rather than additive sufficient-statistic
folding, so the mechanics differ even though the shape of the fault-tolerance story does not:

* **Async DCP snapshots** (:func:`save_checkpoint_async`) -- freezes model, optimizer, loader, and caller
  metadata at one instant, then writes them on a background thread through the same immutable-generation,
  integrity-manifest, and atomic-publication protocol as synchronous training checkpoints. A resume
  therefore restores model + optimizer + data position together without trusting a detached sidecar.
* **Loader-state capture** (:class:`LoaderState`) -- mirrors the resumability contract
  :class:`mixle.data.streaming_corpus.StreamingCorpus` (F3, PR #139) already guarantees:
  ``epoch_batches(epoch)`` is a pure, deterministic function of ``(seed, epoch, rank, world_size)``, so the
  ONLY thing that changes as an epoch progresses is how many batches have been consumed -- capturing
  ``(seed, epoch, rank, world_size, batch_idx)`` is sufficient to reconstruct the identical remaining
  stream. :func:`resume_batches` resumes any loader that exposes that same ``epoch_batches`` contract, F3's
  or a synthetic stand-in.
* **Elastic restart** (:class:`SimulatedRank`, :class:`ElasticTrainingJob`) -- mirrors
  ``ResilientMPEncodedData``'s deterministic kill rendezvous (each rank signals "started" -- here, once its
  forward+backward for a step is done -- then blocks for an explicit "go" from the driver before the
  optimizer step commits, so a chaos test's kill lands at a known point, not a timing race) but adapted to
  data-parallel gradient averaging: a dead rank's gradient is simply excluded from the step's average (the
  job continues with fewer ranks, degrading gracefully) rather than failing the whole step, and a dead rank
  can be elastically respawned at its last checkpointed loader position while joining the current
  canonical model generation (not from scratch and without rewinding surviving ranks).
* **Resume-with-receipts** -- :meth:`ElasticTrainingJob.respawn_rank` marks the NEXT observed step as
  ``restart=True`` when it feeds :class:`mixle.utils.parallel.training_health.TrainingHealthMonitor`
  (F4, PR #147), so every restart automatically gets F4's per-restart continuity verdict for free -- this
  module does not reimplement that check, it wires into it.

Scope note (mirrors the ``dcp_checkpoint`` / ``resilient_em`` modules' own scoping): "10k A100s" framing
aside, checkpointing, async snapshotting, elastic restart bookkeeping, and loss-continuity verification are
all exact regardless of scale -- what genuinely does not exist on a laptop is FSDP2 sharding a model too
big for one device and a real multi-node NCCL all-reduce. Those two are simulated here: ``world_size``
ranks are real concurrent OS threads (not a real distributed job), and gradient "all-reduce" is a plain
mean over surviving ranks' locally computed grads. Every other piece -- DCP save/load, the CPU-clone async
mechanism, loader-state round-tripping, the kill rendezvous, and the continuity check -- is the same code
that would run at 10k GPUs, exercised at small scale.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from numbers import Integral, Real
from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch is optional
    torch = None

from mixle.utils.parallel.dcp_checkpoint import (
    load_training_metadata,
    load_training_state,
    save_frozen_training_state,
)
from mixle.utils.parallel.training_health import TrainingHealthMonitor

__all__ = [
    "LoaderState",
    "resume_batches",
    "AsyncCheckpointHandle",
    "save_checkpoint_async",
    "load_checkpoint",
    "StepResult",
    "SimulatedRank",
    "ElasticTrainingJob",
]


# ---------------------------------------------------------------------------
# 1. Loader-state capture
# ---------------------------------------------------------------------------


def _require_int(value: Any, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {result}")
    return result


@dataclass(frozen=True)
class LoaderState:
    """Resumability state of one rank's data loader: enough to reproduce its exact next batch after a
    restart, without serializing RNG internals or buffered batches.

    Mirrors :class:`mixle.data.streaming_corpus.StreamingCorpus`'s contract: ``epoch_batches(epoch)`` is a
    pure function of ``(seed, epoch, rank, world_size)`` (:func:`~mixle.data.streaming_corpus.
    global_document_order` reseeds from ``(seed, epoch)`` via ``SeedSequence``, then
    :func:`~mixle.data.streaming_corpus.shard_documents_for_rank` deterministically slices per rank) --
    the only thing that varies as an epoch progresses is how many batches of that deterministic stream have
    already been consumed, i.e. ``batch_idx``.
    """

    seed: int
    epoch: int
    rank: int
    world_size: int
    batch_idx: int = 0

    def __post_init__(self) -> None:
        seed = _require_int(self.seed, "seed", minimum=0)
        epoch = _require_int(self.epoch, "epoch", minimum=0)
        world_size = _require_int(self.world_size, "world_size", minimum=1)
        rank = _require_int(self.rank, "rank", minimum=0)
        batch_idx = _require_int(self.batch_idx, "batch_idx", minimum=0)
        if rank >= world_size:
            raise ValueError(f"rank must satisfy 0 <= rank < world_size, got rank={rank} world_size={world_size}")
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "epoch", epoch)
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "world_size", world_size)
        object.__setattr__(self, "batch_idx", batch_idx)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LoaderState:
        return cls(
            seed=d["seed"],
            epoch=d["epoch"],
            rank=d["rank"],
            world_size=d["world_size"],
            batch_idx=d["batch_idx"],
        )

    def advanced(self, n: int = 1) -> LoaderState:
        """The state after ``n`` more batches have been consumed this epoch."""
        n = _require_int(n, "n", minimum=0)
        return LoaderState(self.seed, self.epoch, self.rank, self.world_size, self.batch_idx + n)


def resume_batches(corpus: Any, state: LoaderState):
    """Resume a loader shaped like :class:`mixle.data.streaming_corpus.StreamingCorpus` (anything exposing
    ``epoch_batches(epoch) -> Iterator[(context, targets, loss_mask)]`` with that contract) exactly at
    ``state.batch_idx``.

    Determinism is what makes this correct rather than approximate: re-materializing the whole epoch and
    discarding the already-consumed prefix reproduces bitwise-identical remaining batches to what an
    uninterrupted run would have yielded from that point on -- the same trick
    :func:`mixle.utils.parallel.resilient_em.checkpointed_fold` relies on for exact accumulator recovery.
    """
    it = corpus.epoch_batches(state.epoch)
    for _ in range(state.batch_idx):
        next(it)
    return it


def synthetic_batch_for_state(
    state: LoaderState, *, vocab: int, block: int, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """A synthetic, F3-shaped batch: a pure, deterministic function of ``LoaderState`` alone (no external
    iterator, no corpus needed), used where a real tokenized corpus is out of scope (see
    ``streaming_corpus``'s own scope note: tokenization/corpus data is not this codebase's concern).

    Folds ``(seed, epoch, rank, world_size, batch_idx)`` into one seed via ``SeedSequence`` -- the same
    technique :func:`mixle.data.streaming_corpus.global_document_order` uses for ``(seed, epoch)`` --
    so any two calls with an equal ``LoaderState`` produce a bitwise-identical batch, and calls with
    different ``batch_idx`` never collide. This is what makes "does the resumed loader produce the same
    next batch as the uninterrupted run would have" a directly testable, bitwise claim.
    """
    vocab = _require_int(vocab, "vocab", minimum=1)
    block = _require_int(block, "block", minimum=1)
    batch_size = _require_int(batch_size, "batch_size", minimum=1)
    seed = int(
        np.random.SeedSequence([state.seed, state.epoch, state.rank, state.world_size, state.batch_idx]).generate_state(
            1
        )[0]
    )
    rng = np.random.RandomState(seed)
    x = rng.randint(0, vocab, size=(batch_size, block))
    y = rng.randint(0, vocab, size=(batch_size,))
    return torch.as_tensor(x, dtype=torch.long), torch.as_tensor(y, dtype=torch.long)


# ---------------------------------------------------------------------------
# 2. Async DCP snapshot: synchronous CPU-clone, background-thread write
# ---------------------------------------------------------------------------


@dataclass
class AsyncCheckpointHandle:
    """A checkpoint write in flight (or finished) on a background thread."""

    thread: threading.Thread
    path: str
    prepare_time_s: float  # wall-clock time the CALLER was actually blocked (the D2H clone only)
    errors: list[BaseException] = field(default_factory=list)

    def wait(self, timeout: float | None = None) -> None:
        self.thread.join(timeout=timeout)
        if self.thread.is_alive():
            raise TimeoutError("checkpoint write did not finish within the requested timeout.")
        if self.errors:
            raise RuntimeError("asynchronous checkpoint write failed.") from self.errors[0]

    @property
    def done(self) -> bool:
        return not self.thread.is_alive()


def _clone_state_tree(obj: Any) -> Any:
    """Recursively detach+clone every tensor in a (possibly nested dict/list) optimizer/model state tree,
    leaving plain (picklable) values untouched -- the frozen snapshot the background thread writes from,
    so training mutating the LIVE tensors after this function returns cannot corrupt the write in flight."""
    if torch.is_tensor(obj):
        return obj.detach().clone().cpu()
    if isinstance(obj, dict):
        return {k: _clone_state_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clone_state_tree(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_clone_state_tree(v) for v in obj)
    return obj


def save_checkpoint_async(
    module: Any,
    optimizer: Any,
    path: str,
    loader_state: LoaderState,
    *,
    extra: dict[str, Any] | None = None,
) -> AsyncCheckpointHandle:
    """Snapshot ``(model, optimizer, loader_state)`` to ``path`` without blocking the training loop.

    The ONLY synchronous work is ``get_state_dict`` + a detached CPU clone (a bounded D2H-copy cost that
    does not scale with disk/network write latency). The transactional generation write and publication
    happen on a background thread, so this function returns as soon as the clone is done and the training
    loop's next step can start immediately.

    ``loader_state`` and caller ``extra`` use the same integrity-bound typed
    sidecar and immutable-generation publication protocol as
    :mod:`mixle.utils.parallel.dcp_checkpoint`.
    """
    from torch.distributed.checkpoint.state_dict import get_state_dict

    t0 = time.perf_counter()
    model_sd, optim_sd = get_state_dict(module, optimizer)
    model_sd = _clone_state_tree(model_sd)
    optim_sd = _clone_state_tree(optim_sd)
    prepare_time_s = time.perf_counter() - t0

    loader_payload = loader_state.to_dict()
    extra_payload = dict(extra or {})

    errors: list[BaseException] = []

    def _write() -> None:
        try:
            save_frozen_training_state(
                model_sd,
                optim_sd,
                path,
                step=int(loader_payload["batch_idx"]),
                loader_state=loader_payload,
                extra=extra_payload,
            )
        except BaseException as error:  # noqa: BLE001 - surfaced by AsyncCheckpointHandle.wait
            errors.append(error)

    thread = threading.Thread(target=_write, daemon=True)
    thread.start()
    return AsyncCheckpointHandle(thread=thread, path=str(path), prepare_time_s=prepare_time_s, errors=errors)


def load_checkpoint(module: Any, optimizer: Any, path: str) -> LoaderState:
    """Load a verified transactional checkpoint and return its loader position."""
    payload = load_training_state(module, optimizer, path)
    return LoaderState.from_dict(payload["loader_state"])


def load_checkpoint_extra(path: str) -> dict[str, Any]:
    """Return verified typed ``extra`` metadata without mutating training state."""
    payload = load_training_metadata(path)
    return dict(payload.get("extra") or {})


# ---------------------------------------------------------------------------
# 3. Elastic restart: simulated data-parallel ranks, deterministic kill rendezvous
# ---------------------------------------------------------------------------


@dataclass
class StepResult:
    rank: int
    step: int
    loss: float
    grad_norm: float
    grads: list[torch.Tensor]


class SimulatedRank:
    """One data-parallel rank's local training-step worker, run on a real background thread.

    Mirrors :class:`mixle.utils.parallel.resilient_em.ResilientMPEncodedData`'s rendezvous: after computing
    a full forward+backward pass for a step -- the point at which a real GPU worker would ordinarily
    all-reduce gradients and step the optimizer -- the thread signals ``started`` and BLOCKS waiting for an
    explicit ``go`` from the driver. This pins "mid-step" to a known point (strictly after gradient
    computation, strictly before the step is applied), so a chaos test's kill is deterministic, not a
    timing race: a kill issued at this rendezvous is guaranteed to land before any weight update happens.

    ``kill()`` needs no OS-level teardown (real process kill, as ``resilient_em`` does, is not available
    for an in-process thread): simply never releasing the rendezvous IS the simulated crash -- the thread
    times out and exits with no result, exactly as a real dead worker would leave the driver's ``recv()``
    hanging until it gives up.
    """

    def __init__(self, rank_id: int, model_factory: Callable[[], Any], batch_fn: Callable[[], tuple[Any, Any]]):
        self.rank_id = _require_int(rank_id, "rank_id", minimum=0)
        self.model_factory = model_factory
        self.batch_fn = batch_fn
        self._started = threading.Event()
        self._go = threading.Event()
        self._done = threading.Event()
        self._result: StepResult | None = None
        self._error: BaseException | None = None
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()

    def start_step(self, step: int, canonical_state_dict: dict[str, Any]) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError(f"rank {self.rank_id} still has a step in flight")
        step = _require_int(step, "step", minimum=0)
        started = threading.Event()
        go = threading.Event()
        done = threading.Event()
        cancel = threading.Event()
        self._started = started
        self._go = go
        self._done = done
        self._cancel = cancel
        self._result = None
        self._error = None

        def _run() -> None:
            try:
                local_model = self.model_factory()
                local_model.load_state_dict(canonical_state_dict)
                x, y = self.batch_fn()
                if not torch.is_tensor(x) or not torch.is_tensor(y):
                    raise TypeError("batch_fn must return two tensors")
                if x.ndim < 1 or y.ndim < 1 or x.shape[0] <= 0 or x.shape[0] != y.shape[0]:
                    raise ValueError(
                        f"batch_fn returned incompatible batch dimensions x={tuple(x.shape)} y={tuple(y.shape)}"
                    )
                local_model.zero_grad(set_to_none=True)
                logits = local_model(x)
                loss = torch.nn.functional.cross_entropy(logits, y)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=1e9)
                grads = [
                    (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p))
                    for p in local_model.parameters()
                ]
                if cancel.is_set():
                    return
                started.set()
                go.wait()
                if cancel.is_set():
                    return
                self._result = StepResult(self.rank_id, step, float(loss.item()), float(grad_norm.item()), grads)
            except BaseException as e:  # noqa: BLE001 - surface on the driver, mirroring resilient_em's worker path
                self._error = e
                started.set()
            finally:
                done.set()

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def wait_started(self, timeout: float = 5.0) -> bool:
        return self._started.wait(timeout=timeout)

    def release(self) -> None:
        """Wave this rank through the rendezvous -- it survives this step."""
        self._go.set()

    def cancel(self) -> None:
        """Cancel this step and wake a worker already waiting at the rendezvous."""
        self._cancel.set()
        self._go.set()

    def join(self, timeout: float = 5.0) -> StepResult | None:
        if not self._done.wait(timeout=max(0.0, timeout)):
            raise TimeoutError(f"rank {self.rank_id} did not finish before the step deadline")
        if self._error is not None:
            raise RuntimeError(f"rank {self.rank_id} step {self._error!r} failed") from self._error
        return self._result


def _average_grads(grad_lists: list[list[torch.Tensor]]) -> list[torch.Tensor]:
    n = len(grad_lists)
    return [sum(gs) / n for gs in zip(*grad_lists)]


class ElasticTrainingJob:
    """A data-parallel training loop, chaos-tolerant: a rank dying mid-step degrades gracefully -- the
    job continues averaging over fewer surviving ranks that step, rather than hard-failing -- and a dead
    rank can be elastically respawned at its last checkpointed loader position and rejoin the current
    canonical model generation instead of restarting or rewinding the whole job.

    Holds one canonical ``(model, optimizer)`` (what gets checkpointed and what every live replica receives
    at the start of its next step);
    each :class:`SimulatedRank` computes its OWN local forward+backward against a fresh copy of the
    canonical weights (the data-parallel replica), and the driver applies the mean of surviving ranks'
    gradients to the canonical model once per step -- the plain-mean "all-reduce" this module's docstring
    flags as the one piece that is genuinely simulated rather than exercised for real.

    Every restart is wired into F4's continuity check for free: :meth:`respawn_rank` marks the NEXT
    ``run_step`` call as ``restart=True`` when it feeds :attr:`health`, so ``health.report()["restarts"]``
    always carries a real per-restart continuity verdict, not something the caller has to remember to ask
    for.

    **Single-driver only.** :meth:`run_step`, :meth:`checkpoint`, and :meth:`respawn_rank` share mutable
    state (:attr:`canonical_model`, :attr:`canonical_optimizer`, :attr:`loader_states`) with no internal
    locking, and are meant to be called from the one thread that constructed this job -- calling any of
    them from a second thread while another is in flight is a misuse this class detects and rejects
    (:class:`RuntimeError`) rather than silently letting it race: ``run_step`` advances the canonical
    weights before it advances :attr:`loader_states`, so a concurrent :meth:`checkpoint` can capture a
    model state newer than the loader state written alongside it. Periodic/background checkpointing must
    be driven from the same thread as ``run_step`` (e.g. called between steps), not a separate one.
    """

    def __init__(
        self,
        model_factory: Callable[[], Any],
        world_size: int,
        batch_fn_for_rank: Callable[[int, LoaderState], tuple[Any, Any]],
        checkpoint_dir: str,
        *,
        seed: int = 0,
        lr: float = 1e-2,
        step_timeout_s: float = 5.0,
        health_monitor: TrainingHealthMonitor | None = None,
    ) -> None:
        world_size = _require_int(world_size, "world_size", minimum=1)
        seed = _require_int(seed, "seed", minimum=0)
        if (
            isinstance(step_timeout_s, bool)
            or not isinstance(step_timeout_s, Real)
            or not np.isfinite(step_timeout_s)
            or step_timeout_s <= 0
        ):
            raise ValueError(f"step_timeout_s must be finite and positive, got {step_timeout_s!r}")
        self.model_factory = model_factory
        self.world_size = world_size
        self.batch_fn_for_rank = batch_fn_for_rank
        self.checkpoint_dir = str(checkpoint_dir)
        self.step_timeout_s = float(step_timeout_s)

        self.canonical_model = model_factory()
        self.canonical_optimizer = torch.optim.SGD(self.canonical_model.parameters(), lr=lr)
        self.health = health_monitor or TrainingHealthMonitor(loss_window=10, loss_min_periods=3, loss_z_thresh=6.0)

        self.loader_states: dict[int, LoaderState] = {
            r: LoaderState(seed=seed, epoch=0, rank=r, world_size=self.world_size, batch_idx=0)
            for r in range(self.world_size)
        }
        self.ranks: dict[int, SimulatedRank] = {}
        for r in range(self.world_size):
            self._spawn_rank(r)
        self.dead_ranks: set[int] = set()
        self.pending_restart = False  # True right after a resume -- consumed by the next observed step
        self.last_checkpoint_handle: AsyncCheckpointHandle | None = None
        self.history: list[dict[str, Any]] = []
        self._owner_thread_id = threading.get_ident()

    def _spawn_rank(self, rank_id: int) -> None:
        self.ranks[rank_id] = SimulatedRank(
            rank_id, self.model_factory, lambda r=rank_id: self.batch_fn_for_rank(r, self.loader_states[r])
        )

    def _check_single_driver(self, method_name: str) -> None:
        current = threading.get_ident()
        if current != self._owner_thread_id:
            raise RuntimeError(
                f"ElasticTrainingJob.{method_name}() called from thread {current}, but this job was "
                f"constructed on thread {self._owner_thread_id}. run_step()/checkpoint()/respawn_rank() "
                "share mutable state with no internal locking and must be driven from a single thread "
                '-- see the class docstring\'s "Single-driver only" note.'
            )

    def run_step(self, step: int, kill_ranks: frozenset[int] = frozenset()) -> dict[str, Any]:
        """Run one data-parallel step. ``kill_ranks`` simulates a mid-step death for those ranks: they
        reach the post-backward rendezvous (so their compute genuinely happened) but are never released,
        so their gradient is excluded from this step's average -- the job continues with fewer ranks
        rather than raising."""
        self._check_single_driver("run_step")
        step = _require_int(step, "step", minimum=0)
        kill_set = set(kill_ranks)
        if any(isinstance(rank, bool) or not isinstance(rank, Integral) for rank in kill_set):
            raise TypeError("kill_ranks must contain integer rank identifiers")
        kill_set = {int(rank) for rank in kill_set}
        invalid = sorted(rank for rank in kill_set if rank < 0 or rank >= self.world_size)
        if invalid:
            raise ValueError(f"kill_ranks contains out-of-world rank identifiers: {invalid}")
        canonical_sd = {k: v.detach().clone() for k, v in self.canonical_model.state_dict().items()}
        live = [r for r in self.ranks if r not in self.dead_ranks]
        if not live:
            raise RuntimeError("ElasticTrainingJob has no live ranks left.")
        already_dead = sorted(kill_set.intersection(self.dead_ranks))
        if already_dead:
            raise ValueError(f"kill_ranks contains ranks that are already dead: {already_dead}")

        deadline = time.monotonic() + self.step_timeout_s
        for r in live:
            self.ranks[r].start_step(step, canonical_sd)
        pending = set(live)
        ready: set[int] = set()
        while pending:
            for r in tuple(pending):
                if not self.ranks[r].wait_started(0.0):
                    continue
                pending.remove(r)
                ready.add(r)
                if r in kill_set:
                    self.ranks[r].cancel()
                else:
                    self.ranks[r].release()
            remaining = max(0.0, deadline - time.monotonic())
            if not pending or remaining == 0.0:
                break
            time.sleep(min(0.001, remaining))

        newly_dead = pending | kill_set
        for r in newly_dead:
            self.ranks[r].cancel()
        survivors = [r for r in live if r in ready and r not in kill_set]

        results: dict[int, StepResult] = {}
        rank_failures: dict[int, str] = {}
        for r in survivors:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                res = self.ranks[r].join(remaining)
            except (RuntimeError, TimeoutError) as error:
                self.ranks[r].cancel()
                newly_dead.add(r)
                rank_failures[r] = str(error)
                continue
            if res is not None:
                results[r] = res
            else:
                newly_dead.add(r)

        for r in newly_dead:
            self.dead_ranks.add(r)
        if not results:
            raise RuntimeError(f"all live ranks died at step {step} -- nothing to average")

        avg_grads = _average_grads([res.grads for res in results.values()])
        self.canonical_optimizer.zero_grad(set_to_none=True)
        for p, g in zip(self.canonical_model.parameters(), avg_grads):
            p.grad = g.clone()
        self.canonical_optimizer.step()

        mean_loss = float(np.mean([res.loss for res in results.values()]))
        mean_grad_norm = float(np.mean([res.grad_norm for res in results.values()]))

        for r in results:  # only ranks that actually produced a batch this step advance their position
            self.loader_states[r] = self.loader_states[r].advanced()

        restarted_this_step = self.pending_restart
        anomalies = self.health.observe_step(step, mean_loss, grad_norm=mean_grad_norm, restart=restarted_this_step)
        self.pending_restart = False

        record = {
            "step": step,
            "loss": mean_loss,
            "grad_norm": mean_grad_norm,
            "survivors": sorted(results),
            "newly_dead": sorted(newly_dead),
            "rank_failures": rank_failures,
            "restart": restarted_this_step,
            "anomalies": [a.kind for a in anomalies],
        }
        self.history.append(record)
        return record

    def checkpoint(self, path: str | None = None) -> AsyncCheckpointHandle:
        """Async-snapshot the canonical model/optimizer plus every rank's loader state."""
        self._check_single_driver("checkpoint")
        path = path or self.checkpoint_dir
        rank0_state = self.loader_states[0]
        extra = {"loader_states": {r: s.to_dict() for r, s in self.loader_states.items()}}
        handle = save_checkpoint_async(self.canonical_model, self.canonical_optimizer, path, rank0_state, extra=extra)
        self.last_checkpoint_handle = handle
        return handle

    def respawn_rank(self, rank_id: int, checkpoint_path: str | None = None) -> LoaderState:
        """Bring ``rank_id`` back at its checkpointed data position without rolling back the live job.

        The respawned replica receives the current canonical model at the start of its next step, exactly
        like every other replica. Only its own loader position comes from the checkpoint; committed model
        updates and surviving ranks' positions remain untouched. Marks the next ``run_step`` as a restart
        so F4's continuity check evaluates it.
        """
        self._check_single_driver("respawn_rank")
        rank_id = _require_int(rank_id, "rank_id", minimum=0)
        if rank_id >= self.world_size:
            raise ValueError(f"rank_id must be less than world_size={self.world_size}, got {rank_id}")
        if rank_id not in self.dead_ranks:
            raise ValueError(f"rank {rank_id} is already live and cannot be respawned")
        path = checkpoint_path or self.checkpoint_dir
        # A checkpoint write in flight to this same path must finish before it's read back, or this can
        # load a partially-written / stale directory -- wait rather than trust the caller to have done so.
        # Existing callers (e.g. the respawn test in mixle/tests/fault_tolerant_training_test.py) already
        # wait manually before respawning; this turns that into a guarantee instead of a convention.
        if self.last_checkpoint_handle is not None and self.last_checkpoint_handle.path == path:
            self.last_checkpoint_handle.wait()
        extra = load_checkpoint_extra(path)
        loader_states = extra.get("loader_states")
        if not isinstance(loader_states, dict):
            raise RuntimeError("checkpoint does not contain rank-local loader states")
        saved_state = loader_states.get(rank_id, loader_states.get(str(rank_id)))
        if saved_state is None:
            raise RuntimeError(f"checkpoint does not contain loader state for rank {rank_id}")
        restored_state = LoaderState.from_dict(saved_state)
        if restored_state.rank != rank_id or restored_state.world_size != self.world_size:
            raise RuntimeError(
                "checkpoint loader coordinates do not match this job: "
                f"saved rank={restored_state.rank} world_size={restored_state.world_size}, "
                f"expected rank={rank_id} world_size={self.world_size}"
            )
        self.loader_states[rank_id] = restored_state
        self.dead_ranks.discard(rank_id)
        self._spawn_rank(rank_id)
        self.pending_restart = True
        return self.loader_states[rank_id]

    def continuity_ok(self) -> bool:
        return self.health.continuity_ok()
