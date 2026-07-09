"""Training-health receipts: MFU, loss/grad-norm anomaly detection, precision checks, restart continuity.

The frontier-scale trainer (the vendored TorchTitan/Megatron loop atop :mod:`mixle.models.transformer`,
sharded via :mod:`mixle.utils.parallel.torchrun` / :mod:`mixle.utils.parallel.dcp_checkpoint`) runs for
weeks unattended -- the only way to know it is healthy is receipts computed *from the loop itself*, not a
human staring at a loss curve. :class:`TrainingHealthMonitor` is that receipt machine: call
``observe_step(...)`` once per optimizer step (mirroring :meth:`mixle.telemetry.core.Telemetry.record`) and
call ``report()`` once at the end (mirroring :class:`mixle.evolve.ledger.EvolutionLedger` /
:meth:`mixle.evolve.population.OperatorBandit.report`) for a structured, JSON-serializable summary.

Four things are tracked:

* **MFU** -- :class:`ModelFlopConfig` computes the theoretical FLOPs/step for a transformer config (the
  standard ``6N + attention`` accounting, same formula nanoGPT's ``estimate_mfu`` uses); ``achieved
  FLOPs/sec`` comes from a caller-supplied wall-clock step time (real timing, whatever hardware the loop
  runs on); MFU is the ratio against a caller-supplied hardware peak.
* **Loss-spike / changepoint detection** -- a robust (median/MAD) rolling z-score per step.
* **Grad-norm / precision anomalies** -- the same rolling z-score on grad-norm, plus NaN/Inf checks on both
  streams (these always fire, independent of the rolling window's warmup).
* **Per-restart continuity** -- ``restart=True`` on the first step after a checkpoint resume marks the
  rolling baseline boundary; the next step's z-score is evaluated against the *pre-restart* baseline (it
  has not been updated with any post-restart value yet), so a resume that silently drops optimizer/RNG
  state and produces a real loss jump is caught as ``restart_discontinuity`` -- a well-behaved resume is
  not.
* **Dead-rank liveness** -- ``observe_rank_step(rank, step)`` is a per-rank heartbeat: a data-parallel loop
  calls it once per step per rank (mirroring how :class:`~mixle.utils.parallel.fault_tolerant_training.
  ElasticTrainingJob` already tracks ``dead_ranks`` for gradient-averaging purposes, but that bookkeeping
  never surfaced as a *health receipt* -- this does). ``check_rank_liveness(current_step)`` flags any rank
  that has gone silent for more than ``rank_heartbeat_threshold`` steps as ``dead_rank``, once per outage.

No cluster is required to exercise any of this: the FLOPs accounting and anomaly math are exact regardless
of scale, and the tests drive a real (tiny) :func:`mixle.models.transformer.build_causal_lm` for a handful
of steps. The *absolute* MFU number a laptop/CI runner produces is not comparable to a real cluster's --
that comparison is deferred until the real distributed trainer (roadmap F1) exists.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "Anomaly",
    "ModelFlopConfig",
    "MFUSample",
    "RollingBaseline",
    "StepRecord",
    "TrainingHealthMonitor",
    "theoretical_flops_per_iter",
    "flop_config_from_causal_lm",
]


# ---------------------------------------------------------------------------
# MFU: theoretical FLOPs accounting + achieved/peak ratio
# ---------------------------------------------------------------------------


@dataclass
class ModelFlopConfig:
    """The shape a transformer's FLOPs/step depends on -- enough to compute theoretical FLOPs exactly.

    ``n_params`` should exclude position-embedding params (they are a lookup, not a matmul); use
    :func:`flop_config_from_causal_lm` to derive this correctly from a real
    :class:`mixle.models.transformer.CausalLM`.
    """

    n_params: int
    n_layer: int
    n_head: int
    d_model: int
    seq_len: int

    def __post_init__(self) -> None:
        if self.d_model % self.n_head != 0:
            raise ValueError(f"d_model={self.d_model} not divisible by n_head={self.n_head}")

    def flops_per_iter(self, batch_size: int) -> float:
        return theoretical_flops_per_iter(
            n_params=self.n_params,
            n_layer=self.n_layer,
            n_head=self.n_head,
            d_model=self.d_model,
            seq_len=self.seq_len,
            batch_size=batch_size,
        )


def theoretical_flops_per_iter(
    *, n_params: int, n_layer: int, n_head: int, d_model: int, seq_len: int, batch_size: int
) -> float:
    """Theoretical forward+backward FLOPs for one training iteration of a decoder-only transformer.

    The standard two-term accounting (see nanoGPT's ``estimate_mfu`` / the PaLM paper appendix): ``6*N``
    per token for the dense matmuls (forward+backward is ~3x forward, and each matmul is a multiply-add =
    2 FLOPs, giving the well-known ``6N`` per-token constant), plus ``12*L*H*Q*T`` per token for the
    attention matmuls (QK^T and attn@V), which scale with context length ``T`` and are *not* captured by
    the ``6N`` param-count term. Multiplying by ``T`` (all tokens in the sequence) and ``batch_size`` gives
    the FLOPs for one full iteration.
    """
    if seq_len <= 0 or batch_size <= 0:
        raise ValueError("seq_len and batch_size must be positive")
    head_dim = d_model / n_head
    flops_per_token = 6.0 * n_params + 12.0 * n_layer * n_head * head_dim * seq_len
    flops_per_fwdbwd = flops_per_token * seq_len
    return flops_per_fwdbwd * batch_size


def flop_config_from_causal_lm(model: Any, seq_len: int) -> ModelFlopConfig:
    """Derive a :class:`ModelFlopConfig` from a real :class:`mixle.models.transformer.CausalLM`.

    Position-embedding params are excluded from the count (a lookup table, not a matmul) -- the same
    convention nanoGPT's ``get_num_params(non_embedding=True)`` uses.
    """
    n_params = sum(p.numel() for p in model.parameters())
    pos = getattr(model, "pos", None)
    if pos is not None and hasattr(pos, "weight"):
        n_params -= pos.weight.numel()
    return ModelFlopConfig(
        n_params=int(n_params),
        n_layer=int(model.n_layer),
        n_head=int(model.n_head),
        d_model=int(model.d_model),
        seq_len=int(seq_len),
    )


@dataclass
class MFUSample:
    """One step's MFU measurement: real wall-clock timing against the theoretical FLOPs for that step."""

    step: int
    step_flops: float
    step_time_s: float
    peak_flops_per_sec: float

    @property
    def achieved_flops_per_sec(self) -> float:
        return self.step_flops / self.step_time_s if self.step_time_s > 0 else float("nan")

    @property
    def mfu(self) -> float:
        if self.peak_flops_per_sec <= 0:
            return float("nan")
        return self.achieved_flops_per_sec / self.peak_flops_per_sec


# ---------------------------------------------------------------------------
# Loss-spike / grad-norm changepoint detection: robust (median/MAD) rolling z-score
# ---------------------------------------------------------------------------


class RollingBaseline:
    """A robust rolling baseline (median + scaled MAD) over a trailing window of values.

    Median/MAD rather than mean/std so a *previous* spike does not itself blow up the spread used to judge
    the *next* one. ``z_score(value)`` is evaluated against the window as it stood before ``value`` was
    seen, so calling ``update`` only after scoring makes the check causal (no leakage of the current point
    into its own baseline) -- this is also what makes the restart-continuity check work: the window carries
    only pre-restart history until the caller updates it.
    """

    def __init__(self, window: int = 20, min_periods: int = 5):
        if min_periods < 2:
            raise ValueError("min_periods must be >= 2")
        self.window = int(window)
        self.min_periods = int(min_periods)
        self._values: deque[float] = deque(maxlen=self.window)

    def baseline(self) -> tuple[float, float] | None:
        """``(median, scaled_mad)`` of the current window, or ``None`` during warmup."""
        if len(self._values) < self.min_periods:
            return None
        arr = np.asarray(self._values, dtype=float)
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med))) * 1.4826  # consistent estimator of std under normality
        return med, mad

    def z_score(self, value: float) -> float | None:
        """Robust z-score of ``value`` against the window *before* ``value`` is added, or ``None`` in warmup."""
        b = self.baseline()
        if b is None:
            return None
        med, mad = b
        spread = mad if mad > 1e-12 else 1e-12
        return (value - med) / spread

    def update(self, value: float) -> None:
        self._values.append(float(value))

    def state(self) -> dict[str, Any]:
        """Serializable snapshot -- carry this across a checkpoint/restart to preserve continuity."""
        return {"window": self.window, "min_periods": self.min_periods, "values": list(self._values)}

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> RollingBaseline:
        obj = cls(window=state["window"], min_periods=state["min_periods"])
        obj._values.extend(state["values"])
        return obj


# ---------------------------------------------------------------------------
# The run: one record per step, one report() at the end
# ---------------------------------------------------------------------------


@dataclass
class StepRecord:
    """One observed training step."""

    step: int
    loss: float
    grad_norm: float | None = None
    step_time_s: float | None = None
    restart: bool = False


@dataclass
class Anomaly:
    """One flagged anomaly: what kind, at which step, and the baseline it was judged against."""

    step: int
    kind: str  # "nan_inf_loss" | "nan_inf_grad" | "loss_spike" | "grad_norm_spike" | "restart_discontinuity"
    # | "dead_rank"
    value: float
    baseline: float | None
    z_score: float | None
    detected_at_step: int  # step at which the monitor raised this (== step for spikes, == step for nan/inf)

    def latency(self, injected_at_step: int) -> int:
        """Steps between the true anomaly step and detection -- 0 means caught on the same step."""
        return self.detected_at_step - injected_at_step


class TrainingHealthMonitor:
    """The ``run`` object: ``observe_step(...)`` per optimizer step, ``report()`` once at the end.

    Follows the shape of :class:`mixle.telemetry.core.Telemetry` (``record``/buffer) and
    :class:`mixle.evolve.ledger.EvolutionLedger` (``.report()``-style terminal summary): no I/O, pure
    in-process accounting, JSON-serializable output.
    """

    def __init__(
        self,
        *,
        flop_config: ModelFlopConfig | None = None,
        peak_flops_per_sec: float | None = None,
        loss_window: int = 20,
        loss_min_periods: int = 5,
        loss_z_thresh: float = 6.0,
        grad_window: int = 20,
        grad_min_periods: int = 5,
        grad_z_thresh: float = 6.0,
        rank_heartbeat_threshold: int = 50,
    ) -> None:
        self.flop_config = flop_config
        self.peak_flops_per_sec = peak_flops_per_sec
        self.loss_z_thresh = float(loss_z_thresh)
        self.grad_z_thresh = float(grad_z_thresh)
        self.rank_heartbeat_threshold = int(rank_heartbeat_threshold)

        self._loss_baseline = RollingBaseline(window=loss_window, min_periods=loss_min_periods)
        self._grad_baseline = RollingBaseline(window=grad_window, min_periods=grad_min_periods)

        self.records: list[StepRecord] = []
        self.anomalies: list[Anomaly] = []
        self.mfu_samples: list[MFUSample] = []
        self._pending_restart_step: int | None = None  # step index whose *next* observation is post-restart
        self._rank_last_seen: dict[str, int] = {}  # rank name -> last step it reported a heartbeat
        self._dead_ranks_flagged: set[str] = set()  # ranks already flagged dead in the current outage

    def observe_step(
        self,
        step: int,
        loss: float,
        *,
        grad_norm: float | None = None,
        step_time_s: float | None = None,
        batch_size: int | None = None,
        restart: bool = False,
    ) -> list[Anomaly]:
        """Record one step; returns any anomalies raised at this step (also appended to ``self.anomalies``)."""
        loss = float(loss)
        rec = StepRecord(step=step, loss=loss, grad_norm=grad_norm, step_time_s=step_time_s, restart=restart)
        self.records.append(rec)
        found: list[Anomaly] = []

        is_post_restart = self._pending_restart_step is not None and step == self._pending_restart_step + 1

        # --- precision: NaN/Inf, unconditional, no warmup needed ---------------------------------------
        if not math.isfinite(loss):
            found.append(
                Anomaly(step=step, kind="nan_inf_loss", value=loss, baseline=None, z_score=None, detected_at_step=step)
            )
        if grad_norm is not None and not math.isfinite(grad_norm):
            found.append(
                Anomaly(
                    step=step,
                    kind="nan_inf_grad",
                    value=float(grad_norm),
                    baseline=None,
                    z_score=None,
                    detected_at_step=step,
                )
            )

        # --- loss spike / restart discontinuity (same statistic, different label) ----------------------
        if math.isfinite(loss):
            z = self._loss_baseline.z_score(loss)
            if z is not None and z > self.loss_z_thresh:
                base = self._loss_baseline.baseline()
                kind = "restart_discontinuity" if is_post_restart else "loss_spike"
                found.append(
                    Anomaly(
                        step=step,
                        kind=kind,
                        value=loss,
                        baseline=base[0] if base else None,
                        z_score=z,
                        detected_at_step=step,
                    )
                )
            self._loss_baseline.update(loss)

        # --- grad-norm spike ------------------------------------------------------------------------
        if grad_norm is not None and math.isfinite(grad_norm):
            gz = self._grad_baseline.z_score(grad_norm)
            if gz is not None and gz > self.grad_z_thresh:
                gbase = self._grad_baseline.baseline()
                found.append(
                    Anomaly(
                        step=step,
                        kind="grad_norm_spike",
                        value=float(grad_norm),
                        baseline=gbase[0] if gbase else None,
                        z_score=gz,
                        detected_at_step=step,
                    )
                )
            self._grad_baseline.update(grad_norm)

        if restart:
            self._pending_restart_step = step
        elif is_post_restart:
            self._pending_restart_step = None  # consumed; only the immediate next step is checked

        # --- MFU -------------------------------------------------------------------------------------
        if step_time_s is not None and self.flop_config is not None and self.peak_flops_per_sec:
            bs = int(batch_size) if batch_size is not None else 1
            step_flops = self.flop_config.flops_per_iter(bs)
            self.mfu_samples.append(
                MFUSample(
                    step=step,
                    step_flops=step_flops,
                    step_time_s=float(step_time_s),
                    peak_flops_per_sec=self.peak_flops_per_sec,
                )
            )

        self.anomalies.extend(found)
        return found

    def observe_rank_step(self, rank: str | int, step: int) -> None:
        """Record a per-rank heartbeat: ``rank`` reported liveness (e.g. completed a local forward+backward)
        at ``step``. Call once per step per rank; combine with :meth:`check_rank_liveness` to detect a rank
        that has stopped reporting. A rank reporting again after an outage clears its ``dead_rank`` flag so a
        later re-death can be flagged again.
        """
        name = str(rank)
        self._rank_last_seen[name] = int(step)
        self._dead_ranks_flagged.discard(name)

    def check_rank_liveness(self, current_step: int) -> list[Anomaly]:
        """Flag any known rank that has not reported a heartbeat (:meth:`observe_rank_step`) for more than
        ``rank_heartbeat_threshold`` steps as ``dead_rank``. Raised once per outage (not once per step) --
        the flag is cleared the next time that rank reports in, so a respawned/recovered rank can be caught
        going dead again later.
        """
        found: list[Anomaly] = []
        for rank, last_seen in self._rank_last_seen.items():
            silence = current_step - last_seen
            if silence > self.rank_heartbeat_threshold and rank not in self._dead_ranks_flagged:
                found.append(
                    Anomaly(
                        step=last_seen,
                        kind="dead_rank",
                        value=float(silence),
                        baseline=float(self.rank_heartbeat_threshold),
                        z_score=None,
                        detected_at_step=current_step,
                    )
                )
                self._dead_ranks_flagged.add(rank)
        self.anomalies.extend(found)
        return found

    def continuity_ok(self) -> bool:
        """``True`` unless any ``restart_discontinuity`` was ever flagged."""
        return not any(a.kind == "restart_discontinuity" for a in self.anomalies)

    def report(self) -> dict[str, Any]:
        """A complete, JSON-serializable summary: step count, MFU, anomalies by kind, continuity verdict."""
        by_kind: dict[str, int] = {}
        for a in self.anomalies:
            by_kind[a.kind] = by_kind.get(a.kind, 0) + 1

        mfu_values = [s.mfu for s in self.mfu_samples if math.isfinite(s.mfu)]
        restart_steps = [r.step for r in self.records if r.restart]

        return {
            "n_steps": len(self.records),
            "n_anomalies": len(self.anomalies),
            "anomalies_by_kind": by_kind,
            "anomalies": [
                {
                    "step": a.step,
                    "kind": a.kind,
                    "value": a.value,
                    "baseline": a.baseline,
                    "z_score": a.z_score,
                }
                for a in self.anomalies
            ],
            "mfu": {
                "n_samples": len(self.mfu_samples),
                "mean": float(np.mean(mfu_values)) if mfu_values else None,
                "min": float(np.min(mfu_values)) if mfu_values else None,
                "max": float(np.max(mfu_values)) if mfu_values else None,
            },
            "restarts": {
                "steps": restart_steps,
                "continuity_ok": self.continuity_ok(),
            },
        }
