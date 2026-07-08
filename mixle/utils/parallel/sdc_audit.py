"""Silent-data-corruption (SDC) audit for the resilient mp EM backend (K5).

Status audit this module leans on (see K4's docstring in ``resilient_em.py``, which this
module builds directly on top of): fits are deterministic given seed (#115) and sufficient
statistics are ADDITIVE. K4 already turns those two facts into exact, cheap *failure*
recovery (a crashed worker's shard is just redone). K5's contribution is different: it turns
the SAME two facts into a cheap *corruption* detector.

**The idea.** Silent data corruption (SDC) is a bit-flip in memory or compute that produces a
WRONG result with no exception and no crash -- the fit just silently converges to a corrupted
answer. For gradient-based training this is expensive to catch (you'd have to redo an entire
gradient step, or use redundant hardware). mixle's additive-stat EM makes it cheap: recompute
the SAME shard's accumulator TWICE, from the SAME raw shard bytes / same estimator / same
model / same ``sub_chunks`` split, once on the shard's normal ("primary") rank and once on a
DIFFERENT ("audit") rank. Determinism-given-seed + additive stats + fixed chunking means the
two ``(count, value())`` payloads MUST be bitwise identical if nothing is corrupted -- so any
byte-level difference is proof of a real hardware/software fault, not numerical noise (there
is no tolerance band to tune, unlike a gradient re-check).

**Why zero false positives holds "by construction", not just empirically:**
    1. Both the primary and audit recompute start from ``self._shard_raw[shard_id]`` -- the
       IDENTICAL pickled raw observations the driver has held since construction. Neither rank
       is given a different view of the data.
    2. Both recomputes run the SAME code path: ``_worker_main``'s ``"update_shard"`` handler,
       which pickles the shard with the SAME ``sub_chunks`` value (see ``_encode_shard``), so
       floating-point summation order -- which is NOT associative -- is identical on both
       ranks. This is exactly the "fixed chunking" half of the guarantee: without pinning
       ``sub_chunks``, two honest ranks could legitimately disagree in the last bit.
    3. ``seq_update`` (the E-step) and ``seq_initialize`` are pure, deterministic functions of
       (encoded data, weights, model[, seed]) -- no per-rank RNG state, no nondeterministic
       parallel reduction inside a single call. IEEE-754 arithmetic is deterministic given a
       fixed sequence of operations, so replaying the identical operation sequence on
       different physical hardware still produces the identical bit pattern.
    4. Therefore an uncorrupted primary and an uncorrupted audit recompute of the same shard
       are the same pure computation evaluated twice, and must agree bit-for-bit. A mismatch
       can only arise if at least one of the two computations was NOT the pure computation --
       i.e. something (memory fault, cosmic ray, buggy kernel) perturbed it. This is verified
       empirically too, at scale, in ``mixle/tests/sdc_audit_test.py``.

**NaN/Inf watchdog.** ``mixle.models._neural_serial.check_finite`` already guards individual
density evaluations. K5 extends that same "fail loud, immediately, with the offending
location named" philosophy to the accumulator-fold boundary: :func:`finite_guarded_fold` is a
drop-in replacement for :func:`~mixle.utils.parallel.resilient_em.checkpointed_fold` that
checks finiteness of the running accumulator's ``value()`` immediately after EVERY
``combine()`` call, not just once at the end -- so a NaN/Inf introduced while folding payload
``i`` is caught at payload ``i``'s ``combine()`` boundary, before it silently propagates into
payload ``i+1..n``. Scope note: this wraps ``combine()`` calls made by THIS module's own fold
loop only (mirroring ``checkpointed_fold``'s loop) -- it deliberately does NOT touch
``SequenceEncodableStatisticAccumulator.combine()``'s contract itself, which every other
caller in the codebase still uses unguarded, exactly as before.
"""

from __future__ import annotations

import hashlib
import pickle
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mixle.models._neural_serial import check_finite
from mixle.utils.parallel.resilient_em import ResilientMPEncodedData

__all__ = [
    "AuditedMPEncodedData",
    "SDCAuditReceipt",
    "finite_guarded_fold",
    "inject_bit_flip",
]

_PROTO = pickle.HIGHEST_PROTOCOL


def inject_bit_flip(payload: bytes, bit_offset: int | None = None) -> bytes:
    """Flip exactly one bit of ``payload``, deterministically, and return the corrupted bytes.

    A real, reproducible stand-in for a hardware/software bit-flip: this is the corruption
    primitive the acceptance tests inject via :meth:`AuditedMPEncodedData.arm_corruption`.
    ``bit_offset`` defaults to the middle bit of the payload (an arbitrary but fixed choice --
    determinism of the test does not depend on which bit, only that the two payloads it is
    applied to differ afterward).
    """
    if not payload:
        raise ValueError("cannot flip a bit in an empty payload.")
    n_bits = len(payload) * 8
    if bit_offset is None:
        bit_offset = n_bits // 2
    bit_offset = int(bit_offset) % n_bits
    byte_i, bit_i = divmod(bit_offset, 8)
    corrupted = bytearray(payload)
    corrupted[byte_i] ^= 1 << bit_i
    return bytes(corrupted)


def _assert_finite_value(value: Any, where: str) -> None:
    """Recursively walk a (possibly nested list/tuple/dict) accumulator ``value()`` payload and
    raise via :func:`check_finite` at the first non-finite numeric leaf found."""
    if isinstance(value, np.ndarray):
        check_finite(value, where)
    elif isinstance(value, (int, float, np.floating, np.integer)) and not isinstance(value, bool):
        check_finite(np.asarray([float(value)]), where)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _assert_finite_value(v, where)
    elif isinstance(value, dict):
        for v in value.values():
            _assert_finite_value(v, where)
    # else: non-numeric leaf (str, bool, None, categorical support metadata, ...) -- nothing to check.


def finite_guarded_fold(
    estimator: Any, payloads: Sequence[bytes], where: str = "AuditedMPEncodedData.combine"
) -> tuple[float, Any]:
    """``checkpointed_fold``, but check finiteness of the running accumulator immediately after
    EVERY ``combine()`` call (see module docstring for why this is a wrapper around this
    module's own fold loop rather than a change to ``combine()``'s contract)."""
    accumulator = estimator.accumulator_factory().make()
    nobs = 0.0
    for i, raw in enumerate(payloads):
        count, stats = pickle.loads(raw)
        nobs += count
        accumulator.combine(stats)
        _assert_finite_value(
            accumulator.value(), "%s (payload index %d, %d combine() calls so far)" % (where, i, i + 1)
        )
    stats_dict: dict[str, Any] = {}
    accumulator.key_merge(stats_dict)
    accumulator.key_replace(stats_dict)
    return nobs, accumulator.value()


@dataclass
class SDCAuditReceipt:
    """A structured record of one detected primary-vs-audit mismatch: which shard, which two
    ranks, and a summary of what actually differed (never just a bare "mismatch" boolean)."""

    round: int
    shard_id: int
    primary_worker: int
    audit_worker: int
    primary_nbytes: int
    audit_nbytes: int
    first_diff_byte_offset: int | None
    primary_sha256: str
    audit_sha256: str
    primary_value_repr: str = field(default="", repr=False)
    audit_value_repr: str = field(default="", repr=False)

    def summary(self) -> str:
        return (
            "SDC audit mismatch: shard=%d primary_rank=%d audit_rank=%d "
            "primary=%dB (sha256 %s) audit=%dB (sha256 %s) first_diff_byte=%s"
            % (
                self.shard_id,
                self.primary_worker,
                self.audit_worker,
                self.primary_nbytes,
                self.primary_sha256[:12],
                self.audit_nbytes,
                self.audit_sha256[:12],
                self.first_diff_byte_offset,
            )
        )


def _receipt(
    round_no: int, shard_id: int, primary_worker: int, audit_worker: int, primary: bytes, audit: bytes
) -> SDCAuditReceipt:
    n = min(len(primary), len(audit))
    first_diff = next((i for i in range(n) if primary[i] != audit[i]), None)
    if first_diff is None and len(primary) != len(audit):
        first_diff = n

    def _safe_repr(payload: bytes) -> str:
        try:
            return repr(pickle.loads(payload))[:2000]
        except Exception as e:  # a corrupted payload may not even unpickle -- that's fine, still a receipt
            return "<unpicklable: %s>" % e

    return SDCAuditReceipt(
        round=round_no,
        shard_id=shard_id,
        primary_worker=primary_worker,
        audit_worker=audit_worker,
        primary_nbytes=len(primary),
        audit_nbytes=len(audit),
        first_diff_byte_offset=first_diff,
        primary_sha256=hashlib.sha256(primary).hexdigest(),
        audit_sha256=hashlib.sha256(audit).hexdigest(),
        primary_value_repr=_safe_repr(primary),
        audit_value_repr=_safe_repr(audit),
    )


class AuditedMPEncodedData(ResilientMPEncodedData):
    """:class:`ResilientMPEncodedData` (K4) plus a continuous SDC audit (K5).

    Every round (``pysp_seq_estimate`` / ``pysp_stream_accumulate``), a random ``audit_rate``
    fraction of shards are ALSO recomputed on a second, different rank via the same ad hoc
    ``"update_shard"`` wire command K4 already uses for shard recovery (see
    ``ResilientMPEncodedData._recover_shard``) -- no new worker-side machinery. The primary and
    audit ``(count, value())`` payloads are compared BYTE-FOR-BYTE. A mismatch:

        1. is recorded as a structured :class:`SDCAuditReceipt` (``self.audit_receipts``,
           ``self.last_round_audit_mismatches``);
        2. quarantines BOTH ranks that produced the disagreeing payloads via K4's existing
           ``_blacklist`` / ``_retire_worker`` / ``_migrate_shard_permanently`` machinery --
           this is a deliberately conservative policy: a single 2-way mismatch cannot, by
           itself, prove which of the two ranks is the corrupted one (that needs a third
           witness / majority vote), so both are treated as suspect and the shard is migrated
           to a clean survivor rather than risk silently trusting either.

    The main EM round itself (retry, blacklisting on repeated *failure*, checkpointed fold) is
    entirely K4's, untouched, reused via inheritance -- K5 only adds the audit phase (run
    BEFORE the main round each call, so a quarantine decided by the audit is already reflected
    in that same round's live-worker set) and swaps K4's plain ``checkpointed_fold`` for
    :func:`finite_guarded_fold` via the ``_fold_fn`` hook K4 exposes for exactly this purpose.

    Args:
        audit_rate (float): fraction of shards (``0..1``) redundantly recomputed each round.
        rng (np.random.RandomState | None): drives which shards are audited each round and
            which live rank is picked as the second ("audit") rank. Not used for anything that
            needs to be reproducible bit-for-bit across ranks -- only for *which* shards get
            the (always bit-exact) double-check.

    Testing hook:
        :meth:`arm_corruption` registers a one-shot-per-round hook that can mutate a payload
        right after it comes off the wire from an ad hoc ``"update_shard"`` recompute, letting
        a test inject a deterministic, reproducible corruption (see :func:`inject_bit_flip`)
        into a chosen (rank, shard, role) combination without touching worker internals -- the
        same "observe the wire, mutate deterministically" pattern K4's ``arm_kill`` uses.
    """

    def __init__(
        self,
        data: Sequence[Any],
        estimator: Any | None = None,
        encoder: Any | None = None,
        num_workers: int | None = None,
        sub_chunks: int = 1,
        max_retries: int = 2,
        audit_rate: float = 0.1,
        rng: np.random.RandomState | None = None,
    ) -> None:
        super().__init__(
            data,
            estimator=estimator,
            encoder=encoder,
            num_workers=num_workers,
            sub_chunks=sub_chunks,
            max_retries=max_retries,
        )
        if not (0.0 <= float(audit_rate) <= 1.0):
            raise ValueError("audit_rate must be within [0, 1].")
        self.audit_rate = float(audit_rate)
        self._audit_rng = rng if rng is not None else np.random.RandomState()
        self._fold_fn = finite_guarded_fold
        self._corrupt_hook: Any = None
        self._round = 0

        # instrumentation, mirroring K4's last_round_* fields, exposed for tests/operators.
        self.last_round_audited_shards: set[int] = set()
        self.last_round_audit_mismatches: list[SDCAuditReceipt] = []
        self.last_round_audit_eval_count = 0
        self.audit_receipts: list[SDCAuditReceipt] = []

    # -- testing hook ------------------------------------------------------

    def arm_corruption(self, hook: Any) -> None:
        """Register ``hook(worker_id, shard_id, role, payload_bytes) -> payload_bytes``
        (``role`` is ``"primary"`` or ``"audit"``), applied to every ad hoc ``"update_shard"``
        payload this instance receives until cleared. Unlike ``arm_kill`` this is NOT one-shot
        by default (an SDC fault is typically persistent, e.g. a stuck bit in one DIMM) --
        clear it explicitly with ``arm_corruption(None)`` to simulate a transient fault.
        """
        self._corrupt_hook = hook

    # -- audit phase ---------------------------------------------------------

    def _update_shard_on(self, worker_id: int, estimator_b: bytes, model_b: bytes, shard_id: int, role: str) -> bytes:
        """Ad hoc, single-shard recompute on ``worker_id`` -- the exact same wire command K4's
        ``_recover_shard`` uses, just with an explicit (not "lowest live") target rank."""
        self._send_raw(
            worker_id, ("update_shard", estimator_b, model_b, shard_id, self._shard_raw[shard_id], self.sub_chunks)
        )
        self._recv_raw(worker_id)  # "started" ack
        self._send_raw(worker_id, ("go",))
        _, payload = self._recv_raw(worker_id)
        if self._corrupt_hook is not None:
            payload = self._corrupt_hook(worker_id, shard_id, role, payload)
        return payload

    def _quarantine(self, worker_a: int, worker_b: int, shard_id: int) -> None:
        """Blacklist both suspect ranks and permanently migrate whatever they held (see class
        docstring for why both, not just one, are quarantined on a single mismatch)."""
        for w in (worker_a, worker_b):
            if w not in self._conns or w in self._blacklist:
                continue
            self._failures[w] = self.max_retries  # SDC evidence skips the normal retry grace period
            self._blacklist.add(w)
            shard_ids = self._worker_shards.pop(w, set())
            self._retire_worker(w)
            for sid in shard_ids:
                self._migrate_shard_permanently(sid)

    def _audit_shards(
        self, estimator: Any, model: Any, shard_ids: set[int], quarantine_on_mismatch: bool = True
    ) -> None:
        """Redundantly recompute a random ``audit_rate`` fraction of ``shard_ids`` on a second
        rank and bitwise-compare. ``quarantine_on_mismatch=False`` is a testing knob (used by
        the catch-rate measurement in the test suite) to observe many independent corruption
        trials against one live worker pool without each detection permanently shrinking it.
        """
        self._round += 1
        shard_list = sorted(shard_ids)
        n_audit = int(round(self.audit_rate * len(shard_list)))
        n_audit = max(0, min(n_audit, len(shard_list)))

        self.last_round_audited_shards = set()
        self.last_round_audit_mismatches = []
        self.last_round_audit_eval_count = 0

        if n_audit == 0:
            return
        live = sorted(w for w in self._conns if w not in self._blacklist)
        if len(live) < 2:
            return

        estimator_b = pickle.dumps(estimator, protocol=_PROTO)
        model_b = pickle.dumps(model, protocol=_PROTO)
        chosen = sorted(
            int(s) for s in self._audit_rng.choice(np.array(shard_list, dtype=int), size=n_audit, replace=False)
        )

        audited: set[int] = set()
        mismatches: list[SDCAuditReceipt] = []
        eval_count = 0
        for shard_id in chosen:
            live = sorted(w for w in self._conns if w not in self._blacklist)
            if len(live) < 2:
                break
            owner = next((w for w, sids in self._worker_shards.items() if shard_id in sids and w in live), live[0])
            others = [w for w in live if w != owner]
            secondary = others[int(self._audit_rng.randint(len(others)))]

            primary_payload = self._update_shard_on(owner, estimator_b, model_b, shard_id, "primary")
            audit_payload = self._update_shard_on(secondary, estimator_b, model_b, shard_id, "audit")
            eval_count += 2
            audited.add(shard_id)

            if primary_payload != audit_payload:
                receipt = _receipt(self._round, shard_id, owner, secondary, primary_payload, audit_payload)
                mismatches.append(receipt)
                self.audit_receipts.append(receipt)
                if quarantine_on_mismatch:
                    self._quarantine(owner, secondary, shard_id)

        self.last_round_audited_shards = audited
        self.last_round_audit_mismatches = mismatches
        self.last_round_audit_eval_count = eval_count

    # -- protocol recognized by mixle.stats dispatch (audit, then delegate to K4) --------------

    def pysp_seq_estimate(self, estimator: Any, prev_estimate: Any) -> Any:
        self._audit_shards(estimator, prev_estimate, set(self._shard_raw.keys()))
        return super().pysp_seq_estimate(estimator, prev_estimate)

    def pysp_stream_accumulate(self, estimator: Any, model: Any) -> tuple[float, Any]:
        self._audit_shards(estimator, model, set(self._shard_raw.keys()))
        return super().pysp_stream_accumulate(estimator, model)
