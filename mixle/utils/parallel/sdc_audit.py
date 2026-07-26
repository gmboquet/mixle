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
DIFFERENT ("audit") rank.

The verdict is deliberately not based on pickle-byte equality. Healthy heterogeneous workers
can differ in serialization order or in the last numerical bits because of CPU/GPU kernels,
BLAS implementations, fused operations, or library versions. Payloads are decoded into a
closed typed statistic, compared under the explicitly configured ``audit_rtol``/``audit_atol``
numeric envelope, and separately canonicalized for exact evidence digests. Every mismatch
receipt records that envelope, maximum observed errors, the typed canonical digests, and the
raw payload digests. Exact raw agreement remains useful evidence but is never assumed to be a
portable proof of worker health.

**NaN/Inf watchdog.** ``mixle.models._neural_serial.check_finite`` already guards individual
density evaluations. K5 extends that same "fail loud, immediately, with the offending
location named" philosophy to the accumulator-fold boundary: :func:`finite_guarded_fold` is a
drop-in replacement for :func:`~mixle.utils.parallel.resilient_em.checkpointed_fold` that
checks finiteness of the running accumulator's ``value()`` immediately after EVERY
``combine()`` call, not just once at the end -- so a NaN/Inf introduced while folding payload
``i`` is caught at payload ``i``'s ``combine()`` boundary, before it silently propagates into
payload ``i+1..n``. The fully tied value is checked again after ``key_merge``/``key_replace``
so shared-key transformations cannot introduce an unchecked invalid state. Scope note: this
wraps operations made by THIS module's own fold loop only -- it deliberately does NOT touch
``SequenceEncodableStatisticAccumulator.combine()``'s contract itself, which every other
caller in the codebase still uses unguarded, exactly as before.
"""

from __future__ import annotations

import hashlib
import pickle
import struct
from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass, field, fields, is_dataclass
from numbers import Complex, Integral, Number, Real
from typing import Any

import numpy as np

from mixle.utils.parallel.resilient_em import ResilientMPEncodedData

__all__ = [
    "AuditedMPEncodedData",
    "SDCAuditReceipt",
    "finite_guarded_fold",
    "inject_bit_flip",
]

_PROTO = pickle.HIGHEST_PROTOCOL


@dataclass(frozen=True)
class _StatisticComparison:
    equivalent: bool
    reason: str
    max_abs_error: float
    max_rel_error: float
    primary_canonical_sha256: str
    audit_canonical_sha256: str


def _canonical_statistic_bytes(value: Any) -> bytes:
    """Canonical typed encoding used for evidence digests, never executable loading."""
    if value is None:
        return b"N"
    if isinstance(value, bool):
        return b"B1" if value else b"B0"
    if isinstance(value, np.generic):
        return _canonical_statistic_bytes(value.item())
    if isinstance(value, Integral):
        raw = str(int(value)).encode("ascii")
        return b"I" + len(raw).to_bytes(8, "big") + raw
    if isinstance(value, Real):
        return b"F" + struct.pack(">d", float(value))
    if isinstance(value, Complex):
        return b"C" + struct.pack(">dd", float(value.real), float(value.imag))
    if isinstance(value, str):
        raw = value.encode("utf-8")
        return b"S" + len(raw).to_bytes(8, "big") + raw
    if isinstance(value, bytes):
        return b"Y" + len(value).to_bytes(8, "big") + value
    if isinstance(value, np.ndarray):
        shape = b",".join(str(int(size)).encode("ascii") for size in value.shape)
        header = value.dtype.str.encode("ascii") + b":" + shape
        if value.dtype.kind == "O":
            body = b"".join(_canonical_statistic_bytes(item) for item in value.flat)
        else:
            body = np.ascontiguousarray(value.astype(value.dtype.newbyteorder(">"), copy=False)).tobytes()
        return b"A" + len(header).to_bytes(8, "big") + header + len(body).to_bytes(8, "big") + body
    if isinstance(value, tuple):
        return b"T" + len(value).to_bytes(8, "big") + b"".join(_canonical_statistic_bytes(item) for item in value)
    if isinstance(value, list):
        return b"L" + len(value).to_bytes(8, "big") + b"".join(_canonical_statistic_bytes(item) for item in value)
    if isinstance(value, dict):
        pairs = sorted(
            (
                _canonical_statistic_bytes(key),
                _canonical_statistic_bytes(item),
            )
            for key, item in value.items()
        )
        return b"D" + len(pairs).to_bytes(8, "big") + b"".join(
            len(key).to_bytes(8, "big") + key + len(item).to_bytes(8, "big") + item for key, item in pairs
        )
    raise TypeError(f"unsupported statistic type {type(value).__module__}.{type(value).__qualname__}")


def _compare_statistic_payloads(primary: bytes, audit: bytes, *, rtol: float, atol: float) -> _StatisticComparison:
    """Compare typed statistics numerically while retaining exact canonical evidence digests."""
    try:
        primary_value = pickle.loads(primary)
        audit_value = pickle.loads(audit)
        primary_canonical = _canonical_statistic_bytes(primary_value)
        audit_canonical = _canonical_statistic_bytes(audit_value)
    except Exception as error:  # noqa: BLE001 - corrupt bytes are themselves an audit mismatch
        return _StatisticComparison(
            False,
            f"payload decode/schema failure: {type(error).__name__}: {error}",
            float("inf"),
            float("inf"),
            hashlib.sha256(b"raw:" + primary).hexdigest(),
            hashlib.sha256(b"raw:" + audit).hexdigest(),
        )

    max_abs_error = 0.0
    max_rel_error = 0.0
    mismatch = ""

    def _compare(left: Any, right: Any, path: str) -> None:
        nonlocal max_abs_error, max_rel_error, mismatch
        if mismatch:
            return
        if isinstance(left, np.generic):
            left = left.item()
        if isinstance(right, np.generic):
            right = right.item()
        if isinstance(left, bool) or isinstance(right, bool):
            if type(left) is not type(right) or left != right:
                mismatch = f"{path}: boolean/schema mismatch"
            return
        if isinstance(left, Integral) and isinstance(right, Integral):
            if int(left) != int(right):
                mismatch = f"{path}: integer mismatch"
            return
        if isinstance(left, Complex) and isinstance(right, Complex):
            left_complex = complex(left)
            right_complex = complex(right)
            if not (
                np.isfinite(left_complex.real)
                and np.isfinite(left_complex.imag)
                and np.isfinite(right_complex.real)
                and np.isfinite(right_complex.imag)
            ):
                mismatch = f"{path}: non-finite numeric value"
                return
            absolute = abs(left_complex - right_complex)
            scale = max(abs(left_complex), abs(right_complex))
            relative = absolute / scale if scale > 0.0 else 0.0
            max_abs_error = max(max_abs_error, float(absolute))
            max_rel_error = max(max_rel_error, float(relative))
            if absolute > atol + rtol * scale:
                mismatch = f"{path}: numeric error exceeds tolerance"
            return
        if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
            if left.shape != right.shape or left.dtype.kind != right.dtype.kind:
                mismatch = f"{path}: array shape/dtype-kind mismatch"
                return
            if left.dtype.kind in "biu":
                if not np.array_equal(left, right):
                    mismatch = f"{path}: exact array mismatch"
                return
            if left.dtype.kind in "fc":
                if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
                    mismatch = f"{path}: non-finite array value"
                    return
                absolute = np.abs(left.astype(np.complex128) - right.astype(np.complex128))
                scale = np.maximum(np.abs(left), np.abs(right))
                if absolute.size:
                    max_abs_error = max(max_abs_error, float(np.max(absolute)))
                    relative = np.divide(absolute, scale, out=np.zeros_like(absolute, dtype=float), where=scale > 0)
                    max_rel_error = max(max_rel_error, float(np.max(relative)))
                if np.any(absolute > atol + rtol * scale):
                    mismatch = f"{path}: array error exceeds tolerance"
                return
            if left.dtype.kind == "O":
                for index, (left_item, right_item) in enumerate(zip(left.flat, right.flat)):
                    _compare(left_item, right_item, f"{path}[{index}]")
                return
            if not np.array_equal(left, right):
                mismatch = f"{path}: array mismatch"
            return
        if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
            if len(left) != len(right):
                mismatch = f"{path}: sequence length mismatch"
                return
            for index, (left_item, right_item) in enumerate(zip(left, right)):
                _compare(left_item, right_item, f"{path}[{index}]")
            return
        if isinstance(left, dict) and isinstance(right, dict):
            if left.keys() != right.keys():
                mismatch = f"{path}: mapping key mismatch"
                return
            for key in left:
                _compare(left[key], right[key], f"{path}[{key!r}]")
            return
        if type(left) is not type(right) or left != right:
            mismatch = f"{path}: value/schema mismatch"

    _compare(primary_value, audit_value, "statistic")
    return _StatisticComparison(
        not mismatch,
        mismatch or "within declared numeric envelope",
        max_abs_error,
        max_rel_error,
        hashlib.sha256(primary_canonical).hexdigest(),
        hashlib.sha256(audit_canonical).hexdigest(),
    )


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


def _assert_finite_value(value: Any, where: str, _seen: set[int] | None = None) -> None:
    """Traverse a typed statistic graph and reject every non-finite numeric leaf."""
    if _seen is None:
        _seen = set()

    def _reject() -> None:
        raise ValueError(f"{where} contains a non-finite statistic (NaN or inf)")

    if value is None or isinstance(value, (str, bytes, bool)):
        return
    if isinstance(value, np.generic):
        _assert_finite_value(value.item(), where, _seen)
        return
    if isinstance(value, Number):
        numeric = complex(value)
        if not np.isfinite(numeric.real) or not np.isfinite(numeric.imag):
            _reject()
        return
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "biufc":
            if not np.all(np.isfinite(value)):
                _reject()
        elif value.dtype.kind == "O":
            identity = id(value)
            if identity in _seen:
                return
            _seen.add(identity)
            for item in value.flat:
                _assert_finite_value(item, where, _seen)
        return
    try:
        import torch

        if torch.is_tensor(value):
            if not bool(torch.all(torch.isfinite(value)).item()):
                _reject()
            return
    except ImportError:  # pragma: no cover - torch is optional
        pass

    identity = id(value)
    if identity in _seen:
        return
    _seen.add(identity)
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_finite_value(key, where, _seen)
            _assert_finite_value(item, where, _seen)
    elif isinstance(value, (Sequence, Set)):
        for item in value:
            _assert_finite_value(item, where, _seen)
    elif is_dataclass(value) and not isinstance(value, type):
        for descriptor in fields(value):
            _assert_finite_value(getattr(value, descriptor.name), where, _seen)
    elif hasattr(value, "__dict__"):
        for item in vars(value).values():
            _assert_finite_value(item, where, _seen)
    else:
        slots = getattr(type(value), "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for name in slots:
            if hasattr(value, name):
                _assert_finite_value(getattr(value, name), where, _seen)


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
        _assert_finite_value(count, "%s (payload index %d count)" % (where, i))
        nobs += count
        _assert_finite_value(nobs, "%s (running observation count)" % where)
        accumulator.combine(stats)
        _assert_finite_value(
            accumulator.value(), "%s (payload index %d, %d combine() calls so far)" % (where, i, i + 1)
        )
    stats_dict: dict[str, Any] = {}
    accumulator.key_merge(stats_dict)
    accumulator.key_replace(stats_dict)
    final_value = accumulator.value()
    _assert_finite_value(final_value, "%s (after key_merge/key_replace)" % where)
    return nobs, final_value


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
    comparison_mode: str = "typed_numeric_envelope/v1"
    audit_rtol: float = 0.0
    audit_atol: float = 0.0
    max_abs_error: float = 0.0
    max_rel_error: float = 0.0
    mismatch_reason: str = ""
    primary_canonical_sha256: str = ""
    audit_canonical_sha256: str = ""
    witness_worker: int | None = None
    suspected_workers: tuple[int, ...] = ()
    quarantine_status: str = "not_requested"
    primary_value_repr: str = field(default="", repr=False)
    audit_value_repr: str = field(default="", repr=False)

    def summary(self) -> str:
        return (
            "SDC audit mismatch: shard=%d primary_rank=%d audit_rank=%d "
            "primary=%dB (sha256 %s) audit=%dB (sha256 %s) first_diff_byte=%s "
            "comparison=%s rtol=%g atol=%g reason=%s suspects=%s quarantine=%s"
            % (
                self.shard_id,
                self.primary_worker,
                self.audit_worker,
                self.primary_nbytes,
                self.primary_sha256[:12],
                self.audit_nbytes,
                self.audit_sha256[:12],
                self.first_diff_byte_offset,
                self.comparison_mode,
                self.audit_rtol,
                self.audit_atol,
                self.mismatch_reason,
                self.suspected_workers,
                self.quarantine_status,
            )
        )


def _receipt(
    round_no: int,
    shard_id: int,
    primary_worker: int,
    audit_worker: int,
    primary: bytes,
    audit: bytes,
    comparison: _StatisticComparison,
    *,
    rtol: float,
    atol: float,
    witness_worker: int | None = None,
    suspected_workers: tuple[int, ...] = (),
    quarantine_status: str = "not_requested",
) -> SDCAuditReceipt:
    n = min(len(primary), len(audit))
    first_diff = next((i for i in range(n) if primary[i] != audit[i]), None)
    if first_diff is None and len(primary) != len(audit):
        first_diff = n

    def _safe_repr(payload: bytes) -> str:
        try:
            return repr(pickle.loads(payload))[:2000]
        except Exception as e:  # noqa: BLE001 - a corrupted payload may not even unpickle; still a receipt
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
        audit_rtol=rtol,
        audit_atol=atol,
        max_abs_error=comparison.max_abs_error,
        max_rel_error=comparison.max_rel_error,
        mismatch_reason=comparison.reason,
        primary_canonical_sha256=comparison.primary_canonical_sha256,
        audit_canonical_sha256=comparison.audit_canonical_sha256,
        witness_worker=witness_worker,
        suspected_workers=suspected_workers,
        quarantine_status=quarantine_status,
        primary_value_repr=_safe_repr(primary),
        audit_value_repr=_safe_repr(audit),
    )


class AuditedMPEncodedData(ResilientMPEncodedData):
    """:class:`ResilientMPEncodedData` (K4) plus a continuous SDC audit (K5).

    Every round (``pysp_seq_estimate`` / ``pysp_stream_accumulate``), a random ``audit_rate``
    fraction of shards are ALSO recomputed on a second, different rank via the same ad hoc
    ``"update_shard"`` wire command K4 already uses for shard recovery (see
    ``ResilientMPEncodedData._recover_shard``) -- no new worker-side machinery. The primary and
    audit ``(count, value())`` payloads are compared as typed statistics under an explicit
    numerical envelope. A mismatch:

        1. is recorded as a structured :class:`SDCAuditReceipt` (``self.audit_receipts``,
           ``self.last_round_audit_mismatches``);
        2. obtains a third independent witness before identifying a suspect. A two-way
           disagreement alone is recorded but never used to quarantine either worker;
        3. stages every affected shard on validated survivors before atomically committing
           the suspect's retirement. If no safe witness or placement exists, quarantine is
           deferred and that decision is recorded in the receipt.

    The main EM round itself (retry, blacklisting on repeated *failure*, checkpointed fold) is
    entirely K4's, untouched, reused via inheritance -- K5 only adds the audit phase (run
    BEFORE the main round each call, so a quarantine decided by the audit is already reflected
    in that same round's live-worker set) and swaps K4's plain ``checkpointed_fold`` for
    :func:`finite_guarded_fold` via the ``_fold_fn`` hook K4 exposes for exactly this purpose.

    Args:
        audit_rate (float): fraction of shards (``0..1``) redundantly recomputed each round.
        rng (np.random.RandomState | None): drives which shards are audited each round and
            which live rank is picked as the second ("audit") rank.
        audit_rtol / audit_atol (float): finite nonnegative tolerances used for floating and
            complex statistic leaves. Integer and categorical leaves remain exact.

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
        audit_rtol: float = 1.0e-12,
        audit_atol: float = 1.0e-12,
        rng: np.random.RandomState | None = None,
        worker_timeout_s: float = 30.0,
    ) -> None:
        if isinstance(audit_rate, bool) or not isinstance(audit_rate, Real) or not np.isfinite(audit_rate):
            raise ValueError("audit_rate must be finite and within [0, 1]")
        if not 0.0 <= audit_rate <= 1.0:
            raise ValueError("audit_rate must be finite and within [0, 1]")
        if (
            isinstance(audit_rtol, bool)
            or not isinstance(audit_rtol, Real)
            or not np.isfinite(audit_rtol)
            or audit_rtol < 0.0
        ):
            raise ValueError("audit_rtol must be finite and nonnegative")
        if (
            isinstance(audit_atol, bool)
            or not isinstance(audit_atol, Real)
            or not np.isfinite(audit_atol)
            or audit_atol < 0.0
        ):
            raise ValueError("audit_atol must be finite and nonnegative")
        super().__init__(
            data,
            estimator=estimator,
            encoder=encoder,
            num_workers=num_workers,
            sub_chunks=sub_chunks,
            max_retries=max_retries,
            worker_timeout_s=worker_timeout_s,
        )
        self.audit_rate = float(audit_rate)
        self.audit_rtol = float(audit_rtol)
        self.audit_atol = float(audit_atol)
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

    def _quarantine(self, suspects: set[int]) -> str:
        """Stage a complete survivor placement, then commit suspect retirement."""
        suspects = {worker for worker in suspects if worker in self._conns and worker not in self._blacklist}
        if not suspects:
            return "not_needed"
        survivors = sorted(worker for worker in self._conns if worker not in suspects and worker not in self._blacklist)
        if not survivors:
            return "deferred:no_safe_survivor"
        assignments = {worker: set(self._worker_shards.get(worker, set())) for worker in suspects}
        shard_ids = sorted(shard for owned in assignments.values() for shard in owned)
        if len(shard_ids) != len(set(shard_ids)):
            return "deferred:ambiguous_shard_ownership"
        placement = {
            shard_id: survivors[index % len(survivors)]
            for index, shard_id in enumerate(shard_ids)
        }
        if any(shard_id in self._worker_shards.get(target, set()) for shard_id, target in placement.items()):
            return "deferred:duplicate_survivor_placement"

        staged: list[tuple[int, int]] = []
        try:
            for shard_id, target in placement.items():
                self._send_raw(target, ("add_shard", shard_id, self._shard_raw[shard_id], self.sub_chunks))
                staged.append((target, shard_id))
                self._recv_raw(target)
        except (EOFError, OSError, RuntimeError, TimeoutError):
            for target, shard_id in staged:
                try:
                    self._send_raw(target, ("remove_shard", shard_id))
                    self._recv_raw(target)
                except (EOFError, OSError, RuntimeError, TimeoutError):
                    pass
            return "deferred:staging_failed"

        for worker in sorted(suspects):
            self._failures[worker] = self.max_retries
            self._blacklist.add(worker)
            self._worker_shards.pop(worker, None)
            self._retire_worker(worker)
        for shard_id, target in placement.items():
            self._worker_shards.setdefault(target, set()).add(shard_id)
        return "committed"

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

            comparison = _compare_statistic_payloads(
                primary_payload,
                audit_payload,
                rtol=self.audit_rtol,
                atol=self.audit_atol,
            )
            if not comparison.equivalent:
                witness_worker: int | None = None
                suspects: tuple[int, ...] = ()
                quarantine_status = "disabled" if not quarantine_on_mismatch else "deferred:insufficient_witnesses"
                if quarantine_on_mismatch:
                    witness_candidates = [worker for worker in live if worker not in {owner, secondary}]
                    if witness_candidates:
                        witness_worker = witness_candidates[int(self._audit_rng.randint(len(witness_candidates)))]
                        witness_payload = self._update_shard_on(
                            witness_worker,
                            estimator_b,
                            model_b,
                            shard_id,
                            "witness",
                        )
                        eval_count += 1
                        primary_witness = _compare_statistic_payloads(
                            primary_payload,
                            witness_payload,
                            rtol=self.audit_rtol,
                            atol=self.audit_atol,
                        )
                        audit_witness = _compare_statistic_payloads(
                            audit_payload,
                            witness_payload,
                            rtol=self.audit_rtol,
                            atol=self.audit_atol,
                        )
                        if primary_witness.equivalent and not audit_witness.equivalent:
                            suspects = (secondary,)
                        elif audit_witness.equivalent and not primary_witness.equivalent:
                            suspects = (owner,)
                        elif primary_witness.equivalent and audit_witness.equivalent:
                            quarantine_status = "deferred:nontransitive_envelope"
                        else:
                            quarantine_status = "deferred:three_way_disagreement"
                        if suspects:
                            quarantine_status = self._quarantine(set(suspects))
                receipt = _receipt(
                    self._round,
                    shard_id,
                    owner,
                    secondary,
                    primary_payload,
                    audit_payload,
                    comparison,
                    rtol=self.audit_rtol,
                    atol=self.audit_atol,
                    witness_worker=witness_worker,
                    suspected_workers=suspects,
                    quarantine_status=quarantine_status,
                )
                mismatches.append(receipt)
                self.audit_receipts.append(receipt)

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
