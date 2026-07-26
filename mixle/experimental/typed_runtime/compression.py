"""Explicit exact/approximate delta transport with acknowledged error feedback."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from numbers import Integral, Real
from typing import Any

import numpy as np

from mixle.experimental.typed_runtime.contracts import UpdateContract
from mixle.experimental.typed_runtime.proposal import payload_fingerprint


class CompressionMethod(StrEnum):
    """Wire representation selected for one delta."""

    DENSE = "dense"
    LOW_RANK = "low_rank"
    TOPK = "topk"


@dataclass(frozen=True)
class CompressionReceipt:
    """Measured bytes/error and authorization for one staged transport payload."""

    key: str
    method: CompressionMethod
    input_bytes: int
    payload_bytes: int
    rank_or_nnz: int
    realized_l2_error: float
    relative_l2_error: float
    pending_residual_l2_norm: float
    exact: bool
    approximation_authorized: bool
    maximum_relative_l2_error: float | None

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key:
            raise ValueError("compression receipts require a non-empty key.")
        if not isinstance(self.method, CompressionMethod):
            raise TypeError("compression receipt method must be CompressionMethod.")
        for name in ("input_bytes", "payload_bytes", "rank_or_nnz"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f"compression receipt {name} must be a non-negative integer.")
        for name in ("realized_l2_error", "relative_l2_error", "pending_residual_l2_norm"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                or value < 0.0
            ):
                raise ValueError(f"compression receipt {name} must be finite and non-negative.")
        if not isinstance(self.exact, bool) or not isinstance(self.approximation_authorized, bool):
            raise TypeError("compression receipt exact/authorization fields must be boolean.")
        if self.maximum_relative_l2_error is not None and (
            isinstance(self.maximum_relative_l2_error, bool)
            or not isinstance(self.maximum_relative_l2_error, Real)
            or not math.isfinite(float(self.maximum_relative_l2_error))
            or self.maximum_relative_l2_error < 0.0
        ):
            raise ValueError("maximum_relative_l2_error must be finite and non-negative.")
        if self.exact and (
            self.realized_l2_error != 0.0
            or self.relative_l2_error != 0.0
            or self.pending_residual_l2_norm != 0.0
        ):
            raise ValueError("exact compression receipts cannot report reconstruction error or residual.")
        if not self.exact and (
            not self.approximation_authorized
            or self.maximum_relative_l2_error is None
            or self.relative_l2_error > self.maximum_relative_l2_error
        ):
            raise ValueError("lossy compression requires explicit authorization and a satisfied error budget.")

    @property
    def compression_ratio(self) -> float:
        """Dense bytes divided by transmitted payload bytes."""

        return self.input_bytes / self.payload_bytes if self.payload_bytes else math.inf

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible staged-compression receipt."""

        return {
            "key": self.key,
            "method": self.method.value,
            "input_bytes": self.input_bytes,
            "payload_bytes": self.payload_bytes,
            "rank_or_nnz": self.rank_or_nnz,
            "realized_l2_error": self.realized_l2_error,
            "relative_l2_error": self.relative_l2_error,
            "pending_residual_l2_norm": self.pending_residual_l2_norm,
            "exact": self.exact,
            "approximation_authorized": self.approximation_authorized,
            "maximum_relative_l2_error": self.maximum_relative_l2_error,
            "compression_ratio": self.compression_ratio,
            "residual_state": "pending_until_acknowledged",
        }


@dataclass(frozen=True)
class CompressedDelta:
    """Immutable wire payload whose residual transition is pending acknowledgement."""

    method: CompressionMethod
    shape: tuple[int, ...]
    dtype: str
    arrays: tuple[np.ndarray, ...]
    receipt: CompressionReceipt
    feedback_token: str

    def __post_init__(self) -> None:
        if not isinstance(self.method, CompressionMethod) or self.method is not self.receipt.method:
            raise ValueError("compressed delta method must match its receipt.")
        if (
            not isinstance(self.shape, tuple)
            or not self.shape
            or any(isinstance(size, bool) or not isinstance(size, Integral) or size < 1 for size in self.shape)
        ):
            raise ValueError("compressed delta shape must contain positive integer dimensions.")
        try:
            dtype = np.dtype(self.dtype)
        except (TypeError, ValueError) as error:
            raise ValueError("compressed delta dtype is invalid.") from error
        if not isinstance(self.arrays, tuple) or not self.arrays or any(
            not isinstance(array, np.ndarray) for array in self.arrays
        ):
            raise TypeError("compressed delta arrays must be a non-empty tuple of ndarrays.")
        if not isinstance(self.feedback_token, str) or not self.feedback_token:
            raise ValueError("compressed delta requires a feedback acknowledgement token.")
        if any(array.flags.writeable for array in self.arrays):
            raise ValueError("compressed delta arrays must be immutable.")
        expected_arrays = (
            1 if self.method is CompressionMethod.DENSE else 3 if self.method is CompressionMethod.LOW_RANK else 2
        )
        if len(self.arrays) != expected_arrays:
            raise ValueError("compressed delta array count does not match its method.")
        elements = math.prod(self.shape)
        if self.receipt.input_bytes != elements * dtype.itemsize:
            raise ValueError("compressed delta input byte count does not match shape and dtype.")
        if self.receipt.payload_bytes != sum(array.nbytes for array in self.arrays):
            raise ValueError("compressed delta payload byte count does not match its arrays.")
        if self.method is CompressionMethod.DENSE:
            if not self.receipt.exact or self.arrays[0].size != elements:
                raise ValueError("dense compressed deltas must contain one exact full-size array.")
        elif self.receipt.exact or not (
            np.issubdtype(dtype, np.floating) or np.issubdtype(dtype, np.complexfloating)
        ):
            raise ValueError("lossy compressed deltas require a floating or complex source dtype.")

    def reconstruct(self) -> np.ndarray:
        """Reconstruct a dense delta in the declared dtype and shape."""

        dtype = np.dtype(self.dtype)
        if self.method is CompressionMethod.DENSE:
            return self.arrays[0].astype(dtype, copy=True).reshape(self.shape)
        if self.method is CompressionMethod.LOW_RANK:
            left, singular, right = self.arrays
            reconstructed = (left * singular[None, :]) @ right
            return reconstructed.astype(dtype, copy=False).reshape(self.shape)
        indices, values = self.arrays
        flat = np.zeros(int(np.prod(self.shape)), dtype=dtype)
        flat[indices.astype(np.int64)] = values.astype(dtype, copy=False)
        return flat.reshape(self.shape)

    @property
    def payload_hash(self) -> str:
        """Deterministic transport fingerprint."""

        return payload_fingerprint((self.method.value, self.shape, self.dtype, self.arrays))

    def as_dict(self) -> dict[str, Any]:
        """Return metadata without serializing array contents."""

        return {
            "method": self.method.value,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "payload_hash": self.payload_hash,
            "feedback_token": self.feedback_token,
            "receipt": self.receipt.as_dict(),
        }


@dataclass(frozen=True)
class CompressionAcknowledgement:
    """Committed or rejected residual transition for one delivered payload."""

    key: str
    feedback_token: str
    payload_hash: str
    applied: bool
    residual_l2_norm_before: float
    residual_l2_norm_after: float

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value for value in (self.key, self.feedback_token, self.payload_hash)
        ):
            raise ValueError("compression acknowledgements require complete non-empty identity.")
        if not isinstance(self.applied, bool):
            raise TypeError("compression acknowledgement applied must be boolean.")
        for name in ("residual_l2_norm_before", "residual_l2_norm_after"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                or value < 0.0
            ):
                raise ValueError(f"compression acknowledgement {name} must be finite and non-negative.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "feedback_token": self.feedback_token,
            "payload_hash": self.payload_hash,
            "applied": self.applied,
            "residual_l2_norm_before": self.residual_l2_norm_before,
            "residual_l2_norm_after": self.residual_l2_norm_after,
        }


@dataclass(frozen=True)
class _PendingResidual:
    feedback_token: str
    payload_hash: str
    next_residual: np.ndarray


class ErrorFeedbackCompressor:
    """Compressor whose residual changes only after acknowledged payload application."""

    def __init__(
        self,
        *,
        default_rank: int = 1,
        default_topk_fraction: float = 0.1,
        exact_threshold_bytes: int = 4_096,
    ) -> None:
        if isinstance(default_rank, bool) or not isinstance(default_rank, Integral) or default_rank < 1:
            raise ValueError("default_rank must be a positive integer.")
        if (
            isinstance(default_topk_fraction, bool)
            or not isinstance(default_topk_fraction, Real)
            or not math.isfinite(float(default_topk_fraction))
            or not 0.0 < default_topk_fraction <= 1.0
        ):
            raise ValueError("default_topk_fraction must be finite and in (0, 1].")
        if (
            isinstance(exact_threshold_bytes, bool)
            or not isinstance(exact_threshold_bytes, Integral)
            or exact_threshold_bytes < 0
        ):
            raise ValueError("exact_threshold_bytes must be a non-negative integer.")
        self.default_rank = int(default_rank)
        self.default_topk_fraction = float(default_topk_fraction)
        self.exact_threshold_bytes = int(exact_threshold_bytes)
        self._residuals: dict[str, np.ndarray] = {}
        self._attempts: dict[str, int] = {}
        self._pending: dict[str, _PendingResidual] = {}

    def residual(self, key: str) -> np.ndarray | None:
        """Return a copy of one committed error-feedback residual."""

        value = self._residuals.get(key)
        return None if value is None else value.copy()

    @staticmethod
    def _norm(value: np.ndarray | None) -> float:
        return 0.0 if value is None else float(np.linalg.norm(value.reshape(-1)))

    def _stage(
        self,
        *,
        key: str,
        method: CompressionMethod,
        value: np.ndarray,
        arrays: tuple[np.ndarray, ...],
        receipt: CompressionReceipt,
        next_residual: np.ndarray,
    ) -> CompressedDelta:
        immutable_arrays = []
        for array in arrays:
            detached = np.asarray(array).copy()
            detached.setflags(write=False)
            immutable_arrays.append(detached)
        arrays = tuple(immutable_arrays)
        attempt = self._attempts.get(key, 0) + 1
        self._attempts[key] = attempt
        payload_hash = payload_fingerprint((method.value, value.shape, value.dtype.str, arrays))
        token = payload_fingerprint((key, attempt, payload_hash))
        delta = CompressedDelta(method, value.shape, value.dtype.str, arrays, receipt, token)
        self._pending[key] = _PendingResidual(token, payload_hash, np.asarray(next_residual).copy())
        return delta

    def _dense(
        self,
        key: str,
        corrected: np.ndarray,
        input_bytes: int,
        *,
        approximation_authorized: bool,
        maximum_relative_l2_error: float | None,
    ) -> CompressedDelta:
        dense = corrected.copy()
        receipt = CompressionReceipt(
            key,
            CompressionMethod.DENSE,
            input_bytes,
            dense.nbytes,
            int(dense.size),
            0.0,
            0.0,
            0.0,
            True,
            approximation_authorized,
            maximum_relative_l2_error,
        )
        return self._stage(
            key=key,
            method=CompressionMethod.DENSE,
            value=corrected,
            arrays=(dense,),
            receipt=receipt,
            next_residual=np.zeros_like(corrected),
        )

    def compress(
        self,
        key: str,
        delta: Any,
        contract: UpdateContract,
        *,
        allow_approximation: bool = False,
        maximum_relative_l2_error: float | None = None,
        rank: int | None = None,
        topk_fraction: float | None = None,
    ) -> CompressedDelta:
        """Stage one payload without changing committed residual state.

        Lossy transport requires ``allow_approximation=True``, a finite error budget, and a
        non-exact update contract. Merge-law metadata never authorizes approximation. Call
        :meth:`acknowledge` after delivery/application; until then another payload for ``key``
        is refused.
        """

        if not isinstance(key, str) or not key:
            raise ValueError("compression key must be non-empty.")
        if not isinstance(contract, UpdateContract):
            raise TypeError("compression requires an UpdateContract.")
        if not isinstance(allow_approximation, bool):
            raise TypeError("allow_approximation must be boolean.")
        if key in self._pending:
            raise RuntimeError("compression key has an unacknowledged payload.")
        value = np.asarray(delta)
        if value.ndim == 0 or not np.issubdtype(value.dtype, np.number) or np.issubdtype(value.dtype, np.bool_):
            raise TypeError("delta compression requires a non-scalar numeric array.")
        if not np.all(np.isfinite(value)):
            raise ValueError("delta compression requires finite values.")
        if key in self._residuals and self._residuals[key].shape != value.shape:
            raise ValueError("error-feedback residual shape changed for key %s." % key)
        residual = self._residuals.get(key, np.zeros_like(value))
        corrected = value + residual
        input_bytes = int(value.nbytes)

        if not allow_approximation:
            if maximum_relative_l2_error is not None:
                raise ValueError("maximum_relative_l2_error requires allow_approximation=True.")
            return self._dense(
                key,
                corrected,
                input_bytes,
                approximation_authorized=False,
                maximum_relative_l2_error=None,
            )
        if contract.exact:
            raise ValueError("an exact update contract forbids lossy transport.")
        if (
            isinstance(maximum_relative_l2_error, bool)
            or not isinstance(maximum_relative_l2_error, Real)
            or not math.isfinite(float(maximum_relative_l2_error))
            or maximum_relative_l2_error < 0.0
        ):
            raise ValueError("lossy transport requires a finite non-negative maximum_relative_l2_error.")

        # Integer factorization would round SVD factors/top-k corrections back into the source
        # dtype. Use exact bytes instead; approximation is supported only for floating/complex
        # numeric payloads.
        if np.issubdtype(value.dtype, np.integer) or input_bytes <= self.exact_threshold_bytes:
            return self._dense(
                key,
                corrected,
                input_bytes,
                approximation_authorized=True,
                maximum_relative_l2_error=float(maximum_relative_l2_error),
            )

        if value.ndim == 2:
            selected_rank = self.default_rank if rank is None else rank
            if isinstance(selected_rank, bool) or not isinstance(selected_rank, Integral) or selected_rank < 1:
                raise ValueError("compression rank must be a positive integer.")
            selected_rank = min(int(selected_rank), min(value.shape))
            left, singular, right = np.linalg.svd(corrected, full_matrices=False)
            left = left[:, :selected_rank].astype(value.dtype, copy=False)
            singular = singular[:selected_rank].astype(value.real.dtype, copy=False)
            right = right[:selected_rank, :].astype(value.dtype, copy=False)
            reconstruction = (left * singular[None, :]) @ right
            arrays = (left, singular, right)
            method = CompressionMethod.LOW_RANK
            rank_or_nnz = selected_rank
        else:
            fraction = self.default_topk_fraction if topk_fraction is None else topk_fraction
            if (
                isinstance(fraction, bool)
                or not isinstance(fraction, Real)
                or not math.isfinite(float(fraction))
                or not 0.0 < fraction <= 1.0
            ):
                raise ValueError("topk_fraction must be finite and in (0, 1].")
            flat = corrected.reshape(-1)
            count = max(1, int(math.ceil(flat.size * float(fraction))))
            selected = np.argpartition(np.abs(flat), -count)[-count:]
            selected = np.sort(selected.astype(np.int64))
            values = flat[selected].copy()
            reconstruction = np.zeros_like(flat)
            reconstruction[selected] = values
            reconstruction = reconstruction.reshape(value.shape)
            arrays = (selected, values)
            method = CompressionMethod.TOPK
            rank_or_nnz = count

        payload_bytes = sum(array.nbytes for array in arrays)
        if payload_bytes >= input_bytes:
            return self._dense(
                key,
                corrected,
                input_bytes,
                approximation_authorized=True,
                maximum_relative_l2_error=float(maximum_relative_l2_error),
            )
        next_residual = corrected - reconstruction
        error = float(np.linalg.norm(next_residual.reshape(-1)))
        denominator = float(np.linalg.norm(corrected.reshape(-1)))
        relative = error / denominator if denominator > 0.0 else 0.0
        if relative > maximum_relative_l2_error:
            return self._dense(
                key,
                corrected,
                input_bytes,
                approximation_authorized=True,
                maximum_relative_l2_error=float(maximum_relative_l2_error),
            )
        receipt = CompressionReceipt(
            key,
            method,
            input_bytes,
            payload_bytes,
            rank_or_nnz,
            error,
            relative,
            error,
            False,
            True,
            float(maximum_relative_l2_error),
        )
        return self._stage(
            key=key,
            method=method,
            value=value,
            arrays=arrays,
            receipt=receipt,
            next_residual=next_residual,
        )

    def acknowledge(self, delta: CompressedDelta, *, applied: bool) -> CompressionAcknowledgement:
        """Commit a staged residual only after the payload was applied, or discard it on failure."""

        if not isinstance(delta, CompressedDelta):
            raise TypeError("acknowledge requires a CompressedDelta.")
        if not isinstance(applied, bool):
            raise TypeError("acknowledgement applied must be boolean.")
        key = delta.receipt.key
        pending = self._pending.get(key)
        if pending is None or pending.feedback_token != delta.feedback_token:
            raise ValueError("compressed delta is not the current pending payload for its key.")
        current_hash = delta.payload_hash
        if current_hash != pending.payload_hash:
            raise ValueError("compressed payload changed before acknowledgement.")
        before = self._norm(self._residuals.get(key))
        if applied:
            self._residuals[key] = pending.next_residual.copy()
        del self._pending[key]
        after = self._norm(self._residuals.get(key))
        return CompressionAcknowledgement(key, delta.feedback_token, current_hash, applied, before, after)

    def state_dict(self) -> dict[str, Any]:
        """Return checkpointable committed state; in-flight transitions must be resolved first."""

        if self._pending:
            raise RuntimeError("cannot checkpoint compressor with unacknowledged payloads.")
        return {
            "version": 2,
            "default_rank": self.default_rank,
            "default_topk_fraction": self.default_topk_fraction,
            "exact_threshold_bytes": self.exact_threshold_bytes,
            "attempts": dict(self._attempts),
            "residuals": {key: value.copy() for key, value in self._residuals.items()},
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Atomically restore validated committed error-feedback state."""

        if not isinstance(state, dict) or state.get("version") != 2:
            raise ValueError("unsupported compressor state version.")
        candidate = ErrorFeedbackCompressor(
            default_rank=state.get("default_rank"),
            default_topk_fraction=state.get("default_topk_fraction"),
            exact_threshold_bytes=state.get("exact_threshold_bytes"),
        )
        attempts = state.get("attempts")
        residuals = state.get("residuals")
        if (
            not isinstance(attempts, dict)
            or any(
                not isinstance(key, str)
                or not key
                or isinstance(value, bool)
                or not isinstance(value, Integral)
                or value < 0
                for key, value in attempts.items()
            )
            or not isinstance(residuals, dict)
            or any(not isinstance(key, str) or not key for key in residuals)
        ):
            raise ValueError("compressor attempts/residuals must be valid keyed mappings.")
        restored_residuals: dict[str, np.ndarray] = {}
        for key, value in residuals.items():
            array = np.asarray(value)
            if (
                array.ndim == 0
                or not np.issubdtype(array.dtype, np.number)
                or np.issubdtype(array.dtype, np.bool_)
                or not np.all(np.isfinite(array))
            ):
                raise ValueError("compressor residual state must contain finite non-scalar numeric arrays.")
            restored_residuals[key] = array.copy()
        self.default_rank = candidate.default_rank
        self.default_topk_fraction = candidate.default_topk_fraction
        self.exact_threshold_bytes = candidate.exact_threshold_bytes
        self._attempts = {key: int(value) for key, value in attempts.items()}
        self._residuals = restored_residuals
        self._pending = {}


__all__ = [
    "CompressedDelta",
    "CompressionAcknowledgement",
    "CompressionMethod",
    "CompressionReceipt",
    "ErrorFeedbackCompressor",
]
