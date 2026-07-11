"""Contract-gated low-rank/top-k delta compression with persistent error feedback."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from mixle.experimental.typed_runtime.contracts import MergeLaw, UpdateContract
from mixle.experimental.typed_runtime.proposal import payload_fingerprint


class CompressionMethod(StrEnum):
    """Wire representation selected for one delta."""

    DENSE = "dense"
    LOW_RANK = "low_rank"
    TOPK = "topk"


@dataclass(frozen=True)
class CompressionReceipt:
    """Measured bytes and reconstruction error for one compression action."""

    key: str
    method: CompressionMethod
    input_bytes: int
    payload_bytes: int
    rank_or_nnz: int
    realized_l2_error: float
    relative_l2_error: float
    residual_l2_norm: float
    exact: bool

    @property
    def compression_ratio(self) -> float:
        """Dense bytes divided by transmitted payload bytes."""

        return self.input_bytes / self.payload_bytes if self.payload_bytes else math.inf

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible compression receipt."""

        return {
            "key": self.key,
            "method": self.method.value,
            "input_bytes": self.input_bytes,
            "payload_bytes": self.payload_bytes,
            "rank_or_nnz": self.rank_or_nnz,
            "realized_l2_error": self.realized_l2_error,
            "relative_l2_error": self.relative_l2_error,
            "residual_l2_norm": self.residual_l2_norm,
            "exact": self.exact,
            "compression_ratio": self.compression_ratio,
        }


@dataclass(frozen=True)
class CompressedDelta:
    """Dense, low-rank, or sparse wire payload plus its measured receipt."""

    method: CompressionMethod
    shape: tuple[int, ...]
    dtype: str
    arrays: tuple[np.ndarray, ...]
    receipt: CompressionReceipt

    def reconstruct(self) -> np.ndarray:
        """Reconstruct a dense delta in the original dtype and shape."""

        dtype = np.dtype(self.dtype)
        if self.method is CompressionMethod.DENSE:
            return self.arrays[0].astype(dtype, copy=True).reshape(self.shape)
        if self.method is CompressionMethod.LOW_RANK:
            left, singular, right = self.arrays
            return ((left * singular[None, :]) @ right).astype(dtype, copy=False).reshape(self.shape)
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
            "receipt": self.receipt.as_dict(),
        }


class ErrorFeedbackCompressor:
    """Stateful compressor whose residual buffers survive checkpoint/restart."""

    def __init__(
        self,
        *,
        default_rank: int = 1,
        default_topk_fraction: float = 0.1,
        exact_threshold_bytes: int = 4_096,
    ) -> None:
        if default_rank < 1:
            raise ValueError("default_rank must be positive.")
        if not 0.0 < default_topk_fraction <= 1.0:
            raise ValueError("default_topk_fraction must be in (0, 1].")
        if exact_threshold_bytes < 0:
            raise ValueError("exact_threshold_bytes must be non-negative.")
        self.default_rank = default_rank
        self.default_topk_fraction = default_topk_fraction
        self.exact_threshold_bytes = exact_threshold_bytes
        self._residuals: dict[str, np.ndarray] = {}

    def residual(self, key: str) -> np.ndarray | None:
        """Return a copy of one error-feedback residual."""

        value = self._residuals.get(key)
        return None if value is None else value.copy()

    def _dense(self, key: str, corrected: np.ndarray, input_bytes: int) -> CompressedDelta:
        dense = corrected.copy()
        self._residuals[key] = np.zeros_like(corrected)
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
        )
        return CompressedDelta(CompressionMethod.DENSE, corrected.shape, corrected.dtype.str, (dense,), receipt)

    def compress(
        self,
        key: str,
        delta: Any,
        contract: UpdateContract,
        *,
        rank: int | None = None,
        topk_fraction: float | None = None,
    ) -> CompressedDelta:
        """Compress one finite delta under its typed approximation contract."""

        if not key:
            raise ValueError("compression key must be non-empty.")
        value = np.asarray(delta)
        if value.ndim == 0 or not np.issubdtype(value.dtype, np.number):
            raise TypeError("delta compression requires a non-scalar numeric array.")
        if not np.all(np.isfinite(value)):
            raise ValueError("delta compression requires finite values.")
        if key in self._residuals and self._residuals[key].shape != value.shape:
            raise ValueError("error-feedback residual shape changed for key %s." % key)
        residual = self._residuals.get(key, np.zeros_like(value))
        corrected = value + residual
        input_bytes = int(value.nbytes)
        approximation_allowed = not contract.exact or contract.merge_law in (
            MergeLaw.LOW_RANK,
            MergeLaw.WEIGHTED_SKETCH,
        )
        if not approximation_allowed or input_bytes <= self.exact_threshold_bytes:
            return self._dense(key, corrected, input_bytes)

        if value.ndim == 2:
            selected_rank = min(rank or self.default_rank, min(value.shape))
            if selected_rank < 1:
                raise ValueError("compression rank must be positive.")
            left, singular, right = np.linalg.svd(corrected, full_matrices=False)
            left = left[:, :selected_rank].astype(value.dtype, copy=False)
            singular = singular[:selected_rank].astype(value.dtype, copy=False)
            right = right[:selected_rank, :].astype(value.dtype, copy=False)
            reconstruction = (left * singular[None, :]) @ right
            arrays = (left, singular, right)
            method = CompressionMethod.LOW_RANK
            rank_or_nnz = selected_rank
        else:
            fraction = topk_fraction or self.default_topk_fraction
            if not 0.0 < fraction <= 1.0:
                raise ValueError("topk_fraction must be in (0, 1].")
            flat = corrected.reshape(-1)
            count = max(1, int(math.ceil(flat.size * fraction)))
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
            return self._dense(key, corrected, input_bytes)
        new_residual = corrected - reconstruction
        self._residuals[key] = new_residual
        error = float(np.linalg.norm(new_residual.reshape(-1)))
        denominator = float(np.linalg.norm(corrected.reshape(-1)))
        relative = error / denominator if denominator > 0.0 else 0.0
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
        )
        return CompressedDelta(method, value.shape, value.dtype.str, arrays, receipt)

    def state_dict(self) -> dict[str, Any]:
        """Return checkpointable configuration and residual buffers."""

        return {
            "version": 1,
            "default_rank": self.default_rank,
            "default_topk_fraction": self.default_topk_fraction,
            "exact_threshold_bytes": self.exact_threshold_bytes,
            "residuals": {key: value.copy() for key, value in self._residuals.items()},
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore a validated error-feedback checkpoint."""

        if state.get("version") != 1:
            raise ValueError("unsupported compressor state version.")
        self.default_rank = int(state["default_rank"])
        self.default_topk_fraction = float(state["default_topk_fraction"])
        self.exact_threshold_bytes = int(state["exact_threshold_bytes"])
        residuals = state.get("residuals", {})
        if not isinstance(residuals, dict) or any(not np.all(np.isfinite(value)) for value in residuals.values()):
            raise ValueError("compressor residual state must be a finite array mapping.")
        self._residuals = {str(key): np.asarray(value).copy() for key, value in residuals.items()}


__all__ = ["CompressedDelta", "CompressionMethod", "CompressionReceipt", "ErrorFeedbackCompressor"]
