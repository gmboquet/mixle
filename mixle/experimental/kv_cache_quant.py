"""I2: KV-cache quantization + E2 tails -- int8/fp8 KV for inference, with E2's cluster structure
supplying "quantized exact outliers + G4 parametric tails" for the far-field bank's own outlier bookkeeping.

**What this is.** Two related but separate quantization seams, both built on top of already-existing
mechanisms rather than inventing a new quantizer:

1. :func:`quantize_kv_cache` / :func:`dequantize_kv_cache` -- ordinary affine int8 (or native ``fp8_e4m3``)
   quantization of E1's exact near-field KV window (``SlidingWindowState.cache_k`` / ``cache_v``, or any
   ``(..., d_head)`` K/V tensor at inference time). This is the literal "int8/fp8 KV for inference" half of
   the roadmap card -- a standard per-tensor affine round-trip, not the sorted-profile machinery. It is
   scoped to the near-field window because that is what "the KV cache" means operationally: the thing every
   attention step reads on every token.

2. :func:`quantize_cluster_outliers` -- G4's sorted-profile quantizer (``mixle.models.sorted_profile_quantizer``)
   applied to the E2 ``ClusterBank``'s own outlier/tail bookkeeping. E2 already separates, per cluster, per
   chunk (``birth_and_merge``'s ``receipt["per_cluster_outlier_tokens"]``): tokens whose residual against
   the cluster's Gaussian-affine fit was largest (the ``outlier_top_k`` highest-residual tokens per cluster,
   currently a plain dense fp32 tensor -- E2's own module docstring calls this out as the "I2/G4 storage
   seam", see ``E2_UNAVAILABLE_PIECES["I2/G4"]`` in ``moment_closure_attention.py``). This module closes
   that seam: those flagged outlier tokens are int8-quantized ("quantized exact outliers" -- exact in the
   sense of being carved out and identified individually, not exact in the sense of full float32 precision),
   while the surrounding non-outlier K/V population of the same chunk goes through G4's
   ``fit_sorted_profile`` (head-exact top-k + parametric Gaussian tail fit, its own KS-receipt-gated dense
   fallback) -- the "G4 parametric tails" half of the card.

Both halves reuse existing machinery on purpose: (1) is deliberately NOT routed through G4 (a KS-fit-gated
parametric quantizer is the wrong tool for "quantize this window on every single token" -- it is a
per-tensor batch operation with real fitting cost, appropriate for the once-per-chunk ClusterBank outlier
snapshot in (2), not for a per-step cache write), and (2) is deliberately NOT a new int8 scheme -- it calls
:func:`quantize_kv_cache` for the outlier half and ``mixle.models.sorted_profile_quantizer.fit_sorted_profile``
verbatim for the tail half, so there is exactly one int8 implementation and exactly one parametric-tail
implementation in this codebase, both reused rather than duplicated.

**Honest scope.** fp8 support here is gated on ``torch.float8_e4m3fn`` (available on this environment's
torch 2.12 build, CPU-only -- no fp8 hardware acceleration is claimed or exercised, this is a numerical
round-trip test of the dtype's representable grid, not a throughput benchmark). No custom Triton/CUDA
kernels are written; this module is receipts-and-correctness scoped, not a speed optimization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from mixle.models.sorted_profile_quantizer import (
    SortedProfileEncoding,
    fit_sorted_profile,
)
from mixle.models.sorted_profile_quantizer import (
    reconstruct as reconstruct_sorted_profile,
)

try:
    import torch

    _HAS_TORCH = True
    _HAS_FP8 = hasattr(torch, "float8_e4m3fn")
except ImportError:  # pragma: no cover - torch is optional
    _HAS_TORCH = False
    _HAS_FP8 = False

__all__ = [
    "QuantMode",
    "AffineQuantized",
    "QuantizedClusterOutliers",
    "ClusterTokenConservation",
    "quantize_kv_cache",
    "dequantize_kv_cache",
    "quantize_cluster_outliers",
    "dequantize_cluster_outliers",
    "verify_cluster_token_conservation",
    "quantization_error_per_token",
]

QuantMode = Literal["int8", "fp8"]

# int8 affine quantization uses the full signed range minus one code point (matches the common symmetric
# convention of leaving -128 unused so the zero point is exactly representable and dequant is a single
# multiply, no bias term).
_INT8_QMAX = 127


@dataclass(frozen=True)
class AffineQuantized:
    """Round-tripped quantized tensor: quantized codes plus the (per-tensor) scale needed to dequantize.

    Attributes:
        codes: ``torch.int8`` (int8 mode) or ``torch.float8_e4m3fn`` (fp8 mode) tensor, same shape as the
            input.
        scale (float): Positive finite multiplier used by ``dequant = codes.float() * scale``. Int8 uses
            ``max(|x|) / 127`` (or 1 for an all-zero tensor). FP8 uses 1 when the source fits natively and
            otherwise scales its largest magnitude to the dtype's finite maximum before casting.
        mode: Which quantization scheme produced this.
    """

    codes: Any
    scale: float
    mode: QuantMode


def _require_torch() -> None:
    if not _HAS_TORCH:
        raise ImportError("mixle.experimental.kv_cache_quant requires torch.")


def quantize_kv_cache(x: Any, *, mode: QuantMode = "int8") -> AffineQuantized:
    """Quantize a K or V tensor (any shape, real-valued) to int8 or fp8 for inference-time KV-cache storage.

    int8: symmetric per-tensor affine quantization, ``scale = max(|x|) / 127``, ``codes = round(x / scale)``
    clamped to ``[-127, 127]``. Per-tensor (not per-channel/per-head) scale is the deliberately simple
    baseline this module ships; a per-head scale would shrink error further at the cost of ``n_head`` extra
    floats stored per cache write; the perplexity receipt in ``mixle/tests/kv_cache_quant_test.py`` reports
    the per-tensor baseline honestly rather than tuning against a stronger scheme this module does not
    implement.

    fp8: casts to ``torch.float8_e4m3fn`` (4 exponent bits, 3 mantissa bits). Values outside that dtype's
    finite range are first scaled by ``max(|x|) / finfo.max`` so the cast cannot silently produce NaNs.
    Requires ``torch.float8_e4m3fn`` (torch >= 2.1); raises if unavailable rather than silently falling
    back to int8.

    Returns:
        AffineQuantized
    """
    _require_torch()
    if not torch.is_tensor(x):
        x = torch.as_tensor(x, dtype=torch.float32)
    if x.dtype == torch.bool or x.is_complex():
        raise TypeError("KV-cache quantization requires a real, non-boolean tensor.")
    x = x.float()
    if x.numel() == 0:
        raise ValueError("KV-cache quantization requires a non-empty tensor.")
    if not bool(torch.isfinite(x).all()):
        raise ValueError("KV-cache quantization requires finite tensor values.")

    if mode == "int8":
        max_abs = x.abs().max()
        scale = float(max_abs / _INT8_QMAX) if float(max_abs) > 0 else 1.0
        codes = torch.clamp(torch.round(x / scale), -_INT8_QMAX, _INT8_QMAX).to(torch.int8)
        return AffineQuantized(codes=codes, scale=scale, mode="int8")
    if mode == "fp8":
        if not _HAS_FP8:
            raise RuntimeError("This torch build has no torch.float8_e4m3fn; fp8 KV-cache quantization unavailable.")
        max_abs = float(x.abs().max())
        fp8_max = float(torch.finfo(torch.float8_e4m3fn).max)
        scale = max(max_abs / fp8_max, 1.0)
        codes = (x / scale).to(torch.float8_e4m3fn)
        if not bool(torch.isfinite(codes.float()).all()):
            raise OverflowError("FP8 KV-cache quantization produced non-finite codes after scaling.")
        return AffineQuantized(codes=codes, scale=scale, mode="fp8")
    raise ValueError(f"Unknown quant mode {mode!r}; expected 'int8' or 'fp8'.")


def dequantize_kv_cache(q: AffineQuantized) -> Any:
    """Inverse of :func:`quantize_kv_cache`: returns a float32 tensor, same shape as the original input."""
    _require_torch()
    if not isinstance(q, AffineQuantized):
        raise TypeError("q must be an AffineQuantized value.")
    if not torch.is_tensor(q.codes) or q.codes.numel() == 0:
        raise ValueError("q.codes must be a non-empty tensor.")
    if not np.isfinite(q.scale) or q.scale <= 0:
        raise ValueError("q.scale must be positive and finite.")
    if q.mode == "int8":
        if q.codes.dtype != torch.int8:
            raise TypeError("int8 quantization requires torch.int8 codes.")
        return q.codes.float() * q.scale
    if q.mode == "fp8":
        if not _HAS_FP8 or q.codes.dtype != torch.float8_e4m3fn:
            raise TypeError("fp8 quantization requires torch.float8_e4m3fn codes.")
        return q.codes.float() * q.scale
    raise ValueError(f"Unknown quant mode {q.mode!r}")


@dataclass
class QuantizedClusterOutliers:
    """Storage format for one ``birth_and_merge`` chunk's per-cluster outlier tokens (E2's "I2/G4 storage
    seam", see ``moment_closure_attention.E2_UNAVAILABLE_PIECES["I2/G4"]``): the flagged outlier tokens'
    K/V get :func:`quantize_kv_cache`'d ("quantized exact outliers" -- exact positions, quantized values);
    the surrounding non-outlier chunk population gets G4's :func:`~mixle.models.sorted_profile_quantizer.fit_sorted_profile`
    ("G4 parametric tails").

    Attributes:
        cluster_id (int): Which live cluster slot this chunk's outliers/tail came from.
        outlier_k (AffineQuantized | None): Quantized exact K values of the flagged outlier tokens.
        outlier_v (AffineQuantized | None): Quantized exact V values of the flagged outlier tokens.
        member_indices: Every flat token index assigned to this cluster.
        outlier_indices: The cluster members stored individually.
        tail_indices: The remaining cluster members stored in the tail encoding.
        tail_k (SortedProfileEncoding | None): G4 parametric-tail encoding of the non-outlier K population.
        tail_v (SortedProfileEncoding | None): G4 parametric-tail encoding of the non-outlier V population.
    """

    cluster_id: int
    outlier_k: AffineQuantized | None
    outlier_v: AffineQuantized | None
    member_indices: np.ndarray
    outlier_indices: np.ndarray
    tail_indices: np.ndarray
    tail_k: SortedProfileEncoding | None
    tail_v: SortedProfileEncoding | None


@dataclass(frozen=True)
class ClusterTokenConservation:
    """Verified token-count receipt for one clustered chunk."""

    input_count: int
    member_count: int
    outlier_count: int
    tail_count: int

    @property
    def conserved(self) -> bool:
        return (
            self.input_count == self.member_count
            and self.member_count == self.outlier_count + self.tail_count
        )


def _token_indices(value: Any, *, name: str, n_tokens: int, device: Any) -> Any:
    if torch.is_tensor(value):
        if value.dtype == torch.bool or value.is_floating_point() or value.is_complex():
            raise TypeError(f"{name} must contain exact integer indices.")
        indices = value.to(device=device, dtype=torch.long)
    else:
        array = np.asarray(value)
        if array.dtype.kind not in {"i", "u"} or array.dtype.kind == "b":
            raise TypeError(f"{name} must contain exact integer indices.")
        indices = torch.as_tensor(array, device=device, dtype=torch.long)
    if indices.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if indices.numel() and (
        bool((indices < 0).any())
        or bool((indices >= n_tokens).any())
        or int(torch.unique(indices).numel()) != int(indices.numel())
    ):
        raise ValueError(f"{name} must contain unique indices in [0, {n_tokens}).")
    return indices


def verify_cluster_token_conservation(
    quantized: dict[int, QuantizedClusterOutliers],
    n_tokens: int,
) -> ClusterTokenConservation:
    """Verify that cluster members partition the chunk and each member occurs in exactly one storage tier."""
    if isinstance(n_tokens, (bool, np.bool_)) or not isinstance(n_tokens, (int, np.integer)) or n_tokens <= 0:
        raise ValueError("n_tokens must be a positive exact integer.")
    if not isinstance(quantized, dict) or not quantized:
        raise ValueError("quantized clusters must be a non-empty dictionary.")

    all_members: list[int] = []
    all_outliers: list[int] = []
    all_tails: list[int] = []
    for cluster_id, cluster in quantized.items():
        if not isinstance(cluster_id, (int, np.integer)) or isinstance(cluster_id, (bool, np.bool_)):
            raise TypeError("cluster ids must be exact integers.")
        if not isinstance(cluster, QuantizedClusterOutliers) or int(cluster_id) != cluster.cluster_id:
            raise ValueError("cluster dictionary keys must match QuantizedClusterOutliers.cluster_id.")
        raw_indices = (
            ("member", np.asarray(cluster.member_indices)),
            ("outlier", np.asarray(cluster.outlier_indices)),
            ("tail", np.asarray(cluster.tail_indices)),
        )
        if any(indices.dtype.kind not in {"i", "u"} or indices.dtype.kind == "b" for _, indices in raw_indices):
            raise TypeError(f"cluster {cluster_id} token indices must contain exact integers.")
        members, outliers, tails = (indices.astype(np.int64, copy=False) for _, indices in raw_indices)
        for name, indices in (("member", members), ("outlier", outliers), ("tail", tails)):
            if indices.ndim != 1 or np.unique(indices).size != indices.size:
                raise ValueError(f"cluster {cluster_id} {name} indices must be a unique one-dimensional array.")
            if indices.size and (indices.min() < 0 or indices.max() >= n_tokens):
                raise ValueError(f"cluster {cluster_id} {name} indices are outside the input chunk.")
        if set(members.tolist()) != set(outliers.tolist()) | set(tails.tolist()):
            raise ValueError(f"cluster {cluster_id} outlier and tail indices do not cover its members.")
        if set(outliers.tolist()) & set(tails.tolist()):
            raise ValueError(f"cluster {cluster_id} outlier and tail indices overlap.")

        expected_outliers = len(outliers)
        for name, encoded in (("outlier_k", cluster.outlier_k), ("outlier_v", cluster.outlier_v)):
            actual = 0 if encoded is None else int(encoded.codes.shape[0])
            if actual != expected_outliers:
                raise ValueError(f"cluster {cluster_id} {name} stores {actual} tokens, expected {expected_outliers}.")
        expected_tails = len(tails)
        for name, encoded in (("tail_k", cluster.tail_k), ("tail_v", cluster.tail_v)):
            actual = 0 if encoded is None else int(encoded.shape[0])
            if actual != expected_tails:
                raise ValueError(f"cluster {cluster_id} {name} stores {actual} tokens, expected {expected_tails}.")

        all_members.extend(members.tolist())
        all_outliers.extend(outliers.tolist())
        all_tails.extend(tails.tolist())

    if len(all_members) != n_tokens or sorted(all_members) != list(range(n_tokens)):
        raise ValueError("cluster memberships must form an exact partition of the input token indices.")
    receipt = ClusterTokenConservation(
        input_count=int(n_tokens),
        member_count=len(all_members),
        outlier_count=len(all_outliers),
        tail_count=len(all_tails),
    )
    if not receipt.conserved:
        raise ValueError("cluster token counts are not conserved.")
    return receipt


def quantize_cluster_outliers(
    per_cluster_outlier_tokens: dict,
    flat_k: Any,
    flat_v: Any,
    *,
    mode: QuantMode = "int8",
    tail_family: Any = None,
    tail_top_k: int = 0,
) -> dict[int, QuantizedClusterOutliers]:
    """Close E2's I2/G4 storage seam for one ``birth_and_merge`` chunk.

    Args:
        per_cluster_outlier_tokens: ``birth_and_merge``'s ``receipt["per_cluster_outlier_tokens"]`` --
            ``{cluster_id: {"k": ..., "v": ..., "indices": ..., "member_indices": ...}}``.
        flat_k, flat_v: The full chunk's ``(b*t, n_head, d_head)`` K/V tensors (the same tensors
            ``birth_and_merge`` computed ``flat_k``/``flat_v`` from) -- used to build the non-outlier tail
            population per cluster (members assigned to that cluster but not in its outlier ``indices``).
        mode: Quantization mode for the outlier half (see :func:`quantize_kv_cache`).
        tail_family: Passed through to :func:`~mixle.models.sorted_profile_quantizer.fit_sorted_profile`
            for the tail half (default ``GaussianEstimator()``).
        tail_top_k: Head-exact top-k within the tail fit itself (default 0 -- the "head-exact" carve-out is
            already handled by this function's own outlier/tail split, so G4's internal top-k defaults off
            to avoid double-carving the same outliers twice).

    Returns:
        dict[int, QuantizedClusterOutliers], keyed by cluster id.
    """
    _require_torch()
    if not torch.is_tensor(flat_k) or not torch.is_tensor(flat_v):
        raise TypeError("flat_k and flat_v must be torch tensors.")
    if flat_k.ndim < 1 or flat_k.shape != flat_v.shape or flat_k.shape[0] <= 0:
        raise ValueError("flat_k and flat_v must have the same non-empty token-leading shape.")
    if flat_k.device != flat_v.device:
        raise ValueError("flat_k and flat_v must be on the same device.")
    if not bool(torch.isfinite(flat_k).all()) or not bool(torch.isfinite(flat_v).all()):
        raise ValueError("flat_k and flat_v must contain only finite values.")
    if not isinstance(per_cluster_outlier_tokens, dict) or not per_cluster_outlier_tokens:
        raise ValueError("per_cluster_outlier_tokens must be a non-empty dictionary.")
    n_tokens = flat_k.shape[0]
    out: dict[int, QuantizedClusterOutliers] = {}
    for cluster_id, payload in per_cluster_outlier_tokens.items():
        if not isinstance(payload, dict):
            raise TypeError(f"cluster {cluster_id} payload must be a dictionary.")
        member_indices = _token_indices(
            payload.get("member_indices"),
            name=f"cluster {cluster_id} member_indices",
            n_tokens=n_tokens,
            device=flat_k.device,
        )
        outlier_indices = _token_indices(
            payload.get("indices"),
            name=f"cluster {cluster_id} indices",
            n_tokens=n_tokens,
            device=flat_k.device,
        )
        member_mask = torch.zeros(n_tokens, dtype=torch.bool, device=flat_k.device)
        member_mask[member_indices] = True
        if outlier_indices.numel() and not bool(member_mask[outlier_indices].all()):
            raise ValueError(f"cluster {cluster_id} outlier indices must be cluster members.")
        tail_mask = member_mask.clone()
        tail_mask[outlier_indices] = False
        tail_indices = torch.nonzero(tail_mask, as_tuple=False).flatten()

        payload_k = payload.get("k")
        payload_v = payload.get("v")
        if not torch.is_tensor(payload_k) or not torch.is_tensor(payload_v):
            raise TypeError(f"cluster {cluster_id} outlier k/v values must be torch tensors.")
        expected_shape = (int(outlier_indices.numel()), *flat_k.shape[1:])
        if payload_k.shape != expected_shape or payload_v.shape != expected_shape:
            raise ValueError(f"cluster {cluster_id} outlier k/v values must have shape {expected_shape}.")
        if payload_k.numel() and (
            not bool(torch.isfinite(payload_k).all()) or not bool(torch.isfinite(payload_v).all())
        ):
            raise ValueError(f"cluster {cluster_id} outlier k/v values must be finite.")
        expected_k = flat_k[outlier_indices]
        expected_v = flat_v[outlier_indices]
        if not torch.equal(payload_k.to(device=flat_k.device, dtype=flat_k.dtype), expected_k) or not torch.equal(
            payload_v.to(device=flat_v.device, dtype=flat_v.dtype), expected_v
        ):
            raise ValueError(f"cluster {cluster_id} outlier k/v values do not match their source indices.")

        outlier_k_q = quantize_kv_cache(expected_k, mode=mode) if outlier_indices.numel() else None
        outlier_v_q = quantize_kv_cache(expected_v, mode=mode) if outlier_indices.numel() else None

        tail_k_vals = flat_k[tail_indices]
        tail_v_vals = flat_v[tail_indices]

        tail_k_enc = (
            fit_sorted_profile(tail_k_vals, top_k=tail_top_k, tail_family=tail_family)
            if tail_k_vals.numel() >= 2
            else None
        )
        tail_v_enc = (
            fit_sorted_profile(tail_v_vals, top_k=tail_top_k, tail_family=tail_family)
            if tail_v_vals.numel() >= 2
            else None
        )

        out[cluster_id] = QuantizedClusterOutliers(
            cluster_id=cluster_id,
            outlier_k=outlier_k_q,
            outlier_v=outlier_v_q,
            member_indices=member_indices.detach().cpu().numpy(),
            outlier_indices=outlier_indices.detach().cpu().numpy(),
            tail_indices=tail_indices.detach().cpu().numpy(),
            tail_k=tail_k_enc,
            tail_v=tail_v_enc,
        )
    verify_cluster_token_conservation(out, n_tokens)
    return out


def dequantize_cluster_outliers(q: QuantizedClusterOutliers) -> dict[str, Any]:
    """Inverse of one cluster's :class:`QuantizedClusterOutliers`.

    The returned dictionary includes the member, outlier, and tail token indices needed to place each
    reconstructed population without duplicating tokens from other clusters.
    """
    return {
        "outlier_k": dequantize_kv_cache(q.outlier_k) if q.outlier_k is not None else None,
        "outlier_v": dequantize_kv_cache(q.outlier_v) if q.outlier_v is not None else None,
        "tail_k": reconstruct_sorted_profile(q.tail_k) if q.tail_k is not None else None,
        "tail_v": reconstruct_sorted_profile(q.tail_v) if q.tail_v is not None else None,
        "member_indices": q.member_indices.copy(),
        "outlier_indices": q.outlier_indices.copy(),
        "tail_indices": q.tail_indices.copy(),
    }


def quantization_error_per_token(x: Any, *, mode: QuantMode = "int8") -> Any:
    """Per-token (leading-axis) mean absolute quantize/dequantize round-trip error of ``x`` under
    :func:`quantize_kv_cache` -- ``x``: ``(n_tokens, ...)``, returns ``(n_tokens,)``.

    Used by the receipt-correlation acceptance test (roadmap I2, "receipt correlation inside E2") to ask
    whether E2's own per-cluster misfit signal lines up with where naive KV quantization error is largest.
    """
    _require_torch()
    if not torch.is_tensor(x):
        x = torch.as_tensor(x, dtype=torch.float32)
    q = quantize_kv_cache(x, mode=mode)
    recon = dequantize_kv_cache(q)
    err = (x.float() - recon).abs()
    reduce_dims = tuple(range(1, err.dim()))
    return err.mean(dim=reduce_dims) if reduce_dims else err
