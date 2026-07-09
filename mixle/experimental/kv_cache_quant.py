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
    "quantize_kv_cache",
    "dequantize_kv_cache",
    "quantize_cluster_outliers",
    "dequantize_cluster_outliers",
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
        scale (float): For int8, ``max(|x|) / 127`` -- ``dequant = codes.float() * scale``. For fp8, always
            ``1.0`` (fp8's own exponent field already spans the input's dynamic range for the K/V magnitudes
            this module targets; see :func:`quantize_kv_cache`'s docstring for the honest caveat about very
            large-magnitude tensors).
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

    fp8: a direct cast to ``torch.float8_e4m3fn`` (4 exponent bits, 3 mantissa bits) and back -- no scale
    computation needed since fp8's floating exponent already tracks the input's dynamic range (unlike int8's
    fixed-point grid). Requires ``torch.float8_e4m3fn`` (torch >= 2.1); raises if unavailable rather than
    silently falling back to int8.

    Returns:
        AffineQuantized
    """
    _require_torch()
    if not torch.is_tensor(x):
        x = torch.as_tensor(x, dtype=torch.float32)
    x = x.float()

    if mode == "int8":
        max_abs = x.abs().max()
        scale = float(max_abs / _INT8_QMAX) if float(max_abs) > 0 else 1.0
        codes = torch.clamp(torch.round(x / scale), -_INT8_QMAX, _INT8_QMAX).to(torch.int8)
        return AffineQuantized(codes=codes, scale=scale, mode="int8")
    if mode == "fp8":
        if not _HAS_FP8:
            raise RuntimeError("This torch build has no torch.float8_e4m3fn; fp8 KV-cache quantization unavailable.")
        codes = x.to(torch.float8_e4m3fn)
        return AffineQuantized(codes=codes, scale=1.0, mode="fp8")
    raise ValueError(f"Unknown quant mode {mode!r}; expected 'int8' or 'fp8'.")


def dequantize_kv_cache(q: AffineQuantized) -> Any:
    """Inverse of :func:`quantize_kv_cache`: returns a float32 tensor, same shape as the original input."""
    _require_torch()
    if q.mode == "int8":
        return q.codes.float() * q.scale
    if q.mode == "fp8":
        return q.codes.float()
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
        outlier_indices (np.ndarray | None): Flat (batch*time) token indices the outliers came from.
        tail_k (SortedProfileEncoding | None): G4 parametric-tail encoding of the non-outlier K population.
        tail_v (SortedProfileEncoding | None): G4 parametric-tail encoding of the non-outlier V population.
    """

    cluster_id: int
    outlier_k: AffineQuantized | None
    outlier_v: AffineQuantized | None
    outlier_indices: np.ndarray | None
    tail_k: SortedProfileEncoding | None
    tail_v: SortedProfileEncoding | None


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
            ``{cluster_id: {"k": (n_out, n_head, d_head), "v": ..., "indices": (n_out,)}}``.
        flat_k, flat_v: The full chunk's ``(b*t, n_head, d_head)`` K/V tensors (the same tensors
            ``birth_and_merge`` computed ``flat_k``/``flat_v`` from) -- used to build the non-outlier tail
            population per cluster (every token NOT in that cluster's ``indices``).
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
    n_tokens = flat_k.shape[0]
    out: dict[int, QuantizedClusterOutliers] = {}
    for cluster_id, payload in per_cluster_outlier_tokens.items():
        indices = payload["indices"]
        idx_np = indices.detach().cpu().numpy() if torch.is_tensor(indices) else np.asarray(indices)

        outlier_k_q = quantize_kv_cache(payload["k"], mode=mode) if payload["k"].numel() > 0 else None
        outlier_v_q = quantize_kv_cache(payload["v"], mode=mode) if payload["v"].numel() > 0 else None

        tail_mask = np.ones(n_tokens, dtype=bool)
        tail_mask[idx_np] = False
        tail_k_vals = flat_k[torch.as_tensor(tail_mask)]
        tail_v_vals = flat_v[torch.as_tensor(tail_mask)]

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
            outlier_indices=idx_np,
            tail_k=tail_k_enc,
            tail_v=tail_v_enc,
        )
    return out


def dequantize_cluster_outliers(q: QuantizedClusterOutliers) -> dict[str, Any]:
    """Inverse of one cluster's :class:`QuantizedClusterOutliers`: returns
    ``{"outlier_k": tensor|None, "outlier_v": tensor|None, "tail_k": np.ndarray|None, "tail_v": np.ndarray|None}``.
    """
    return {
        "outlier_k": dequantize_kv_cache(q.outlier_k) if q.outlier_k is not None else None,
        "outlier_v": dequantize_kv_cache(q.outlier_v) if q.outlier_v is not None else None,
        "tail_k": reconstruct_sorted_profile(q.tail_k) if q.tail_k is not None else None,
        "tail_v": reconstruct_sorted_profile(q.tail_v) if q.tail_v is not None else None,
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
