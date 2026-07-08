"""Memory efficiency for training state (roadmap F6): fp8 hardening + compressed optimizer
moments + a per-block selective activation-recompute policy.

Three pieces, per the roadmap spec:

1. **fp8 hardening** (:func:`fp8_cast_with_guard`) -- the existing fp8 mention in this codebase
   (``mixle/utils/parallel/torch_neural.py``'s ``precision`` docstring: ``"fp32"|"bf16" (fp8 =
   torchao, vendored)``) is, honestly, just a comment: no fp8 code path actually exists there yet.
   "Hardening" here means building the real thing with real edge-case handling -- using torch's
   NATIVE ``float8_e4m3fn``/``float8_e5m2`` dtypes (no torchao dependency needed for the guard
   logic itself) plus explicit overflow detection (values exceeding the format's representable
   range, which fp8 hardware silently clamps to +/-inf rather than raising), underflow detection
   (fp8's tiny dynamic range flushing a large fraction of small-but-nonzero values to zero), and a
   graceful fallback to a wider dtype (bf16/fp32) whenever either guard fires -- rather than
   accepting a silently-corrupted fp8 tensor.

2. **Optimizer-state compression** (:class:`CompressedOptimizerState`, :class:`CompressedAdam`) --
   Adam-style optimizers keep two moment buffers (``m``, ``v``) at the SAME size as the parameters
   they track (states are 2x params fp32, per the F6 spec). This directly reuses G4
   (``mixle.models.sorted_profile_quantizer.fit_sorted_profile``/``reconstruct``), which was
   explicitly built with "optimizer states (F6)" as its first named use case, as one compression
   path, PLUS a simpler/cheaper 8-bit blockwise quantization path (the standard bitsandbytes-style
   8-bit-Adam technique) as a fast alternative when G4's fuller (and more expensive to fit) sorted-
   profile machinery is not worth its cost. :func:`choose_compression_method` picks between them
   per tensor using a goodness-of-fit-based rule (reusing G4's own KS-statistic receipt for the G4
   path, and a real reconstruction-error check for the int8 path), with a DENSE fallback -- mirroring
   G4's own "receipt-driven dense fallback" pattern -- when neither compressed representation is
   trustworthy for a given (possibly adversarial) tensor.

3. **Selective activation-recompute policy** (:class:`SelectiveRecomputePolicy`) -- extends
   ``mixle.models.transformer.CausalLM``'s previously all-or-nothing ``gradient_checkpointing``
   bool flag to a PER-BLOCK decision, using a real cost model: a block's stored-activation memory
   footprint (the benefit of recomputing it instead) versus its recompute FLOP cost. This mirrors
   the cost/benefit-tradeoff SHAPE of D6's compile economics
   (``mixle.inference.backend_respecialization.estimate_compile_cost``/``estimate_compile_benefit``,
   PR #153) -- a fixed/proxy-unit cost estimate compared against a fixed/proxy-unit benefit estimate,
   with a net-benefit-positive gate -- applied here to a different decision (recompute vs. store,
   not eager vs. compiled). D6 itself lives on a separate, not-yet-merged branch
   (``backend-respecialization``) so it is not imported here; the pattern is reapplied standalone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mixle.models.sorted_profile_quantizer import (
    DEFAULT_GOF_THRESHOLD,
    SortedProfileEncoding,
    fit_sorted_profile,
)
from mixle.models.sorted_profile_quantizer import (
    reconstruct as _g4_reconstruct,
)

try:
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - torch is optional
    _HAS_TORCH = False

__all__ = [
    "fp8_cast_with_guard",
    "Fp8CastResult",
    "quantize_int8_blockwise",
    "dequantize_int8_blockwise",
    "CompressedMomentEncoding",
    "choose_compression_method",
    "compress_moment",
    "decompress_moment",
    "CompressedOptimizerState",
    "CompressedAdam",
    "RecomputeDecision",
    "SelectiveRecomputePolicy",
    "estimate_block_activation_bytes",
    "estimate_block_recompute_flops",
    "estimate_recompute_benefit",
    "estimate_recompute_cost",
]

# ---------------------------------------------------------------------------------------------
# 1. fp8 hardening
# ---------------------------------------------------------------------------------------------

# Representable-magnitude ceilings for torch's two native fp8 formats (IEEE-ish, finite-only for
# e4m3fn). Values above these are NOT raised on cast -- fp8 hardware/software casts silently clamp
# to +/-inf (e5m2, which has an inf encoding) or to NaN (e4m3fn, which has no inf encoding and
# reserves its would-be-inf bit pattern for NaN) -- so an explicit pre-cast magnitude check is the
# only way to catch this before it silently corrupts a tensor.
_FP8_FORMAT_MAX = {
    "float8_e4m3fn": 448.0,
    "float8_e5m2": 57344.0,
}

_DEFAULT_UNDERFLOW_FRACTION_THRESHOLD = 0.5


@dataclass(frozen=True)
class Fp8CastResult:
    """Receipt of one :func:`fp8_cast_with_guard` call -- never silently swallowed.

    Attributes:
        tensor: The output tensor -- either the fp8-cast tensor (``used_fp8=True``) or the
            ``fallback_dtype`` cast (``used_fp8=False``).
        used_fp8 (bool): Whether the fp8 cast was accepted.
        reason (str): Human-readable reason for the decision (acceptance or the specific guard
            that fired).
        max_abs (float): The input tensor's max absolute value (0.0 for an empty tensor) -- the
            statistic the overflow guard checks.
        underflow_fraction (float): Fraction of the input's nonzero values that flushed to exactly
            zero under the fp8 round-trip -- the statistic the underflow guard checks. ``0.0`` when
            the overflow guard fired first (the round-trip was never attempted).
    """

    tensor: Any
    used_fp8: bool
    reason: str
    max_abs: float
    underflow_fraction: float


def fp8_cast_with_guard(
    tensor: Any,
    fp8_dtype: str = "float8_e4m3fn",
    fallback_dtype: Any = None,
    underflow_fraction_threshold: float = _DEFAULT_UNDERFLOW_FRACTION_THRESHOLD,
) -> Fp8CastResult:
    """Attempt an fp8 cast of ``tensor``, guarded against the two ways fp8 silently corrupts data.

    Args:
        tensor: A torch tensor (any float dtype).
        fp8_dtype (str): ``"float8_e4m3fn"`` (higher precision, smaller range -- the default,
            matching common fp8-training practice for weights/activations) or ``"float8_e5m2"``
            (wider range, coarser precision).
        fallback_dtype: Dtype to fall back to when a guard fires. Defaults to ``torch.bfloat16``
            (the codebase's existing wide-range training dtype, per
            ``mixle.utils.parallel.torch_neural``).
        underflow_fraction_threshold (float): Maximum tolerable fraction of nonzero input values
            that are allowed to flush to exactly zero under the fp8 round-trip before the underflow
            guard fires.

    Returns:
        Fp8CastResult: always carries the real, computed guard statistics, whether or not the fp8
        cast was ultimately accepted -- consistent with this codebase's receipt-over-silent-
        assumption convention (see :mod:`mixle.models.sorted_profile_quantizer`'s
        goodness-of-fit receipt).
    """
    if not _HAS_TORCH:
        raise ImportError("fp8_cast_with_guard requires torch.")
    if fallback_dtype is None:
        fallback_dtype = torch.bfloat16

    if not hasattr(torch, fp8_dtype):
        return Fp8CastResult(
            tensor=tensor.to(fallback_dtype),
            used_fp8=False,
            reason=f"torch build lacks native {fp8_dtype!r} -- falling back to {fallback_dtype}",
            max_abs=0.0,
            underflow_fraction=0.0,
        )
    torch_fp8_dtype = getattr(torch, fp8_dtype)

    finite_mask = torch.isfinite(tensor)
    if not bool(finite_mask.all()):
        return Fp8CastResult(
            tensor=tensor.to(fallback_dtype),
            used_fp8=False,
            reason="input already contains non-finite (inf/nan) values -- refusing to fp8-cast",
            max_abs=float("inf"),
            underflow_fraction=0.0,
        )

    max_abs = float(tensor.detach().abs().max().item()) if tensor.numel() else 0.0
    fmt_max = _FP8_FORMAT_MAX[fp8_dtype]
    if max_abs > fmt_max:
        return Fp8CastResult(
            tensor=tensor.to(fallback_dtype),
            used_fp8=False,
            reason=f"max abs value {max_abs:.6g} exceeds {fp8_dtype} format max {fmt_max:.6g} -- overflow risk",
            max_abs=max_abs,
            underflow_fraction=0.0,
        )

    cast = tensor.to(torch_fp8_dtype)
    round_tripped = cast.to(tensor.dtype)

    nonzero_before = tensor != 0
    n_nonzero = int(nonzero_before.sum().item())
    if n_nonzero > 0:
        flushed = int(((round_tripped == 0) & nonzero_before).sum().item())
        underflow_fraction = flushed / n_nonzero
    else:
        underflow_fraction = 0.0

    if underflow_fraction > underflow_fraction_threshold:
        return Fp8CastResult(
            tensor=tensor.to(fallback_dtype),
            used_fp8=False,
            reason=(
                f"{underflow_fraction:.1%} of nonzero values flushed to zero under the fp8 round-trip "
                f"(threshold {underflow_fraction_threshold:.1%}) -- underflow risk"
            ),
            max_abs=max_abs,
            underflow_fraction=underflow_fraction,
        )

    if not bool(torch.isfinite(round_tripped).all()):
        return Fp8CastResult(
            tensor=tensor.to(fallback_dtype),
            used_fp8=False,
            reason="fp8 round-trip produced non-finite values despite passing the magnitude guard",
            max_abs=max_abs,
            underflow_fraction=underflow_fraction,
        )

    return Fp8CastResult(
        tensor=cast,
        used_fp8=True,
        reason="fp8 cast accepted -- within representable range, underflow within tolerance",
        max_abs=max_abs,
        underflow_fraction=underflow_fraction,
    )


# ---------------------------------------------------------------------------------------------
# 2. Optimizer-state compression: G4 (sorted-profile) + 8-bit blockwise moments
# ---------------------------------------------------------------------------------------------

# Blockwise int8 quantization is the fast, cheap alternative to G4's full sorted-profile fit
# (which pays a real cost: a distribution fit via mixle.inference.estimate plus a KS goodness-of-
# fit test, both O(n log n) or worse). Per-block dynamic scaling (rather than one global scale)
# bounds the damage a single outlier block can do to the rest of the tensor's resolution -- the
# same reasoning bitsandbytes' 8-bit optimizers use.
_DEFAULT_INT8_BLOCK_SIZE = 2048

# Below this element count, G4's fixed per-tensor overhead (a distribution fit, a KS test, a
# permutation array) is not worth paying relative to a tensor this small -- int8 (or dense) is
# cheaper and, at this size, comparably accurate.
_DEFAULT_MIN_SIZE_FOR_G4 = 4096

# Fraction of a tensor's elements carved out as G4's head-exact top-k outliers when this module
# picks the G4 path -- kept small and proportional (rather than a fixed count) so it scales with
# tensor size while staying negligible relative to the tail.
_DEFAULT_G4_TOP_K_FRACTION = 0.001

# Above this relative-L2 reconstruction error, an int8 quantization is treated as untrustworthy
# and the dense fallback engages -- mirroring G4's own receipt-driven dense-fallback pattern, just
# with a reconstruction-error receipt instead of a KS-statistic receipt. On its own this catches
# global magnitude blowups, but NOT a single-outlier-crushes-its-block failure mode: when one
# extreme value sets a block's scale, the outlier itself still reconstructs almost exactly (its
# own quantization error is tiny relative to ITS OWN magnitude), so it also dominates the
# tensor-wide L2 norm and can mask a large fraction of the block's other values being flushed to
# zero. :data:`_DEFAULT_INT8_FLUSHED_FRACTION_THRESHOLD` (below) is the second, independent guard
# that catches exactly that case -- the same "fraction of nonzero values flushed to zero" idea
# :func:`fp8_cast_with_guard` already uses for its own underflow guard, reused here.
_DEFAULT_INT8_ADVERSARIAL_RELATIVE_ERROR = 0.5

# Above this fraction of nonzero input values reconstructing as exactly zero, an int8
# quantization is treated as untrustworthy regardless of the global relative-L2 error (see the
# note above) -- e.g. a block dominated by one extreme outlier that flushes most of the rest of
# the block to zero while barely moving the tensor-wide L2 norm.
_DEFAULT_INT8_FLUSHED_FRACTION_THRESHOLD = 0.3


def _flatten_to_numpy(tensor: Any) -> np.ndarray:
    if hasattr(tensor, "detach"):  # torch.Tensor
        return tensor.detach().cpu().numpy().reshape(-1).astype(np.float64)
    return np.asarray(tensor).reshape(-1).astype(np.float64)


def _tensor_shape(tensor: Any) -> tuple:
    if hasattr(tensor, "shape"):
        return tuple(tensor.shape)
    return (np.asarray(tensor).size,)


def quantize_int8_blockwise(
    flat: np.ndarray, block_size: int = _DEFAULT_INT8_BLOCK_SIZE
) -> tuple[np.ndarray, np.ndarray]:
    """Dynamic per-block symmetric int8 quantization (bitsandbytes-style 8-bit-Adam technique).

    Each contiguous block of ``block_size`` elements gets its own scale (``absmax / 127``), so one
    extreme value only degrades the resolution of ITS OWN block rather than the whole tensor.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(codes, scales)`` -- ``codes`` is ``int8`` and the same
        length as ``flat``; ``scales`` is one ``float32`` per block.
    """
    flat = np.asarray(flat, dtype=np.float64).reshape(-1)
    n = flat.size
    block_size = max(int(block_size), 1)
    n_blocks = int(np.ceil(n / block_size)) if n else 0
    codes = np.zeros(n, dtype=np.int8)
    scales = np.zeros(n_blocks, dtype=np.float32)
    for b in range(n_blocks):
        lo, hi = b * block_size, min((b + 1) * block_size, n)
        block = flat[lo:hi]
        absmax = float(np.abs(block).max()) if block.size else 0.0
        scale = absmax / 127.0 if absmax > 0 else 1.0
        scales[b] = scale
        codes[lo:hi] = np.clip(np.round(block / scale), -127, 127).astype(np.int8)
    return codes, scales


def dequantize_int8_blockwise(
    codes: np.ndarray, scales: np.ndarray, block_size: int = _DEFAULT_INT8_BLOCK_SIZE
) -> np.ndarray:
    """Invert :func:`quantize_int8_blockwise`. Returns a ``float32`` array the same length as ``codes``."""
    codes = np.asarray(codes)
    n = codes.size
    block_size = max(int(block_size), 1)
    out = np.zeros(n, dtype=np.float32)
    for b, scale in enumerate(scales):
        lo, hi = b * block_size, min((b + 1) * block_size, n)
        out[lo:hi] = codes[lo:hi].astype(np.float32) * float(scale)
    return out


@dataclass
class CompressedMomentEncoding:
    """Storage format for ONE moment tensor (``m`` or ``v``), compressed by exactly one of the
    three available methods -- only the fields for ``method`` are populated, mirroring
    :class:`~mixle.models.sorted_profile_quantizer.SortedProfileEncoding`'s single-active-branch
    convention.

    Attributes:
        method (str): ``"g4"`` (sorted-profile, :mod:`mixle.models.sorted_profile_quantizer`),
            ``"int8"`` (blockwise quantization), or ``"dense"`` (fallback -- neither compressed
            representation was trustworthy for this tensor).
        shape (tuple[int, ...]): Original tensor shape.
        g4_encoding: Populated iff ``method == "g4"``.
        int8_codes / int8_scales / int8_block_size: Populated iff ``method == "int8"``.
        dense_values: Populated iff ``method == "dense"``.
    """

    method: str
    shape: tuple
    g4_encoding: SortedProfileEncoding | None = None
    int8_codes: np.ndarray | None = None
    int8_scales: np.ndarray | None = None
    int8_block_size: int = _DEFAULT_INT8_BLOCK_SIZE
    dense_values: np.ndarray | None = None

    def nbytes(self) -> int:
        """Measured storage footprint, in bytes -- delegates to G4's own receipt-carrying
        ``nbytes()`` for the G4 branch; computes int8/dense directly."""
        if self.method == "g4":
            return self.g4_encoding.nbytes()
        if self.method == "int8":
            return int(self.int8_codes.nbytes + self.int8_scales.astype(np.float32).nbytes)
        return int(self.dense_values.astype(np.float32).nbytes)


def choose_compression_method(
    tensor: Any,
    *,
    min_size_for_g4: int = _DEFAULT_MIN_SIZE_FOR_G4,
    g4_top_k_fraction: float = _DEFAULT_G4_TOP_K_FRACTION,
    g4_gof_threshold: float = DEFAULT_GOF_THRESHOLD,
    tail_family: Any = None,
) -> str:
    """Goodness-of-fit-based per-tensor method picker (the "one level down" method-picker pattern
    I1 uses for its own picker -- ``mixle.task.bandit`` is a reasonable fit for a LEARNED, reward-
    driven picker, but the choice here is cheaper and just as principled as a fixed rule: G4 already
    computes a real KS goodness-of-fit receipt as part of fitting, so reusing THAT receipt directly
    is simpler than bolting on a bandit that would need to learn what the receipt already tells us
    for free).

    Rule: for tensors at or above ``min_size_for_g4`` (below which G4's fixed per-tensor fitting
    overhead is not worth it), attempt a G4 fit; if it does not dense-fall-back on its own
    goodness-of-fit receipt, use G4 (the more accurate, more expensive path). Otherwise, use int8
    (the cheap path) -- callers should still check :func:`compress_moment`'s returned
    ``method`` for a possible further downgrade to ``"dense"`` if int8 itself proves untrustworthy
    for this specific tensor (see :data:`_DEFAULT_INT8_ADVERSARIAL_RELATIVE_ERROR`).
    """
    flat = _flatten_to_numpy(tensor)
    n = flat.size
    if n >= min_size_for_g4:
        top_k = max(0, int(g4_top_k_fraction * n))
        g4_encoding = fit_sorted_profile(tensor, top_k=top_k, tail_family=tail_family, gof_threshold=g4_gof_threshold)
        if not g4_encoding.used_dense_fallback:
            return "g4"
    return "int8"


def compress_moment(
    tensor: Any,
    method: str = "auto",
    *,
    min_size_for_g4: int = _DEFAULT_MIN_SIZE_FOR_G4,
    g4_top_k_fraction: float = _DEFAULT_G4_TOP_K_FRACTION,
    g4_gof_threshold: float = DEFAULT_GOF_THRESHOLD,
    int8_block_size: int = _DEFAULT_INT8_BLOCK_SIZE,
    int8_adversarial_relative_error: float = _DEFAULT_INT8_ADVERSARIAL_RELATIVE_ERROR,
    int8_flushed_fraction_threshold: float = _DEFAULT_INT8_FLUSHED_FRACTION_THRESHOLD,
    tail_family: Any = None,
) -> CompressedMomentEncoding:
    """Compress one Adam moment tensor (``m`` or ``v``) via G4, int8, or dense storage.

    Args:
        tensor: A torch tensor or numpy array (one Adam moment buffer, any shape).
        method (str): ``"auto"`` (use :func:`choose_compression_method`), ``"g4"``, ``"int8"``, or
            ``"dense"`` to force a specific path. A forced ``"g4"``/``"int8"`` still honestly
            downgrades to ``"dense"`` if the chosen method's own receipt (G4's KS statistic, or
            int8's reconstruction error) rejects the fit -- this function never returns a silently
            bad compressed representation.

    Returns:
        CompressedMomentEncoding
    """
    flat = _flatten_to_numpy(tensor)
    shape = _tensor_shape(tensor)

    chosen = method
    if chosen == "auto":
        chosen = choose_compression_method(
            tensor,
            min_size_for_g4=min_size_for_g4,
            g4_top_k_fraction=g4_top_k_fraction,
            g4_gof_threshold=g4_gof_threshold,
            tail_family=tail_family,
        )

    if chosen == "g4":
        top_k = max(0, int(g4_top_k_fraction * flat.size))
        g4_encoding = fit_sorted_profile(tensor, top_k=top_k, tail_family=tail_family, gof_threshold=g4_gof_threshold)
        if g4_encoding.used_dense_fallback:
            # G4's own receipt rejected the fit -- honor it rather than force a bad G4 encoding.
            return CompressedMomentEncoding(method="dense", shape=shape, dense_values=flat.astype(np.float32))
        return CompressedMomentEncoding(method="g4", shape=shape, g4_encoding=g4_encoding)

    if chosen == "int8":
        codes, scales = quantize_int8_blockwise(flat, block_size=int8_block_size)
        recon = dequantize_int8_blockwise(codes, scales, block_size=int8_block_size)
        denom = float(np.linalg.norm(flat))
        rel_error = float(np.linalg.norm(recon - flat) / denom) if denom > 0 else 0.0
        nonzero_before = flat != 0
        n_nonzero = int(nonzero_before.sum())
        flushed_fraction = float(((recon == 0) & nonzero_before).sum()) / n_nonzero if n_nonzero > 0 else 0.0
        if rel_error > int8_adversarial_relative_error or flushed_fraction > int8_flushed_fraction_threshold:
            # Adversarial tensor for blockwise int8 -- either a global magnitude blowup (caught by
            # rel_error) or a single-outlier-dominated block flushing most of its OTHER values to
            # zero while barely moving the tensor-wide L2 norm (caught by flushed_fraction) -- fall
            # back to dense.
            return CompressedMomentEncoding(method="dense", shape=shape, dense_values=flat.astype(np.float32))
        return CompressedMomentEncoding(
            method="int8",
            shape=shape,
            int8_codes=codes,
            int8_scales=scales.astype(np.float32),
            int8_block_size=int8_block_size,
        )

    return CompressedMomentEncoding(method="dense", shape=shape, dense_values=flat.astype(np.float32))


def decompress_moment(encoding: CompressedMomentEncoding) -> np.ndarray:
    """Invert :func:`compress_moment`. Returns a ``float32`` array reshaped to ``encoding.shape``."""
    if encoding.method == "g4":
        return _g4_reconstruct(encoding.g4_encoding)
    if encoding.method == "int8":
        flat = dequantize_int8_blockwise(encoding.int8_codes, encoding.int8_scales, encoding.int8_block_size)
        return flat.reshape(encoding.shape)
    return encoding.dense_values.reshape(encoding.shape)


class CompressedOptimizerState:
    """Adam-style ``m``/``v`` moment storage for ONE parameter tensor, held COMPRESSED between
    optimizer steps (rather than as two dense fp32 buffers).

    ``set`` compresses; ``get`` decompresses back to plain tensors for the optimizer math to use.

    ``v`` (Adam's second moment, an EMA of squared gradients) is stored as its COMPRESSED SQUARE
    ROOT rather than compressed directly. This is a real, load-bearing design choice, not
    cosmetic: ``v`` routinely spans many orders of magnitude within one parameter tensor (a few
    large-gradient elements next to many near-zero ones), which is exactly the dynamic range that
    defeats both compression paths -- linear int8 quantization sets one block-wide scale from the
    block's max, so any element more than ~127x smaller than that max quantizes to LITERALLY zero;
    G4's Gaussian tail fit is symmetric and can reconstruct small true values as slightly negative.
    Either failure, fed straight into Adam's ``1/(sqrt(v_hat) + eps)`` denominator, produces a
    step blown up by orders of magnitude (empirically confirmed: an early version of this module,
    compressing ``v`` directly with int8, diverged to a >100x loss spike within ~5 steps on the
    tiny transformer this module's own test suite trains). ``sqrt(v)`` roughly halves the dynamic
    range in log-space (Adam's own second-moment buffer already tracks squared gradients FOR this
    reason -- ``sqrt(v)`` is the RMS gradient-magnitude scale, the quantity Adam's update actually
    normalizes by), and squaring the reconstructed value back on :meth:`get` is a nonnegative
    projection for free -- it can never reconstruct a negative ``v``, closing the negative-v/NaN
    failure mode without a separate clamp needing to paper over it (a defensive ``clamp_min(0.0)``
    is still applied as a last line of defense in :class:`CompressedAdam`, since callers of
    ``CompressedOptimizerState`` directly should not have to know this internal detail to stay safe).
    """

    def __init__(self, shape: tuple, method: str = "auto", **compress_kwargs: Any) -> None:
        self.shape = tuple(shape)
        self.method = method
        self._compress_kwargs = compress_kwargs
        self._m: CompressedMomentEncoding | None = None
        self._v: CompressedMomentEncoding | None = None

    def set(self, m: Any, v: Any) -> None:
        self._m = compress_moment(m, method=self.method, **self._compress_kwargs)
        sqrt_v = v.clamp_min(0.0).sqrt() if hasattr(v, "clamp_min") else np.sqrt(np.clip(v, 0.0, None))
        self._v = compress_moment(sqrt_v, method=self.method, **self._compress_kwargs)

    def get(self, device: Any = None, dtype: Any = None) -> tuple[Any, Any]:
        if self._m is None or self._v is None:
            raise RuntimeError("CompressedOptimizerState.get() called before set()")
        if not _HAS_TORCH:
            raise ImportError("CompressedOptimizerState.get requires torch.")
        m = torch.from_numpy(decompress_moment(self._m))
        sqrt_v = torch.from_numpy(decompress_moment(self._v))
        v = sqrt_v * sqrt_v  # nonnegative by construction, regardless of any reconstruction noise
        if dtype is not None:
            m, v = m.to(dtype), v.to(dtype)
        if device is not None:
            m, v = m.to(device), v.to(device)
        return m, v

    def nbytes(self) -> int:
        """Total measured compressed footprint of both moment buffers, in bytes."""
        m_bytes = self._m.nbytes() if self._m is not None else 0
        v_bytes = self._v.nbytes() if self._v is not None else 0
        return m_bytes + v_bytes

    @property
    def methods(self) -> tuple[str | None, str | None]:
        """``(m_method, v_method)`` -- the ACTUAL method used for each buffer (post any honest
        downgrade-to-dense), not just the one requested."""
        return (
            self._m.method if self._m is not None else None,
            self._v.method if self._v is not None else None,
        )


if _HAS_TORCH:

    class CompressedAdam(torch.optim.Optimizer):
        """Adam whose per-parameter ``m``/``v`` moment buffers are held COMPRESSED
        (:class:`CompressedOptimizerState`) between steps, instead of as two dense fp32 buffers --
        the "optimizer-state compression" half of F6.

        Mirrors the well-known 8-bit-Adam pattern (bitsandbytes): decompress -> take the exact Adam
        update in the parameter's own dtype -> recompress. The per-step update math is byte-for-byte
        standard Adam; only the AT-REST storage between steps differs, so :class:`CompressedAdam`
        with ``compression_method="dense"`` is Adam with no approximation at all (a useful sanity
        check, exercised by the loss-parity test).

        Honest cost note: this reference implementation recompresses BOTH moment buffers every
        step, including (for ``compression_method="g4"``/``"auto"``) re-fitting G4's distribution
        and KS test every step -- far more compute than a real deployment would spend (a production
        system would compress at a coarser cadence, e.g. only on optimizer-state checkpoint/
        offload, not every step). Nothing here changes the per-step Adam math itself; only the
        wall-clock cost of this particular reference cadence differs from a production one.
        """

        def __init__(
            self,
            params: Any,
            lr: float = 1e-3,
            betas: tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-8,
            weight_decay: float = 0.0,
            compression_method: str = "auto",
            **compress_kwargs: Any,
        ) -> None:
            defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
            super().__init__(params, defaults)
            self.compression_method = compression_method
            self._compress_kwargs = compress_kwargs

        @torch.no_grad()
        def step(self, closure: Any = None) -> Any:
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()

            for group in self.param_groups:
                lr = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad
                    if weight_decay != 0:
                        grad = grad.add(p, alpha=weight_decay)

                    state = self.state[p]
                    if "step" not in state:
                        state["step"] = 0
                        state["compressed"] = CompressedOptimizerState(
                            tuple(p.shape), method=self.compression_method, **self._compress_kwargs
                        )
                        state["compressed"].set(torch.zeros_like(p), torch.zeros_like(p))

                    state["step"] += 1
                    t = state["step"]
                    compressed: CompressedOptimizerState = state["compressed"]
                    m, v = compressed.get(device=p.device, dtype=p.dtype)
                    # v is Adam's second-moment buffer -- strictly non-negative by construction --
                    # but a symmetric-family compression path (G4's default GaussianEstimator tail
                    # fit is unbounded) can reconstruct a near-zero true v as a small NEGATIVE
                    # value. Clamp before it ever reaches v_hat.sqrt(): an unclamped negative v_hat
                    # produces NaN there, which then poisons every future step irrecoverably.
                    v = v.clamp_min(0.0)

                    m = beta1 * m + (1.0 - beta1) * grad
                    v = beta2 * v + (1.0 - beta2) * grad * grad

                    bias_correction1 = 1.0 - beta1**t
                    bias_correction2 = 1.0 - beta2**t
                    m_hat = m / bias_correction1
                    v_hat = v / bias_correction2

                    p.add_(-lr * m_hat / (v_hat.sqrt() + eps))
                    compressed.set(m, v)

            return loss

    __all__.append("CompressedAdam")


# ---------------------------------------------------------------------------------------------
# 3. Selective, per-block activation-recompute policy
# ---------------------------------------------------------------------------------------------

# Proxy units (not wall-clock/real bytes-of-value), same convention as D6's compile-economics
# constants (mixle.inference.backend_respecialization, PR #153): callers with real measured
# per-byte memory value / per-FLOP compute cost should pass them explicitly.
_DEFAULT_MEMORY_VALUE_PER_BYTE = 1.0
_DEFAULT_FLOP_COST_PER_UNIT = 2e-9


def estimate_block_activation_bytes(batch: int, seq_len: int, d_model: int, dtype_bytes: int = 4) -> float:
    """Estimate the memory footprint of ONE transformer block's stored output activation
    (``mixle.models.transformer.Block``'s output, shape ``(batch, seq_len, d_model)``) -- the
    memory that activation checkpointing (recomputing instead of storing) frees."""
    return float(batch) * float(seq_len) * float(d_model) * float(dtype_bytes)


def estimate_block_recompute_flops(batch: int, seq_len: int, d_model: int) -> float:
    """Estimate the FLOP cost of recomputing ONE transformer block's forward pass, tied directly
    to ``mixle.models.transformer.Block``'s actual layer shapes: ``qkv`` (``d -> 3d``), ``proj``
    (``d -> d``), and the two MLP linears (``d -> 4d -> d``) give ``3d^2 + d^2 + 4d^2 + 4d^2 =
    12*d_model^2`` linear-layer parameters per block; the standard "2 FLOPs per parameter per
    token" forward-pass heuristic turns that into a FLOP estimate, plus the attention score/value
    matmuls (``QK^T`` and ``attn @ V``, each ``~2*batch*seq_len^2*d_model`` FLOPs) that scale with
    ``seq_len^2`` rather than with parameter count."""
    params_per_block = 12.0 * float(d_model) ** 2
    linear_flops = 2.0 * float(batch) * float(seq_len) * params_per_block
    attn_score_flops = 4.0 * float(batch) * float(seq_len) ** 2 * float(d_model)
    return linear_flops + attn_score_flops


def estimate_recompute_benefit(
    activation_bytes: float, memory_value_per_byte: float = _DEFAULT_MEMORY_VALUE_PER_BYTE
) -> float:
    """The value of the memory freed by recomputing (rather than storing) one block's activation."""
    return float(activation_bytes) * float(memory_value_per_byte)


def estimate_recompute_cost(recompute_flops: float, flop_cost_per_unit: float = _DEFAULT_FLOP_COST_PER_UNIT) -> float:
    """The cost of the extra compute spent recomputing one block's activation during backward."""
    return float(recompute_flops) * float(flop_cost_per_unit)


@dataclass(frozen=True)
class RecomputeDecision:
    """The cost/benefit tradeoff and chosen action for ONE block's activation-recompute policy --
    mirrors D6's :class:`~mixle.inference.backend_respecialization.RespecializationDecision`
    shape: a flat, inspectable dataclass carrying both the raw estimates and the derived decision,
    not just a boolean.
    """

    block_index: int
    should_recompute: bool
    estimated_cost: float
    estimated_benefit: float
    activation_bytes: float
    recompute_flops: float
    rationale: str

    @property
    def net_benefit(self) -> float:
        """``estimated_benefit - estimated_cost`` -- positive iff recomputing this block is worth it."""
        return self.estimated_benefit - self.estimated_cost


class SelectiveRecomputePolicy:
    """Per-block, cost-model-driven activation-checkpointing decision -- extends
    ``mixle.models.transformer.CausalLM``'s previously all-or-nothing ``gradient_checkpointing``
    bool flag (see that module's ``forward``, which now also accepts a per-block list) to a
    PER-BLOCK decision.

    A block is recommended for recompute when the value of the memory freed (its stored-
    activation footprint, valued at ``memory_value_per_byte``) exceeds the cost of the extra
    compute spent recomputing it (its recompute FLOPs, valued at ``flop_cost_per_unit``) -- the
    same cost-vs-benefit tradeoff SHAPE as D6's compile economics, applied to this different
    decision (recompute-vs-store, not eager-vs-compiled).
    """

    def __init__(
        self,
        memory_value_per_byte: float = _DEFAULT_MEMORY_VALUE_PER_BYTE,
        flop_cost_per_unit: float = _DEFAULT_FLOP_COST_PER_UNIT,
    ) -> None:
        self.memory_value_per_byte = float(memory_value_per_byte)
        self.flop_cost_per_unit = float(flop_cost_per_unit)

    def decide_block(self, block_index: int, activation_bytes: float, recompute_flops: float) -> RecomputeDecision:
        benefit = estimate_recompute_benefit(activation_bytes, self.memory_value_per_byte)
        cost = estimate_recompute_cost(recompute_flops, self.flop_cost_per_unit)
        should_recompute = benefit > cost
        rationale = (
            f"block {block_index}: benefit(memory freed)={benefit:.4g} "
            f"{'>' if should_recompute else '<='} cost(recompute flops)={cost:.4g} "
            f"-> {'recompute' if should_recompute else 'store'}"
        )
        return RecomputeDecision(
            block_index=block_index,
            should_recompute=should_recompute,
            estimated_cost=cost,
            estimated_benefit=benefit,
            activation_bytes=float(activation_bytes),
            recompute_flops=float(recompute_flops),
            rationale=rationale,
        )

    def decide_model(self, lm: Any, batch: int, seq_len: int, dtype_bytes: int = 4) -> list[RecomputeDecision]:
        """Decide per-block recompute for every block of a ``CausalLM``
        (``mixle.models.transformer.build_causal_lm``), using its own ``d_model``/``n_layer``."""
        d_model = int(lm.d_model)
        n_layer = int(lm.n_layer)
        decisions = []
        for i in range(n_layer):
            activation_bytes = estimate_block_activation_bytes(batch, seq_len, d_model, dtype_bytes)
            recompute_flops = estimate_block_recompute_flops(batch, seq_len, d_model)
            decisions.append(self.decide_block(i, activation_bytes, recompute_flops))
        return decisions

    def apply_to_model(self, lm: Any, batch: int, seq_len: int, dtype_bytes: int = 4) -> list[RecomputeDecision]:
        """Compute the per-block decisions and set them directly as ``lm.gradient_checkpointing``
        (a per-block bool list -- see ``mixle.models.transformer.CausalLM.forward``, which accepts
        either a single bool for all blocks or a per-block list)."""
        decisions = self.decide_model(lm, batch, seq_len, dtype_bytes)
        lm.gradient_checkpointing = [d.should_recompute for d in decisions]
        return decisions
