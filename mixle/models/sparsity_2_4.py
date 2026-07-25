"""2:4 structured sparsity, end to end (roadmap I4): training-time mask ramp + cuSPARSELt-format export.

.. warning::

   **Experimental frontier-training prototype.** The mask-ramp and export mechanics are exact and tested,
   but the actual cuSPARSELt speedup needs a real CUDA device with that kernel; this is not a production
   sparsity-training pipeline. Treat it as a prototype and measure end-to-end on your target hardware.

Two pieces, glued by ONE borrowed primitive rather than two reimplementations:

1. :class:`TwoFourSparsityRamp` -- a schedulable training-time mask ramp. It does not reinvent 2:4
   masking: at every ramp step it calls G2's :func:`mixle.models.sigma_weighted_projection.
   sigma_weighted_block_sparse` with the literal ``"2:4"`` pattern, which is the actual masking/value
   -readjustment mechanism (see that module's docstring for the alternating-projection algorithm). This
   module's own job is just the RAMP: which fraction of a weight matrix's rows are already under the hard
   2:4 constraint at a given training step, growing from 0% at ``start_step`` to 100% at ``end_step``. The
   ramp's step-to-fraction map is a plain, swappable callable (``schedule``) precisely so a future
   roadmap-H3 "structure-edit schedule" controller could drive it (supply a different ``schedule``, or
   mutate ``start_step``/``end_step`` on the fly) without touching this module -- H3 itself is NOT built
   here, only the seam it would plug into.

2. :func:`export_2_4_compressed` / :func:`decompress` -- the actual cuSPARSELt-style compressed-matrix
   format: for every contiguous group of 4 weights along the input (last) axis that already satisfies the
   2:4 constraint (exactly 2 nonzeros), store the 2 surviving VALUES plus a small INDEX recording which 2
   of the 4 in-group positions they came from. This is the documented shape of NVIDIA's semi-structured
   sparse storage (see e.g. the cuSPARSELt / Ampere structured-sparse-tensor-core docs and
   ``torch.sparse.SparseSemiStructuredTensor``): compressed values at half the density, plus small
   per-group metadata recording nonzero *positions* (not full-size, since only ``C(4,2) = 6`` patterns are
   possible per group). NVIDIA does not publish the exact bit-for-bit metadata layout their kernels consume
   (it is treated as an opaque, hardware/kernel-generation-specific detail even inside
   ``torch.sparse.SparseSemiStructuredTensor``), so the exact packing implemented here (2 bits per in-group
   position, 2 positions packed per byte-nibble, 2 nibbles per byte) is THIS module's own documented,
   round-trip-correct encoding of the publicly documented "values + position indices" shape -- not a claim
   of bit-exact compatibility with a specific cuSPARSELt release. Values preserve portable
   float16/float32/float64 input precision; a
   real cuSPARSELt handle would additionally repack this into its internal opaque compressed-matrix object
   via ``cusparseLtSpMMACompress`` (or, in this torch build, ``torch.sparse.to_sparse_semi_structured``),
   which requires a CUDA tensor and a cuSPARSELt-capable GPU -- see :func:`cusparselt_status` and the module
   docstring in ``mixle/tests/sparsity_2_4_test.py`` for what was actually checked/measured in THIS
   environment (no CUDA device here -- see that check's output).

Environment note (checked, not assumed): this environment has no CUDA device (``torch.cuda.is_available()
is False``), so ``torch.backends.cusparselt`` reports unavailable and
``torch.sparse.to_sparse_semi_structured`` raises (CPU tensors are not supported by that call at all, CUDA
or not). The compress/decompress round trip below is pure CPU numpy/torch and is exercised directly
(round-trip exactness + measured byte-size compression ratio are pinned by tests); no accelerated
cuSPARSELt kernel is run anywhere in this module. See ``cusparselt_status()`` for a queryable summary of
what this torch build actually offers.
"""

from __future__ import annotations

import struct
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Real
from typing import Any, ClassVar

import numpy as np

from mixle.models.sigma_weighted_projection import sigma_weighted_block_sparse

__all__ = [
    "TwoFourSparsityRamp",
    "Compressed2to4",
    "export_2_4_compressed",
    "decompress",
    "cusparselt_status",
    "is_2_4_sparse",
]


# --------------------------------------------------------------------------------------------------------
# 1. training-time mask ramp
# --------------------------------------------------------------------------------------------------------


def _linear_ramp(step: int, start_step: int, end_step: int) -> float:
    """Default schedule: plain linear ramp of the CONSTRAINED-ROW FRACTION from 0 at ``start_step`` to 1 at
    ``end_step``. Held at 0 before ``start_step`` and at 1 from ``end_step`` onward. This is the "fraction
    of weights subject to the 2:4 constraint" ramp the roadmap item asks for -- not a soft/relaxed density,
    a genuine growing set of ROWS that are hard-projected onto the exact 2:4 pattern every step, so that by
    ``end_step`` the whole matrix satisfies the structural constraint the export format requires.
    """
    if end_step <= start_step:
        return 1.0 if step >= end_step else 0.0
    if step <= start_step:
        return 0.0
    if step >= end_step:
        return 1.0
    return (step - start_step) / (end_step - start_step)


ScheduleFn = Callable[[int, int, int], float]


def _exact_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _finite_real(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


class TwoFourSparsityRamp:
    """Schedulable training-time 2:4 mask ramp.

    ``start_step``/``end_step``: training steps between which the constrained-row fraction grows from 0 to
    1 (see ``schedule``). ``target_density``: the density the 2:4 CONSTRAINT itself enforces once applied
    -- structurally fixed at ``0.5`` (2 of every 4), kept as an explicit constructor argument (rather than
    hardcoded) so the acceptance criterion "2:4 model at stated ... density" is documented at the call site
    and so a mismatched caller expectation fails loudly at construction time instead of silently.
    ``schedule(step, start_step, end_step) -> fraction in [0, 1]``: the pluggable ramp-shape callable --
    THE seam a future roadmap-H3 structure-edit-schedule controller would drive (a non-linear cubic ramp, a
    warmup-then-hold schedule, one keyed off validation loss instead of step count, etc.); defaults to
    :func:`_linear_ramp`. H3 itself is not implemented here -- this is deliberately just a callable/
    parameter another component could supply.
    """

    def __init__(
        self,
        start_step: int,
        end_step: int,
        target_density: float = 0.5,
        schedule: ScheduleFn | None = None,
    ) -> None:
        start_step = _exact_int(start_step, "start_step")
        end_step = _exact_int(end_step, "end_step")
        if end_step < start_step:
            raise ValueError(f"end_step ({end_step}) must be >= start_step ({start_step})")
        target_density = _finite_real(target_density, "target_density")
        if abs(target_density - 0.5) > 1e-9:
            # 2:4 semi-structured sparsity keeps exactly 2 of every 4 entries -- density 0.5 is not a
            # tunable knob of the *pattern* itself (a different target density would need a different
            # structural pattern, e.g. N:M for other N/M, which is out of scope here). Kept as an explicit
            # argument (rather than removed) so callers state the density they expect and get a loud error
            # if it doesn't match what 2:4 actually delivers.
            raise ValueError(f"2:4 sparsity has fixed density 0.5; got target_density={target_density}")
        if schedule is not None and not callable(schedule):
            raise TypeError("schedule must be callable or None")
        self.start_step = start_step
        self.end_step = end_step
        self.target_density = target_density
        self.schedule: ScheduleFn = schedule or _linear_ramp

    def fraction(self, step: int) -> float:
        """Fraction of the weight matrix's OUTPUT ROWS that are under the hard 2:4 constraint at ``step``,
        in ``[0, 1]``. Invalid custom-schedule results fail closed rather than being silently clipped."""
        step = _exact_int(step, "step")
        fraction = _finite_real(self.schedule(step, self.start_step, self.end_step), "schedule result")
        if fraction < 0.0 or fraction > 1.0:
            raise ValueError(f"schedule result must be in [0, 1]; got {fraction}")
        return fraction

    def n_constrained_rows(self, step: int, n_rows: int) -> int:
        n_rows = _exact_int(n_rows, "n_rows")
        return int(round(self.fraction(step) * n_rows))

    def project(self, weight: Any, step: int, sigma: Any = None) -> Any:
        """Apply the ramp to a weight matrix (``d_out x d_in``, ``d_in`` a multiple of 4): the first
        ``n_constrained_rows(step, d_out)`` rows are hard-projected onto the 2:4 pattern via G2's
        :func:`sigma_weighted_block_sparse` (the real masking/value-adjustment mechanism -- this is not a
        magnitude-only reimplementation); the remaining rows are left untouched (still fully dense at this
        point in the ramp). Returns a NEW array/tensor of the same type and shape as ``weight``; does not
        mutate ``weight`` in place (callers doing in-place training updates copy the result back
        themselves, see :func:`apply_ramp_to_linear_`).

        ``sigma``: optional ``d_in x d_in`` covariance to weight the projection by (e.g. a real propagated
        law from G1/``moment_propagation``, wired the same way G2's own acceptance test wires it). Defaults
        to the identity, which reduces the Sigma-weighted objective to plain Frobenius reconstruction --
        i.e. plain magnitude-based 2:4 value selection when no activation-covariance estimate is available,
        which is the common case during ordinary token-level training where no data-free law has been
        propagated for this weight.

        Rows are constrained from index 0 upward (a fixed, deterministic ordering) rather than by
        magnitude or another heuristic -- keeps the ramp's mask-growth trajectory the same across calls for
        the same ``step``, which is what the ramp-correctness test below checks against.
        """
        torch = _torch_or_none()
        is_tensor = torch is not None and isinstance(weight, torch.Tensor)
        w_np = weight.detach().cpu().numpy().astype(np.float64) if is_tensor else np.asarray(weight, dtype=np.float64)

        if w_np.ndim != 2 or min(w_np.shape) == 0:
            raise ValueError("2:4 sparsity ramp requires a non-empty two-dimensional weight matrix")
        if not np.all(np.isfinite(w_np)):
            raise ValueError("2:4 sparsity ramp requires finite weights")
        d_out, d_in = w_np.shape
        if d_in % 4 != 0:
            raise ValueError(f"2:4 sparsity ramp requires the input dim to be a multiple of 4; got {d_in}")
        sigma_np = np.eye(d_in) if sigma is None else np.asarray(sigma, dtype=np.float64)

        n_constrained = self.n_constrained_rows(step, d_out)
        out = w_np.copy()
        if n_constrained > 0:
            out[:n_constrained] = sigma_weighted_block_sparse(w_np[:n_constrained], sigma_np, "2:4")

        if is_tensor:
            return torch.as_tensor(out, dtype=weight.dtype, device=weight.device)
        return out

    def apply_(self, linear: Any, step: int, sigma: Any = None) -> None:
        """In-place convenience: project ``linear.weight.data`` through :meth:`project` and copy the result
        back. This is the "mask-then-continue-training" step a caller runs after every optimizer step (see
        ``train_with_ramp`` below) -- gradients keep flowing through the dense parameter between projections
        (straight-through), and the projection is re-applied (with a re-selected pattern, per G2) every
        call, which is what lets already-constrained rows keep adapting their surviving VALUES as training
        continues, not just their support.
        """
        with_no_grad = _torch_or_none()
        projected = self.project(linear.weight.data, step, sigma=sigma)
        if with_no_grad is not None:
            with with_no_grad.no_grad():
                linear.weight.data.copy_(projected)
        else:  # pragma: no cover - torch is required for nn.Linear callers anyway
            linear.weight.data = projected


def is_2_4_sparse(weight: Any) -> bool:
    """True iff every contiguous group of 4 entries along the last axis has AT MOST 2 nonzeros (the 2:4
    structural constraint the export format below requires as a precondition)."""
    w_np = _to_numpy(weight)
    d_out, d_in = w_np.shape
    if d_in % 4 != 0:
        return False
    groups_nnz = (w_np.reshape(d_out, d_in // 4, 4) != 0).sum(axis=-1)
    return bool(np.all(groups_nnz <= 2))


def _to_numpy(x: Any) -> np.ndarray:
    torch = _torch_or_none()
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _torch_or_none() -> Any:
    try:
        import torch

        return torch
    except ImportError:  # pragma: no cover
        return None


def cusparselt_status() -> dict[str, Any]:
    """Honest, queryable snapshot of what THIS torch build/environment actually offers for real
    cuSPARSELt-accelerated 2:4 sparse GEMM -- used by the tests to decide (and clearly LABEL) whether an
    "inference speedup" number is a real measurement or a theoretical bound. Never raises: every field
    degrades to ``False``/``None`` if the relevant torch API is missing entirely (older torch builds).
    """
    torch = _torch_or_none()
    if torch is None:
        return {"torch_available": False}
    status: dict[str, Any] = {
        "torch_available": True,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    cusparselt = getattr(torch.backends, "cusparselt", None)
    status["has_cusparselt_backend"] = cusparselt is not None
    if cusparselt is not None:
        try:
            status["cusparselt_is_available"] = bool(cusparselt.is_available())
        except Exception as e:  # pragma: no cover - defensive  # noqa: BLE001
            status["cusparselt_is_available"] = False
            status["cusparselt_is_available_error"] = repr(e)
        try:
            status["cusparselt_version"] = cusparselt.version()
        except Exception as e:  # pragma: no cover - defensive  # noqa: BLE001
            status["cusparselt_version"] = None
            status["cusparselt_version_error"] = repr(e)
    status["has_sparse_semi_structured_tensor"] = hasattr(torch.sparse, "SparseSemiStructuredTensor")
    status["capable_of_real_cusparselt_gemm"] = bool(
        status["cuda_available"] and status.get("cusparselt_is_available", False)
    )
    return status


# --------------------------------------------------------------------------------------------------------
# 2. cuSPARSELt-format compressed export (CPU-side format conversion, real and round-trip correct)
# --------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Compressed2to4:
    """A 2:4 semi-structured-sparse compressed matrix, cuSPARSELt-shaped: the 2 surviving values per group
    of 4 (``values``, half the density of the dense matrix) plus a small per-group index recording which 2
    of the 4 in-group positions they occupy (``indices``). See the module docstring for exactly how
    ``indices`` is packed (2 bits per in-group position, 2 groups' worth of index nibbles per byte) and why
    that specific bit layout is this module's own documented encoding rather than a claim of bit-for-bit
    compatibility with a specific NVIDIA driver's opaque compressed-matrix object.

    ``values``: float array, shape ``(d_out, d_in // 4, 2)`` -- the 2 surviving values per group, in
    ASCENDING in-group-position order (position of ``values[..., 0]`` <= position of ``values[..., 1]``;
    this ordering convention is what lets ``indices`` unambiguously address them on decompress).
    ``indices``: ``uint8`` array, shape ``(d_out, ceil(d_in // 4 / 2))`` -- packed 2-bits-per-position
    metadata, 2 groups per byte (low nibble = group ``2k``'s two 2-bit positions, high nibble = group
    ``2k+1``'s, when present).
    ``shape``: the original dense ``(d_out, d_in)`` shape, needed to reconstruct ``decompress``'s output
    (the last packed index byte may cover an odd group count).
    """

    MAGIC: ClassVar[bytes] = b"M24\x00"
    FORMAT_VERSION: ClassVar[int] = 1
    _HEADER: ClassVar[struct.Struct] = struct.Struct("<4sBBQQ")
    _DTYPE_TO_CODE: ClassVar[dict[np.dtype, int]] = {
        np.dtype(np.float16): 1,
        np.dtype(np.float32): 2,
        np.dtype(np.float64): 3,
    }
    _CODE_TO_DTYPE: ClassVar[dict[int, np.dtype]] = {
        1: np.dtype(np.float16),
        2: np.dtype(np.float32),
        3: np.dtype(np.float64),
    }

    values: np.ndarray
    indices: np.ndarray
    shape: tuple[int, int]
    format_version: int = FORMAT_VERSION

    def __post_init__(self) -> None:
        version = _exact_int(self.format_version, "format_version", minimum=1)
        if version != self.FORMAT_VERSION:
            raise ValueError(f"unsupported Compressed2to4 format_version {version}")
        if not isinstance(self.shape, (tuple, list)) or len(self.shape) != 2:
            raise ValueError("shape must be a (d_out, d_in) pair")
        d_out = _exact_int(self.shape[0], "shape[0]", minimum=1)
        d_in = _exact_int(self.shape[1], "shape[1]", minimum=1)
        if d_in % 4 != 0:
            raise ValueError("Compressed2to4 input dimension must be a multiple of 4")
        n_groups = d_in // 4

        values = np.asarray(self.values)
        if values.dtype not in self._DTYPE_TO_CODE:
            raise TypeError("Compressed2to4 values must use float16, float32, or float64")
        if values.shape != (d_out, n_groups, 2):
            raise ValueError(
                "Compressed2to4 values must have shape "
                f"{(d_out, n_groups, 2)}; got {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("Compressed2to4 values must be finite")

        indices = np.asarray(self.indices)
        expected_indices = (d_out, (n_groups + 1) // 2)
        if indices.dtype != np.uint8 or indices.shape != expected_indices:
            raise ValueError(
                f"Compressed2to4 indices must be uint8 with shape {expected_indices}; "
                f"got dtype={indices.dtype}, shape={indices.shape}"
            )
        for group in range(n_groups):
            byte = indices[:, group // 2]
            nibble = byte >> 4 if group % 2 else byte & 0x0F
            pos0 = nibble & 0b11
            pos1 = (nibble >> 2) & 0b11
            if np.any(pos0 >= pos1):
                raise ValueError("Compressed2to4 packed positions must be distinct and ascending")
        if n_groups % 2 and np.any(indices[:, -1] & 0xF0):
            raise ValueError("Compressed2to4 unused high metadata nibble must be zero")

        values = np.ascontiguousarray(values).copy()
        indices = np.ascontiguousarray(indices).copy()
        values.setflags(write=False)
        indices.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "shape", (d_out, d_in))
        object.__setattr__(self, "format_version", version)

    def nbytes(self) -> int:
        """Return the exact size of :meth:`to_bytes`, including its versioned header."""
        return len(self.to_bytes())

    def to_bytes(self) -> bytes:
        """Serialize the validated payload without pickle."""
        dtype_code = self._DTYPE_TO_CODE[self.values.dtype]
        header = self._HEADER.pack(
            self.MAGIC,
            self.format_version,
            dtype_code,
            self.shape[0],
            self.shape[1],
        )
        return header + self.values.tobytes(order="C") + self.indices.tobytes(order="C")

    @classmethod
    def from_bytes(cls, payload: Any) -> Compressed2to4:
        """Parse a bounded, versioned payload and reject truncation or trailing data."""
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise TypeError("Compressed2to4 payload must be bytes-like")
        view = memoryview(payload)
        if len(view) < cls._HEADER.size:
            raise ValueError("Compressed2to4 payload is truncated")
        magic, version, dtype_code, d_out, d_in = cls._HEADER.unpack(view[: cls._HEADER.size])
        if magic != cls.MAGIC:
            raise ValueError("Compressed2to4 payload has an invalid magic value")
        if version != cls.FORMAT_VERSION:
            raise ValueError(f"unsupported Compressed2to4 payload version {version}")
        dtype = cls._CODE_TO_DTYPE.get(dtype_code)
        if dtype is None:
            raise ValueError(f"unsupported Compressed2to4 dtype code {dtype_code}")
        if d_out == 0 or d_in == 0 or d_in % 4:
            raise ValueError("Compressed2to4 payload has an invalid dense shape")
        n_groups = d_in // 4
        values_count = d_out * n_groups * 2
        values_bytes = values_count * dtype.itemsize
        indices_count = d_out * ((n_groups + 1) // 2)
        expected = cls._HEADER.size + values_bytes + indices_count
        if len(view) != expected:
            raise ValueError(f"Compressed2to4 payload length must be exactly {expected} bytes; got {len(view)}")
        values_start = cls._HEADER.size
        values = np.frombuffer(view[values_start : values_start + values_bytes], dtype=dtype).reshape(
            d_out, n_groups, 2
        )
        indices = np.frombuffer(view[values_start + values_bytes :], dtype=np.uint8).reshape(
            d_out, (n_groups + 1) // 2
        )
        return cls(values, indices, (d_out, d_in), format_version=version)


def _pack_group_index(pos0: int, pos1: int) -> int:
    """Pack two 2-bit in-group positions (each in ``0..3``) into one nibble: ``pos0`` in bits 0-1, ``pos1``
    in bits 2-3."""
    return (pos0 & 0b11) | ((pos1 & 0b11) << 2)


def _unpack_group_index(nibble: int) -> tuple[int, int]:
    return nibble & 0b11, (nibble >> 2) & 0b11


def export_2_4_compressed(weight_2_4_masked: Any) -> Compressed2to4:
    """Convert an already-2:4-masked dense weight matrix into the compressed (values + indices) storage
    format cuSPARSELt-style semi-structured sparse GEMM kernels consume. Precondition: every contiguous
    group of 4 entries along the last axis already has AT MOST 2 nonzeros (checked; raises if violated --
    this function does not itself *select* the 2:4 pattern, that's :func:`TwoFourSparsityRamp.project` /
    G2's ``sigma_weighted_block_sparse``, this is purely the storage-format conversion of an already-
    constrained matrix, matching how a real cuSPARSELt workflow separates "prune/select the pattern" from
    "compress for the kernel").
    """
    w_np = _to_numpy(weight_2_4_masked)
    if w_np.ndim != 2 or min(w_np.shape) == 0:
        raise ValueError("2:4 export requires a non-empty two-dimensional weight matrix")
    if w_np.dtype not in Compressed2to4._DTYPE_TO_CODE:
        raise TypeError("2:4 export supports float16, float32, and float64 weights")
    if not np.all(np.isfinite(w_np)):
        raise ValueError("2:4 export requires finite weights")
    d_out, d_in = w_np.shape
    if d_in % 4 != 0:
        raise ValueError(f"2:4 export requires the input dim to be a multiple of 4; got {d_in}")
    n_groups = d_in // 4
    groups = w_np.reshape(d_out, n_groups, 4)
    nnz_per_group = (groups != 0).sum(axis=-1)
    if not np.all(nnz_per_group <= 2):
        bad = np.argwhere(nnz_per_group > 2)
        raise ValueError(
            f"export_2_4_compressed requires every group of 4 to have <=2 nonzeros; "
            f"found a violation at (row, group)={tuple(bad[0])}"
        )

    values = np.zeros((d_out, n_groups, 2), dtype=w_np.dtype)
    packed_len = (n_groups + 1) // 2
    indices = np.zeros((d_out, packed_len), dtype=np.uint8)

    for i in range(d_out):
        for g in range(n_groups):
            nz_positions = np.flatnonzero(groups[i, g] != 0)
            # pad with an arbitrary (value=0) "phantom" position when a group has <2 real nonzeros (e.g. a
            # weight that was already exactly-zero pre-masking): pick the lowest UNUSED position so
            # positions stay well-defined and decompress reconstructs the same (all-zero-there) values.
            if nz_positions.size < 2:
                used = set(nz_positions.tolist())
                for cand in range(4):
                    if len(nz_positions) >= 2:
                        break
                    if cand not in used:
                        nz_positions = np.append(nz_positions, cand)
                        used.add(cand)
                nz_positions = np.sort(nz_positions)
            pos0, pos1 = int(nz_positions[0]), int(nz_positions[1])
            values[i, g, 0] = groups[i, g, pos0]
            values[i, g, 1] = groups[i, g, pos1]
            nibble = _pack_group_index(pos0, pos1)
            byte_idx, high = divmod(g, 2)
            if high == 0:
                indices[i, byte_idx] = (indices[i, byte_idx] & 0b11110000) | nibble
            else:
                indices[i, byte_idx] = (indices[i, byte_idx] & 0b00001111) | (nibble << 4)

    return Compressed2to4(values=values, indices=indices, shape=(d_out, d_in))


def decompress(compressed: Compressed2to4) -> np.ndarray:
    """Exact inverse of :func:`export_2_4_compressed`: reconstruct the dense (2:4-sparse) matrix from
    ``values``/``indices``. Round-trips EXACTLY (bit-for-bit on the surviving values, genuine zeros
    elsewhere) -- pinned directly by the compress/decompress test."""
    if not isinstance(compressed, Compressed2to4):
        raise TypeError("compressed must be a Compressed2to4 payload")
    d_out, d_in = compressed.shape
    n_groups = d_in // 4
    out = np.zeros((d_out, n_groups, 4), dtype=compressed.values.dtype)
    for i in range(d_out):
        for g in range(n_groups):
            byte_idx, high = divmod(g, 2)
            byte = int(compressed.indices[i, byte_idx])
            nibble = (byte >> 4) if high else (byte & 0b00001111)
            pos0, pos1 = _unpack_group_index(nibble)
            out[i, g, pos0] = compressed.values[i, g, 0]
            out[i, g, pos1] = compressed.values[i, g, 1]
    return out.reshape(d_out, d_in)
