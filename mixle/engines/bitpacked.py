"""Packed binary / ternary COMPUTE: exact {-1,+1} and {-1,0,+1} dot products via popcount.

The one sub-byte path that is *real arithmetic* (not dequant-then-fp32): pack a vector to bits and
``a . b = D - 2*popcount(a XOR b)`` (binary) / a two-plane popcount (ternary). The hardware popcount does
64 lanes per instruction with no rounding, and the packed data is 32x smaller than float64.

Performance depends on the fp32 baseline. On Apple silicon, cache-resident GEMM goes through the AMX
matrix coprocessor (Accelerate BLAS), which can make this popcount kernel slower; the advantage there is
storage and bandwidth rather than compute. On CPUs without a matrix unit, memory-bound problems, or
native binary/ternary models where fp32 wastes 32x the bytes, it can be a compute win. The kernel is
always *exact*; select it for the measured regime. The compiled extension is optional
(``build_kernels.compile_bitpacked_kernels``); a correct but slower numpy ``bitwise_count`` fallback runs
when it is absent.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from mixle.engines._bitpacked import binary_gemm as _binary_gemm_c
    from mixle.engines._bitpacked import ternary_gemm as _ternary_gemm_c

    HAS_BITPACKED = True
except ImportError:  # pragma: no cover - extension optional
    HAS_BITPACKED = False


def pack_pm1(x: Any) -> np.ndarray:
    """Pack a ``{-1,+1}`` (or ``{0,1}``) array's rows to ``uint64`` words; last axis padded to a 64-multiple."""
    bits = (np.asarray(x) > 0).astype(np.uint8)
    if bits.ndim == 1:
        bits = bits[None, :]
    pad = (-bits.shape[1]) % 64
    if pad:
        bits = np.pad(bits, ((0, 0), (0, pad)))
    return np.ascontiguousarray(np.packbits(bits, axis=1)).view(np.uint64)


def binary_gemm(a_packed: Any, b_packed: Any, dim: int) -> np.ndarray:
    """Exact ``A @ B.T`` for ``{-1,+1}`` matrices packed by :func:`pack_pm1`.

    ``a_packed`` is ``(N, words)`` packed rows of A, ``b_packed`` is ``(M, words)`` packed rows of B (the
    operand whose columns are dotted), ``dim`` is the true bit length. Returns ``int32`` ``(N, M)``.
    """
    a = np.ascontiguousarray(a_packed, dtype=np.uint64)
    b = np.ascontiguousarray(b_packed, dtype=np.uint64)
    if HAS_BITPACKED:
        return _binary_gemm_c(a, b, int(dim))
    # correct, memory-bounded numpy fallback: one row of A vs all rows of B at a time
    ab = a.view(np.uint8).reshape(a.shape[0], -1)
    bb = b.view(np.uint8).reshape(b.shape[0], -1)
    out = np.empty((a.shape[0], b.shape[0]), dtype=np.int32)
    for i in range(a.shape[0]):
        ham = np.bitwise_count(np.bitwise_xor(ab[i], bb)).sum(axis=1)
        out[i] = int(dim) - 2 * ham
    return out


def binary_dot(a: Any, b: Any) -> np.ndarray:
    """Exact dot products of a batch of ``{-1,+1}`` vectors ``a`` (N, D) against ``b`` (M, D). Returns (N, M)."""
    a = np.asarray(a)
    b = np.asarray(b)
    dim = a.shape[-1]
    return binary_gemm(pack_pm1(a), pack_pm1(b), dim)


def ternary_gemm(a_sign: Any, a_nz: Any, b_sign: Any, b_nz: Any) -> np.ndarray:
    """Exact ``{-1,0,+1}`` ``A @ B.T`` from packed sign + nonzero-mask bit-planes (compiled path only)."""
    if not HAS_BITPACKED:
        raise RuntimeError("ternary_gemm requires the compiled _bitpacked extension (compile_bitpacked_kernels)")
    cast = lambda p: np.ascontiguousarray(p, dtype=np.uint64)  # noqa: E731
    return _ternary_gemm_c(cast(a_sign), cast(a_nz), cast(b_sign), cast(b_nz))


def pack_ternary(x: Any) -> tuple[np.ndarray, np.ndarray]:
    """Pack a ``{-1,0,+1}`` array's rows into (sign, nonzero) bit-plane uint64 words for :func:`ternary_gemm`."""
    x = np.asarray(x)
    sign = pack_pm1(x > 0)  # sign bit set where value > 0 (the 0/-1 entries are gated by the nz mask)
    nz = pack_pm1(x != 0)
    return sign, nz
