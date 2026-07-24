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

# pack_pm1's declared alphabet is {-1,+1} OR {0,1} (see its docstring); the union is {-1,0,1}. pack_ternary's
# declared alphabet is {-1,0,+1}. Both are the same set, so one check serves both entry points.
_PM1_ALPHABET = (-1, 0, 1)


def _require_exact_int(value: Any, name: str) -> int:
    """Coerce ``value`` to a plain ``int``, rejecting bools, non-numeric types, non-finite floats, and any
    value with a fractional part."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an int, got bool {value!r}")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}")
        truncated = int(value)
        if value != truncated:
            raise ValueError(f"{name} must be an exact integer, got {value!r}")
        return truncated
    raise TypeError(f"{name} must be an int, got {value!r} ({type(value).__name__})")


def _require_nonnegative_int(value: Any, name: str) -> int:
    v = _require_exact_int(value, name)
    if v < 0:
        raise ValueError(f"{name} must be nonnegative, got {v}")
    return v


def _validate_alphabet(x: np.ndarray, name: str) -> None:
    """Raise ``ValueError`` unless every value of ``x`` is an exact member of the declared ``{-1,0,+1}``
    alphabet. Packing silently mapped every positive value to ``+1`` and everything else to ``-1``, so
    out-of-alphabet inputs like ``2``, ``-9``, or ``0.2`` were reinterpreted via the sign test instead of
    being rejected.
    """
    if x.dtype == np.bool_ or x.size == 0:
        return  # booleans are always {0,1}-valid by construction
    if np.issubdtype(x.dtype, np.floating) and not np.all(np.isfinite(x)):
        raise ValueError(f"{name} contains non-finite values (NaN/Inf); expected values in {_PM1_ALPHABET!r}")
    mask = np.zeros(x.shape, dtype=bool)
    for v in _PM1_ALPHABET:
        mask |= x == v
    if not np.all(mask):
        bad = np.unique(np.asarray(x)[~mask])
        raise ValueError(
            f"{name} values must be exactly one of {_PM1_ALPHABET!r}; got out-of-alphabet value(s) {bad[:8].tolist()!r}"
        )


def _check_2d(a: np.ndarray, name: str) -> None:
    if a.ndim != 2:
        raise ValueError(f"{name} must be a 2D packed array (rows, words); got shape {a.shape!r}")


def _check_same_shape(a: np.ndarray, b: np.ndarray, a_name: str, b_name: str) -> None:
    if a.shape != b.shape:
        raise ValueError(
            f"{a_name} and {b_name} must have identical shape (same rows and packed words); "
            f"got {a.shape!r} and {b.shape!r}"
        )


def _check_word_count(a: np.ndarray, b: np.ndarray, a_name: str, b_name: str) -> None:
    if a.shape[1] != b.shape[1]:
        raise ValueError(
            f"{a_name} and {b_name} have mismatched packed word counts ({a.shape[1]} vs {b.shape[1]}); "
            "their true bit-vector dimensions differ, or one operand is corrupt"
        )


def pack_pm1(x: Any) -> np.ndarray:
    """Pack a ``{-1,+1}`` (or ``{0,1}``) array's rows to ``uint64`` words; last axis padded to a 64-multiple."""
    arr = np.asarray(x)
    _validate_alphabet(arr, "pack_pm1 input")
    bits = (arr > 0).astype(np.uint8)
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

    Both operands' rank, packed word count, and word count implied by ``dim`` are validated up front: the
    compiled kernel derives its loop bound from ``a_packed`` alone and has no way to notice a shorter
    ``b_packed`` on its own (MXR-080-0131), so a caller-visible ``ValueError`` here is what stands between
    a mismatched pair and an out-of-bounds read in the ``nogil`` kernel.
    """
    dim = _require_nonnegative_int(dim, "dim")
    a = np.ascontiguousarray(a_packed, dtype=np.uint64)
    b = np.ascontiguousarray(b_packed, dtype=np.uint64)
    _check_2d(a, "a_packed")
    _check_2d(b, "b_packed")
    _check_word_count(a, b, "a_packed", "b_packed")
    expected_words = (dim + 63) // 64
    if a.shape[1] != expected_words:
        raise ValueError(
            f"dim={dim} implies {expected_words} packed word(s) per row, but a_packed/b_packed have {a.shape[1]}"
        )
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
    if a.ndim == 0 or b.ndim == 0:
        raise ValueError("binary_dot inputs must have at least one dimension")
    if a.shape[-1] != b.shape[-1]:
        raise ValueError(
            f"binary_dot requires matching trailing (true, unpacked) dimensions, got {a.shape[-1]} and {b.shape[-1]}"
        )
    dim = a.shape[-1]
    return binary_gemm(pack_pm1(a), pack_pm1(b), dim)


def ternary_gemm(a_sign: Any, a_nz: Any, b_sign: Any, b_nz: Any) -> np.ndarray:
    """Exact ``{-1,0,+1}`` ``A @ B.T`` from packed sign + nonzero-mask bit-planes (compiled path only).

    Each operand's sign and nonzero-mask planes must have identical shape (they describe the same rows),
    and the two operands' packed word counts must match. The compiled kernel derives its loop bound from
    ``a_sign`` alone and indexes into the other three planes without any bounds check of its own
    (MXR-080-0131), so this validation is what turns a shorter plane into a clean exception instead of an
    out-of-bounds read.
    """
    if not HAS_BITPACKED:
        raise RuntimeError("ternary_gemm requires the compiled _bitpacked extension (compile_bitpacked_kernels)")
    cast = lambda p: np.ascontiguousarray(p, dtype=np.uint64)  # noqa: E731
    a_sign, a_nz, b_sign, b_nz = cast(a_sign), cast(a_nz), cast(b_sign), cast(b_nz)
    _check_2d(a_sign, "a_sign")
    _check_2d(a_nz, "a_nz")
    _check_2d(b_sign, "b_sign")
    _check_2d(b_nz, "b_nz")
    _check_same_shape(a_sign, a_nz, "a_sign", "a_nz")
    _check_same_shape(b_sign, b_nz, "b_sign", "b_nz")
    _check_word_count(a_sign, b_sign, "a_sign", "b_sign")
    return _ternary_gemm_c(a_sign, a_nz, b_sign, b_nz)


def pack_ternary(x: Any) -> tuple[np.ndarray, np.ndarray]:
    """Pack a ``{-1,0,+1}`` array's rows into (sign, nonzero) bit-plane uint64 words for :func:`ternary_gemm`."""
    x = np.asarray(x)
    _validate_alphabet(x, "pack_ternary input")
    sign = pack_pm1(x > 0)  # sign bit set where value > 0 (the 0/-1 entries are gated by the nz mask)
    nz = pack_pm1(x != 0)
    return sign, nz
