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

from mixle.engines._optional_extension import load_optional_extension

_BITPACKED_EXTENSION = load_optional_extension(
    "mixle.engines._bitpacked",
    ("binary_gemm", "ternary_gemm"),
)
HAS_BITPACKED = _BITPACKED_EXTENSION.available
BITPACKED_EXTENSION_DIAGNOSTIC = _BITPACKED_EXTENSION.diagnostic
if HAS_BITPACKED:  # pragma: no cover - depends on the optional local build
    _binary_gemm_c, _ternary_gemm_c = _BITPACKED_EXTENSION.values

_PM1_ALPHABET = (-1, 1)
_ZERO_ONE_ALPHABET = (0, 1)
_TERNARY_ALPHABET = (-1, 0, 1)


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


def _require_vector_or_matrix(x: Any, name: str) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim not in (1, 2):
        raise ValueError(f"{name} must be a vector or matrix; got shape {arr.shape!r}")
    return arr


def _validate_finite(x: np.ndarray, name: str, alphabet: tuple[int, ...]) -> None:
    if np.issubdtype(x.dtype, np.floating) and not np.all(np.isfinite(x)):
        raise ValueError(f"{name} contains non-finite values (NaN/Inf); expected values in {alphabet!r}")


def _matches_alphabet(x: np.ndarray, alphabet: tuple[int, ...]) -> bool:
    mask = np.zeros(x.shape, dtype=bool)
    for v in alphabet:
        mask |= x == v
    return bool(np.all(mask))


def _validate_pm1_alphabet(x: np.ndarray, name: str) -> None:
    """Require one complete binary alphabet, never the incompatible alphabets' three-value union."""
    _validate_finite(x, name, _TERNARY_ALPHABET)
    if x.dtype == np.bool_ or x.size == 0:
        return
    if _matches_alphabet(x, _PM1_ALPHABET) or _matches_alphabet(x, _ZERO_ONE_ALPHABET):
        return
    bad = np.unique(x)
    raise ValueError(
        f"{name} must use either {_PM1_ALPHABET!r} or {_ZERO_ONE_ALPHABET!r} consistently; "
        f"got values {bad[:8].tolist()!r}"
    )


def _validate_ternary_alphabet(x: np.ndarray, name: str) -> None:
    _validate_finite(x, name, _TERNARY_ALPHABET)
    if not _matches_alphabet(x, _TERNARY_ALPHABET):
        bad = np.unique(x)
        raise ValueError(f"{name} values must be in {_TERNARY_ALPHABET!r}; got {bad[:8].tolist()!r}")


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


def _as_packed_matrix(value: Any, name: str) -> np.ndarray:
    """Validate packed words before uint64 conversion, preventing fractional truncation and wrapping."""
    raw = np.asarray(value)
    _check_2d(raw, name)
    if raw.dtype == np.bool_:
        raise ValueError(f"{name} must contain uint64 words, not booleans")
    if np.issubdtype(raw.dtype, np.integer):
        if raw.size and np.any(raw < 0):
            raise ValueError(f"{name} must not contain negative words")
    elif np.issubdtype(raw.dtype, np.floating):
        if raw.size and (
            not np.all(np.isfinite(raw))
            or not np.array_equal(raw, np.trunc(raw))
            or np.any(raw < 0)
            or np.any(raw >= 2**64)
        ):
            raise ValueError(f"{name} must contain exact finite uint64 words")
    else:
        checked = np.empty(raw.shape, dtype=np.uint64)
        for index, item in np.ndenumerate(raw):
            word = _require_nonnegative_int(item, name)
            if word > np.iinfo(np.uint64).max:
                raise ValueError(f"{name} word exceeds uint64 range")
            checked[index] = word
        return np.ascontiguousarray(checked)
    return np.ascontiguousarray(raw, dtype=np.uint64)


def _require_canonical_padding(packed: np.ndarray, dim: int, name: str) -> None:
    """Require unused trailing packbits lanes to be zero."""
    if dim == 0:
        return
    byte_rows = packed.view(np.uint8).reshape(packed.shape[0], -1)
    used_bytes, remaining_bits = divmod(dim, 8)
    if remaining_bits:
        padding_mask = (1 << (8 - remaining_bits)) - 1
        if np.any(byte_rows[:, used_bytes] & padding_mask):
            raise ValueError(f"{name} has nonzero padding bits beyond dim={dim}")
        used_bytes += 1
    if np.any(byte_rows[:, used_bytes:]):
        raise ValueError(f"{name} has nonzero padding bytes beyond dim={dim}")


def pack_pm1(x: Any) -> np.ndarray:
    """Pack a ``{-1,+1}`` (or ``{0,1}``) array's rows to ``uint64`` words; last axis padded to a 64-multiple."""
    arr = _require_vector_or_matrix(x, "pack_pm1 input")
    _validate_pm1_alphabet(arr, "pack_pm1 input")
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
    a = _as_packed_matrix(a_packed, "a_packed")
    b = _as_packed_matrix(b_packed, "b_packed")
    _check_word_count(a, b, "a_packed", "b_packed")
    expected_words = (dim + 63) // 64
    if a.shape[1] != expected_words:
        raise ValueError(
            f"dim={dim} implies {expected_words} packed word(s) per row, but a_packed/b_packed have {a.shape[1]}"
        )
    _require_canonical_padding(a, dim, "a_packed")
    _require_canonical_padding(b, dim, "b_packed")
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
    a = _require_vector_or_matrix(a, "binary_dot a")
    b = _require_vector_or_matrix(b, "binary_dot b")
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
    x = _require_vector_or_matrix(x, "pack_ternary input")
    _validate_ternary_alphabet(x, "pack_ternary input")
    sign = pack_pm1(x > 0)  # sign bit set where value > 0 (the 0/-1 entries are gated by the nz mask)
    nz = pack_pm1(x != 0)
    return sign, nz
