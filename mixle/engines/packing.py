"""Sub-byte bit-packing -- genuine fp1/fp2/fp4 (and below-byte index) STORAGE compression in pure numpy.

The low-bit end of mixle's spectrum is a storage win, not a CPU compute speedup (sub-byte arithmetic has
no native CPU support and dequant-to-fp32 is slower than fp32; that fast-dequant kernel is the Cython/C
tail). But the *packing* -- cramming ``bits``-wide codes into bytes -- vectorizes cleanly with numpy
shifts and is what actually shrinks the bytes on disk / on the wire. Power-of-two widths {1,2,4,8} pack
exactly ``8/bits`` codes per byte; this is the codec :class:`~mixle.engines.formats.CodebookFormat` and
the low-bit float formats use to realize their advertised compression ratio.
"""

from __future__ import annotations

from typing import Any

import numpy as np

_SUPPORTED = (1, 2, 4, 8)


def _require_exact_int(value: Any, name: str) -> int:
    """Coerce ``value`` to a plain ``int``, rejecting bools, non-numeric types, non-finite floats, and any
    value with a fractional part.

    A blind ``int(value)`` truncation would silently change behavior instead of failing loudly.
    """
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


def _require_exact_nonneg_int_array(values: Any, name: str) -> np.ndarray:
    """Validate ``values`` are exact, finite, nonnegative integers and return them as ``uint64``.

    Casting straight to ``uint64`` (the previous behavior) silently truncates fractional codes
    (``1.9`` -> ``1``) and wraps negative codes into huge unsigned values instead of failing loudly.
    """
    arr = np.asarray(values)
    if arr.dtype == np.bool_:
        return arr.astype(np.uint64)
    if np.issubdtype(arr.dtype, np.integer):
        if arr.size and np.any(arr < 0):
            raise ValueError(f"{name} must be nonnegative, got a negative value")
        return arr.astype(np.uint64)
    if np.issubdtype(arr.dtype, np.floating):
        if arr.size and not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must be finite, got NaN/Inf")
        if arr.size and not np.array_equal(arr, np.trunc(arr)):
            raise ValueError(f"{name} must be exact integers, got a fractional value")
        if arr.size and np.any(arr < 0):
            raise ValueError(f"{name} must be nonnegative, got a negative value")
        return arr.astype(np.uint64)
    # object / mixed dtype -- fall back to an elementwise check so no exotic input silently coerces.
    flat = arr.ravel()
    checked = np.empty(flat.size, dtype=np.uint64)
    for i, v in enumerate(flat):
        checked[i] = _require_nonnegative_int(v, name)
    return checked.reshape(arr.shape)


def _require_width(bits: Any, operation: str) -> int:
    bits = _require_exact_int(bits, "bits")
    if bits not in _SUPPORTED:
        raise ValueError("%s supports bit widths %r, got %d" % (operation, _SUPPORTED, bits))
    return bits


def _require_byte_array(values: Any) -> np.ndarray:
    """Validate packed bytes before conversion so fractions never truncate and integers never wrap."""
    raw = np.asarray(values)
    if raw.dtype == np.bool_:
        raise ValueError("packed must contain bytes, not booleans")
    checked = _require_exact_nonneg_int_array(raw, "packed")
    if checked.size and np.any(checked > 255):
        raise ValueError("packed values must be bytes in [0, 255]")
    return checked.astype(np.uint8).ravel()


def pack_bits(codes: Any, bits: int) -> np.ndarray:
    """Pack unsigned ``codes`` (each ``< 2**bits``) into a ``uint8`` array, ``8//bits`` per byte.

    ``bits`` must be a power of two in {1, 2, 4, 8} (the widths that tile a byte exactly). Little-endian
    within each byte: code ``j`` of a group occupies bit positions ``[j*bits, (j+1)*bits)``.
    """
    bits = _require_width(bits, "pack_bits")
    c = _require_exact_nonneg_int_array(codes, "codes").ravel()
    if np.any(c >= (1 << bits)):
        raise ValueError("a code does not fit in %d bits" % bits)
    if bits == 8:
        return c.astype(np.uint8)
    per_byte = 8 // bits
    pad = (-c.size) % per_byte
    if pad:
        c = np.concatenate([c, np.zeros(pad, dtype=np.uint64)])
    groups = c.reshape(-1, per_byte).astype(np.uint8)
    packed = np.zeros(groups.shape[0], dtype=np.uint8)
    for j in range(per_byte):
        packed |= groups[:, j] << np.uint8(j * bits)
    return packed


def unpack_bits(packed: Any, bits: int, count: int) -> np.ndarray:
    """Inverse of :func:`pack_bits`: recover the first ``count`` codes as a ``uint64`` array.

    ``count`` must be an exact nonnegative integer that fits within the packed payload's capacity; a
    negative ``count`` is rejected rather than falling through to Python's negative-slice semantics
    (``arr[:-3]``), and a ``count`` beyond the payload's actual capacity is rejected rather than silently
    returning fewer values than requested.
    """
    bits = _require_width(bits, "unpack_bits")
    count = _require_nonnegative_int(count, "count")
    p = _require_byte_array(packed)
    per_byte = 1 if bits == 8 else 8 // bits
    capacity = p.size * per_byte
    if count > capacity:
        raise ValueError("count %d exceeds packed capacity %d (short by %d)" % (count, capacity, count - capacity))
    if bits == 8:
        return p.astype(np.uint64)[:count]
    mask = np.uint8((1 << bits) - 1)
    out = np.empty((p.size, per_byte), dtype=np.uint8)
    for j in range(per_byte):
        out[:, j] = (p >> np.uint8(j * bits)) & mask
    return out.ravel()[:count].astype(np.uint64)


def packed_nbytes(count: int, bits: int) -> int:
    """Number of bytes :func:`pack_bits` produces for ``count`` codes of width ``bits``."""
    bits = _require_width(bits, "packed_nbytes")
    count = _require_nonnegative_int(count, "count")
    per_byte = 8 // bits
    return (count + per_byte - 1) // per_byte
