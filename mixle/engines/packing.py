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


def pack_bits(codes: Any, bits: int) -> np.ndarray:
    """Pack unsigned ``codes`` (each ``< 2**bits``) into a ``uint8`` array, ``8//bits`` per byte.

    ``bits`` must be a power of two in {1, 2, 4, 8} (the widths that tile a byte exactly). Little-endian
    within each byte: code ``j`` of a group occupies bit positions ``[j*bits, (j+1)*bits)``.
    """
    if bits not in _SUPPORTED:
        raise ValueError("pack_bits supports bit widths %r, got %d" % (_SUPPORTED, bits))
    c = np.asarray(codes, dtype=np.uint64).ravel()
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
    """Inverse of :func:`pack_bits`: recover the first ``count`` codes as a ``uint64`` array."""
    if bits not in _SUPPORTED:
        raise ValueError("unpack_bits supports bit widths %r, got %d" % (_SUPPORTED, bits))
    p = np.asarray(packed, dtype=np.uint8).ravel()
    if bits == 8:
        return p.astype(np.uint64)[:count]
    per_byte = 8 // bits
    mask = np.uint8((1 << bits) - 1)
    out = np.empty((p.size, per_byte), dtype=np.uint8)
    for j in range(per_byte):
        out[:, j] = (p >> np.uint8(j * bits)) & mask
    return out.ravel()[:count].astype(np.uint64)


def packed_nbytes(count: int, bits: int) -> int:
    """Number of bytes :func:`pack_bits` produces for ``count`` codes of width ``bits``."""
    if bits not in _SUPPORTED:
        raise ValueError("packed_nbytes supports bit widths %r, got %d" % (_SUPPORTED, bits))
    per_byte = 8 // bits
    return (count + per_byte - 1) // per_byte
