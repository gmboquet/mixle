"""Numeric format codecs across mixle's precision spectrum: low-bit float, fixed-point, and
codebook/VQ -- the compression + "fixed precision" end, each with a provable error bound so a
precision-allocation pass can pick the smallest format that preserves accuracy.

A :class:`NumericFormat` is a lossy codec: ``quantize`` maps a ``float64`` array to a compact
representation, ``dequantize`` maps it back, and ``max_abs_error`` / ``max_rel_error`` bound the round
trip. The bound is the "logic" a caller uses to spend minimal bits while keeping error under a target
(see :func:`min_float_mantissa_bits`). Fixed-point and codebook codecs store an actually smaller array
(real compression, vectorized in numpy); the float codec rounds to an ``n``-bit float's representable set
to *measure* that band's accuracy (true sub-byte bit-packing is the Cython/C tail).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

# Exponent-bit width per total float width (IEEE-ish: binary16/32/64/128 use 5/8/11/15; fp8 e4m3 uses 4).
_EXP_BITS = {1: 0, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 4, 16: 5, 32: 8, 64: 11, 128: 15, 256: 19}


def _exp_bits_for(total_bits: int) -> int:
    if total_bits in _EXP_BITS:
        return _EXP_BITS[total_bits]
    # IEEE 754-2019 interchange formula for binary-k (k >= 128): w = round(4*log2(k)) - 13.
    return max(2, min(total_bits - 1, round(4 * math.log2(total_bits)) - 13))


class NumericFormat:
    """Base codec: ``quantize`` / ``dequantize`` plus round-trip error bounds and a storage bit count."""

    name = "identity"
    bits_per_value = 64.0

    def quantize(self, x: Any) -> Any:  # pragma: no cover - overridden
        """Encode values into the format's storage representation."""
        raise NotImplementedError

    def dequantize(self, q: Any) -> np.ndarray:  # pragma: no cover - overridden
        """Decode stored values back to float64."""
        raise NotImplementedError

    def round_trip(self, x: Any) -> np.ndarray:
        """Quantize and immediately dequantize values."""
        return self.dequantize(self.quantize(x))

    def measured_max_abs_error(self, x: Any) -> float:
        """Empirical max absolute round-trip error over ``x`` (codecs also expose analytic bounds)."""
        x = np.asarray(x, dtype=np.float64)
        return float(np.max(np.abs(self.round_trip(x) - x))) if x.size else 0.0

    def compression_ratio(self) -> float:
        """Stored bits per value relative to float64."""
        return 64.0 / self.bits_per_value


class FloatFormat(NumericFormat):
    """A float with ``mantissa_bits`` of mantissa -- rounds ``x`` to that band's representable set.

    Below ~52 mantissa bits this is lossy (fp4/fp8/fp16 storage); the round trip *measures* the band's
    accuracy. ``max_rel_error == 2**-(mantissa_bits+1)`` from round-to-nearest.
    """

    def __init__(self, mantissa_bits: int, exp_bits: int = 11) -> None:
        self.mantissa_bits = int(mantissa_bits)
        self.exp_bits = int(exp_bits)
        self.name = "fp%d" % (1 + self.exp_bits + self.mantissa_bits)
        self.bits_per_value = float(1 + self.exp_bits + self.mantissa_bits)

    @classmethod
    def fp(cls, total_bits: int) -> FloatFormat:
        """Build an ``n``-bit float format with an IEEE-like exponent/mantissa split (n = 1..1024+)."""
        exp = _exp_bits_for(int(total_bits))
        mant = max(0, int(total_bits) - 1 - exp)
        return cls(mantissa_bits=mant, exp_bits=exp)

    @property
    def max_rel_error(self) -> float:
        """Return the nominal worst-case relative rounding error."""
        return 2.0 ** -(self.mantissa_bits + 1)

    def quantize(self, x: Any) -> np.ndarray:
        """Round values to this floating mantissa precision."""
        x = np.asarray(x, dtype=np.float64)
        if self.mantissa_bits >= 52:
            return x.copy()  # float64 already carries 52 mantissa bits
        m, e = np.frexp(x)  # x == m * 2**e, m in [0.5, 1)
        # Round the significand to mantissa_bits explicit bits + 1 implicit leading bit, so the
        # round-to-nearest relative error is 2**-(mantissa_bits+1), matching ``max_rel_error``.
        scale = float(1 << (self.mantissa_bits + 1))
        return np.ldexp(np.round(m * scale) / scale, e)

    def dequantize(self, q: Any) -> np.ndarray:
        """Decode quantized floating values to float64."""
        return np.asarray(q, dtype=np.float64)


class FixedPointFormat(NumericFormat):
    """Fixed-point: store ``round(x * 2**frac_bits)`` as an integer; real compression + a hard error bound.

    ``int_bits`` sets the representable magnitude ``[-2**int_bits, 2**int_bits)``; out-of-range clamps.
    ``max_abs_error == 2**-(frac_bits+1)`` (round-to-nearest), independent of ``x``.
    """

    def __init__(self, frac_bits: int, int_bits: int = 31) -> None:
        self.frac_bits = int(frac_bits)
        self.int_bits = int(int_bits)
        total = 1 + self.int_bits + self.frac_bits
        self.name = "fixed(i%d.f%d)" % (self.int_bits, self.frac_bits)
        self.bits_per_value = float(total)
        self._scale = float(2**self.frac_bits)
        self._limit = 2 ** (self.int_bits + self.frac_bits)  # max magnitude in scaled integer units
        self._store_dtype = np.int32 if total <= 32 else np.int64

    @property
    def max_abs_error(self) -> float:  # type: ignore[override]
        """Return the fixed-point half-step absolute error bound."""
        return 2.0 ** -(self.frac_bits + 1)

    def quantize(self, x: Any) -> np.ndarray:
        """Encode values as clipped scaled integers."""
        x = np.asarray(x, dtype=np.float64)
        scaled = np.round(x * self._scale)
        np.clip(scaled, -self._limit, self._limit - 1, out=scaled)
        return scaled.astype(self._store_dtype)

    def dequantize(self, q: Any) -> np.ndarray:
        """Decode scaled integers back to float64 values."""
        return np.asarray(q, dtype=np.float64) / self._scale


class CodebookFormat(NumericFormat):
    """Scalar vector-quantization: store an index into a learned codebook; ``log2(K)`` bits per value.

    The genuine pure-numpy compression codec -- quantize gathers the nearest code (an unsigned index
    array), dequantize gathers the code values back. Fit the codebook to data with :meth:`fit`.
    """

    def __init__(self, codebook: Any) -> None:
        self.codebook = np.asarray(codebook, dtype=np.float64)
        self.codebook.sort()  # sorted codes let quantize use searchsorted (O(n log K))
        k = self.codebook.size
        self.name = "codebook(K=%d)" % k
        self.bits_per_value = float(max(1, math.ceil(math.log2(max(2, k)))))
        self._idx_dtype = np.uint8 if k <= 256 else (np.uint16 if k <= 65536 else np.uint32)

    @classmethod
    def fit(cls, data: Any, n_codes: int, iters: int = 25, seed: int = 0) -> CodebookFormat:
        """Learn ``n_codes`` codes by 1-D k-means (Lloyd) on ``data``; codes are the cluster means."""
        x = np.asarray(data, dtype=np.float64).ravel()
        if x.size == 0:
            return cls(np.zeros(1))
        n_codes = int(min(n_codes, np.unique(x).size))
        # init at quantiles (a good 1-D start), then refine.
        centers = np.quantile(x, np.linspace(0.0, 1.0, n_codes)) if n_codes > 1 else np.array([x.mean()])
        for _ in range(iters):
            edges = (centers[:-1] + centers[1:]) / 2.0
            idx = np.searchsorted(edges, x)
            new = centers.copy()
            for k in range(n_codes):
                sel = x[idx == k]
                if sel.size:
                    new[k] = sel.mean()
            if np.allclose(new, centers):
                break
            centers = new
        return cls(centers)

    def quantize(self, x: Any) -> np.ndarray:
        """Map values to nearest codebook indices."""
        x = np.asarray(x, dtype=np.float64)
        edges = (self.codebook[:-1] + self.codebook[1:]) / 2.0
        idx = np.searchsorted(edges, x)  # nearest code by the sorted-codebook midpoints
        return idx.astype(self._idx_dtype)

    def dequantize(self, q: Any) -> np.ndarray:
        """Map codebook indices back to representative values."""
        return self.codebook[np.asarray(q, dtype=np.intp)]

    def _pack_bits(self) -> int:
        """Power-of-two index width used by :meth:`compress`; rounds ``bits_per_value`` up to {1,2,4,8}."""
        b = int(self.bits_per_value)
        return next(w for w in (1, 2, 4, 8) if w >= b) if b <= 8 else 8

    def compress(self, x: Any) -> tuple[np.ndarray, int]:
        """Quantize ``x`` and bit-pack the indices to bytes: returns ``(packed_uint8, count)``.

        For ``K <= 16`` codes the indices are sub-byte and pack ``8//bits`` per byte, realizing the
        advertised compression (e.g. 16 codes -> 4-bit indices -> 2 values/byte -> 16x vs float64).
        """
        from mixle.engines.packing import pack_bits

        idx = self.quantize(x)
        return pack_bits(idx, self._pack_bits()), int(np.asarray(x).size)

    def decompress(self, packed: Any, count: int) -> np.ndarray:
        """Inverse of :meth:`compress`: unpack indices and gather the codebook back to ``float64``."""
        from mixle.engines.packing import unpack_bits

        return self.dequantize(unpack_bits(packed, self._pack_bits(), count))


def min_float_mantissa_bits(target_rel_error: float) -> int:
    """Smallest mantissa-bit count whose round-to-nearest relative error meets ``target_rel_error``.

    The error-tracing primitive: given a tolerated relative error, return the minimal float precision
    that preserves it -- i.e. spend the fewest bits the accuracy budget allows.
    """
    if target_rel_error <= 0:
        raise ValueError("target_rel_error must be positive")
    return max(0, math.ceil(-math.log2(target_rel_error) - 1))
