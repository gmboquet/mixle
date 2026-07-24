"""Numeric format codecs across mixle's precision spectrum: low-bit float, fixed-point, and
codebook/VQ -- the compression + "fixed precision" end, each with a provable error bound so a
precision-allocation pass can pick the smallest format that preserves accuracy.

A :class:`NumericFormat` is a lossy codec: ``quantize`` maps a ``float64`` array to a compact
representation, ``dequantize`` maps it back, and ``max_abs_error`` / ``max_rel_error`` bound the round
trip. The bound is the "logic" a caller uses to spend minimal bits while keeping error under a target
(see :func:`min_float_mantissa_bits`). Fixed-point and codebook codecs store an actually smaller array
(real compression, vectorized in numpy) with a hard range/index limit enforced at construction or
quantize time; the float codec instead rounds each value's MANTISSA to an ``n``-bit band's precision with
an unbounded exponent (no overflow/underflow/subnormal handling -- see :class:`FloatFormat`) to *measure*
that band's rounding accuracy at any magnitude (true sub-byte bit-packing, and a real range-limited fpN
encoding, are the Cython/C tail).
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
    """Round-to-nearest MANTISSA rounding at ``mantissa_bits``, nominally styled after an IEEE-ish fpN
    split (see :meth:`fp`).

    ``quantize`` rounds each value's significand to ``mantissa_bits`` explicit bits (round-to-nearest);
    the exponent passes through UNCHANGED, with no range limit. ``exp_bits`` therefore only feeds
    ``name`` / ``bits_per_value`` (the advertised bit budget an allocation pass compares against) -- it is
    NOT enforced: there is no overflow/underflow, no saturation or flush-to-zero, no subnormals, and no
    NaN/Inf bit-pattern encoding. This is a deliberate scope choice, not an oversight: an unbounded
    exponent is what makes ``max_rel_error`` a genuine bound for *any* finite magnitude (see that
    property's docstring), so a caller can read the round trip as a pure mantissa-rounding measurement
    independent of range.

    A consequence: this codec does NOT faithfully emulate a real bounded fpN storage format. A real 8-bit
    float cannot represent ``1e100`` at all, but ``FloatFormat.fp(8).round_trip([1e100])`` rounds
    ``1e100``'s mantissa to 3 bits and returns a value that is still ~1e100 in magnitude -- by design, per
    the above, not a bug. Do not use ``fp(8)``/``fp(16)``/etc. where actual overflow/underflow/saturation
    behavior matters; they measure mantissa-rounding error only.
    """

    def __init__(self, mantissa_bits: int, exp_bits: int = 11) -> None:
        self.mantissa_bits = int(mantissa_bits)
        self.exp_bits = int(exp_bits)
        total_bits = 1 + self.exp_bits + self.mantissa_bits
        # "nominal_fpN": N is the IEEE-ish total bit count exp_bits/mantissa_bits would imply for a real
        # bounded float, but only mantissa_bits is actually enforced by quantize -- see the class
        # docstring. Deliberately not named "fpN" alone: that would claim a bounded-range storage format
        # this codec does not implement.
        self.name = "mantissa_round(m%d,nominal_fp%d)" % (self.mantissa_bits, total_bits)
        self.bits_per_value = float(total_bits)

    @classmethod
    def fp(cls, total_bits: int) -> FloatFormat:
        """Build a format whose ``mantissa_bits``/``exp_bits`` follow an IEEE-like split of
        ``total_bits`` (n = 1..1024+). Only the resulting ``mantissa_bits`` affects ``quantize``'s
        numeric behavior -- ``exp_bits`` is bit-budget bookkeeping only (see the class docstring); this
        does NOT build a range-limited fpN codec.
        """
        exp = _exp_bits_for(int(total_bits))
        mant = max(0, int(total_bits) - 1 - exp)
        return cls(mantissa_bits=mant, exp_bits=exp)

    @property
    def max_rel_error(self) -> float:
        """Return the worst-case relative rounding error -- a universal bound, true for every finite
        input regardless of magnitude (the exponent is never range-limited; see the class docstring)."""
        return 2.0 ** -(self.mantissa_bits + 1)

    def quantize(self, x: Any) -> np.ndarray:
        """Round values to this mantissa precision; the exponent passes through exactly, unclamped."""
        x = np.asarray(x, dtype=np.float64)
        if self.mantissa_bits >= 52:
            return x.copy()  # float64 already carries 52 mantissa bits
        m, e = np.frexp(x)  # x == m * 2**e, m in [0.5, 1)
        # Round the significand to mantissa_bits explicit bits + 1 implicit leading bit, so the
        # round-to-nearest relative error is 2**-(mantissa_bits+1), matching ``max_rel_error`` for any
        # finite x -- e passes through unclamped, so no magnitude can overflow or underflow this rounding.
        scale = float(1 << (self.mantissa_bits + 1))
        return np.ldexp(np.round(m * scale) / scale, e)

    def dequantize(self, q: Any) -> np.ndarray:
        """Decode quantized floating values to float64."""
        return np.asarray(q, dtype=np.float64)


class FixedPointFormat(NumericFormat):
    """Fixed-point: store ``round(x * 2**frac_bits)`` as an integer; real compression + a hard error bound.

    ``int_bits`` sets the representable magnitude ``[-2**int_bits, 2**int_bits)``; out-of-range clamps.
    ``max_abs_error == 2**-(frac_bits+1)`` (round-to-nearest), independent of ``x``. Storage is a plain
    ``int32``/``int64`` -- there is no arbitrary-width packed storage -- so the declared total width
    ``1 + int_bits + frac_bits`` cannot exceed 64 bits; wider requests raise at construction instead of
    silently overflowing the int64 cast (and falsifying ``max_abs_error``).
    """

    _MAX_TOTAL_BITS = 64  # storage ceiling: this codec only ever picks int32 or int64, nothing wider

    def __init__(self, frac_bits: int, int_bits: int = 31) -> None:
        self.frac_bits = int(frac_bits)
        self.int_bits = int(int_bits)
        total = 1 + self.int_bits + self.frac_bits
        if total > self._MAX_TOTAL_BITS:
            raise ValueError(
                "fixed(i%d.f%d) declares %d bits, but FixedPointFormat only supports up to %d bits "
                "(int64 storage; no arbitrary-width packed storage is implemented) -- construct a "
                "narrower format" % (self.int_bits, self.frac_bits, total, self._MAX_TOTAL_BITS)
            )
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


def _require_positive_int(name: str, value: Any) -> int:
    """Validate a positive, exact-integer count (e.g. ``n_codes``/``iters``); returns it as ``int``."""
    try:
        as_int = int(value)
    except (TypeError, ValueError):
        raise ValueError("%s must be a positive integer, got %r" % (name, value)) from None
    if as_int != value or as_int <= 0:
        raise ValueError("%s must be a positive integer, got %r" % (name, value))
    return as_int


class CodebookFormat(NumericFormat):
    """Scalar vector-quantization: store an index into a learned codebook; ``log2(K)`` bits per value.

    The genuine pure-numpy compression codec -- quantize gathers the nearest code (an unsigned index
    array), dequantize gathers the code values back. Fit the codebook to data with :meth:`fit`.

    The codebook must be finite and nonempty (validated at construction, ``ValueError`` otherwise).
    Duplicate entries are allowed: a repeated code costs a wasted, unreachable index but does not break
    nearest-code assignment or decoding (both entries decode to the same, correct value), and
    :meth:`fit`'s Lloyd iteration can legitimately converge two cluster means onto the same value on
    degenerate/skewed data -- rejecting that outcome would be pure loss, not a correctness fix.
    """

    def __init__(self, codebook: Any) -> None:
        self.codebook = np.asarray(codebook, dtype=np.float64)
        if self.codebook.size == 0:
            raise ValueError("codebook must be nonempty")
        if not np.all(np.isfinite(self.codebook)):
            raise ValueError("codebook entries must be finite (no NaN/Inf)")
        self.codebook.sort()  # sorted codes let quantize use searchsorted (O(n log K))
        k = self.codebook.size
        self.name = "codebook(K=%d)" % k
        self.bits_per_value = float(max(1, math.ceil(math.log2(max(2, k)))))
        self._idx_dtype = np.uint8 if k <= 256 else (np.uint16 if k <= 65536 else np.uint32)

    @classmethod
    def fit(cls, data: Any, n_codes: int, iters: int = 25, seed: int = 0) -> CodebookFormat:
        """Learn ``n_codes`` codes by 1-D k-means (Lloyd) on ``data``; codes are the cluster means.

        Raises ``ValueError`` if ``n_codes`` or ``iters`` is not a positive integer.
        """
        n_codes = _require_positive_int("n_codes", n_codes)
        iters = _require_positive_int("iters", iters)
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
        """Index width used by :meth:`compress`: sub-byte {1,2,4,8} widths for ``K <= 256`` (bit-packed
        via :mod:`mixle.engines.packing`), else a byte-aligned 16 or 32 bits matching ``_idx_dtype``.

        Wide codebooks are NOT capped down to 8 bits: every code :meth:`quantize` can produce also fits
        through :meth:`compress` (a 300-code format's code 299 needs 9 bits and gets a 16-bit slot, not a
        forced-and-broken 8-bit one).
        """
        b = int(self.bits_per_value)
        if b <= 8:
            return next(w for w in (1, 2, 4, 8) if w >= b)
        return 16 if b <= 16 else 32

    def compress(self, x: Any) -> tuple[np.ndarray, int]:
        """Quantize ``x`` and pack the indices to bytes: returns ``(packed_uint8, count)``.

        For ``K <= 256`` codes the indices are sub-byte or byte-exact and :func:`~mixle.engines.packing.
        pack_bits` bit-packs them (e.g. 16 codes -> 4-bit indices -> 2 values/byte -> 16x vs float64).
        Larger codebooks need 16- or 32-bit indices, which are not sub-byte-packable -- those are stored
        byte-aligned (little-endian), still returned as a flat ``uint8`` array.
        """
        idx = self.quantize(x)
        bits = self._pack_bits()
        if bits <= 8:
            from mixle.engines.packing import pack_bits

            packed = pack_bits(idx, bits)
        else:
            dtype = np.dtype("<u2") if bits == 16 else np.dtype("<u4")
            packed = idx.astype(dtype).view(np.uint8)
        return packed, int(np.asarray(x).size)

    def decompress(self, packed: Any, count: int) -> np.ndarray:
        """Inverse of :meth:`compress`: unpack indices and gather the codebook back to ``float64``."""
        bits = self._pack_bits()
        if bits <= 8:
            from mixle.engines.packing import unpack_bits

            idx = unpack_bits(packed, bits, count)
        else:
            dtype = np.dtype("<u2") if bits == 16 else np.dtype("<u4")
            idx = np.frombuffer(np.asarray(packed, dtype=np.uint8).tobytes(), dtype=dtype)[:count]
        return self.dequantize(idx)


def min_float_mantissa_bits(target_rel_error: float) -> int:
    """Smallest mantissa-bit count whose round-to-nearest relative error meets ``target_rel_error``.

    The error-tracing primitive: given a tolerated relative error, return the minimal float precision
    that preserves it -- i.e. spend the fewest bits the accuracy budget allows.
    """
    if target_rel_error <= 0:
        raise ValueError("target_rel_error must be positive")
    return max(0, math.ceil(-math.log2(target_rel_error) - 1))
