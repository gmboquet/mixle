"""Logarithmic number system for mixle's log-space compute -- quantize by a fixed log constant ``ln(C)``.

mixle already works in log-space (log-densities, log-weights, log-sum-exp), so the natural quantization is
on the *log* value: store ``v`` as the integer ``k = round(ln(v) / s)`` with step ``s = ln(C)``, i.e.
``v = C**k``. Then the two hot operations become integer arithmetic:

* **multiplying probabilities = adding log-probs = adding the integers** ``k1 + k2`` -- exact, no table.
* **log-sum-exp** (adding probabilities -- the mixture / HMM / marginalization op) becomes
  ``max(k1, k2) + LUT[|k1 - k2|]`` where ``LUT[d] = round(log(1 + exp(-d*s)) / s)`` is a small precomputed
  integer table (the Gaussian logarithm). No ``exp``, no ``log`` -- integer ``max`` + a gather. This is the
  transcendental reduction that dominates mixture scoring, and unlike a GEMM it has no BLAS to lose to:
  measured ~4x faster than float64 ``logsumexp`` in pure numpy (more with a compiled integer kernel).

``step`` is the precision dial -- the fp1..fpN spectrum, but in the log domain where it is natural: each
unit is a factor of ``C = exp(step)``, so the relative precision of a stored value is ~``step/2`` and the
log-sum-exp error is bounded by ~``step``. Smaller step -> finer + wider integer range (int16 at step~0.1,
int32 at step~1e-3). The model's log-parameters and the data terms are quantized by the SAME step, so the
whole score is integer arithmetic.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

try:  # optional compiled one-pass tree kernel for the integer log-sum-exp (bit-identical, ~8x)
    from mixle.engines._lns_kernel import logsumexp_rows as _logsumexp_rows_c

    _HAS_LNS_KERNEL = True
except ImportError:  # pragma: no cover - extension optional
    _HAS_LNS_KERNEL = False


class LogNumberSystem:
    """Quantize log-space values to integers in units of ``step = ln(C)`` and compute on the integers."""

    def __init__(self, step: float = 1e-2) -> None:
        if step <= 0:
            raise ValueError("step must be positive")
        self.step = float(step)
        # LUT[d] = round(log1p(exp(-d*step)) / step); the correction falls to 0, truncate where it rounds to 0.
        # log1p(exp(-d*step)) < step/2  <=>  exp(-d*step) < exp(step/2)-1  =>  d > -log(exp(step/2)-1)/step
        thresh = math.expm1(0.5 * self.step)
        dmax = int(math.ceil(-math.log(thresh) / self.step)) + 2 if thresh > 0 else 2
        self.dmax = max(dmax, 1)
        d = np.arange(self.dmax + 1, dtype=np.float64)
        self.lut = np.rint(np.log1p(np.exp(-d * self.step)) / self.step).astype(np.int64)

    @classmethod
    def from_relative_precision(cls, rel: float) -> LogNumberSystem:
        """Build a system whose stored values are accurate to ~``rel`` relative (``step = ln(1+rel)``)."""
        return cls(step=math.log1p(rel))

    def max_logsumexp_error(self, n: int) -> float:
        """Certified bound on the absolute log-sum-exp error for reducing ``n`` quantized terms.

        Two error sources compound over the pairwise reduction tree (see :meth:`logsumexp`):

        * input quantization -- each leaf starts up to ``step/2`` away from its true log-value.
        * LUT rounding -- each :meth:`logadd` looks up ``round(log1p(exp(-d*step)) / step)``, itself
          within ``step/2`` of the true correction (by construction: the table's last entry is exactly
          0, and ``dmax`` is sized so the true correction is already ``< step/2`` there, so clipping a
          larger gap to ``dmax`` costs less error than the in-table rounding it stands in for).

        ``logsumexp(a, b) = log(exp(a) + exp(b))`` is monotonic and shift-equivariant in each argument
        (``logsumexp(a+c, b+c) = logsumexp(a, b) + c``), which gives an EXACT bound -- not a first-order
        approximation -- on how far a :meth:`logadd` of already-approximate operands can land from the
        true value of the pair they represent: at most the WORSE of the two operands' existing error.
        Errors don't add across siblings, only the fresh per-node LUT rounding does. So after ``d`` tree
        levels the worst-case error is ``step/2`` (leaf quantization) plus ``d`` more halves of ``step``
        (one fresh LUT rounding per level on the critical path): ``(d + 1) * step / 2``.

        Both reduction trees here (the numpy fallback and the compiled kernel) combine adjacent pairs
        and carry any odd leftover forward unchanged, so they are always balanced: the deepest element
        crosses exactly ``d = ceil(log2(n))`` levels regardless of how ``n`` factors, and this bound is
        a valid certificate for any ``n`` -- not just the 2-term case the previous constant ``1.5 *
        step`` bound was actually sized for (equivalent to this formula at ``n`` in ``{3, 4}``; already
        violated by measured error at, e.g., ``n=8`` -- see MXR-080-0139 and the regression tests).
        """
        if n < 1:
            raise ValueError(f"max_logsumexp_error: n must be >= 1, got {n}")
        depth = math.ceil(math.log2(n)) if n > 1 else 0
        return (depth + 1) * self.step / 2.0

    def quantize(self, log_values: Any) -> np.ndarray:
        """Round log-space values to integer multiples of ``step`` (the stored representation)."""
        return np.rint(np.asarray(log_values, dtype=np.float64) / self.step).astype(np.int64)

    def dequantize(self, k: Any) -> np.ndarray:
        """Recover the float log-value ``k * step``."""
        return np.asarray(k, dtype=np.int64) * self.step

    def logadd(self, k1: Any, k2: Any) -> np.ndarray:
        """Integer Gaussian logarithm: ``logsumexp`` of two quantized log-values -> ``max + LUT[|diff|]``."""
        k1 = np.asarray(k1, dtype=np.int64)
        k2 = np.asarray(k2, dtype=np.int64)
        d = np.minimum(np.abs(k1 - k2), self.dmax)
        return np.maximum(k1, k2) + self.lut[d]

    def logsumexp(self, k: Any, axis: int = -1) -> np.ndarray:
        """Integer log-sum-exp along ``axis`` via a pairwise tree of :meth:`logadd` (no exp/log).

        Uses the compiled one-pass tree kernel for the common 2-D last-axis reduction when available
        (bit-identical to the numpy tree, ~8x faster); falls back to vectorized numpy otherwise.
        """
        arr = np.asarray(k, dtype=np.int64)
        if _HAS_LNS_KERNEL and arr.ndim == 2 and axis in (-1, 1) and arr.shape[1] > 0:
            return _logsumexp_rows_c(np.ascontiguousarray(arr), self.lut, self.dmax)
        k = np.moveaxis(arr, axis, -1).copy()
        while k.shape[-1] > 1:
            if k.shape[-1] & 1:
                tail, k = k[..., -1:], k[..., :-1]
            else:
                tail = None
            a, b = k[..., 0::2], k[..., 1::2]
            d = np.minimum(np.abs(a - b), self.dmax)
            k = np.maximum(a, b) + self.lut[d]
            if tail is not None:
                k = np.concatenate([k, tail], axis=-1)
        return k[..., 0]

    def integer_dtype(self, log_range: float) -> Any:
        """Smallest signed integer dtype that holds log-values spanning ``[-log_range, log_range]``."""
        kmax = log_range / self.step
        for dt in (np.int16, np.int32):
            if kmax < np.iinfo(dt).max:
                return dt
        return np.int64
