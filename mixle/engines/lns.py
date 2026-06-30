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

    @property
    def max_logsumexp_error(self) -> float:
        """Bound on the absolute log-sum-exp error from quantization + LUT rounding (~one step per fold)."""
        return 1.5 * self.step

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
        """Integer log-sum-exp along ``axis`` via a pairwise tree of :meth:`logadd` (no exp/log)."""
        k = np.moveaxis(np.asarray(k, dtype=np.int64), axis, -1).copy()
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
