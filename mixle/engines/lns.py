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

A true zero (``log(0) = -inf``) has no finite code, so it gets one reserved sentinel, ``LOG_ZERO_CODE``,
and every arithmetic entry point (``quantize``, ``logadd``, ``multiply``, the compiled kernel) special
cases it explicitly before doing any int64 math -- see the constants below and MXR-080-0138.
"""

from __future__ import annotations

import math
from typing import Any, Final

import numpy as np

from mixle.engines._optional_extension import load_optional_extension

_LNS_EXTENSION = load_optional_extension("mixle.engines._lns_kernel", ("logsumexp_rows",))
_HAS_LNS_KERNEL = _LNS_EXTENSION.available
LNS_EXTENSION_DIAGNOSTIC = _LNS_EXTENSION.diagnostic
if _HAS_LNS_KERNEL:  # pragma: no cover - depends on the optional local build
    (_logsumexp_rows_c,) = _LNS_EXTENSION.values


# --- MXR-080-0138: log-zero sentinel + saturating code range -------------------------------------------
#
# An LNS code is `round(log(v) / step)`. Casting a non-finite (log of exactly zero, or NaN) or wildly
# out-of-range log-value straight to int64 is not something this module can rely on: numpy's actual
# behavior for that float->int64 cast is version/platform dependent (empirically, current numpy
# saturates -inf/+inf to int64 min/max and NaN to 0 -- but NaN silently becoming code 0, i.e. "log-value
# exactly 0", is itself a silent correctness bug, not just a cast-safety one). And even where the cast
# happens to saturate, two such extreme codes can still overflow when SUBTRACTED (in `logadd`) or ADDED
# (in `multiply`, i.e. log-space multiplication) -- an overflowed, wrapped-negative difference is exactly
# the kind of value that ends up as a raw LUT index in the boundscheck-disabled compiled kernel, reading
# out-of-bounds memory instead of raising. So log-zero gets one reserved, explicit sentinel code, and
# every other code saturates into a fixed safe range instead of ever being allowed to overflow int64:
#
#   LOG_ZERO_CODE  the ONLY code that means "log(0)". Never produced by rounding a float (quantize()
#                  detects exact -inf itself and substitutes this directly); never touched by raw int64
#                  arithmetic, since every operation below checks for it before doing any math -- so its
#                  literal value only has to be far from anything quantize() can otherwise produce, not
#                  safe-to-add-to.
#   CODE_MIN/MAX   the saturating range for every other code. `2**61` leaves 2 bits of headroom under
#                  int64's `2**63-1`: the sum OR difference of any two in-range codes stays inside int64
#                  (`2*2**61 = 2**62 < 2**63-1`), so clamping into this range before arithmetic makes
#                  overflow structurally impossible rather than merely unlikely. Both bounds are exact
#                  powers of two, so converting them to/from float64 (used mid-computation in `quantize`)
#                  is exact at any magnitude, unlike e.g. `2**61 - 1`.
#
# `_lns_kernel.pyx` duplicates these same three values as compiled constants (it cannot import them
# without a circular import, since this module conditionally imports the compiled kernel above it) --
# keep the two definitions in sync if either changes.
LOG_ZERO_CODE: Final = np.iinfo(np.int64).min
CODE_MAX: Final = np.int64(2**61)
CODE_MIN: Final = np.int64(-(2**61))


class LogNumberSystem:
    """Quantize log-space values to integers in units of ``step = ln(C)`` and compute on the integers."""

    LOG_ZERO_CODE = LOG_ZERO_CODE
    CODE_MIN = CODE_MIN
    CODE_MAX = CODE_MAX

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
        """Round log-space values to integer multiples of ``step`` (the stored representation).

        Exact ``-inf`` (the log of a true zero) maps to the reserved :data:`LOG_ZERO_CODE` sentinel --
        never through a float-to-int64 cast, whose behavior for a non-finite input this module does not
        rely on (see MXR-080-0138). NaN and ``+inf`` have no LNS code and raise. Any other value that
        would round outside ``[CODE_MIN, CODE_MAX]`` saturates to that boundary instead of overflowing.
        """
        x = np.asarray(log_values, dtype=np.float64)
        n_nan = int(np.isnan(x).sum())
        if n_nan:
            raise ValueError(f"quantize: log_values has {n_nan} NaN value(s), which have no LNS code")
        n_posinf = int(np.isposinf(x).sum())
        if n_posinf:
            raise ValueError(f"quantize: log_values has {n_posinf} +inf value(s), which have no LNS code")
        is_log_zero = np.isneginf(x)
        # np.clip on an array containing -inf is well-defined (it clamps to CODE_MIN), so every lane is
        # finite and within [CODE_MIN, CODE_MAX] before the int64 cast below -- the cast itself never has
        # to handle a non-finite or out-of-range float, regardless of numpy version/platform cast rules.
        clamped = np.clip(np.rint(x / self.step), float(CODE_MIN), float(CODE_MAX))
        codes = clamped.astype(np.int64)
        return np.where(is_log_zero, LOG_ZERO_CODE, codes).astype(np.int64)

    def dequantize(self, k: Any) -> np.ndarray:
        """Recover the float log-value ``k * step``; the reserved zero sentinel maps back to exactly ``-inf``."""
        k = np.asarray(k, dtype=np.int64)
        value = k.astype(np.float64) * self.step
        return np.where(k == LOG_ZERO_CODE, -np.inf, value)

    def logadd(self, k1: Any, k2: Any) -> np.ndarray:
        """Integer Gaussian logarithm: ``logsumexp`` of two quantized log-values -> ``max + LUT[|diff|]``.

        ``LOG_ZERO_CODE`` (log of an exact zero) is absorbing -- zero-plus-anything is the other operand,
        computed with no LUT lookup at all. Ordinary operands are clamped into ``[CODE_MIN, CODE_MAX]``
        first, so the ``|k1 - k2|`` difference below can never overflow int64 even for adversarial inputs
        (MXR-080-0138): an overflowed, wrapped-negative difference would otherwise become the LUT index
        a few lines down.
        """
        k1 = np.asarray(k1, dtype=np.int64)
        k2 = np.asarray(k2, dtype=np.int64)
        is_zero1 = k1 == LOG_ZERO_CODE
        is_zero2 = k2 == LOG_ZERO_CODE
        c1 = np.clip(k1, CODE_MIN, CODE_MAX)
        c2 = np.clip(k2, CODE_MIN, CODE_MAX)
        d = np.minimum(np.abs(c1 - c2), self.dmax)
        result = np.maximum(c1, c2) + self.lut[d]
        result = np.where(is_zero2, k1, result)
        result = np.where(is_zero1, k2, result)
        return result

    def multiply(self, k1: Any, k2: Any) -> np.ndarray:
        """Integer code for the PRODUCT of two LNS values: ``log(a*b) = log(a) + log(b)`` -> code add.

        Exact for ordinary operands (integer add, no LUT, no rounding). ``LOG_ZERO_CODE`` is absorbing
        (anything times zero is zero); an ordinary sum that would leave ``[CODE_MIN, CODE_MAX]`` saturates
        to that boundary rather than silently wrapping through int64 overflow (MXR-080-0138). Prefer this
        over raw ``k1 + k2`` for any code that might be (or derive from) a quantized true zero.
        """
        k1 = np.asarray(k1, dtype=np.int64)
        k2 = np.asarray(k2, dtype=np.int64)
        is_zero = (k1 == LOG_ZERO_CODE) | (k2 == LOG_ZERO_CODE)
        c1 = np.clip(k1, CODE_MIN, CODE_MAX)
        c2 = np.clip(k2, CODE_MIN, CODE_MAX)
        result = np.clip(c1 + c2, CODE_MIN, CODE_MAX)
        return np.where(is_zero, LOG_ZERO_CODE, result)

    def logsumexp(self, k: Any, axis: int = -1) -> np.ndarray:
        """Integer log-sum-exp along ``axis`` via a pairwise tree of :meth:`logadd` (no exp/log).

        Uses the compiled one-pass tree kernel for the common 2-D last-axis reduction when available
        (bit-identical to the numpy tree, ~8x faster); falls back to vectorized numpy otherwise. Both
        trees combine adjacent pairs and carry an odd leftover element forward unchanged, so the deepest
        element always makes exactly ``ceil(log2(m))`` logadd hops (``m`` = reduced axis length) -- see
        :meth:`max_logsumexp_error`, which bounds the resulting error in terms of that depth.
        """
        arr = np.asarray(k, dtype=np.int64)
        if _HAS_LNS_KERNEL and arr.ndim == 2 and axis in (-1, 1) and arr.shape[1] > 0:
            return _logsumexp_rows_c(np.ascontiguousarray(arr), self.lut, self.dmax, LOG_ZERO_CODE, CODE_MIN, CODE_MAX)
        k = np.moveaxis(arr, axis, -1).copy()
        while k.shape[-1] > 1:
            if k.shape[-1] & 1:
                tail, k = k[..., -1:], k[..., :-1]
            else:
                tail = None
            a, b = k[..., 0::2], k[..., 1::2]
            k = self.logadd(a, b)
            if tail is not None:
                k = np.concatenate([k, tail], axis=-1)
        return k[..., 0]

    def integer_dtype(self, log_range: float) -> Any:
        """Smallest signed integer dtype that holds log-values spanning ``[-log_range, log_range]``.

        Sized for ordinary codes only: ``LOG_ZERO_CODE`` is representable in int64 alone, so a caller
        that needs to store exact zeros (rather than pass them straight through ``quantize``/``dequantize``)
        should not narrow the returned dtype below int64.
        """
        kmax = log_range / self.step
        for dt in (np.int16, np.int32):
            if kmax < np.iinfo(dt).max:
                return dt
        return np.int64
