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


def _exact_code_array(values: Any, name: str = "codes") -> np.ndarray:
    """Validate exact canonical LNS codes before any integer conversion or arithmetic."""
    raw = np.asarray(values)
    if raw.dtype == np.bool_:
        raise ValueError(f"{name} must contain integer LNS codes, not booleans")
    if np.issubdtype(raw.dtype, np.integer):
        if np.issubdtype(raw.dtype, np.unsignedinteger) and raw.size and np.any(raw > CODE_MAX):
            raise ValueError(f"{name} contains a code outside the canonical LNS range")
        codes = raw.astype(np.int64)
    elif np.issubdtype(raw.dtype, np.floating):
        if raw.size and (
            not np.all(np.isfinite(raw))
            or not np.array_equal(raw, np.trunc(raw))
            or np.any(raw < float(LOG_ZERO_CODE))
            or np.any(raw > float(CODE_MAX))
        ):
            raise ValueError(f"{name} must contain exact finite integer LNS codes")
        codes = raw.astype(np.int64)
    else:
        checked = np.empty(raw.size, dtype=np.int64)
        for i, value in enumerate(raw.ravel()):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must contain exact integer LNS codes")
            value = int(value)
            if value != LOG_ZERO_CODE and not CODE_MIN <= value <= CODE_MAX:
                raise ValueError(f"{name} contains a code outside the canonical LNS range")
            checked[i] = value
        codes = checked.reshape(raw.shape)
    invalid = (codes != LOG_ZERO_CODE) & ((codes < CODE_MIN) | (codes > CODE_MAX))
    if np.any(invalid):
        raise ValueError(f"{name} contains a code outside the canonical LNS range")
    return codes


class LogNumberSystem:
    """Quantize log-space values to integers in units of ``step = ln(C)`` and compute on the integers."""

    LOG_ZERO_CODE = LOG_ZERO_CODE
    CODE_MIN = CODE_MIN
    CODE_MAX = CODE_MAX

    def __init__(self, step: float = 1e-2) -> None:
        if isinstance(step, (bool, np.bool_)):
            raise ValueError("step must be finite and positive")
        try:
            self.step = float(step)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("step must be finite and positive") from None
        if not math.isfinite(self.step) or self.step <= 0:
            raise ValueError("step must be finite and positive")
        # LUT[d] = round(log1p(exp(-d*step)) / step); the correction falls to 0, truncate where it rounds to 0.
        # log1p(exp(-d*step)) < step/2  <=>  exp(-d*step) < exp(step/2)-1  =>  d > -log(exp(step/2)-1)/step
        thresh = math.expm1(0.5 * self.step)
        dmax = int(math.ceil(-math.log(thresh) / self.step)) + 2 if thresh > 0 else 2
        self.dmax = max(dmax, 1)
        d = np.arange(self.dmax + 1, dtype=np.float64)
        self.lut = np.rint(np.log1p(np.exp(-d * self.step)) / self.step).astype(np.int64)
        self.lut.setflags(write=False)

    @classmethod
    def from_relative_precision(cls, rel: float) -> LogNumberSystem:
        """Build a system whose stored values are accurate to ~``rel`` relative (``step = ln(1+rel)``)."""
        if isinstance(rel, (bool, np.bool_)) or not np.isscalar(rel) or not math.isfinite(rel) or rel <= 0:
            raise ValueError("rel must be a finite positive scalar")
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
        if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 1:
            raise ValueError(f"max_logsumexp_error: n must be >= 1, got {n}")
        depth = math.ceil(math.log2(n)) if n > 1 else 0
        return (depth + 1) * self.step / 2.0

    def quantize(self, log_values: Any) -> np.ndarray:
        """Round log-space values to integer multiples of ``step`` (the stored representation).

        Exact ``-inf`` (the log of a true zero) maps to the reserved :data:`LOG_ZERO_CODE` sentinel --
        never through a float-to-int64 cast, whose behavior for a non-finite input this module does not
        rely on (see MXR-080-0138). NaN and ``+inf`` have no LNS code and raise. Any other value that
        would round outside ``[CODE_MIN, CODE_MAX]`` is rejected: clipping would invalidate the
        format's quantization-error certificate.
        """
        x = np.asarray(log_values, dtype=np.float64)
        n_nan = int(np.isnan(x).sum())
        if n_nan:
            raise ValueError(f"quantize: log_values has {n_nan} NaN value(s), which have no LNS code")
        n_posinf = int(np.isposinf(x).sum())
        if n_posinf:
            raise ValueError(f"quantize: log_values has {n_posinf} +inf value(s), which have no LNS code")
        is_log_zero = np.isneginf(x)
        with np.errstate(over="ignore", invalid="ignore"):
            rounded = np.rint(x / self.step)
        ordinary = ~is_log_zero
        invalid = ordinary & (
            ~np.isfinite(rounded) | (rounded < float(CODE_MIN)) | (rounded > float(CODE_MAX))
        )
        if np.any(invalid):
            raise OverflowError("quantize: finite log value is outside the representable LNS code range")
        safe = np.where(is_log_zero, 0.0, rounded)
        codes = safe.astype(np.int64)
        return np.where(is_log_zero, LOG_ZERO_CODE, codes).astype(np.int64)

    def dequantize(self, k: Any) -> np.ndarray:
        """Recover the float log-value ``k * step``; the reserved zero sentinel maps back to exactly ``-inf``."""
        k = _exact_code_array(k)
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
        k1, k2 = np.broadcast_arrays(_exact_code_array(k1, "k1"), _exact_code_array(k2, "k2"))
        is_zero1 = k1 == LOG_ZERO_CODE
        is_zero2 = k2 == LOG_ZERO_CODE
        safe1 = np.where(is_zero1, 0, k1)
        safe2 = np.where(is_zero2, 0, k2)
        d = np.minimum(np.abs(safe1 - safe2), self.dmax)
        ordinary = np.clip(np.maximum(safe1, safe2) + self.lut[d], CODE_MIN, CODE_MAX)
        result = np.where(is_zero2, k1, ordinary)
        return np.where(is_zero1, k2, result).astype(np.int64)

    def multiply(self, k1: Any, k2: Any) -> np.ndarray:
        """Integer code for the PRODUCT of two LNS values: ``log(a*b) = log(a) + log(b)`` -> code add.

        Exact for ordinary operands (integer add, no LUT, no rounding). ``LOG_ZERO_CODE`` is absorbing
        (anything times zero is zero); an ordinary sum that would leave ``[CODE_MIN, CODE_MAX]`` saturates
        to that boundary rather than silently wrapping through int64 overflow (MXR-080-0138). Prefer this
        over raw ``k1 + k2`` for any code that might be (or derive from) a quantized true zero.
        """
        k1, k2 = np.broadcast_arrays(_exact_code_array(k1, "k1"), _exact_code_array(k2, "k2"))
        is_zero = (k1 == LOG_ZERO_CODE) | (k2 == LOG_ZERO_CODE)
        safe1 = np.where(k1 == LOG_ZERO_CODE, 0, k1)
        safe2 = np.where(k2 == LOG_ZERO_CODE, 0, k2)
        result = np.clip(safe1 + safe2, CODE_MIN, CODE_MAX)
        return np.where(is_zero, LOG_ZERO_CODE, result)

    def logsumexp(self, k: Any, axis: int = -1) -> np.ndarray:
        """Integer log-sum-exp along ``axis`` via a pairwise tree of :meth:`logadd` (no exp/log).

        Uses the compiled one-pass tree kernel for the common 2-D last-axis reduction when available
        (bit-identical to the numpy tree, ~8x faster); falls back to vectorized numpy otherwise. Both
        trees combine adjacent pairs and carry an odd leftover element forward unchanged, so the deepest
        element always makes exactly ``ceil(log2(m))`` logadd hops (``m`` = reduced axis length) -- see
        :meth:`max_logsumexp_error`, which bounds the resulting error in terms of that depth.
        """
        arr = _exact_code_array(k)
        if arr.ndim == 0:
            raise ValueError("logsumexp input must have at least one dimension")
        if isinstance(axis, (bool, np.bool_)) or not isinstance(axis, (int, np.integer)):
            raise ValueError(f"axis {axis!r} is invalid for shape {arr.shape}")
        axis = int(axis)
        if axis < 0:
            axis += arr.ndim
        if not 0 <= axis < arr.ndim:
            raise ValueError(f"axis {axis!r} is invalid for shape {arr.shape}") from None
        if arr.shape[axis] == 0:
            raise ValueError("logsumexp cannot reduce an empty axis")
        if _HAS_LNS_KERNEL and arr.ndim == 2 and axis in (-1, 1) and arr.shape[1] > 0:
            result = _logsumexp_rows_c(
                np.ascontiguousarray(arr), self.lut, self.dmax, LOG_ZERO_CODE, CODE_MIN, CODE_MAX
            )
            return _exact_code_array(result, "compiled logsumexp result")
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
        if not np.isscalar(log_range) or not math.isfinite(log_range) or log_range < 0:
            raise ValueError("log_range must be a finite nonnegative scalar")
        kmax = log_range / self.step
        for dt in (np.int16, np.int32):
            if kmax < np.iinfo(dt).max:
                return dt
        return np.int64
