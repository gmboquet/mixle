# cython: boundscheck=False, wraparound=False, cdivision=True
"""Compiled integer log-sum-exp -- a pairwise TREE fold of streaming logadd over each row.

A balanced tree keeps comparable magnitudes combining, so only ~log(M) LUT roundings land on the critical
path (a streaming left-fold accumulates ~M roundings and drifts). This matches the numpy tree's accuracy
(error ~step) while running in one compiled pass over a reused scratch buffer -- pure int64 ``max`` + a
cache-resident LUT gather, no exp/log, no per-pass numpy temporary.
"""

from libc.stdint cimport int64_t

import numpy as np

# MXR-080-0138: must match LOG_ZERO_CODE / CODE_MIN / CODE_MAX in mixle/engines/lns.py exactly. Duplicated
# here as compiled constants (not imported) to avoid a circular import -- lns.py conditionally imports this
# extension module at load time, so this module cannot import back from lns.py.
#
# LOG_ZERO_CODE is assigned from the same `np.iinfo(np.int64).min` expression the Python side uses (a
# runtime call, converted via numpy/CPython's own int64 conversion), not a bare `-9223372036854775808`
# literal: that literal's magnitude itself exceeds what a signed 64-bit C literal can hold before the
# unary minus applies, which is well-defined nowhere -- Cython constant-folds even the portable two-step
# `-9223372036854775807 - 1` idiom back into that same oversized literal, so it does not help here.
cdef int64_t LOG_ZERO_CODE = np.iinfo(np.int64).min
cdef int64_t CODE_MIN = -2305843009213693952  # == -(2**61); safely in range, no literal-folding hazard
cdef int64_t CODE_MAX = 2305843009213693952  # == 2**61


cdef inline int64_t _logadd(
    int64_t a, int64_t b, int64_t[::1] lut, int dmax, int64_t log_zero, int64_t code_min, int64_t code_max
) nogil:
    """Sentinel-and-overflow-safe integer Gaussian logarithm (MXR-080-0138).

    ``log_zero`` (log of an exact zero) is absorbing: zero-plus-anything is the other operand, with no LUT
    lookup. Ordinary operands are clamped into ``[code_min, code_max]`` first so ``d = |a - b|`` below can
    never overflow int64 -- this must hold even for adversarial ``a``/``b``, since boundscheck is disabled
    file-wide and an overflowed, wrapped-negative ``d`` would otherwise be used directly as the ``lut``
    index, reading out-of-bounds memory instead of raising. The explicit ``d < 0`` clamp right before the
    table access is defense in depth: given the operand clamp above, it should be unreachable, but a LUT
    index derived from adversarial input should never be trusted without its own direct bounds check.
    """
    cdef int64_t d, mx, ca, cb
    if a == log_zero:
        return b
    if b == log_zero:
        return a
    ca = a
    if ca < code_min:
        ca = code_min
    elif ca > code_max:
        ca = code_max
    cb = b
    if cb < code_min:
        cb = code_min
    elif cb > code_max:
        cb = code_max
    if ca >= cb:
        mx = ca
        d = ca - cb
    else:
        mx = cb
        d = cb - ca
    if d < 0:
        d = 0
    elif d > dmax:
        d = dmax
    return mx + lut[d]


def logsumexp_rows(int64_t[:, ::1] k, int64_t[::1] lut, int dmax, int64_t log_zero, int64_t code_min, int64_t code_max):
    """Integer log-sum-exp along axis 1 via a per-row pairwise tree fold: ``(N, M)`` codes -> ``(N,)``.

    ``log_zero``/``code_min``/``code_max`` mirror :data:`mixle.engines.lns.LOG_ZERO_CODE` / ``CODE_MIN`` /
    ``CODE_MAX`` -- passed in explicitly by the caller (:meth:`mixle.engines.lns.LogNumberSystem.logsumexp`)
    rather than hardcoded here, so the compiled kernel and the numpy tree can never disagree on what the
    sentinel is even if one of the two definitions is ever changed without the other.
    """
    cdef Py_ssize_t n = k.shape[0], m = k.shape[1], i, j, half, sz
    out_np = np.empty(n, dtype=np.int64)
    buf_np = np.empty(m if m > 0 else 1, dtype=np.int64)
    cdef int64_t[::1] out = out_np
    cdef int64_t[::1] buf = buf_np
    with nogil:
        for i in range(n):
            for j in range(m):
                buf[j] = k[i, j]
            sz = m
            while sz > 1:
                half = sz // 2
                for j in range(half):
                    buf[j] = _logadd(buf[2 * j], buf[2 * j + 1], lut, dmax, log_zero, code_min, code_max)
                if sz & 1:
                    buf[half] = buf[sz - 1]
                    sz = half + 1
                else:
                    sz = half
            out[i] = buf[0]
    return out_np


def cross_entropy_rows(int64_t[:, ::1] k, int64_t[::1] targets, int64_t[::1] lut, int dmax):
    """Sum of per-row ``(logsumexp(k[i]) - k[i, target[i]])`` in code units -- the fused LM/classifier NLL.

    The tree-fold log-partition and the target-logit gather in one pass over a reused buffer, no temporaries.
    Caller multiplies by ``step`` and divides by N for the mean negative log-likelihood.

    Target/shape validation is out of scope here (see MXR-080-0140/0141, tracked separately against
    mixle/engines/lns_nn.py); the internal tree fold below passes the module-level sentinel/range
    constants into `_logadd` purely so its MXR-080-0138 fix (this fold has the identical overflowed-LUT-
    index hazard as `logsumexp_rows`) applies here too, without changing this function's own signature.
    """
    cdef Py_ssize_t n = k.shape[0], m = k.shape[1], i, j, half, sz
    cdef int64_t total = 0
    buf_np = np.empty(m if m > 0 else 1, dtype=np.int64)
    cdef int64_t[::1] buf = buf_np
    with nogil:
        for i in range(n):
            for j in range(m):
                buf[j] = k[i, j]
            sz = m
            while sz > 1:
                half = sz // 2
                for j in range(half):
                    buf[j] = _logadd(buf[2 * j], buf[2 * j + 1], lut, dmax, LOG_ZERO_CODE, CODE_MIN, CODE_MAX)
                if sz & 1:
                    buf[half] = buf[sz - 1]
                    sz = half + 1
                else:
                    sz = half
            total += buf[0] - k[i, targets[i]]
    return total
