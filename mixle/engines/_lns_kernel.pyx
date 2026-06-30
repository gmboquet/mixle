# cython: boundscheck=False, wraparound=False, cdivision=True
"""Compiled integer log-sum-exp -- a pairwise TREE fold of streaming logadd over each row.

A balanced tree keeps comparable magnitudes combining, so only ~log(M) LUT roundings land on the critical
path (a streaming left-fold accumulates ~M roundings and drifts). This matches the numpy tree's accuracy
(error ~step) while running in one compiled pass over a reused scratch buffer -- pure int64 ``max`` + a
cache-resident LUT gather, no exp/log, no per-pass numpy temporary.
"""

from libc.stdint cimport int64_t

import numpy as np


cdef inline int64_t _logadd(int64_t a, int64_t b, int64_t[::1] lut, int dmax) nogil:
    cdef int64_t d, mx
    if a >= b:
        mx = a
        d = a - b
    else:
        mx = b
        d = b - a
    if d > dmax:
        d = dmax
    return mx + lut[d]


def logsumexp_rows(int64_t[:, ::1] k, int64_t[::1] lut, int dmax):
    """Integer log-sum-exp along axis 1 via a per-row pairwise tree fold: ``(N, M)`` codes -> ``(N,)``."""
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
                    buf[j] = _logadd(buf[2 * j], buf[2 * j + 1], lut, dmax)
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
                    buf[j] = _logadd(buf[2 * j], buf[2 * j + 1], lut, dmax)
                if sz & 1:
                    buf[half] = buf[sz - 1]
                    sz = half + 1
                else:
                    sz = half
            total += buf[0] - k[i, targets[i]]
    return total
