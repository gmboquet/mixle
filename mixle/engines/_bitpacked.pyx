# cython: boundscheck=False, wraparound=False, cdivision=True
"""Packed binary/ternary COMPUTE kernels -- the one genuine sub-byte arithmetic path (popcount dot).

A {-1,+1} (binary) dot product is exact integer arithmetic with NO rounding: pack each vector to bits,
and ``a . b = D - 2 * popcount(a XOR b)``. The hardware popcount (NEON CNT / x86 POPCNT) does this on
64 packed values per instruction. Ternary {-1,0,+1} uses two bit-planes (sign + nonzero mask). These run
on packed uint64 words with no fp intermediate -- a 32x storage shrink and real packed arithmetic. Whether
it beats fp32 BLAS depends on the hardware (it does not on Apple-AMX for cache-resident GEMMs; it does on
memory-bound / non-BLAS / GPU paths) -- the kernel is correct and exact regardless.
"""

from libc.stdint cimport int32_t, uint64_t

import numpy as np

cdef extern from *:
    int __builtin_popcountll(unsigned long long) nogil


def binary_gemm(uint64_t[:, ::1] a, uint64_t[:, ::1] b, int dim):
    """Exact ``A @ B^T`` for {-1,+1} matrices packed to uint64 words.

    ``a`` is (N, words) packed rows, ``b`` is (M, words) packed rows (the columns of the operand), ``dim``
    is the true bit length. Returns int32 (N, M) with ``out[n,m] = dim - 2*hamming(a[n], b[m])``.
    """
    cdef Py_ssize_t n = a.shape[0], m = b.shape[0], words = a.shape[1]
    cdef Py_ssize_t i, j, k
    cdef int ham
    out_np = np.empty((n, m), dtype=np.int32)
    cdef int32_t[:, ::1] out = out_np
    with nogil:
        for i in range(n):
            for j in range(m):
                ham = 0
                for k in range(words):
                    ham += __builtin_popcountll(a[i, k] ^ b[j, k])
                out[i, j] = <int32_t>(dim - 2 * ham)
    return out_np


def ternary_gemm(
    uint64_t[:, ::1] a_sign, uint64_t[:, ::1] a_nz, uint64_t[:, ::1] b_sign, uint64_t[:, ::1] b_nz
):
    """Exact ternary {-1,0,+1} ``A @ B^T`` from sign + nonzero-mask bit-planes.

    Per element the product is +1 (both nonzero, signs agree), -1 (both nonzero, signs differ), else 0.
    ``out = popcount(active & ~(sign_a ^ sign_b)) - popcount(active & (sign_a ^ sign_b))`` with
    ``active = nz_a & nz_b``.
    """
    cdef Py_ssize_t n = a_sign.shape[0], m = b_sign.shape[0], words = a_sign.shape[1]
    cdef Py_ssize_t i, j, k
    cdef int acc
    cdef uint64_t active, diff
    out_np = np.empty((n, m), dtype=np.int32)
    cdef int32_t[:, ::1] out = out_np
    with nogil:
        for i in range(n):
            for j in range(m):
                acc = 0
                for k in range(words):
                    active = a_nz[i, k] & b_nz[j, k]
                    diff = a_sign[i, k] ^ b_sign[j, k]
                    acc += __builtin_popcountll(active & ~diff)
                    acc -= __builtin_popcountll(active & diff)
                out[i, j] = <int32_t>acc
    return out_np
