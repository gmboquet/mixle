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
    cdef Py_ssize_t i, j, k, expected_words
    cdef int ham
    # Defense in depth (MXR-080-0131): mixle.engines.bitpacked.binary_gemm already validates shapes
    # before calling in, but this compiled entry point is also reachable directly (bypassing that Python
    # wrapper). `words` below is read from `a` alone; boundscheck is disabled file-wide for speed, so
    # without this check a shorter `b` would be read past the end of its buffer inside the nogil loop
    # instead of raising. These checks run while still holding the GIL, before the unsafe fast path.
    if dim < 0:
        raise ValueError("binary_gemm: dim must be nonnegative, got %d" % dim)
    if b.shape[1] != words:
        raise ValueError(
            "binary_gemm: a has %d packed word(s) per row but b has %d; shapes are inconsistent"
            % (words, b.shape[1])
        )
    expected_words = (dim + 63) // 64
    if words != expected_words:
        raise ValueError(
            "binary_gemm: dim=%d implies %d packed word(s) per row, but arrays have %d"
            % (dim, expected_words, words)
        )
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
    # Defense in depth (MXR-080-0131): mixle.engines.bitpacked.ternary_gemm already validates shapes
    # before calling in, but this compiled entry point is also reachable directly. `words` below is read
    # from `a_sign` alone and used to index all three other planes; boundscheck is disabled file-wide, so
    # a shorter a_nz/b_sign/b_nz would be read past the end of its buffer inside the nogil loop instead of
    # raising. These checks run while still holding the GIL, before the unsafe fast path.
    if a_nz.shape[0] != n or a_nz.shape[1] != words:
        raise ValueError(
            "ternary_gemm: a_sign is (%d, %d) but a_nz is (%d, %d)" % (n, words, a_nz.shape[0], a_nz.shape[1])
        )
    if b_sign.shape[1] != words:
        raise ValueError(
            "ternary_gemm: a_sign has %d packed word(s) per row but b_sign has %d" % (words, b_sign.shape[1])
        )
    if b_nz.shape[0] != m or b_nz.shape[1] != words:
        raise ValueError(
            "ternary_gemm: b_sign is (%d, %d) but b_nz is (%d, %d)" % (m, words, b_nz.shape[0], b_nz.shape[1])
        )
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
