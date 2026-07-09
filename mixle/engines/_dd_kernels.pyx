# cython: boundscheck=False, wraparound=False, cdivision=True
"""Compiled double-double kernels using hardware FMA -- the optional C accelerator for the EFT path.

numpy exposes no fused-multiply-add, so the pure-numpy ``two_prod`` pays a ~6-op Veltkamp split per
element. With ``libc.math.fma`` the exact product error is a SINGLE instruction (``fma(a, b, -a*b)``),
so an accurate double-double dot/sum runs in one pass of low-overhead FMAs -- faster than the vectorized
multi-pass numpy version and exactly as accurate. Built optionally; mixle falls back to pure numpy when
this is not compiled (so a compiler is never required to import the package).
"""

from libc.math cimport fma


def dd_dot_c(double[::1] a, double[::1] b):
    """Accurate dot product sum(a_i * b_i) in double-double, returned as (hi, lo). One FMA pass."""
    cdef Py_ssize_t n = a.shape[0], i
    cdef double s_hi = 0.0, s_lo = 0.0
    cdef double p, pe, t, bb, err
    for i in range(n):
        p = a[i] * b[i]
        pe = fma(a[i], b[i], -p)            # exact rounding error of a*b
        t = s_hi + p                        # two_sum(s_hi, p)
        bb = t - s_hi
        err = (s_hi - (t - bb)) + (p - bb)
        s_hi = t
        s_lo += err + pe
        t = s_hi + s_lo                     # renormalize (quick_two_sum)
        s_lo = s_lo - (t - s_hi)
        s_hi = t
    return s_hi, s_lo


def dd_sum_c(double[::1] x):
    """Accurate sum of x in double-double, returned as (hi, lo). One pass."""
    cdef Py_ssize_t n = x.shape[0], i
    cdef double s_hi = 0.0, s_lo = 0.0
    cdef double xi, t, bb, err
    for i in range(n):
        xi = x[i]
        t = s_hi + xi
        bb = t - s_hi
        err = (s_hi - (t - bb)) + (xi - bb)
        s_hi = t
        s_lo += err
        t = s_hi + s_lo
        s_lo = s_lo - (t - s_hi)
        s_hi = t
    return s_hi, s_lo
