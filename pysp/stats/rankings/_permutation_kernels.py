"""Numba permutation-distance kernels shared by the ranking distributions.

Every right-invariant permutation distance ``d(a, b)`` between two orderings (``a[r]`` / ``b[r]`` is the
item at rank ``r``) is a function of the single *relative-rank* permutation ``r``, where ``r[i]`` is the
rank, under ``b``, of the item placed at rank ``i`` by ``a`` (``r = rank_b[a]``). Writing each distance
as a property of ``r`` versus the identity lets one O(n^2)/O(n log n) integer kernel serve all of them:

    Kendall tau     inversions(r)              (discordant pairs)
    Cayley          n - cycles(r)              (minimum transpositions)
    Hamming         #{i : r[i] != i}           (displaced items)
    footrule        sum_i |r[i] - i|           (Spearman footrule, L1)
    Spearman rho    sum_i (r[i] - i)^2         (squared L2)
    Ulam            n - LIS(r)                  (n - longest increasing subsequence)

All kernels are ``@numba.njit(cache=True)`` integer loops, so they JIT to native code and fall back to
pure Python (via the numba shim) when numba is absent -- the results are identical either way.
"""

from __future__ import annotations

import numpy as np

from pysp.utils.optional_deps import numba

METRICS = ("kendall", "cayley", "hamming", "footrule", "spearman", "ulam")
_METRIC_ID = {name: i for i, name in enumerate(METRICS)}


def metric_id(metric: str) -> int:
    """Map a metric name to its integer id (raises on an unknown name)."""
    try:
        return _METRIC_ID[metric]
    except KeyError:
        raise ValueError(f"metric must be one of {METRICS}, got {metric!r}.") from None


# --- per-permutation kernels (distance of the relative permutation r from the identity) ----------
@numba.njit("int64(int64[:])", cache=True)
def kendall_perm(r: np.ndarray) -> int:
    n = r.shape[0]
    c = 0
    for i in range(n):
        ri = r[i]
        for j in range(i + 1, n):
            if ri > r[j]:
                c += 1
    return c


@numba.njit("int64(int64[:])", cache=True)
def cayley_perm(r: np.ndarray) -> int:
    n = r.shape[0]
    seen = np.zeros(n, dtype=np.bool_)
    cycles = 0
    for i in range(n):
        if not seen[i]:
            cycles += 1
            j = i
            while not seen[j]:
                seen[j] = True
                j = r[j]
    return n - cycles


@numba.njit("int64(int64[:])", cache=True)
def hamming_perm(r: np.ndarray) -> int:
    n = r.shape[0]
    c = 0
    for i in range(n):
        if r[i] != i:
            c += 1
    return c


@numba.njit("int64(int64[:])", cache=True)
def footrule_perm(r: np.ndarray) -> int:
    n = r.shape[0]
    c = 0
    for i in range(n):
        d = r[i] - i
        c += d if d >= 0 else -d
    return c


@numba.njit("int64(int64[:])", cache=True)
def spearman_perm(r: np.ndarray) -> int:
    n = r.shape[0]
    c = 0
    for i in range(n):
        d = r[i] - i
        c += d * d
    return c


@numba.njit("int64(int64[:])", cache=True)
def ulam_perm(r: np.ndarray) -> int:
    n = r.shape[0]
    tails = np.empty(n, dtype=np.int64)  # tails[k] = smallest possible tail of an increasing run of length k+1
    size = 0
    for i in range(n):
        x = r[i]
        lo, hi = 0, size
        while lo < hi:  # first tail >= x (strictly increasing LIS)
            mid = (lo + hi) // 2
            if tails[mid] < x:
                lo = mid + 1
            else:
                hi = mid
        tails[lo] = x
        if lo == size:
            size += 1
    return n - size


# --- RIM insertion code: the per-stage statistic of the Generalized Mallows Model ----------------
@numba.njit("int64[:, :](int64[:, :], int64[:])", cache=True)
def _seq_rim_code(orderings: np.ndarray, sigma0: np.ndarray) -> np.ndarray:
    """Repeated-Insertion-Model code ``J[i] = #{m < i : q[m] > q[i]}`` (q = observed rank of sigma0[i]).

    ``J[i] in {0..i}`` is the back-jump of central item ``i`` under the RIM, ``sum_i J[i] = kendall``;
    returns columns ``J[1..n-1]`` (``J[0] = 0`` is dropped). This is exactly the statistic the per-stage
    RIM sampler inverts, so density and sampling stay consistent.
    """
    big_n, n = orderings.shape
    out = np.empty((big_n, n - 1), dtype=np.int64)
    rank = np.empty(n, dtype=np.int64)
    q = np.empty(n, dtype=np.int64)
    for t in range(big_n):
        sig = orderings[t]
        for rpos in range(n):
            rank[sig[rpos]] = rpos  # observed rank of each item
        for i in range(n):
            q[i] = rank[sigma0[i]]
        for i in range(1, n):
            c = 0
            qi = q[i]
            for m in range(i):
                if q[m] > qi:
                    c += 1
            out[t, i - 1] = c
    return out


# --- batched drivers: distance of every row of R (relative-rank vectors) from the identity -------
@numba.njit("int64[:](int64[:,:], int64)", cache=True)
def _seq_distance(R: np.ndarray, mid: int) -> np.ndarray:
    n = R.shape[0]
    out = np.empty(n, dtype=np.int64)
    for k in range(n):
        r = R[k]
        if mid == 0:
            out[k] = kendall_perm(r)
        elif mid == 1:
            out[k] = cayley_perm(r)
        elif mid == 2:
            out[k] = hamming_perm(r)
        elif mid == 3:
            out[k] = footrule_perm(r)
        elif mid == 4:
            out[k] = spearman_perm(r)
        else:
            out[k] = ulam_perm(r)
    return out


# --- python-facing helpers -----------------------------------------------------------------------
def relative_ranks(orderings: np.ndarray, rank_center: np.ndarray) -> np.ndarray:
    """Compose orderings into the center's rank frame: ``R[k, i] = rank_center[orderings[k, i]]``."""
    return np.ascontiguousarray(rank_center[np.asarray(orderings, dtype=np.int64)], dtype=np.int64)


def seq_distance_to_center(orderings: np.ndarray, rank_center: np.ndarray, metric: str) -> np.ndarray:
    """Vectorized distance of each ordering (row of an ``(N, n)`` array) to the center, under ``metric``."""
    o = np.atleast_2d(np.asarray(orderings, dtype=np.int64))
    return _seq_distance(relative_ranks(o, np.asarray(rank_center, dtype=np.int64)), metric_id(metric))


def seq_rim_code(orderings: np.ndarray, sigma0: np.ndarray) -> np.ndarray:
    """RIM insertion codes ``(N, n-1)`` of each ordering relative to the central permutation ``sigma0``."""
    o = np.atleast_2d(np.asarray(orderings, dtype=np.int64))
    return _seq_rim_code(o, np.asarray(sigma0, dtype=np.int64))


def permutation_distance(a: np.ndarray, b: np.ndarray, metric: str = "kendall") -> int:
    """Distance between two orderings ``a`` and ``b`` (permutations of ``0..n-1``) under ``metric``."""
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    rank_b = np.empty(b.shape[0], dtype=np.int64)
    rank_b[b] = np.arange(b.shape[0], dtype=np.int64)
    r = np.ascontiguousarray(rank_b[a], dtype=np.int64)
    return int(_seq_distance(r.reshape(1, -1), metric_id(metric))[0])
