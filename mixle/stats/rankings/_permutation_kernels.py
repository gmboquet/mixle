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

import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from mixle.utils.optional_deps import numba

METRICS = ("kendall", "cayley", "hamming", "footrule", "spearman", "ulam")
_METRIC_ID = {name: i for i, name in enumerate(METRICS)}


def metric_id(metric: str) -> int:
    """Map a metric name to its integer id (raises on an unknown name)."""
    try:
        return _METRIC_ID[metric]
    except (KeyError, TypeError):
        raise ValueError(f"metric must be one of {METRICS}, got {metric!r}.") from None


def _validate_permutation(value: np.ndarray, *, label: str, expected_dim: int | None = None) -> np.ndarray:
    """Return one exact permutation as owned contiguous ``int64`` data."""
    raw = np.asarray(value)
    if raw.ndim != 1:
        raise ValueError(f"{label} must be a one-dimensional permutation.")
    if expected_dim is not None and len(raw) != expected_dim:
        raise ValueError(f"{label} must have length {expected_dim}.")
    if np.issubdtype(raw.dtype, np.bool_):
        raise TypeError(f"{label} must contain exact integer item identifiers.")
    if np.iscomplexobj(raw):
        raise TypeError(f"{label} must contain exact integer item identifiers.")
    try:
        converted = np.asarray(raw, dtype=np.int64)
        numeric = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{label} must contain exact integer item identifiers.") from exc
    if not np.all(np.isfinite(numeric)) or not np.array_equal(numeric, converted):
        raise ValueError(f"{label} must contain exact integer item identifiers.")
    expected = np.arange(len(converted), dtype=np.int64)
    if not np.array_equal(np.sort(converted), expected):
        raise ValueError(f"{label} must be a permutation of 0,...,{len(converted) - 1}.")
    return np.ascontiguousarray(converted, dtype=np.int64)


def _validate_orderings(
    value: np.ndarray,
    *,
    label: str,
    expected_dim: int | None = None,
) -> np.ndarray:
    """Return a two-dimensional batch of exact equal-width permutations."""
    raw = np.asarray(value)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    if raw.ndim != 2:
        raise ValueError(f"{label} must be a one- or two-dimensional permutation array.")
    if expected_dim is not None and raw.shape[1] != expected_dim:
        raise ValueError(f"{label} rows must have length {expected_dim}.")
    rows = [
        _validate_permutation(row, label=f"{label} row {index}", expected_dim=expected_dim)
        for index, row in enumerate(raw)
    ]
    if not rows:
        dim = 0 if expected_dim is None else expected_dim
        return np.empty((0, dim), dtype=np.int64)
    return np.ascontiguousarray(np.vstack(rows), dtype=np.int64)


# --- per-permutation kernels (distance of the relative permutation r from the identity) ----------
@numba.njit("int64(int64[:])", cache=True)
def _kendall_perm(r: np.ndarray) -> int:
    n = r.shape[0]
    c = 0
    for i in range(n):
        ri = r[i]
        for j in range(i + 1, n):
            if ri > r[j]:
                c += 1
    return c


@numba.njit("int64(int64[:])", cache=True)
def _cayley_perm(r: np.ndarray) -> int:
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
def _hamming_perm(r: np.ndarray) -> int:
    n = r.shape[0]
    c = 0
    for i in range(n):
        if r[i] != i:
            c += 1
    return c


@numba.njit("int64(int64[:])", cache=True)
def _footrule_perm(r: np.ndarray) -> int:
    n = r.shape[0]
    c = 0
    for i in range(n):
        d = r[i] - i
        c += d if d >= 0 else -d
    return c


@numba.njit("int64(int64[:])", cache=True)
def _spearman_perm(r: np.ndarray) -> int:
    n = r.shape[0]
    c = 0
    for i in range(n):
        d = r[i] - i
        c += d * d
    return c


@numba.njit("int64(int64[:])", cache=True)
def _ulam_perm(r: np.ndarray) -> int:
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


def kendall_perm(r: np.ndarray) -> int:
    """Return the Kendall distance of one validated relative permutation."""
    return int(_kendall_perm(_validate_permutation(r, label="relative permutation")))


def cayley_perm(r: np.ndarray) -> int:
    """Return the Cayley distance of one validated relative permutation."""
    return int(_cayley_perm(_validate_permutation(r, label="relative permutation")))


def hamming_perm(r: np.ndarray) -> int:
    """Return the Hamming distance of one validated relative permutation."""
    return int(_hamming_perm(_validate_permutation(r, label="relative permutation")))


def footrule_perm(r: np.ndarray) -> int:
    """Return the Spearman-footrule distance of one validated relative permutation."""
    return int(_footrule_perm(_validate_permutation(r, label="relative permutation")))


def spearman_perm(r: np.ndarray) -> int:
    """Return the Spearman-rho distance of one validated relative permutation."""
    return int(_spearman_perm(_validate_permutation(r, label="relative permutation")))


def ulam_perm(r: np.ndarray) -> int:
    """Return the Ulam distance of one validated relative permutation."""
    return int(_ulam_perm(_validate_permutation(r, label="relative permutation")))


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
            out[k] = _kendall_perm(r)
        elif mid == 1:
            out[k] = _cayley_perm(r)
        elif mid == 2:
            out[k] = _hamming_perm(r)
        elif mid == 3:
            out[k] = _footrule_perm(r)
        elif mid == 4:
            out[k] = _spearman_perm(r)
        else:
            out[k] = _ulam_perm(r)
    return out


# --- assignment-model normalizers: exact permanent + Sinkhorn/Bethe approximation ----------------
@numba.njit("float64(float64[:, :])", cache=True)
def _log_permanent_dp(M: np.ndarray) -> float:
    """Stable subset DP for the log permanent of a validated nonnegative square matrix."""
    n = M.shape[0]
    if n == 0:
        return 0.0
    size = 1 << n
    dp = np.full(size, -np.inf)
    dp[0] = 0.0
    for mask in range(1, size):
        count = 0
        temp = mask
        while temp:
            count += temp & 1
            temp >>= 1
        row = count - 1
        value = -np.inf
        for column in range(n):
            bit = 1 << column
            weight = M[row, column]
            if (mask & bit) and weight > 0.0:
                candidate = dp[mask ^ bit] + math.log(weight)
                if value == -np.inf:
                    value = candidate
                elif candidate > value:
                    value = candidate + math.log1p(math.exp(value - candidate))
                else:
                    value = value + math.log1p(math.exp(candidate - value))
        dp[mask] = value
    return dp[size - 1]


@numba.njit("Tuple((float64[:, :], float64))(float64[:, :], int64)", cache=True)
def _sinkhorn_bethe(s: np.ndarray, n_iter: int) -> tuple[np.ndarray, float]:
    """Log-domain Sinkhorn on the kernel ``exp(s)``: returns the doubly-stochastic marginals ``P`` and a
    Bethe estimate of ``log permanent(exp(s))`` (the scalable approximation for the assignment model)."""
    n = s.shape[0]
    f = np.zeros(n)
    g = np.zeros(n)
    for _ in range(n_iter):
        for i in range(n):  # row scaling: make row i sum to 1
            mx = -np.inf
            for j in range(n):
                v = s[i, j] + g[j]
                if v > mx:
                    mx = v
            acc = 0.0
            for j in range(n):
                acc += math.exp(s[i, j] + g[j] - mx)
            f[i] = -(mx + math.log(acc))
        for j in range(n):  # column scaling: make column j sum to 1
            mx = -np.inf
            for i in range(n):
                v = s[i, j] + f[i]
                if v > mx:
                    mx = v
            acc = 0.0
            for i in range(n):
                acc += math.exp(s[i, j] + f[i] - mx)
            g[j] = -(mx + math.log(acc))
    p = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            p[i, j] = math.exp(s[i, j] + f[i] + g[j])
    # Bethe free energy: log Z ~ sum P*s + Bethe entropy (Vontobel 2013)
    logz = 0.0
    for i in range(n):
        for j in range(n):
            pij = p[i, j]
            if pij > 0.0:
                logz += pij * s[i, j] - pij * math.log(pij)
            if pij < 1.0:
                logz += (1.0 - pij) * math.log1p(-pij)
    return p, logz


def ryser_log_permanent(M: np.ndarray) -> float:
    """Return the exact log permanent of a finite nonnegative square matrix.

    The compatibility name is retained, but the implementation uses a log-domain subset
    dynamic program rather than cancellation-prone inclusion/exclusion.
    """
    if np.iscomplexobj(M):
        raise TypeError("permanent input must be a real matrix.")
    matrix = np.asarray(M, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("permanent input must be a square matrix.")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError("permanent input must contain finite nonnegative values.")
    if matrix.shape[0] > 22:
        raise ValueError("exact permanent evaluation is limited to dimension 22.")
    result = float(_log_permanent_dp(np.ascontiguousarray(matrix)))
    if math.isnan(result) or result == np.inf:
        raise FloatingPointError("permanent evaluation produced a non-finite numerical result.")
    return result


def sinkhorn_bethe(s: np.ndarray, n_iter: int) -> tuple[np.ndarray, float]:
    """Return finite Sinkhorn marginals and a Bethe log-normalizer approximation."""
    if np.iscomplexobj(s):
        raise TypeError("Sinkhorn scores must be a real matrix.")
    scores = np.asarray(s, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("Sinkhorn scores must be a square matrix.")
    if np.any(np.isnan(scores)) or np.any(scores == np.inf):
        raise ValueError("Sinkhorn scores must contain only finite values or -inf exclusions.")
    if isinstance(n_iter, (bool, np.bool_)):
        raise TypeError("n_iter must be a positive exact integer.")
    raw_iterations = np.asarray(n_iter)
    if raw_iterations.ndim != 0:
        raise TypeError("n_iter must be a positive exact integer.")
    try:
        iterations = int(raw_iterations.item())
        numeric_iterations = float(raw_iterations.item())
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("n_iter must be a positive exact integer.") from exc
    if not math.isfinite(numeric_iterations) or numeric_iterations != iterations or iterations <= 0:
        raise ValueError("n_iter must be a positive exact integer.")
    if scores.shape[0]:
        rows, columns = linear_sum_assignment(np.where(np.isfinite(scores), 0.0, 1.0))
        if np.any(~np.isfinite(scores[rows, columns])):
            raise ValueError("finite Sinkhorn support must contain a perfect matching.")
    plan, logz = _sinkhorn_bethe(np.ascontiguousarray(scores), iterations)
    if not np.all(np.isfinite(plan)) or not math.isfinite(float(logz)):
        raise FloatingPointError("Sinkhorn iteration produced a non-finite plan or normalizer.")
    return plan, float(logz)


# --- python-facing helpers -----------------------------------------------------------------------
def relative_ranks(orderings: np.ndarray, rank_center: np.ndarray) -> np.ndarray:
    """Compose orderings into the center's rank frame: ``R[k, i] = rank_center[orderings[k, i]]``."""
    center = _validate_permutation(rank_center, label="rank_center")
    rows = _validate_orderings(orderings, label="orderings", expected_dim=len(center))
    return np.ascontiguousarray(center[rows], dtype=np.int64)


def seq_distance_to_center(orderings: np.ndarray, rank_center: np.ndarray, metric: str) -> np.ndarray:
    """Vectorized distance of each ordering (row of an ``(N, n)`` array) to the center, under ``metric``."""
    return _seq_distance(relative_ranks(orderings, rank_center), metric_id(metric))


def seq_rim_code(orderings: np.ndarray, sigma0: np.ndarray) -> np.ndarray:
    """RIM insertion codes ``(N, n-1)`` of each ordering relative to the central permutation ``sigma0``."""
    center = _validate_permutation(sigma0, label="sigma0")
    if len(center) == 0:
        raise ValueError("RIM permutations must contain at least one item.")
    rows = _validate_orderings(orderings, label="orderings", expected_dim=len(center))
    return _seq_rim_code(rows, center)


def permutation_distance(a: np.ndarray, b: np.ndarray, metric: str = "kendall") -> int:
    """Distance between two orderings ``a`` and ``b`` (permutations of ``0..n-1``) under ``metric``."""
    first = _validate_permutation(a, label="a")
    second = _validate_permutation(b, label="b", expected_dim=len(first))
    rank_b = np.empty(second.shape[0], dtype=np.int64)
    rank_b[second] = np.arange(second.shape[0], dtype=np.int64)
    r = np.ascontiguousarray(rank_b[first], dtype=np.int64)
    return int(_seq_distance(r.reshape(1, -1), metric_id(metric))[0])
