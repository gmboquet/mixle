"""Approximate p-value and rank utilities for composite Bernoulli evidence.

The main helper builds a discretized log-likelihood histogram for products of
binomial terms so callers can estimate tail ranks without enumerating every
binary outcome.
"""

import itertools
from operator import index

import numpy as np
from scipy.special import gammaln


def binomial_rank(
    log_p_vec: list[float] | np.ndarray,
    log_p1_vec: list[float] | np.ndarray | None = None,
    count_vec: list | np.ndarray | None = None,
    ll_eps: float = 1.0e-4,
    max_len: int | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    """Approximates the log-density histogram for a composite of binomials.

    x, y, (LL0, DLL, cnt) =  binomial_rank(np.log([0.3, 0.2]), count_vec=[3, 2], max_len=10000)

    # p_mat([1, 0, 0, 1, 1])
    LL = np.log([0.3, 0.7, 0.7, 0.2, 0.2]).sum()
    approx_rank = y[int((LL - LL0)/DLL):].sum() * np.power(2.0, cnt)


        rtype(Tuple[np.ndarray, np.ndarray, Tuple[float, float, float]])

    Args:
        log_p_vec: Vector with log probabilities for each binomial distribution
        log_p1_vec: Optional vector with log one minus probabilities for each binomial distribution (for high-precision)
        count_vec: Vector with the number of draws for each binomial distribution
        ll_eps: Bin spacing is determined so that ``|LL - floor(LL/space)*space| < ll_eps``
        max_len: Maximum number of bins for histogram
    Returns:
        log_density array, corresponding probs array, Tuple[ll0, dll, total_count]
    """
    try:
        log_p_vec = np.asarray(log_p_vec, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("log_p_vec must be a one-dimensional log-probability vector") from exc
    if log_p_vec.ndim != 1 or log_p_vec.size == 0:
        raise ValueError("log_p_vec must be a non-empty one-dimensional vector")
    if np.any(np.isnan(log_p_vec)) or np.any(np.isposinf(log_p_vec)) or np.any(log_p_vec > 0.0):
        raise ValueError("log_p_vec must contain log probabilities in [-inf, 0]")

    if log_p1_vec is None:
        with np.errstate(divide="ignore", invalid="ignore"):
            log_p1_vec = np.log1p(-np.exp(log_p_vec))
    else:
        try:
            log_p1_vec = np.asarray(log_p1_vec, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("log_p1_vec must be a one-dimensional log-probability vector") from exc
        if log_p1_vec.shape != log_p_vec.shape:
            raise ValueError("log_p_vec and log_p1_vec must have identical lengths")
        if np.any(np.isnan(log_p1_vec)) or np.any(np.isposinf(log_p1_vec)) or np.any(log_p1_vec > 0.0):
            raise ValueError("log_p1_vec must contain log probabilities in [-inf, 0]")
        probability_sums = np.exp(log_p_vec) + np.exp(log_p1_vec)
        if not np.allclose(probability_sums, 1.0, rtol=1e-10, atol=1e-12):
            raise ValueError("log_p_vec and log_p1_vec must encode complementary probabilities")

    if count_vec is None:
        counts = np.ones(len(log_p_vec), dtype=int)
    else:
        if isinstance(count_vec, (str, bytes)):
            raise TypeError("count_vec must be a sequence of non-negative integers")
        raw_counts = list(count_vec)
        if len(raw_counts) != len(log_p_vec):
            raise ValueError("count_vec must have exactly one count per probability")
        counts = np.empty(len(raw_counts), dtype=int)
        for position, count in enumerate(raw_counts):
            if isinstance(count, (bool, np.bool_)):
                raise ValueError("count_vec must contain non-negative integers")
            try:
                counts[position] = index(count)
            except TypeError as exc:
                raise ValueError("count_vec must contain non-negative integers") from exc
            if counts[position] < 0:
                raise ValueError("count_vec must contain non-negative integers")
    if (
        isinstance(ll_eps, (bool, np.bool_))
        or not isinstance(ll_eps, (int, float, np.integer, np.floating))
        or not np.isfinite(ll_eps)
        or float(ll_eps) <= 0.0
    ):
        raise ValueError("ll_eps must be a positive finite number")
    ll_eps = float(ll_eps)
    if max_len is not None:
        if isinstance(max_len, (bool, np.bool_)):
            raise ValueError("max_len must be a positive integer")
        try:
            max_len = index(max_len)
        except TypeError as exc:
            raise ValueError("max_len must be a positive integer") from exc
        if max_len <= 0:
            raise ValueError("max_len must be a positive integer")

    entries = []

    # Compute binomial log-densities and probabilities
    for log_p, log_p1, n in zip(log_p_vec, log_p1_vec, counts, strict=True):
        if n == 0:
            entries.append((np.array([0.0]), np.array([1.0]), 0))
            continue
        if np.isneginf(log_p) or np.isneginf(log_p1):
            # A deterministic Bernoulli contributes one possible finite
            # likelihood, zero, but its draw count must remain in the rank
            # receipt instead of disappearing from the experiment.
            entries.append((np.array([0.0]), np.array([1.0]), int(n)))
            continue
        nn = np.arange(0, n + 1)
        llv = log_p * nn + log_p1 * (n - nn)
        ell = gammaln(n + 1) - gammaln(nn + 1) - gammaln(n - nn + 1)
        ell = np.exp(ell - ell.max())
        ell /= np.sum(ell)
        llv = llv[ell > 0]
        ell = ell[ell > 0]

        entries.append((llv, ell, int(n)))

    # Find parameters for a common fixed-space grid [ll0, ll0 + dll, ll0 + 2*dll, ...]
    min_vec = np.asarray([entry[0].min() for entry in entries])
    llv_vec = np.concatenate([entry[0] - entry[0].min() for entry in entries])
    llv_vec = np.sort(np.unique(llv_vec))

    mll = float(np.sum([entry[0].max() - entry[0].min() for entry in entries]))
    if mll == 0.0:
        dll = ll_eps
    elif max_len is not None:
        if max_len == 1:
            raise ValueError("max_len=1 is only feasible when every likelihood is identical")
        dll = mll / (max_len - 1)
    else:
        differences = np.diff(llv_vec)
        differences = differences[differences > 0.0]
        if differences.size == 0:
            dll = ll_eps
        else:
            dll = float(differences.min())
            if dll > ll_eps:
                dll /= 2.0 ** int(np.ceil(np.log2(dll / ll_eps)))
    if not np.isfinite(dll) or dll <= 0.0:
        raise ValueError("binomial likelihood grid has no positive finite spacing")

    # Adjust log-density histograms to a common grid and convolve
    temp_idx = np.floor((entries[0][0] - entries[0][0].min()) / dll).astype(int)
    acc_prob = np.bincount(temp_idx, weights=entries[0][1])
    acc_count = entries[0][2]

    for next_llv, next_ell, next_count in entries[1:]:
        next_idx = np.floor((next_llv - next_llv.min()) / dll).astype(int)

        next_prob = np.bincount(next_idx, weights=next_ell)
        max_count = max(next_count, acc_count)
        acc_weight = np.power(2.0, acc_count - max_count)
        next_weight = np.power(2.0, next_count - max_count)

        acc_prob = np.convolve(acc_prob * acc_weight, next_prob * next_weight)
        acc_prob /= np.sum(acc_prob)
        acc_count += next_count

    ll0 = min_vec.sum()
    acc_ll = ll0 + np.arange(len(acc_prob)) * dll
    return acc_ll, acc_prob, (ll0, dll, acc_count)


if __name__ == "__main__":
    pvec = np.asarray([0.3, 0.8, 0.4])
    pvec = np.log(pvec)
    nvec = np.log1p(-np.exp(pvec))
    cvec = np.asarray([2, 3, 3])

    pvec_long = np.concatenate([[u] * n for u, n in zip(pvec, cvec)])
    nvec_long = np.concatenate([[u] * n for u, n in zip(nvec, cvec)])

    test = np.asarray([1, 0, 1, 1, 0, 1, 0, 1])
    ll = np.where(test == 1, pvec_long, nvec_long).sum()

    acc_ll, acc_prob, (ll0, dll, acc_count) = binomial_rank(pvec, count_vec=cvec, max_len=100000)
    left = acc_prob[(int((ll - ll0) / dll) - 1) :].sum() * np.power(2, acc_count)
    mid = acc_prob[int((ll - ll0) / dll) :].sum() * np.power(2, acc_count)
    right = acc_prob[(int((ll - ll0) / dll) + 1) :].sum() * np.power(2, acc_count)
    print("Approximate rank: %f ( Somewhere in [%f, %f] )" % (mid, right, left))

    # Verify this
    temp = np.asarray(
        [
            np.where([u == 1 for u in x], pvec_long, nvec_long).sum()
            for x in itertools.product([0, 1], repeat=len(pvec_long))
        ]
    )
    print("True rank:" + str((temp >= ll).sum()))
