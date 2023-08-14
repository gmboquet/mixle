from typing import Tuple, Union, List, Optional
import numpy as np
import itertools
from scipy.special import gammaln


def binomial_rank(log_p_vec: Union[List[float], np.ndarray],
                  log_p1_vec: Optional[Union[List[float], np.ndarray]] = None,
                  count_vec: Optional[Union[List, np.ndarray]] = None, ll_eps: float = 1.0e-4,
                  max_len: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float]]:
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
        ll_eps: Bin spacing is determined so that |LL - floor(LL/space)*space| < ll_eps
        max_len: Maximum number of bins for histogram
    Returns:
        log_density array, corresponding probs array, Tuple[ll0, dll, total_count]
    """
    entries = []

    if log_p1_vec is None:
        log_p1_vec = np.log1p(-np.exp(log_p_vec))

    if count_vec is None:
        count_vec = np.ones(len(log_p_vec))

    # Compute binomial log-densities and probabilities
    for log_p, log_p1, n in zip(log_p_vec, log_p1_vec, count_vec):
        if n == 0 or log_p == -np.inf or log_p1 == -np.inf:
            continue
        nn = np.arange(0, n + 1)
        llv = log_p * nn + log_p1 * (n - nn)
        ell = gammaln(n + 1) - gammaln(nn + 1) - gammaln(n - nn + 1)
        ell = np.exp(ell - ell.max())
        ell /= np.sum(ell)
        llv = llv[ell > 0]
        ell = ell[ell > 0]

        entries.append((llv, ell, n))

    # Find parameters for a common fixed-space grid [ll0, ll0 + dll, ll0 + 2*dll, ...]
    min_vec = np.asarray([entry[0].min() for entry in entries])
    llv_vec = np.concatenate([entry[0] - entry[0].min() for entry in entries])
    llv_vec = np.sort(np.unique(llv_vec))

    if max_len is not None:
        mll = np.sum([entry[0].max() - entry[0].min() for entry in entries])
        dll = mll / max_len
    else:
        dll = np.diff(llv_vec).min()
        while np.abs(llv_vec - np.floor(llv_vec / dll) * dll).max() > ll_eps:
            dll /= 2

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
    acc_ll = ll0 + np.arange(len(acc_prob))*dll
    return acc_ll, acc_prob, (ll0, dll, acc_count)



if __name__ == '__main__':

    pvec = np.asarray([0.3, 0.8, 0.4])
    pvec = np.log(pvec)
    nvec = np.log1p(-np.exp(pvec))
    cvec = np.asarray([2, 3, 3])

    pvec_long = np.concatenate([[u] * n for u, n in zip(pvec, cvec)])
    nvec_long = np.concatenate([[u] * n for u, n in zip(nvec, cvec)])

    test = np.asarray([1, 0, 1, 1, 0, 1, 0, 1])
    ll = np.where(test==1, pvec_long, nvec_long).sum()

    acc_ll, acc_prob, (ll0, dll, acc_count) = binomial_rank(pvec, count_vec=cvec, max_len=100000)
    left  = acc_prob[(int((ll - ll0) / dll) - 1):].sum() * np.power(2, acc_count)
    mid   = acc_prob[int((ll - ll0) / dll):].sum() * np.power(2, acc_count)
    right = acc_prob[(int((ll - ll0) / dll) + 1):].sum() * np.power(2, acc_count)
    print('Approximate rank: %f ( Somewhere in [%f, %f] )'%(mid, right, left))

    # Verify this
    temp = np.asarray([np.where([u == 1 for u in x], pvec_long, nvec_long).sum() for x in itertools.product([0, 1], repeat=len(pvec_long))])
    print('True rank:' + str((temp >= ll).sum()))

