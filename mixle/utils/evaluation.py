"""Held-out evaluation and data-partitioning utilities.

Empirical KL divergence between a fitted model and data, plus index/data partitioning helpers
(k-fold split, proportional split) used for validation and cross-validation.
"""

from collections.abc import Sequence
from typing import Any, TypeVar

import numpy as np
from numpy.random import RandomState

from mixle.stats import (
    seq_log_density,
)
from mixle.stats.compute.pdist import SequenceEncodableProbabilityDistribution

T = TypeVar("T")
E0 = TypeVar("E0")


def empirical_kl_divergence(
    dist1: SequenceEncodableProbabilityDistribution,
    dist2: SequenceEncodableProbabilityDistribution,
    enc_data: list[tuple[int, Any]],
) -> tuple[float, float, float]:
    """Computes the emirical KL-divergence between two densities.

    Compute the KL-divergence between dist1 and dist2, for encoded sequence of data. Dists must both have the
    same encodings.

    Args:
        dist1 (SequenceEncodableProbabilityDistribution): Distribution compatible with enc_data.
        dist2 (SequenceEncodableProbabilityDistribution): Distribution compatible with enc_data.
        enc_data (List[Tuple[int, Any]]): List of Tuple containing chunk size and encoded sequence for chunked data.

    Returns:
        Tuple of KL-divergence estimate and counts of invalid likelihood values for each distribution.

    """

    ll = seq_log_density(enc_data, estimate=(dist1, dist2))
    ll = np.hstack(ll)

    l1 = ll[0, :]
    l2 = ll[1, :]
    g1 = np.bitwise_and(l1 != -np.inf, ~np.isnan(l1))
    g2 = np.bitwise_and(l2 != -np.inf, ~np.isnan(l2))
    gg = np.bitwise_and(g1, g2)

    max_l1 = np.max(l1[gg])
    max_l2 = np.max(l2[gg])

    p1 = np.exp(l1[gg] - max_l1)
    p1 /= p1.sum()

    p2 = np.exp(l2[gg] - max_l2)
    p2 /= p2.sum()

    r1 = (p1 * (np.log(p1) - np.log(p2))).sum()
    r2 = (~g1).sum()
    r3 = (~g2).sum()

    return r1, r2, r3


def ks_test(data: Sequence[float], dist: Any) -> tuple[float, float]:
    """One-sample Kolmogorov-Smirnov goodness-of-fit test of ``data`` against ``dist``.

    Returns ``(D, p_value)`` where ``D = sup_x |F_n(x) - dist.cdf(x)|`` is the KS statistic and the
    two-sided p-value is the exact Kolmogorov distribution ``P(D_n >= D)`` (``scipy.stats.kstwo``). ``dist``
    must expose a scalar ``cdf`` (the ``HasCDF`` capability). A small p-value is evidence that ``data`` is
    not distributed as ``dist`` -- continuous goodness-of-fit / model checking.
    """
    from scipy.stats import kstwo

    x = np.sort(np.asarray(data, dtype=np.float64))
    n = x.size
    if n == 0:
        raise ValueError("ks_test requires at least one observation.")
    cdf = np.array([float(dist.cdf(float(xi))) for xi in x])
    idx = np.arange(1, n + 1, dtype=np.float64)
    d_plus = float(np.max(idx / n - cdf))
    d_minus = float(np.max(cdf - (idx - 1.0) / n))
    d = max(d_plus, d_minus)
    return d, float(kstwo.sf(d, n))


def chi_square_test(
    data: Sequence[int], dist: Any, *, lo: int | None = None, hi: int | None = None
) -> tuple[float, int, float]:
    """Pearson chi-square goodness-of-fit test for a discrete ``dist`` against integer ``data``.

    Bins the observations over each value in ``[lo, hi]`` plus a single combined tail bin for everything
    outside that window (so the expected cell probabilities sum to 1, using ``dist.cdf`` for the tail).
    Returns ``(chi2, dof, p_value)`` with ``dof = #cells - 1`` and the upper-tail chi-square p-value;
    ``lo``/``hi`` default to the data's min/max. A small p-value is evidence of misfit.
    """
    import math

    from scipy.stats import chi2 as _chi2

    x = np.asarray(data)
    n = x.size
    if n == 0:
        raise ValueError("chi_square_test requires at least one observation.")
    lo = int(np.min(x)) if lo is None else int(lo)
    hi = int(np.max(x)) if hi is None else int(hi)
    ks = list(range(lo, hi + 1))
    observed = np.array([np.sum(x == k) for k in ks] + [np.sum((x < lo) | (x > hi))], dtype=np.float64)
    tail_p = max(0.0, 1.0 - (float(dist.cdf(hi)) - float(dist.cdf(lo - 1))))
    probs = np.array([math.exp(dist.log_density(k)) for k in ks] + [tail_p], dtype=np.float64)
    expected = n * probs
    mask = expected > 0.0
    chi2 = float(np.sum((observed[mask] - expected[mask]) ** 2 / expected[mask]))
    dof = int(mask.sum()) - 1
    return chi2, dof, float(_chi2.sf(chi2, dof))


def k_fold_split_index(sz: int, k: int, rng: RandomState) -> np.ndarray:
    """Returns integer numpy index vector for k-fold split. Entry j is the fold-id for the j^{th} data point.

    Args:
        sz (int): Integer length of data points in data set.
        k (int): Integer number of folds for k-folds.
        rng (RandomState): RandomState for setting seed.

    Returns:
        1-d np.ndarray[int] of indices for each data points fold-id.

    """
    idx = rng.rand(sz)
    sidx = np.argsort(idx)

    rv = np.zeros(sz, dtype=int)
    for i in range(k):
        rv[sidx[np.arange(start=i, stop=sz, step=k, dtype=int)]] = i

    return rv


def partition_data_index(sz: int, pvec: list[float] | np.ndarray, rng: RandomState) -> list[np.ndarray]:
    """Returns List of np.ndarray[int] containing integers indexes for data partitions proportional to pvec.

    Args:
        sz (int): Integer value of total number of data observations.
        pvec (Union[List[float], np.ndarray]): Vector of proportions for each partition.
        rng (RandomState): RandomState for setting seed of random partitioning.

    Returns:
        List of numpy arrays containing indexes of each partition.

    """
    idx = rng.rand(sz)
    sidx = np.argsort(idx)

    rv = []
    p_tot = 0
    prev_idx = 0

    for p in pvec:
        next_idx = int(round(sz * (p_tot + p), 0))
        rv.append(sidx[prev_idx:next_idx])
        p_tot += p
        prev_idx = next_idx

    return rv


def partition_data(data: Sequence[T], pvec: list[float] | np.ndarray, rng: RandomState) -> list[list[T]]:
    """Partitions List of data into partitions, each with size equal to the proportion of pvec.

    Args:

        data (Sequence[T]): Sequence of data observations, each entry of type T.
        pvec (Union[List[float], np.ndarray]): List of length n, containing proportion of data to be held in each data
            partition.
        rng (RandomState): RandomState for setting seed on random partitioning of data.

    Returns:
        List of List containing data partitions of proportion equal to pvec.

    """
    idx_list = partition_data_index(len(data), pvec, rng)

    return [[data[i] for i in u] for u in idx_list]
