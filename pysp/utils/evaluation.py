"""Held-out evaluation and data-partitioning utilities.

Empirical KL divergence between a fitted model and data, plus index/data partitioning helpers
(k-fold split, proportional split) used for validation and cross-validation.
"""

from collections.abc import Sequence
from typing import Any, TypeVar

import numpy as np
from numpy.random import RandomState

from pysp.stats import (
    seq_log_density,
)
from pysp.stats.compute.pdist import SequenceEncodableProbabilityDistribution

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
        Tuple of KL-div estiamte, number of 'bad' likelihood values for dist1, 'bad' likelihood values for dist2.

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
