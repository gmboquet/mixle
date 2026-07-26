"""Held-out evaluation and data-partitioning utilities.

Empirical KL divergence between a fitted model and data, plus index/data partitioning helpers
(k-fold split, proportional split) used for validation and cross-validation.
"""

from collections.abc import Sequence
from operator import index
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
) -> tuple[float, int, int]:
    """Estimate ``E[log dist1(X) - log dist2(X)]`` on encoded observations.

    The estimator is the arithmetic mean of per-observation log-density
    differences on rows where both scores are finite. Invalid-score counts for
    each model are returned separately. At least one jointly finite row is
    required; positive infinity, negative infinity, and NaN are all invalid.

    Args:
        dist1 (SequenceEncodableProbabilityDistribution): Distribution compatible with enc_data.
        dist2 (SequenceEncodableProbabilityDistribution): Distribution compatible with enc_data.
        enc_data (List[Tuple[int, Any]]): List of Tuple containing chunk size and encoded sequence for chunked data.

    Returns:
        Estimate followed by invalid-score counts for ``dist1`` and ``dist2``.

    """

    if not enc_data:
        raise ValueError("empirical_kl_divergence requires non-empty encoded data")
    chunks = seq_log_density(enc_data, estimate=(dist1, dist2))
    if not chunks:
        raise ValueError("density evaluation returned no score chunks")
    try:
        ll = np.hstack(chunks)
    except ValueError as exc:
        raise ValueError("density score chunks must have one aligned row per model") from exc
    if ll.ndim != 2 or ll.shape[0] != 2 or ll.shape[1] == 0:
        raise ValueError("density evaluation must return a non-empty 2 by n score matrix")

    l1, l2 = np.asarray(ll[0], dtype=float), np.asarray(ll[1], dtype=float)
    finite1, finite2 = np.isfinite(l1), np.isfinite(l2)
    jointly_finite = finite1 & finite2
    if not jointly_finite.any():
        raise ValueError("empirical KL has no observations with two finite log densities")
    estimate = float(np.mean(l1[jointly_finite] - l2[jointly_finite]))
    if not np.isfinite(estimate):
        raise ValueError("empirical KL produced a non-finite estimate")
    return estimate, int((~finite1).sum()), int((~finite2).sum())


def _exact_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    try:
        return int(index(value))
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _scalar_probability(value: Any, name: str) -> float:
    array = np.asarray(value, dtype=float)
    if array.size != 1:
        raise ValueError(f"{name} must return exactly one scalar value")
    result = float(array.reshape(-1)[0])
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must return a finite probability in [0, 1]")
    return result


def _scalar_log_probability(value: Any, name: str) -> float:
    array = np.asarray(value, dtype=float)
    if array.size != 1:
        raise ValueError(f"{name} must return exactly one scalar value")
    result = float(array.reshape(-1)[0])
    if np.isnan(result) or np.isposinf(result):
        raise ValueError(f"{name} must return a finite log probability or negative infinity")
    return result


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
    from scipy.stats import chi2 as _chi2

    if isinstance(data, (str, bytes)):
        raise TypeError("chi_square_test data must be a sequence of integers")
    raw = list(data)
    if not raw:
        raise ValueError("chi_square_test requires at least one observation.")
    x = np.asarray([_exact_integer(value, "observations") for value in raw], dtype=np.int64)
    n = x.size
    lo = int(np.min(x)) if lo is None else _exact_integer(lo, "lo")
    hi = int(np.max(x)) if hi is None else _exact_integer(hi, "hi")
    if lo > hi:
        raise ValueError("lo must be less than or equal to hi")
    if not callable(getattr(dist, "cdf", None)) or not callable(getattr(dist, "log_density", None)):
        raise TypeError("dist must expose callable cdf and log_density methods")
    ks = list(range(lo, hi + 1))
    observed = np.array(
        [np.sum(x == k) for k in ks] + [np.sum((x < lo) | (x > hi))],
        dtype=np.float64,
    )
    cdf_hi = _scalar_probability(dist.cdf(hi), "dist.cdf")
    cdf_before = _scalar_probability(dist.cdf(lo - 1), "dist.cdf")
    if cdf_before > cdf_hi:
        raise ValueError("dist.cdf must be non-decreasing over the requested bounds")
    tail_p = 1.0 - (cdf_hi - cdf_before)
    log_probs = [_scalar_log_probability(dist.log_density(k), "dist.log_density") for k in ks]
    probs = np.array(
        [0.0 if np.isneginf(log_p) else float(np.exp(log_p)) for log_p in log_probs] + [tail_p],
        dtype=np.float64,
    )
    if (
        not np.all(np.isfinite(probs))
        or np.any(probs < 0.0)
        or not np.isclose(float(probs.sum()), 1.0, rtol=1e-8, atol=1e-10)
    ):
        raise ValueError("discrete cell probabilities must be finite, non-negative, and sum to one")
    expected = n * probs
    positive = expected > 0.0
    dof = int(positive.sum()) - 1
    if np.any((observed > 0.0) & ~positive):
        return float("inf"), max(dof, 0), 0.0
    if dof < 1:
        raise ValueError("chi-square p-value requires at least two positive-probability cells")
    statistic = float(np.sum((observed[positive] - expected[positive]) ** 2 / expected[positive]))
    return statistic, dof, float(_chi2.sf(statistic, dof))


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
