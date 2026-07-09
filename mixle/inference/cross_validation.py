"""Cross-validation fold generators, including the dependence-aware schemes.

Random k-fold silently assumes the rows are exchangeable. When they are not -- a time series, a
spatial field, repeated measures on the same subject -- a random split leaks information from
neighbours of a test point into the training set, so the held-out score overstates real
out-of-sample skill. This is the single most common silent error in evaluating models on
autocorrelated data. The fix is to make the fold boundaries respect the dependence:

  * **i.i.d.** -- :func:`kfold`, :func:`leave_one_out`, :func:`stratified_kfold` (preserve class
    balance).
  * **grouped** (repeated measures) -- :func:`group_kfold`, :func:`leave_one_group_out`: never split a
    group across train and test.
  * **temporal** -- :func:`blocked_kfold` (contiguous blocks), :func:`time_series_split` (forward-
    chaining, train always precedes test), :func:`purged_kfold` (an embargo buffer around each test
    block removed from train).
  * **spatial** -- :func:`spatial_block_kfold`: hold out contiguous spatial blocks, not scattered
    points.
  * **nested** -- :func:`nested_kfold`: an inner CV inside each outer fold for tuned-model
    evaluation.

Every generator returns a list of ``(train_index, test_index)`` integer-array pairs, so they are
interchangeable wherever a list of folds is expected (a pluggable fold source).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.random import RandomState

Fold = tuple[np.ndarray, np.ndarray]


def _as_rng(seed: int | RandomState | None) -> RandomState:
    if isinstance(seed, RandomState):
        return seed
    return RandomState(seed)


def _fold_sizes(n: int, k: int) -> np.ndarray:
    """Near-equal fold sizes: the first ``n % k`` folds get one extra element (sklearn convention)."""
    sizes = np.full(k, n // k, dtype=int)
    sizes[: n % k] += 1
    return sizes


def kfold(n: int, n_splits: int = 5, *, shuffle: bool = False, seed: int | RandomState | None = 0) -> list[Fold]:
    """Standard k-fold split of ``n`` rows.

    Args:
        n: number of observations.
        n_splits: number of folds ``k`` (each row is in exactly one test fold).
        shuffle: shuffle the row order before splitting (use for i.i.d. data; leave False to keep
            contiguous folds, i.e. :func:`blocked_kfold`).
        seed: RNG seed when ``shuffle`` is True.

    Returns:
        ``k`` ``(train_index, test_index)`` pairs.
    """
    if not 2 <= n_splits <= n:
        raise ValueError("n_splits must be in [2, n].")
    idx = np.arange(n)
    if shuffle:
        _as_rng(seed).shuffle(idx)
    folds = []
    start = 0
    for size in _fold_sizes(n, n_splits):
        test = idx[start : start + size]
        train = np.concatenate([idx[:start], idx[start + size :]])
        folds.append((train, test))
        start += size
    return folds


def blocked_kfold(n: int, n_splits: int = 5) -> list[Fold]:
    """Contiguous-block k-fold (no shuffle) for serially dependent data.

    Identical to :func:`kfold` with ``shuffle=False``; named separately because keeping blocks
    contiguous is the *point* for time series, not an incidental default.
    """
    return kfold(n, n_splits, shuffle=False)


def leave_one_out(n: int) -> list[Fold]:
    """Leave-one-out CV: ``n`` folds, each holding out a single observation."""
    return kfold(n, n, shuffle=False)


def stratified_kfold(
    y: np.ndarray, n_splits: int = 5, *, shuffle: bool = True, seed: int | RandomState | None = 0
) -> list[Fold]:
    """Stratified k-fold preserving class proportions in every fold.

    Distributes each class's indices across the folds so that class frequencies in each test fold
    mirror the overall frequencies -- important for imbalanced classification.

    Args:
        y: ``(n,)`` class labels.
        n_splits: number of folds.
        shuffle: shuffle within each class before round-robin assignment.
        seed: RNG seed when ``shuffle`` is True.

    Returns:
        ``n_splits`` ``(train_index, test_index)`` pairs.
    """
    y = np.asarray(y)
    n = y.shape[0]
    rng = _as_rng(seed)
    test_assign = np.empty(n, dtype=int)
    for c in np.unique(y):
        pos = np.nonzero(y == c)[0]
        if shuffle:
            rng.shuffle(pos)
        test_assign[pos] = np.arange(len(pos)) % n_splits
    folds = []
    all_idx = np.arange(n)
    for f in range(n_splits):
        test = all_idx[test_assign == f]
        train = all_idx[test_assign != f]
        folds.append((train, test))
    return folds


def leave_one_group_out(groups: np.ndarray) -> list[Fold]:
    """Leave-one-group-out CV: one fold per distinct group held out entirely.

    Args:
        groups: ``(n,)`` group labels (e.g. subject / site / genus id).

    Returns:
        One ``(train_index, test_index)`` pair per unique group.
    """
    groups = np.asarray(groups)
    all_idx = np.arange(groups.shape[0])
    folds = []
    for g in np.unique(groups):
        test = all_idx[groups == g]
        train = all_idx[groups != g]
        folds.append((train, test))
    return folds


def group_kfold(groups: np.ndarray, n_splits: int = 5) -> list[Fold]:
    """Group k-fold: partition *groups* into ``k`` folds so no group spans train and test.

    Greedily assigns whole groups (largest first) to the currently-smallest fold, balancing fold sizes
    while keeping each group intact -- the right scheme for repeated measures when there are more
    groups than folds.
    """
    groups = np.asarray(groups)
    uniq, counts = np.unique(groups, return_counts=True)
    if n_splits > len(uniq):
        raise ValueError("n_splits cannot exceed the number of groups.")
    order = np.argsort(-counts)
    fold_of_group: dict = {}
    fold_load = np.zeros(n_splits, dtype=int)
    for gi in order:
        f = int(np.argmin(fold_load))
        fold_of_group[uniq[gi]] = f
        fold_load[f] += counts[gi]
    assign = np.array([fold_of_group[g] for g in groups])
    all_idx = np.arange(groups.shape[0])
    return [(all_idx[assign != f], all_idx[assign == f]) for f in range(n_splits)]


def time_series_split(n: int, n_splits: int = 5, *, gap: int = 0, max_train_size: int | None = None) -> list[Fold]:
    """Forward-chaining time-series CV: train always precedes test (expanding window).

    The series is cut into ``n_splits + 1`` contiguous blocks; fold ``i`` tests on block ``i+1`` and
    trains on everything strictly before it. A ``gap`` (buffer) drops the ``gap`` points just before
    each test block from the training set so leakage through short-range autocorrelation is avoided.

    Args:
        n: number of time-ordered observations.
        n_splits: number of train/test splits.
        gap: number of observations to drop between the train and test segments.
        max_train_size: cap on the training-window length (sliding window); None keeps it expanding.

    Returns:
        ``n_splits`` ``(train_index, test_index)`` pairs.
    """
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1.")
    test_size = n // (n_splits + 1)
    if test_size < 1:
        raise ValueError("n is too small for the requested n_splits.")
    folds = []
    for i in range(n_splits):
        test_start = test_size * (i + 1)
        test_end = n if i == n_splits - 1 else test_start + test_size
        train_end = test_start - gap
        if train_end <= 0:
            continue
        train_start = 0 if max_train_size is None else max(0, train_end - max_train_size)
        folds.append((np.arange(train_start, train_end), np.arange(test_start, test_end)))
    return folds


def purged_kfold(n: int, n_splits: int = 5, *, embargo: int = 0) -> list[Fold]:
    """Purged/embargoed blocked k-fold (a.k.a. buffered CV).

    Contiguous test blocks, but the ``embargo`` observations on *each* side of a test block are
    removed from the training set, so no training point sits within the autocorrelation reach of a
    test point. The standard guard for serially dependent data when you still want every block to
    serve as a test fold (cf. de Prado 2018).

    Args:
        n: number of time-ordered observations.
        n_splits: number of contiguous folds.
        embargo: buffer width removed from train on both sides of each test block.

    Returns:
        ``n_splits`` ``(train_index, test_index)`` pairs.
    """
    folds = []
    start = 0
    for size in _fold_sizes(n, n_splits):
        test = np.arange(start, start + size)
        lo = max(0, start - embargo)
        hi = min(n, start + size + embargo)
        train = np.concatenate([np.arange(0, lo), np.arange(hi, n)])
        folds.append((train, test))
        start += size
    return folds


def spatial_block_kfold(
    coords: np.ndarray,
    n_splits: int = 5,
    *,
    block_size: float | None = None,
    n_side: int | None = None,
    seed: int | RandomState | None = 0,
) -> list[Fold]:
    """Spatial-block k-fold: hold out contiguous spatial blocks, not scattered points.

    Partitions space into a regular grid of blocks and assigns whole blocks to folds at random, so a
    test point's spatial neighbours are held out with it rather than leaking into training. This is the
    right scheme for geostatistical / spatially autocorrelated data.

    Args:
        coords: ``(n, d)`` spatial coordinates (typically ``d = 2``).
        n_splits: number of folds.
        block_size: grid-cell width in coordinate units; if None it is derived from ``n_side``.
        n_side: number of grid cells per axis; defaults so there are comfortably more blocks than
            folds.
        seed: RNG seed for assigning blocks to folds.

    Returns:
        ``n_splits`` ``(train_index, test_index)`` pairs.
    """
    coords = np.atleast_2d(np.asarray(coords, dtype=float))
    n, d = coords.shape
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    if block_size is not None:
        cell = np.full(d, float(block_size))
    else:
        if n_side is None:
            n_side = max(2, int(np.ceil((4 * n_splits) ** (1.0 / d))))
        cell = span / n_side
    block_idx = np.floor((coords - lo) / cell).astype(int)
    block_idx = np.minimum(block_idx, np.floor(span / cell).astype(int))  # clamp the max edge
    _, block_id = np.unique(block_idx, axis=0, return_inverse=True)
    n_blocks = block_id.max() + 1
    rng = _as_rng(seed)
    block_fold = rng.permutation(n_blocks) % n_splits
    assign = block_fold[block_id]
    all_idx = np.arange(n)
    return [(all_idx[assign != f], all_idx[assign == f]) for f in range(n_splits)]


@dataclass
class NestedFold:
    """One outer fold of :func:`nested_kfold`.

    Attributes:
        train: outer training indices (used to build the inner CV).
        test: outer test indices (held out for the final, untuned-on evaluation).
        inner: inner ``(train_index, test_index)`` folds, **indexing into the original array** (not
            into ``train``), so they can be applied directly.
    """

    train: np.ndarray
    test: np.ndarray
    inner: list[Fold]


def nested_kfold(
    n: int,
    *,
    outer_splits: int = 5,
    inner_splits: int = 4,
    shuffle: bool = False,
    seed: int | RandomState | None = 0,
) -> list[NestedFold]:
    """Nested k-fold: an inner CV (for tuning) inside each outer fold (for evaluation).

    The outer loop estimates generalisation; the inner loop selects hyper-parameters using only the
    outer-training data, so the reported score is not optimistically biased by tuning on the test set.

    Args:
        n: number of observations.
        outer_splits: number of outer folds.
        inner_splits: number of inner folds within each outer training set.
        shuffle: shuffle before splitting (i.i.d. data only).
        seed: RNG seed.

    Returns:
        A list of :class:`NestedFold`; inner folds index the original array.
    """
    outer = kfold(n, outer_splits, shuffle=shuffle, seed=seed)
    out = []
    for train, test in outer:
        inner_local = kfold(len(train), inner_splits, shuffle=shuffle, seed=seed)
        inner = [(train[tr], train[te]) for tr, te in inner_local]
        out.append(NestedFold(train=train, test=test, inner=inner))
    return out


__all__ = [
    "Fold",
    "kfold",
    "blocked_kfold",
    "leave_one_out",
    "stratified_kfold",
    "leave_one_group_out",
    "group_kfold",
    "time_series_split",
    "purged_kfold",
    "spatial_block_kfold",
    "nested_kfold",
    "NestedFold",
]
