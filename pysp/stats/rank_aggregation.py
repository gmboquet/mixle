"""Rank aggregation, consensus rankings, and permutation distances.

Given several orderings of the same items (voters' preferences, judges' rankings, search results to
fuse), recover a single *consensus* ordering and quantify how dispersed the inputs are:

  * :func:`borda_count` -- positional scoring (fast, the average-rank consensus).
  * :func:`copeland` -- pairwise-majority (Condorcet-flavoured) scoring.
  * :func:`kemeny_consensus` -- the median ranking minimising total Kendall-tau distance to the inputs
    (the maximum-likelihood Condorcet aggregation); exact for small item sets, local search beyond.
  * :func:`mallows_fit` -- fit a Mallows model: its central ranking (Kemeny consensus) plus a
    dispersion ``theta`` (larger = more agreement among voters).

Permutation distances :func:`kendall_distance`, :func:`spearman_footrule`, and :func:`cayley_distance`
are exposed directly. Orderings are passed as sequences of item ids, best first (a permutation of
``0..m-1``); ``rankings`` is a 2-D array with one ordering per row.
"""

from __future__ import annotations

from itertools import permutations

import numpy as np


def _as_rankings(rankings: np.ndarray) -> np.ndarray:
    r = np.atleast_2d(np.asarray(rankings, dtype=int))
    m = r.shape[1]
    for row in r:
        if sorted(row.tolist()) != list(range(m)):
            raise ValueError("each ranking must be a permutation of 0..m-1 (item ids, best first).")
    return r


def _positions(ordering: np.ndarray) -> np.ndarray:
    """Map an ordering (item ids best-first) to position-of-each-item."""
    pos = np.empty(len(ordering), dtype=int)
    pos[np.asarray(ordering, dtype=int)] = np.arange(len(ordering))
    return pos


def kendall_distance(a: np.ndarray, b: np.ndarray) -> int:
    """Kendall-tau distance: the number of item pairs ordered oppositely by ``a`` and ``b``."""
    pa, pb = _positions(a), _positions(b)
    m = len(pa)
    d = 0
    for i in range(m):
        for j in range(i + 1, m):
            if np.sign(pa[i] - pa[j]) != np.sign(pb[i] - pb[j]):
                d += 1
    return int(d)


def spearman_footrule(a: np.ndarray, b: np.ndarray) -> int:
    """Spearman footrule distance: the sum of absolute position differences across items."""
    return int(np.sum(np.abs(_positions(a) - _positions(b))))


def cayley_distance(a: np.ndarray, b: np.ndarray) -> int:
    """Cayley distance: the minimum number of transpositions turning ``a`` into ``b`` (m - #cycles)."""
    pa, pb = _positions(a), _positions(b)
    # permutation taking a's order to b's order
    perm = pb[np.argsort(pa)]
    seen = np.zeros(len(perm), dtype=bool)
    cycles = 0
    for i in range(len(perm)):
        if not seen[i]:
            cycles += 1
            j = i
            while not seen[j]:
                seen[j] = True
                j = perm[j]
    return int(len(perm) - cycles)


def borda_count(rankings: np.ndarray) -> dict[str, np.ndarray]:
    """Borda positional aggregation: each item scores ``(m - 1 - position)`` summed over voters.

    Returns:
        ``{'consensus', 'scores'}`` -- the consensus ordering (best first) and per-item Borda scores.
    """
    r = _as_rankings(rankings)
    m = r.shape[1]
    scores = np.zeros(m)
    for row in r:
        scores[row] += (m - 1) - np.arange(m)
    consensus = np.argsort(-scores, kind="stable")
    return {"consensus": consensus, "scores": scores}


def copeland(rankings: np.ndarray) -> dict[str, np.ndarray]:
    """Copeland pairwise-majority aggregation.

    Each ordered pair contributes a win/loss by majority across voters; an item's score is wins minus
    losses. Closely tracks the Condorcet winner when one exists.

    Returns:
        ``{'consensus', 'scores', 'wins'}``.
    """
    r = _as_rankings(rankings)
    n, m = r.shape
    pos = np.stack([_positions(row) for row in r])
    score = np.zeros(m)
    wins = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            i_before_j = np.sum(pos[:, i] < pos[:, j])
            wins[i, j] = i_before_j
            if i_before_j > n / 2:
                score[i] += 1
            elif i_before_j < n / 2:
                score[i] -= 1
    consensus = np.argsort(-score, kind="stable")
    return {"consensus": consensus, "scores": score, "wins": wins}


def _total_kendall(candidate: np.ndarray, rankings: np.ndarray) -> int:
    return int(sum(kendall_distance(candidate, row) for row in rankings))


def kemeny_consensus(rankings: np.ndarray, *, exact_max_items: int = 8) -> dict:
    """Kemeny median ranking: minimise the total Kendall-tau distance to all input orderings.

    The Kemeny consensus is the maximum-likelihood aggregation under the Mallows--Kendall model and the
    Condorcet-consistent choice. Exact by enumeration when ``m <= exact_max_items``; otherwise a local
    search (adjacent transpositions) from the Borda ordering.

    Returns:
        ``{'consensus', 'distance', 'exact'}`` -- the consensus, its total Kendall distance, and whether
        the result is exact.
    """
    r = _as_rankings(rankings)
    m = r.shape[1]
    if m <= exact_max_items:
        best, best_d = None, np.inf
        for perm in permutations(range(m)):
            d = _total_kendall(np.array(perm), r)
            if d < best_d:
                best, best_d = np.array(perm), d
        return {"consensus": best, "distance": int(best_d), "exact": True}

    # local search from Borda
    cur = borda_count(r)["consensus"].copy()
    cur_d = _total_kendall(cur, r)
    improved = True
    while improved:
        improved = False
        for i in range(m - 1):
            cand = cur.copy()
            cand[i], cand[i + 1] = cand[i + 1], cand[i]
            d = _total_kendall(cand, r)
            if d < cur_d:
                cur, cur_d = cand, d
                improved = True
    return {"consensus": cur, "distance": int(cur_d), "exact": False}


def _mallows_expected_distance(theta: float, m: int) -> float:
    """E[Kendall distance] under Mallows(theta) via the independent V_j decomposition."""
    total = 0.0
    for j in range(1, m):  # component j ranges over v in 0..(m-j)
        vmax = m - j
        v = np.arange(vmax + 1)
        w = np.exp(-theta * v)
        total += float(np.sum(v * w) / np.sum(w))
    return total


def mallows_fit(rankings: np.ndarray, *, exact_max_items: int = 8) -> dict:
    """Fit a Mallows model (Kendall): central ranking + dispersion ``theta``.

    The central ranking is the :func:`kemeny_consensus`; ``theta`` is the MLE concentration, found by
    matching the observed mean Kendall distance to its expectation under the model. Larger ``theta``
    means tighter agreement (``theta -> 0`` is uniform/no-consensus).

    Returns:
        ``{'center', 'theta', 'mean_distance'}``.
    """
    r = _as_rankings(rankings)
    m = r.shape[1]
    km = kemeny_consensus(r, exact_max_items=exact_max_items)
    center = km["consensus"]
    mean_d = float(np.mean([kendall_distance(center, row) for row in r]))
    max_d = m * (m - 1) / 2
    if mean_d <= 1e-9:
        theta = float("inf")
    elif mean_d >= max_d / 2:
        theta = 0.0
    else:
        lo, hi = 1e-6, 50.0
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if _mallows_expected_distance(mid, m) > mean_d:
                lo = mid
            else:
                hi = mid
        theta = 0.5 * (lo + hi)
    return {"center": center, "theta": theta, "mean_distance": mean_d}


__all__ = [
    "kendall_distance",
    "spearman_footrule",
    "cayley_distance",
    "borda_count",
    "copeland",
    "kemeny_consensus",
    "mallows_fit",
]
