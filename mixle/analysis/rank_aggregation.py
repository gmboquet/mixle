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

import operator
from itertools import permutations
from typing import Any

import numpy as np


def _exact_nonnegative_integer(value: int, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a non-Boolean nonnegative integer.")
    try:
        integer = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a non-Boolean nonnegative integer.") from exc
    if integer < 0:
        raise ValueError(f"{name} must be nonnegative, got {integer}.")
    return int(integer)


def _validate_permutation(ordering: Any, *, m: int | None = None) -> np.ndarray:
    """Validate that ``ordering`` is an exact integer permutation of ``0..n-1`` and return it as a
    length-``n`` int array (``n == m`` if ``m`` is given).

    Every public entry point in this module funnels through here before any distance or aggregation
    runs (MXR-080-0107), so a malformed ranking is always rejected with a clear error rather than
    silently mishandled:

    - Fractional/non-finite entries are rejected *before* any cast to int -- a truncating int cast must
      never be allowed to manufacture a spurious permutation (``[0.9, 1.1]`` truncates to the
      superficially-valid ``[0, 1]`` if cast first, validated second).
    - Duplicate or out-of-range item ids are rejected outright, instead of silently scattering into
      uninitialized memory (a duplicate id) or crashing with a raw ``IndexError`` (an out-of-range id)
      the way unchecked fancy indexing (``pos[ordering] = ...``) does downstream.
    - When ``m`` is given (comparing/aggregating against another ranking of that size), a length
      mismatch is rejected explicitly instead of silently truncating the comparison to the shorter
      ranking's length -- which could otherwise report a meaningless zero distance between two rankings
      of different sizes.
    """
    raw = np.asarray(ordering)
    if raw.ndim != 1:
        raise ValueError(f"ranking must be a 1-D sequence of item ids (best first); got shape {raw.shape}")
    if raw.size == 0:
        raise ValueError("ranking must contain at least one item.")
    if raw.dtype == np.bool_ or (
        raw.dtype == object and any(isinstance(value, (bool, np.bool_)) for value in raw.tolist())
    ):
        raise ValueError("ranking item ids must not be Boolean values.")
    if np.issubdtype(raw.dtype, np.integer):
        arr = raw.astype(int)
    elif np.issubdtype(raw.dtype, np.floating):
        as_float = raw.astype(np.float64)
        if not np.all(np.isfinite(as_float)) or not np.all(as_float == np.round(as_float)):
            raise ValueError(
                f"ranking must contain only finite integer item ids, got {raw.tolist()} -- "
                "a fractional or non-finite id is never a valid permutation entry, even if truncating "
                "it to int would happen to land in range."
            )
        arr = np.round(as_float).astype(int)
    else:
        raise ValueError("ranking must contain numeric integer item ids.")
    n = arr.shape[0]
    if m is not None and n != m:
        raise ValueError(f"rankings must be permutations of the same {m} items; got a length-{n} ranking")
    if sorted(arr.tolist()) != list(range(n)):
        raise ValueError(
            f"ranking must be a permutation of 0..{n - 1} (item ids, best first, no duplicates/gaps); "
            f"got {arr.tolist()}"
        )
    return arr


def _as_rankings(rankings: np.ndarray) -> np.ndarray:
    raw = np.asarray(rankings)
    if raw.ndim == 1:
        raw = raw[None, :]
    if raw.ndim != 2 or raw.shape[0] == 0 or raw.shape[1] == 0:
        raise ValueError(
            f"rankings must contain at least one ranking over a nonempty common item set; got shape {raw.shape}"
        )
    m = raw.shape[1]
    return np.stack([_validate_permutation(row, m=m) for row in raw])


def _positions(ordering: np.ndarray) -> np.ndarray:
    """Map an ordering (item ids best-first) to position-of-each-item."""
    arr = _validate_permutation(ordering)
    pos = np.empty(len(arr), dtype=int)
    pos[arr] = np.arange(len(arr))
    return pos


def _validate_pair(a: Any, b: Any) -> tuple[np.ndarray, np.ndarray]:
    """Validate ``a`` and ``b`` are each a permutation, and permutations of the *same* domain size --
    two rankings of different lengths are incomparable and must be rejected, not silently truncated to
    the shorter one's length (which could otherwise report a meaningless zero distance)."""
    va = _validate_permutation(a)
    vb = _validate_permutation(b, m=len(va))
    return va, vb


def kendall_distance(a: np.ndarray, b: np.ndarray) -> int:
    """Kendall-tau distance: the number of item pairs ordered oppositely by ``a`` and ``b``."""
    a, b = _validate_pair(a, b)
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
    a, b = _validate_pair(a, b)
    return int(np.sum(np.abs(_positions(a) - _positions(b))))


def cayley_distance(a: np.ndarray, b: np.ndarray) -> int:
    """Cayley distance: the minimum number of transpositions turning ``a`` into ``b`` (m - #cycles)."""
    a, b = _validate_pair(a, b)
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
    search (adjacent transpositions) from the Borda ordering. Exact enumeration evaluates ``m!``
    permutations, so ``exact_max_items`` is an explicit factorial resource boundary; it must be an
    exact non-Boolean nonnegative integer.

    Returns:
        ``{'consensus', 'distance', 'exact', 'search_mode', 'exact_max_items'}`` -- the consensus,
        total Kendall distance, whether the result is exact, and the requested/achieved search mode.
    """
    exact_max_items = _exact_nonnegative_integer(exact_max_items, name="exact_max_items")
    r = _as_rankings(rankings)
    m = r.shape[1]
    if m <= exact_max_items:
        best, best_d = None, np.inf
        for perm in permutations(range(m)):
            d = _total_kendall(np.array(perm), r)
            if d < best_d:
                best, best_d = np.array(perm), d
        return {
            "consensus": best,
            "distance": int(best_d),
            "exact": True,
            "search_mode": "exact_enumeration",
            "exact_max_items": exact_max_items,
        }

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
    return {
        "consensus": cur,
        "distance": int(cur_d),
        "exact": False,
        "search_mode": "adjacent_swap_local_search",
        "exact_max_items": exact_max_items,
    }


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
        ``{'center', 'theta', 'mean_distance', 'consensus_distance', 'exact', 'search_mode',
        'exact_max_items'}``, preserving the Kemeny search guarantee used to obtain the center.
    """
    exact_max_items = _exact_nonnegative_integer(exact_max_items, name="exact_max_items")
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
    return {
        "center": center,
        "theta": theta,
        "mean_distance": mean_d,
        "consensus_distance": km["distance"],
        "exact": km["exact"],
        "search_mode": km["search_mode"],
        "exact_max_items": km["exact_max_items"],
    }


__all__ = [
    "kendall_distance",
    "spearman_footrule",
    "cayley_distance",
    "borda_count",
    "copeland",
    "kemeny_consensus",
    "mallows_fit",
]
