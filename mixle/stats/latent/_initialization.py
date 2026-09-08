"""Symmetry-breaking starts shared by the latent families.

EM cannot leave a symmetric fixed point. A latent model initialized by drawing each observation's
component or state uniformly at random gives every component the same random ~1/k subsample of the
data, so every component's emission starts at the marginal, every transition row starts uniform, and
the first several dozen iterations move the objective by hundredths of a nat while the model sits on
that plateau -- or never leaves it. The mixture family answered this with a k-means++ start; this
module is that start, factored out so the hidden-Markov families can take it too (A-01, P09-F07).
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

LLOYD_ITERATIONS = 20
"""Lloyd refinements after seeding. Enough to pull an outlier seed onto its cluster's mass."""

MIN_CLUSTER_FRACTION = 0.05
"""A cluster below this share of ``n / k`` rows is an outlier the seeding latched onto, not a group."""


def numeric_feature_matrix(encoded: Any, rows: int | None = None) -> np.ndarray | None:
    """Best-effort ``(n, d)`` float matrix from an encoded batch, or ``None`` when there is none.

    ``None`` is the signal to fall back to the random start: a composite or ragged encoding, a
    non-numeric dtype, or a non-finite value has no vector space for k-means to work in.

    FLOATING dtypes only, and that is the whole point rather than an incidental strictness. k-means
    needs the distance between two encoded values to mean something, and for a categorical family the
    encoded value is a LABEL: ``IntegerCategoricalDistribution`` writes symbol codes as ``int64``, and
    the distance between code 3 and code 4 says nothing about the symbols. Clustering them anyway
    handed each state a contiguous slice of the alphabet, so an eight-state HMM over twelve symbols
    came back with one or two symbols in each state's support where 0.8.1 kept ten to twelve -- and a
    streaming re-estimate then met evidence no state could score. Every family whose values carry
    metric meaning encodes as float, counts included (``PoissonDistribution`` encodes ``float64``),
    so this keeps the start exactly where it belongs.
    """
    try:
        arr = np.asarray(encoded)
    except (TypeError, ValueError):
        return None
    if arr.dtype == object or not np.issubdtype(arr.dtype, np.floating):
        return None
    if arr.ndim == 1:
        arr = arr[:, None]
    elif arr.ndim != 2:
        return None
    if rows is not None and arr.shape[0] != rows:
        return None
    if arr.shape[0] == 0 or not np.isfinite(arr).all():
        return None
    return np.asarray(arr, dtype=float)


def kmeanspp_assignment(
    features: np.ndarray,
    groups: int,
    rng: np.random.RandomState,
    *,
    require_balanced: bool = True,
) -> np.ndarray | None:
    """Assign each row of ``features`` to one of ``groups`` by k-means++ seeding plus Lloyd steps.

    Returns ``None`` -- meaning "use the random start instead" -- when there are fewer rows than
    groups, or when a cluster ends up below :data:`MIN_CLUSTER_FRACTION` of its fair share and
    ``require_balanced`` is set. That guard is the 0.8.1 repair for a start that latched onto an
    outlier and handed one component a handful of rows, from which EM shrank it into a degenerate
    spike it never left.
    """
    n = int(features.shape[0])
    k = int(groups)
    if k < 1 or n < k:
        return None
    if k == 1:
        return np.zeros(n, dtype=int)

    centers_idx = np.empty(k, dtype=int)
    centers_idx[0] = rng.randint(n)
    closest_sq = np.sum((features - features[centers_idx[0]]) ** 2, axis=1)
    for c in range(1, k):
        total = float(closest_sq.sum())
        if total <= 0.0 or not np.isfinite(total):
            centers_idx[c] = rng.randint(n)
        else:
            centers_idx[c] = int(rng.choice(n, p=closest_sq / total))
        closest_sq = np.minimum(closest_sq, np.sum((features - features[centers_idx[c]]) ** 2, axis=1))

    centers = features[centers_idx].copy()
    assign = np.argmin(np.sum((features[:, None, :] - centers[None, :, :]) ** 2, axis=2), axis=1)
    for _lloyd in range(LLOYD_ITERATIONS):
        for c in range(k):
            members = assign == c
            if np.any(members):
                centers[c] = features[members].mean(axis=0)
        new_assign = np.argmin(np.sum((features[:, None, :] - centers[None, :, :]) ** 2, axis=2), axis=1)
        if np.array_equal(new_assign, assign):
            break
        assign = new_assign

    if require_balanced:
        min_members = max(features.shape[1] + 1, int(MIN_CLUSTER_FRACTION * n / k))
        if int(np.min(np.bincount(assign, minlength=k))) < min_members:
            return None
    return np.asarray(assign, dtype=int)


def derived_rng(material: np.ndarray) -> np.random.RandomState:
    """A generator derived from ``material``, so ATTEMPTING a start consumes no shared stream.

    The symmetry-breaking start may decline (non-numeric emissions, unbalanced clusters, a split that
    leaves a state with no weight), and the caller then falls back to the random draw it already made.
    Had the attempt drawn from that same generator, the fallback would no longer reproduce what the
    old code produced -- the fallback path would change even where the new start does not apply. A
    generator seeded from the draw itself is deterministic, independent of the caller's stream, and
    leaves every declined case bit-identical.
    """
    digest = hashlib.blake2b(np.ascontiguousarray(material).tobytes(), digest_size=8).digest()
    return np.random.RandomState(int.from_bytes(digest, "big") % (2**32))


def broken_symmetry_states(
    encoded: Any,
    num_states: int,
    rng: np.random.RandomState,
    *,
    rows: int | None = None,
    weights: np.ndarray | None = None,
) -> np.ndarray | None:
    """A per-observation state assignment that does not start every state at the marginal.

    The hidden-Markov families drew each observation's state uniformly, which makes every state's
    emission the sample marginal and every transition row uniform -- the symmetric fixed point EM
    cannot leave. On a well-separated two-state problem three of five seeds collapsed to identical
    emissions and stayed there through 400 iterations (P09-F07), and an ordinary categorical-emission
    HMM spent its first ~35 iterations moving the objective by 0.05 (A-01).

    ``None`` means the emissions are not a numeric vector space -- categorical, composite, ragged --
    and the caller keeps the random draw it has always used.
    """
    features = numeric_feature_matrix(encoded, rows)
    if features is None:
        return None
    assign = kmeanspp_assignment(features, num_states, rng)
    if assign is None or weights is None:
        return assign
    # A cluster is a group of OBSERVATIONS; a state needs a group of observations that carry WEIGHT.
    # Under sub-sampled initialization (``seq_initialize``'s ``p``) or explicit per-row weights, a
    # cluster can be made entirely of zero-weight rows, which would leave that state with no data at
    # all and its emission at the estimator's parameter floor -- worse than the marginal start this
    # replaces. When that happens the caller keeps its random draw.
    per_state = np.bincount(assign, weights=np.asarray(weights, dtype=float), minlength=num_states)
    return None if not np.all(per_state > 0.0) else assign
