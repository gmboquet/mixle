"""Sparse mixture scoring with a CERTIFIED tail bound -- the DSA 'top-k + bound the rest' idea.

DeepSeek Sparse Attention scores all candidates with a low-cost indexer, keeps the top-k, computes the
expensive thing only on those, and accepts a (bounded) error. The mixture analogue: each component has a
low-cost, x-independent upper bound ``log w_k + sup_x log p_k(x)`` (weight times peak density). Rank by it,
score the exact ``log w_k + log p_k(x)`` for only the top ``max_components``, and bound the dropped tail by
the sum of the remaining upper bounds. That yields a *certified bracket* ``[lower, upper]`` provably
containing the true ``log p(x)`` -- unlike DSA, the error is certified, not just hoped small.

The bound is valid only where every contributing component's density is BOUNDED (a finite peak): the
design review's scoping. Families with an unbounded density (Gamma shape<1, Beta with a<1 or b<1, ...)
return ``None`` from :func:`log_density_sup`, and scoring falls back to exact (no certification) for those.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


def log_density_sup(dist: Any) -> float | None:
    """Maximum ``log p(x)`` of a single distribution, or ``None`` if unbounded / unknown.

    A finite value certifies the component can contribute at most this much, which is what makes the
    dropped-tail bound sound. ``None`` (unbounded density or an unrecognized family) forces the exact path.
    """
    t = type(dist).__name__
    if t == "GaussianDistribution":  # peak at the mean
        return -0.5 * math.log(2.0 * math.pi * float(dist.sigma2))
    if t == "CategoricalDistribution":
        pm = getattr(dist, "pmap", None)
        if pm:
            m = max(float(v) for v in pm.values())
            return math.log(m) if m > 0 else None
    if t == "PoissonDistribution":  # max pmf at the mode floor(lam)
        lam = float(dist.lam)
        if lam <= 0:
            return 0.0
        k = math.floor(lam)
        return k * math.log(lam) - lam - math.lgamma(k + 1)
    if t == "BernoulliDistribution":
        p = float(dist.p)
        return math.log(max(p, 1.0 - p))
    return None  # unbounded (Gamma shape<1, Beta a<1/b<1, ...) or unrecognized -> exact fallback


def _logsumexp(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr) | (arr == -np.inf)]
    if arr.size == 0 or np.all(arr == -np.inf):
        return float("-inf")
    m = float(np.max(arr))
    if m == -np.inf:
        return float("-inf")
    return m + math.log(float(np.sum(np.exp(arr - m))))


@dataclass(frozen=True)
class SparseScore:
    """A certified bracket on a mixture's ``log p(x)`` from scoring only the top components.

    ``lower <= log p(x) <= upper`` always holds; ``exact`` is True when the bracket collapsed (all
    components scored, or ``max_components`` covered the support). ``n_scored`` components were exactly
    evaluated. When ``exact``, ``lower == upper == log p(x)``.
    """

    lower: float
    upper: float
    exact: bool
    n_scored: int


def sparse_mixture_score(mixture: Any, x: Any, max_components: int) -> SparseScore:
    """Score ``mixture.log_density(x)`` exactly on the top ``max_components`` and certify the rest.

    Ranks components by the low-cost upper bound ``log w_k + sup_k`` (x-independent), scores the exact
    contribution of only the top ones, and bounds the dropped tail by the remaining upper bounds. Returns
    a :class:`SparseScore` bracket. If any positive-weight component's density is unbounded
    (``log_density_sup`` is ``None``), falls back to exact full scoring (``lower == upper``, no speedup).
    """
    comps = list(mixture.components)
    log_w = np.asarray(mixture.log_w, dtype=np.float64)
    active = [k for k in range(len(comps)) if log_w[k] > -np.inf]

    sups = {k: log_density_sup(comps[k]) for k in active}
    if any(sups[k] is None for k in active):  # cannot certify -> exact
        exact = float(mixture.log_density(x))
        return SparseScore(exact, exact, True, len(active))

    bounds = sorted(active, key=lambda k: log_w[k] + sups[k], reverse=True)
    keep = bounds[:max_components]
    drop = bounds[max_components:]

    kept_scores = [float(log_w[k]) + float(comps[k].log_density(x)) for k in keep]
    lower = _logsumexp(kept_scores)
    tail_upper = _logsumexp([float(log_w[k]) + float(sups[k]) for k in drop]) if drop else float("-inf")
    upper = _logsumexp([lower, tail_upper])
    return SparseScore(lower, upper, not drop, len(keep))


def collapse_identical(mixture: Any) -> Any:
    """Merge components that are identical (same family + parameters) by summing their weights -- EXACT.

    A fitted huge mixture often carries duplicate components; pooling exact duplicates leaves ``log p(x)``
    unchanged while shrinking K. Identity is keyed on the component's string form (which encodes its
    parameters). 'Blend a mixture of closed forms' with zero approximation.
    """
    from mixle.stats.latent.mixture import MixtureDistribution

    comps = list(mixture.components)
    w = np.asarray(mixture.w, dtype=np.float64)
    groups: dict[str, list[int]] = {}
    order: list[str] = []
    for k, c in enumerate(comps):
        key = str(c)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(k)
    new_comps = [comps[groups[key][0]] for key in order]
    new_w = [float(sum(float(w[k]) for k in groups[key])) for key in order]
    return MixtureDistribution(new_comps, new_w)


def _merge_gaussians(comps: list[Any], weights: list[float]) -> tuple[Any, float]:
    """Moment-match a weighted group of Gaussians to one Gaussian preserving combined weight/mean/var."""
    import mixle.stats as st

    w = np.asarray(weights, dtype=np.float64)
    wt = float(w.sum())
    mus = np.array([float(c.mu) for c in comps])
    s2 = np.array([float(c.sigma2) for c in comps])
    mu = float((w * mus).sum() / wt)
    var = float((w * (s2 + mus**2)).sum() / wt - mu**2)
    return st.GaussianDistribution(mu, max(var, 1e-12)), wt


def collapse_gaussian_mixture(mixture: Any, max_components: int) -> Any:
    """Reduce an all-Gaussian mixture to ``<= max_components`` by moment-matching the nearest pair.

    Greedily merges the two closest components (by mean) into the single Gaussian preserving their
    combined weight, mean, and variance, until ``max_components`` remain. Approximate (it widens), but it
    preserves the OVERALL mixture mean and variance exactly -- the analytic 'collapse a huge mixture of
    closed forms' that lets scoring/sampling scale.
    """
    from mixle.stats.latent.mixture import MixtureDistribution

    comps = list(mixture.components)
    w = [float(x) for x in mixture.w]
    if any(type(c).__name__ != "GaussianDistribution" for c in comps):
        raise ValueError("collapse_gaussian_mixture requires all-Gaussian components")
    while len(comps) > max_components:
        best = None
        for i in range(len(comps)):
            for j in range(i + 1, len(comps)):
                d = abs(float(comps[i].mu) - float(comps[j].mu))
                if best is None or d < best[0]:
                    best = (d, i, j)
        _, i, j = best  # type: ignore[misc]
        merged, wt = _merge_gaussians([comps[i], comps[j]], [w[i], w[j]])
        comps = [c for k, c in enumerate(comps) if k not in (i, j)] + [merged]
        w = [x for k, x in enumerate(w) if k not in (i, j)] + [wt]
    return MixtureDistribution(comps, w)
