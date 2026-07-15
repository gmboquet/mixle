"""Regular vine (R-vine): the general vine that subsumes both the C-vine and the D-vine.

A C-vine forces every tree to be a star (one root linked to all); a D-vine forces every tree to be a path.
A **regular vine** lifts both restrictions: each tree may be ANY spanning tree (subject to the proximity
condition that consecutive trees nest correctly), so the pair-copula construction can follow whatever
dependence graph the data actually has. This is the general object (Bedford & Cooke 2002; Dißmann et al.
2013) of which C- and D-vines are special cases.

The practical payoff is **automatic structure selection**: rather than fixing the order by hand,
:class:`RVineCopulaEstimator` runs Dißmann's greedy algorithm -- tree 1 is the maximum spanning tree over
``|Kendall's tau|`` among all pairs; each deeper tree is the maximum spanning tree over the previous tree's
edges (respecting proximity), weighted by the conditional ``|tau|`` -- and fits the best pair-copula family
per edge as it goes. So the vine picks both its shape and its per-edge families from the data.

Because a vine IS a copula on ``(0,1)^d``, :class:`RVineCopulaDistribution` is a drop-in dependence core for
:class:`~mixle.stats.combinator.copula.CopulaDistribution`, exactly like the C-vine, D-vine, and the
elliptical/Archimedean cores.

This module reuses the bivariate pair copulas (with their ``h``-functions) from
:mod:`mixle.stats.multivariate.vine_copula`.

Reference: Dißmann, Brechmann, Czado & Kurowicka, "Selecting and estimating regular vine copulae and
application to financial returns" (Computational Statistics & Data Analysis, 2013).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mixle.stats.compute.pdist import (
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
)
from mixle.stats.multivariate._copula_common import (
    BufferedUScoreAccumulatorFactory,
    UScoreEncoder,
    weighted_kendall_tau,
)
from mixle.stats.multivariate.vine_copula import (
    _CLIP,
    _DEFAULT_CANDIDATES,
    _clip01,
    _fit_best_pair,
)


class _Edge:
    """One pair-copula in the vine: conditioned pair ``{a, b}``, conditioning set ``cond``, fitted copula.

    ``parents`` maps each conditioned variable to ``(prev_edge_index, variable)`` -- the previous-tree edge
    whose stored conditional CDF is this edge's input for that variable. ``None`` in tree 1 (inputs are the
    raw uniform columns ``a`` and ``b``).
    """

    __slots__ = ("a", "b", "cond", "copula", "parents")

    def __init__(self, a: int, b: int, cond: frozenset, copula: Any, parents: Any) -> None:
        self.a = a
        self.b = b
        self.cond = cond
        self.copula = copula
        self.parents = parents  # None (tree 1) or {a: (idx, var), b: (idx, var)}

    def constraint(self) -> frozenset:
        return self.cond | {self.a, self.b}


def _max_spanning_tree(
    n: int, weight: dict[tuple[int, int], float], allowed: set[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Prim's algorithm for a MAXIMUM spanning tree over nodes ``0..n-1`` using only ``allowed`` edges.

    Assumes the allowed graph is connected (true for a vine tree by the proximity condition). Returns the
    chosen ``(i, j)`` edges (``i < j``).
    """
    in_tree = {0}
    chosen: list[tuple[int, int]] = []
    while len(in_tree) < n:
        best, best_w = None, -np.inf
        for i, j in allowed:
            a, b = (i, j) if i < j else (j, i)
            if (a in in_tree) != (b in in_tree):  # exactly one endpoint inside -> a growing edge
                if weight[(a, b)] > best_w:
                    best, best_w = (a, b), weight[(a, b)]
        if best is None:  # disconnected under `allowed` -- add any missing node arbitrarily (defensive)
            missing = next(k for k in range(n) if k not in in_tree)
            in_tree.add(missing)
            continue
        chosen.append(best)
        in_tree.add(best[0] if best[1] in in_tree else best[1])
    return chosen


def _forward_vals(trees: list[list[_Edge]], u: np.ndarray) -> tuple[np.ndarray, list[list[dict[int, np.ndarray]]]]:
    """Forward pass: accumulate the log-density and the per-edge conditional CDFs ``val[edge][var]``."""
    u = _clip01(u)
    loglik = np.zeros(u.shape[0])
    all_vals: list[list[dict[int, np.ndarray]]] = []
    for t, tree in enumerate(trees):
        tree_vals: list[dict[int, np.ndarray]] = []
        for e in tree:
            if t == 0:
                ia, ib = u[:, e.a], u[:, e.b]
            else:
                pa, va = e.parents[e.a]
                pb, vb = e.parents[e.b]
                ia = all_vals[t - 1][pa][va]
                ib = all_vals[t - 1][pb][vb]
            loglik = loglik + e.copula.logpdf(ia, ib)
            tree_vals.append({e.a: e.copula.h(ia, ib), e.b: e.copula.h(ib, ia)})
        all_vals.append(tree_vals)
    return loglik, all_vals


def _select_and_fit(u: np.ndarray, w: np.ndarray, candidates: tuple[str, ...]) -> list[list[_Edge]]:
    """Dißmann's greedy selection + sequential fit: build tree by tree, max-spanning-tree on conditional |tau|."""
    u = _clip01(u)
    d = u.shape[1]
    trees: list[list[_Edge]] = []
    prev_vals: list[dict[int, np.ndarray]] = []

    # --- tree 1: MST over the raw variables, weight |tau(i, j)| ---
    weight: dict[tuple[int, int], float] = {}
    allowed: set[tuple[int, int]] = set()
    for i in range(d):
        for j in range(i + 1, d):
            weight[(i, j)] = abs(weighted_kendall_tau(u[:, i], u[:, j], w))
            allowed.add((i, j))
    tree1: list[_Edge] = []
    vals1: list[dict[int, np.ndarray]] = []
    for i, j in _max_spanning_tree(d, weight, allowed):
        pc = _fit_best_pair(u[:, i], u[:, j], w, candidates)
        tree1.append(_Edge(i, j, frozenset(), pc, None))
        vals1.append({i: pc.h(u[:, i], u[:, j]), j: pc.h(u[:, j], u[:, i])})
    trees.append(tree1)
    prev_vals = vals1

    # --- deeper trees: nodes = previous-tree edges; join adjacent edges (proximity); MST on conditional |tau| ---
    for t in range(1, d - 1):
        prev = trees[t - 1]
        m = len(prev)
        cand_weight: dict[tuple[int, int], float] = {}
        cand_allowed: set[tuple[int, int]] = set()
        cand_info: dict[tuple[int, int], tuple[int, int, frozenset, np.ndarray, np.ndarray, Any]] = {}
        for x in range(m):
            for y in range(x + 1, m):
                ex, ey = prev[x], prev[y]
                shared = ex.constraint() & ey.constraint()
                if len(shared) != t:  # proximity: must share exactly t common variables
                    continue
                ux = ex.constraint() - shared  # ex's unique variable
                uy = ey.constraint() - shared
                if len(ux) != 1 or len(uy) != 1:
                    continue
                a, b = next(iter(ux)), next(iter(uy))
                if a not in (ex.a, ex.b) or b not in (ey.a, ey.b):  # unique var must be CONDITIONED in its parent
                    continue
                ia, ib = prev_vals[x][a], prev_vals[y][b]
                cand_weight[(x, y)] = abs(weighted_kendall_tau(ia, ib, w))
                cand_allowed.add((x, y))
                cand_info[(x, y)] = (a, b, shared, ia, ib, (x, y))
        tree_t: list[_Edge] = []
        vals_t: list[dict[int, np.ndarray]] = []
        for x, y in _max_spanning_tree(m, cand_weight, cand_allowed):
            a, b, shared, ia, ib, (px, py) = cand_info[(x, y)]
            pc = _fit_best_pair(ia, ib, w, candidates)
            tree_t.append(_Edge(a, b, shared, pc, {a: (px, a), b: (py, b)}))
            vals_t.append({a: pc.h(ia, ib), b: pc.h(ib, ia)})
        trees.append(tree_t)
        prev_vals = vals_t
    return trees


class RVineCopulaDistribution(SequenceEncodableProbabilityDistribution):
    """A regular-vine copula on ``(0,1)^d``: an arbitrary tree sequence of bivariate pair copulas.

    Construct with :class:`RVineCopulaEstimator` (Dißmann selection), which picks the tree structure AND a
    pair-copula family per edge from data. ``trees`` is the fitted structure (a list of trees, each a list of
    :class:`_Edge`); ``dim`` the number of variables.
    """

    def __init__(
        self,
        dim: int,
        trees: list[list[_Edge]],
        candidates: tuple[str, ...] = _DEFAULT_CANDIDATES,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        if int(dim) < 2:
            raise ValueError("RVineCopulaDistribution needs dim >= 2; got %d" % dim)
        self.dim = int(dim)
        self.trees = trees
        self.candidates = tuple(candidates)
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        fams = ",".join(e.copula.family for tree in self.trees for e in tree)
        return "RVineCopulaDistribution(dim=%d, [%s])" % (self.dim, fams)

    def log_density(self, u: np.ndarray) -> float:
        return float(self.seq_log_density(np.atleast_2d(np.asarray(u, dtype=np.float64)))[0])

    def seq_log_density(self, u: np.ndarray) -> np.ndarray:
        if not self.trees or not self.trees[0]:  # unfitted / independence
            return np.zeros(np.atleast_2d(u).shape[0])
        return _forward_vals(self.trees, u)[0]

    def sampler(self, seed: int | None = None) -> RVineCopulaSampler:
        return RVineCopulaSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> RVineCopulaEstimator:
        return RVineCopulaEstimator(self.dim, candidates=self.candidates, name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> UScoreEncoder:
        return UScoreEncoder()


class RVineCopulaSampler(DistributionSampler):
    """Sample by inverse Rosenblatt over the fitted structure (generic numerical conditional inversion)."""

    def __init__(self, dist: RVineCopulaDistribution, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = np.random.RandomState(seed)

    def sample(self, size: int | None = None, *, batched: bool = True) -> np.ndarray:
        # Generic (family-agnostic) inverse Rosenblatt: sample variables one at a time, inverting each new
        # variable's conditional CDF (built from the vine) against a uniform via bisection. O(d^2) h-evals
        # per accepted variable; correct for any pair-copula families.
        n = 1 if size is None else int(size)
        d = self.dist.dim
        w = self.rng.uniform(_CLIP, 1.0 - _CLIP, size=(n, d))
        x = np.empty((n, d))
        x[:, 0] = w[:, 0]
        for k in range(1, d):
            lo = np.full(n, _CLIP)
            hi = np.full(n, 1.0 - _CLIP)
            for _ in range(50):  # bisection on the conditional CDF F(x_k | x_0..x_{k-1})
                mid = 0.5 * (lo + hi)
                cand = x.copy()
                cand[:, k] = mid
                cdf = self._conditional_cdf(cand, k)
                under = cdf < w[:, k]
                lo = np.where(under, mid, lo)
                hi = np.where(under, hi, mid)
            x[:, k] = 0.5 * (lo + hi)
        out = _clip01(x)
        return out[0] if size is None else out

    def _conditional_cdf(self, u: np.ndarray, k: int) -> np.ndarray:
        """``F(u_k | u_0,...,u_{k-1})`` from the vine's h-functions -- the Rosenblatt transform's k-th coord.

        Recomputes the vine forward pass and reads off the conditional CDF of variable ``k`` given the
        earlier variables by chaining the h-functions of the edges that touch ``k`` in successive trees.
        """
        # forward pass to get every edge's conditional CDFs at u
        _, all_vals = _forward_vals(self.dist.trees, _clip01(u))
        # F(u_k | earlier) is the tree-t conditional CDF of k where its conditioning set is exactly {0..k-1}
        # restricted to the vine; walk down trees taking the deepest edge whose conditioning ⊆ {0..k-1}.
        earlier = set(range(k))
        best = _clip01(u)[:, k]  # tree-0 fallback: F(u_k) = u_k (marginal is uniform)
        for t, tree in enumerate(self.dist.trees):
            for idx, e in enumerate(tree):
                if k in (e.a, e.b) and e.cond <= earlier and (e.constraint() - {k}) <= earlier:
                    best = all_vals[t][idx][k]
        return best


class RVineCopulaEstimator(ParameterEstimator):
    """Dißmann selection + sequential MLE: choose the tree structure and per-edge family from data."""

    def __init__(
        self,
        dim: int,
        candidates: tuple[str, ...] = _DEFAULT_CANDIDATES,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        self.dim = dim
        self.candidates = tuple(candidates)
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> BufferedUScoreAccumulatorFactory:
        return BufferedUScoreAccumulatorFactory(self.dim, keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat: tuple[np.ndarray, np.ndarray]) -> RVineCopulaDistribution:
        u, w = suff_stat
        if len(u) < 2:
            return RVineCopulaDistribution(self.dim, [], candidates=self.candidates, name=self.name, keys=self.keys)
        trees = _select_and_fit(_clip01(u), np.asarray(w, dtype=np.float64), self.candidates)
        return RVineCopulaDistribution(self.dim, trees, candidates=self.candidates, name=self.name, keys=self.keys)
