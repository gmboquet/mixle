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
    reject_unsupported_pseudo_count,
    u_score_batch,
    validated_buffered_statistic,
    validated_dimension,
    validated_sample_size,
    weighted_kendall_tau,
)
from mixle.stats.multivariate.vine_copula import (
    _CLIP,
    _DEFAULT_CANDIDATES,
    _fit_best_pair,
    _validated_candidates,
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

    Raises if the allowed graph is disconnected rather than returning a forest.
    """
    n = validated_dimension(n, minimum=1, label="spanning-tree node count")
    canonical_allowed: set[tuple[int, int]] = set()
    for edge in allowed:
        if not isinstance(edge, tuple) or len(edge) != 2:
            raise ValueError("allowed spanning-tree edges must be (i, j) pairs")
        i, j = edge
        if not isinstance(i, int) or isinstance(i, bool) or not isinstance(j, int) or isinstance(j, bool):
            raise TypeError("spanning-tree node IDs must be integers")
        if not 0 <= i < n or not 0 <= j < n or i == j:
            raise ValueError("allowed spanning-tree edges must join distinct nodes in range(0, n)")
        canonical_allowed.add((min(i, j), max(i, j)))
    missing_weights = canonical_allowed - set(weight)
    if missing_weights:
        raise ValueError("spanning-tree weights are missing allowed edge(s): %s" % sorted(missing_weights))
    in_tree = {0}
    chosen: list[tuple[int, int]] = []
    while len(in_tree) < n:
        best, best_w = None, -np.inf
        for i, j in canonical_allowed:
            a, b = (i, j) if i < j else (j, i)
            if (a in in_tree) != (b in in_tree):  # exactly one endpoint inside -> a growing edge
                if weight[(a, b)] > best_w:
                    best, best_w = (a, b), weight[(a, b)]
        if best is None:
            missing = sorted(set(range(n)) - in_tree)
            raise ValueError(
                "allowed spanning-tree graph is disconnected; cannot reach node(s) %s from node 0" % missing
            )
        chosen.append(best)
        in_tree.add(best[0] if best[1] in in_tree else best[1])
    return chosen


def _forward_vals(trees: list[list[_Edge]], u: np.ndarray) -> tuple[np.ndarray, list[list[dict[int, np.ndarray]]]]:
    """Forward pass: accumulate the log-density and the per-edge conditional CDFs ``val[edge][var]``."""
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
    candidates = _validated_candidates(candidates)
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
        pc = _fit_best_pair(
            u[:, i],
            u[:, j],
            w,
            candidates,
            edge_context="R-vine tree 1 variables (%d, %d)" % (i, j),
        )
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
            pc = _fit_best_pair(
                ia,
                ib,
                w,
                candidates,
                edge_context="R-vine tree %d variables (%d, %d) conditioned on %s" % (t + 1, a, b, sorted(shared)),
            )
            tree_t.append(_Edge(a, b, shared, pc, {a: (px, a), b: (py, b)}))
            vals_t.append({a: pc.h(ia, ib), b: pc.h(ib, ia)})
        trees.append(tree_t)
        prev_vals = vals_t
    return trees


def _independence_trees(dim: int) -> list[list[_Edge]]:
    """Construct a complete, valid C-vine-shaped independence tree sequence."""
    trees: list[list[_Edge]] = []
    for tree_index in range(dim - 1):
        cond = frozenset(range(tree_index))
        tree: list[_Edge] = []
        for variable in range(tree_index + 1, dim):
            parents = None
            if tree_index:
                parents = {
                    tree_index: (0, tree_index),
                    variable: (variable - tree_index, variable),
                }
            tree.append(
                _Edge(
                    tree_index,
                    variable,
                    cond,
                    _independence_pair(),
                    parents,
                )
            )
        trees.append(tree)
    return trees


def _independence_pair() -> Any:
    """Delay the pair-class import cycle through the already imported family table."""
    from mixle.stats.multivariate.vine_copula import IndependencePairCopula

    return IndependencePairCopula()


def _tree_is_connected(node_count: int, edges: list[tuple[int, int]]) -> bool:
    if node_count == 1:
        return not edges
    adjacency = {node: set() for node in range(node_count)}
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    reached = {0}
    frontier = [0]
    while frontier:
        node = frontier.pop()
        for neighbor in adjacency[node] - reached:
            reached.add(neighbor)
            frontier.append(neighbor)
    return len(reached) == node_count


def _conditional_lookup(
    trees: list[list[_Edge]],
) -> dict[tuple[int, frozenset[int]], tuple[_Edge, int]]:
    lookup: dict[tuple[int, frozenset[int]], tuple[_Edge, int]] = {}
    for tree in trees:
        for edge in tree:
            for variable, other in ((edge.a, edge.b), (edge.b, edge.a)):
                key = (variable, edge.cond | {other})
                if key in lookup:
                    raise ValueError(
                        "R-vine contains duplicate conditional representation for variable %d given %s"
                        % (variable, sorted(key[1]))
                    )
                lookup[key] = (edge, other)
    return lookup


def _find_sampling_order(
    dim: int,
    lookup: dict[tuple[int, frozenset[int]], tuple[_Edge, int]],
) -> tuple[int, ...]:
    """Find a Rosenblatt order whose required conditional scores exist in the vine."""

    def search(prefix: tuple[int, ...]) -> tuple[int, ...] | None:
        if len(prefix) == dim:
            return prefix
        conditioning = frozenset(prefix)
        for variable in range(dim):
            if variable in conditioning:
                continue
            if conditioning and (variable, conditioning) not in lookup:
                continue
            answer = search(prefix + (variable,))
            if answer is not None:
                return answer
        return None

    result = search(())
    if result is None:
        raise ValueError("R-vine tree sequence has no complete Rosenblatt sampling order")
    return result


def _validated_trees(dim: int, trees: Any) -> tuple[list[list[_Edge]], tuple[int, ...]]:
    """Validate cardinality, IDs, topology, proximity, parents, and sampling completeness."""
    if not isinstance(trees, (list, tuple)):
        raise TypeError("R-vine trees must be a sequence of tree edge sequences")
    if len(trees) != dim - 1:
        raise ValueError("R-vine must contain exactly %d trees" % (dim - 1))

    checked: list[list[_Edge]] = []
    signatures: set[tuple[frozenset[int], frozenset[int]]] = set()
    for tree_index, raw_tree in enumerate(trees):
        if not isinstance(raw_tree, (list, tuple)):
            raise TypeError("R-vine tree %d must be an edge sequence" % (tree_index + 1))
        expected_edges = dim - tree_index - 1
        if len(raw_tree) != expected_edges:
            raise ValueError("R-vine tree %d must contain exactly %d edges" % (tree_index + 1, expected_edges))
        tree: list[_Edge] = []
        topology_edges: list[tuple[int, int]] = []
        for edge_index, raw_edge in enumerate(raw_tree):
            if not isinstance(raw_edge, _Edge):
                raise TypeError("R-vine tree %d edge %d must be an R-vine edge" % (tree_index + 1, edge_index))
            if (
                not isinstance(raw_edge.a, int)
                or isinstance(raw_edge.a, bool)
                or not isinstance(raw_edge.b, int)
                or isinstance(raw_edge.b, bool)
                or raw_edge.a == raw_edge.b
                or not 0 <= raw_edge.a < dim
                or not 0 <= raw_edge.b < dim
            ):
                raise ValueError("R-vine conditioned variable IDs must be distinct integers in range(0, dim)")
            try:
                cond = frozenset(raw_edge.cond)
            except TypeError as exc:
                raise TypeError("R-vine edge conditioning set must be iterable") from exc
            if (
                len(cond) != tree_index
                or raw_edge.a in cond
                or raw_edge.b in cond
                or any(not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < dim for value in cond)
            ):
                raise ValueError(
                    "R-vine tree %d edge conditioning set must contain exactly %d other valid variables"
                    % (tree_index + 1, tree_index)
                )
            copula = raw_edge.copula
            if (
                not isinstance(getattr(copula, "family", None), str)
                or not callable(getattr(copula, "logpdf", None))
                or not callable(getattr(copula, "h", None))
                or not callable(getattr(copula, "h_inv", None))
            ):
                raise TypeError("R-vine edge copula is not pair-copula compatible")
            signature = (frozenset((raw_edge.a, raw_edge.b)), cond)
            if signature in signatures:
                raise ValueError("R-vine contains a duplicate conditioned edge")
            signatures.add(signature)

            parents = raw_edge.parents
            if tree_index == 0:
                if parents is not None:
                    raise ValueError("R-vine tree 1 edges must not have parent references")
                topology_edges.append((raw_edge.a, raw_edge.b))
            else:
                if not isinstance(parents, dict) or set(parents) != {raw_edge.a, raw_edge.b}:
                    raise ValueError("R-vine deeper-tree parents must map both conditioned variables")
                parent_nodes: list[int] = []
                normalized_parents: dict[int, tuple[int, int]] = {}
                previous = checked[tree_index - 1]
                for variable in (raw_edge.a, raw_edge.b):
                    reference = parents[variable]
                    if (
                        not isinstance(reference, tuple)
                        or len(reference) != 2
                        or not isinstance(reference[0], int)
                        or isinstance(reference[0], bool)
                        or reference[1] != variable
                        or not 0 <= reference[0] < len(previous)
                    ):
                        raise ValueError("R-vine parent references must be valid (edge_index, variable) pairs")
                    parent = previous[reference[0]]
                    if parent.constraint() != cond | {variable} or variable not in (parent.a, parent.b):
                        raise ValueError("R-vine parent reference violates the proximity/conditioning contract")
                    parent_nodes.append(reference[0])
                    normalized_parents[variable] = reference
                if parent_nodes[0] == parent_nodes[1]:
                    raise ValueError("R-vine edge must join two distinct previous-tree nodes")
                topology_edges.append((parent_nodes[0], parent_nodes[1]))
                parents = normalized_parents
            tree.append(_Edge(raw_edge.a, raw_edge.b, cond, copula, parents))

        node_count = dim if tree_index == 0 else len(checked[tree_index - 1])
        if not _tree_is_connected(node_count, topology_edges):
            raise ValueError("R-vine tree %d is cyclic or disconnected" % (tree_index + 1))
        checked.append(tree)

    lookup = _conditional_lookup(checked)
    return checked, _find_sampling_order(dim, lookup)


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
        self.dim = validated_dimension(dim, label="R-vine copula dimension")
        self.trees, self._sampling_order = _validated_trees(self.dim, trees)
        self.candidates = _validated_candidates(candidates)
        self.name = name
        self.keys = keys

    @classmethod
    def independence(
        cls,
        dim: int,
        candidates: tuple[str, ...] = _DEFAULT_CANDIDATES,
        name: str | None = None,
        keys: str | None = None,
    ) -> RVineCopulaDistribution:
        """Construct an explicit, structurally complete independence R-vine."""
        checked_dim = validated_dimension(dim, label="R-vine copula dimension")
        return cls(
            checked_dim,
            _independence_trees(checked_dim),
            candidates=candidates,
            name=name,
            keys=keys,
        )

    def __str__(self) -> str:
        fams = ",".join(e.copula.family for tree in self.trees for e in tree)
        return "RVineCopulaDistribution(dim=%d, [%s])" % (self.dim, fams)

    def log_density(self, u: np.ndarray) -> float:
        return float(self.seq_log_density(np.atleast_2d(np.asarray(u, dtype=np.float64)))[0])

    def seq_log_density(self, u: np.ndarray) -> np.ndarray:
        u = u_score_batch(u, self.dim)
        return _forward_vals(self.trees, u)[0]

    def sampler(self, seed: int | None = None) -> RVineCopulaSampler:
        return RVineCopulaSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> RVineCopulaEstimator:
        reject_unsupported_pseudo_count(pseudo_count, family="R-vine copula")
        return RVineCopulaEstimator(self.dim, candidates=self.candidates, name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> UScoreEncoder:
        return UScoreEncoder(self.dim)


class RVineCopulaSampler(DistributionSampler):
    """Sample exactly by the inverse Rosenblatt transform encoded by the validated vine."""

    def __init__(self, dist: RVineCopulaDistribution, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = np.random.RandomState(seed)
        self._lookup = _conditional_lookup(dist.trees)

    def sample(self, size: int | None = None, *, batched: bool = True) -> np.ndarray:
        n = 1 if size is None else validated_sample_size(size)
        d = self.dist.dim
        innovations = self.rng.uniform(_CLIP, 1.0 - _CLIP, size=(n, d))
        out = np.full((n, d), np.nan, dtype=np.float64)
        forward_cache: dict[tuple[int, frozenset[int]], np.ndarray] = {}

        def forward_score(variable: int, conditioning: frozenset[int]) -> np.ndarray:
            key = (variable, conditioning)
            if key in forward_cache:
                return forward_cache[key]
            if not conditioning:
                result = out[:, variable]
                if np.any(~np.isfinite(result)):
                    raise RuntimeError("R-vine sampling requested an ungenerated conditioning variable")
            else:
                edge, other = self._lookup[key]
                own_input = forward_score(variable, edge.cond)
                other_input = forward_score(other, edge.cond)
                result = edge.copula.h(own_input, other_input)
            forward_cache[key] = result
            return result

        def invert_score(
            variable: int,
            conditioning: frozenset[int],
            target: np.ndarray,
        ) -> np.ndarray:
            if not conditioning:
                return target
            edge, other = self._lookup[(variable, conditioning)]
            other_input = forward_score(other, edge.cond)
            lower_target = edge.copula.h_inv(target, other_input)
            return invert_score(variable, edge.cond, lower_target)

        generated: list[int] = []
        for variable in self.dist._sampling_order:
            conditioning = frozenset(generated)
            out[:, variable] = invert_score(variable, conditioning, innovations[:, variable])
            forward_cache[(variable, frozenset())] = out[:, variable]
            generated.append(variable)
        return out[0] if size is None else out


class RVineCopulaEstimator(ParameterEstimator):
    """Dißmann selection + sequential MLE: choose the tree structure and per-edge family from data."""

    def __init__(
        self,
        dim: int,
        candidates: tuple[str, ...] = _DEFAULT_CANDIDATES,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        self.dim = validated_dimension(dim, label="R-vine copula dimension")
        self.candidates = _validated_candidates(candidates)
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> BufferedUScoreAccumulatorFactory:
        return BufferedUScoreAccumulatorFactory(self.dim, keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat: tuple[np.ndarray, np.ndarray]) -> RVineCopulaDistribution:
        u, w = validated_buffered_statistic(suff_stat, self.dim, minimum_rows=2, require_positive_weight=True)
        trees = _select_and_fit(u, w, self.candidates)
        return RVineCopulaDistribution(self.dim, trees, candidates=self.candidates, name=self.name, keys=self.keys)
