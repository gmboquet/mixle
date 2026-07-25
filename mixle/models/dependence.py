"""Conditional-dependence and small causal-structure learning utilities.

The module provides Gaussian and discrete conditional-independence measures plus
lightweight PC-style skeleton and collider-orientation helpers for exploratory
causal structure analysis.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

Edge = tuple[int, int]


@dataclass
class ConditionalIndependenceResult:
    """Result from a conditional independence calculation."""

    measure: float | None
    statistic: float | None
    p_value: float | None
    independent: bool | None
    status: Literal["independent", "dependent", "inconclusive"]
    sample_size: int
    degrees_of_freedom: int | None
    method: str
    reason: str | None = None


@dataclass
class CausalSkeleton:
    """Undirected skeleton plus separating sets from a PC-style search."""

    edges: set[Edge]
    separating_sets: dict[Edge, frozenset[int]]
    variable_names: list[Any]
    all_separating_sets: dict[Edge, tuple[frozenset[int], ...]] | None = None

    def has_edge(self, i: int, j: int) -> bool:
        """Return whether the undirected skeleton contains edge ``i``--``j``."""
        return _edge(i, j) in self.edges


@dataclass
class PartiallyDirectedGraph:
    """Partially directed graph after collider orientation."""

    directed_edges: set[Edge]
    undirected_edges: set[Edge]
    variable_names: list[Any]


def gaussian_partial_correlation(
    data: Any,
    x: int,
    y: int,
    given: Sequence[int] = (),
    ridge: float = 1.0e-10,
) -> float:
    """Return partial correlation rho_xy.given for continuous data."""
    arr = _as_2d_data(data)
    x, y, given = _validate_indices(arr.shape[1], x, y, given)
    ridge = _positive_finite(ridge, "ridge")
    x_vec = arr[:, x]
    y_vec = arr[:, y]
    if len(given) == 0:
        return _corr(x_vec, y_vec)
    z = arr[:, given]
    x_res = _residualize(x_vec, z, ridge)
    y_res = _residualize(y_vec, z, ridge)
    return _corr(x_res, y_res)


def gaussian_conditional_independence(
    data: Any, x: int, y: int, given: Sequence[int] = (), alpha: float = 0.05, ridge: float = 1.0e-10
) -> ConditionalIndependenceResult:
    """Fisher-z Gaussian conditional independence test."""
    arr = _as_2d_data(data)
    x, y, given = _validate_indices(arr.shape[1], x, y, given)
    alpha = _probability(alpha, "alpha")
    ridge = _positive_finite(ridge, "ridge")
    dof = arr.shape[0] - len(given) - 3
    if dof <= 0:
        return _inconclusive(
            sample_size=len(arr),
            degrees_of_freedom=dof,
            method="gaussian-fisher-z",
            reason=(
                f"Fisher-z requires n - |given| - 3 > 0; got n={len(arr)}, "
                f"|given|={len(given)}, degrees_of_freedom={dof}"
            ),
        )
    try:
        rho = gaussian_partial_correlation(arr, x, y, given=given, ridge=ridge)
    except (ValueError, np.linalg.LinAlgError) as exc:
        return _inconclusive(
            sample_size=len(arr),
            degrees_of_freedom=dof,
            method="gaussian-fisher-z",
            reason=f"partial correlation is undefined: {exc}",
        )
    rho = float(np.clip(rho, -0.999999, 0.999999))
    statistic = 0.5 * math.log((1.0 + rho) / (1.0 - rho)) * math.sqrt(dof)
    p_value = math.erfc(abs(statistic) / math.sqrt(2.0))
    independent = bool(p_value > alpha)
    return ConditionalIndependenceResult(
        measure=rho,
        statistic=float(statistic),
        p_value=float(p_value),
        independent=independent,
        status="independent" if independent else "dependent",
        sample_size=len(arr),
        degrees_of_freedom=dof,
        method="gaussian-fisher-z",
    )


def discrete_conditional_mutual_information(data: Any, x: int, y: int, given: Sequence[int] = ()) -> float:
    """Estimate I(X;Y | Z) from categorical samples using empirical counts."""
    arr = _as_discrete_data(data)
    x, y, given = _validate_indices(arr.shape[1], x, y, given)
    n = arr.shape[0]
    xyz: dict[tuple[Any, Any, tuple[Any, ...]], int] = {}
    xz: dict[tuple[Any, tuple[Any, ...]], int] = {}
    yz: dict[tuple[Any, tuple[Any, ...]], int] = {}
    zc: dict[tuple[Any, ...], int] = {}
    for row in arr:
        z = tuple(row[g] for g in given)
        xv = row[x]
        yv = row[y]
        xyz[(xv, yv, z)] = xyz.get((xv, yv, z), 0) + 1
        xz[(xv, z)] = xz.get((xv, z), 0) + 1
        yz[(yv, z)] = yz.get((yv, z), 0) + 1
        zc[z] = zc.get(z, 0) + 1
    cmi = 0.0
    for (xv, yv, z), c_xyz in xyz.items():
        cmi += (c_xyz / n) * math.log((c_xyz * zc[z]) / (xz[(xv, z)] * yz[(yv, z)]))
    return float(max(0.0, cmi))


def discrete_conditional_independence(
    data: Any,
    x: int,
    y: int,
    given: Sequence[int] = (),
    alpha: float = 0.05,
) -> ConditionalIndependenceResult:
    """Likelihood-ratio (G-test) conditional-independence test for categorical samples.

    The statistic is ``2 n I(X;Y|Z)`` and is compared with its chi-square null distribution. The
    asymptotic result is reported as inconclusive unless there are at least five observations per
    degree of freedom; callers that need sparse-table exactness should use a permutation test.
    """
    arr = _as_discrete_data(data)
    x, y, given = _validate_indices(arr.shape[1], x, y, given)
    alpha = _probability(alpha, "alpha")
    cmi = discrete_conditional_mutual_information(arr, x, y, given)
    degrees_of_freedom = _discrete_degrees_of_freedom(arr, x, y, given)
    if degrees_of_freedom <= 0:
        return _inconclusive(
            sample_size=len(arr),
            degrees_of_freedom=degrees_of_freedom,
            method="discrete-g-test",
            reason="the observed contingency tables have zero degrees of freedom",
            measure=cmi,
        )
    if len(arr) < 5 * degrees_of_freedom:
        return _inconclusive(
            sample_size=len(arr),
            degrees_of_freedom=degrees_of_freedom,
            method="discrete-g-test",
            reason=(
                "chi-square calibration requires at least five observations per degree of freedom; "
                f"got n={len(arr)}, degrees_of_freedom={degrees_of_freedom}"
            ),
            measure=cmi,
        )

    from scipy.stats import chi2

    statistic = 2.0 * len(arr) * cmi
    p_value = float(chi2.sf(statistic, degrees_of_freedom))
    if not np.isfinite(p_value):
        return _inconclusive(
            sample_size=len(arr),
            degrees_of_freedom=degrees_of_freedom,
            method="discrete-g-test",
            reason="chi-square survival function returned a non-finite p-value",
            measure=cmi,
        )
    independent = bool(p_value > alpha)
    return ConditionalIndependenceResult(
        measure=cmi,
        statistic=statistic,
        p_value=p_value,
        independent=independent,
        status="independent" if independent else "dependent",
        sample_size=len(arr),
        degrees_of_freedom=degrees_of_freedom,
        method="discrete-g-test",
    )


def learn_pc_skeleton(
    data: Any,
    variable_names: Sequence[Any] | None = None,
    alpha: float = 0.05,
    max_cond_set: int = 2,
    method: str = "gaussian",
) -> CausalSkeleton:
    """Learn a PC-style undirected skeleton from conditional independences."""
    if method not in {"gaussian", "discrete"}:
        raise ValueError("method must be 'gaussian' or 'discrete'.")
    arr = _as_2d_data(data) if method == "gaussian" else _as_discrete_data(data)
    alpha = _probability(alpha, "alpha")
    max_cond_set = _nonnegative_int(max_cond_set, "max_cond_set")
    p = arr.shape[1]
    names = list(range(p)) if variable_names is None else list(variable_names)
    if len(names) != p:
        raise ValueError("variable_names length must match data columns.")
    if len(set(names)) != len(names):
        raise ValueError("variable_names must be unique.")
    edges: set[Edge] = {_edge(i, j) for i in range(p) for j in range(i + 1, p)}
    sepsets: dict[Edge, frozenset[int]] = {}
    all_sepsets: dict[Edge, tuple[frozenset[int], ...]] = {}
    for cond_size in range(min(max_cond_set, max(p - 2, 0)) + 1):
        level_edges = tuple(sorted(edges))
        adjacency = {node: set() for node in range(p)}
        for i, j in level_edges:
            adjacency[i].add(j)
            adjacency[j].add(i)
        removals: dict[Edge, tuple[frozenset[int], ...]] = {}
        for i, j in level_edges:
            candidate_pools = (
                tuple(sorted(adjacency[i] - {j})),
                tuple(sorted(adjacency[j] - {i})),
            )
            candidate_sets = sorted(
                {
                    tuple(given)
                    for pool in candidate_pools
                    if len(pool) >= cond_size
                    for given in itertools.combinations(pool, cond_size)
                }
            )
            separating = []
            for given in candidate_sets:
                result = _conditional_independence(arr, i, j, given, alpha, method)
                if result.status == "independent":
                    separating.append(frozenset(given))
            if separating:
                removals[(i, j)] = tuple(separating)
        for edge in sorted(removals):
            edges.discard(edge)
            all_sepsets[edge] = removals[edge]
            sepsets[edge] = removals[edge][0]
    return CausalSkeleton(edges, sepsets, names, all_sepsets)


def orient_v_structures(skeleton: CausalSkeleton) -> PartiallyDirectedGraph:
    """Build a deterministic, conflict-checked CPDAG from skeleton and separating evidence."""
    _validate_skeleton(skeleton)
    p = len(skeleton.variable_names)
    directed: set[Edge] = set()
    undirected = set(skeleton.edges)
    proposals: set[Edge] = set()
    for i, j in itertools.combinations(range(p), 2):
        if skeleton.has_edge(i, j):
            continue
        edge = _edge(i, j)
        all_separators = (
            skeleton.all_separating_sets.get(edge, ())
            if skeleton.all_separating_sets is not None
            else ((skeleton.separating_sets[edge],) if edge in skeleton.separating_sets else ())
        )
        if not all_separators:
            continue
        for k in range(p):
            if k == i or k == j:
                continue
            if (
                skeleton.has_edge(i, k)
                and skeleton.has_edge(j, k)
                and all(k not in separator for separator in all_separators)
            ):
                proposals.add((i, k))
                proposals.add((j, k))
    _apply_orientations(directed, undirected, proposals)
    _apply_meek_rules(directed, undirected, p)
    _ensure_acyclic(directed, p)
    return PartiallyDirectedGraph(directed, undirected, skeleton.variable_names)


def _conditional_independence(
    data: np.ndarray,
    i: int,
    j: int,
    given: Sequence[int],
    alpha: float,
    method: str,
) -> ConditionalIndependenceResult:
    if method == "gaussian":
        return gaussian_conditional_independence(data, i, j, given=given, alpha=alpha)
    if method == "discrete":
        return discrete_conditional_independence(data, i, j, given=given, alpha=alpha)
    raise ValueError("method must be 'gaussian' or 'discrete'.")


def _as_2d_data(data: Any) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("data must be a two-dimensional array.")
    if arr.shape[0] == 0:
        raise ValueError("data must contain at least one row.")
    if arr.shape[1] == 0:
        raise ValueError("data must contain at least one column.")
    if np.any(~np.isfinite(arr)):
        raise ValueError("data must be finite.")
    return arr


def _as_discrete_data(data: Any) -> np.ndarray:
    arr = np.asarray(data)
    if arr.ndim != 2:
        raise ValueError("data must be a two-dimensional array.")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError("data must contain at least one row and column.")
    for row in arr:
        for value in row:
            if value is None:
                raise ValueError("discrete data must not contain missing values.")
            if isinstance(value, (float, np.floating, complex, np.complexfloating)) and not np.isfinite(value):
                raise ValueError("discrete data must not contain non-finite values.")
            try:
                hash(value)
            except TypeError as exc:
                raise ValueError("every discrete value must be hashable.") from exc
    return arr


def _residualize(y: np.ndarray, z: np.ndarray, ridge: float) -> np.ndarray:
    yy = y - y.mean()
    zz = z - z.mean(axis=0, keepdims=True)
    gram = zz.T.dot(zz) + float(ridge) * np.eye(zz.shape[1])
    coef = np.linalg.solve(gram, zz.T.dot(yy))
    return yy - zz.dot(coef)


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    xx = x - x.mean()
    yy = y - y.mean()
    denom = math.sqrt(float(np.dot(xx, xx) * np.dot(yy, yy)))
    if denom <= 0.0:
        raise ValueError("correlation is undefined for a constant residual")
    return float(np.dot(xx, yy) / denom)


def _edge(i: int, j: int) -> Edge:
    return (int(i), int(j)) if i < j else (int(j), int(i))


def _index(value: Any, name: str, width: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer column index")
    result = int(value)
    if not 0 <= result < width:
        raise ValueError(f"{name}={result} is outside [0, {width})")
    return result


def _validate_indices(width: int, x: Any, y: Any, given: Sequence[int]) -> tuple[int, int, tuple[int, ...]]:
    x = _index(x, "x", width)
    y = _index(y, "y", width)
    if x == y:
        raise ValueError("x and y must identify different columns")
    if isinstance(given, (str, bytes)) or not isinstance(given, Sequence):
        raise ValueError("given must be a sequence of column indices")
    result = tuple(_index(value, f"given[{index}]", width) for index, value in enumerate(given))
    if len(set(result)) != len(result):
        raise ValueError("given must not contain repeated column indices")
    if x in result or y in result:
        raise ValueError("given must not overlap x or y")
    return x, y, result


def _positive_finite(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a strictly positive finite scalar")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a strictly positive finite scalar") from exc
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a strictly positive finite scalar")
    return result


def _probability(value: Any, name: str) -> float:
    result = _positive_finite(value, name)
    if result >= 1.0:
        raise ValueError(f"{name} must lie strictly between 0 and 1")
    return result


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def _inconclusive(
    *,
    sample_size: int,
    degrees_of_freedom: int | None,
    method: str,
    reason: str,
    measure: float | None = None,
) -> ConditionalIndependenceResult:
    return ConditionalIndependenceResult(
        measure=measure,
        statistic=None,
        p_value=None,
        independent=None,
        status="inconclusive",
        sample_size=sample_size,
        degrees_of_freedom=degrees_of_freedom,
        method=method,
        reason=reason,
    )


def _discrete_degrees_of_freedom(
    data: np.ndarray,
    x: int,
    y: int,
    given: Sequence[int],
) -> int:
    strata: dict[tuple[Any, ...], tuple[set[Any], set[Any]]] = {}
    for row in data:
        key = tuple(row[index] for index in given)
        x_values, y_values = strata.setdefault(key, (set(), set()))
        x_values.add(row[x])
        y_values.add(row[y])
    return int(sum((len(x_values) - 1) * (len(y_values) - 1) for x_values, y_values in strata.values()))


def _validate_skeleton(skeleton: CausalSkeleton) -> None:
    if not isinstance(skeleton, CausalSkeleton):
        raise TypeError("skeleton must be a CausalSkeleton")
    p = len(skeleton.variable_names)
    if len(set(skeleton.variable_names)) != p:
        raise ValueError("skeleton variable_names must be unique")
    for edge in skeleton.edges:
        if (
            not isinstance(edge, tuple)
            or len(edge) != 2
            or edge != _edge(*edge)
            or edge[0] == edge[1]
            or not 0 <= edge[0] < p
            or not 0 <= edge[1] < p
        ):
            raise ValueError(f"invalid canonical skeleton edge {edge!r}")


def _apply_orientations(directed: set[Edge], undirected: set[Edge], proposals: set[Edge]) -> None:
    for source, target in sorted(proposals):
        edge = _edge(source, target)
        if (target, source) in proposals or (target, source) in directed:
            raise ValueError(f"conflicting orientations proposed for edge {edge}")
        if edge not in undirected:
            if (source, target) not in directed:
                raise ValueError(f"cannot orient absent edge {edge}")
            continue
        undirected.remove(edge)
        directed.add((source, target))


def _adjacent(a: int, b: int, directed: set[Edge], undirected: set[Edge]) -> bool:
    return _edge(a, b) in undirected or (a, b) in directed or (b, a) in directed


def _apply_meek_rules(directed: set[Edge], undirected: set[Edge], p: int) -> None:
    while True:
        proposals: set[Edge] = set()
        for a, b in sorted(undirected):
            for source, target in ((a, b), (b, a)):
                # R1: c -> source - target, with c and target non-adjacent.
                if any(
                    (c, source) in directed and not _adjacent(c, target, directed, undirected)
                    for c in range(p)
                    if c not in {source, target}
                ):
                    proposals.add((source, target))
                # R2: source - target and source -> c -> target.
                if any(
                    (source, c) in directed and (c, target) in directed
                    for c in range(p)
                    if c not in {source, target}
                ):
                    proposals.add((source, target))
                # R3: source-c and source-d undirected, c -> target, d -> target, c/d non-adjacent.
                parents = [
                    c
                    for c in range(p)
                    if c not in {source, target}
                    and _edge(source, c) in undirected
                    and (c, target) in directed
                ]
                if any(
                    not _adjacent(c, d, directed, undirected)
                    for c, d in itertools.combinations(parents, 2)
                ):
                    proposals.add((source, target))
        if not proposals:
            return
        _apply_orientations(directed, undirected, proposals)


def _ensure_acyclic(directed: set[Edge], p: int) -> None:
    children = {node: [] for node in range(p)}
    indegree = [0] * p
    for source, target in directed:
        children[source].append(target)
        indegree[target] += 1
    queue = [node for node in range(p) if indegree[node] == 0]
    visited = 0
    while queue:
        node = queue.pop()
        visited += 1
        for child in children[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if visited != p:
        raise ValueError("oriented graph contains a directed cycle")
