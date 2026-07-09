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
from typing import Any

import numpy as np

Edge = tuple[int, int]


@dataclass
class ConditionalIndependenceResult:
    """Result from a conditional independence calculation."""

    measure: float
    statistic: float
    p_value: float | None
    independent: bool


@dataclass
class CausalSkeleton:
    """Undirected skeleton plus separating sets from a PC-style search."""

    edges: set[Edge]
    separating_sets: dict[Edge, frozenset[int]]
    variable_names: list[Any]

    def has_edge(self, i: int, j: int) -> bool:
        """Return whether the undirected skeleton contains edge ``i``--``j``."""
        return _edge(i, j) in self.edges


@dataclass
class PartiallyDirectedGraph:
    """Partially directed graph after collider orientation."""

    directed_edges: set[Edge]
    undirected_edges: set[Edge]
    variable_names: list[Any]


def gaussian_partial_correlation(data: Any, x: int, y: int, given: Sequence[int] = (), ridge: float = 1.0e-10) -> float:
    """Return partial correlation rho_xy.given for continuous data."""
    arr = _as_2d_data(data)
    x_vec = arr[:, int(x)]
    y_vec = arr[:, int(y)]
    given = tuple(int(g) for g in given)
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
    rho = gaussian_partial_correlation(arr, x, y, given=given, ridge=ridge)
    rho = float(np.clip(rho, -0.999999, 0.999999))
    dof = max(arr.shape[0] - len(tuple(given)) - 3, 1)
    statistic = 0.5 * math.log((1.0 + rho) / (1.0 - rho)) * math.sqrt(dof)
    p_value = math.erfc(abs(statistic) / math.sqrt(2.0))
    return ConditionalIndependenceResult(
        measure=rho, statistic=float(statistic), p_value=float(p_value), independent=bool(p_value > alpha)
    )


def discrete_conditional_mutual_information(data: Any, x: int, y: int, given: Sequence[int] = ()) -> float:
    """Estimate I(X;Y | Z) from categorical samples using empirical counts."""
    arr = np.asarray(data)
    if arr.ndim != 2:
        raise ValueError("data must be a two-dimensional array.")
    x = int(x)
    y = int(y)
    given = tuple(int(g) for g in given)
    n = arr.shape[0]
    if n == 0:
        raise ValueError("data must contain at least one row.")
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


def learn_pc_skeleton(
    data: Any,
    variable_names: Sequence[Any] | None = None,
    alpha: float = 0.05,
    max_cond_set: int = 2,
    method: str = "gaussian",
) -> CausalSkeleton:
    """Learn a PC-style undirected skeleton from conditional independences."""
    arr = _as_2d_data(data) if method == "gaussian" else np.asarray(data)
    if arr.ndim != 2:
        raise ValueError("data must be a two-dimensional array.")
    p = arr.shape[1]
    names = list(range(p)) if variable_names is None else list(variable_names)
    if len(names) != p:
        raise ValueError("variable_names length must match data columns.")
    edges: set[Edge] = {_edge(i, j) for i in range(p) for j in range(i + 1, p)}
    sepsets: dict[Edge, frozenset[int]] = {}
    for cond_size in range(max(0, int(max_cond_set)) + 1):
        for i, j in list(edges):
            candidates = [v for v in range(p) if v != i and v != j]
            for given in itertools.combinations(candidates, cond_size):
                if _is_independent(arr, i, j, given, alpha, method):
                    e = _edge(i, j)
                    edges.discard(e)
                    sepsets[e] = frozenset(given)
                    break
    return CausalSkeleton(edges, sepsets, names)


def orient_v_structures(skeleton: CausalSkeleton) -> PartiallyDirectedGraph:
    """Orient unshielded colliders i -> k <- j using separating sets."""
    directed: set[Edge] = set()
    p = len(skeleton.variable_names)
    for i, j in itertools.combinations(range(p), 2):
        if skeleton.has_edge(i, j):
            continue
        sep = skeleton.separating_sets.get(_edge(i, j), frozenset())
        for k in range(p):
            if k == i or k == j:
                continue
            if skeleton.has_edge(i, k) and skeleton.has_edge(j, k) and k not in sep:
                directed.add((i, k))
                directed.add((j, k))
    undirected = set()
    for i, j in skeleton.edges:
        if (i, j) not in directed and (j, i) not in directed:
            undirected.add((i, j))
    return PartiallyDirectedGraph(directed, undirected, skeleton.variable_names)


def _is_independent(data: np.ndarray, i: int, j: int, given: Sequence[int], alpha: float, method: str) -> bool:
    if method == "gaussian":
        return gaussian_conditional_independence(data, i, j, given=given, alpha=alpha).independent
    if method == "discrete":
        return discrete_conditional_mutual_information(data, i, j, given=given) <= alpha
    raise ValueError("method must be 'gaussian' or 'discrete'.")


def _as_2d_data(data: Any) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("data must be a two-dimensional array.")
    if arr.shape[0] == 0:
        raise ValueError("data must contain at least one row.")
    if np.any(~np.isfinite(arr)):
        raise ValueError("data must be finite.")
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
        return 0.0
    return float(np.dot(xx, yy) / denom)


def _edge(i: int, j: int) -> Edge:
    return (int(i), int(j)) if i < j else (int(j), int(i))
