"""Unstructured finite elements: P1 (linear triangular) FEM for the Poisson equation.

Structured grids cannot conform to real geology -- faults, pinch-outs, irregular basin outlines, local
refinement near a well. Finite elements on an unstructured triangular mesh can. This is the canonical
piece: linear (P1) elements assembling the stiffness matrix from per-triangle contributions, solving
``-div(kappa grad u) = f`` with Dirichlet boundaries on an arbitrary triangulation (e.g. a Delaunay mesh
of scattered points). Part of the earth-science/multiphysics work (Phase 5).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = ["fem_poisson", "boundary_nodes"]


def boundary_nodes(triangles: np.ndarray) -> np.ndarray:
    """The boundary node indices of a triangulation -- the vertices of edges that belong to one triangle."""
    edges: dict[tuple[int, int], int] = {}
    for tri in triangles:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            e = (min(a, b), max(a, b))
            edges[e] = edges.get(e, 0) + 1
    bnd = {v for e, cnt in edges.items() if cnt == 1 for v in e}
    return np.array(sorted(bnd), dtype=int)


def fem_poisson(nodes, triangles, source, *, conductivity=1.0, dirichlet=None) -> np.ndarray:
    """Solve ``-div(kappa grad u) = source`` by P1 finite elements on a triangular mesh.

    Args:
        nodes: ``(N, 2)`` vertex coordinates.
        triangles: ``(M, 3)`` integer vertex indices per element.
        source: ``f`` -- scalar, per-node array, or a callable ``f(xy) -> value``.
        conductivity: ``kappa`` -- scalar or per-element.
        dirichlet: ``{node_index: value}`` boundary conditions; default pins every boundary node to 0.

    Returns:
        ``u`` of shape ``(N,)`` -- the FEM solution at the nodes.
    """
    nodes = np.asarray(nodes, dtype=float)
    tris = np.asarray(triangles, dtype=int)
    nn = len(nodes)
    kappa = np.full(len(tris), float(conductivity)) if np.isscalar(conductivity) else np.asarray(conductivity, float)
    if callable(source):
        fval = np.array([source(p) for p in nodes])
    elif np.isscalar(source):
        fval = np.full(nn, float(source))
    else:
        fval = np.asarray(source, dtype=float).ravel()

    rows, cols, vals = [], [], []
    f = np.zeros(nn)
    for e, tri in enumerate(tris):
        (x1, y1), (x2, y2), (x3, y3) = nodes[tri]
        area = 0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
        if abs(area) < 1e-14:
            continue
        b = np.array([y2 - y3, y3 - y1, y1 - y2])  # d(basis)/dx * 2A
        c = np.array([x3 - x2, x1 - x3, x2 - x1])  # d(basis)/dy * 2A
        ke = kappa[e] * (np.outer(b, b) + np.outer(c, c)) / (4.0 * abs(area))  # P1 element stiffness
        for i in range(3):
            f[tri[i]] += abs(area) / 3.0 * fval[tri].mean()  # lumped load
            for j in range(3):
                rows.append(tri[i])
                cols.append(tri[j])
                vals.append(ke[i, j])
    k = sp.csr_matrix((vals, (rows, cols)), shape=(nn, nn)).tolil()

    bc = (
        {int(v): 0.0 for v in boundary_nodes(tris)}
        if dirichlet is None
        else {int(k_): float(v) for k_, v in dirichlet.items()}
    )
    for node, val in bc.items():  # Dirichlet: identity row, fixed RHS
        k.rows[node] = [node]
        k.data[node] = [1.0]
        f[node] = val
    return spla.spsolve(k.tocsr(), f)
