"""Multiphysics forward solvers: nD steady diffusion/Poisson and coupled PDE systems.

The adjoint inverse stack (pde_solve.py) already assembles ``-div(kappa grad u)`` in *any* dimension, but
only 1-D/2-D forward problems were wired up. This adds (1) a clean nD steady-state solver -- 3-D works
out of the box -- and (2) a CoupledPDESystem that block-assembles several physical fields with node-local
coupling (thermo-elastic-style exchange, reaction-diffusion between species, ...) into one linear solve.
Part of the earth-science/multiphysics/UQ plan (Phase 5).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = ["solve_poisson", "CoupledPDESystem", "solve_elasticity"]


def _diffusion_blocks(shape, kappa, spacing):
    """The COO pieces of ``-div(kappa grad)`` with Dirichlet (identity) boundary rows, on a grid (numpy)."""
    from pysp.ppl.physics.pde_solve import _grid_faces

    g = _grid_faces(shape, spacing)
    n = g["n"]
    kappa = np.full(n, float(kappa)) if np.isscalar(kappa) else np.asarray(kappa, dtype=float).ravel()
    fa, fb, fw = g["face_a"], g["face_b"], g["face_w"]
    cond = 0.5 * (kappa[fa] + kappa[fb]) * fw  # face conductance
    bmask = g["boundary_mask"]
    a_int, b_int = ~bmask[fa], ~bmask[fb]
    diag = np.zeros(n)
    np.add.at(diag, fa[a_int], cond[a_int])
    np.add.at(diag, fb[b_int], cond[b_int])
    rows = np.concatenate([fa[a_int], fb[b_int], g["interior"], g["boundary"]])
    cols = np.concatenate([fb[a_int], fa[b_int], g["interior"], g["boundary"]])
    vals = np.concatenate([-cond[a_int], -cond[b_int], diag[g["interior"]], np.ones(len(g["boundary"]))])
    return rows, cols, vals, n, g["boundary"], g["interior"]


def solve_poisson(shape, source, conductivity=1.0, *, dirichlet=0.0, spacing=1.0) -> np.ndarray:
    """Solve the steady diffusion / Poisson equation ``-div(kappa grad u) = source`` on a structured grid.

    Works in any dimension (1-D, 2-D, 3-D, ...) -- ``shape`` sets it. Dirichlet boundary conditions: the
    boundary nodes are pinned to ``dirichlet`` (scalar or a per-node array). ``conductivity`` is a scalar
    or a per-node field (heterogeneous media). Returns ``u`` reshaped to ``shape``.

    Args:
        shape: grid shape, e.g. ``(nx, ny, nz)``.
        source: right-hand side ``f``, scalar or per-node array (length ``prod(shape)`` or ``shape``).
        conductivity: ``kappa``, scalar or per-node.
        dirichlet: boundary values, scalar or per-node.
        spacing: grid spacing, scalar or per-axis.
    """
    shape = tuple(int(s) for s in np.atleast_1d(shape))
    rows, cols, vals, n, bnd, _ = _diffusion_blocks(shape, conductivity, spacing)
    a = sp.csc_matrix((vals, (rows, cols)), shape=(n, n))
    b = (np.full(n, float(source)) if np.isscalar(source) else np.asarray(source, dtype=float).ravel()).copy()
    dvals = np.full(n, float(dirichlet)) if np.isscalar(dirichlet) else np.asarray(dirichlet, dtype=float).ravel()
    b[bnd] = dvals[bnd]  # boundary rows are identity, so b sets the Dirichlet values there
    return spla.spsolve(a, b).reshape(shape)


class CoupledPDESystem:
    """Several diffusion fields on one grid, coupled node-locally -- a forward multiphysics solver.

    Field ``i`` obeys ``-div(kappa_i grad u_i) + sum_j C[i,j] u_j = f_i`` on the interior (Dirichlet
    boundaries). The coupling matrix ``C`` (``K x K``) is a node-local exchange/reaction between fields:
    e.g. ``C = [[k, -k], [-k, k]]`` is symmetric exchange of strength ``k`` between two fields (heat <->
    deformation, two reacting species). ``C = 0`` decouples into independent solves. The whole block
    system is assembled sparse and solved once.

    Args:
        shape: grid shape (shared by all fields).
        conductivities: one ``kappa`` (scalar or per-node) per field.
        coupling: ``K x K`` node-local coupling matrix.
        spacing: grid spacing.
    """

    def __init__(self, shape, conductivities: Sequence[Any], coupling: np.ndarray, *, spacing=1.0):
        self.shape = tuple(int(s) for s in np.atleast_1d(shape))
        self.kappas = list(conductivities)
        self.k = len(self.kappas)
        self.coupling = np.asarray(coupling, dtype=float)
        if self.coupling.shape != (self.k, self.k):
            raise ValueError(f"coupling must be {self.k}x{self.k} for {self.k} fields.")
        self.spacing = spacing

    def solve(self, sources: Sequence[Any], dirichlet: Sequence[Any] | float = 0.0) -> list[np.ndarray]:
        """Solve for all fields. ``sources`` is one RHS per field; returns one ``shape``-array per field."""
        blocks = [_diffusion_blocks(self.shape, kap, self.spacing) for kap in self.kappas]
        n = blocks[0][3]
        interior = blocks[0][5]
        rhs = np.zeros(self.k * n)
        dir_list = [dirichlet] * self.k if np.isscalar(dirichlet) else list(dirichlet)
        b_rows, b_cols, b_vals = [], [], []  # the full block system in COO
        for i in range(self.k):
            rows, cols, vals, _, bnd, _ = blocks[i]
            b_rows.append(i * n + np.asarray(rows))
            b_cols.append(i * n + np.asarray(cols))
            b_vals.append(np.asarray(vals))
            src = sources[i]
            bi = (np.full(n, float(src)) if np.isscalar(src) else np.asarray(src, dtype=float).ravel()).copy()
            dv = dir_list[i]
            dvals = np.full(n, float(dv)) if np.isscalar(dv) else np.asarray(dv, dtype=float).ravel()
            bi[bnd] = dvals[bnd]
            rhs[i * n : (i + 1) * n] = bi
            for j in range(self.k):  # node-local coupling C[i,j] u_j on interior rows of field i
                if self.coupling[i, j] != 0.0:
                    b_rows.append(i * n + interior)
                    b_cols.append(j * n + interior)
                    b_vals.append(np.full(len(interior), self.coupling[i, j]))
        big = sp.csc_matrix(
            (np.concatenate(b_vals), (np.concatenate(b_rows), np.concatenate(b_cols))), shape=(self.k * n,) * 2
        )
        u = spla.spsolve(big, rhs)
        return [u[i * n : (i + 1) * n].reshape(self.shape) for i in range(self.k)]


def solve_elasticity(shape, body_force, *, lame_lambda=1.0, lame_mu=1.0, spacing=1.0, dirichlet=0.0):
    """Solve 2-D static linear elasticity (the Navier-Cauchy displacement equations) on a structured grid.

    ``mu lap(u) + (lambda + mu) grad(div u) + f = 0`` for the displacement field ``u = (ux, uy)``, with
    isotropic Lame parameters ``lame_lambda`` / ``lame_mu`` (stiffness) and Dirichlet (clamped) boundaries.
    The structural / geomechanics forward map behind seismic and stress modelling: given a stiffness model
    and loads, it returns the displacement, from which strain and stress follow.

    Args:
        shape: ``(nx, ny)`` grid.
        body_force: ``(nx, ny, 2)`` or ``(2,)`` body force ``(fx, fy)`` per node (scalar broadcasts).
        lame_lambda, lame_mu: isotropic Lame parameters (may be scalars; constant-coefficient form).
        spacing: grid spacing (scalar or ``(hx, hy)``).
        dirichlet: boundary displacement, ``(nx, ny, 2)``, ``(2,)`` or scalar.

    Returns:
        ``u`` of shape ``(nx, ny, 2)`` -- the displacement field.
    """
    nx, ny = (int(s) for s in shape)
    n = nx * ny
    hx, hy = np.broadcast_to(np.asarray(spacing, dtype=float), (2,))
    lam, mu = float(lame_lambda), float(lame_mu)
    idx = np.arange(n).reshape(nx, ny)
    on_bnd = np.zeros((nx, ny), dtype=bool)
    on_bnd[[0, -1], :] = True
    on_bnd[:, [0, -1]] = True
    rows, cols, vals = [], [], []

    def add(r, c, v):
        rows.append(r)
        cols.append(c)
        vals.append(v)

    cxx, cyy = (lam + 2 * mu) / hx**2, mu / hy**2  # x-equation diagonal stencils (uy: swap)
    cmix = (lam + mu) / (4 * hx * hy)  # mixed-derivative coupling
    for i in range(nx):
        for j in range(ny):
            r = idx[i, j]
            if on_bnd[i, j]:  # clamped boundary: identity rows for both components
                add(r, r, 1.0)
                add(n + r, n + r, 1.0)
                continue
            # x-equation: (lam+2mu) uxx + mu uyy + (lam+mu) d2uy/dxdy = -fx
            add(r, idx[i + 1, j], cxx)
            add(r, idx[i - 1, j], cxx)
            add(r, idx[i, j + 1], cyy)
            add(r, idx[i, j - 1], cyy)
            add(r, r, -2 * cxx - 2 * cyy)
            for (di, dj), s in (((1, 1), cmix), ((1, -1), -cmix), ((-1, 1), -cmix), ((-1, -1), cmix)):
                add(r, n + idx[i + di, j + dj], s)
            # y-equation: mu uxx + (lam+2mu) uyy + (lam+mu) d2ux/dxdy = -fy
            add(n + r, n + idx[i + 1, j], mu / hx**2)
            add(n + r, n + idx[i - 1, j], mu / hx**2)
            add(n + r, n + idx[i, j + 1], (lam + 2 * mu) / hy**2)
            add(n + r, n + idx[i, j - 1], (lam + 2 * mu) / hy**2)
            add(n + r, n + r, -2 * mu / hx**2 - 2 * (lam + 2 * mu) / hy**2)
            for (di, dj), s in (((1, 1), cmix), ((1, -1), -cmix), ((-1, 1), -cmix), ((-1, -1), cmix)):
                add(n + r, idx[i + di, j + dj], s)

    a = sp.csc_matrix((vals, (rows, cols)), shape=(2 * n, 2 * n))
    f = np.broadcast_to(np.asarray(body_force, dtype=float), (nx, ny, 2)).reshape(n, 2)
    dv = np.broadcast_to(np.asarray(dirichlet, dtype=float), (nx, ny, 2)).reshape(n, 2)
    b = np.zeros(2 * n)
    interior = ~on_bnd.ravel()
    b[:n][interior] = -f[interior, 0]
    b[n:][interior] = -f[interior, 1]
    b[:n][~interior] = dv[~interior, 0]  # boundary identity rows set the clamped displacement
    b[n:][~interior] = dv[~interior, 1]
    u = spla.spsolve(a, b)
    return np.stack([u[:n], u[n:]], axis=1).reshape(nx, ny, 2)
