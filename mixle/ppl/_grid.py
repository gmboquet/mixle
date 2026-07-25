"""Structured-grid adjacency helper shared by field priors and the PDE solvers.

``_grid_faces`` precomputes the fixed face list of a structured grid (every adjacent node pair along
an axis, with ``1/h^2`` weights, plus the boundary mask). It is pure NumPy with no PDE-solver
dependency, so it lives in mixle proper: the edge-preserving field priors (``TotalVariation`` / ``Potts``
in :mod:`mixle.ppl.priors`) use it, and so does the PDE divergence-form assembly (now in the
``mixle-pde`` package, which imports it from here). Kept here so neither side depends on the other.
"""

from __future__ import annotations

from numbers import Integral

import numpy as np


def _grid_faces(shape, spacing):
    """Precompute the fixed face list of a structured grid: every adjacent node pair ``(a, b)`` along an
    axis with its ``1/h^2``, plus the boundary mask. Boundary nodes get a Dirichlet identity row; an
    interior node keeps the flux coupling to a boundary neighbour (whose value the source sets)."""
    try:
        raw_shape = tuple(shape)
    except TypeError as error:
        raise TypeError("shape must be a non-empty sequence of positive integers") from error
    if not raw_shape:
        raise ValueError("shape must contain at least one grid axis")
    if any(isinstance(extent, (bool, np.bool_)) or not isinstance(extent, Integral) for extent in raw_shape):
        raise TypeError("every shape extent must be a positive integer")
    shape = tuple(int(extent) for extent in raw_shape)
    if any(extent <= 0 for extent in shape):
        raise ValueError("every shape extent must be a positive integer")
    ndim = len(shape)
    try:
        spacing = np.broadcast_to(np.asarray(spacing, dtype=float), (ndim,))
    except (TypeError, ValueError) as error:
        raise ValueError(f"spacing must be scalar or broadcastable to {ndim} axes") from error
    if not np.all(np.isfinite(spacing)) or np.any(spacing <= 0.0):
        raise ValueError("every spacing value must be positive and finite")
    n = int(np.prod(shape))
    idx = np.arange(n).reshape(shape)
    on_boundary = np.zeros(shape, dtype=bool)
    for ax in range(ndim):
        sl0 = [slice(None)] * ndim
        sl1 = [slice(None)] * ndim
        sl0[ax] = 0
        sl1[ax] = shape[ax] - 1
        on_boundary[tuple(sl0)] = True
        on_boundary[tuple(sl1)] = True
    face_a, face_b, face_w = [], [], []
    for ax in range(ndim):
        lo = [slice(None)] * ndim
        hi = [slice(None)] * ndim
        lo[ax] = slice(0, shape[ax] - 1)
        hi[ax] = slice(1, shape[ax])
        a = idx[tuple(lo)].ravel()
        b = idx[tuple(hi)].ravel()
        face_a.append(a)
        face_b.append(b)
        face_w.append(np.full(len(a), 1.0 / spacing[ax] ** 2))
    return {
        "n": n,
        "shape": shape,
        "boundary": np.where(on_boundary.ravel())[0],
        "interior": np.where(~on_boundary.ravel())[0],
        "boundary_mask": on_boundary.ravel(),
        "face_a": np.concatenate(face_a) if face_a else np.array([], dtype=int),
        "face_b": np.concatenate(face_b) if face_b else np.array([], dtype=int),
        "face_w": np.concatenate(face_w) if face_w else np.array([], dtype=float),
    }
