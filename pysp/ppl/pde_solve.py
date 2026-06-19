"""Adjoint-capable sparse PDE solves for large-scale inverse problems.

The keystone of scalable PDE-constrained inference: a sparse linear solve ``A(theta) u = b(theta)`` whose
gradient is computed by the adjoint method rather than by differentiating through the factorization. The
forward pass factorizes ``A`` once (sparse LU); the backward pass solves a single adjoint system
``A^T lambda = dL/du`` and reads the parameter gradients off ``lambda`` and ``u``. Cost is one extra solve
regardless of how many coefficients ``theta`` has, and memory is ``O(nnz)`` -- so a 2-D or 3-D
coefficient field with 10^4-10^6 unknowns is tractable where a dense ``torch.linalg.solve`` (O(n^2) memory,
O(n^3) work, dense backward) is not.

The assembly helpers build the sparse operator differentiably from a coefficient field on a structured
grid: the variable-coefficient divergence form ``-div(kappa grad u)`` (diffusion / conduction / Darcy /
seismic slowness) and the Helmholtz operator ``-lap u - (omega^2 / c(x)^2) u`` (radar / sonar / acoustic
scattering). Each returns ``(rows, cols, vals)`` with fixed integer patterns and ``vals`` a torch tensor
that carries gradients back to the field, so the whole forward map is differentiable end to end and plugs
into :class:`pysp.ppl.inverse.DifferentialProxy` via ``solver='sparse'``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["sparse_solve", "divergence_form", "helmholtz_operator", "laplacian"]


def _torch():
    import torch

    return torch


def _integrate_ops(rhs, y0, t_grid, torch, method="rk4"):
    """Integrate ``du/dt = rhs(u, t)`` from ``y0`` over ``t_grid`` (the ops-namespace ODE integrator)."""
    y = y0 if torch.is_tensor(y0) else torch.as_tensor(y0, dtype=torch.float64)
    states = [y]
    for i in range(len(t_grid) - 1):
        t = t_grid[i]
        h = t_grid[i + 1] - t_grid[i]
        if method == "euler":
            y = y + h * rhs(y, t)
        elif method == "rk4":
            k1 = rhs(y, t)
            k2 = rhs(y + 0.5 * h * k1, t + 0.5 * h)
            k3 = rhs(y + 0.5 * h * k2, t + 0.5 * h)
            k4 = rhs(y + h * k3, t + h)
            y = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            raise ValueError(f"unknown method {method!r}; use 'rk4' or 'euler'.")
        states.append(y)
    return torch.stack(states)


def _matvec(rows, cols, vals, n, x, torch):
    """Differentiable sparse matrix-vector product ``A x`` for ``A = sparse(rows, cols, vals)`` (n x n).

    Lets a time-dependent PDE apply an assembled operator (e.g. the Laplacian) explicitly without a solve,
    with gradients flowing to both ``vals`` (the coefficient field) and ``x``.
    """
    rows = torch.as_tensor(rows, dtype=torch.long)
    cols = torch.as_tensor(cols, dtype=torch.long)
    out = torch.zeros(int(n), dtype=vals.dtype)
    return out.index_add(0, rows, vals * x[cols])


def _integrate_record(step, y0, n_steps, record, torch, checkpoint=None):
    """Step a time-dependent system and record an observation at each step, with optional checkpointing.

    ``step(y, i) -> y_next`` advances one step; ``record(y, i) -> obs`` extracts the observed quantity at
    step ``i`` (e.g. the field at receiver nodes). Returns the stacked records for steps ``0..n_steps``.

    With ``checkpoint=K``, segments of ``K`` steps are wrapped in gradient checkpointing: the full
    intermediate states are *not* retained for the backward pass but recomputed segment by segment, so the
    memory is O(n_steps/K) boundary states + O(K) recomputed -- the adjoint-state pattern that makes
    full-waveform inversion (many steps over a large field) feasible. Records (usually small, e.g. a few
    receivers) are kept. Gradients are identical to the non-checkpointed integration.
    """

    def run_segment(y_start, i0, length):
        y = y_start
        seg = []
        for j in range(int(length)):
            seg.append(record(y, int(i0) + j))  # state before step i0+j
            y = step(y, int(i0) + j)
        return y, torch.stack(seg)

    if not checkpoint or checkpoint <= 0:
        y, recs = run_segment(y0, 0, n_steps)
        return torch.cat([recs, record(y, n_steps).unsqueeze(0)])

    from torch.utils.checkpoint import checkpoint as ckpt

    k = int(checkpoint)
    y = y0
    chunks = []
    i = 0
    while i < n_steps:
        length = min(k, n_steps - i)
        y, seg = ckpt(run_segment, y, i, length, use_reentrant=False)
        chunks.append(seg)
        i += length
    chunks.append(record(y, n_steps).unsqueeze(0))
    return torch.cat(chunks)


def _sparse_solve_function(torch):
    """Build (once) the autograd Function bound to this torch import."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    class _SparseSolve(torch.autograd.Function):
        @staticmethod
        def forward(ctx, vals, rows, cols, n, b):
            r = rows.detach().cpu().numpy()
            c = cols.detach().cpu().numpy()
            A = sp.csc_matrix((vals.detach().cpu().numpy(), (r, c)), shape=(int(n), int(n)))
            lu = spla.splu(A)
            bnp = b.detach().cpu().numpy()
            u = lu.solve(bnp)
            ctx.lu = lu
            ctx.save_for_backward(vals, rows, cols, torch.as_tensor(u, dtype=vals.dtype))
            return torch.as_tensor(u, dtype=vals.dtype)

        @staticmethod
        def backward(ctx, grad_u):
            vals, rows, cols, u = ctx.saved_tensors
            # adjoint system A^H lambda = dL/du -- a single extra solve. The conjugate transpose (and the
            # conjugate on u below) is what makes the gradient correct for COMPLEX systems (e.g. the
            # Helmholtz operator in radar / acoustic scattering); for real A it is identical to A^T.
            lam = ctx.lu.solve(grad_u.detach().cpu().numpy(), trans="H")
            lam_t = torch.as_tensor(lam, dtype=vals.dtype)
            r = rows.long()
            c = cols.long()
            # dL/dA = -lambda u^H, read only at the (row, col) pattern: grad_vals[k] = -lambda[r_k] conj(u[c_k])
            grad_vals = -lam_t[r] * u[c].conj()
            grad_b = lam_t
            return grad_vals, None, None, None, grad_b

    return _SparseSolve


_CACHE: dict = {}
_USED = [False]  # set when sparse_solve runs, so fit_field can block the (silently wrong) dense Hessian


def sparse_used_since(reset: bool = False) -> bool:
    """Whether sparse_solve has run since the last reset (guards the second-order Laplace path)."""
    was = _USED[0]
    if reset:
        _USED[0] = False
    return was


def sparse_solve(vals, rows, cols, n, b):
    """Solve the sparse system ``A u = b`` where ``A = sparse(rows, cols, vals)`` (n x n), with adjoint grads.

    ``vals`` (nnz,) and ``b`` (n,) are torch tensors (gradients flow to both); ``rows``/``cols`` are fixed
    long tensors giving the sparsity pattern; duplicate ``(row, col)`` entries are summed. Returns ``u`` (n,).

    The adjoint backward uses a factorization (not autograd), so it is first-order only: it powers MAP and
    Gauss-Newton (``how='gauss_newton'``), but a forward using it must not be fit with the second-order
    ``how='laplace'`` (the dense Hessian would be silently wrong); fit_field detects this and raises.
    """
    torch = _torch()
    fn = _CACHE.get("fn")
    if fn is None:
        fn = _CACHE["fn"] = _sparse_solve_function(torch)
    _USED[0] = True
    rows = torch.as_tensor(rows, dtype=torch.long)
    cols = torch.as_tensor(cols, dtype=torch.long)
    return fn.apply(vals, rows, cols, int(n), b)


# --------------------------------------------------------------------------------------------------
# Differentiable assembly of structured-grid operators (vals carry gradients to the coefficient field).
# --------------------------------------------------------------------------------------------------
def _grid_faces(shape, spacing):
    """Precompute the fixed face list of a structured grid: every adjacent node pair ``(a, b)`` along an
    axis with its ``1/h^2``, plus the boundary mask. Boundary nodes get a Dirichlet identity row; an
    interior node keeps the flux coupling to a boundary neighbour (whose value the source sets)."""
    shape = tuple(int(s) for s in shape)
    ndim = len(shape)
    spacing = np.broadcast_to(np.asarray(spacing, dtype=float), (ndim,))
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


def divergence_form(kappa, shape, *, spacing=1.0, torch=None):
    """Assemble ``-div(kappa grad u)`` with Dirichlet boundaries on a structured grid.

    ``kappa`` is a torch tensor of per-node conductivities (length ``prod(shape)``); the face conductance
    is the arithmetic mean of the two adjacent nodes. Returns ``(rows, cols, vals, n)`` with ``vals``
    differentiable in ``kappa``. Boundary rows are the identity (so the source sets the Dirichlet values);
    an interior node retains the flux term coupling it to any boundary neighbour.
    """
    torch = torch or _torch()
    g = _grid_faces(shape, spacing)
    fa = torch.as_tensor(g["face_a"], dtype=torch.long)
    fb = torch.as_tensor(g["face_b"], dtype=torch.long)
    fw = torch.as_tensor(g["face_w"], dtype=kappa.dtype)
    cond = 0.5 * (kappa[fa] + kappa[fb]) * fw  # face conductance between the two nodes
    bmask = torch.as_tensor(g["boundary_mask"])
    n = g["n"]
    a_int = ~bmask[fa]  # emit a face's contribution to an endpoint's row only if that endpoint is interior
    b_int = ~bmask[fb]
    diag = torch.zeros(n, dtype=kappa.dtype)
    diag = diag.index_add(0, fa[a_int], cond[a_int]).index_add(0, fb[b_int], cond[b_int])
    interior = torch.as_tensor(g["interior"], dtype=torch.long)
    bnd = torch.as_tensor(g["boundary"], dtype=torch.long)
    rows = torch.cat([fa[a_int], fb[b_int], interior, bnd])
    cols = torch.cat([fb[a_int], fa[b_int], interior, bnd])
    vals = torch.cat([-cond[a_int], -cond[b_int], diag[interior], torch.ones(len(bnd), dtype=kappa.dtype)])
    return rows, cols, vals, n


def laplacian(shape, *, spacing=1.0, torch=None):
    """The constant-coefficient negative Laplacian ``-lap`` (Dirichlet) as ``(rows, cols, vals, n)``."""
    torch = torch or _torch()
    n = int(np.prod([int(s) for s in shape]))
    return divergence_form(torch.ones(n, dtype=torch.float64), shape, spacing=spacing, torch=torch)


def helmholtz_operator(slowness2, shape, *, omega, spacing=1.0, torch=None):
    """Assemble the Helmholtz operator ``-lap u - omega^2 * slowness2(x) * u`` (Dirichlet) on a grid.

    ``slowness2`` is a per-node field of ``1/c(x)^2`` (squared slowness); ``omega`` the angular frequency.
    Returns ``(rows, cols, vals, n)`` differentiable in ``slowness2`` -- the acoustic-scattering forward
    operator (real-valued here; complex absorbing boundaries are a later phase).
    """
    torch = torch or _torch()
    n = int(np.prod([int(s) for s in shape]))
    rows, cols, vals, n = divergence_form(torch.ones(n, dtype=slowness2.dtype), shape, spacing=spacing, torch=torch)
    # subtract omega^2 * slowness2 on the diagonal (interior nodes); boundary stays identity
    g = _grid_faces(shape, spacing)
    interior = torch.as_tensor(g["interior"], dtype=torch.long)
    extra_rows = interior
    extra_cols = interior
    extra_vals = -(omega**2) * slowness2[interior]
    rows = torch.cat([rows, extra_rows])
    cols = torch.cat([cols, extra_cols])
    vals = torch.cat([vals, extra_vals])
    return rows, cols, vals, n
