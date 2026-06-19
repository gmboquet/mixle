"""The math namespace handed to differential forward models, so user callbacks never import a backend.

A forward model for a PDE/ODE inverse problem is the one part that cannot be written as `pysp.ppl`
distribution algebra -- the physics is a function. To keep that escape hatch consistent with the rest of
the library, the callback is handed an ``ops`` namespace (curated math + grid assembly + solves) and a
``p`` namespace (the latent drivers by name), instead of a raw tensor library and a parameter dict:

    rhs = lambda u, t, p, ops: -p.k * u                       # an ODE right-hand side
    forward = lambda p, ops: ops.sparse_solve(*ops.divergence_form(ops.exp(p.field), shape), b)

``ops`` delegates to the autograd backend under the hood; user code stays backend-agnostic.
"""

from __future__ import annotations

from typing import Any


class _Params:
    """Latent drivers exposed by name: ``p.k``, ``p.field`` -- the bound values at the current iterate."""

    def __init__(self, values: dict):
        object.__setattr__(self, "_v", values)

    def __getattr__(self, name: str) -> Any:
        try:
            return self._v[name]
        except KeyError as e:
            raise AttributeError(f"no driver {name!r}; declared drivers are {sorted(self._v)}.") from e


class _Ops:
    """A backend-agnostic math namespace: elementwise functions, reductions, an ODE integrator, and the
    differentiable grid operators / sparse solve from :mod:`pysp.ppl.pde_solve`."""

    def __init__(self):
        import torch

        self._t = torch

    # elementwise / reductions (delegate to the backend; arithmetic operators work on the tensors directly)
    def exp(self, x):
        return self._t.exp(x)

    def log(self, x):
        return self._t.log(x)

    def sin(self, x):
        return self._t.sin(x)

    def cos(self, x):
        return self._t.cos(x)

    def sqrt(self, x):
        return self._t.sqrt(x)

    def tanh(self, x):
        return self._t.tanh(x)

    def abs(self, x):
        return self._t.abs(x)

    def clamp(self, x, lo=None, hi=None):
        return self._t.clamp(x, min=lo, max=hi)

    def sum(self, x, axis=None):
        return self._t.sum(x) if axis is None else self._t.sum(x, dim=axis)

    def stack(self, xs, axis=0):
        return self._t.stack(list(xs), dim=axis)

    def cat(self, xs, axis=0):
        return self._t.cat(list(xs), dim=axis)

    def tensor(self, x):
        return self._t.as_tensor(x, dtype=self._t.float64)

    def zeros(self, *shape):
        return self._t.zeros(*shape, dtype=self._t.float64)

    def arange(self, n):
        return self._t.arange(n)

    def matmul(self, a, b):
        return a @ b

    def solve(self, A, b):
        """Dense linear solve (small systems). For large/sparse systems use ``sparse_solve``."""
        return self._t.linalg.solve(A, b)

    # ODE integration (a forward model convenience): rhs(u, t) -> du/dt
    def integrate(self, rhs, y0, t_grid, *, method: str = "rk4"):
        from pysp.ppl.pde_solve import _integrate_ops

        return _integrate_ops(rhs, y0, self.tensor(t_grid), self._t, method)

    # differentiable grid assembly + adjoint sparse solve (the PDE forward operators)
    def divergence_form(self, kappa, shape, *, spacing=1.0):
        from pysp.ppl.pde_solve import divergence_form

        return divergence_form(kappa, shape, spacing=spacing, torch=self._t)

    def helmholtz_operator(self, slowness2, shape, *, omega, spacing=1.0):
        from pysp.ppl.pde_solve import helmholtz_operator

        return helmholtz_operator(slowness2, shape, omega=omega, spacing=spacing, torch=self._t)

    def laplacian(self, shape, *, spacing=1.0):
        from pysp.ppl.pde_solve import laplacian

        return laplacian(shape, spacing=spacing, torch=self._t)

    def sparse_solve(self, rows, cols, vals, n, b):
        """Solve ``A u = b`` for ``A = sparse(rows, cols, vals)`` with adjoint gradients (one extra solve)."""
        from pysp.ppl.pde_solve import sparse_solve

        return sparse_solve(vals, rows, cols, n, b)


def make_ops() -> _Ops:
    return _Ops()
