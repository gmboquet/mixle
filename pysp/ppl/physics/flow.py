"""2D incompressible Navier-Stokes for flow inverse problems (phase 5).

The capstone: a differentiable 2D incompressible Navier-Stokes solver in the streamfunction-vorticity
formulation, built on the rest of the stack -- the streamfunction Poisson solve is the Phase-1 adjoint
``sparse_solve``, the time loop is the Phase-2 checkpointed integrator, and the posterior over the
unobservable upstream/initial condition (or viscosity) comes from the Phase-1b Gauss-Newton path.

Streamfunction-vorticity avoids the pressure / incompressibility saddle point: with streamfunction ``psi``
(velocity ``u = d psi/dy``, ``v = -d psi/dx``, automatically divergence-free) and vorticity ``omega``,

    d omega / dt + (u . grad) omega = nu * laplacian(omega),    laplacian(psi) = -omega,

so each step is a Poisson solve for ``psi`` plus an explicit vorticity-transport update. The inverse
problem -- recover the upstream flow configuration that produced an observed downstream flow -- is then a
``Differential`` observation whose forward integrates this stepper and records velocities at sensors.

The explicit scheme suits moderate Reynolds numbers on modest grids (the regime where these inverse
problems are well posed); high Reynolds / large 3-D needs implicit or stabilized schemes (the frontier).
"""

from __future__ import annotations

import numpy as np

from pysp.ppl.physics.pde_solve import laplacian


class NavierStokes2D:
    """A differentiable 2D incompressible Navier-Stokes stepper (streamfunction-vorticity, explicit).

    ``NavierStokes2D(n, viscosity=..., dt=...)`` builds the forward on an ``n x n`` grid with no-penetration
    walls (``psi = 0``, ``omega = 0`` on the boundary). In a forward callback, advance the vorticity with
    ``step(omega, ops)`` and read flow with ``streamfunction``/``velocity``; the latent driver (an upstream
    or initial vorticity, an inlet strength, a viscosity) flows through to the recorded velocities.
    """

    def __init__(
        self,
        n: int,
        *,
        viscosity: float,
        dt: float,
        spacing: float | None = None,
        implicit_diffusion: bool = False,
    ):
        self.n = int(n)
        self.nu = float(viscosity)
        self.dt = float(dt)
        self.h = float(spacing) if spacing is not None else 1.0 / (n - 1)
        self._poisson = laplacian((n, n), spacing=self.h)  # -laplacian with Dirichlet identity rows
        mask = np.ones((n, n))
        mask[0] = mask[-1] = mask[:, 0] = mask[:, -1] = 0.0
        self._mask = mask.ravel()
        self.implicit_diffusion = bool(implicit_diffusion)
        self._implicit = self._build_implicit() if implicit_diffusion else None

    def _build_implicit(self):
        """Assemble ``I + dt*nu*(-laplacian)`` (interior; identity on the walls) for an implicit diffusion
        step: ``(I + dt*nu*(-lap)) omega_{n+1} = omega_n - dt*advection``. Removes the diffusion CFL limit
        (stable for any dt), so the explicit advection step alone bounds dt -- robust at higher viscosity."""
        import scipy.sparse as sp

        n2 = self.n * self.n
        rows, cols, vals, _ = self._poisson
        L = sp.csc_matrix((vals.numpy(), (rows.numpy(), cols.numpy())), shape=(n2, n2)).tolil()
        for b in np.where(self._mask == 0.0)[0]:  # zero the boundary rows of L (kept identity by the +I below)
            L.rows[b] = []
            L.data[b] = []
        M = (sp.identity(n2, format="csc") + self.dt * self.nu * L.tocsc()).tocoo()
        import torch

        return (torch.as_tensor(M.row), torch.as_tensor(M.col), torch.as_tensor(M.data, dtype=torch.float64), n2)

    def _interior_mask(self, ops):
        return ops.tensor(self._mask)

    def _lap(self, a, ops):
        n, h = self.n, self.h
        A = a.reshape(n, n)
        out = ops.zeros(n, n)
        out[1:-1, 1:-1] = (A[2:, 1:-1] + A[:-2, 1:-1] + A[1:-1, 2:] + A[1:-1, :-2] - 4 * A[1:-1, 1:-1]) / h**2
        return out.reshape(-1)

    def streamfunction(self, omega, ops):
        """Solve ``laplacian(psi) = -omega`` with ``psi = 0`` on the walls (the Phase-1 adjoint solve)."""
        rows, cols, vals, n = self._poisson
        return ops.sparse_solve(rows, cols, vals, n, omega * self._interior_mask(ops))

    def velocity(self, psi, ops):
        """Divergence-free velocity ``(u, v) = (d psi/dy, -d psi/dx)`` from the streamfunction."""
        shape = (self.n, self.n)
        return ops.grad(psi, shape, 1, spacing=self.h), -ops.grad(psi, shape, 0, spacing=self.h)

    def step(self, omega, ops):
        """Advance the vorticity one time step (explicit, or implicit-diffusion if requested)."""
        shape = (self.n, self.n)
        mask = self._interior_mask(ops)
        psi = self.streamfunction(omega, ops)
        u, v = self.velocity(psi, ops)
        advection = u * ops.grad(omega, shape, 0, spacing=self.h) + v * ops.grad(omega, shape, 1, spacing=self.h)
        if self.implicit_diffusion:
            rhs = (omega - self.dt * advection) * mask  # diffusion handled by the implicit solve
            r, c, vv, nn = self._implicit
            return ops.sparse_solve(r, c, vv, nn, rhs) * mask
        omega_next = omega + self.dt * (-advection + self.nu * self._lap(omega, ops))
        return omega_next * mask
