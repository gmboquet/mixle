"""Pseudo-spectral incompressible Navier-Stokes (2-D and 3-D, with an optional Smagorinsky LES closure).

This is a Fourier pseudo-spectral solver for the incompressible Navier-Stokes equations on a periodic box
``[0, length)^d`` (``d = 2`` or ``3``):

    du/dt + (u . grad) u = -grad p + nu * laplacian(u),   div u = 0.

Working in Fourier space makes incompressibility trivial -- the pressure projection is the algebraic
removal of the longitudinal part of each mode, ``u_hat <- u_hat - k (k . u_hat)/|k|^2`` -- and the viscous
term is the exact multiplier ``-nu |k|^2``. The nonlinear advection is evaluated pseudo-spectrally
(derivatives in Fourier space, products in physical space) and time-stepped with classical RK4. Optional
2/3-rule dealiasing removes the aliasing error of the quadratic nonlinearity for well-resolved turbulent
runs, and an optional Smagorinsky eddy viscosity ``nu_t = (C_s * Delta)^2 |S|`` provides a large-eddy
(LES) subgrid closure for high-Reynolds-number flows that the grid cannot resolve directly.

The DNS core (``smagorinsky = 0``) is exact on the analytic benchmarks: it reproduces the 2-D Taylor-Green
vortex decay ``u = -cos x sin y * e^{-2 nu t}`` and the 3-D ABC/Beltrami flow decay ``u = u_0 * e^{-nu t}``
to machine precision. The LES term reduces to the DNS solver as ``C_s -> 0`` and is strictly dissipative.

Reference: Canuto, Hussaini, Quarteroni & Zang, *Spectral Methods in Fluid Dynamics* (1988); Orszag &
Patterson (1972); Smagorinsky, *Monthly Weather Review* 91 (1963).
"""

import math
from typing import Any

import numpy as np


def _wavenumbers(shape: tuple[int, ...], length: float) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Return the per-axis wavenumber grids ``K[j]``, ``|k|^2`` and a zero-safe ``|k|^2`` for division."""
    ks = [2.0 * np.pi * np.fft.fftfreq(n, d=length / n) for n in shape]
    grids = list(np.meshgrid(*ks, indexing="ij"))
    k2 = sum(k * k for k in grids)
    k2_safe = np.where(k2 == 0.0, 1.0, k2)
    return grids, k2, k2_safe


def _dealias_mask(shape: tuple[int, ...]) -> np.ndarray:
    """Return the 2/3-rule mask zeroing the top third of wavenumbers along every axis."""
    mask = np.ones(shape, dtype=bool)
    for axis, n in enumerate(shape):
        freq = np.fft.fftfreq(n, d=1.0 / n)
        keep = np.abs(freq) < (n / 3.0)
        mask &= np.expand_dims(keep, tuple(a for a in range(len(shape)) if a != axis))
    return mask


def incompressible_ns_spectral(
    velocity: Any,
    nu: float,
    dt: float,
    n_steps: int,
    *,
    length: float = 2.0 * math.pi,
    dealias: bool = False,
    smagorinsky: float = 0.0,
) -> tuple[np.ndarray, ...]:
    """Evolve a periodic incompressible velocity field with the pseudo-spectral RK4 solver.

    Args:
        velocity: tuple/list of ``d`` real arrays (``d = 2`` or ``3``), each of identical shape
            ``(N,)*d``, giving the velocity components on a uniform grid over ``[0, length)^d``.
        nu: kinematic viscosity (``1/Re`` in non-dimensional units).
        dt: time step.
        n_steps: number of RK4 steps to take.
        length: side length of the periodic box (default ``2*pi``).
        dealias: apply the 2/3-rule to the nonlinear term (recommended for turbulent/under-resolved runs).
        smagorinsky: Smagorinsky constant ``C_s`` for the LES eddy viscosity ``(C_s*Delta)^2 |S|``; ``0``
            (the default) is direct numerical simulation with no subgrid model.

    Returns:
        The evolved velocity components as a tuple of real arrays (same shapes as the input).
    """
    fields = [np.asarray(u, dtype=np.float64) for u in velocity]
    d = len(fields)
    if d not in (2, 3):
        raise ValueError("incompressible_ns_spectral supports 2-D or 3-D velocity fields.")
    shape = fields[0].shape
    if any(f.shape != shape for f in fields) or len(shape) != d:
        raise ValueError("velocity components must all have shape (N,)*d.")

    grids, k2, k2_safe = _wavenumbers(shape, float(length))
    mask = _dealias_mask(shape) if dealias else None
    delta = float(length) / shape[0]
    cs2_delta2 = (float(smagorinsky) * delta) ** 2

    fftn, ifftn = np.fft.fftn, np.fft.ifftn

    def physical_grad(uh: np.ndarray, j: int) -> np.ndarray:
        return ifftn(1j * grids[j] * uh).real

    def rhs(uh_stack: list[np.ndarray]) -> list[np.ndarray]:
        u = [ifftn(uh).real for uh in uh_stack]
        # nonlinear advection  N_i = -(u . grad) u_i  (pseudo-spectral, products in physical space)
        nonlin = []
        grad = [[physical_grad(uh_stack[i], j) for j in range(d)] for i in range(d)]
        for i in range(d):
            nonlin.append(-sum(u[j] * grad[i][j] for j in range(d)))
        # optional Smagorinsky LES: add the divergence of the subgrid stress 2 nu_t S_ij
        if cs2_delta2 > 0.0:
            strain = [[0.5 * (grad[i][j] + grad[j][i]) for j in range(d)] for i in range(d)]
            s_mag = np.sqrt(2.0 * sum(strain[i][j] ** 2 for i in range(d) for j in range(d)))
            nu_t = cs2_delta2 * s_mag
            for i in range(d):
                sgs = sum(ifftn(1j * grids[j] * fftn(2.0 * nu_t * strain[i][j])).real for j in range(d))
                nonlin[i] = nonlin[i] + sgs
        nh = [fftn(n) for n in nonlin]
        if mask is not None:
            nh = [n * mask for n in nh]
        # project the forcing to be divergence-free, then add the exact viscous multiplier
        div = sum(grids[j] * nh[j] for j in range(d)) / k2_safe
        return [nh[i] - grids[i] * div - nu * k2 * uh_stack[i] for i in range(d)]

    uh = [fftn(f) for f in fields]
    for _ in range(int(n_steps)):
        k1 = rhs(uh)
        k2_ = rhs([uh[i] + 0.5 * dt * k1[i] for i in range(d)])
        k3 = rhs([uh[i] + 0.5 * dt * k2_[i] for i in range(d)])
        k4 = rhs([uh[i] + dt * k3[i] for i in range(d)])
        uh = [uh[i] + dt / 6.0 * (k1[i] + 2.0 * k2_[i] + 2.0 * k3[i] + k4[i]) for i in range(d)]
    return tuple(ifftn(h).real for h in uh)


def kinetic_energy(velocity: Any, length: float = 2.0 * math.pi) -> float:
    """Return the total kinetic energy ``(1/2) integral |u|^2 dV`` of a periodic velocity field."""
    fields = [np.asarray(u, dtype=np.float64) for u in velocity]
    d = len(fields)
    cell = (float(length) / fields[0].shape[0]) ** d
    return 0.5 * float(sum(np.sum(f * f) for f in fields)) * cell
