"""Compressible gas dynamics -- the 1-D Euler equations with an exact Riemann solver.

The compressible Euler equations govern inviscid high-speed / high-pressure / high-temperature gas flow:

    d/dt [rho, rho u, E] + d/dx [rho u, rho u^2 + p, u (E + p)] = 0,   p = (gamma - 1)(E - rho u^2 / 2),

for density ``rho``, velocity ``u``, pressure ``p`` and total energy ``E`` of an ideal gas with ratio of
specific heats ``gamma``. This module provides two complementary pieces:

* :func:`exact_riemann_solution` -- the *exact* self-similar solution of the Riemann problem (arbitrary
  left/right states separated by a discontinuity), via Toro's pressure-function Newton iteration for the
  star region followed by exact sampling of the left/right shocks, rarefactions and contact. This is the
  analytic reference for shock-capturing schemes (e.g. the Sod shock tube).
* :func:`solve_euler_1d` -- a conservative finite-volume solver with the HLL approximate-Riemann flux and
  adaptive CFL time stepping, which captures shocks and contacts at the correct speeds and converges to
  the exact Riemann solution as the grid is refined.

Reference: Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics* (2009), ch. 3-4; Sod, *J.
Computational Physics* 27 (1978).
"""

import math
from typing import Any

import numpy as np

State = tuple[float, float, float]  # (rho, u, p)


def _wave_function(p: float, rho_k: float, p_k: float, a_k: float, gamma: float) -> float:
    """Toro's ``f_K(p)``: the velocity change across the left/right wave (shock if ``p>p_k`` else fan)."""
    if p > p_k:  # shock
        a = 2.0 / ((gamma + 1.0) * rho_k)
        b = (gamma - 1.0) / (gamma + 1.0) * p_k
        return (p - p_k) * math.sqrt(a / (p + b))
    return 2.0 * a_k / (gamma - 1.0) * ((p / p_k) ** ((gamma - 1.0) / (2.0 * gamma)) - 1.0)


def _star_region(left: State, right: State, gamma: float) -> tuple[float, float]:
    """Solve for the star-region pressure and velocity ``(p*, u*)`` by Newton iteration on the pressure."""
    rho_l, u_l, p_l = left
    rho_r, u_r, p_r = right
    a_l = math.sqrt(gamma * p_l / rho_l)
    a_r = math.sqrt(gamma * p_r / rho_r)

    def total(p: float) -> float:
        return _wave_function(p, rho_l, p_l, a_l, gamma) + _wave_function(p, rho_r, p_r, a_r, gamma) + (u_r - u_l)

    p = 0.5 * (p_l + p_r)
    for _ in range(100):
        dp = 1.0e-7 * p
        deriv = (total(p + dp) - total(p - dp)) / (2.0 * dp)
        p_new = max(p - total(p) / deriv, 1.0e-9)
        if abs(p_new - p) < 1.0e-13 * p:
            p = p_new
            break
        p = p_new
    u_star = 0.5 * (u_l + u_r) + 0.5 * (
        _wave_function(p, rho_r, p_r, a_r, gamma) - _wave_function(p, rho_l, p_l, a_l, gamma)
    )
    return p, u_star


def exact_riemann_solution(left: State, right: State, x: Any, t: float, *, gamma: float = 1.4) -> np.ndarray:
    """Return the exact solution ``(rho, u, p)`` of the Euler Riemann problem at positions ``x``, time ``t``.

    Args:
        left: the left state ``(rho, u, p)`` (for ``x < 0`` at ``t = 0``).
        right: the right state ``(rho, u, p)`` (for ``x > 0``).
        x: positions relative to the initial discontinuity (the solution is self-similar in ``x/t``).
        t: time (> 0).
        gamma: ratio of specific heats.

    Returns:
        Array of shape ``(len(x), 3)`` with columns ``rho, u, p``.
    """
    rho_l, u_l, p_l = left
    rho_r, u_r, p_r = right
    a_l = math.sqrt(gamma * p_l / rho_l)
    a_r = math.sqrt(gamma * p_r / rho_r)
    p_star, u_star = _star_region(left, right, gamma)
    g1 = (gamma - 1.0) / (gamma + 1.0)
    out = np.empty((len(np.asarray(x)), 3))
    for i, xi in enumerate(np.asarray(x, dtype=np.float64)):
        s = xi / t
        if s < u_star:  # left of the contact
            if p_star > p_l:  # left shock
                s_shock = u_l - a_l * math.sqrt((gamma + 1.0) / (2.0 * gamma) * p_star / p_l + (gamma - 1.0) / (2.0 * gamma))
                if s < s_shock:
                    rho, u, p = rho_l, u_l, p_l
                else:
                    rho = rho_l * (p_star / p_l + g1) / (g1 * p_star / p_l + 1.0)
                    u, p = u_star, p_star
            else:  # left rarefaction
                a_star_l = a_l * (p_star / p_l) ** ((gamma - 1.0) / (2.0 * gamma))
                if s < u_l - a_l:
                    rho, u, p = rho_l, u_l, p_l
                elif s > u_star - a_star_l:
                    rho = rho_l * (p_star / p_l) ** (1.0 / gamma)
                    u, p = u_star, p_star
                else:  # inside the fan
                    u = 2.0 / (gamma + 1.0) * (a_l + (gamma - 1.0) / 2.0 * u_l + s)
                    c = 2.0 / (gamma + 1.0) * (a_l + (gamma - 1.0) / 2.0 * (u_l - s))
                    rho = rho_l * (c / a_l) ** (2.0 / (gamma - 1.0))
                    p = p_l * (c / a_l) ** (2.0 * gamma / (gamma - 1.0))
        else:  # right of the contact
            if p_star > p_r:  # right shock
                s_shock = u_r + a_r * math.sqrt((gamma + 1.0) / (2.0 * gamma) * p_star / p_r + (gamma - 1.0) / (2.0 * gamma))
                if s > s_shock:
                    rho, u, p = rho_r, u_r, p_r
                else:
                    rho = rho_r * (p_star / p_r + g1) / (g1 * p_star / p_r + 1.0)
                    u, p = u_star, p_star
            else:  # right rarefaction
                a_star_r = a_r * (p_star / p_r) ** ((gamma - 1.0) / (2.0 * gamma))
                if s > u_r + a_r:
                    rho, u, p = rho_r, u_r, p_r
                elif s < u_star + a_star_r:
                    rho = rho_r * (p_star / p_r) ** (1.0 / gamma)
                    u, p = u_star, p_star
                else:  # inside the fan
                    u = 2.0 / (gamma + 1.0) * (-a_r + (gamma - 1.0) / 2.0 * u_r + s)
                    c = 2.0 / (gamma + 1.0) * (a_r - (gamma - 1.0) / 2.0 * (u_r - s))
                    rho = rho_r * (c / a_r) ** (2.0 / (gamma - 1.0))
                    p = p_r * (c / a_r) ** (2.0 * gamma / (gamma - 1.0))
        out[i] = (rho, u, p)
    return out


def _to_conserved(prim: np.ndarray, gamma: float) -> np.ndarray:
    rho, u, p = prim[0], prim[1], prim[2]
    return np.array([rho, rho * u, p / (gamma - 1.0) + 0.5 * rho * u * u])


def _to_primitive(cons: np.ndarray, gamma: float) -> np.ndarray:
    rho = cons[0]
    u = cons[1] / rho
    p = (gamma - 1.0) * (cons[2] - 0.5 * rho * u * u)
    return np.array([rho, u, p])


def _euler_flux(cons: np.ndarray, gamma: float) -> np.ndarray:
    rho, u, p = _to_primitive(cons, gamma)
    return np.array([rho * u, rho * u * u + p, u * (cons[2] + p)])


def _hll_flux(ul: np.ndarray, ur: np.ndarray, gamma: float) -> np.ndarray:
    rho_l, u_l, p_l = _to_primitive(ul, gamma)
    rho_r, u_r, p_r = _to_primitive(ur, gamma)
    a_l = math.sqrt(gamma * p_l / rho_l)
    a_r = math.sqrt(gamma * p_r / rho_r)
    sl = min(u_l - a_l, u_r - a_r)
    sr = max(u_l + a_l, u_r + a_r)
    if sl >= 0.0:
        return _euler_flux(ul, gamma)
    if sr <= 0.0:
        return _euler_flux(ur, gamma)
    return (sr * _euler_flux(ul, gamma) - sl * _euler_flux(ur, gamma) + sl * sr * (ur - ul)) / (sr - sl)


def solve_euler_1d(rho0: Any, u0: Any, p0: Any, dx: float, t_final: float, *, gamma: float = 1.4,
                   cfl: float = 0.4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evolve the 1-D Euler equations with a conservative HLL finite-volume scheme (outflow boundaries).

    Args:
        rho0, u0, p0: initial density, velocity and pressure arrays on a uniform grid of spacing ``dx``.
        dx: cell width.
        t_final: final time.
        gamma: ratio of specific heats.
        cfl: CFL number for the adaptive time step.

    Returns:
        The final ``(rho, u, p)`` arrays. Captures shocks/contacts at the correct speeds and converges to
        :func:`exact_riemann_solution` under grid refinement.
    """
    prim = np.vstack([np.asarray(rho0, dtype=np.float64), np.asarray(u0, dtype=np.float64), np.asarray(p0, dtype=np.float64)])
    n = prim.shape[1]
    u = np.array([_to_conserved(prim[:, i], gamma) for i in range(n)]).T
    t = 0.0
    while t < t_final:
        pr = np.array([_to_primitive(u[:, i], gamma) for i in range(n)])
        smax = float(np.max(np.abs(pr[:, 1]) + np.sqrt(gamma * pr[:, 2] / pr[:, 0])))
        dt = min(cfl * dx / smax, t_final - t)
        f = np.zeros((3, n + 1))
        for i in range(1, n):
            f[:, i] = _hll_flux(u[:, i - 1], u[:, i], gamma)
        f[:, 0] = _euler_flux(u[:, 0], gamma)  # zero-gradient (outflow) boundaries
        f[:, n] = _euler_flux(u[:, n - 1], gamma)
        u = u - dt / dx * (f[:, 1:] - f[:, :-1])
        t += dt
    pr = np.array([_to_primitive(u[:, i], gamma) for i in range(n)])
    return pr[:, 0], pr[:, 1], pr[:, 2]
