"""2-D acoustic wave propagation with a perfectly-matched-layer (PML) absorbing boundary.

For seismic / radar forward modelling on a truncated grid you must stop outgoing waves from reflecting off
the domain edge. A heuristic sponge wastes domain and still leaks; a PML is the standard clean solution --
a thin layer in which a coordinate-stretching damping absorbs outgoing waves at (continuously) zero
reflection. This is the split-field PML for the first-order acoustic system (pressure split into px+py,
each damped in its own direction). Part of the earth-science/multiphysics work (Phase 5).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

__all__ = ["solve_wave_pml"]


def _ddx(f, dx):
    g = np.zeros_like(f)
    g[1:-1, :] = (f[2:, :] - f[:-2, :]) / (2 * dx)
    return g


def _ddy(f, dy):
    g = np.zeros_like(f)
    g[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2 * dy)
    return g


def _pml_profile(n: int, width: int, dx: float, c: float, strength: float) -> np.ndarray:
    """Quadratic damping ramp sigma(x): 0 in the interior, rising to ``strength`` over the edge layers."""
    sig = np.zeros(n)
    if width <= 0:
        return sig
    smax = strength * 3.0 * c / (2.0 * width * dx)
    ramp = (np.arange(width) / width) ** 2
    sig[:width] = smax * ramp[::-1]
    sig[-width:] = smax * ramp
    return sig


def solve_wave_pml(
    shape: tuple[int, int],
    *,
    c: float = 1.0,
    dx: float = 1.0,
    dt: float | None = None,
    n_steps: int = 400,
    source: Callable[[float], float] | None = None,
    source_loc: tuple[int, int] | None = None,
    pml_width: int = 20,
    pml_strength: float = 2.0,
    absorb: bool = True,
):
    """Propagate an acoustic pulse and return ``(final_pressure, energy_history)``.

    A Ricker-pulse point source (default) is injected at ``source_loc`` (default: centre). With
    ``absorb=True`` the four edges are PML layers of width ``pml_width``; with ``absorb=False`` they are
    hard reflecting walls (for comparison). ``energy_history`` is the total field energy per step -- with a
    working PML it decays to ~0 once the pulse leaves the domain, whereas a hard wall traps it.
    """
    nx, ny = (int(s) for s in shape)
    dt = float(dt) if dt is not None else 0.4 * dx / c
    sx_1d = _pml_profile(nx, pml_width, dx, c, pml_strength) if absorb else np.zeros(nx)
    sy_1d = _pml_profile(ny, pml_width, dx, c, pml_strength) if absorb else np.zeros(ny)
    sx = sx_1d[:, None] * np.ones((1, ny))
    sy = sy_1d[None, :] * np.ones((nx, 1))
    if source_loc is None:
        source_loc = (nx // 2, ny // 2)
    if source is None:
        t0, fpk = 0.06 * n_steps * dt, 8.0 / (n_steps * dt)

        def source(t):  # Ricker wavelet
            a = (np.pi * fpk * (t - t0)) ** 2
            return (1 - 2 * a) * np.exp(-a)

    px = np.zeros((nx, ny))
    py = np.zeros((nx, ny))
    vx = np.zeros((nx, ny))
    vy = np.zeros((nx, ny))
    energy = np.empty(n_steps)
    c2 = c * c
    for it in range(n_steps):
        p = px + py
        vx += dt * (-_ddx(p, dx) - sx * vx)
        vy += dt * (-_ddy(p, dx) - sy * vy)
        px += dt * (-c2 * _ddx(vx, dx) - sx * px)
        py += dt * (-c2 * _ddy(vy, dx) - sy * py)
        px[source_loc] += dt * source(it * dt)
        energy[it] = float(np.sum((px + py) ** 2) + np.sum(vx**2) + np.sum(vy**2))
    return px + py, energy
