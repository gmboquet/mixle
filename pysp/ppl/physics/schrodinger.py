"""Time-dependent Schrodinger equation -- the split-step Fourier (Strang) propagator.

Evolves a complex wavefunction under the time-dependent Schrodinger equation on a periodic box (1-D, 2-D
or 3-D):

    i hbar d psi/dt = -(hbar^2 / 2 m) laplacian(psi) + V(x) psi.

The split-step Fourier method applies a Strang splitting of the kinetic and potential operators: a
half-step of the potential (a pointwise phase ``exp(-i V dt / 2 hbar)``), a full kinetic step applied
exactly in Fourier space (``exp(-i hbar |k|^2 dt / 2 m)``), and a second potential half-step. Each factor
is unitary, so the scheme conserves the norm to machine precision and is second-order accurate in ``dt``.
It reproduces the exact quantum benchmarks: a harmonic-oscillator eigenstate is stationary in ``|psi|^2``
and conserves energy, and a free Gaussian wave packet spreads at the exact rate
``sigma(t) = sigma_0 sqrt(1 + (hbar t / 2 m sigma_0^2)^2)``.

Reference: Feit, Fleck & Steiger, "Solution of the Schrodinger equation by a spectral method", *J.
Computational Physics* 47 (1982); Bandrauk & Shen (1993).
"""

import math
from typing import Any

import numpy as np


def schrodinger_split_step(
    psi: Any,
    potential: Any,
    dt: float,
    n_steps: int,
    *,
    length: float = 2.0 * math.pi,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> np.ndarray:
    """Propagate a wavefunction with the split-step Fourier method (unitary, 2nd-order in ``dt``).

    Args:
        psi: initial complex wavefunction on a uniform periodic grid (shape ``(N,)*d``, ``d`` in 1..3).
        potential: real potential ``V(x)`` of the same shape (static).
        dt: time step.
        n_steps: number of steps.
        length: periodic box side length.
        mass: particle mass ``m``.
        hbar: reduced Planck constant.

    Returns:
        The evolved complex wavefunction (same shape).
    """
    wave = np.asarray(psi, dtype=np.complex128)
    v = np.asarray(potential, dtype=np.float64)
    if wave.shape != v.shape:
        raise ValueError("psi and potential must have the same shape.")
    d = wave.ndim
    if d not in (1, 2, 3):
        raise ValueError("schrodinger_split_step supports 1-, 2- or 3-D grids.")
    ks = [2.0 * np.pi * np.fft.fftfreq(n, d=float(length) / n) for n in wave.shape]
    k2 = sum(k * k for k in np.meshgrid(*ks, indexing="ij"))
    half_potential = np.exp(-0.5j * v * dt / hbar)
    kinetic = np.exp(-0.5j * hbar * k2 * dt / mass)
    fftn, ifftn = np.fft.fftn, np.fft.ifftn

    for _ in range(int(n_steps)):
        wave = half_potential * wave
        wave = ifftn(kinetic * fftn(wave))
        wave = half_potential * wave
    return wave


def probability_density(psi: Any) -> np.ndarray:
    """Return ``|psi|^2``, the Born-rule probability density of a wavefunction."""
    w = np.asarray(psi, dtype=np.complex128)
    return np.abs(w) ** 2


def norm(psi: Any, length: float = 2.0 * math.pi) -> float:
    """Return the squared norm ``integral |psi|^2 dV`` (conserved exactly by the unitary propagator)."""
    w = np.asarray(psi, dtype=np.complex128)
    cell = (float(length) / w.shape[0]) ** w.ndim
    return float(np.sum(np.abs(w) ** 2)) * cell
