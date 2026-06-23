"""2D acoustic wave equation for full-waveform-inversion-style inverse problems (phase 2 completion).

A turnkey forward for time-domain wave propagation, ``u_tt = c(x)^2 laplacian(u) + source(t)``, solved by an
explicit symplectic (leapfrog) step on the first-order system ``(u, w=u_t)``. An absorbing sponge layer near
the boundary damps outgoing waves so the finite domain does not reflect them back (the practical stand-in
for a perfectly-matched layer). Built on the checkpointed time integrator (Phase 2), so the gradient w.r.t.
the velocity field -- the full-waveform-inversion sensitivity -- is available at feasible memory.

The inverse problem -- recover the velocity field ``c(x)`` (or a localized perturbation) from waveforms
recorded at a few receivers -- is a :class:`~pysp.ppl.Differential` observation that integrates this stepper
and records the displacement at the receivers; fit it with ``how='gauss_newton'``.
"""

from __future__ import annotations

import numpy as np


class WaveEquation2D:
    """A differentiable 2D acoustic wave-equation stepper with an absorbing sponge boundary.

    ``WaveEquation2D(n, dt=..., spacing=..., absorb_width=..., absorb_strength=...)`` builds the forward on
    an ``n x n`` grid. The state is the packed ``(u, w=u_t)`` (``pack``/``displacement``); ``step(state,
    c2, ops, source=...)`` advances one leapfrog step given the squared-velocity field ``c2`` (a driver or
    a fixed array) and an optional per-node source term for that step.
    """

    def __init__(
        self,
        n: int,
        *,
        dt: float,
        spacing: float | None = None,
        absorb_width: int = 0,
        absorb_strength: float = 2.0,
    ):
        self.n = int(n)
        self.dt = float(dt)
        self.h = float(spacing) if spacing is not None else 1.0 / (n - 1)
        self._gamma = self._build_sponge(absorb_width, absorb_strength)

    def _build_sponge(self, width, strength):
        n = self.n
        gamma = np.zeros((n, n))
        if width > 0:
            idx = np.arange(n)
            d = np.minimum(idx, n - 1 - idx)  # distance (in nodes) to the nearest edge
            ramp = np.where(d < width, (1.0 - d / width) ** 2, 0.0)
            g = np.maximum(ramp[:, None], ramp[None, :])  # union of the two-axis ramps
            gamma = float(strength) * g
        return gamma.ravel()

    def pack(self, u, w):
        """Pack displacement ``u`` and velocity ``w = u_t`` into the integrator state."""
        import torch

        return torch.cat([torch.as_tensor(u), torch.as_tensor(w)])

    def displacement(self, state):
        """The displacement field ``u`` from a packed state."""
        return state[: self.n * self.n]

    def _lap(self, u, ops):
        n, h = self.n, self.h
        A = u.reshape(n, n)
        out = ops.zeros(n, n)
        out[1:-1, 1:-1] = (A[2:, 1:-1] + A[:-2, 1:-1] + A[1:-1, 2:] + A[1:-1, :-2] - 4 * A[1:-1, 1:-1]) / h**2
        return out.reshape(-1)

    def step(self, state, c2, ops, source=0.0):
        """Advance ``(u, w)`` one leapfrog step under ``u_tt = c2 * lap(u) + source - gamma * u_t``."""
        nn = self.n * self.n
        u, w = state[:nn], state[nn:]
        gamma = ops.tensor(self._gamma)
        u_next = u + self.dt * w
        accel = c2 * self._lap(u_next, ops) + source - gamma * w
        w_next = w + self.dt * accel
        return ops.cat([u_next, w_next])
