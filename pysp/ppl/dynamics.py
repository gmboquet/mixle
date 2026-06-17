"""Pluggable spatial dynamics operators for PDE-constrained models (method-of-lines).

A PDE for a field ``u(x, t)`` is turned into a finite set of coupled ODEs by discretizing the
spatial derivatives on a 1-D grid (the *method of lines*): ``du/dt = G u`` where ``G`` is the
discretized spatial operator (a Laplacian for diffusion, an upwind difference for advection,
...). Integrating one time step ``dt`` gives a linear state transition ``u_{t+1} = A u_t`` with
``A = transition_matrix(dt)`` -- exactly the transition of a multivariate linear-Gaussian state
space (see :mod:`pysp.ppl.pde`), so the existing Kalman/RTS/EM machinery applies unchanged.

Operators are pluggable via :func:`register_dynamics_operator` -- the same "register, don't
branch" pattern the compute engines and encoded-data backends use -- so a new PDE plugs in by
supplying its discretized ``operator_matrix`` without touching the solver.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


def laplacian_matrix(n: int, h: float, bc: str = "neumann") -> np.ndarray:
    """Second-difference Laplacian ``d^2/dx^2`` on ``n`` uniform points (spacing ``h``).

    ``bc`` selects the boundary condition: ``'dirichlet'`` (field pinned to 0 outside),
    ``'neumann'`` (zero-flux / reflecting), or ``'periodic'`` (wrap-around).
    """
    if n < 3:
        raise ValueError("laplacian_matrix needs at least 3 grid points.")
    lap = np.zeros((n, n), dtype=float)
    for i in range(n):
        lap[i, i] = -2.0
        if i > 0:
            lap[i, i - 1] = 1.0
        if i < n - 1:
            lap[i, i + 1] = 1.0
    if bc == "neumann":
        lap[0, 0] = -1.0  # zero-flux: u_{-1} = u_0
        lap[-1, -1] = -1.0
    elif bc == "periodic":
        lap[0, -1] = 1.0
        lap[-1, 0] = 1.0
    elif bc != "dirichlet":
        raise ValueError("bc must be 'dirichlet', 'neumann', or 'periodic'.")
    return lap / (h * h)


def upwind_gradient_matrix(n: int, h: float, velocity: float, bc: str = "periodic") -> np.ndarray:
    """First-order upwind difference for ``d/dx`` (sign chosen by ``velocity`` direction)."""
    grad = np.zeros((n, n), dtype=float)
    if velocity >= 0.0:
        for i in range(n):
            grad[i, i] = 1.0
            if i > 0:
                grad[i, i - 1] = -1.0
            elif bc == "periodic":
                grad[i, -1] = -1.0
    else:
        for i in range(n):
            grad[i, i] = -1.0
            if i < n - 1:
                grad[i, i + 1] = 1.0
            elif bc == "periodic":
                grad[i, 0] = 1.0
    return grad / h


def _matrix_exp(m: np.ndarray) -> np.ndarray:
    try:
        from scipy.linalg import expm

        return np.asarray(expm(m), dtype=float)
    except Exception:
        # Scaling-and-squaring fallback (no scipy): exp(m) = (exp(m/2^k))^(2^k) via Taylor.
        norm = float(np.max(np.sum(np.abs(m), axis=1)))
        k = max(0, int(np.ceil(np.log2(norm + 1.0))))
        a = m / (2.0**k)
        term = np.eye(m.shape[0])
        result = np.eye(m.shape[0])
        for j in range(1, 19):
            term = term @ a / j
            result = result + term
        for _ in range(k):
            result = result @ result
        return result


class DynamicsOperator(ABC):
    """A spatial operator ``G`` (method-of-lines) plus its time-step transition ``A``.

    Subclasses implement :meth:`operator_matrix` returning the ``(n, n)`` discretized spatial
    operator. :meth:`transition_matrix` integrates it over ``dt`` by the chosen ``scheme``:
    ``'implicit'`` Euler ``(I - dt G)^{-1}`` (unconditionally stable, the default),
    ``'explicit'`` Euler ``I + dt G`` (cheap; needs small ``dt``), or ``'exact'`` ``expm(dt G)``.
    """

    def __init__(self, n: int, length: float = 1.0, bc: str = "neumann", scheme: str = "implicit") -> None:
        if scheme not in ("implicit", "explicit", "exact"):
            raise ValueError("scheme must be 'implicit', 'explicit', or 'exact'.")
        self.n = int(n)
        self.length = float(length)
        self.bc = bc
        self.scheme = scheme
        self.h = self.length / (self.n - 1)
        self.grid = np.linspace(0.0, self.length, self.n)

    @abstractmethod
    def operator_matrix(self) -> np.ndarray:
        """Return the ``(n, n)`` discretized spatial operator ``G`` (``du/dt = G u``)."""

    def transition_matrix(self, dt: float) -> np.ndarray:
        """Return the one-step linear transition ``A`` such that ``u_{t+1} = A u_t``."""
        g = self.operator_matrix()
        if self.scheme == "explicit":
            return np.eye(self.n) + dt * g
        if self.scheme == "exact":
            return _matrix_exp(dt * g)
        return np.linalg.solve(np.eye(self.n) - dt * g, np.eye(self.n))  # implicit Euler


class DiffusionOperator(DynamicsOperator):
    """Heat / diffusion equation ``du/dt = D d^2u/dx^2`` (``D`` = diffusivity)."""

    def __init__(
        self, diffusivity: float, n: int, length: float = 1.0, bc: str = "neumann", scheme: str = "implicit"
    ) -> None:
        super().__init__(n=n, length=length, bc=bc, scheme=scheme)
        self.diffusivity = float(diffusivity)

    def operator_matrix(self) -> np.ndarray:
        return self.diffusivity * laplacian_matrix(self.n, self.h, self.bc)


class AdvectionOperator(DynamicsOperator):
    """Linear advection ``du/dt = -c du/dx`` (transport at velocity ``c``)."""

    def __init__(
        self, velocity: float, n: int, length: float = 1.0, bc: str = "periodic", scheme: str = "implicit"
    ) -> None:
        super().__init__(n=n, length=length, bc=bc, scheme=scheme)
        self.velocity = float(velocity)

    def operator_matrix(self) -> np.ndarray:
        return -self.velocity * upwind_gradient_matrix(self.n, self.h, self.velocity, self.bc)


class AdvectionDiffusionOperator(DynamicsOperator):
    """Advection-diffusion ``du/dt = D d^2u/dx^2 - c du/dx``."""

    def __init__(
        self,
        diffusivity: float,
        velocity: float,
        n: int,
        length: float = 1.0,
        bc: str = "periodic",
        scheme: str = "implicit",
    ) -> None:
        super().__init__(n=n, length=length, bc=bc, scheme=scheme)
        self.diffusivity = float(diffusivity)
        self.velocity = float(velocity)

    def operator_matrix(self) -> np.ndarray:
        diff = self.diffusivity * laplacian_matrix(self.n, self.h, self.bc)
        adv = -self.velocity * upwind_gradient_matrix(self.n, self.h, self.velocity, self.bc)
        return diff + adv


# --- operator registry ("register, don't branch") ---------------------------------------
_DYNAMICS_OPERATORS: dict[str, Any] = {}


def register_dynamics_operator(name: str, factory: Any) -> None:
    """Register a :class:`DynamicsOperator` factory under ``name`` for :func:`make_operator`."""
    if not callable(factory):
        raise TypeError("dynamics-operator factory must be callable.")
    _DYNAMICS_OPERATORS[name.lower()] = factory


def available_dynamics_operators() -> list[str]:
    """Return the sorted names of all registered dynamics operators."""
    return sorted(_DYNAMICS_OPERATORS)


def make_operator(name: str, **kwargs: Any) -> DynamicsOperator:
    """Construct a registered dynamics operator by ``name`` (see :func:`available_dynamics_operators`)."""
    factory = _DYNAMICS_OPERATORS.get(name.lower())
    if factory is None:
        raise ValueError(
            "unknown dynamics operator %r; registered: %s" % (name, ", ".join(available_dynamics_operators()))
        )
    return factory(**kwargs)


register_dynamics_operator("diffusion", DiffusionOperator)
register_dynamics_operator("advection", AdvectionOperator)
register_dynamics_operator("advection_diffusion", AdvectionDiffusionOperator)
