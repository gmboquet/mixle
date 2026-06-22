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


# ---------------------------------------------------------------------------
# Adaptive explicit ODE integrator (Dormand-Prince RK45)
# ---------------------------------------------------------------------------
# Butcher tableau for the Dormand-Prince 5(4) embedded pair (the method behind MATLAB ode45 /
# scipy RK45): a 5th-order solution with an embedded 4th-order estimate for adaptive step control.
_DP_C = (0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0)
_DP_A = (
    (),
    (1 / 5,),
    (3 / 40, 9 / 40),
    (44 / 45, -56 / 15, 32 / 9),
    (19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729),
    (9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656),
    (35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84),
)
_DP_B5 = np.array([35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0.0])
_DP_B4 = np.array([5179 / 57600, 0.0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40])


def integrate_adaptive(
    rhs: Any,
    y0: Any,
    t_eval: Any,
    *,
    t0: float = 0.0,
    rtol: float = 1.0e-7,
    atol: float = 1.0e-9,
    max_step_halving: int = 60,
) -> np.ndarray:
    """Integrate ``dy/dt = rhs(t, y)`` with an adaptive-step Dormand-Prince RK45 method.

    A 5th-order explicit solver with an embedded 4th-order error estimate that grows/shrinks the step
    to meet the ``rtol``/``atol`` tolerance, so smooth stretches take big steps and fast transients take
    small ones (the same adaptive method as ``scipy.integrate.solve_ivp(method="RK45")``). ``rhs(t, y)``
    returns the derivative (scalar or vector); ``t_eval`` is the increasing array of output times (the
    last is the final time). Returns an array of shape ``(len(t_eval), len(y0))`` of the state at each
    requested time -- each output is produced by a single high-order step from the last accepted point,
    so it is consistent with the adaptive trajectory. Unlike :meth:`Ops.integrate` (fixed-step, engine
    differentiable for adjoints), this is a NumPy accuracy-focused forward integrator.
    """
    f = lambda t, y: np.atleast_1d(np.asarray(rhs(t, y), dtype=np.float64))  # noqa: E731
    y = np.atleast_1d(np.asarray(y0, dtype=np.float64)).copy()
    times = np.asarray(t_eval, dtype=np.float64)
    tf = float(times[-1])
    t = float(t0)
    h = (tf - t) / 100.0 if tf > t else 1.0e-3

    def step(t: float, y: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray]:
        k: list[np.ndarray] = []
        for i in range(7):
            yi = y + h * sum(_DP_A[i][j] * k[j] for j in range(len(_DP_A[i])))
            k.append(f(t + _DP_C[i] * h, yi))
        kk = np.array(k)
        return y + h * (_DP_B5 @ kk), y + h * (_DP_B4 @ kk)

    out: list[np.ndarray] = []
    idx = 0
    while idx < len(times):
        hh = min(h, tf - t)
        y5, y4 = step(t, y, hh)
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y5))
        err = float(np.max(np.abs(y5 - y4) / scale))
        tiny = hh <= (tf - t0) * 2.0**-max_step_halving
        if err <= 1.0 or tiny:
            t_new = t + hh
            while idx < len(times) and times[idx] <= t_new + 1.0e-12:
                ys, _ = step(t, y, times[idx] - t)  # one high-order step to the exact output time
                out.append(ys)
                idx += 1
            t, y = t_new, y5
            h = hh * (5.0 if err == 0.0 else min(5.0, 0.9 * err ** (-0.2)))
        else:
            h = hh * max(0.2, 0.9 * err ** (-0.2))
    return np.array(out)


# ---------------------------------------------------------------------------
# Implicit stiff ODE integrator (L-stable SDIRK2)
# ---------------------------------------------------------------------------
_SDIRK2_GAMMA = 1.0 - 1.0 / np.sqrt(2.0)  # the L-stable 2nd-order singly-diagonally-implicit choice


def integrate_stiff(
    rhs: Any,
    y0: Any,
    t_eval: Any,
    *,
    t0: float = 0.0,
    jac: Any = None,
    h_max: float = 0.05,
    newton_tol: float = 1.0e-11,
    max_newton: int = 50,
) -> np.ndarray:
    """Integrate a STIFF system ``dy/dt = rhs(t, y)`` with the L-stable 2nd-order SDIRK2 method.

    Stiff problems (widely separated time scales) make explicit solvers like :func:`integrate_adaptive`
    take impractically tiny steps; this two-stage singly-diagonally-implicit Runge-Kutta method
    (``gamma = 1 - 1/sqrt(2)``) is **L-stable**, so it damps the fast modes correctly at any step size
    while staying 2nd-order accurate on the slow ones. Each stage solves ``k = rhs(t, base + h*gamma*k)``
    by Newton iteration using the Jacobian ``jac(t, y)`` (a finite-difference Jacobian is used when
    ``jac`` is ``None``). ``t_eval`` is the increasing array of output times; each interval is covered by
    substeps capped at ``h_max``. Returns the state at each output time, shape ``(len(t_eval), len(y0))``.
    """
    f = lambda t, y: np.atleast_1d(np.asarray(rhs(t, y), dtype=np.float64))  # noqa: E731
    y = np.atleast_1d(np.asarray(y0, dtype=np.float64)).copy()
    n = y.size
    times = np.asarray(t_eval, dtype=np.float64)
    t = float(t0)
    g = _SDIRK2_GAMMA

    def jacobian(tc: float, yc: np.ndarray) -> np.ndarray:
        if jac is not None:
            return np.atleast_2d(np.asarray(jac(tc, yc), dtype=np.float64))
        eps = 1.0e-7
        f0 = f(tc, yc)
        out = np.empty((n, n))
        for i in range(n):
            yp = yc.copy()
            yp[i] += eps
            out[:, i] = (f(tc, yp) - f0) / eps
        return out

    def solve_stage(tc: float, base: np.ndarray, h: float) -> np.ndarray:
        k = f(tc, base)  # explicit guess
        for _ in range(max_newton):
            resid = k - f(tc, base + h * g * k)
            mat = np.eye(n) - h * g * jacobian(tc, base + h * g * k)
            dk = np.linalg.solve(mat, -resid)
            k = k + dk
            if np.max(np.abs(dk)) < newton_tol:
                break
        return k

    result: list[np.ndarray] = []
    for t_next in times:
        nsub = max(1, int(np.ceil((t_next - t) / h_max)))
        h = (t_next - t) / nsub
        for _ in range(nsub):
            k1 = solve_stage(t + g * h, y, h)
            k2 = solve_stage(t + h, y + h * (1.0 - g) * k1, h)
            y = y + h * ((1.0 - g) * k1 + g * k2)
            t = t + h
        result.append(y.copy())
    return np.array(result)
