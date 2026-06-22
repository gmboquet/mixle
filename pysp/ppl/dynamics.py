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


# ---------------------------------------------------------------------------
# Forward sensitivity of an ODE solution to its parameters
# ---------------------------------------------------------------------------
def integrate_sensitivity(
    rhs: Any,
    y0: Any,
    t_eval: Any,
    params: Any,
    *,
    t0: float = 0.0,
    rtol: float = 1.0e-9,
    atol: float = 1.0e-11,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate ``dy/dt = rhs(t, y, p)`` together with the forward sensitivities ``S = dy/dp``.

    Returns ``(Y, S)`` where ``Y`` has shape ``(len(t_eval), n)`` (the trajectory) and ``S`` has shape
    ``(len(t_eval), n, n_params)`` with ``S[k, i, j] = d y_i(t_eval[k]) / d p_j``. The sensitivity obeys
    the variational equation ``dS/dt = (df/dy) S + (df/dp)`` (with ``S(t0) = 0`` since the initial state
    is parameter-independent); the augmented ``[y, S]`` system is solved with the adaptive RK45 of
    :func:`integrate_adaptive`, and ``df/dy``/``df/dp`` are obtained by finite differences. This is the
    forward-mode answer to "how does the solution move as I perturb the parameters" -- gradients for
    calibration, design, and optimal-control without re-running the solve per parameter.
    """
    y_init = np.atleast_1d(np.asarray(y0, dtype=np.float64))
    n = y_init.size
    p = np.atleast_1d(np.asarray(params, dtype=np.float64))
    n_par = p.size
    eps = 1.0e-7

    def aug(t: float, z: np.ndarray) -> np.ndarray:
        y = z[:n]
        s = z[n:].reshape(n, n_par)
        f0 = np.atleast_1d(np.asarray(rhs(t, y, p), dtype=np.float64))
        jy = np.empty((n, n))
        for i in range(n):
            yp = y.copy()
            yp[i] += eps
            jy[:, i] = (np.atleast_1d(np.asarray(rhs(t, yp, p), dtype=np.float64)) - f0) / eps
        jp = np.empty((n, n_par))
        for j in range(n_par):
            pp = p.copy()
            pp[j] += eps
            jp[:, j] = (np.atleast_1d(np.asarray(rhs(t, y, pp), dtype=np.float64)) - f0) / eps
        return np.concatenate([f0, (jy @ s + jp).ravel()])

    z0 = np.concatenate([y_init, np.zeros(n * n_par)])
    z = integrate_adaptive(aug, z0, t_eval, t0=t0, rtol=rtol, atol=atol)
    times = np.asarray(t_eval, dtype=np.float64)
    return z[:, :n], z[:, n:].reshape(len(times), n, n_par)


# ---------------------------------------------------------------------------
# Differential-algebraic equations (semi-explicit index-1, mass-matrix form)
# ---------------------------------------------------------------------------
def integrate_dae(
    rhs: Any,
    y0: Any,
    t_eval: Any,
    mass: Any,
    *,
    t0: float = 0.0,
    jac: Any = None,
    h_max: float = 0.02,
    newton_tol: float = 1.0e-12,
    max_newton: int = 60,
) -> np.ndarray:
    """Integrate a mass-matrix DAE/ODE ``M y' = rhs(t, y)`` with the L-stable SDIRK2 method.

    Generalizes :func:`integrate_stiff` to a constant (possibly **singular**) mass matrix ``M``: rows
    where ``M`` is zero are algebraic constraints ``0 = rhs_row(t, y)``, so this solves semi-explicit
    index-1 differential-algebraic equations (and ordinary stiff ODEs when ``M`` is invertible). Each
    SDIRK stage solves ``M k = rhs(t, base + h*gamma*k)`` for the stage slope ``k`` by Newton, using the
    linearization ``(M - h*gamma*df/dy)``. The initial condition must be consistent (satisfy the
    algebraic constraints). Returns the state at each ``t_eval`` time, shape ``(len(t_eval), len(y0))``.
    """
    f = lambda t, y: np.atleast_1d(np.asarray(rhs(t, y), dtype=np.float64))  # noqa: E731
    m = np.asarray(mass, dtype=np.float64)
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
        k = np.linalg.lstsq(m, f(tc, base), rcond=None)[0]  # consistent initial guess (handles singular M)
        for _ in range(max_newton):
            resid = m @ k - f(tc, base + h * g * k)
            mat = m - h * g * jacobian(tc, base + h * g * k)
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


# ---------------------------------------------------------------------------
# Nonlinear PDE right-hand sides (method of lines) -- Burgers' equation
# ---------------------------------------------------------------------------
def burgers_rhs(nu: float, dx: float, *, bc: str = "dirichlet") -> Any:
    """Build the method-of-lines right-hand side of the viscous Burgers equation.

    Returns a callable ``rhs(t, u)`` giving ``du/dt = -u u_x + nu u_xx`` on a uniform 1-D grid of spacing
    ``dx`` (central differences for both the nonlinear advection and the viscosity ``nu``), to be passed
    to :func:`integrate_adaptive` (or :func:`integrate_stiff` for small ``nu``). ``bc="dirichlet"`` holds
    the two endpoints fixed (their initial values); ``bc="periodic"`` wraps the grid. Burgers is the
    canonical nonlinear convection-diffusion test -- it forms and smears shocks -- and admits the exact
    travelling wave ``u = (uL+uR)/2 - (uL-uR)/2 tanh((uL-uR)(x - s t)/(4 nu))`` with ``s = (uL+uR)/2``.
    """
    nu = float(nu)
    dx = float(dx)
    if bc not in ("dirichlet", "periodic"):
        raise ValueError("bc must be 'dirichlet' or 'periodic'.")

    def rhs(t: float, u: Any) -> np.ndarray:
        u = np.asarray(u, dtype=np.float64)
        flux = 0.5 * u * u  # conservative form: -(u^2/2)_x telescopes -> discrete mass is conserved
        if bc == "periodic":
            dflux = (np.roll(flux, -1) - np.roll(flux, 1)) / (2.0 * dx)
            uxx = (np.roll(u, -1) - 2.0 * u + np.roll(u, 1)) / (dx * dx)
            return -dflux + nu * uxx
        du = np.zeros_like(u)
        dflux = (flux[2:] - flux[:-2]) / (2.0 * dx)
        uxx = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / (dx * dx)
        du[1:-1] = -dflux + nu * uxx
        return du

    return rhs
