"""Differential-equation forward models for Bayesian inverse problems.

Many quantities are observable only through a dynamical system they drive: a rate constant through a
decay curve, a contaminant source through downstream concentrations, an initial state through a later
trajectory, a diffusivity through a steady temperature field. The inverse problem is to recover a
posterior over those hidden drivers from noisy, partial, indirect observations of the system's output.

`DifferentialProxy` is a :class:`pysp.ppl.field.Proxy` whose forward model is the solution of an ODE or
PDE. The latent drivers are declared as parameters (coefficients, initial conditions, noise scale) or
supplied by the shared field of :func:`pysp.ppl.fit_field` (a source term, a spatially varying
coefficient, an initial-condition profile), so the joint fit returns a posterior over any of them. The
solve is differentiable (a torch Runge-Kutta / Euler integrator for initial-value problems, or a linear
solve for steady-state problems), so gradients flow to the drivers for MAP and Laplace inference.

This is the inverse-problem / data-assimilation pattern, and it is domain-agnostic: pharmacokinetics,
epidemiology (SIR/SEIR), chemical kinetics, predator-prey, heat-equation source recovery, groundwater and
contaminant transport, tomography. The specific equation is the user's; the inference is general.

Example -- recover a decay rate from a noisy decay curve (no shared field)::

    t = np.linspace(0, 5, 40)
    proxy = DifferentialProxy(
        y_obs, t_grid=t, y0=1.0, scale=0.05,
        rhs=lambda state, t, th, torch: -th["k"] * state,   # dy/dt = -k y
        params=[("k", "positive", 0.5)],
    )
    post = fit_field(None, [proxy], how="laplace")
    k_mean, k_sd = post.posterior("ode.k")

Example -- recover a source field from a steady diffusion equation (the shared field is the source)::

    proxy = DifferentialProxy(
        u_sensors, solver="linear_steady", uses_field=True, scale=0.02,
        operator=lambda th, torch: D * Laplacian_t,                 # -D u'' = q
        source=lambda th, torch: th["field"],                       # q is the latent field
        observe=lambda u, th, torch: u[sensor_idx],
    )
    post = fit_field(source_field, [proxy], how="laplace")          # GaussianField over space
    q_mean, q_sd = post.posterior(source_field.name)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from pysp.ppl.field import Proxy, _ParamSpec

__all__ = ["DifferentialProxy", "integrate_ode"]


def integrate_ode(rhs, y0, t_grid, theta, torch, method: str = "rk4"):
    """Integrate ``d state/dt = rhs(state, t, theta, torch)`` from ``y0`` over ``t_grid`` (differentiable).

    ``rk4`` is fixed-step classical Runge-Kutta; ``euler`` is explicit Euler. Returns the stacked
    trajectory with shape ``(len(t_grid),) + y0.shape``. Steps follow the spacing of ``t_grid`` (use a
    fine grid for stiff or fast dynamics).
    """
    y = y0
    states = [y]
    for i in range(len(t_grid) - 1):
        t = t_grid[i]
        h = t_grid[i + 1] - t_grid[i]
        if method == "euler":
            y = y + h * rhs(y, t, theta, torch)
        elif method == "rk4":
            k1 = rhs(y, t, theta, torch)
            k2 = rhs(y + 0.5 * h * k1, t + 0.5 * h, theta, torch)
            k3 = rhs(y + 0.5 * h * k2, t + 0.5 * h, theta, torch)
            k4 = rhs(y + h * k3, t + h, theta, torch)
            y = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            raise ValueError(f"unknown method {method!r}; use 'rk4' or 'euler'.")
        states.append(y)
    return torch.stack(states)


class DifferentialProxy(Proxy):
    """An ODE/PDE forward-model likelihood: observe data downstream of a differential equation, infer the drivers.

    Parameters
    ----------
    y : array
        The observations to compare against the (operator-applied) solution.
    rhs : callable ``(state, t, theta, torch) -> d state/dt``
        The right-hand side, for an initial-value problem (``solver='rk4'`` or ``'euler'``). For a PDE,
        this is the method-of-lines spatial discretization.
    operator, source : callables ``(theta, torch) -> matrix`` / ``-> vector``
        For ``solver='linear_steady'``: solve ``operator(theta) @ u = source(theta)`` for the steady state.
    y0 : array or callable ``(theta, torch) -> state``
        Initial condition for an IVP; a callable lets the initial state itself be a latent driver.
    t_grid : array
        Integration grid for an IVP.
    observe : callable ``(trajectory, theta, torch) -> predicted`` or None
        Maps the solution to the observed quantities (a sensor/sampling operator). Defaults to the full
        solution (it must then match ``y``'s shape).
    scale : float or the ``free`` token
        Observation-noise scale (Gaussian) -- fixed (calibrated) or estimated.
    family : {'gaussian', 'poisson'}
        Observation model. ``'poisson'`` treats the prediction as a rate (counts).
    params : sequence of ``(name, support, init)``
        The latent drivers carried by this proxy (coefficients, initial conditions): ``support`` is
        ``'real'`` or ``'positive'``, ``init`` sets the shape.
    uses_field : bool
        If True, the shared field of :func:`fit_field` is passed as ``theta['field']`` (e.g. a source
        term or spatially varying coefficient), so its posterior is inferred jointly.
    """

    def __init__(
        self,
        y: np.ndarray,
        *,
        rhs: Callable | None = None,
        operator: Callable | None = None,
        source: Callable | None = None,
        y0: Any = None,
        t_grid: np.ndarray | None = None,
        observe: Callable | None = None,
        scale: Any = 1.0,
        family: str = "gaussian",
        solver: str = "rk4",
        params: Sequence[tuple] = (),
        uses_field: bool = False,
        prefix: str = "ode",
    ):
        if solver in ("rk4", "euler") and (rhs is None or t_grid is None or y0 is None):
            raise ValueError("an initial-value solver needs rhs, t_grid and y0.")
        if solver == "linear_steady" and (operator is None or source is None):
            raise ValueError("solver='linear_steady' needs operator and source.")
        if family not in ("gaussian", "poisson"):
            raise ValueError("family must be 'gaussian' or 'poisson'.")
        self.y = np.asarray(y, dtype=float)
        self.rhs = rhs
        self.operator = operator
        self.source = source
        self.y0 = y0
        self.t_grid = None if t_grid is None else np.asarray(t_grid, dtype=float)
        self.observe = observe
        self.family = family
        self.solver = solver
        self.uses_field = uses_field
        self.prefix = prefix
        self.param_specs = list(params)
        # the noise scale is a fixed float or a free positive parameter
        from pysp.ppl.core import _is_free

        if family == "gaussian" and _is_free(scale):
            self._scale_name = f"{prefix}.scale"
            self._scale_fixed = None
        else:
            self._scale_name = None
            self._scale_fixed = float(scale) if family == "gaussian" else None

    def params(self) -> list[_ParamSpec]:
        specs = []
        for name, support, init in self.param_specs:
            arr = np.asarray(init, dtype=float)
            specs.append(_ParamSpec(f"{self.prefix}.{name}", arr.shape, support, arr))
        if self._scale_name is not None:
            specs.append(_ParamSpec(self._scale_name, (), "positive", np.array(float(np.std(self.y)) or 1.0)))
        return specs

    def _solve(self, theta, torch):
        if self.solver == "linear_steady":
            return torch.linalg.solve(self.operator(theta, torch), self.source(theta, torch))
        y0 = self.y0(theta, torch) if callable(self.y0) else torch.as_tensor(np.asarray(self.y0, dtype=float))
        return integrate_ode(self.rhs, y0, torch.as_tensor(self.t_grid), theta, torch, self.solver)

    def loglik(self, field_t, params, torch):
        theta = {name: params[f"{self.prefix}.{name}"] for name, _, _ in self.param_specs}
        if self.uses_field:
            theta["field"] = field_t
        traj = self._solve(theta, torch)
        pred = self.observe(traj, theta, torch) if self.observe is not None else traj
        y = torch.as_tensor(self.y)
        if self.family == "poisson":
            rate = torch.clamp(pred, min=1e-12)
            return torch.sum(y * torch.log(rate) - rate)
        scale = params[self._scale_name] if self._scale_name is not None else self._scale_fixed
        resid = (y - pred) / scale
        log_scale = torch.log(scale) if torch.is_tensor(scale) else float(np.log(scale))
        return -0.5 * torch.sum(resid * resid) - y.numel() * (log_scale + 0.5 * np.log(2 * np.pi))
