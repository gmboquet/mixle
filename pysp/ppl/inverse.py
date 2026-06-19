"""Differential-equation forward models for Bayesian inverse problems.

Many quantities are observable only through a dynamical system they drive: a rate constant through a
decay curve, a contaminant source through downstream concentrations, an initial state through a later
trajectory, a diffusivity through a steady temperature field. The inverse problem is to recover a
posterior over those hidden drivers from noisy, partial, indirect observations of the system's output.

`Differential` is an observation whose forward model is the solution of an ODE or PDE. It attaches to the
field surface like any other observation: the latent drivers are ``free`` handles (``free(1,
name="k", support="positive")``) or the shared field a ``GP`` carries, and a single callback supplies the
physics. The callback is handed a ``p`` namespace (drivers by name: ``p.k``, ``p.field``) and an ``ops``
namespace (backend-agnostic math + grid assembly + the adjoint sparse solve), so it never imports a tensor
library. Fit with ``joint([...]).fit(how=...)`` and read a posterior off any node.

Example -- recover a decay rate from a noisy decay curve (no shared field)::

    k = free(1, name="k", support="positive")
    obs = Differential(y_obs, drivers=[k], y0=1.0, t_grid=t, scale=0.05,
                       rhs=lambda u, t, p, ops: -p.k * u)        # dy/dt = -k y
    post = joint([obs]).fit(how="map")
    k_mean, k_sd = post.posterior("k")

Example -- recover a source field from a steady diffusion equation (the shared field is the source)::

    q = GP("q", index=coords, kernel=RandomWalk(scale=0.3, ridge=5.0))
    obs = Differential(u_sensors, over=q, scale=0.02, observe=lambda u, p, ops: u[sensor_idx],
                       forward=lambda p, ops: ops.sparse_solve(*ops.divergence_form(p.field, shape), b))
    post = joint([obs]).fit(how="map")
    q_mean, q_sd = post.posterior("q")
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from pysp.ppl.field import Proxy, _ParamSpec
from pysp.ppl.ops import _Params, make_ops

__all__ = ["Differential"]


def _driver_spec(handle):
    """Read ``(name, support, dim)`` off a ``free(...)`` handle used as a latent driver."""
    if getattr(handle, "_kind", None) != "param":
        raise TypeError("a driver must be a free(...) handle, e.g. free(1, name='k', support='positive').")
    spec = handle._args[0]
    name = getattr(spec, "name", None) or getattr(handle, "_name", None)
    if name is None:
        raise ValueError("drivers must be named, e.g. free(1, name='k').")
    return name, getattr(spec, "support", "real"), int(getattr(spec, "dim", 1))


class _DifferentialProxy(Proxy):
    """Internal proxy: solve a forward model from the drivers/field and score the observed output."""

    def __init__(self, y, *, forward, observe, drivers, over_name, scale, family, prefix="diff"):
        self.y = np.asarray(y, dtype=float)
        self.forward = forward
        self.observe = observe
        self.drivers = drivers  # list of (name, support, dim)
        self.over_name = over_name
        self.family = family
        self.prefix = prefix
        from pysp.ppl.core import _is_free

        if family == "gaussian" and _is_free(scale):
            self._scale_name = f"{prefix}.scale"
            self._scale_fixed = None
        else:
            self._scale_name = None
            self._scale_fixed = float(scale) if family == "gaussian" else None

    def params(self) -> list[_ParamSpec]:
        specs = [
            _ParamSpec(name, (dim,) if dim > 1 else (), support, np.zeros(dim)) for name, support, dim in self.drivers
        ]
        if self._scale_name is not None:
            specs.append(_ParamSpec(self._scale_name, (), "positive", np.array(0.0)))
        return specs

    def loglik(self, field_t, params, torch):
        ops = make_ops()
        values = {name: params[name] for name, _, _ in self.drivers}
        if self.over_name is not None:
            values["field"] = field_t
        p = _Params(values)
        solution = self.forward(p, ops)
        pred = self.observe(solution, p, ops) if self.observe is not None else solution
        y = torch.as_tensor(self.y)
        if self.family == "poisson":
            rate = torch.clamp(pred, min=1e-12)
            return torch.sum(y * torch.log(rate) - rate)
        scale = params[self._scale_name] if self._scale_name is not None else self._scale_fixed
        resid = (y - pred) / scale
        log_scale = torch.log(scale) if torch.is_tensor(scale) else float(np.log(scale))
        return -0.5 * torch.sum(resid * resid) - y.numel() * (log_scale + 0.5 * np.log(2 * np.pi))

    def residual(self, field_t, params, torch):
        if self.family != "gaussian":
            return None
        ops = make_ops()
        values = {name: params[name] for name, _, _ in self.drivers}
        if self.over_name is not None:
            values["field"] = field_t
        p = _Params(values)
        solution = self.forward(p, ops)
        pred = self.observe(solution, p, ops) if self.observe is not None else solution
        scale = params[self._scale_name] if self._scale_name is not None else self._scale_fixed
        return (torch.as_tensor(self.y) - pred) / scale


def Differential(
    y: np.ndarray,
    *,
    forward: Callable | None = None,
    rhs: Callable | None = None,
    y0: Any = None,
    t_grid: np.ndarray | None = None,
    method: str = "rk4",
    observe: Callable | None = None,
    drivers: Sequence = (),
    over=None,
    scale: Any = 1.0,
    family: str = "gaussian",
) -> tuple:
    """An observation whose forward model is an ODE/PDE solve; recovers a posterior over the drivers.

    Provide either ``forward(p, ops) -> solution`` (general: any solve, e.g. ``ops.sparse_solve`` of a
    ``ops.divergence_form`` operator), or an initial-value problem via ``rhs(u, t, p, ops)`` plus ``y0``
    and ``t_grid`` (the framework integrates it). ``observe(solution, p, ops) -> predicted`` maps the
    solution to the observed quantities (default: the whole solution). ``drivers`` are ``free(...)``
    handles (coefficients, initial conditions); ``over`` is a ``GP`` whose field is a driver (a source
    term, a spatially varying coefficient) exposed as ``p.field``. ``scale`` is the Gaussian noise level
    (fixed or ``free``); ``family`` is ``'gaussian'`` or ``'poisson'``. Returns the ``(field, proxy)`` pair
    consumed by :func:`joint`.
    """
    if family not in ("gaussian", "poisson"):
        raise ValueError("family must be 'gaussian' or 'poisson'.")
    if forward is None and rhs is None:
        raise ValueError("provide forward(p, ops) or an initial-value problem via rhs/y0/t_grid.")
    if rhs is not None and (y0 is None or t_grid is None):
        raise ValueError("an initial-value rhs needs y0 and t_grid.")

    from pysp.ppl.field import GaussianField

    field = None
    over_name = None
    if over is not None:
        field = over.field if hasattr(over, "field") else over
        if not isinstance(field, GaussianField):
            raise TypeError("over must be a GP (or GaussianField) carrying the shared latent field.")
        over_name = field.name

    if forward is None:
        tg = np.asarray(t_grid, dtype=float)

        def forward(p, ops, _rhs=rhs, _y0=y0, _tg=tg, _m=method):
            y0v = _y0(p, ops) if callable(_y0) else ops.tensor(np.asarray(_y0, dtype=float))
            return ops.integrate(lambda u, t: _rhs(u, t, p, ops), y0v, _tg, method=_m)

    drivers_spec = [_driver_spec(h) for h in drivers]
    proxy = _DifferentialProxy(
        y, forward=forward, observe=observe, drivers=drivers_spec, over_name=over_name, scale=scale, family=family
    )
    return field, proxy
