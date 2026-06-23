"""PDE-constrained state-space models for pysp.ppl (multivariate Kalman/RTS + EM).

A latent spatial field ``u_t in R^n`` evolves by a PDE discretized via the method of lines
(see :mod:`pysp.ppl.dynamics`), giving a linear transition ``u_t = A u_{t-1} + w_t`` with
``A`` fixed by the physics and process noise ``w ~ N(0, q I)``. Noisy observations
``y_t = H u_t + v_t`` (``v ~ N(0, r I)``; ``H`` an optional sensor/sampling operator, identity
by default) are assimilated by a multivariate Kalman filter + RTS smoother. Fitting is EM: the
E-step is the smoother, the M-step updates the scalar noise levels ``q`` and ``r`` while the
PDE-determined transition ``A`` is held fixed (the "PDE constraint").

This is the multivariate generalization of :mod:`pysp.ppl.statespace`; it reuses the same
filter/smoother/EM structure with vector states and a physics-derived transition matrix.
"""

from __future__ import annotations

import math

import numpy as np

from pysp.ppl.core import RandomVariable
from pysp.ppl.physics.dynamics import DynamicsOperator


class PDEStateSpaceResult:
    """Smoothed latent field, fitted noise levels, and forecasts for a PDE state-space fit."""

    def __init__(
        self,
        operator: DynamicsOperator,
        dt: float,
        q: float,
        r: float,
        A: np.ndarray,
        H: np.ndarray,
        smoothed: np.ndarray,
        smoothed_var: np.ndarray,
        loglik: float,
    ) -> None:
        self.operator = operator
        self.dt = float(dt)
        self.process_var = float(q)
        self.obs_var = float(r)
        self.process_sd = float(math.sqrt(q))
        self.obs_sd = float(math.sqrt(r))
        self.transition = np.asarray(A)
        self.observation = np.asarray(H)
        self.smoothed = np.asarray(smoothed)  # (T, n) E[u_t | y_{1:T}]
        self.smoothed_sd = np.sqrt(np.clip(np.asarray(smoothed_var), 0.0, None))  # (T, n) marginal sd
        self.loglik = float(loglik)
        self.acceptance_rate = None
        self.predictive = None
        self.coefficients = {"process_sd": self.process_sd, "obs_sd": self.obs_sd, "dt": self.dt}

    def forecast(self, h: int) -> np.ndarray:
        """Forecast the field ``h`` steps ahead from the last smoothed state (mean dynamics)."""
        u = self.smoothed[-1].copy()
        out = []
        for _ in range(h):
            u = self.transition @ u
            out.append(u.copy())
        return np.asarray(out)

    def summary(self) -> dict:
        return {"process_sd": self.process_sd, "obs_sd": self.obs_sd, "dt": self.dt, "loglik": self.loglik}


def _observation_matrix(H, n: int) -> np.ndarray:
    if H is None:
        return np.eye(n)
    H = np.asarray(H, dtype=float)
    if H.ndim != 2 or H.shape[1] != n:
        raise ValueError("observation operator H must be (m, n) with n matching the grid size.")
    return H


def kalman_rts_em(
    observations: np.ndarray,
    operator: DynamicsOperator,
    dt: float = 1.0,
    H=None,
    max_its: int = 100,
    tol: float = 1e-6,
) -> PDEStateSpaceResult:
    """Fit scalar process/observation noise of a PDE state space by EM; smooth the latent field.

    Args:
        observations: ``(T, m)`` array of noisy field observations (``m == n`` for full
            observation, fewer for sparse sensors paired with an ``H`` operator).
        operator: a :class:`DynamicsOperator` supplying the transition ``A = operator.transition_matrix(dt)``.
        dt: time step between consecutive observations.
        H: optional ``(m, n)`` observation operator (identity when ``None``).
        max_its, tol: EM iteration controls (stop on log-likelihood change below ``tol``).
    """
    y = np.asarray(observations, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    T, m = y.shape
    n = operator.n
    A = operator.transition_matrix(dt)
    H = _observation_matrix(H, n)
    if H.shape[0] != m:
        raise ValueError("observation width %d does not match H rows %d." % (m, H.shape[0]))

    eye_n = np.eye(n)
    # Initial state: least-squares back-project the first observation; diffuse prior.
    x0 = np.linalg.lstsq(H, y[0], rcond=None)[0]
    scale = max(float(np.var(y)), 1e-6)
    P0 = scale * eye_n
    q = 0.1 * scale
    r = 0.5 * scale
    prev_ll = None
    xs = np.tile(x0, (T, 1))
    Ps = np.tile(P0, (T, 1, 1))

    for _ in range(max_its):
        Q = q * eye_n
        R = r * np.eye(m)
        xp = np.empty((T, n))
        Pp = np.empty((T, n, n))
        xf = np.empty((T, n))
        Pf = np.empty((T, n, n))
        xprev, Pprev = x0, P0
        ll = 0.0
        for t in range(T):
            xpr = A @ xprev
            Ppr = A @ Pprev @ A.T + Q
            S = H @ Ppr @ H.T + R
            Sinv = np.linalg.inv(S)
            K = Ppr @ H.T @ Sinv
            innov = y[t] - H @ xpr
            xf[t] = xpr + K @ innov
            Pf[t] = (eye_n - K @ H) @ Ppr
            xp[t], Pp[t] = xpr, Ppr
            sign, logdet = np.linalg.slogdet(S)
            ll += -0.5 * (m * math.log(2.0 * math.pi) + logdet + float(innov @ Sinv @ innov))
            xprev, Pprev = xf[t], Pf[t]

        # RTS smoother (with lag-one smoothed cross-covariance for the M-step).
        xs = np.empty((T, n))
        Ps = np.empty((T, n, n))
        Pcov = np.zeros((T, n, n))
        xs[-1], Ps[-1] = xf[-1], Pf[-1]
        for t in range(T - 2, -1, -1):
            J = Pf[t] @ A.T @ np.linalg.inv(Pp[t + 1])
            xs[t] = xf[t] + J @ (xs[t + 1] - xp[t + 1])
            Ps[t] = Pf[t] + J @ (Ps[t + 1] - Pp[t + 1]) @ J.T
            Pcov[t + 1] = J @ Ps[t + 1]

        # M-step: scalar process and observation variances (A held fixed -- the PDE constraint).
        proc = 0.0
        for t in range(1, T):
            Exx = Ps[t] + np.outer(xs[t], xs[t])
            Ex1 = Pcov[t] + np.outer(xs[t], xs[t - 1])
            Exx_1 = Ps[t - 1] + np.outer(xs[t - 1], xs[t - 1])
            resid = Exx - A @ Ex1.T - Ex1 @ A.T + A @ Exx_1 @ A.T
            proc += np.trace(resid)
        q = max(proc / (n * (T - 1)), 1e-10)

        obs = 0.0
        for t in range(T):
            innov = y[t] - H @ xs[t]
            obs += float(innov @ innov) + np.trace(H @ Ps[t] @ H.T)
        r = max(obs / (m * T), 1e-10)

        x0, P0 = xs[0].copy(), Ps[0].copy()
        if prev_ll is not None and abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

    smoothed_var = np.stack([np.diag(Ps[t]) for t in range(T)], axis=0)
    return PDEStateSpaceResult(operator, dt, q, r, A, H, xs, smoothed_var, ll)


def pde_fit(
    rv: RandomVariable, data, *, dt: float = 1.0, H=None, max_its: int = 100, tol: float = 1e-6, **_
) -> RandomVariable:
    """Fit a ``PDE(operator)`` random variable to spatiotemporal ``data`` (T x m observations)."""
    (operator,) = rv._args
    if not isinstance(operator, DynamicsOperator):
        raise TypeError("PDE() requires a DynamicsOperator (see pysp.ppl.dynamics).")
    result = kalman_rts_em(np.asarray(data, dtype=float), operator, dt=dt, H=H, max_its=max_its, tol=tol)
    return RandomVariable._bound(None, name=rv._name, result=result)


def fit_diffusivity(
    observations,
    *,
    length: float = 1.0,
    bc: str = "neumann",
    scheme: str = "exact",
    dt: float = 1.0,
    init_diffusivity: float = 1.0,
    max_its: int = 500,
    lr: float = 0.05,
):
    """Infer a 1-D diffusion coefficient (and observation noise) from spatiotemporal snapshots.

    Unlike :func:`kalman_rts_em` (which holds the PDE transition fixed and fits only the noise
    levels), this estimates the **PDE parameter** itself. The latent field evolves by the
    discretized heat equation ``u_t = A(D) u_{t-1}`` with ``A(D)`` the method-of-lines transition for
    diffusivity ``D`` (the same ``explicit`` / ``implicit`` / ``exact`` schemes as
    :class:`~pysp.ppl.dynamics.DiffusionOperator`). ``D`` (and the observation noise) are fit by
    maximizing the one-step predictive Gaussian likelihood
    ``sum_t log N(y_t; A(D) y_{t-1}, sigma^2 I)``; gradients flow by reverse-mode autodiff through the
    differentiable transition (``torch.matrix_exp`` / linear solve) -- i.e. the discrete **adjoint**
    of the forward solve -- so no hand-derived adjoint equations are needed.

    Args:
        observations: ``(T, n)`` array of noisy field snapshots on a uniform grid.
        length, bc, scheme, dt: spatial/temporal discretization (see ``DiffusionOperator``).
        init_diffusivity, max_its, lr: optimizer seed and budget.

    Returns:
        dict with the fitted ``diffusivity``, ``obs_sd``, and the maximized ``loglik``.
    """
    import torch

    from pysp.inference.objectives import optimize_torch_objective
    from pysp.ppl.physics.dynamics import laplacian_matrix

    y = np.asarray(observations, dtype=float)
    if y.ndim != 2 or y.shape[0] < 2:
        raise ValueError("observations must be a (T, n) array with T >= 2 time steps.")
    n = y.shape[1]
    h = float(length) / (n - 1)
    lap = torch.tensor(laplacian_matrix(n, h, bc), dtype=torch.float64)
    eye = torch.eye(n, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64)
    prev, curr = yt[:-1], yt[1:]

    raw_d = torch.tensor(math.log(float(init_diffusivity)), dtype=torch.float64, requires_grad=True)
    log_sd = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)

    def transition(diffusivity):
        g = diffusivity * dt * lap
        if scheme == "explicit":
            return eye + g
        if scheme == "exact":
            return torch.linalg.matrix_exp(g)
        return torch.linalg.solve(eye - g, eye)  # implicit Euler

    def objective():
        diffusivity = torch.exp(raw_d)
        var = torch.exp(2.0 * log_sd) + 1.0e-12
        resid = curr - prev @ transition(diffusivity).T  # y_t - A(D) y_{t-1}
        return -0.5 * (resid.pow(2).sum() / var + resid.numel() * torch.log(2.0 * math.pi * var))

    loglik = optimize_torch_objective([raw_d, log_sd], objective, max_its=max_its, lr=lr, maximize=True, out=None)
    return {
        "diffusivity": float(torch.exp(raw_d).detach()),
        "obs_sd": float(torch.exp(log_sd).detach()),
        "loglik": float(loglik) if np.isscalar(loglik) or hasattr(loglik, "__float__") else float(loglik[0]),
    }


def fit_pde_parameters(
    observations,
    transition,
    init_params: dict[str, float],
    *,
    max_its: int = 500,
    lr: float = 0.05,
):
    """Estimate arbitrary (possibly nonlinear) PDE parameters by the autograd adjoint.

    Generalizes :func:`fit_diffusivity` to multiple parameters and nonlinear dynamics. ``transition``
    is a differentiable one-step solver ``transition(u_prev, params) -> u_pred`` where ``u_prev`` is a
    ``(B, n)`` torch tensor of fields and ``params`` is a dict of (positive) torch scalars; it may be
    nonlinear in ``u`` (e.g. reaction terms). The named parameters in ``init_params`` (and the scalar
    observation noise) are fit by maximizing the one-step predictive Gaussian likelihood
    ``sum_t log N(y_t; transition(y_{t-1}, params), sigma^2 I)``; gradients flow by reverse-mode
    autodiff through ``transition`` -- the discrete adjoint of the forward solve -- with no
    hand-derived adjoint equations.

    Returns a dict of the fitted parameters plus ``obs_sd`` and the maximized ``loglik``.
    """
    import torch

    from pysp.inference.objectives import optimize_torch_objective

    y = np.asarray(observations, dtype=float)
    if y.ndim != 2 or y.shape[0] < 2:
        raise ValueError("observations must be a (T, n) array with T >= 2 time steps.")
    yt = torch.tensor(y, dtype=torch.float64)
    prev, curr = yt[:-1], yt[1:]

    names = list(init_params)
    raw = {k: torch.tensor(math.log(float(v)), dtype=torch.float64, requires_grad=True) for k, v in init_params.items()}
    log_sd = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)

    def objective():
        params = {k: torch.exp(raw[k]) for k in names}
        var = torch.exp(2.0 * log_sd) + 1.0e-12
        resid = curr - transition(prev, params)
        return -0.5 * (resid.pow(2).sum() / var + resid.numel() * torch.log(2.0 * math.pi * var))

    loglik = optimize_torch_objective(
        [raw[k] for k in names] + [log_sd], objective, max_its=max_its, lr=lr, maximize=True, out=None
    )
    fitted = {k: float(torch.exp(raw[k]).detach()) for k in names}
    fitted["obs_sd"] = float(torch.exp(log_sd).detach())
    fitted["loglik"] = float(loglik) if (np.isscalar(loglik) or hasattr(loglik, "__float__")) else float(loglik[0])
    return fitted


def fit_reaction_diffusion(
    observations,
    *,
    length: float = 1.0,
    bc: str = "neumann",
    dt: float = 1.0,
    init_diffusivity: float = 1.0,
    init_growth: float = 1.0,
    max_its: int = 800,
    lr: float = 0.05,
):
    """Infer the (nonlinear) Fisher-KPP parameters ``du/dt = D u_xx + r u (1 - u)`` from snapshots.

    A two-parameter, nonlinear instance of :func:`fit_pde_parameters`: an explicit-Euler step
    ``u + dt (D L u + r u (1 - u))`` (``L`` the discrete Laplacian) is differentiable in the
    diffusivity ``D`` and growth rate ``r``, both fit by the autograd adjoint. Returns ``diffusivity``,
    ``growth``, ``obs_sd``, ``loglik``.
    """
    import torch

    from pysp.ppl.physics.dynamics import laplacian_matrix

    y = np.asarray(observations, dtype=float)
    n = y.shape[1]
    lap = torch.tensor(laplacian_matrix(n, float(length) / (n - 1), bc), dtype=torch.float64)

    def transition(u_prev, params):
        diffusion = u_prev @ (params["diffusivity"] * lap).T
        reaction = params["growth"] * u_prev * (1.0 - u_prev)
        return u_prev + dt * (diffusion + reaction)

    fitted = fit_pde_parameters(
        y, transition, {"diffusivity": init_diffusivity, "growth": init_growth}, max_its=max_its, lr=lr
    )
    return fitted
