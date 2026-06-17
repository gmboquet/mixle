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
from pysp.ppl.dynamics import DynamicsOperator


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
