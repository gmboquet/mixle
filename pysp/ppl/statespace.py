"""Linear-Gaussian state-space models for pysp.ppl (Kalman filter + RTS smoother + EM).

A univariate latent state evolves as ``x_t = phi * x_{t-1} + w_t`` (``w ~ N(0, q)``) and is
observed as ``y_t = x_t + v_t`` (``v ~ N(0, r)``). ``LocalLevel()`` fixes ``phi = 1`` (a
random walk + noise / trend smoother); ``AR1()`` estimates ``phi``. Fitting is EM: the
E-step is the Kalman/RTS smoother, the M-step updates ``phi, q, r``.
"""

from __future__ import annotations

import math

import numpy as np

from pysp.ppl.core import RandomVariable


class StateSpaceResult:
    def __init__(self, phi, q, r, smoothed, smoothed_var, loglik):
        self.phi = float(phi)
        self.level_sd = float(math.sqrt(q))  # state innovation sd
        self.obs_sd = float(math.sqrt(r))  # observation noise sd
        self.smoothed = np.asarray(smoothed)  # E[x_t | y_{1:T}]
        self.smoothed_sd = np.sqrt(np.asarray(smoothed_var))
        self.loglik = float(loglik)
        self.acceptance_rate = None
        self.predictive = None
        # exposed through RandomVariable.params (no single emission distribution)
        self.coefficients = {"phi": self.phi, "level_sd": self.level_sd, "obs_sd": self.obs_sd}

    def forecast(self, h: int):
        """Point forecasts h steps ahead from the last smoothed state."""
        x = self.smoothed[-1]
        out = []
        for _ in range(h):
            x = self.phi * x
            out.append(x)
        return np.asarray(out)

    def summary(self):
        return {"phi": self.phi, "level_sd": self.level_sd, "obs_sd": self.obs_sd, "loglik": self.loglik}


def _kalman_em(y, phi_free, max_its, tol):
    y = np.asarray(y, dtype=float).reshape(-1)
    T = y.size
    v0 = max(float(np.var(y)), 1e-6)
    phi = 0.5 if phi_free else 1.0
    q, r = 0.1 * v0, 0.5 * v0
    x0, P0 = float(y[0]), v0
    prev_ll = None
    for _ in range(max_its):
        xp = np.empty(T)
        Pp = np.empty(T)
        xf = np.empty(T)
        Pf = np.empty(T)
        xprev, Pprev, ll = x0, P0, 0.0
        for t in range(T):
            xpr = phi * xprev
            Ppr = phi * phi * Pprev + q
            S = Ppr + r
            K = Ppr / S
            innov = y[t] - xpr
            xf[t] = xpr + K * innov
            Pf[t] = (1.0 - K) * Ppr
            xp[t], Pp[t] = xpr, Ppr
            ll += -0.5 * (math.log(2.0 * math.pi * S) + innov * innov / S)
            xprev, Pprev = xf[t], Pf[t]

        xs = np.empty(T)
        Ps = np.empty(T)
        Pcov = np.zeros(T)
        xs[-1], Ps[-1] = xf[-1], Pf[-1]
        for t in range(T - 2, -1, -1):
            J = phi * Pf[t] / Pp[t + 1]
            xs[t] = xf[t] + J * (xs[t + 1] - xp[t + 1])
            Ps[t] = Pf[t] + J * J * (Ps[t + 1] - Pp[t + 1])
            Pcov[t + 1] = J * Ps[t + 1]  # lag-one smoothed covariance

        Exx = Ps + xs**2
        Exx1 = Pcov[1:] + xs[1:] * xs[:-1]
        if phi_free:
            phi = float(np.sum(Exx1) / max(np.sum(Exx[:-1]), 1e-12))
        q = max(float(np.mean(Exx[1:] - 2 * phi * Exx1 + phi * phi * Exx[:-1])), 1e-8)
        r = max(float(np.mean((y - xs) ** 2 + Ps)), 1e-8)
        x0, P0 = float(xs[0]), float(Ps[0])
        if prev_ll is not None and abs(ll - prev_ll) < tol:
            break
        prev_ll = ll
    return StateSpaceResult(phi, q, r, xs, Ps, ll)


def statespace_fit(rv: RandomVariable, data, *, max_its: int = 200, tol: float = 1e-6, **_) -> RandomVariable:
    (phi_free,) = rv._args
    result = _kalman_em(data, bool(phi_free), max_its, tol)
    return RandomVariable._bound(None, name=rv._name, result=result)
