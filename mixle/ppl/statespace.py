"""Linear-Gaussian state-space models for mixle.ppl (Kalman filter + RTS smoother + EM).

A univariate latent state evolves as ``x_t = phi * x_{t-1} + w_t`` (``w ~ N(0, q)``) and is
observed as ``y_t = x_t + v_t`` (``v ~ N(0, r)``). ``LocalLevel()`` fixes ``phi = 1`` (a
random walk + noise / trend smoother); ``AR1()`` estimates ``phi``. Fitting is EM: the
E-step is the Kalman/RTS smoother, the M-step updates ``phi, q, r``.
"""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any

import numpy as np

from mixle.ppl.core import RandomVariable, register_composite


class StateSpaceResult:
    """Fitted univariate linear-Gaussian state-space model and smoothed latent path."""

    def __init__(
        self,
        phi,
        q,
        r,
        x0,
        P0,
        smoothed,
        smoothed_var,
        loglik,
        *,
        converged,
        iterations,
        objective_trace,
        termination_reason,
    ):
        self.phi = float(phi)
        self.level_sd = float(math.sqrt(q))  # state innovation sd
        self.obs_sd = float(math.sqrt(r))  # observation noise sd
        self.initial_mean = float(x0)
        self.initial_sd = float(math.sqrt(P0))
        self.smoothed = np.asarray(smoothed, dtype=float)  # E[x_t | y_{1:T}]
        variance = np.asarray(smoothed_var, dtype=float)
        if (
            self.smoothed.ndim != 1
            or self.smoothed.size == 0
            or variance.shape != self.smoothed.shape
            or not np.all(np.isfinite(self.smoothed))
            or not np.all(np.isfinite(variance))
            or np.any(variance < 0.0)
        ):
            raise ValueError("state-space smoothing must return a finite non-empty path and non-negative variances")
        self.smoothed_sd = np.sqrt(variance)
        self.loglik = float(loglik)
        self.converged = bool(converged)
        self.iterations = int(iterations)
        self.objective_trace = tuple(float(value) for value in objective_trace)
        self.termination_reason = str(termination_reason)
        if (
            not all(
                np.isfinite(value)
                for value in (self.phi, self.level_sd, self.obs_sd, self.initial_mean, self.initial_sd, self.loglik)
            )
            or self.level_sd <= 0.0
            or self.obs_sd <= 0.0
            or self.initial_sd <= 0.0
            or self.iterations <= 0
            or len(self.objective_trace) != self.iterations + 1
            or not np.all(np.isfinite(self.objective_trace))
        ):
            raise ValueError("state-space fit produced an invalid parameter or termination receipt")
        self.termination = {
            "converged": self.converged,
            "iterations": self.iterations,
            "reason": self.termination_reason,
            "objective": self.loglik,
            "objective_trace": self.objective_trace,
        }
        self.acceptance_rate = None
        self.predictive = None
        # exposed through RandomVariable.params (no single emission distribution)
        self.coefficients = {
            "phi": self.phi,
            "level_sd": self.level_sd,
            "obs_sd": self.obs_sd,
            "initial_mean": self.initial_mean,
            "initial_sd": self.initial_sd,
        }

    def forecast(self, h: int):
        """Point forecasts h steps ahead from the last smoothed state."""
        h = _positive_int(h, "forecast horizon")
        x = self.smoothed[-1]
        out = []
        for _ in range(h):
            x = self.phi * x
            out.append(x)
        return np.asarray(out)

    def summary(self):
        """Return fitted dynamics, noise scales, initialization, and log likelihood."""
        return {
            "phi": self.phi,
            "level_sd": self.level_sd,
            "obs_sd": self.obs_sd,
            "initial_mean": self.initial_mean,
            "initial_sd": self.initial_sd,
            "loglik": self.loglik,
            "converged": self.converged,
            "iterations": self.iterations,
            "termination_reason": self.termination_reason,
            "objective_trace": self.objective_trace,
        }


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _nonnegative_finite(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite non-negative real number")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite non-negative real number")
    return result


def _series(value: Any, missing: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise TypeError("state-space observations must be a numeric one-dimensional series") from error
    if result.ndim != 1 or result.size == 0:
        raise ValueError("state-space observations must be a non-empty one-dimensional series")
    if np.any(np.isinf(result)):
        raise ValueError("state-space observations must not contain infinite values")
    if np.any(np.isnan(result)):
        if missing == "error":
            raise ValueError(
                "state-space observations contain missing values; pass missing='marginalize' to integrate them out"
            )
        if missing != "marginalize":
            raise ValueError(f"missing={missing!r}; choose 'error' or 'marginalize'")
    elif missing not in {"error", "marginalize"}:
        raise ValueError(f"missing={missing!r}; choose 'error' or 'marginalize'")
    if not np.any(np.isfinite(result)):
        raise ValueError("state-space observations must contain at least one observed value")
    return result


def _kalman_smooth(y, phi, q, r, x0, P0):
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or y.size == 0:
        raise ValueError("Kalman smoothing requires a non-empty one-dimensional series")
    if np.any(np.isinf(y)) or not np.any(np.isfinite(y)):
        raise ValueError("Kalman smoothing requires finite values or explicitly marginalized NaNs")
    if not all(np.isfinite(value) for value in (phi, q, r, x0, P0)) or min(q, r, P0) <= 0.0:
        raise ValueError("Kalman parameters must be finite and q, r, and P0 must be positive")
    T = y.size
    xp = np.empty(T)
    Pp = np.empty(T)
    xf = np.empty(T)
    Pf = np.empty(T)
    xprev, Pprev, ll = x0, P0, 0.0
    for t in range(T):
        xpr = phi * xprev
        Ppr = phi * phi * Pprev + q
        if np.isnan(y[t]):
            xf[t], Pf[t] = xpr, Ppr
        else:
            S = Ppr + r
            K = Ppr / S
            innov = y[t] - xpr
            xf[t] = xpr + K * innov
            Pf[t] = (1.0 - K) * Ppr
            ll += -0.5 * (math.log(2.0 * math.pi * S) + innov * innov / S)
        xp[t], Pp[t] = xpr, Ppr
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

    return xs, Ps, Pcov, ll


def _kalman_em(y, phi_free, max_its, tol, *, missing="error"):
    y = _series(y, missing)
    max_its = _positive_int(max_its, "max_its")
    tol = _nonnegative_finite(tol, "tol")
    T = y.size
    observed = np.isfinite(y)
    v0 = max(float(np.nanvar(y)), 1e-6)
    phi = 0.5 if phi_free else 1.0
    q, r = 0.1 * v0, 0.5 * v0
    x0, P0 = float(y[observed][0]), v0
    xs, Ps, Pcov, ll = _kalman_smooth(y, phi, q, r, x0, P0)
    trace = [float(ll)]
    converged = False
    iterations = 0
    for iteration in range(max_its):
        # The filter treats (x0, P0) as the state BEFORE the first observation (its first
        # prediction is phi*x0), so the E-step must smooth one extra step back to that same
        # pre-sample state (Shumway & Stoffer): assigning the time-0 posterior instead mixes
        # timing conventions and breaks EM monotonicity on short series.
        Pp0 = phi * phi * P0 + q  # the filter's first prediction from (x0, P0)
        J0 = phi * P0 / Pp0
        xs_init = x0 + J0 * (xs[0] - phi * x0)
        Ps_init = P0 + J0 * J0 * (Ps[0] - Pp0)
        Pcov0 = J0 * Ps[0]  # lag-one smoothed covariance Cov(x_init, x_0 | y)
        Exx = Ps + xs**2
        Exx1 = Pcov[1:] + xs[1:] * xs[:-1]
        # transition sums include the pre-sample -> time-0 step, matching the filter's timing
        num = float(Pcov0 + xs[0] * xs_init) + float(np.sum(Exx1))  # sum E[x_t x_{t-1}]
        den = float(Ps_init + xs_init * xs_init) + float(np.sum(Exx[:-1]))  # sum E[x_{t-1}^2]
        if phi_free:
            phi = float(num / max(den, 1e-12))
        q = max(float((np.sum(Exx) - 2 * phi * num + phi * phi * den) / T), 1e-8)
        r = max(float(np.mean((y[observed] - xs[observed]) ** 2 + Ps[observed])), 1e-8)
        x0, P0 = float(xs_init), max(float(Ps_init), 1e-8)
        xs, Ps, Pcov, next_ll = _kalman_smooth(y, phi, q, r, x0, P0)
        if not np.isfinite(next_ll):
            raise RuntimeError(f"state-space EM objective became non-finite at iteration {iteration + 1}")
        trace.append(float(next_ll))
        iterations = iteration + 1
        if tol > 0.0 and abs(next_ll - ll) < tol:
            converged = True
            ll = next_ll
            break
        ll = next_ll
    termination_reason = "objective_tolerance" if converged else "iteration_limit"
    return StateSpaceResult(
        phi,
        q,
        r,
        x0,
        P0,
        xs,
        Ps,
        ll,
        converged=converged,
        iterations=iterations,
        objective_trace=trace,
        termination_reason=termination_reason,
    )


def statespace_fit(
    rv: RandomVariable,
    data,
    *,
    max_its: int = 200,
    tol: float | None = None,
    delta: float = 1e-8,
    missing: str = "error",
    **_,
) -> RandomVariable:
    """Fit a ``LocalLevel`` or ``AR1`` state-space expression by Kalman EM."""
    (phi_free,) = rv._args
    tolerance = delta if tol is None else tol
    result = _kalman_em(data, bool(phi_free), max_its, tolerance, missing=missing)
    return RandomVariable._bound(None, name=rv._name, result=result)


def _ss_err(*a, **k):
    raise NotImplementedError("state-space models are fit via fit(); they have no single dist.")


# Self-register the StateSpace composite with its bespoke fitter (the fit_fn hook), so core dispatches
# to statespace_fit without a per-family branch. LocalLevel()/AR1() build RandomVariables of this family.
register_composite("StateSpace", _ss_err, _ss_err, fit_fn=statespace_fit)
