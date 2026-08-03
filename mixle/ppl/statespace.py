"""Linear-Gaussian state-space models for mixle.ppl (Kalman filter + RTS smoother + EM).

A univariate latent state evolves as ``x_t = phi * x_{t-1} + w_t`` (``w ~ N(0, q)``) and is
observed as ``y_t = x_t + v_t`` (``v ~ N(0, r)``). ``LocalLevel()`` fixes ``phi = 1`` (a
random walk + noise / trend smoother); ``AR1()`` estimates ``phi``. Fitting is EM: the
E-step is the Kalman/RTS smoother, the M-step updates ``phi, q, r``.
"""

from __future__ import annotations

import math
import warnings
from numbers import Integral, Real
from typing import Any

import numpy as np

from mixle.ppl.core import RandomVariable, register_composite
from mixle.utils.exact import require_exact_bool

# Roundoff allowance for the EM monotonicity claim (MXR-080-1897). EM's log likelihood is
# non-decreasing by construction, so a *material* decrease means the E-step and M-step disagree --
# but a plateaued fit still jitters at float precision, and refusing that would refuse fits the
# library legitimately produces. Measured over ~14,000 fits (AR1 + LocalLevel; T from 1 to 200;
# Gaussian / random-walk / near-constant / heavy-tailed / marginalized-NaN / huge- and tiny-scale
# series; tol from 0 to 1e-4) the worst single-step decrease was 1.4e-15 relative to |loglik| --
# pure roundoff, and never the step that fired the tolerance test. A relative allowance of 1e-9
# leaves six orders of headroom over the measured noise while still catching any decrease big enough
# to matter. `mixle.stats.latent.lda` uses the same relative-allowance shape at 1e-12.
#
# NOT checked, deliberately: whether the M-step's `max(..., 1e-8)` floors on q, r, and P0 can break
# monotonicity. They cannot in theory (clipping a concave-in-the-parameter Q at a bound still yields
# the constrained maximizer) and no sweep produced a counterexample, so no allowance is spent on it.
_MONOTONE_RELATIVE_SLACK = 1.0e-9

# Declared stopping semantics (MXR-080-1897): the reason names the rule that fired, and the
# `converged` verdict is not free to disagree with it. Adding a stopping rule means adding its
# reason here -- which is the point: a receipt whose verdict and reason are set independently can
# report "converged" alongside a reason that says the loop ran out of iterations.
_TERMINATION_REASONS = {
    "objective_tolerance": True,  # a non-decreasing step smaller than tol
    "iteration_limit": False,  # max_its exhausted without such a step
}


def _monotone_slack(objective: float) -> float:
    """Largest objective decrease attributable to float roundoff at this objective scale."""
    return _MONOTONE_RELATIVE_SLACK * max(1.0, abs(objective))


class StateSpaceResult:
    """Fitted univariate linear-Gaussian state-space model and smoothed latent path.

    The reported ``loglik`` is bound to ``objective_trace[-1]`` and the ``converged`` verdict is
    bound to ``termination_reason``: a receipt that can state a final objective unrelated to the
    trace it came from, or a verdict unrelated to the rule that stopped the loop, is not evidence
    (MXR-080-1897). The smoothed path and its standard deviations are copied and sealed, so neither
    the caller's arrays nor a later write through the result can rewrite geometry that was validated
    at construction.
    """

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
        self.phi = _finite_float(phi, "phi")
        self.level_sd = float(math.sqrt(_positive_finite(q, "state innovation variance q")))  # state innovation sd
        self.obs_sd = float(math.sqrt(_positive_finite(r, "observation variance r")))  # observation noise sd
        self.initial_mean = _finite_float(x0, "initial mean x0")
        self.initial_sd = float(math.sqrt(_positive_finite(P0, "initial variance P0")))
        # `np.array(..., copy=True)` + `writeable = False`, not `np.asarray`: `np.asarray` returns the
        # caller's own float64 array, so every check below described an array the caller could still
        # rewrite afterwards -- a finite, non-negative-variance path that becomes NaN one statement
        # later (MXR-080-1897).
        self.smoothed = np.array(smoothed, dtype=float, copy=True)  # E[x_t | y_{1:T}]
        variance = np.array(smoothed_var, dtype=float, copy=True)
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
        self.smoothed.flags.writeable = False
        self.smoothed_sd.flags.writeable = False
        self.loglik = _finite_float(loglik, "loglik")
        self.converged = require_exact_bool(converged, "converged")
        self.iterations = _positive_int(iterations, "iterations")
        self.objective_trace = tuple(_finite_float(value, "objective_trace entry") for value in objective_trace)
        if not isinstance(termination_reason, str):
            raise TypeError(f"termination_reason must be one of {sorted(_TERMINATION_REASONS)}, not a coerced object")
        self.termination_reason = termination_reason
        if self.termination_reason not in _TERMINATION_REASONS:
            raise ValueError(
                f"termination_reason={self.termination_reason!r} is not a declared state-space stopping rule "
                f"(expected one of {sorted(_TERMINATION_REASONS)})"
            )
        if _TERMINATION_REASONS[self.termination_reason] is not self.converged:
            raise ValueError(
                f"termination_reason={self.termination_reason!r} and converged={self.converged} disagree; "
                "the verdict must be the one the named stopping rule implies"
            )
        if len(self.objective_trace) != self.iterations + 1:
            raise ValueError(
                f"objective_trace has {len(self.objective_trace)} entries for {self.iterations} iterations; "
                "it must hold the pre-iteration objective plus one entry per iteration"
            )
        if self.loglik != self.objective_trace[-1]:
            raise ValueError(
                f"loglik={self.loglik!r} is not the objective the fit ended on ({self.objective_trace[-1]!r}); "
                "the reported log likelihood must be the one the returned parameters produced"
            )
        # Recorded, not enforced (MXR-080-1897): a decrease is *disclosed* rather than refused,
        # because the trace is the honest record of what EM did and a result that cannot hold a
        # non-monotone trace cannot report one. `_kalman_em` is what refuses to call a decreasing
        # step "converged"; this is the evidence a reader needs to see that it happened.
        steps = np.diff(np.asarray(self.objective_trace, dtype=float))
        self.max_objective_decrease = float(min(0.0, steps.min())) if steps.size else 0.0
        self.monotone = bool(
            all(step >= -_monotone_slack(previous) for previous, step in zip(self.objective_trace, steps))
        )
        self.termination = {
            "converged": self.converged,
            "iterations": self.iterations,
            "reason": self.termination_reason,
            "objective": self.loglik,
            "objective_trace": self.objective_trace,
            "monotone": self.monotone,
            "max_objective_decrease": self.max_objective_decrease,
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
            "monotone": self.monotone,
            "max_objective_decrease": self.max_objective_decrease,
        }


def _finite_float(value: Any, name: str) -> float:
    """Return ``value`` as a finite float, refusing anything that is not already a real number.

    ``float(value)`` accepted ``"0.5"`` and ``True`` for a fitted parameter (MXR-080-1897); a result
    object records what a fit produced, so a value that had to be parsed or reinterpreted to become a
    number did not come from the fit.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number, got {type(value).__name__} ({value!r})")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real number, got {result!r}")
    return result


def _positive_finite(value: Any, name: str) -> float:
    result = _finite_float(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be a finite positive real number, got {result!r}")
    return result


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
    warned_about_decrease = False
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
        # Declared stopping semantics (MXR-080-1897). The old rule was `abs(next_ll - ll) < tol`,
        # which reads a *decrease* as convergence: EM's objective is non-decreasing by construction,
        # so a decrease is evidence the E-step and M-step disagree, and calling it "converged" hands
        # the caller a receipt asserting the opposite of what happened. What the rule now requires is
        # spelled out: a step that did not go downhill (beyond float roundoff at this objective's
        # scale) *and* moved less than tol. The magnitude test stays `abs(change) < tol` so the
        # caller's tolerance keeps meaning exactly what it did; the only steps that stop qualifying
        # are the ones that went materially backwards.
        change = next_ll - ll
        slack = _monotone_slack(ll)
        if change < -slack:
            # Not raised. A material decrease is a defect in the updates, but the fit still has a
            # usable smoothed path and the caller keeps the full trace to judge it; a raise here
            # would also discard results for any future update rule that is only weakly monotone.
            # It cannot be reported as convergence, and it is disclosed twice: here, and in the
            # result's `monotone` / `max_objective_decrease` receipt fields.
            if not warned_about_decrease:
                warned_about_decrease = True
                warnings.warn(
                    f"state-space EM objective decreased by {-change:.3e} at iteration {iterations} "
                    f"(from {ll!r} to {next_ll!r}). EM is non-decreasing by construction, so this is "
                    "not convergence and is not reported as such; inspect result.objective_trace.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        elif tol > 0.0 and abs(change) < tol:
            converged = True
            ll = next_ll
            break
        ll = next_ll
    termination_reason = "objective_tolerance" if converged else "iteration_limit"
    return StateSpaceResult(
        phi=phi,
        q=q,
        r=r,
        x0=x0,
        P0=P0,
        smoothed=xs,
        smoothed_var=Ps,
        loglik=ll,
        converged=converged,
        iterations=iterations,
        objective_trace=trace,
        termination_reason=termination_reason,
    )


def statespace_fit(
    rv: RandomVariable,
    data,
    *,
    how: str = "auto",
    max_its: int = 200,
    tol: float | None = None,
    delta: float = 1e-8,
    missing: str = "error",
    backend: str = "local",
    num_workers: int | None = None,
    engine: Any = None,
    precision: Any = None,
    print_iter: int = 0,
    **unknown,
) -> RandomVariable:
    """Fit a ``LocalLevel`` or ``AR1`` state-space expression by Kalman EM."""
    if unknown:
        raise TypeError(f"unsupported state-space fit control(s): {', '.join(sorted(unknown))}")
    if how not in {"auto", "em"}:
        raise NotImplementedError(f"state-space fitting implements EM, not how={how!r}")
    if backend != "local" or num_workers is not None or engine is not None or precision is not None:
        raise NotImplementedError("state-space fitting currently supports only local execution")
    if print_iter != 0:
        raise NotImplementedError("state-space fitting does not implement print_iter")
    if tol is not None and delta != 1e-8:
        raise ValueError("tol and delta are aliases; pass only one")
    (phi_free,) = rv._args
    tolerance = delta if tol is None else tol
    result = _kalman_em(data, bool(phi_free), max_its, tolerance, missing=missing)
    return RandomVariable._bound(None, name=rv._name, result=result)


def _ss_err(*a, **k):
    raise NotImplementedError("state-space models are fit via fit(); they have no single dist.")


# Self-register the StateSpace composite with its bespoke fitter (the fit_fn hook), so core dispatches
# to statespace_fit without a per-family branch. LocalLevel()/AR1() build RandomVariables of this family.
register_composite("StateSpace", _ss_err, _ss_err, fit_fn=statespace_fit)
