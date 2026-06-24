"""Censored and truncated maximum-likelihood fitting for the pysp PPL.

Survival / reliability / detection-limit data are *partially observed*: a subject still alive at the
end of a study, a component that had not failed, or a measurement below an instrument's threshold are
all **right-censored** -- we know only that the value exceeds some bound. **Truncation** is the dual:
the sample is drawn conditionally on lying in a window (values outside it are never seen). The ordinary
likelihood is wrong for both; this module fits a distribution's free parameters under the correct one,
using each distribution's ``cdf``. It closes the censored-leaf gap (a capability Stan has and most PPLs
lack a clean surface for).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from pysp.ppl.core import RandomVariable, _is_free

_TINY = 1e-300


def _to_unconstrained(value: float, support: str) -> float:
    if support == "positive":
        return float(np.log(max(value, 1e-8)))
    if support == "unit":
        v = min(max(value, 1e-6), 1.0 - 1e-6)
        return float(np.log(v / (1.0 - v)))
    return float(value)


def _to_constrained(theta: float, support: str) -> float:
    theta = 0.0 if not np.isfinite(theta) else float(theta)
    if support == "positive":
        return float(np.exp(np.clip(theta, -50.0, 50.0)))
    if support == "unit":
        return float(1.0 / (1.0 + np.exp(-np.clip(theta, -50.0, 50.0))))
    return theta


def censored_loglik(dist, time: Sequence[float], *, event=None, lower=None, upper=None) -> float:
    """Total log-likelihood of right-censored and/or truncated ``time`` under a fitted ``dist``.

    ``event[i]`` true (default all true) means ``time[i]`` is an observed event contributing
    ``log f(time[i])``; false means it is right-censored, contributing the log-survival
    ``log(1 - F(time[i]))``. ``lower``/``upper`` truncate the support: every point then also subtracts
    ``log(F(upper) - F(lower))`` (use ``None`` for an open end). Requires ``dist.cdf``.
    """
    if not hasattr(dist, "cdf"):
        raise ValueError(f"{type(dist).__name__} has no cdf; censoring/truncation needs one.")
    t = np.asarray(time, dtype=float)
    ev = np.ones(t.size, dtype=bool) if event is None else np.asarray(event, dtype=bool)
    f_lo = 0.0 if lower is None else float(dist.cdf(float(lower)))
    f_hi = 1.0 if upper is None else float(dist.cdf(float(upper)))
    log_norm = np.log(max(f_hi - f_lo, _TINY)) if (lower is not None or upper is not None) else 0.0
    total = 0.0
    for ti, ei in zip(t, ev):
        if ei:
            total += float(dist.log_density(float(ti)))
        else:
            total += float(np.log(max(1.0 - float(dist.cdf(float(ti))), _TINY)))
        total -= log_norm
    return float(total)


def fit_censored(
    model: RandomVariable,
    time: Sequence[float],
    *,
    event: Sequence[Any] | None = None,
    lower: float | None = None,
    upper: float | None = None,
    seed: int = 0,
) -> RandomVariable:
    """Fit a distribution's free parameters to right-censored and/or truncated data by ML.

    ``model`` is a flat PPL distribution with ``free`` parameter slots, e.g. ``Weibull(free, free)`` or
    ``Exponential(free)``. ``time`` are the (possibly censored) values; ``event`` flags which are
    observed events vs right-censored (default all observed); ``lower``/``upper`` mark truncation of the
    sampling window. Maximizes :func:`censored_loglik` over the free slots (Nelder-Mead in the
    unconstrained space, respecting each slot's positivity/unit support) and returns the fitted model as
    a bound :class:`RandomVariable` (with ``.summary()``).
    """
    from scipy.optimize import minimize

    fam = getattr(model, "_family", None)
    if fam is None or not hasattr(fam, "make_dist"):
        raise ValueError("fit_censored needs a flat distribution model, e.g. Weibull(free, free).")
    args = list(model._args)
    free_idx = [i for i, a in enumerate(args) if _is_free(a)]
    if not free_idx:
        raise ValueError("model has no free parameters to fit.")
    support = fam.support
    t = np.asarray(time, dtype=float)
    med, spread = float(np.median(t)), float(np.std(t) or 1.0)

    def _init(i: int) -> float:
        s = support[i] if i < len(support) else "real"
        if s == "positive":
            return max(spread, 1e-2)
        if s == "unit":
            return 0.5
        return med

    theta0 = np.array([_to_unconstrained(_init(i), support[i] if i < len(support) else "real") for i in free_idx])

    def _build(theta: np.ndarray):
        full = list(args)
        for k, i in enumerate(free_idx):
            full[i] = _to_constrained(float(theta[k]), support[i] if i < len(support) else "real")
        return fam.make_dist(tuple(full), model._name)

    def neg_loglik(theta: np.ndarray) -> float:
        try:
            d = _build(theta)
            ll = censored_loglik(d, t, event=event, lower=lower, upper=upper)
        except (ValueError, FloatingPointError, ZeroDivisionError):
            return 1e18
        return 1e18 if not np.isfinite(ll) else -ll

    res = minimize(neg_loglik, theta0, method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 4000})
    best = res.x if (res.success and np.all(np.isfinite(res.x)) and neg_loglik(res.x) < 1e17) else theta0
    fitted = _build(best)
    return RandomVariable._bound(fitted, name=model._name)


def kaplan_meier(time: Sequence[float], event: Sequence[Any] | None = None) -> dict[str, np.ndarray]:
    """Kaplan-Meier nonparametric survival estimate ``S(t)`` from right-censored data.

    Returns ``{'time', 'survival', 'at_risk', 'events'}`` over the distinct event times -- the standard
    model-free survival curve to plot against, or compare a fitted parametric model to.
    """
    t = np.asarray(time, dtype=float)
    ev = np.ones(t.size, dtype=bool) if event is None else np.asarray(event, dtype=bool)
    order = np.argsort(t)
    t, ev = t[order], ev[order]
    times = np.unique(t[ev]) if ev.any() else np.unique(t)
    surv = np.ones(times.size, dtype=float)
    at_risk = np.empty(times.size, dtype=float)
    events = np.empty(times.size, dtype=float)
    s = 1.0
    for k, tau in enumerate(times):
        n_risk = float(np.sum(t >= tau))
        d = float(np.sum((t == tau) & ev))
        at_risk[k] = n_risk
        events[k] = d
        if n_risk > 0:
            s *= 1.0 - d / n_risk
        surv[k] = s
    return {"time": times, "survival": surv, "at_risk": at_risk, "events": events}


__all__ = ["censored_loglik", "fit_censored", "kaplan_meier"]
