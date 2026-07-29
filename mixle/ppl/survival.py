"""Censored and truncated maximum-likelihood fitting for the mixle PPL.

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
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np

from mixle.ppl.core import RandomVariable, _is_free


def _to_unconstrained(value: float, support: str) -> float:
    if support == "positive":
        return float(np.log(max(value, 1e-8)))
    if support == "unit":
        v = min(max(value, 1e-6), 1.0 - 1e-6)
        return float(np.log(v / (1.0 - v)))
    return float(value)


def _to_constrained(theta: float, support: str) -> float:
    if not np.isfinite(theta):
        raise ValueError("unconstrained parameter must be finite")
    theta = float(theta)
    if support == "positive":
        return float(np.exp(np.clip(theta, -50.0, 50.0)))
    if support == "unit":
        return float(1.0 / (1.0 + np.exp(-np.clip(theta, -50.0, 50.0))))
    return theta


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _survival_data(
    time: Sequence[float],
    event: Sequence[Any] | None,
    *,
    lower: float | None = None,
    upper: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float | None, float | None]:
    try:
        times = np.asarray(time, dtype=float)
    except (TypeError, ValueError) as error:
        raise TypeError("survival times must be a finite numeric vector") from error
    if times.ndim != 1 or times.size == 0:
        raise ValueError("survival times must be a non-empty one-dimensional vector")
    if not np.all(np.isfinite(times)):
        raise ValueError("survival times must contain only finite values")

    if event is None:
        events = np.ones(times.size, dtype=bool)
    else:
        raw_events = np.asarray(event)
        if raw_events.ndim != 1 or raw_events.shape != times.shape:
            raise ValueError(f"event must be one-dimensional with exactly {times.size} entries")
        if raw_events.dtype.kind not in {"b", "i", "u", "f"}:
            raise ValueError("event indicators must contain exact binary values 0 or 1")
        try:
            numeric_events = raw_events.astype(float)
        except (TypeError, ValueError) as error:
            raise ValueError("event indicators must contain exact binary values 0 or 1") from error
        if not np.all(np.isfinite(numeric_events)) or not np.all((numeric_events == 0.0) | (numeric_events == 1.0)):
            raise ValueError("event indicators must contain exact binary values 0 or 1")
        events = numeric_events.astype(bool)

    bounds = []
    for value, name in ((lower, "lower"), (upper, "upper")):
        if value is None:
            bounds.append(None)
            continue
        try:
            bound = float(value)
        except (TypeError, ValueError) as error:
            raise TypeError(f"{name} truncation bound must be finite") from error
        if not np.isfinite(bound):
            raise ValueError(f"{name} truncation bound must be finite")
        bounds.append(bound)
    lower, upper = bounds
    if lower is not None and upper is not None and lower >= upper:
        raise ValueError("lower truncation bound must be strictly less than upper")
    if lower is not None and np.any(times < lower):
        raise ValueError("survival times must not fall below the lower truncation bound")
    if upper is not None and np.any(times > upper):
        raise ValueError("survival times must not exceed the upper truncation bound")
    return times, events, lower, upper


def _cdf(dist: Any, value: float, name: str) -> float:
    try:
        result = float(dist.cdf(float(value)))
    except (TypeError, ValueError, FloatingPointError) as error:
        raise ValueError(f"distribution cdf failed at {name}={value!r}") from error
    if not np.isfinite(result) or result < -1e-12 or result > 1.0 + 1e-12:
        raise ValueError(f"distribution cdf at {name}={value!r} must be a finite probability")
    return min(max(result, 0.0), 1.0)


def _log_survival_probability(dist: Any, value: float, name: str) -> float:
    from mixle.stats.combinator.survival import _log_survival

    _cdf(dist, value, name)
    result = float(_log_survival(dist, value))
    if np.isnan(result) or result > 1e-12:
        raise ValueError(f"distribution survival probability at {name}={value!r} is invalid")
    return min(result, 0.0)


def _log_difference(log_a: float, log_b: float, name: str) -> float:
    """Return log(exp(log_a) - exp(log_b)) for probabilities a >= b."""
    if log_a == -np.inf:
        if log_b == -np.inf:
            return -np.inf
        raise ValueError(f"{name} has reversed probability bounds")
    if log_b == -np.inf:
        return log_a
    if log_b > log_a + 1e-12:
        raise ValueError(f"{name} has reversed probability bounds")
    ratio = min(float(np.exp(log_b - log_a)), 1.0)
    return -np.inf if ratio >= 1.0 else float(log_a + np.log1p(-ratio))


def censored_loglik(dist, time: Sequence[float], *, event=None, lower=None, upper=None) -> float:
    """Total log-likelihood of right-censored and/or truncated ``time`` under a fitted ``dist``.

    ``event[i]`` true (default all true) means ``time[i]`` is an observed event contributing
    ``log f(time[i])``; false means it is right-censored, contributing the log-survival
    ``log(1 - F(time[i]))``. ``lower``/``upper`` truncate the support: every point then also subtracts
    ``log(F(upper) - F(lower))`` (use ``None`` for an open end). Requires ``dist.cdf``.
    """
    if not hasattr(dist, "cdf"):
        raise ValueError(f"{type(dist).__name__} has no cdf; censoring/truncation needs one.")
    t, ev, lower, upper = _survival_data(time, event, lower=lower, upper=upper)
    if lower is None and upper is None:
        log_norm = 0.0
    elif lower is None:
        f_hi = _cdf(dist, upper, "upper")
        log_norm = -np.inf if f_hi <= 0.0 else float(np.log(f_hi))
    elif upper is None:
        log_norm = _log_survival_probability(dist, lower, "lower")
    else:
        log_norm = _log_difference(
            _log_survival_probability(dist, lower, "lower"),
            _log_survival_probability(dist, upper, "upper"),
            "truncation window",
        )
    if log_norm == -np.inf:
        raise ValueError("truncation window has zero probability under the distribution")
    total = 0.0
    for ti, ei in zip(t, ev):
        if ei:
            contribution = float(dist.log_density(float(ti)))
            if np.isnan(contribution):
                raise ValueError(f"distribution returned NaN log density at time {ti!r}")
        else:
            log_survival_at_time = _log_survival_probability(dist, float(ti), "censoring time")
            contribution = (
                log_survival_at_time
                if upper is None
                else _log_difference(
                    log_survival_at_time,
                    _log_survival_probability(dist, upper, "upper"),
                    "upper-truncated censoring probability",
                )
            )
        total += contribution - log_norm
    return float(total)


@dataclass(frozen=True)
class CensoredFitResult:
    """Optimization receipt attached to a successfully fitted censored model."""

    converged: bool
    iterations: int
    evaluations: int
    objective: float
    seed: int
    starts: int
    termination_reason: str
    message: str


def fit_censored(
    model: RandomVariable,
    time: Sequence[float],
    *,
    event: Sequence[Any] | None = None,
    lower: float | None = None,
    upper: float | None = None,
    seed: int = 0,
    max_its: int = 4000,
    n_starts: int = 2,
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
    t, events, lower, upper = _survival_data(time, event, lower=lower, upper=upper)
    max_its = _positive_int(max_its, "max_its")
    n_starts = _positive_int(n_starts, "n_starts")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral):
        raise TypeError("seed must be an integer")
    seed = int(seed)
    if seed < 0 or seed >= 2**32:
        raise ValueError("seed must lie in [0, 2**32)")
    rng = np.random.RandomState(seed)
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
        if not np.all(np.isfinite(theta)):
            return 1e18
        try:
            d = _build(theta)
            ll = censored_loglik(d, t, event=events, lower=lower, upper=upper)
        except (TypeError, ValueError, FloatingPointError, OverflowError, ZeroDivisionError):
            return 1e18
        return 1e18 if not np.isfinite(ll) else -ll

    starts = [theta0]
    starts.extend(theta0 + rng.normal(0.0, 0.25, size=theta0.shape) for _ in range(n_starts - 1))
    successful = []
    failures = []
    for start in starts:
        result = minimize(
            neg_loglik,
            start,
            method="Nelder-Mead",
            options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": max_its},
        )
        objective = neg_loglik(np.asarray(result.x, dtype=float))
        if result.success and np.all(np.isfinite(result.x)) and np.isfinite(objective) and objective < 1e17:
            successful.append((objective, result))
        else:
            failures.append(str(getattr(result, "message", "unknown optimizer failure")))
    if not successful:
        messages = "; ".join(failures) or "no valid optimizer result"
        raise RuntimeError(f"censored maximum-likelihood optimization did not converge: {messages}")
    objective, best = min(successful, key=lambda item: item[0])
    fitted = _build(np.asarray(best.x, dtype=float))
    receipt = CensoredFitResult(
        converged=True,
        iterations=int(getattr(best, "nit", 0)),
        evaluations=int(getattr(best, "nfev", 0)),
        objective=float(objective),
        seed=seed,
        starts=n_starts,
        termination_reason="optimizer_converged",
        message=str(getattr(best, "message", "")),
    )
    return RandomVariable._bound(fitted, name=model._name, result=receipt)


def kaplan_meier(time: Sequence[float], event: Sequence[Any] | None = None) -> dict[str, np.ndarray]:
    """Kaplan-Meier nonparametric survival estimate ``S(t)`` from right-censored data.

    Returns ``{'time', 'survival', 'at_risk', 'events'}`` over the distinct event times -- the standard
    model-free survival curve to plot against, or compare a fitted parametric model to.
    """
    t, ev, _, _ = _survival_data(time, event)
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


__all__ = ["CensoredFitResult", "censored_loglik", "fit_censored", "kaplan_meier"]
