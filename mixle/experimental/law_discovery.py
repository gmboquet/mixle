"""Discover a *validated* empirical law from a black-box simulator -- a scoped, honest step into the
"derive an equation / propose a relationship" tier of the discovery vision.

Distinct from :mod:`mixle.experimental.equation_discovery` (SINDy recovery of a *known* ODE operator,
graded against ground truth): this takes a real simulator you can only *call* -- ``simulate(x) -> y``
-- sweeps its input, fits a small library of candidate functional forms, and selects the winner by
**out-of-sample validation on held-out inputs the fit never saw**. That held-out score is the whole
point: it is what separates a *discovered law* (predicts new inputs) from an *overfit* (memorizes the
sweep). If no form generalizes, that is reported honestly (a low held-out R^2), not dressed up as a
discovery -- verification is the referee, exactly as it must be for anything calling itself discovery.

Exploratory ``mixle.experimental`` code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

__all__ = ["CANDIDATE_FORMS", "DiscoveredLaw", "discover_law"]

# name -> (fitter-function(x, *params), n_params, human-readable expression, requires_positive_x)
CANDIDATE_FORMS: dict[str, tuple[Callable[..., np.ndarray], int, str, bool]] = {
    "linear": (lambda x, a, b: a * x + b, 2, "y = a*x + b", False),
    "quadratic": (lambda x, a, b, c: a * x**2 + b * x + c, 3, "y = a*x^2 + b*x + c", False),
    "power": (lambda x, a, b: a * np.power(np.clip(x, 1e-12, None), b), 2, "y = a*x^b", True),
    "logarithmic": (lambda x, a, b: a * np.log(np.clip(x, 1e-12, None)) + b, 2, "y = a*ln(x) + b", True),
    "sqrt": (lambda x, a, b: a * np.sqrt(np.clip(x, 0.0, None)) + b, 2, "y = a*sqrt(x) + b", True),
    "exponential": (lambda x, a, b: a * np.exp(np.clip(b * x, -50, 50)), 2, "y = a*e^(b*x)", False),
}


@dataclass
class DiscoveredLaw:
    """A discovered relationship + the verification that earns the word 'discovered'.

    ``holdout_r2`` (out-of-sample, on inputs the fit never saw) is the referee: high means the law
    generalizes; low means no clean law was found at this budget, and ``passed`` says so honestly.
    """

    form: str
    expression: str
    params: dict[str, float]
    train_r2: float
    holdout_r2: float
    n_train: int
    n_holdout: int
    passed: bool  # holdout_r2 >= min_holdout_r2 -- did any form actually generalize?
    ranking: list[tuple[str, float]] = field(default_factory=list)  # (form, holdout_r2), best first


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot < 1e-30:
        return 1.0 if ss_res < 1e-30 else 0.0
    return 1.0 - ss_res / ss_tot


def discover_law(
    simulate: Callable[[float], float],
    input_range: tuple[float, float],
    *,
    n_samples: int = 24,
    holdout_fraction: float = 1.0 / 3.0,
    min_holdout_r2: float = 0.9,
    forms: tuple[str, ...] | None = None,
    log_spaced: bool = False,
    seed: int = 0,
) -> DiscoveredLaw:
    """Sweep ``simulate`` over ``input_range``, fit candidate functional forms, and return the one that
    best predicts a held-out slice of inputs.

    Args:
        simulate: the black-box ``x -> y`` (a real simulator wrapped to scalar in/out).
        input_range: ``(lo, hi)`` to sweep the input over.
        n_samples: total simulator evaluations across the sweep.
        holdout_fraction: fraction of the swept points reserved for out-of-sample validation. The
            held-out points are taken *interleaved* across the range (every k-th), so validation spans
            the whole domain rather than only its far end -- a law must generalize everywhere.
        min_holdout_r2: the bar the best form's held-out R^2 must clear for ``passed=True``.
        forms: restrict the candidate library (default: all of :data:`CANDIDATE_FORMS` whose positivity
            requirement the input range satisfies).
        log_spaced: sweep inputs geometrically (for laws probed over decades, e.g. resistivity).
        seed: unused here (the sweep is deterministic); kept for interface parity with the rest of the
            discovery API.

    Returns:
        A :class:`DiscoveredLaw` -- the winning form, its fitted params, train and (the referee)
        held-out R^2, and the full ranking. ``passed`` is honest: if nothing generalizes, it is False.
    """
    from scipy.optimize import curve_fit

    lo, hi = float(input_range[0]), float(input_range[1])
    xs = np.geomspace(max(lo, 1e-9), hi, n_samples) if log_spaced else np.linspace(lo, hi, n_samples)
    ys = np.array([float(simulate(float(x))) for x in xs], dtype=float)

    # interleaved holdout: every k-th point spans the whole domain, so validation isn't just extrapolation.
    k = max(2, int(round(1.0 / max(holdout_fraction, 1e-6))))
    is_holdout = (np.arange(n_samples) % k == 1)
    x_tr, y_tr = xs[~is_holdout], ys[~is_holdout]
    x_ho, y_ho = xs[is_holdout], ys[is_holdout]

    positive_domain = lo > 0
    candidate_names = forms or tuple(CANDIDATE_FORMS)
    results: list[tuple[str, float, float, dict[str, float]]] = []  # (form, train_r2, holdout_r2, params)
    for name in candidate_names:
        fn, n_params, _expr, needs_pos = CANDIDATE_FORMS[name]
        if needs_pos and not positive_domain:
            continue
        try:
            popt, _ = curve_fit(fn, x_tr, y_tr, p0=np.ones(n_params), maxfev=10000)
        except Exception:  # noqa: BLE001 -- a non-converging form is simply not the law; skip it
            continue
        train_r2 = _r2(y_tr, fn(x_tr, *popt))
        holdout_r2 = _r2(y_ho, fn(x_ho, *popt))
        param_names = list("abcdefg")[:n_params]
        results.append((name, train_r2, holdout_r2, dict(zip(param_names, (round(float(p), 6) for p in popt)))))

    if not results:
        return DiscoveredLaw("none", "no candidate form fit", {}, 0.0, float("-inf"), len(x_tr), len(x_ho), False, [])

    results.sort(key=lambda r: r[2], reverse=True)  # rank by the referee: held-out R^2
    best_name, best_train, best_holdout, best_params = results[0]
    return DiscoveredLaw(
        form=best_name, expression=CANDIDATE_FORMS[best_name][2], params=best_params,
        train_r2=round(best_train, 4), holdout_r2=round(best_holdout, 4),
        n_train=len(x_tr), n_holdout=len(x_ho), passed=best_holdout >= min_holdout_r2,
        ranking=[(n, round(h, 4)) for n, _t, h, _p in results],
    )
