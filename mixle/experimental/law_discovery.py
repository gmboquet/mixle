"""Discover a *validated* empirical law from a black-box simulator -- a scoped, honest step into the
"derive an equation / propose a relationship" tier of the discovery vision.

Distinct from :mod:`mixle.experimental.equation_discovery` (SINDy recovery of a *known* ODE operator,
graded against ground truth): this takes a real simulator you can only *call* -- ``simulate(x) -> y``
-- sweeps its input, fits a small library of candidate functional forms, selects the winner on a
validation split, and confirms that winner once on an untouched holdout split. The independent
confirmation score is the whole point: it is what separates a *discovered law* (predicts new inputs)
from a model that merely won a finite candidate search. If no form generalizes, that is reported
honestly, not dressed up as a discovery.

Exploratory ``mixle.experimental`` code.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

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

    The candidate library is ranked by ``selection_r2`` on a validation split. ``holdout_r2``
    is then computed once for the winner on an untouched confirmation split and is the referee for
    ``passed``.
    """

    form: str
    expression: str
    params: dict[str, float]
    train_r2: float
    holdout_r2: float
    n_train: int
    n_holdout: int
    passed: bool  # holdout_r2 >= min_holdout_r2 -- did any form actually generalize?
    ranking: list[tuple[str, float]] = field(default_factory=list)  # (form, selection_r2), best first
    selection_r2: float = float("-inf")
    n_selection: int = 0


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
    wins validation and then passes an independent holdout confirmation.

    Args:
        simulate: the black-box ``x -> y`` (a real simulator wrapped to scalar in/out).
        input_range: ``(lo, hi)`` to sweep the input over.
        n_samples: total simulator evaluations across the sweep.
        holdout_fraction: fraction reserved for each of validation and independent confirmation.
            The remainder is used for fitting. A seeded partition keeps the two evaluation sets
            disjoint from fitting and from each other.
        min_holdout_r2: the bar the selected form's independent holdout R^2 must clear.
        forms: restrict the candidate library (default: all of :data:`CANDIDATE_FORMS` whose positivity
            requirement the input range satisfies).
        log_spaced: sweep inputs geometrically (for laws probed over decades, e.g. resistivity).
        seed: controls the deterministic train/validation/confirmation partition.

    Returns:
        A :class:`DiscoveredLaw` -- the winning form, its refitted params, train, selection, and
        independent holdout R^2 values. ``passed`` is false when confirmation misses the threshold.
    """
    from scipy.optimize import curve_fit

    if not callable(simulate):
        raise TypeError("simulate must be callable")
    if len(input_range) != 2:
        raise ValueError("input_range must contain exactly (lo, hi)")
    lo, hi = float(input_range[0]), float(input_range[1])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        raise ValueError("input_range bounds must be finite with lo < hi")
    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 1:
        raise ValueError("n_samples must be a positive integer")
    if not np.isfinite(holdout_fraction) or not 0.0 < holdout_fraction < 0.5:
        raise ValueError("holdout_fraction must be finite and strictly between 0 and 0.5")
    if not np.isfinite(min_holdout_r2) or min_holdout_r2 > 1.0:
        raise ValueError("min_holdout_r2 must be finite and no greater than 1")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if log_spaced and lo <= 0.0:
        raise ValueError("log_spaced discovery requires a strictly positive input range")

    candidate_names = tuple(CANDIDATE_FORMS) if forms is None else tuple(forms)
    if not candidate_names:
        raise ValueError("forms must contain at least one candidate")
    unknown_forms = [name for name in candidate_names if name not in CANDIDATE_FORMS]
    if unknown_forms:
        raise ValueError(f"unknown candidate forms: {unknown_forms}")
    if len(set(candidate_names)) != len(candidate_names):
        raise ValueError("forms must not contain duplicates")

    n_evaluation = int(round(n_samples * holdout_fraction))
    if n_evaluation < 2:
        raise ValueError("holdout_fraction and n_samples must allocate at least two validation points")
    n_fit = n_samples - 2 * n_evaluation
    max_params = max(CANDIDATE_FORMS[name][1] for name in candidate_names)
    if n_fit <= max_params:
        raise ValueError("n_samples leaves too few fitting points after independent validation and holdout splits")

    xs = np.geomspace(max(lo, 1e-9), hi, n_samples) if log_spaced else np.linspace(lo, hi, n_samples)
    ys = np.array([float(simulate(float(x))) for x in xs], dtype=float)
    if not np.all(np.isfinite(ys)):
        raise ValueError("simulate must return finite scalar values")

    order = np.random.RandomState(seed).permutation(n_samples)
    selection_idx = order[:n_evaluation]
    holdout_idx = order[n_evaluation : 2 * n_evaluation]
    train_idx = order[2 * n_evaluation :]
    x_tr, y_tr = xs[train_idx], ys[train_idx]
    x_sel, y_sel = xs[selection_idx], ys[selection_idx]
    x_ho, y_ho = xs[holdout_idx], ys[holdout_idx]

    positive_domain = lo > 0
    results: list[tuple[str, float, float, np.ndarray]] = []  # form, train_r2, selection_r2, parameters
    for name in candidate_names:
        fn, n_params, _expr, needs_pos = CANDIDATE_FORMS[name]
        if needs_pos and not positive_domain:
            continue
        try:
            popt, _ = curve_fit(fn, x_tr, y_tr, p0=np.ones(n_params), maxfev=10000)
        except Exception:  # noqa: BLE001 -- a non-converging form is simply not the law; skip it
            continue
        train_r2 = _r2(y_tr, fn(x_tr, *popt))
        selection_prediction = fn(x_sel, *popt)
        if not np.all(np.isfinite(selection_prediction)):
            continue
        selection_r2 = _r2(y_sel, selection_prediction)
        if np.isfinite(train_r2) and np.isfinite(selection_r2):
            results.append((name, train_r2, selection_r2, popt))

    if not results:
        return DiscoveredLaw(
            "none",
            "no candidate form fit",
            {},
            0.0,
            float("-inf"),
            len(x_tr),
            len(x_ho),
            False,
            [],
            float("-inf"),
            len(x_sel),
        )

    results.sort(key=lambda result: result[2], reverse=True)
    best_name, _initial_train, best_selection, initial_params = results[0]
    best_fn, n_params, _expression, _needs_positive = CANDIDATE_FORMS[best_name]
    x_refit = np.concatenate([x_tr, x_sel])
    y_refit = np.concatenate([y_tr, y_sel])
    best_params, _ = curve_fit(best_fn, x_refit, y_refit, p0=initial_params, maxfev=10000)
    train_prediction = best_fn(x_refit, *best_params)
    holdout_prediction = best_fn(x_ho, *best_params)
    if not np.all(np.isfinite(train_prediction)) or not np.all(np.isfinite(holdout_prediction)):
        raise ValueError("selected form produced non-finite predictions during independent confirmation")
    best_train = _r2(y_refit, train_prediction)
    best_holdout = _r2(y_ho, holdout_prediction)
    param_names = list("abcdefg")[:n_params]
    rounded_params = dict(zip(param_names, (round(float(param), 6) for param in best_params)))
    return DiscoveredLaw(
        form=best_name,
        expression=CANDIDATE_FORMS[best_name][2],
        params=rounded_params,
        train_r2=round(best_train, 4),
        holdout_r2=round(best_holdout, 4),
        n_train=len(x_refit),
        n_holdout=len(x_ho),
        passed=best_holdout >= min_holdout_r2,
        ranking=[(name, round(selection, 4)) for name, _train, selection, _params in results],
        selection_r2=round(best_selection, 4),
        n_selection=len(x_sel),
    )
