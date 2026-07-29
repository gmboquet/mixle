"""Predictive checks for the mixle PPL -- the model-criticism half of the Bayesian workflow.

A fit is not finished when the parameters are estimated; you check whether the model can reproduce the
data. These helpers implement the two standard checks:

* :func:`posterior_predictive_check` -- simulate replicate datasets from the *fitted* model, compare a
  test statistic on the replicates against its observed value, and report the Bayesian p-value
  ``P(T(y_rep) >= T(y_obs))``. A p-value near 0 or 1 means the model fails to capture that feature of
  the data (e.g. its skew or its tails); near 0.5 is a good fit.
* :func:`prior_predictive` -- simulate datasets from the *prior* (before seeing data) by drawing every
  prior parameter and sampling, so you can sanity-check that the prior implies plausible data.

Both accept a dict of named test statistics (callables on a dataset); the defaults cover location,
spread, and the extremes.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from numbers import Integral
from typing import Any

import numpy as np

from mixle.ppl.core import RandomVariable

_DEFAULT_STATS: dict[str, Callable[[np.ndarray], float]] = {
    "mean": lambda y: float(np.mean(y)),
    "std": lambda y: float(np.std(y)),
    "min": lambda y: float(np.min(y)),
    "max": lambda y: float(np.max(y)),
    "median": lambda y: float(np.median(y)),
}


def _stats(statistics: dict[str, Callable] | None) -> dict[str, Callable]:
    selected = _DEFAULT_STATS if statistics is None else statistics
    if not isinstance(selected, Mapping):
        raise TypeError("statistics must be a mapping of names to callables")
    if not selected:
        raise ValueError("statistics must contain at least one named statistic")
    result = {}
    for name, function in selected.items():
        if not isinstance(name, str) or not name:
            raise ValueError("statistic names must be non-empty strings")
        if not callable(function):
            raise TypeError(f"statistic {name!r} must be callable")
        result[name] = function
    return result


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _dataset(
    value: Any,
    name: str,
    *,
    rows: int | None = None,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a finite numeric dataset") from error
    if result.ndim == 0 or result.shape[0] == 0 or any(size == 0 for size in result.shape[1:]):
        raise ValueError(f"{name} must have shape (rows, ...non-empty observation axes)")
    if rows is not None and result.shape[0] != rows:
        raise ValueError(f"{name} must contain exactly {rows} observations; got {result.shape[0]}")
    if shape is not None and result.shape != shape:
        raise ValueError(f"{name} must match observed shape {shape}; got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _statistic_value(name: str, function: Callable, values: np.ndarray, source: str) -> float:
    try:
        raw = function(values)
    except Exception as error:
        raise ValueError(f"statistic {name!r} failed on {source}: {error}") from error
    array = np.asarray(raw)
    if array.ndim != 0:
        raise ValueError(f"statistic {name!r} must return one scalar for {source}; got shape {array.shape}")
    try:
        result = float(array)
    except (TypeError, ValueError) as error:
        raise TypeError(f"statistic {name!r} must return a numeric scalar for {source}") from error
    if not np.isfinite(result):
        raise ValueError(f"statistic {name!r} returned a non-finite value for {source}")
    return result


def _evaluate_statistics(
    statistics: dict[str, Callable],
    values: np.ndarray,
    source: str,
) -> dict[str, float]:
    return {name: _statistic_value(name, function, values, source) for name, function in statistics.items()}


def _draw_one_prior_parameter(
    prior: RandomVariable,
    rng: np.random.RandomState,
) -> Any:
    dist = _draw_prior_dist(prior, rng)
    raw = dist.sampler(seed=int(rng.randint(1, 2**31))).sample(1)
    try:
        draws = np.asarray(raw)
    except (TypeError, ValueError) as error:
        raise TypeError(f"prior parameter {prior._name or '<unnamed>'!r} produced non-array draws") from error
    if draws.ndim == 0:
        draw = draws
    elif draws.shape[0] == 1:
        draw = draws[0]
    else:
        raise ValueError(
            f"prior parameter {prior._name or '<unnamed>'!r} ignored sample size 1; got shape {draws.shape}"
        )
    try:
        numeric = np.asarray(draw, dtype=float)
    except (TypeError, ValueError) as error:
        raise TypeError(f"prior parameter {prior._name or '<unnamed>'!r} must produce numeric draws") from error
    if not np.all(np.isfinite(numeric)):
        raise ValueError(f"prior parameter {prior._name or '<unnamed>'!r} produced a non-finite draw")
    if numeric.ndim == 0:
        return numeric.item()
    return numeric.copy()


def posterior_predictive_check(
    fitted: RandomVariable,
    data: Sequence[Any],
    *,
    statistics: dict[str, Callable[[np.ndarray], float]] | None = None,
    n_rep: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Posterior predictive check of a fitted PPL model against ``data``.

    Draws ``n_rep`` replicate datasets (each the size of ``data``) from ``fitted.predict`` -- which
    integrates over parameter uncertainty for a Bayesian fit (conjugate/mcmc/hmc) and is the plug-in
    predictive for a point fit (em/map) -- evaluates each named statistic on every replicate and on the
    observed data, and returns the Bayesian p-value per statistic.

    Returns ``{'observed', 'replicated', 'p_value', 'n_rep'}``: ``observed[name]`` the statistic on the
    data, ``replicated[name]`` its ``(n_rep,)`` replicate values, ``p_value[name] = P(T_rep >= T_obs)``.
    """
    stats = _stats(statistics)
    n_rep = _positive_int(n_rep, "n_rep")
    obs_arr = _dataset(data, "observed data")
    n_obs = obs_arr.shape[0]
    predict = getattr(fitted, "predict", None)
    if not callable(predict):
        raise TypeError("fitted must expose a callable predict(size, rng=...) method")
    rng = np.random.RandomState(seed)
    observed = _evaluate_statistics(stats, obs_arr, "observed data")
    replicated = {k: np.empty(n_rep, dtype=float) for k in stats}
    for r in range(n_rep):
        sim = _dataset(
            predict(n_obs, rng=rng),
            f"posterior predictive replicate {r}",
            rows=n_obs,
            shape=obs_arr.shape,
        )
        values = _evaluate_statistics(stats, sim, f"posterior predictive replicate {r}")
        for name, value in values.items():
            replicated[name][r] = value
    p_value = {k: float(np.mean(replicated[k] >= observed[k])) for k in stats}
    return {"observed": observed, "replicated": replicated, "p_value": p_value, "n_rep": n_rep}


def _draw_prior_dist(rv: RandomVariable, rng: np.random.RandomState):
    """Lower ``rv`` to a concrete mixle distribution, drawing every prior-distribution parameter slot.

    Recurses through hyperpriors: a slot holding a ``RandomVariable`` is replaced by a single draw from
    that prior (itself resolved the same way), then the family builds a concrete distribution.
    """
    args = []
    for a in rv._args:
        if isinstance(a, RandomVariable) and a._kind == "sample":
            args.append(_draw_one_prior_parameter(a, rng))
        else:
            args.append(a)
    if rv._family is None:
        raise ValueError("prior_predictive needs a distribution-valued model.")
    return rv._family.make_dist(tuple(args), rv._name)


def prior_predictive(
    model: RandomVariable,
    size: int,
    *,
    n_rep: int = 1000,
    statistics: dict[str, Callable[[np.ndarray], float]] | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Prior predictive simulation: ``n_rep`` datasets of ``size`` drawn from ``model``'s prior.

    For each replicate it draws every prior parameter (and hyperparameter) and then ``size`` data
    points, so the result reflects what the model believes *before* seeing data -- the check that a
    prior is neither absurdly tight nor absurdly diffuse. Returns
    ``{'replicated': {stat: (n_rep,)}, 'samples': (n_rep, size), 'n_rep'}`` with the per-replicate
    statistics and the raw simulated datasets.
    """
    stats = _stats(statistics)
    if not isinstance(model, RandomVariable):
        raise TypeError("model must be a PPL RandomVariable")
    size = _positive_int(size, "size")
    n_rep = _positive_int(n_rep, "n_rep")
    rng = np.random.RandomState(seed)
    sample_replicates = []
    replicated = {k: np.empty(n_rep, dtype=float) for k in stats}
    replicate_shape = None
    for r in range(n_rep):
        dist = _draw_prior_dist(model, rng)
        sim = _dataset(
            dist.sampler(seed=int(rng.randint(1, 2**31))).sample(size),
            f"prior predictive replicate {r}",
            rows=size,
        )
        if replicate_shape is None:
            replicate_shape = sim.shape
        elif sim.shape != replicate_shape:
            raise ValueError(f"prior predictive replicate {r} changed shape from {replicate_shape} to {sim.shape}")
        sample_replicates.append(sim)
        values = _evaluate_statistics(stats, sim, f"prior predictive replicate {r}")
        for name, value in values.items():
            replicated[name][r] = value
    samples = np.stack(sample_replicates, axis=0)
    return {"replicated": replicated, "samples": samples, "n_rep": n_rep}


def prior_predictive_check(
    model: RandomVariable,
    data: Sequence[Any],
    *,
    statistics: dict[str, Callable[[np.ndarray], float]] | None = None,
    n_rep: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Prior predictive check: where the observed statistics sit in the prior predictive distribution.

    Like :func:`posterior_predictive_check` but the replicates come from the prior (via
    :func:`prior_predictive`), so a p-value near 0 or 1 flags a prior that is inconsistent with the data
    before any fitting -- often a sign the prior is mis-scaled.
    """
    stats = _stats(statistics)
    obs_arr = _dataset(data, "observed data")
    n_rep = _positive_int(n_rep, "n_rep")
    pp = prior_predictive(model, obs_arr.shape[0], n_rep=n_rep, statistics=stats, seed=seed)
    observed = _evaluate_statistics(stats, obs_arr, "observed data")
    p_value = {k: float(np.mean(pp["replicated"][k] >= observed[k])) for k in stats}
    return {"observed": observed, "replicated": pp["replicated"], "p_value": p_value, "n_rep": n_rep}


__all__ = ["posterior_predictive_check", "prior_predictive", "prior_predictive_check"]
