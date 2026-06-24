"""Predictive checks for the pysp PPL -- the model-criticism half of the Bayesian workflow.

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

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from pysp.ppl.core import RandomVariable

_DEFAULT_STATS: dict[str, Callable[[np.ndarray], float]] = {
    "mean": lambda y: float(np.mean(y)),
    "std": lambda y: float(np.std(y)),
    "min": lambda y: float(np.min(y)),
    "max": lambda y: float(np.max(y)),
    "median": lambda y: float(np.median(y)),
}


def _stats(statistics: dict[str, Callable] | None) -> dict[str, Callable]:
    return dict(_DEFAULT_STATS if statistics is None else statistics)


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
    obs_arr = np.asarray(data, dtype=float)
    n_obs = len(obs_arr)
    if n_obs == 0:
        raise ValueError("data is empty.")
    rng = np.random.RandomState(seed)
    observed = {k: float(fn(obs_arr)) for k, fn in stats.items()}
    replicated = {k: np.empty(int(n_rep), dtype=float) for k in stats}
    for r in range(int(n_rep)):
        sim = np.asarray(fitted.predict(n_obs, rng=rng), dtype=float).ravel()
        for k, fn in stats.items():
            replicated[k][r] = float(fn(sim))
    p_value = {k: float(np.mean(replicated[k] >= observed[k])) for k in stats}
    return {"observed": observed, "replicated": replicated, "p_value": p_value, "n_rep": int(n_rep)}


def _draw_prior_dist(rv: RandomVariable, rng: np.random.RandomState):
    """Lower ``rv`` to a concrete pysp distribution, drawing every prior-distribution parameter slot.

    Recurses through hyperpriors: a slot holding a ``RandomVariable`` is replaced by a single draw from
    that prior (itself resolved the same way), then the family builds a concrete distribution.
    """
    args = []
    for a in rv._args:
        if isinstance(a, RandomVariable) and a._kind == "sample":
            sub = _draw_prior_dist(a, rng)
            draw = np.atleast_1d(sub.sampler(seed=int(rng.randint(1, 2**31))).sample(1))[0]
            args.append(float(draw))
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
    rng = np.random.RandomState(seed)
    samples = np.empty((int(n_rep), int(size)), dtype=float)
    replicated = {k: np.empty(int(n_rep), dtype=float) for k in stats}
    for r in range(int(n_rep)):
        dist = _draw_prior_dist(model, rng)
        sim = np.asarray(dist.sampler(seed=int(rng.randint(1, 2**31))).sample(int(size)), dtype=float).ravel()
        samples[r] = sim
        for k, fn in stats.items():
            replicated[k][r] = float(fn(sim))
    return {"replicated": replicated, "samples": samples, "n_rep": int(n_rep)}


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
    obs_arr = np.asarray(data, dtype=float)
    pp = prior_predictive(model, len(obs_arr), n_rep=n_rep, statistics=stats, seed=seed)
    observed = {k: float(fn(obs_arr)) for k, fn in stats.items()}
    p_value = {k: float(np.mean(pp["replicated"][k] >= observed[k])) for k in stats}
    return {"observed": observed, "replicated": pp["replicated"], "p_value": p_value, "n_rep": int(n_rep)}


__all__ = ["posterior_predictive_check", "prior_predictive", "prior_predictive_check"]
