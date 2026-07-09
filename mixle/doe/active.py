"""Active learning and Bayesian optimal design: sequential design to *learn*, not to optimize.

Where Bayesian optimization places points to find an optimum, **active learning** places points to make
a surrogate accurate everywhere, and **Bayesian optimal design** places them to learn model parameters.

Active learning (GP surrogate):
* :func:`alm_scores` -- Active Learning MacKay: the posterior predictive variance (pick the most
  uncertain point). Low-cost but myopic.
* :func:`alc_scores` -- Active Learning Cohn / IMSE: the *integrated* reduction in posterior variance a
  candidate would buy over a reference set -- the principled criterion.
* :func:`active_learning_design` -- the sequential loop that grows an accurate surrogate.

Bayesian optimal design (parametric model):
* :func:`expected_information_gain_linear` -- the exact EIG of a linear-Gaussian model (= Bayesian
  D-optimality), in closed form.
* :func:`expected_information_gain_nmc` -- the nested-Monte-Carlo EIG for a general nonlinear simulator,
  from a prior sampler and a log-likelihood.

The GP-based functions fit the torch surrogate; the EIG functions are pure NumPy.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.random import RandomState

from mixle.doe.bayesopt import _fit_surrogate, _validate_xy
from mixle.doe.designs import Bounds, _as_bounds, _as_rng, latin_hypercube


def alm_scores(gp: Any, x: np.ndarray, y: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Active Learning MacKay scores: the GP posterior predictive variance at each candidate."""
    _, cov = gp.predict(x, y, candidates, return_cov=True)
    return np.clip(np.diag(np.atleast_2d(np.asarray(cov, dtype=np.float64))), 0.0, None)


def alc_scores(gp: Any, x: np.ndarray, y: np.ndarray, candidates: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Active Learning Cohn / IMSE scores: integrated posterior-variance reduction per candidate.

    Adding candidate ``c`` reduces the posterior variance at a reference point ``r`` by
    ``cov_post(r, c)^2 / var_post(c)``; this returns the sum over the reference set (the negative change
    in integrated MSE), so the maximizer is the most globally informative next point.
    """
    pts = np.vstack([reference, candidates])
    nr = reference.shape[0]
    _, cov = gp.predict(x, y, pts, return_cov=True)
    cov = np.atleast_2d(np.asarray(cov, dtype=np.float64))
    cov_rc = cov[:nr, nr:]  # (n_ref, n_cand) posterior cov between reference and candidates
    var_c = np.clip(np.diag(cov)[nr:], 1e-12, None)
    return np.asarray((cov_rc**2).sum(axis=0) / var_c)


def propose_active_learning(
    x: Any,
    y: Any,
    bounds: Bounds,
    *,
    method: str = "alc",
    n_candidates: int = 512,
    n_reference: int = 256,
    seed: int | RandomState | None = None,
    gp: Any = None,
    fit_kwargs: dict[str, Any] | None = None,
) -> np.ndarray:
    """Propose the next active-learning point (``method='alc'`` IMSE, or ``'alm'`` max variance)."""
    if int(n_candidates) <= 0:
        raise ValueError("n_candidates must be positive.")
    b = _as_bounds(bounds)
    rng = _as_rng(seed)
    xs, ys = _validate_xy(x, y)
    gp = _fit_surrogate(xs, ys, gp, fit_kwargs)
    cand = latin_hypercube(b, int(n_candidates), rng)
    if method == "alm":
        scores = alm_scores(gp, xs, ys, cand)
    elif method == "alc":
        scores = alc_scores(gp, xs, ys, cand, latin_hypercube(b, int(n_reference), rng))
    else:
        raise ValueError("method must be 'alc' or 'alm'.")
    return cand[int(np.argmax(scores))]


def active_learning_design(
    objective: Callable[[np.ndarray], float],
    bounds: Bounds,
    *,
    n_init: int | None = None,
    max_evals: int = 40,
    method: str = "alc",
    seed: int | RandomState | None = None,
    fit_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Sequentially place points to maximize surrogate accuracy, returning the design and its responses.

    Starts from a Latin-hypercube design, then repeatedly fits the GP and adds the most informative
    point (by ``method``) until ``max_evals`` evaluations. Returns ``{'X', 'Y'}`` -- a design tailored to
    learn ``objective`` well everywhere, not just near an optimum.
    """
    b = _as_bounds(bounds)
    d = b.shape[0]
    rng = _as_rng(seed)
    n_init = int(n_init) if n_init else 2 * d
    x_all = latin_hypercube(b, n_init, rng)
    y_all = np.array([float(objective(np.asarray(p, dtype=np.float64))) for p in x_all], dtype=np.float64)
    while y_all.shape[0] < max_evals:
        xn = propose_active_learning(x_all, y_all, b, method=method, seed=rng, fit_kwargs=fit_kwargs)
        x_all = np.vstack([x_all, xn])
        y_all = np.append(y_all, float(objective(np.asarray(xn, dtype=np.float64))))
    return {"X": x_all, "Y": y_all}


def expected_information_gain_linear(
    model_matrix: np.ndarray, *, noise: float = 1.0, prior_cov: np.ndarray | None = None
) -> float:
    """Exact expected information gain of a linear-Gaussian design ``y = F.theta + eps`` (= Bayesian D-opt).

    For a Gaussian prior ``theta ~ N(0, Sigma0)`` and observation noise ``eps ~ N(0, noise^2 I)``, the
    mutual information between the data and ``theta`` is ``0.5 * log det(I + noise^-2 Sigma0 F^T F)``.
    ``model_matrix`` is the ``(n, p)`` design matrix ``F`` (e.g. from
    :func:`mixle.doe.optimal.polynomial_features`). Higher EIG = a more informative design.
    """
    f = np.asarray(model_matrix, dtype=np.float64)
    p = f.shape[1]
    sigma0 = np.eye(p) if prior_cov is None else np.asarray(prior_cov, dtype=np.float64)
    m = np.eye(p) + (sigma0 @ (f.T @ f)) / (float(noise) ** 2)
    sign, logdet = np.linalg.slogdet(m)
    return float(0.5 * logdet) if sign > 0 else -np.inf


def expected_information_gain_nmc(
    prior_sampler: Callable[[RandomState, int], np.ndarray],
    log_likelihood: Callable[[np.ndarray, np.ndarray], np.ndarray],
    simulate: Callable[[np.ndarray, RandomState], np.ndarray],
    *,
    n_outer: int = 256,
    n_inner: int = 256,
    seed: int | RandomState | None = None,
) -> float:
    """Nested-Monte-Carlo expected information gain for a general (nonlinear) design.

    Estimates ``EIG = E_{theta, y}[ log p(y|theta) - log E_{theta'}[p(y|theta')] ]`` (Ryan 2003): draw
    outer ``theta_i ~ prior`` and ``y_i ~ p(y|theta_i)`` via ``simulate``; the inner expectation is a
    mean over ``n_inner`` prior draws of ``exp(log_likelihood(theta', y_i))``. ``prior_sampler(rng, n)``
    returns ``(n, k)`` parameter draws; ``log_likelihood(thetas, y)`` returns a log-density per row of
    ``thetas`` at the single observation ``y``; ``simulate(theta, rng)`` draws one ``y`` given ``theta``.
    """
    if int(n_outer) <= 0 or int(n_inner) <= 0:
        raise ValueError("expected_information_gain_nmc requires n_outer > 0 and n_inner > 0.")
    rng = seed if isinstance(seed, RandomState) else RandomState(seed)
    thetas_outer = np.asarray(prior_sampler(rng, int(n_outer)), dtype=np.float64)
    total = 0.0
    for theta_i in thetas_outer:
        y_i = np.asarray(simulate(theta_i, rng), dtype=np.float64)
        ll_true = float(np.atleast_1d(log_likelihood(theta_i[None, :], y_i))[0])
        thetas_inner = np.asarray(prior_sampler(rng, int(n_inner)), dtype=np.float64)
        ll_inner = np.asarray(log_likelihood(thetas_inner, y_i), dtype=np.float64).ravel()
        log_evidence = float(np.logaddexp.reduce(ll_inner) - np.log(ll_inner.size))
        total += ll_true - log_evidence
    return float(total / thetas_outer.shape[0])


__all__ = [
    "alm_scores",
    "alc_scores",
    "propose_active_learning",
    "active_learning_design",
    "expected_information_gain_linear",
    "expected_information_gain_nmc",
]
