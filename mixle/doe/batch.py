"""Rigorous batch (multi-point) Bayesian optimization for parallel experiment campaigns.

The kriging-believer batch in :mod:`mixle.doe.bayesopt` fantasizes the posterior *mean* at each pick --
low-cost, but it discards the correlation between the batch points and the posterior uncertainty they
share, so it can place near-duplicate points. This module proposes batches under the *true joint* GP
posterior:

* :func:`monte_carlo_qei` -- the multi-point Expected Improvement ``E[max(best - min_i f(x_i), 0)]`` of
  a candidate batch with joint posterior ``N(mu, Sigma)``, estimated by Monte Carlo (Ginsbourger et al.
  2010); the exact generalization of EI to ``q`` simultaneous evaluations.
* :func:`propose_qei_batch` -- greedily builds a ``q``-point batch, each new point maximizing the q-EI
  of the batch-so-far-plus-candidate under the joint posterior. Rigorous (no fantasies) and tractable.
* :func:`propose_local_penalization` -- the Gonzalez et al. (2016) local-penalization batch: pick
  points one at a time but multiply the acquisition by a soft exclusion zone around the pending picks,
  sized by a Lipschitz estimate of the objective. Scales to large ``q`` without joint sampling.

``monte_carlo_qei`` is pure NumPy. The proposal drivers fit the torch GP surrogate (like the rest of
the BO layer), so they require PyTorch.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.random import RandomState

from mixle.doe.bayesopt import _fit_surrogate, _validate_xy
from mixle.doe.designs import Bounds, _as_bounds, _as_rng, latin_hypercube


def _safe_cholesky(sigma: np.ndarray) -> np.ndarray:
    """Cholesky of a posterior covariance, adding escalating jitter; diagonal fallback if all fails."""
    q = sigma.shape[0]
    base = 1e-10 * max(1.0, float(np.trace(sigma)) / q)
    jit = base
    for _ in range(7):
        try:
            return np.linalg.cholesky(sigma + jit * np.eye(q))
        except np.linalg.LinAlgError:
            jit *= 10.0
    return np.diag(np.sqrt(np.maximum(np.diag(sigma), 0.0)))


def monte_carlo_qei(
    mean: Any, cov: Any, best: float, *, maximize: bool = False, samples: int = 512, seed: int | RandomState = 0
) -> float:
    """Monte-Carlo multi-point Expected Improvement of a batch with joint posterior ``N(mean, cov)``.

    Draws ``samples`` joint posterior realizations of the ``q`` batch points and averages the batch
    improvement over the incumbent ``best`` -- ``max(best - min_i f_i, 0)`` for minimization, or
    ``max(max_i f_i - best, 0)`` for maximization. For ``q = 1`` this reduces to ordinary EI.
    """
    mu = np.asarray(mean, dtype=np.float64).ravel()
    sigma = np.atleast_2d(np.asarray(cov, dtype=np.float64))
    q = mu.size
    rng = seed if isinstance(seed, RandomState) else RandomState(seed)
    chol = _safe_cholesky(sigma)
    draws = mu[None, :] + rng.standard_normal((int(samples), q)) @ chol.T
    if maximize:
        improvement = np.maximum(draws.max(axis=1) - best, 0.0)
    else:
        improvement = np.maximum(best - draws.min(axis=1), 0.0)
    return float(improvement.mean())


def propose_qei_batch(
    x: Any,
    y: Any,
    bounds: Bounds,
    q: int,
    *,
    n_candidates: int = 256,
    mc_samples: int = 256,
    maximize: bool = False,
    seed: int | RandomState | None = None,
    gp: Any = None,
    fit_kwargs: dict[str, Any] | None = None,
) -> np.ndarray:
    """Propose a ``q``-point batch by greedy Monte-Carlo q-EI under the joint GP posterior.

    Fits the GP to ``(x, y)``, then builds the batch one point at a time: each new point is the
    Latin-hypercube candidate maximizing the q-EI of ``{batch so far} + candidate`` (evaluated with
    *common random numbers* so the greedy comparison is fair). Because the joint posterior is used, an
    already-chosen point lowers the marginal value of nearby candidates, so the batch self-diversifies
    without any fantasized observations. Returns a ``(q, d)`` array.
    """
    if int(q) <= 0:
        raise ValueError("q must be positive.")
    b = _as_bounds(bounds)
    rng = _as_rng(seed)
    xs, ys = _validate_xy(x, y)
    gp = _fit_surrogate(xs, ys, gp, fit_kwargs)
    best = float(ys.max() if maximize else ys.min())
    candidates = latin_hypercube(b, int(n_candidates), rng)
    mc_seed = int(rng.randint(2**31))  # common random numbers across candidates and steps
    batch: list[np.ndarray] = []
    for _ in range(int(q)):
        best_c, best_val = None, -np.inf
        for c in candidates:
            pts = np.vstack([*batch, c]) if batch else c[None, :]
            mean, cov = gp.predict(xs, ys, pts, return_cov=True)
            val = monte_carlo_qei(mean, cov, best, maximize=maximize, samples=mc_samples, seed=mc_seed)
            if val > best_val:
                best_val, best_c = val, c
        batch.append(np.asarray(best_c, dtype=np.float64))
    return np.asarray(batch)


def propose_local_penalization(
    x: Any,
    y: Any,
    bounds: Bounds,
    q: int,
    *,
    n_candidates: int = 512,
    maximize: bool = False,
    seed: int | RandomState | None = None,
    gp: Any = None,
    fit_kwargs: dict[str, Any] | None = None,
) -> np.ndarray:
    """Propose a ``q``-point batch by local penalization (Gonzalez et al. 2016).

    Picks points sequentially from a single GP fit (no refitting): each pick maximizes the expected
    improvement multiplied by a soft exclusion factor around every pending pick. The exclusion radius is
    set from a Lipschitz estimate ``L`` of the objective (the largest posterior-mean gradient over the
    candidates) and the gap to the incumbent, so the penalty is principled rather than a fixed distance.
    Cheaper than q-EI for large ``q`` (one GP fit, closed-form penalties). Returns a ``(q, d)`` array.
    """
    from scipy.stats import norm

    if int(q) <= 0:
        raise ValueError("q must be positive.")
    if int(q) > int(n_candidates):
        # once every candidate's merit is set to -inf (line 174), np.argmax deterministically returns
        # index 0 again (ties broken by first occurrence) -- the batch would silently contain
        # duplicate points instead of raising. Name the actual constraint instead.
        raise ValueError(f"propose_local_penalization requires q <= n_candidates (q={q}, n_candidates={n_candidates}).")
    b = _as_bounds(bounds)
    rng = _as_rng(seed)
    xs, ys = _validate_xy(x, y)
    gp = _fit_surrogate(xs, ys, gp, fit_kwargs)
    best = float(ys.max() if maximize else ys.min())
    cand = latin_hypercube(b, int(n_candidates), rng)
    mean, std = _posterior_mean_std(gp, xs, ys, cand)

    # Lipschitz estimate of the (minimization-oriented) objective: the largest posterior-mean slope
    # |mu(a) - mu(b)| / ||a - b|| over a subsample of candidate pairs -- a valid empirical lower bound.
    obj_mean = -mean if maximize else mean  # penalize in a minimization sense
    sub = rng.choice(cand.shape[0], size=min(cand.shape[0], 64), replace=False)
    lipschitz = 1e-6
    for i in sub:
        dd = np.linalg.norm(cand[sub] - cand[i], axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            slopes = np.abs(obj_mean[sub] - obj_mean[i]) / dd
        slopes[~np.isfinite(slopes)] = 0.0
        lipschitz = max(lipschitz, float(slopes.max()))
    # Optimistic estimate of the global minimum: one posterior-sd below the best mean. Using the plain
    # running min would give the best pick a zero-radius exclusion ball (-> clustering); the optimistic
    # bound keeps every pick's exclusion radius r_j = mu(x_j) - M_hat strictly positive.
    m_star = float((obj_mean - std).min())

    signed = (mean - best) if maximize else (best - mean)
    z = signed / np.maximum(std, 1e-12)
    ei = np.maximum(signed, 0.0) * norm.cdf(z) + std * norm.pdf(z)
    merit = np.log(np.maximum(ei, 1e-300))  # log-acquisition so penalties multiply as sums

    diag = float(np.linalg.norm(b[:, 1] - b[:, 0]))
    batch: list[np.ndarray] = []
    for _ in range(int(q)):
        k = int(np.argmax(merit))
        batch.append(cand[k].copy())
        # Exclude a ball around the new pick whose radius is the Lipschitz reach toward the (optimistic)
        # minimum, (mu(x_j) - M_hat)/L, floored at 5% of the domain diagonal so the batch always spreads.
        rj = max((obj_mean[k] - m_star) / lipschitz, 0.05 * diag)
        dist = np.linalg.norm(cand - cand[k], axis=1)
        phi = norm.cdf((dist - rj) / (0.3 * rj))  # ~0 inside the ball, ~1 beyond it
        merit = merit + np.log(np.maximum(phi, 1e-12))
        merit[k] = -np.inf
    return np.asarray(batch)


def _posterior_mean_std(gp: Any, xs: np.ndarray, ys: np.ndarray, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Posterior mean and marginal std at ``pts`` from a fitted surrogate."""
    mean, cov = gp.predict(xs, ys, pts, return_cov=True)
    var = np.clip(np.diag(np.atleast_2d(np.asarray(cov, dtype=np.float64))), 0.0, None)
    return np.asarray(mean, dtype=np.float64).ravel(), np.sqrt(var)


__all__ = ["monte_carlo_qei", "propose_qei_batch", "propose_local_penalization"]
