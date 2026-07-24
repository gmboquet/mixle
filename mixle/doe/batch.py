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

import warnings
from typing import Any

import numpy as np
from numpy.random import RandomState

from mixle.doe.bayesopt import _fit_surrogate, _validate_xy
from mixle.doe.designs import Bounds, _as_bounds, _as_rng, latin_hypercube

_PSD_EIGENVALUE_RATIO = 1e-9
"""Relative-tolerance bound (matching ``mixle.inference.belief.GaussianBelief``'s PSD gate) on how
negative ``_safe_cholesky``'s worst eigenvalue may be before a covariance is refused outright rather
than jitter-healed. A genuine posterior covariance is PSD in exact arithmetic; float roundoff can only
push it a few ULPs south of zero, so a worst eigenvalue this far below zero (relative to the matrix's
own eigenvalue scale) means the input is not a covariance at all -- no amount of principled jitter makes
that the RIGHT fix, only a plausible-looking wrong one."""


def _require_positive_int(value: Any, name: str) -> int:
    """Validate ``value`` is an exact positive integer, rejecting nonpositive and fractional counts.

    ``int(value)`` alone silently truncates (``2.7`` becomes ``2``, no error); this instead requires the
    float and int forms to agree, so a fractional count is named as invalid rather than quietly rounded.
    """
    try:
        as_float = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.") from exc
    as_int = int(as_float)
    if as_int <= 0 or as_float != as_int:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return as_int


def _safe_cholesky(sigma: np.ndarray) -> np.ndarray:
    """Cholesky of a posterior covariance, for sampling the ``q`` batch points' TRUE JOINT distribution.

    This is the mechanism that makes q-EI/Thompson batches genuinely joint (as opposed to the cheaper
    independent approximation): a caller's requested dependence structure must never be silently swapped
    out, so this validates before it heals and heals before it gives up.

    * ``sigma`` finite and PSD within ``_PSD_EIGENVALUE_RATIO`` of its own eigenvalue scale, but merely
      numerically singular (e.g. a duplicate/near-duplicate candidate pair shares an almost-perfectly-
      correlated row/column): recoverable. An escalating diagonal jitter, starting at ``1e-10`` of the
      matrix's eigenvalue scale and backing off by 10x for up to 7 attempts, nudges it just inside the
      numerically-decomposable PD cone -- the SAME dependence structure, only its conditioning changes.
      A ``RuntimeWarning`` reports exactly how much jitter was needed, so this is never invisible.
    * ``sigma`` not finite, or indefinite well beyond float noise (e.g. ``[[1, 2], [2, 1]]``, whose
      eigenvalues are ``3`` and ``-1``: no jitter this small makes that PD): not a valid covariance at
      all. Raises ``ValueError`` immediately -- this never falls through to a diagonal-only fallback,
      which would silently swap the caller's TRUE JOINT request for an INDEPENDENT approximation while
      still returning a plausible-looking number.

    Raises:
        ValueError: if ``sigma`` is non-finite, fails the PSD check, or (in a pathological case that
            should not occur for anything that passed the PSD check) still fails to factor after the
            full jitter budget is exhausted.
    """
    sigma = np.asarray(sigma, dtype=np.float64)
    if not np.all(np.isfinite(sigma)):
        raise ValueError("covariance is not finite (contains NaN/Inf); cannot sample the joint posterior.")
    q = sigma.shape[0]
    sym = 0.5 * (sigma + sigma.T)  # symmetrize first: cholesky only reads one triangle, so an asymmetric-
    # but-PD input would otherwise "succeed" against a matrix that silently differs from what was passed.
    evals = np.linalg.eigvalsh(sym)
    scale = float(np.abs(evals).max())
    if evals.min() < -_PSD_EIGENVALUE_RATIO * scale:
        raise ValueError(
            "covariance is not positive semi-definite (worst eigenvalue "
            f"{evals.min():.6g} vs eigenvalue scale {scale:.6g}); refusing to silently substitute an "
            "independent (diagonal-only) approximation for a fundamentally invalid joint covariance."
        )
    base = 1e-10 * max(scale, 1e-12)
    jitters = [0.0] + [base * (10.0**i) for i in range(7)]  # attempt 0 unperturbed, then 7 escalating retries
    eye = np.eye(q)
    for attempt, jit in enumerate(jitters):
        try:
            chol = np.linalg.cholesky(sym + jit * eye)
        except np.linalg.LinAlgError:
            continue
        if jit > 0.0:
            warnings.warn(
                "joint posterior covariance was numerically singular under a direct Cholesky "
                f"factorization; healed with diagonal jitter={jit:.3g} ({jit / scale:.3g} of the "
                f"eigenvalue scale {scale:.3g}) after {attempt} failed attempt(s). The dependence "
                "structure is unchanged -- only the numerical conditioning was adjusted.",
                RuntimeWarning,
                stacklevel=2,
            )
        return chol
    raise ValueError(
        "covariance passed the positive-semidefinite check (worst eigenvalue "
        f"{evals.min():.6g} vs eigenvalue scale {scale:.6g}) but Cholesky still failed after escalating "
        f"jitter up to {jitters[-1]:.3g}; refusing to silently substitute a diagonal (independent) "
        "approximation."
    )


def monte_carlo_qei(
    mean: Any, cov: Any, best: float, *, maximize: bool = False, samples: int = 512, seed: int | RandomState = 0
) -> float:
    """Monte-Carlo multi-point Expected Improvement of a batch with joint posterior ``N(mean, cov)``.

    Draws ``samples`` joint posterior realizations of the ``q`` batch points and averages the batch
    improvement over the incumbent ``best`` -- ``max(best - min_i f_i, 0)`` for minimization, or
    ``max(max_i f_i - best, 0)`` for maximization. For ``q = 1`` this reduces to ordinary EI.

    Raises:
        ValueError: if ``mean`` is empty, ``cov``'s shape does not match ``mean``, either contains a
            non-finite value, ``cov`` is not a valid (finite, PSD-within-tolerance) covariance, or
            ``samples`` is not a positive integer. See :func:`_safe_cholesky` for the covariance
            validation/healing policy -- it never silently downgrades a true joint request to an
            independent one.
    """
    mu = np.asarray(mean, dtype=np.float64).ravel()
    q = mu.size
    if q == 0:
        raise ValueError("mean must have at least one element (q >= 1).")
    sigma = np.atleast_2d(np.asarray(cov, dtype=np.float64))
    if sigma.shape != (q, q):
        raise ValueError(f"cov must be a ({q}, {q}) matrix matching mean's length {q}; got shape {sigma.shape}.")
    if not np.all(np.isfinite(mu)):
        raise ValueError("mean contains non-finite values (NaN/Inf); cannot compute q-EI.")
    n_samples = _require_positive_int(samples, "samples")
    rng = seed if isinstance(seed, RandomState) else RandomState(seed)
    chol = _safe_cholesky(sigma)
    draws = mu[None, :] + rng.standard_normal((n_samples, q)) @ chol.T
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

    Raises:
        ValueError: if ``q``, ``n_candidates``, or ``mc_samples`` is not a positive integer, if there are
            zero observations, or if every candidate scores a non-finite q-EI merit at some batch step
            (rather than silently leaving that step's pick undefined).
    """
    q = _require_positive_int(q, "q")
    n_candidates = _require_positive_int(n_candidates, "n_candidates")
    mc_samples = _require_positive_int(mc_samples, "mc_samples")
    b = _as_bounds(bounds)
    rng = _as_rng(seed)
    xs, ys = _validate_xy(x, y)
    if ys.size == 0:
        # ys.min()/ys.max() on an empty ys crashes with an opaque "zero-size array" ValueError. Same
        # documented path as bayesopt._propose_one: BayesianOptimizer.ask(q) with q > n_init before any
        # tell() calls propose_qei_batch with zero observations -- there is no incumbent to score q-EI
        # against yet, so name that clearly instead of a generic numpy crash.
        raise ValueError("cannot propose a q-EI batch with zero observations; call tell() first.")
    gp = _fit_surrogate(xs, ys, gp, fit_kwargs)
    best = float(ys.max() if maximize else ys.min())
    candidates = latin_hypercube(b, n_candidates, rng)
    if candidates.shape[0] != n_candidates:
        raise ValueError(f"latin_hypercube returned {candidates.shape[0]} candidates, expected {n_candidates}.")
    mc_seed = int(rng.randint(2**31))  # common random numbers across candidates and steps
    batch: list[np.ndarray] = []
    for step in range(q):
        best_c, best_val = None, -np.inf
        for c in candidates:
            pts = np.vstack([*batch, c]) if batch else c[None, :]
            mean, cov = gp.predict(xs, ys, pts, return_cov=True)
            val = monte_carlo_qei(mean, cov, best, maximize=maximize, samples=mc_samples, seed=mc_seed)
            if val > best_val:  # NaN val never wins: comparisons against NaN are always False
                best_val, best_c = val, c
        if best_c is None:
            # every candidate's q-EI merit was non-finite (NaN) -- e.g. a NaN incumbent from a NaN in y,
            # or a degenerate GP posterior. Silently falling through would append None -> np.asarray(None,
            # dtype=float64) is a 0-d NaN array, corrupting every subsequent step's np.vstack. Name it.
            raise ValueError(
                f"no candidate produced a finite q-EI merit at batch step {step + 1}/{q}; all "
                f"{n_candidates} candidates scored non-finite (NaN) -- check the GP posterior mean/"
                "covariance and the observed y for non-finite values."
            )
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

    Raises:
        ValueError: if ``q`` or ``n_candidates`` is not a positive integer, if ``q > n_candidates``, if
            there are zero observations, or if every candidate scores a non-finite acquisition merit
            (rather than letting ``argmax`` -- which treats NaN as the maximum -- silently select one).
    """
    from scipy.stats import norm

    q = _require_positive_int(q, "q")
    n_candidates = _require_positive_int(n_candidates, "n_candidates")
    if q > n_candidates:
        # once every candidate's merit is set to -inf (below, after each pick), np.argmax deterministically
        # returns index 0 again (ties broken by first occurrence) -- the batch would silently contain
        # duplicate points instead of raising. Name the actual constraint instead.
        raise ValueError(f"propose_local_penalization requires q <= n_candidates (q={q}, n_candidates={n_candidates}).")
    b = _as_bounds(bounds)
    rng = _as_rng(seed)
    xs, ys = _validate_xy(x, y)
    if ys.size == 0:
        # same defect class as propose_qei_batch/bayesopt._propose_one: ys.min()/ys.max() on an empty
        # ys crashes with an opaque "zero-size array" ValueError instead of naming the real cause.
        raise ValueError("cannot propose a local-penalization batch with zero observations; call tell() first.")
    gp = _fit_surrogate(xs, ys, gp, fit_kwargs)
    best = float(ys.max() if maximize else ys.min())
    cand = latin_hypercube(b, n_candidates, rng)
    if cand.shape[0] != n_candidates:
        raise ValueError(f"latin_hypercube returned {cand.shape[0]} candidates, expected {n_candidates}.")
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
    # np.argmax treats NaN as the maximum -- np.argmax([1, nan, 5]) returns 1, not 2 -- so a single
    # non-finite merit would otherwise silently outrank every genuinely-scored candidate. Mask any
    # non-finite entry to -inf first so it can never win selection over a finite one, then require at
    # least one candidate to still be admissible; -inf is unambiguous here since a legitimately-computed
    # merit is floored at log(1e-300) (~-691), never -inf.
    merit = np.where(np.isfinite(merit), merit, -np.inf)
    if not np.any(np.isfinite(merit)):
        raise ValueError(
            f"no candidate produced a finite acquisition merit; all {n_candidates} candidates scored "
            "non-finite (NaN) -- check the GP posterior mean/std and the observed y for non-finite values."
        )

    diag = float(np.linalg.norm(b[:, 1] - b[:, 0]))
    batch: list[np.ndarray] = []
    for _ in range(q):
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
