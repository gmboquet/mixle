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


def _positive_int(name: str, value: Any) -> int:
    """Validate that ``value`` is an exact, finite, positive integer count and return it as ``int``.

    Rejects ``bool``, non-numeric types, non-finite values, fractional values, and non-positive values
    -- MXR-080-0158 found budget/draw counts silently truncated (a fractional count via a bare
    ``int()`` cast) or silently substituted (a zero count replaced by a default) instead of rejected.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    if not np.isfinite(value):
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    ivalue = int(value)
    if ivalue != value or ivalue <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return ivalue


def _reject_if_not_psd(eigvals: np.ndarray, p: int) -> None:
    """Raise unless the ascending eigenvalues ``eigvals`` of a claimed ``(p, p)`` covariance are all
    non-negative, up to a floating-point tolerance at the zero boundary (a genuinely singular but
    positive-*semi*definite covariance -- e.g. a hard zero-variance prior direction -- must not be
    rejected merely for being singular; MXR-080-0159)."""
    tol = max(float(eigvals[-1]), 1.0) * p * np.finfo(np.float64).eps
    if eigvals[0] < -tol:
        raise ValueError(f"prior_cov must be positive-semidefinite; smallest eigenvalue is {eigvals[0]:.6g}.")


def alm_scores(gp: Any, x: np.ndarray, y: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Active Learning MacKay scores: the GP posterior predictive variance at each candidate."""
    candidates = np.asarray(candidates)
    if candidates.shape[0] == 0:
        raise ValueError("alm_scores requires a nonempty candidates set.")
    _, cov = gp.predict(x, y, candidates, return_cov=True)
    scores = np.clip(np.diag(np.atleast_2d(np.asarray(cov, dtype=np.float64))), 0.0, None)
    if not np.all(np.isfinite(scores)):
        raise ValueError("alm_scores produced non-finite merits; check the surrogate's covariance prediction.")
    return scores


def alc_scores(gp: Any, x: np.ndarray, y: np.ndarray, candidates: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Active Learning Cohn / IMSE scores: integrated posterior-variance reduction per candidate.

    Adding candidate ``c`` reduces the posterior variance at a reference point ``r`` by
    ``cov_post(r, c)^2 / var_post(c)``; this returns the sum over the reference set (the negative change
    in integrated MSE), so the maximizer is the most globally informative next point.
    """
    candidates = np.asarray(candidates)
    reference = np.asarray(reference)
    if reference.shape[0] == 0:
        raise ValueError(
            "alc_scores requires a nonempty reference set (an empty reference collapses every merit to "
            "zero -- an integral over nothing -- making the subsequent argmax pick an arbitrary "
            "candidate instead of an informative one)."
        )
    if candidates.shape[0] == 0:
        raise ValueError("alc_scores requires a nonempty candidates set.")
    pts = np.vstack([reference, candidates])
    nr = reference.shape[0]
    _, cov = gp.predict(x, y, pts, return_cov=True)
    cov = np.atleast_2d(np.asarray(cov, dtype=np.float64))
    cov_rc = cov[:nr, nr:]  # (n_ref, n_cand) posterior cov between reference and candidates
    var_c = np.clip(np.diag(cov)[nr:], 1e-12, None)
    scores = np.asarray((cov_rc**2).sum(axis=0) / var_c)
    if not np.all(np.isfinite(scores)):
        raise ValueError("alc_scores produced non-finite merits; check the surrogate's covariance prediction.")
    return scores


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
    """Propose the next active-learning point (``method='alc'`` IMSE, or ``'alm'`` max variance).

    ``n_candidates`` (and ``n_reference``, for ``method='alc'``) must be exact positive integers
    (MXR-080-0158): a fractional count used to be silently truncated by a bare ``int()`` cast, and an
    ``n_reference=0`` reduced ALC to an empty reference set (all-zero merits, an arbitrary argmax).
    """
    n_candidates = _positive_int("n_candidates", n_candidates)
    if method == "alc":
        n_reference = _positive_int("n_reference", n_reference)
    b = _as_bounds(bounds)
    rng = _as_rng(seed)
    xs, ys = _validate_xy(x, y)
    gp = _fit_surrogate(xs, ys, gp, fit_kwargs)
    cand = latin_hypercube(b, n_candidates, rng)
    if method == "alm":
        scores = alm_scores(gp, xs, ys, cand)
    elif method == "alc":
        scores = alc_scores(gp, xs, ys, cand, latin_hypercube(b, n_reference, rng))
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

    ``n_init`` (default ``2 * len(bounds)``) and ``max_evals`` must both be exact positive integers with
    ``1 <= n_init <= max_evals`` (MXR-080-0158): an initial design bigger than the whole budget used to
    be evaluated in full and returned over budget instead of rejected, a fractional count was silently
    truncated, and an explicit ``n_init=0`` was silently replaced by the default rather than treated as
    the caller error it almost certainly is.
    """
    b = _as_bounds(bounds)
    d = b.shape[0]
    rng = _as_rng(seed)
    max_evals = _positive_int("max_evals", max_evals)
    n_init = 2 * d if n_init is None else _positive_int("n_init", n_init)
    if n_init > max_evals:
        raise ValueError(
            f"n_init={n_init} exceeds max_evals={max_evals}: the initial design alone would already "
            "evaluate over budget. Pass an explicit n_init <= max_evals (n_init defaults to "
            "2 * len(bounds) when omitted, which must also fit within max_evals)."
        )
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
    mutual information between the data and ``theta`` is ``0.5 * log det(I_p + noise^-2 Sigma0 F^T F)``.
    ``model_matrix`` is the ``(n, p)`` design matrix ``F`` (e.g. from
    :func:`mixle.doe.optimal.polynomial_features`). Higher EIG = a more informative design.

    By Sylvester's determinant identity, ``det(I_p + Sigma0 F^T F / noise^2) == det(I_n + F Sigma0 F^T
    / noise^2)`` exactly -- the two sides differ only in which of the ``(p, p)`` or ``(n, n)`` matrix
    gets formed and factorized. This function forms whichever is smaller. That matters beyond
    speed: scoring a handful of candidate observations (``n`` small, e.g. a few new monitoring
    sites) against a model with many parameters (``p`` large, e.g. hundreds of grid cells) is exactly
    this function's typical caller (:mod:`mixle_pde.monitoring_design`,
    :mod:`mixle_pde.geophysics`), and forming the full dense ``(p, p)`` matrix in that regime is
    reliably ill-conditioned for ``slogdet`` -- verified to throw spurious divide-by-zero/overflow
    ``RuntimeWarning``s on a real ``p=720`` case that the ``(n, n)`` formulation computes cleanly
    (identical value, to float64 precision, with zero warnings).

    ``noise`` must be finite and strictly positive, and ``prior_cov`` (when given) must be a finite,
    symmetric, positive-semidefinite ``(p, p)`` matrix -- a real covariance, not merely a matrix that
    happens to make a determinant come out positive (MXR-080-0159: invalid inputs used to reach
    ``slogdet`` unchecked and could return ``-inf`` or a plausible-looking scalar for a model that is
    not actually a probability distribution). Given valid inputs, the information matrix is guaranteed
    symmetric positive-definite by construction (an identity plus a PSD term), so its log-determinant
    is computed from a Cholesky factorization -- ``2 * sum(log(diag(L)))`` -- rather than ``slogdet``:
    more numerically stable, and it doubles as a final validity check. ``F Sigma0 F^T`` (the ``n <= p``
    fast path) is symmetric for *any* symmetric ``Sigma0``, since it is sandwiched by ``F``/``F^T``, but
    ``Sigma0 F^T F`` (the other order, needed when ``p < n``) is generally not, even though both factors
    are -- the product of two symmetric matrices need not commute -- so that branch instead sandwiches
    ``F^T F`` between a symmetric square root of ``Sigma0``, which keeps the determinant exactly
    (Sylvester's identity again) while keeping the matrix handed to Cholesky genuinely symmetric.
    """
    f = np.asarray(model_matrix, dtype=np.float64)
    if f.ndim != 2:
        raise ValueError(f"model_matrix must be a 2-D (n, p) array, got shape {f.shape}.")
    n, p = f.shape
    noise = float(noise)
    if not np.isfinite(noise) or noise <= 0.0:
        raise ValueError(f"noise must be finite and strictly positive, got {noise!r}.")
    if prior_cov is None:
        sigma0 = np.eye(p)
    else:
        sigma0 = np.asarray(prior_cov, dtype=np.float64)
        if sigma0.shape != (p, p):
            raise ValueError(
                f"prior_cov must have shape ({p}, {p}) to match model_matrix's {p} columns, got {sigma0.shape}."
            )
        if not np.all(np.isfinite(sigma0)):
            raise ValueError("prior_cov must be finite.")
        if not np.allclose(sigma0, sigma0.T):
            raise ValueError("prior_cov must be symmetric (it is a covariance matrix).")

    if n <= p:
        # F Sigma0 F^T is symmetric regardless of Sigma0's own factorization, so only Sigma0's
        # eigenvalues are needed here -- never its eigenvectors, which matters because this is exactly
        # the large-p regime this branch exists to avoid p-scale work in.
        if p > 0:
            _reject_if_not_psd(np.linalg.eigvalsh(sigma0), p)
        m = np.eye(n) + (f @ sigma0 @ f.T) / (noise**2)
    else:
        if p > 0:
            eigvals, eigvecs = np.linalg.eigh(sigma0)
            _reject_if_not_psd(eigvals, p)
            sigma0_half = (eigvecs * np.sqrt(np.clip(eigvals, 0.0, None))) @ eigvecs.T
        else:
            sigma0_half = sigma0
        m = np.eye(p) + (sigma0_half @ (f.T @ f) @ sigma0_half) / (noise**2)

    try:
        chol = np.linalg.cholesky(m)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "Failed to Cholesky-factor the information matrix; it should be positive-definite whenever "
            "noise > 0 and prior_cov is positive-semidefinite -- check model_matrix for non-finite or "
            "extreme entries."
        ) from exc
    logdet = 2.0 * float(np.sum(np.log(np.diag(chol))))
    return 0.5 * logdet


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
