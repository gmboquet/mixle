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
  from a prior sampler and a log-likelihood; :func:`expected_information_gain_nmc_estimate` returns the
  same point value together with its Monte Carlo standard error and confidence interval.

The GP-based functions fit the torch surrogate; the EIG functions are pure NumPy.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np
from numpy.random import RandomState

from mixle.doe.bayesopt import _fit_surrogate, _validate_xy
from mixle.doe.designs import Bounds, _as_bounds, _as_rng, latin_hypercube


def _positive_int(name: str, value: Any) -> int:
    """Validate that ``value`` is an exact, finite, positive integer count and return it as ``int``.

    Rejects ``bool``, non-numeric types, non-finite values, fractional values, and non-positive values
    -- MXR-080-0158/0160 found budget/draw counts silently truncated (a fractional count via a bare
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


def _validated_posterior(
    gp: Any,
    x: np.ndarray,
    y: np.ndarray,
    points: np.ndarray,
    *,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a certified finite posterior mean and covariance for exactly ``points``."""
    mean_raw, cov_raw = gp.predict(x, y, points, return_cov=True)
    n = points.shape[0]
    mean = np.asarray(mean_raw, dtype=np.float64)
    if mean.shape == (n, 1):
        mean = mean[:, 0]
    elif mean.shape != (n,):
        raise ValueError(f"{label} posterior mean must have shape ({n},) or ({n}, 1), got {mean.shape}.")
    if not np.all(np.isfinite(mean)):
        raise ValueError(f"{label} posterior mean must be finite.")
    cov = np.asarray(cov_raw, dtype=np.float64)
    if cov.shape != (n, n):
        raise ValueError(f"{label} posterior covariance must have shape ({n}, {n}), got {cov.shape}.")
    if not np.all(np.isfinite(cov)):
        raise ValueError(f"{label} posterior covariance must be finite.")
    scale = max(float(np.linalg.norm(cov, ord=np.inf)), np.finfo(np.float64).tiny)
    tolerance = 64.0 * np.finfo(np.float64).eps * scale * max(n, 1)
    asymmetry = float(np.linalg.norm(cov - cov.T, ord=np.inf))
    if asymmetry > tolerance:
        raise ValueError(
            f"{label} posterior covariance must be symmetric within {tolerance:.6g}; asymmetry is {asymmetry:.6g}."
        )
    cov = 0.5 * (cov + cov.T)
    eigvals = np.linalg.eigvalsh(cov)
    if eigvals[0] < -tolerance:
        raise ValueError(
            f"{label} posterior covariance must be positive-semidefinite; smallest eigenvalue is {eigvals[0]:.6g}."
        )
    if eigvals[0] < 0.0:
        eigvecs = np.linalg.eigh(cov)[1]
        cov = (eigvecs * np.clip(eigvals, 0.0, None)) @ eigvecs.T
    return mean, cov


def _as_point_set(values: Any, name: str) -> np.ndarray:
    points = np.asarray(values, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty 2-D array, got shape {points.shape}.")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must contain only finite values.")
    return points


def alm_scores(gp: Any, x: np.ndarray, y: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Active Learning MacKay scores: the GP posterior predictive variance at each candidate."""
    candidates = _as_point_set(candidates, "candidates")
    _, cov = _validated_posterior(gp, x, y, candidates, label="ALM")
    scores = np.maximum(np.diag(cov), 0.0)
    if not np.all(np.isfinite(scores)):
        raise ValueError("alm_scores produced non-finite merits; check the surrogate's covariance prediction.")
    return scores


def alc_scores(gp: Any, x: np.ndarray, y: np.ndarray, candidates: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Active Learning Cohn / IMSE scores: integrated posterior-variance reduction per candidate.

    Adding candidate ``c`` reduces the posterior variance at a reference point ``r`` by
    ``cov_post(r, c)^2 / var_post(c)``; this returns the sum over the reference set (the negative change
    in integrated MSE), so the maximizer is the most globally informative next point.
    """
    candidates = _as_point_set(candidates, "candidates")
    reference = _as_point_set(reference, "reference")
    if candidates.shape[1] != reference.shape[1]:
        raise ValueError(
            f"candidates and reference must have the same width, got {candidates.shape[1]} and {reference.shape[1]}."
        )
    pts = np.vstack([reference, candidates])
    nr = reference.shape[0]
    _, cov = _validated_posterior(gp, x, y, pts, label="ALC")
    cov_rc = cov[:nr, nr:]  # (n_ref, n_cand) posterior cov between reference and candidates
    var_c = np.maximum(np.diag(cov)[nr:], 0.0)
    scores = np.zeros(candidates.shape[0], dtype=np.float64)
    informative = var_c > 0.0
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        scores[informative] = (cov_rc[:, informative] ** 2).sum(axis=0) / var_c[informative]
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
    x_rows: list[np.ndarray] = []
    y_values: list[float] = []
    failed_evaluations: list[dict[str, Any]] = []
    n_evaluations = 0

    def evaluate(point: np.ndarray) -> float | None:
        nonlocal n_evaluations
        candidate = np.array(point, dtype=np.float64, copy=True)
        n_evaluations += 1
        value = float(objective(candidate.copy()))
        if not np.isfinite(value):
            failed_evaluations.append(
                {
                    "evaluation": n_evaluations,
                    "x": candidate,
                    "status": "nonfinite_observation",
                    "observation": value,
                }
            )
            return None
        return value

    def result(stopped_reason: str) -> dict[str, Any]:
        return {
            "X": np.asarray(x_rows, dtype=np.float64).reshape(-1, d),
            "Y": np.asarray(y_values, dtype=np.float64),
            "n_evaluations": n_evaluations,
            "failed_evaluations": tuple(failed_evaluations),
            "stopped_reason": stopped_reason,
        }

    for point in latin_hypercube(b, n_init, rng):
        value = evaluate(point)
        if value is None:
            return result("objective_failed")
        x_rows.append(np.array(point, dtype=np.float64, copy=True))
        y_values.append(value)

    while n_evaluations < max_evals:
        x_all = np.asarray(x_rows, dtype=np.float64)
        y_all = np.asarray(y_values, dtype=np.float64)
        xn = propose_active_learning(x_all, y_all, b, method=method, seed=rng, fit_kwargs=fit_kwargs)
        value = evaluate(xn)
        if value is None:
            return result("objective_failed")
        x_rows.append(np.array(xn, dtype=np.float64, copy=True))
        y_values.append(value)
    return result("budget_exhausted")


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
    if not np.all(np.isfinite(f)):
        raise ValueError("model_matrix must contain only finite values.")
    n, p = f.shape
    if isinstance(noise, (bool, np.bool_)):
        raise TypeError("noise must be a finite positive real number, got bool.")
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
        covariance_scale = (
            max(float(np.linalg.norm(sigma0, ord=np.inf)), np.finfo(np.float64).tiny)
            if sigma0.size
            else np.finfo(np.float64).tiny
        )
        covariance_tolerance = 64.0 * np.finfo(np.float64).eps * covariance_scale * max(p, 1)
        if float(np.linalg.norm(sigma0 - sigma0.T, ord=np.inf)) > covariance_tolerance:
            raise ValueError("prior_cov must be symmetric within floating-point roundoff.")
        sigma0 = 0.5 * (sigma0 + sigma0.T)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        inv_noise_sq = np.square(1.0 / noise)
        if not np.isfinite(inv_noise_sq):
            raise ValueError("noise is too small to form finite information evidence.")
        if n <= p:
            # F Sigma0 F^T is symmetric regardless of Sigma0's own factorization, so only Sigma0's
            # eigenvalues are needed here -- never its eigenvectors, which matters because this is exactly
            # the large-p regime this branch exists to avoid p-scale work in.
            if p > 0:
                _reject_if_not_psd(np.linalg.eigvalsh(sigma0), p)
            m = np.eye(n) + (f @ sigma0 @ f.T) * inv_noise_sq
        else:
            if p > 0:
                eigvals, eigvecs = np.linalg.eigh(sigma0)
                _reject_if_not_psd(eigvals, p)
                sigma0_half = (eigvecs * np.sqrt(np.clip(eigvals, 0.0, None))) @ eigvecs.T
            else:
                sigma0_half = sigma0
            m = np.eye(p) + (sigma0_half @ (f.T @ f) @ sigma0_half) * inv_noise_sq

    if not np.all(np.isfinite(m)):
        raise ValueError("linear information matrix became non-finite from the supplied finite inputs.")
    matrix_scale = (
        max(float(np.linalg.norm(m, ord=np.inf)), np.finfo(np.float64).tiny) if m.size else np.finfo(np.float64).tiny
    )
    matrix_tolerance = 64.0 * np.finfo(np.float64).eps * matrix_scale * max(m.shape[0], 1)
    if m.size and float(np.linalg.norm(m - m.T, ord=np.inf)) > matrix_tolerance:
        raise ValueError("linear information matrix is not symmetric within floating-point roundoff.")
    m = 0.5 * (m + m.T)

    try:
        chol = np.linalg.cholesky(m)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "Failed to Cholesky-factor the information matrix; it should be positive-definite whenever "
            "noise > 0 and prior_cov is positive-semidefinite -- check model_matrix for non-finite or "
            "extreme entries."
        ) from exc
    if not np.all(np.isfinite(chol)) or np.any(np.diag(chol) <= 0.0):
        raise ValueError("linear information Cholesky factor is not finite and strictly positive.")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        logdet = 2.0 * float(np.sum(np.log(np.diag(chol))))
        result = 0.5 * logdet
    if not np.isfinite(result):
        raise ValueError("expected information gain became non-finite.")
    if result < -64.0 * np.finfo(np.float64).eps * max(abs(result), 1.0):
        raise ValueError(f"expected information gain must be non-negative, got {result!r}.")
    return max(result, 0.0)


class ExpectedInformationGainEstimate(NamedTuple):
    """A nested-Monte-Carlo EIG estimate, reported with its own Monte Carlo uncertainty (MXR-080-0160).

    ``value`` is the point estimate: the mean, over ``n_outer`` independent outer replicates, of
    ``log p(y_i|theta_i) - log E_theta'[p(y_i|theta')]`` (Ryan 2003). ``standard_error`` is the Monte
    Carlo standard error of that mean (the outer replicates' sample standard deviation / ``sqrt(n_outer)``;
    ``None`` when ``n_outer == 1``, since a single replicate carries no information about its own spread).
    ``ci_low``/``ci_high`` are an approximate two-sided 95% confidence interval (``value +- 1.96 *
    standard_error``) -- a normal approximation to the outer average that does not capture the *inner*
    estimate's own (asymptotically vanishing, but nonzero at finite ``n_inner``) bias; increase
    ``n_inner`` to shrink that separate source of error. ``n_outer``/``n_inner`` record the actual
    (validated) draw counts used.
    """

    value: float
    standard_error: float | None
    ci_low: float | None
    ci_high: float | None
    n_outer: int
    n_inner: int


_EIG_CI95_Z = 1.959963984540054  # two-sided 95% normal-approximation multiplier


def expected_information_gain_nmc_estimate(
    prior_sampler: Callable[[RandomState, int], np.ndarray],
    log_likelihood: Callable[[np.ndarray, np.ndarray], np.ndarray],
    simulate: Callable[[np.ndarray, RandomState], np.ndarray],
    *,
    n_outer: int = 256,
    n_inner: int = 256,
    seed: int | RandomState | None = None,
) -> ExpectedInformationGainEstimate:
    """Nested-Monte-Carlo expected information gain, with its Monte Carlo standard error (MXR-080-0160).

    Estimates ``EIG = E_{theta, y}[ log p(y|theta) - log E_{theta'}[p(y|theta')] ]`` (Ryan 2003): draw
    outer ``theta_i ~ prior`` and ``y_i ~ p(y|theta_i)`` via ``simulate``; the inner expectation is a
    mean over ``n_inner`` prior draws of ``exp(log_likelihood(theta', y_i))``. ``prior_sampler(rng, n)``
    returns ``(n, k)`` parameter draws; ``log_likelihood(thetas, y)`` returns a log-density per row of
    ``thetas`` at the single observation ``y``; ``simulate(theta, rng)`` draws one ``y`` given ``theta``.

    ``n_outer``/``n_inner`` must be exact positive integers. Every oracle call is validated against its
    contract before use: ``prior_sampler`` must return exactly the requested number of draws, at *every*
    call (outer and inner alike) -- a sampler that silently under-delivers used to shrink the estimate's
    effective sample size without changing its reported one, and an empty inner sample used to fail deep
    inside ``logaddexp.reduce`` instead of at the point the malformed draw actually appeared;
    ``log_likelihood`` must return exactly one log-density per row it was given; and every draw and
    log-density must be finite, checked at each oracle boundary so a non-finite value cannot silently
    propagate into the final average.
    """
    n_outer = _positive_int("n_outer", n_outer)
    n_inner = _positive_int("n_inner", n_inner)
    rng = seed if isinstance(seed, RandomState) else RandomState(seed)

    thetas_outer = np.asarray(prior_sampler(rng, n_outer), dtype=np.float64)
    if thetas_outer.ndim != 2 or thetas_outer.shape[0] != n_outer:
        raise ValueError(
            f"prior_sampler(rng, {n_outer}) must return an (n, k) array of exactly {n_outer} outer "
            f"draws, got shape {thetas_outer.shape}."
        )
    if not np.all(np.isfinite(thetas_outer)):
        raise ValueError("prior_sampler returned non-finite outer draws.")

    terms = np.empty(n_outer, dtype=np.float64)
    for i, theta_i in enumerate(thetas_outer):
        y_i = np.asarray(simulate(theta_i, rng), dtype=np.float64)
        if not np.all(np.isfinite(y_i)):
            raise ValueError(f"simulate returned a non-finite observation at outer draw {i}.")

        ll_true_arr = np.asarray(log_likelihood(theta_i[None, :], y_i), dtype=np.float64)
        if ll_true_arr.shape != (1,):
            raise ValueError(
                f"log_likelihood(theta[None, :], y) must return shape (1,), got {ll_true_arr.shape} (outer draw {i})."
            )
        ll_true = float(ll_true_arr[0])
        if not np.isfinite(ll_true):
            raise ValueError(f"log_likelihood returned a non-finite log-density at outer draw {i}.")

        thetas_inner = np.asarray(prior_sampler(rng, n_inner), dtype=np.float64)
        if thetas_inner.ndim != 2 or thetas_inner.shape[0] != n_inner:
            raise ValueError(
                f"prior_sampler(rng, {n_inner}) must return an (n, k) array of exactly {n_inner} inner "
                f"draws, got shape {thetas_inner.shape} (outer draw {i})."
            )
        if not np.all(np.isfinite(thetas_inner)):
            raise ValueError(f"prior_sampler returned non-finite inner draws at outer draw {i}.")

        ll_inner = np.asarray(log_likelihood(thetas_inner, y_i), dtype=np.float64)
        if ll_inner.shape != (n_inner,):
            raise ValueError(
                f"log_likelihood(thetas, y) must return shape ({n_inner},) (one density per inner "
                f"draw), got {ll_inner.shape} (outer draw {i})."
            )
        if not np.all(np.isfinite(ll_inner)):
            raise ValueError(f"log_likelihood returned non-finite log-densities at outer draw {i}.")

        log_evidence = float(np.logaddexp.reduce(ll_inner) - np.log(n_inner))
        if not np.isfinite(log_evidence):
            raise ValueError(f"derived log evidence is non-finite at outer draw {i}.")
        with np.errstate(over="ignore", invalid="ignore"):
            term = ll_true - log_evidence
        if not np.isfinite(term):
            raise ValueError(f"derived information-gain term is non-finite at outer draw {i}.")
        terms[i] = term

    scale = float(np.max(np.abs(terms)))
    if scale == 0.0:
        value = 0.0
    else:
        value = float(scale * np.mean(terms / scale))
    if not np.isfinite(value):
        raise ValueError("nested-Monte-Carlo expected information gain is non-finite.")
    if n_outer > 1:
        deviations = terms - value
        deviation_scale = float(np.max(np.abs(deviations)))
        if deviation_scale == 0.0:
            standard_error = 0.0
        else:
            with np.errstate(over="ignore", invalid="ignore"):
                standard_error = float(
                    deviation_scale
                    * np.sqrt(np.sum(np.square(deviations / deviation_scale)) / (n_outer - 1))
                    / np.sqrt(n_outer)
                )
        if not np.isfinite(standard_error):
            raise ValueError("nested-Monte-Carlo standard error is non-finite.")
        with np.errstate(over="ignore", invalid="ignore"):
            ci_low = value - _EIG_CI95_Z * standard_error
            ci_high = value + _EIG_CI95_Z * standard_error
        if not np.isfinite(ci_low) or not np.isfinite(ci_high):
            raise ValueError("nested-Monte-Carlo confidence interval is non-finite.")
    else:
        standard_error = None
        ci_low = None
        ci_high = None
    return ExpectedInformationGainEstimate(
        value=value,
        standard_error=standard_error,
        ci_low=ci_low,
        ci_high=ci_high,
        n_outer=n_outer,
        n_inner=n_inner,
    )


def expected_information_gain_nmc(
    prior_sampler: Callable[[RandomState, int], np.ndarray],
    log_likelihood: Callable[[np.ndarray, np.ndarray], np.ndarray],
    simulate: Callable[[np.ndarray, RandomState], np.ndarray],
    *,
    n_outer: int = 256,
    n_inner: int = 256,
    seed: int | RandomState | None = None,
) -> float:
    """The nested-Monte-Carlo EIG point estimate.

    A thin wrapper around :func:`expected_information_gain_nmc_estimate` for callers that only want the
    point value; see its docstring for the full draw-contract validation (MXR-080-0160) and for how to
    also get the Monte Carlo standard error / confidence interval.
    """
    return expected_information_gain_nmc_estimate(
        prior_sampler, log_likelihood, simulate, n_outer=n_outer, n_inner=n_inner, seed=seed
    ).value


__all__ = [
    "alm_scores",
    "alc_scores",
    "propose_active_learning",
    "active_learning_design",
    "expected_information_gain_linear",
    "ExpectedInformationGainEstimate",
    "expected_information_gain_nmc_estimate",
    "expected_information_gain_nmc",
]
