"""Forward uncertainty propagation: push an input distribution through a model to output statistics.

Given input uncertainty (a Gaussian or a sampler) and a model ``f``, report the induced uncertainty on
the output -- mean, standard deviation, and quantiles. Monte Carlo is general; the unscented transform
propagates the first two moments with ``2d+1`` deterministic sigma points (exact for a linear model, a
useful low-cost approximation for mild nonlinearity). The back half of the UQ loop, after sensitivity.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np

__all__ = ["propagate", "register_propagator", "unscented_transform"]

_PSD_EIGENVALUE_RATIO = 1e-9
"""Relative-tolerance bound (matching ``mixle.inference.belief.GaussianBelief``'s PSD gate, and
mirrored from ``mixle.doe.batch._safe_cholesky``'s MXR-080-0166 fix) on how negative a covariance's
worst eigenvalue may be -- relative to its own eigenvalue scale -- before it is refused outright as not
a valid covariance at all, rather than treated as merely numerically singular."""


def _require_positive_int(value: Any, name: str) -> int:
    """Validate ``value`` is an exact positive integer, rejecting nonpositive and fractional counts.

    Mirrors ``mixle.doe.batch._require_positive_int``: ``int(value)`` alone silently truncates
    (``10.5`` becomes ``10``, no error) and an invalid size otherwise fails deep inside numpy with an
    unrelated-looking error (a fractional Monte Carlo sample count raises ``TypeError: 'float' object
    is unsliceable``; zero raises ``IndexError: index -1 is out of bounds for axis 0 with size 0``) --
    neither names the actual problem.
    """
    try:
        as_float = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.") from exc
    as_int = int(as_float)
    if as_int <= 0 or as_float != as_int:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return as_int


def _validate_covariance(cov: np.ndarray) -> tuple[np.ndarray, float]:
    """Reject a non-finite or non-PSD covariance; return ``(symmetrized_cov, eigenvalue_scale)``.

    Shared by both propagation methods (and by :func:`_safe_cholesky`) so an invalid covariance is
    refused identically regardless of which is requested. Without this, ``'unscented'`` would eventually
    be caught by ``_safe_cholesky``'s Cholesky attempt, but ``'montecarlo'`` would sail straight through
    numpy's own ``multivariate_normal(..., check_valid='warn')`` -- which, by default, only warns (easy
    to miss, and silenced by any ambient warnings filter) and then samples anyway from a matrix that was
    never a covariance in the first place.
    """
    cov = np.asarray(cov, dtype=np.float64)
    if not np.all(np.isfinite(cov)):
        raise ValueError("covariance is not finite (contains NaN/Inf).")
    sym = 0.5 * (cov + cov.T)  # symmetrize first: an asymmetric-but-PD input would otherwise "succeed"
    # against a matrix that silently differs from what was passed.
    evals = np.linalg.eigvalsh(sym)
    scale = float(np.abs(evals).max()) if evals.size else 0.0
    if evals.min() < -_PSD_EIGENVALUE_RATIO * max(scale, 1e-12):
        raise ValueError(
            "covariance is not positive semi-definite (worst eigenvalue "
            f"{evals.min():.6g} vs eigenvalue scale {scale:.6g}); refusing to silently substitute an "
            "independent (diagonal-only) approximation -- or silently sample from an invalid matrix -- "
            "for a fundamentally invalid covariance."
        )
    return sym, scale


def _safe_cholesky(sigma: np.ndarray) -> np.ndarray:
    """Cholesky of a covariance, for deriving the unscented transform's sigma points.

    Mirrors ``mixle.doe.batch._safe_cholesky``'s validate-before-heal policy (MXR-080-0166): a caller's
    requested covariance structure must never be silently swapped out for an independent approximation.

    * ``sigma`` finite and PSD within ``_PSD_EIGENVALUE_RATIO`` of its own eigenvalue scale, but merely
      numerically singular (e.g. a fixed/zero-variance input dimension, or perfectly correlated inputs):
      recoverable. Escalating diagonal jitter, starting at ``1e-10`` of the matrix's eigenvalue scale and
      backing off by 10x for up to 7 attempts, nudges it just inside the numerically-decomposable PD
      cone -- the SAME dependence structure, only its conditioning changes. The first attempt is always
      unperturbed: this callsite's covariance can be at any scale (the unscented transform's own
      ``(d + lambda) * cov`` factor can shrink it to ~1e-6 for small ``alpha``), so an always-on jitter
      sized relative to a fixed floor would be a large RELATIVE perturbation on a small-scale matrix.
      Whenever jitter *is* needed, a ``RuntimeWarning`` reports the exact amount, so it is quantified and
      visible rather than invisible.
    * ``sigma`` not finite, or indefinite well beyond float noise (e.g. ``[[1, 2], [2, 1]]``, whose
      eigenvalues are ``3`` and ``-1``: no jitter this small makes that PD): not a valid covariance at
      all. Raises ``ValueError`` immediately, via :func:`_validate_covariance` -- this never falls
      through to a diagonal-only fallback, which would silently discard both the off-diagonal dependence
      structure AND the fact the input wasn't PSD to begin with, while still returning a plausible-
      looking sigma-point spread.

    Raises:
        ValueError: if ``sigma`` is non-finite, fails the PSD check, or (in a pathological case that
            should not occur for anything that passed the PSD check) still fails to factor after the
            full jitter budget is exhausted.
    """
    sym, scale = _validate_covariance(sigma)
    d = sym.shape[0]
    base = 1e-10 * max(scale, 1e-12)
    jitters = [0.0] + [base * (10.0**i) for i in range(7)]  # attempt 0 unperturbed, then 7 escalating
    eye = np.eye(d)
    for attempt, jit in enumerate(jitters):
        try:
            chol = np.linalg.cholesky(sym + jit * eye)
        except np.linalg.LinAlgError:
            continue
        if jit > 0.0:
            warnings.warn(
                "input covariance was numerically singular under a direct Cholesky factorization; "
                f"healed with diagonal jitter={jit:.3g} ({jit / max(scale, 1e-300):.3g} of the "
                f"eigenvalue scale {scale:.3g}) after {attempt} failed attempt(s). The dependence "
                "structure is unchanged -- only the numerical conditioning was adjusted.",
                RuntimeWarning,
                stacklevel=2,
            )
        return chol
    raise ValueError(
        f"covariance passed the positive-semidefinite check (eigenvalue scale {scale:.6g}) but Cholesky "
        f"still failed after escalating jitter up to {jitters[-1]:.3g}; refusing to silently substitute "
        "a diagonal (independent) approximation."
    )


#: Registry of forward-propagation methods, keyed by ``method`` name. Each entry is a callable
#: ``f(func, mean, cov, *, n, quantiles, seed) -> dict`` -- the "register, don't branch" pattern
#: shared with the doe acquisition/criterion registries.
_PROPAGATORS: dict[str, Callable[..., dict[str, Any]]] = {}


def register_propagator(name: str) -> Callable[[Callable[..., dict[str, Any]]], Callable[..., dict[str, Any]]]:
    """Decorator registering a propagation method under ``name`` for :func:`propagate`.

    The decorated callable receives ``(func, mean, cov, *, n, quantiles, seed)`` (``mean``/``cov``
    already coerced to float arrays) and returns the output-statistics dict.
    """

    def decorator(fn: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
        _PROPAGATORS[name] = fn
        return fn

    return decorator


def propagate(
    func: Callable[[np.ndarray], np.ndarray],
    mean: np.ndarray,
    cov: np.ndarray | None = None,
    *,
    n: int = 10000,
    method: str = "montecarlo",
    quantiles: tuple[float, ...] = (0.05, 0.5, 0.95),
    seed: int = 0,
) -> dict[str, Any]:
    """Propagate a Gaussian input ``N(mean, cov)`` through ``func`` to output statistics.

    Args:
        func: a *vectorized* model ``f(X) -> y`` mapping ``(m, d)`` inputs to ``(m,)`` (or ``(m, k)``)
            outputs.
        mean: length-``d`` input mean. ``cov``: ``(d, d)`` input covariance (defaults to identity).
        n: Monte Carlo sample size (ignored by the unscented method).
        method: ``'montecarlo'`` (sample + summarize, gives quantiles) or ``'unscented'`` (sigma-point
            moment propagation, mean + std only). Looked up through the propagator registry.
        quantiles: output quantiles to report (Monte Carlo only).

    Returns:
        ``{'mean', 'std', 'quantiles' (mc only), 'samples' (mc only)}``.

    Raises:
        ValueError: if ``mean`` is non-finite, ``cov``'s shape doesn't match ``mean``, ``cov`` is not a
            valid (finite, PSD-within-tolerance) covariance, ``n`` is not a positive integer, any
            ``quantiles`` entry is outside ``[0, 1]``, or ``method`` is not registered.
    """
    mean = np.atleast_1d(np.asarray(mean, dtype=float))
    d = len(mean)
    cov = np.eye(d) if cov is None else np.atleast_2d(np.asarray(cov, dtype=float))
    if not np.all(np.isfinite(mean)):
        raise ValueError("mean is not finite (contains NaN/Inf).")
    if cov.shape != (d, d):
        raise ValueError(f"covariance shape {cov.shape} is incompatible with mean of length {d}; expected ({d}, {d}).")
    _validate_covariance(cov)  # raises on non-finite/non-PSD; see _safe_cholesky for the unscented path
    n = _require_positive_int(n, "n")
    quantiles = tuple(quantiles)
    if quantiles:
        q_arr = np.asarray(quantiles, dtype=float)
        if not np.all(np.isfinite(q_arr)) or np.any(q_arr < 0.0) or np.any(q_arr > 1.0):
            raise ValueError(f"quantiles must be finite values in [0, 1]; got {quantiles!r}.")
    try:
        propagator = _PROPAGATORS[method]
    except KeyError:
        raise ValueError(
            "unknown propagation method %r; registered methods are %s."
            % (method, ", ".join(repr(name) for name in sorted(_PROPAGATORS)))
        ) from None
    return propagator(func, mean, cov, n=n, quantiles=quantiles, seed=seed)


@register_propagator("unscented")
def _propagate_unscented(
    func: Callable[[np.ndarray], np.ndarray],
    mean: np.ndarray,
    cov: np.ndarray,
    *,
    n: int,
    quantiles: tuple[float, ...],
    seed: int,
) -> dict[str, Any]:
    m, c = unscented_transform(func, mean, cov)
    std = np.sqrt(np.clip(np.diag(np.atleast_2d(c)), 0.0, None))
    return {"mean": m, "std": float(std[0]) if np.ndim(m) == 0 else std}


@register_propagator("montecarlo")
def _propagate_montecarlo(
    func: Callable[[np.ndarray], np.ndarray],
    mean: np.ndarray,
    cov: np.ndarray,
    *,
    n: int,
    quantiles: tuple[float, ...],
    seed: int,
) -> dict[str, Any]:
    rng = np.random.RandomState(seed)
    x = rng.multivariate_normal(mean, cov, size=n)
    y = np.asarray(func(x), dtype=float)
    if y.ndim not in (1, 2) or y.shape[0] != n:
        raise ValueError(
            f"model function must return an array whose leading dimension matches the {n} input "
            f"samples (shape ({n},) or ({n}, k)); got shape {y.shape}. Refusing to use a result whose "
            "shape doesn't match what was sampled."
        )
    if not np.all(np.isfinite(y)):
        raise ValueError("model function returned non-finite (NaN/Inf) output.")
    return {
        "mean": y.mean(axis=0),
        "std": y.std(axis=0),
        "quantiles": {q: np.quantile(y, q, axis=0) for q in quantiles},
        "samples": y,
    }


def unscented_transform(
    func: Callable[[np.ndarray], np.ndarray],
    mean: np.ndarray,
    cov: np.ndarray,
    *,
    alpha: float = 1e-3,
    beta: float = 2.0,
    kappa: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate ``(mean, cov)`` through ``func`` with the unscented (sigma-point) transform.

    Returns the output ``(mean, covariance)``. Exact for an affine ``func``; for a nonlinear one it
    captures the mean and covariance to second order with only ``2d+1`` evaluations.

    Raises:
        ValueError: if ``mean`` is non-finite, ``cov``'s shape doesn't match ``mean``, ``cov`` is not a
            valid covariance (see :func:`_safe_cholesky`), ``alpha``/``beta``/``kappa`` are not finite,
            ``d + lambda <= 0``, ``func``'s return doesn't have a leading dimension of ``2d+1`` (rejected
            explicitly, not silently reinterpreted via a total-size-only reshape), or ``func``'s return
            is non-finite.
    """
    mean = np.atleast_1d(np.asarray(mean, dtype=float))
    cov = np.atleast_2d(np.asarray(cov, dtype=float))
    d = len(mean)
    if not np.all(np.isfinite(mean)):
        raise ValueError("mean is not finite (contains NaN/Inf).")
    if cov.shape != (d, d):
        raise ValueError(f"covariance shape {cov.shape} is incompatible with mean of length {d}; expected ({d}, {d}).")
    if not (np.isfinite(alpha) and np.isfinite(beta) and np.isfinite(kappa)):
        raise ValueError(
            f"unscented transform parameters must be finite; got alpha={alpha!r}, beta={beta!r}, kappa={kappa!r}."
        )
    lam = alpha**2 * (d + kappa) - d
    if not (d + lam > 0):  # `> 0` (not `<= 0`) so a NaN lambda -- e.g. from a NaN alpha/kappa slipping
        # past the finite check above -- is caught too: NaN comparisons are always False, so `<= 0` would
        # silently let it through where `> 0` correctly does not.
        raise ValueError(
            f"unscented_transform requires d + lambda > 0, got {d + lam:.6g}; "
            f"choose kappa > -d (here d={d}) so the sigma-point spread is positive."
        )
    chol = _safe_cholesky((d + lam) * cov)
    sigma = np.vstack([mean, mean + chol.T, mean - chol.T])  # 2d+1 sigma points
    wm = np.full(2 * d + 1, 1.0 / (2.0 * (d + lam)))
    wc = wm.copy()
    wm[0] = lam / (d + lam)
    wc[0] = lam / (d + lam) + (1.0 - alpha**2 + beta)
    expected_leading = 2 * d + 1
    y_raw = np.asarray(func(sigma), dtype=float)
    if y_raw.ndim == 1:
        if y_raw.shape[0] != expected_leading:
            raise ValueError(
                f"model function must return an array whose leading dimension is {expected_leading} "
                f"(one row per sigma point, shape ({expected_leading},) or ({expected_leading}, k)); "
                f"got shape {y_raw.shape}. Refusing to silently reinterpret a wrong-shaped result via a "
                "total-size-only reshape."
            )
        y = y_raw.reshape(expected_leading, 1)
    elif y_raw.ndim == 2:
        if y_raw.shape[0] != expected_leading:
            raise ValueError(
                f"model function must return an array whose leading dimension is {expected_leading} "
                f"(one row per sigma point, shape ({expected_leading},) or ({expected_leading}, k)); "
                f"got shape {y_raw.shape}. Refusing to silently reinterpret a wrong-shaped result via a "
                "total-size-only reshape."
            )
        y = y_raw
    else:
        raise ValueError(f"model function must return a 1-D or 2-D array; got shape {y_raw.shape} (ndim={y_raw.ndim}).")
    if not np.all(np.isfinite(y)):
        raise ValueError("model function returned non-finite (NaN/Inf) output.")
    y_mean = wm @ y
    dy = y - y_mean
    y_cov = (wc[:, None] * dy).T @ dy
    out_dim = y.shape[1]
    return (y_mean[0], float(y_cov[0, 0])) if out_dim == 1 else (y_mean, y_cov)
