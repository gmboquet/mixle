"""Errors-in-variables regression: fit a relationship when the predictor is itself measured with error.

When you regress a property on a position/depth/another measurement that carries its own uncertainty --
uncertain well locations, picked stratigraphic depths, one noisy proxy against another -- ordinary least
squares is *biased*: input noise attenuates the slope toward zero (regression dilution). The
errors-in-variables model ``y = a + b x* + e_y``, ``x = x* + e_x`` corrects this. With a known noise
variance ratio it is Deming regression (total least squares when the ratio is 1); it also recovers the
latent true predictor values ``x*`` (the denoised positions).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.random import RandomState

__all__ = ["deming_regression", "DemingFit", "simex", "propagate_uncertainty"]


class DemingFit:
    """Result of :func:`deming_regression`: slope/intercept plus the recovered latent predictor values."""

    def __init__(self, slope, intercept, variance_ratio, x, y):
        self.slope = float(slope)
        self.intercept = float(intercept)
        self.variance_ratio = float(variance_ratio)
        # latent true predictor x* (orthogonal-style projection given the variance ratio)
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        self.x_latent = x + (self.slope / (variance_ratio + self.slope**2)) * (y - self.intercept - self.slope * x)

    def conditional_mean(self, x_star: np.ndarray) -> np.ndarray:
        """The conditional mean ``E[y | x*] = a + b x*`` at *true* predictor values ``x*``."""
        return self.intercept + self.slope * np.asarray(x_star, dtype=float)


def deming_regression(x, y, variance_ratio: float = 1.0) -> DemingFit:
    """Errors-in-variables (Deming) regression of ``y`` on ``x`` when both are noisy.

    Args:
        x, y: paired measurements; both may carry error.
        variance_ratio: ``var(e_y) / var(e_x)`` -- the ratio of output to input noise variance. ``1.0``
            is total least squares (orthogonal regression); a large value -> ordinary least squares (no
            input error); a small value -> inverse regression (predictor dominated by error).

    Returns:
        A :class:`DemingFit` with the unbiased ``slope`` / ``intercept`` and the recovered latent ``x*``.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be one-dimensional paired measurements")
    if x.shape != y.shape:
        raise ValueError("x and y must contain the same number of observations")
    if x.shape[0] < 2:
        raise ValueError("deming_regression requires at least 2 observations")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("x and y must contain only finite values")
    lam = float(variance_ratio)
    if not np.isfinite(lam) or lam <= 0:
        raise ValueError("variance_ratio must be finite and > 0")
    xb, yb = x.mean(), y.mean()
    sxx = np.mean((x - xb) ** 2)
    syy = np.mean((y - yb) ** 2)
    sxy = np.mean((x - xb) * (y - yb))
    if sxx == 0.0:
        raise ValueError(
            "deming_regression requires x to have nonzero variance (a constant x carries no information about the slope)"
        )
    covariance_scale = max(sxx, syy, 1.0)
    if abs(sxy) <= np.finfo(float).eps * covariance_scale:
        # With zero cross-covariance, the Deming principal axis is horizontal only
        # when the x direction has the larger error-standardised variance. The
        # alternative is vertical (an infinite y-on-x slope), or non-identifiable
        # when both directions tie; neither has a finite slope representation.
        if syy < lam * sxx:
            slope = 0.0
        else:
            raise ValueError("Deming slope is vertical or unidentified for this zero-covariance design")
    else:
        slope = (syy - lam * sxx + np.sqrt((syy - lam * sxx) ** 2 + 4.0 * lam * sxy**2)) / (2.0 * sxy)
    intercept = yb - slope * xb
    return DemingFit(slope, intercept, lam, x, y)


def simex(
    fit_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    sigma_u: float | np.ndarray,
    *,
    lambdas: np.ndarray | None = None,
    n_sims: int = 100,
    extrapolation: str = "quadratic",
    seed: int | RandomState | None = 0,
) -> dict:
    """SIMEX: simulation--extrapolation correction for a predictor measured with known error.

    When a predictor ``x`` is observed as ``x = x* + u`` with ``u ~ N(0, sigma_u^2)``, naive estimates
    are biased (attenuation). SIMEX *adds* further noise of variance ``lambda sigma_u^2`` for a grid of
    ``lambda >= 0``, refits at each level (averaging over ``n_sims`` noise draws), then extrapolates the
    estimate back to ``lambda = -1`` (zero measurement error). Works for any estimator returning a
    parameter vector.

    Args:
        fit_fn: ``f(x, y) -> theta`` returning the parameter vector for (possibly multi-column) ``x``.
        x: ``(n,)`` or ``(n, p)`` error-prone predictor(s).
        y: ``(n,)`` response.
        sigma_u: measurement-error standard deviation (scalar, or per-column for matrix ``x``).
        lambdas: extra-noise levels (defaults to ``0, 0.5, 1.0, 1.5, 2.0``).
        n_sims: noise replications per level.
        extrapolation: ``"quadratic"`` or ``"linear"`` extrapolant in ``lambda``.
        seed: RNG seed.

    Returns:
        ``{'estimate', 'naive', 'lambdas', 'curve'}`` -- the SIMEX-corrected parameter vector, the naive
        ``lambda=0`` estimate, and the per-level averaged estimates.
    """
    rng = seed if isinstance(seed, RandomState) else RandomState(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim not in {1, 2} or x.shape[0] < 1 or (x.ndim == 2 and x.shape[1] < 1):
        raise ValueError("x must be a non-empty one- or two-dimensional predictor array")
    if y.ndim != 1 or y.shape[0] != x.shape[0]:
        raise ValueError("y must be one-dimensional and aligned with x")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("x and y must contain only finite values")
    vec = x.ndim == 1
    Xc = x[:, None] if vec else x
    n, p = Xc.shape
    try:
        sig = np.broadcast_to(np.asarray(sigma_u, dtype=float), (p,))
    except ValueError as exc:
        raise ValueError("sigma_u must be scalar or contain one value per predictor column") from exc
    if not np.all(np.isfinite(sig)) or np.any(sig < 0):
        raise ValueError("sigma_u must contain finite non-negative standard deviations")
    if lambdas is None:
        lambdas = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    else:
        lambdas = np.asarray(lambdas, dtype=float)
    if lambdas.ndim != 1 or lambdas.size < 2 or not np.all(np.isfinite(lambdas)) or np.any(lambdas < 0):
        raise ValueError("lambdas must be a finite one-dimensional array of at least two non-negative levels")
    if isinstance(n_sims, (bool, np.bool_)) or not isinstance(n_sims, (int, np.integer)) or n_sims < 1:
        raise ValueError("n_sims must be a positive integer")
    if extrapolation not in {"linear", "quadratic"}:
        raise ValueError("extrapolation must be 'linear' or 'quadratic'")
    deg = 2 if extrapolation == "quadratic" else 1
    if np.unique(lambdas).size <= deg:
        raise ValueError(f"{extrapolation} extrapolation requires at least {deg + 1} distinct lambda levels")

    def fit_parameters(predictors: np.ndarray) -> np.ndarray:
        value = np.asarray(fit_fn(predictors, y), dtype=float)
        if value.ndim > 1 or value.size < 1 or not np.all(np.isfinite(value)):
            raise ValueError("fit_fn must return a non-empty finite scalar or one-dimensional parameter vector")
        return np.atleast_1d(value)

    theta0 = fit_parameters(x)
    curve = np.empty((len(lambdas), theta0.shape[0]))
    for i, lam in enumerate(lambdas):
        acc = np.zeros_like(theta0)
        for _ in range(n_sims):
            noisy = Xc + np.sqrt(lam) * sig[None, :] * rng.standard_normal((n, p))
            fitted = fit_parameters(noisy[:, 0] if vec else noisy)
            if fitted.shape != theta0.shape:
                raise ValueError("fit_fn returned inconsistent parameter dimensions")
            acc += fitted
        curve[i] = acc / n_sims
    estimate = np.array([np.polyval(np.polyfit(lambdas, curve[:, j], deg), -1.0) for j in range(theta0.shape[0])])
    if not np.all(np.isfinite(estimate)):
        raise RuntimeError("SIMEX extrapolation produced non-finite estimates")
    # "naive" is the documented lambda=0 (no added noise) estimate -- theta0, the direct fit on
    # the actual data. curve[0] is instead the average of n_sims NOISY refits at whatever
    # lambdas[0] happens to be, which only coincides with theta0 when the caller's own lambdas
    # array happens to start at exactly 0.0 (the default does, but a custom one need not).
    return {"estimate": estimate, "naive": theta0, "lambdas": lambdas, "curve": curve}


def propagate_uncertainty(
    func: Callable[[np.ndarray], np.ndarray],
    samples: np.ndarray,
    *,
    quantiles: tuple[float, ...] = (0.025, 0.5, 0.975),
) -> dict:
    """Monte-Carlo propagation of an uncertainty set through an arbitrary functional.

    Pushes input draws (a posterior sample, a bootstrap set, any uncertainty representation) through
    ``func`` and summarises the output distribution -- the general "what is the uncertainty of
    ``g(theta)``?" operation. ``func`` may be vectorised (accept the whole ``(n, ...)`` array) or act on
    a single draw; both are detected.

    Args:
        func: the functional to propagate. Returns a scalar or fixed-length vector per input draw.
        samples: ``(n, ...)`` input draws (rows are draws).
        quantiles: output quantiles to report.

    Returns:
        ``{'mean', 'std', 'quantiles', 'levels', 'samples'}`` over the propagated outputs.
    """
    s = np.asarray(samples, dtype=float)
    if s.ndim < 1 or s.shape[0] < 2:
        raise ValueError("samples must contain at least two draws along the first axis")
    if not np.all(np.isfinite(s)):
        raise ValueError("samples must contain only finite values")
    levels = np.asarray(quantiles, dtype=float)
    if levels.ndim != 1 or levels.size < 1 or not np.all(np.isfinite(levels)) or np.any((levels < 0) | (levels > 1)):
        raise ValueError("quantiles must be a non-empty finite one-dimensional sequence in [0, 1]")
    # Matching the outer shape is NECESSARY but not SUFFICIENT to prove `func` is genuinely
    # vectorised (row-independent): a per-draw function written with a numpy reduction that forgot
    # `axis=` (e.g. `lambda row: row / row.sum()`, intended per-row but summing over everything when
    # handed the whole (n, ...) array at once) preserves the outer shape by coincidence while
    # silently computing the wrong thing for every row but the one whose own sum matches the global
    # sum. Cross-check the vectorised call's own first row against `func` applied to that single
    # draw -- the same call shape the fallback loop below already relies on -- before trusting it.
    try:
        row0_ref = np.asarray(func(s[0]), dtype=float)
    except Exception:  # noqa: BLE001
        row0_ref = None  # func doesn't support a single-draw call; fall back to the shape-only check
    try:
        out = np.asarray(func(s), dtype=float)
        matches_single_draw = row0_ref is None or (
            out.ndim >= 1 and out[0].shape == row0_ref.shape and np.allclose(out[0], row0_ref, equal_nan=True)
        )
        if out.ndim < 1 or out.shape[0] != s.shape[0] or not matches_single_draw:
            raise ValueError
    except Exception:  # noqa: BLE001
        out = np.array([np.asarray(func(row), dtype=float) for row in s])
    if out.ndim < 1 or out.shape[0] != s.shape[0] or not np.all(np.isfinite(out)):
        raise ValueError("func must return finite outputs with a consistent shape for every draw")
    return {
        "mean": out.mean(axis=0),
        "std": out.std(axis=0, ddof=1),
        "quantiles": np.quantile(out, levels, axis=0),
        "levels": levels,
        "samples": out,
    }
