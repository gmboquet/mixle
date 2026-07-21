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
    x, y = np.asarray(x, dtype=float).ravel(), np.asarray(y, dtype=float).ravel()
    if x.shape[0] < 2:
        raise ValueError("deming_regression requires at least 2 observations")
    lam = float(variance_ratio)
    xb, yb = x.mean(), y.mean()
    sxx = np.mean((x - xb) ** 2)
    syy = np.mean((y - yb) ** 2)
    sxy = np.mean((x - xb) * (y - yb))
    if sxx == 0.0:
        raise ValueError(
            "deming_regression requires x to have nonzero variance (a constant x carries no information about the slope)"
        )
    if sxy == 0.0:
        # x and y are exactly uncorrelated: the Deming quadratic lam*sxy*b^2 + (lam*sxx-syy)*b - sxy = 0
        # degenerates to (lam*sxx - syy)*b = 0, whose solution is slope=0 (the coefficient is nonzero
        # here, since sxx > 0 is already guaranteed above and generically lam*sxx != syy) -- not the
        # division by `2*sxy` below, which would otherwise produce +-inf/nan.
        slope = 0.0
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
    y = np.asarray(y, dtype=float).ravel()
    vec = x.ndim == 1
    Xc = x[:, None] if vec else x
    n, p = Xc.shape
    sig = np.broadcast_to(np.asarray(sigma_u, dtype=float), (p,))
    if lambdas is None:
        lambdas = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    theta0 = np.atleast_1d(np.asarray(fit_fn(x, y), dtype=float))
    curve = np.empty((len(lambdas), theta0.shape[0]))
    for i, lam in enumerate(lambdas):
        acc = np.zeros_like(theta0)
        for _ in range(n_sims):
            noisy = Xc + np.sqrt(lam) * sig[None, :] * rng.standard_normal((n, p))
            acc += np.atleast_1d(np.asarray(fit_fn(noisy[:, 0] if vec else noisy, y), dtype=float))
        curve[i] = acc / n_sims
    deg = 2 if extrapolation == "quadratic" else 1
    estimate = np.array([np.polyval(np.polyfit(lambdas, curve[:, j], deg), -1.0) for j in range(theta0.shape[0])])
    return {"estimate": estimate, "naive": curve[0], "lambdas": lambdas, "curve": curve}


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
        matches_single_draw = row0_ref is None or np.allclose(out[0], row0_ref, equal_nan=True)
        if out.shape[0] != s.shape[0] or not matches_single_draw:
            raise ValueError
    except Exception:  # noqa: BLE001
        out = np.array([np.asarray(func(row), dtype=float) for row in s])
    levels = np.asarray(quantiles, dtype=float)
    return {
        "mean": out.mean(axis=0),
        "std": out.std(axis=0, ddof=1),
        "quantiles": np.quantile(out, levels, axis=0),
        "levels": levels,
        "samples": out,
    }
