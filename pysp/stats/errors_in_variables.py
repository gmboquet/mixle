"""Errors-in-variables regression: fit a relationship when the predictor is itself measured with error.

When you regress a property on a position/depth/another measurement that carries its own uncertainty --
uncertain well locations, picked stratigraphic depths, one noisy proxy against another -- ordinary least
squares is *biased*: input noise attenuates the slope toward zero (regression dilution). The
errors-in-variables model ``y = a + b x* + e_y``, ``x = x* + e_x`` corrects this. With a known noise
variance ratio it is Deming regression (total least squares when the ratio is 1); it also recovers the
latent true predictor values ``x*`` (the denoised positions). Part of the earth-science/UQ work (Phase 6).
"""

from __future__ import annotations

import numpy as np

__all__ = ["deming_regression", "DemingFit"]


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
    lam = float(variance_ratio)
    xb, yb = x.mean(), y.mean()
    sxx = np.mean((x - xb) ** 2)
    syy = np.mean((y - yb) ** 2)
    sxy = np.mean((x - xb) * (y - yb))
    slope = (syy - lam * sxx + np.sqrt((syy - lam * sxx) ** 2 + 4.0 * lam * sxy**2)) / (2.0 * sxy)
    intercept = yb - slope * xb
    return DemingFit(slope, intercept, lam, x, y)
