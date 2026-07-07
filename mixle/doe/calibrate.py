"""Kennedy-O'Hagan calibration: infer a simulator's parameters with an explicit model-discrepancy term.

Field data rarely equals the simulator even at the true parameters -- there is model-form error. Fitting
parameters by plain least squares absorbs that bias and gives wrong (over-confident) parameters. The
Kennedy-O'Hagan model writes ``y(x) = eta(x, theta) + delta(x) + noise`` with ``delta`` a GP discrepancy,
and infers ``theta`` *and* ``delta`` jointly, so the parameters are not contaminated by the bias.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from scipy.optimize import minimize

from mixle.models._kernels import stationary_kernel

__all__ = ["calibrate", "KOCalibration"]


def _rbf(x1: np.ndarray, x2: np.ndarray, ls: float, amp: float) -> np.ndarray:
    # Squared-exponential covariance via mixle's shared NumPy kernel: identical
    # `sum((x1-x2)**2)/ls**2 -> amp**2 * exp(-d2/2)` shape, so results are unchanged.
    return stationary_kernel(x1, x2, ls, amp, "rbf")


class KOCalibration:
    """Result of :func:`calibrate`: the fitted parameters, discrepancy GP, and a calibrated predictor."""

    def __init__(self, theta, ls, amp, noise, simulator, x, y):
        self.theta = theta
        self.lengthscale, self.amplitude, self.noise = ls, amp, noise
        self._sim, self._x, self._y = simulator, x, y
        self._resid = y - simulator(x, theta)  # discrepancy + noise at the fitted theta

    def predict(self, x_new: np.ndarray, *, with_discrepancy: bool = True) -> np.ndarray:
        """Calibrated prediction at ``x_new``: simulator at the fitted ``theta``, plus the GP discrepancy
        (the bias-corrected estimate of reality) unless ``with_discrepancy=False`` (the pure simulator)."""
        eta = self._sim(x_new, self.theta)
        if not with_discrepancy:
            return eta
        k = _rbf(self._x, self._x, self.lengthscale, self.amplitude) + self.noise**2 * np.eye(len(self._x))
        ks = _rbf(np.atleast_1d(x_new), self._x, self.lengthscale, self.amplitude)
        return eta + ks @ np.linalg.solve(k, self._resid)


def calibrate(
    simulator: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    theta0: Sequence[float],
    *,
    discrepancy: bool = True,
    discrepancy_lengthscale: float | None = None,
    seed: int = 0,
    max_iter: int = 300,
) -> KOCalibration:
    """Calibrate ``simulator(x, theta)`` to field data ``(x, y)`` with a GP discrepancy term.

    Maximizes the marginal likelihood of the residual ``r(theta) = y - eta(x, theta)`` under a GP +
    noise model, over ``theta`` and the discrepancy amplitude + noise. ``discrepancy=False`` drops the GP
    (plain nonlinear least squares) -- useful to *show* the bias the discrepancy removes.

    The discrepancy correlation length is **fixed** (``discrepancy_lengthscale``, default 10% of the input
    domain) rather than fitted: this is the standard resolution of the Kennedy-O'Hagan ``theta``/``delta``
    identifiability problem -- a *short* discrepancy length forces the GP to model only local model-form
    error, leaving the smooth global trend to the parametric simulator so ``theta`` stays identifiable.
    Set it to the scale of model error you expect.

    Args:
        simulator: ``eta(x, theta) -> predictions`` (vectorized over the rows of ``x``).
        x, y: field inputs and observations.
        theta0: initial calibration parameters (its length sets the parameter count).
        discrepancy: include the GP discrepancy term (the Kennedy-O'Hagan model).
        discrepancy_lengthscale: fixed GP correlation length (default: 10% of the input domain).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    theta0 = np.asarray(theta0, dtype=float)
    nth = len(theta0)
    n = len(y)
    scale = np.std(y) + 1e-9
    xx = x if x.ndim > 1 else x[:, None]
    dom = float(np.max(np.ptp(xx, axis=0))) + 1e-9
    ls = float(discrepancy_lengthscale) if discrepancy_lengthscale else 0.1 * dom  # fixed: local discrepancy

    def neg_ll(p):
        theta = p[:nth]
        r = y - np.asarray(simulator(x, theta), dtype=float).ravel()
        if not discrepancy:
            noise = np.exp(p[nth])
            # + 1e-8 floor mirrors the discrepancy branch's kernel-diagonal jitter below -- noise=exp(...)
            # can't hit exactly 0, but Nelder-Mead can still drive p[nth] negative enough that noise**2
            # underflows toward 0, blowing up this division; the two sibling branches should defend
            # against degenerate noise the same way.
            return 0.5 * np.sum(r**2) / (noise**2 + 1e-8) + n * np.log(noise)  # Gaussian iid residual
        amp, noise = np.exp(p[nth : nth + 2])
        k = _rbf(x, x, ls, amp) + (noise**2 + 1e-8) * np.eye(n)
        try:
            chol = np.linalg.cholesky(k)
        except np.linalg.LinAlgError:
            return 1e12
        alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, r))
        return 0.5 * r @ alpha + np.sum(np.log(np.diag(chol))) + 0.5 * n * np.log(2 * np.pi)

    if not discrepancy:
        p0 = np.concatenate([theta0, [np.log(0.1 * scale)]])
    else:
        # Warm-start theta from the no-discrepancy (least-squares) fit so the optimizer does not fall into
        # the degenerate mode where the GP absorbs the whole signal and theta drifts off.
        theta_ls = calibrate(simulator, x, y, theta0, discrepancy=False, max_iter=max_iter).theta
        p0 = np.concatenate([theta_ls, np.log([0.3 * scale, 0.1 * scale])])
    res = minimize(neg_ll, p0, method="Nelder-Mead", options={"maxiter": max_iter, "xatol": 1e-4, "fatol": 1e-6})
    theta = res.x[:nth]
    if discrepancy:
        amp, noise = np.exp(res.x[nth : nth + 2])
    else:
        ls, amp, noise = 1.0, 0.0, np.exp(res.x[nth])
    return KOCalibration(theta, ls, amp, noise, simulator, x, y)
