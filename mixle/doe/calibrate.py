"""Kennedy-O'Hagan calibration: infer a simulator's parameters with an explicit model-discrepancy term.

Field data rarely equals the simulator even at the true parameters -- there is model-form error. Fitting
parameters by plain least squares absorbs that bias and gives wrong (over-confident) parameters. The
Kennedy-O'Hagan model writes ``y(x) = eta(x, theta) + delta(x) + noise`` with ``delta`` a GP discrepancy,
and infers ``theta`` *and* ``delta`` jointly, so the parameters are not contaminated by the bias.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from scipy.optimize import minimize

from mixle.models._kernels import stationary_kernel

__all__ = ["calibrate", "KOCalibration"]

_NOISE_VAR_FLOOR = 1e-8  # one positive-variance floor, shared by every likelihood term (MXR-080-0171)
_CI95_Z = 1.959963984540054  # two-sided 95% normal-approximation multiplier


def _rbf(x1: np.ndarray, x2: np.ndarray, ls: float, amp: float) -> np.ndarray:
    # Squared-exponential covariance via mixle's shared NumPy kernel: identical
    # `sum((x1-x2)**2)/ls**2 -> amp**2 * exp(-d2/2)` shape, so results are unchanged.
    return stationary_kernel(x1, x2, ls, amp, "rbf")


def _iid_gaussian_neg_ll(r: np.ndarray, noise: float) -> float:
    """Negative log-likelihood of iid residuals ``r`` under ``N(0, noise**2)``, the no-discrepancy
    (plain least-squares) branch's noise model.

    The quadratic penalty and the log normalizer both use the SAME floored variance (``noise**2 +
    _NOISE_VAR_FLOOR``) -- coherent at every ``noise``, including 0. Flooring only the quadratic
    term's denominator while leaving the normalizer's ``log(noise)`` unfloored (the previous bug)
    let the optimizer drive ``noise`` toward 0 for an unbounded improvement in the normalizer while
    the quadratic penalty stayed bounded by its floor: a spurious global "optimum" at zero noise
    regardless of how large the actual residuals were.
    """
    n = len(r)
    var = noise**2 + _NOISE_VAR_FLOOR
    return 0.5 * np.sum(r**2) / var + 0.5 * n * np.log(var) + 0.5 * n * np.log(2 * np.pi)


def _positive_int(name: str, value: Any) -> int:
    """Validate that ``value`` is an exact, finite, positive integer count and return it as ``int``.

    Rejects ``bool``, non-numeric types, non-finite values, and fractional or non-positive values
    (MXR-080-0172): an out-of-range ``max_iter`` used to reach the optimizer unchecked, where scipy
    silently treats it as "stop after ~1 iteration" instead of rejecting it as caller error.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    if not np.isfinite(value):
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    ivalue = int(value)
    if ivalue != value or ivalue <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return ivalue


def _numerical_hessian(f: Callable[[np.ndarray], float], x0: np.ndarray) -> np.ndarray:
    """Central-difference Hessian of scalar ``f`` at ``x0`` (the observed information at the MLE).

    Used to turn the point ``theta`` estimate into an asymptotic (Laplace) standard error -- see
    ``KOCalibration.theta_standard_error`` -- via the usual finite-sample approximation ``Cov(theta)
    ~= H^-1`` where ``H`` is the Hessian of the negative log-likelihood at the optimum. Step sizes
    scale with each coordinate's magnitude (floored at 1) so the stencil is well-conditioned whether
    a parameter sits near 0 or is large.
    """
    d = len(x0)
    eps = np.maximum(np.abs(x0), 1.0) * 1e-4
    hess = np.empty((d, d))
    for i in range(d):
        for j in range(i, d):
            xpp, xpm, xmp, xmm = (x0.copy() for _ in range(4))
            xpp[i] += eps[i]
            xpp[j] += eps[j]
            xpm[i] += eps[i]
            xpm[j] -= eps[j]
            xmp[i] -= eps[i]
            xmp[j] += eps[j]
            xmm[i] -= eps[i]
            xmm[j] -= eps[j]
            hess[i, j] = hess[j, i] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4 * eps[i] * eps[j])
    return hess


class KOCalibration:
    """Result of :func:`calibrate`: a POINT MLE of the fitted parameters (not a posterior), the
    discrepancy GP, and a calibrated predictor.

    ``theta`` is the maximizer of the (marginal) likelihood, not a full Bayesian inference of it --
    there is no prior and no posterior distribution over ``theta`` here. ``theta_standard_error`` is
    an asymptotic (Laplace / observed-Fisher-information) standard error for each ``theta`` component,
    from the inverse Hessian of the negative log-likelihood at the MLE; ``theta_ci_low``/``ci_high``
    are the corresponding two-sided 95% Wald intervals (``theta +- 1.96 * theta_standard_error``).
    This is a local, asymptotic approximation (it can be optimistic for small ``n``, strong parameter
    correlation, or a non-quadratic likelihood near the optimum) -- not a substitute for a real
    posterior when that matters. All three are arrays of ``nan`` when the Hessian at the optimum is
    not numerically invertible (or not positive-definite), which the point estimate itself does not
    depend on and remains usable.
    """

    def __init__(self, theta, ls, amp, noise, simulator, x, y, theta_standard_error=None):
        self.theta = theta
        self.lengthscale, self.amplitude, self.noise = ls, amp, noise
        self._sim, self._x, self._y = simulator, x, y
        self._resid = y - simulator(x, theta)  # discrepancy + noise at the fitted theta
        self.theta_standard_error = (
            np.asarray(theta_standard_error, dtype=float)
            if theta_standard_error is not None
            else np.full(len(theta), np.nan)
        )
        self.theta_ci_low = theta - _CI95_Z * self.theta_standard_error
        self.theta_ci_high = theta + _CI95_Z * self.theta_standard_error

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
    max_iter: int = 1000,
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
        x, y: field inputs and observations. ``x`` and ``y`` must have the same number of rows, and
            ``x``, ``y``, ``theta0``, and ``simulator(x, theta0)`` must all be finite.
        theta0: initial calibration parameters (its length sets the parameter count). There must be
            more observations than free ``theta`` parameters, or ``theta`` is not identifiable.
        discrepancy: include the GP discrepancy term (the Kennedy-O'Hagan model).
        discrepancy_lengthscale: fixed GP correlation length; ``None`` (default) uses 10% of the
            input domain. If given explicitly it must be positive -- unlike ``None``, ``0`` is
            *not* silently treated as "use the default": a caller-supplied non-positive value is a
            contract violation, not a request for the default.
        seed: seeds the small, reproducible random perturbation used to initialize the optimizer, so
            the same seed value reproduces the same fit and a different seed can escape an exact tie
            or a degenerate flat ridge at the unperturbed starting point.
        max_iter: positive optimizer iteration budget. Reused, unmodified, for the internal
            no-discrepancy warm-start fit when ``discrepancy=True``.

    Raises:
        ValueError: on an ``x``/``y`` row-count mismatch, non-finite ``x``/``y``/``theta0``/
            ``simulator(x, theta0)``, a ``simulator(x, theta0)`` shape that does not match ``y``, a
            non-positive ``discrepancy_lengthscale`` or ``max_iter``, fewer observations than
            ``theta`` parameters, or if the optimizer fails to converge to a finite objective.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    theta0 = np.asarray(theta0, dtype=float)
    nth = len(theta0)
    n = len(y)

    if nth == 0:
        raise ValueError("theta0 must have at least one parameter to calibrate.")
    if not np.all(np.isfinite(theta0)):
        raise ValueError(f"theta0 must be finite, got {theta0}.")
    if len(x) != n:
        raise ValueError(f"x and y must have the same number of observations; got len(x)={len(x)}, len(y)={n}.")
    if not np.all(np.isfinite(x)):
        raise ValueError("x contains non-finite values (NaN/Inf).")
    if not np.all(np.isfinite(y)):
        raise ValueError("y contains non-finite values (NaN/Inf).")
    if n <= nth:
        raise ValueError(
            f"n={n} observations is not enough to identify {nth} theta parameter(s); theta is "
            "unidentifiable without more observations than free parameters."
        )
    max_iter = _positive_int("max_iter", max_iter)
    if discrepancy_lengthscale is not None:
        discrepancy_lengthscale = float(discrepancy_lengthscale)
        if not discrepancy_lengthscale > 0:
            raise ValueError(f"discrepancy_lengthscale must be positive, got {discrepancy_lengthscale!r}.")

    y0 = np.asarray(simulator(x, theta0), dtype=float)
    if y0.shape != y.shape:
        raise ValueError(f"simulator(x, theta0) must return shape {y.shape} matching y; got {y0.shape}.")
    if not np.all(np.isfinite(y0)):
        raise ValueError("simulator(x, theta0) returned non-finite values; check the simulator and theta0.")

    scale = np.std(y) + 1e-9
    xx = x if x.ndim > 1 else x[:, None]
    dom = float(np.max(np.ptp(xx, axis=0))) + 1e-9
    ls = discrepancy_lengthscale if discrepancy_lengthscale is not None else 0.1 * dom  # fixed: local discrepancy
    rng = np.random.default_rng(seed)

    def neg_ll(p):
        theta = p[:nth]
        r = y - np.asarray(simulator(x, theta), dtype=float).ravel()
        if not discrepancy:
            return _iid_gaussian_neg_ll(r, np.exp(p[nth]))
        amp, noise = np.exp(p[nth : nth + 2])
        k = _rbf(x, x, ls, amp) + (noise**2 + _NOISE_VAR_FLOOR) * np.eye(n)
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
        theta_ls = calibrate(simulator, x, y, theta0, discrepancy=False, seed=seed, max_iter=max_iter).theta
        p0 = np.concatenate([theta_ls, np.log([0.3 * scale, 0.1 * scale])])

    # Multi-start: always try the unperturbed p0 (so this can never do worse than the single-shot fit),
    # plus a couple of seeded perturbations of it, and keep the best CONVERGED result. `seed` used to
    # be accepted but never consumed -- every run was identical regardless of `seed`. This both makes
    # `seed` control something real and reproducible (same seed -> same perturbations -> same fit) and
    # gives the optimizer a chance to escape a poor starting basin instead of a single fixed shot. Note
    # a perturbation is NOT applied by nudging p0 directly: Nelder-Mead's default initial simplex uses a
    # tiny fixed step for any coordinate that is exactly 0 (as `theta0` conventionally is) but a step
    # *relative to its own magnitude* otherwise -- so nudging a 0 to a small nonzero value shrinks, not
    # grows, its effective initial step, and can make that coordinate's search collapse. Perturbing at a
    # fixed 5% (floored) scale relative to the problem's own units sidesteps that.
    candidates = [p0] + [p0 + np.maximum(np.abs(p0), 0.1) * rng.normal(scale=0.05, size=p0.shape) for _ in range(2)]
    best_res = None
    for cand in candidates:
        res = minimize(neg_ll, cand, method="Nelder-Mead", options={"maxiter": max_iter, "xatol": 1e-4, "fatol": 1e-6})
        if res.success and np.isfinite(res.fun) and (best_res is None or res.fun < best_res.fun):
            best_res = res
    if best_res is None:
        raise ValueError(
            f"calibration optimizer did not converge to a finite objective from any of {len(candidates)} "
            "initializations; increase max_iter or check the simulator/data."
        )
    res = best_res
    theta = res.x[:nth]

    # Asymptotic (Laplace/observed-information) standard error for theta: the marginal covariance is
    # the theta-block of the FULL inverse Hessian (not the inverse of just the theta sub-block), which
    # correctly folds in its correlation with the noise/amplitude nuisance parameters. Degrades to nan
    # (not an error) if the Hessian is not numerically invertible/positive-definite at the optimum --
    # the point estimate above does not depend on this and stays valid either way.
    theta_se = None
    try:
        cov = np.linalg.inv(_numerical_hessian(neg_ll, res.x))
        theta_var = np.diag(cov)[:nth]
        if np.all(np.isfinite(theta_var)) and np.all(theta_var >= 0):
            theta_se = np.sqrt(theta_var)
    except np.linalg.LinAlgError:
        theta_se = None

    if discrepancy:
        amp, noise = np.exp(res.x[nth : nth + 2])
    else:
        ls, amp, noise = 1.0, 0.0, np.exp(res.x[nth])
    return KOCalibration(theta, ls, amp, noise, simulator, x, y, theta_standard_error=theta_se)
