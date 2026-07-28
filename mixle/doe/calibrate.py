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

__all__ = ["calibrate", "CalibrationIdentifiabilityError", "KOCalibration"]

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


class CalibrationIdentifiabilityError(ValueError):
    """Raised when simulator sensitivity cannot identify every calibration-parameter direction."""

    def __init__(
        self,
        *,
        rank: int,
        n_parameters: int,
        singular_values: np.ndarray,
        non_identifiable_directions: np.ndarray,
    ) -> None:
        super().__init__(
            f"simulator sensitivity rank is {rank} for {n_parameters} theta parameters; "
            f"{n_parameters - rank} parameter direction(s) are not locally identifiable."
        )
        self.rank = rank
        self.n_parameters = n_parameters
        self.singular_values = np.asarray(singular_values, dtype=float).copy()
        self.non_identifiable_directions = np.asarray(non_identifiable_directions, dtype=float).copy()


def _sensitivity_diagnostics(
    simulator: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x: np.ndarray,
    theta: np.ndarray,
    expected_shape: tuple[int, ...],
) -> tuple[int, np.ndarray, np.ndarray, float]:
    """Finite-difference local simulator sensitivity rank in relative parameter coordinates."""
    n_parameters = theta.size
    parameter_scale = np.maximum(np.abs(theta), 1.0)
    step = parameter_scale * 1e-4
    jacobian = np.empty((int(np.prod(expected_shape)), n_parameters), dtype=float)
    for index in range(n_parameters):
        plus = theta.copy()
        minus = theta.copy()
        plus[index] += step[index]
        minus[index] -= step[index]
        y_plus = np.asarray(simulator(x, plus), dtype=float)
        y_minus = np.asarray(simulator(x, minus), dtype=float)
        if y_plus.shape != expected_shape or y_minus.shape != expected_shape:
            raise ValueError(
                "simulator output shape changed during identifiability probing; "
                f"expected {expected_shape}, got {y_plus.shape} and {y_minus.shape}."
            )
        if not np.all(np.isfinite(y_plus)) or not np.all(np.isfinite(y_minus)):
            raise ValueError("simulator returned non-finite values during identifiability probing.")
        jacobian[:, index] = ((y_plus - y_minus) / (2.0 * step[index])).reshape(-1)
    relative_jacobian = jacobian * parameter_scale[None, :]
    _, singular_values, right_vectors = np.linalg.svd(relative_jacobian, full_matrices=True)
    largest = float(singular_values[0]) if singular_values.size else 0.0
    tolerance = np.sqrt(np.finfo(float).eps) * max(1.0, largest)
    rank = int(np.sum(singular_values > tolerance))
    non_identifiable = right_vectors[rank:, :]
    condition_number = (
        float(singular_values[0] / singular_values[-1])
        if rank == n_parameters and singular_values[-1] > 0.0
        else float("inf")
    )
    return rank, singular_values, non_identifiable, condition_number


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
    depend on. ``sensitivity_rank``, ``sensitivity_singular_values``, and
    ``sensitivity_condition_number`` certify local structural identifiability; a rank-deficient fit
    raises :class:`CalibrationIdentifiabilityError` instead of returning an arbitrary point.

    All fitted arrays are privately copied and frozen. Array properties return detached copies, so
    neither caller-owned inputs nor a returned view can retroactively change the fitted residual,
    interval, or predictor.
    """

    def __init__(
        self,
        theta,
        ls,
        amp,
        noise,
        simulator,
        x,
        y,
        theta_standard_error=None,
        *,
        sensitivity_rank: int | None = None,
        sensitivity_singular_values: np.ndarray | None = None,
        sensitivity_condition_number: float | None = None,
    ):
        theta_array = np.asarray(theta, dtype=float).copy()
        x_array = np.asarray(x, dtype=float).copy()
        y_array = np.asarray(y, dtype=float).copy()
        if theta_array.ndim != 1 or theta_array.size == 0 or not np.all(np.isfinite(theta_array)):
            raise ValueError("theta must be a nonempty finite one-dimensional array.")
        if not np.all(np.isfinite(x_array)) or not np.all(np.isfinite(y_array)):
            raise ValueError("fitted x and y state must be finite.")
        fitted = np.asarray(simulator(x_array, theta_array), dtype=float)
        if fitted.shape != y_array.shape or not np.all(np.isfinite(fitted)):
            raise ValueError("simulator output at fitted theta must be finite and match y.")
        self.lengthscale = float(ls)
        self.amplitude = float(amp)
        self.noise = float(noise)
        if (
            not np.isfinite(self.lengthscale)
            or self.lengthscale <= 0.0
            or not np.isfinite(self.amplitude)
            or self.amplitude < 0.0
            or not np.isfinite(self.noise)
            or self.noise < 0.0
        ):
            raise ValueError("lengthscale must be positive and amplitude/noise must be finite and nonnegative.")
        self._sim = simulator
        self._theta = theta_array
        self._x = x_array
        self._y = y_array
        self._resid = y_array - fitted
        self._theta_standard_error = (
            np.asarray(theta_standard_error, dtype=float).copy()
            if theta_standard_error is not None
            else np.full(theta_array.size, np.nan)
        )
        if self._theta_standard_error.shape != theta_array.shape:
            raise ValueError("theta_standard_error must have one entry per theta parameter.")
        self._theta_ci_low = theta_array - _CI95_Z * self._theta_standard_error
        self._theta_ci_high = theta_array + _CI95_Z * self._theta_standard_error
        self.effective_noise_variance = self.noise**2 + _NOISE_VAR_FLOOR
        covariance = (
            np.zeros((len(x_array), len(x_array)), dtype=float)
            if self.amplitude == 0.0
            else _rbf(x_array, x_array, self.lengthscale, self.amplitude)
        )
        covariance = covariance + self.effective_noise_variance * np.eye(len(x_array))
        self._training_cholesky = np.linalg.cholesky(covariance)
        for array in (
            self._theta,
            self._x,
            self._y,
            self._resid,
            self._theta_standard_error,
            self._theta_ci_low,
            self._theta_ci_high,
            self._training_cholesky,
        ):
            array.setflags(write=False)
        if sensitivity_rank is None:
            (
                sensitivity_rank,
                computed_singular_values,
                non_identifiable,
                computed_condition,
            ) = _sensitivity_diagnostics(simulator, x_array, theta_array, y_array.shape)
            if sensitivity_rank < theta_array.size:
                raise CalibrationIdentifiabilityError(
                    rank=sensitivity_rank,
                    n_parameters=theta_array.size,
                    singular_values=computed_singular_values,
                    non_identifiable_directions=non_identifiable,
                )
            sensitivity_singular_values = computed_singular_values
            sensitivity_condition_number = computed_condition
        self.sensitivity_rank = int(sensitivity_rank)
        singular_values = (
            np.full(theta_array.size, np.nan)
            if sensitivity_singular_values is None
            else np.asarray(sensitivity_singular_values, dtype=float)
        )
        self._sensitivity_singular_values = singular_values.copy()
        self._sensitivity_singular_values.setflags(write=False)
        self.sensitivity_condition_number = (
            float("nan") if sensitivity_condition_number is None else float(sensitivity_condition_number)
        )
        self.identifiable = self.sensitivity_rank == theta_array.size

    @property
    def theta(self) -> np.ndarray:
        """Detached fitted parameter vector."""
        return self._theta.copy()

    @property
    def theta_standard_error(self) -> np.ndarray:
        return self._theta_standard_error.copy()

    @property
    def theta_ci_low(self) -> np.ndarray:
        return self._theta_ci_low.copy()

    @property
    def theta_ci_high(self) -> np.ndarray:
        return self._theta_ci_high.copy()

    @property
    def sensitivity_singular_values(self) -> np.ndarray:
        return self._sensitivity_singular_values.copy()

    def predict(self, x_new: np.ndarray, *, with_discrepancy: bool = True) -> np.ndarray:
        """Calibrated prediction at ``x_new``: simulator at the fitted ``theta``, plus the GP discrepancy
        (the bias-corrected estimate of reality) unless ``with_discrepancy=False`` (the pure simulator)."""
        x_new = np.asarray(x_new, dtype=float)
        if not np.all(np.isfinite(x_new)):
            raise ValueError("x_new must be finite.")
        eta = np.asarray(self._sim(x_new, self._theta), dtype=float)
        expected_rows = len(np.atleast_1d(x_new))
        if eta.shape != (expected_rows,) or not np.all(np.isfinite(eta)):
            raise ValueError(f"simulator(x_new, theta) must return a finite ({expected_rows},) vector.")
        if not with_discrepancy:
            return eta.copy()
        ks = (
            np.zeros((expected_rows, len(self._x)), dtype=float)
            if self.amplitude == 0.0
            else _rbf(np.atleast_1d(x_new), self._x, self.lengthscale, self.amplitude)
        )
        alpha = np.linalg.solve(self._training_cholesky.T, np.linalg.solve(self._training_cholesky, self._resid))
        return eta + ks @ alpha


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
            more observations than free ``theta`` parameters, and the fitted simulator sensitivity
            matrix must have full parameter-column rank.
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
            ``theta`` parameters, rank-deficient simulator sensitivity, or if the optimizer fails to
            converge to a finite objective.
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
    if not np.all(np.isfinite([ls, amp, noise])):
        raise ValueError("calibration optimizer returned non-finite covariance parameters.")
    sensitivity_rank, singular_values, non_identifiable, sensitivity_condition = _sensitivity_diagnostics(
        simulator,
        x,
        theta,
        y.shape,
    )
    if sensitivity_rank < nth:
        raise CalibrationIdentifiabilityError(
            rank=sensitivity_rank,
            n_parameters=nth,
            singular_values=singular_values,
            non_identifiable_directions=non_identifiable,
        )
    return KOCalibration(
        theta,
        ls,
        amp,
        noise,
        simulator,
        x,
        y,
        theta_standard_error=theta_se,
        sensitivity_rank=sensitivity_rank,
        sensitivity_singular_values=singular_values,
        sensitivity_condition_number=sensitivity_condition,
    )
