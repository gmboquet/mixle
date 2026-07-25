"""Inducing-point sparse Gaussian-process regression (FITC) -- scalable GP inference.

Exact GP regression costs O(n^3) in the number of training points, which caps the field/emulator size
for continental grids or large survey sets. This fits a sparse GP with ``m << n`` inducing points via the
Fully Independent Training Conditional (FITC) approximation (Snelson & Ghahramani, 2006), costing
O(n m^2 + m^3) -- linear in ``n``. As ``m -> n`` (and the inducing points cover the data) it recovers the
exact GP.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from mixle.models._kernels import stationary_kernel as _kernel

__all__ = ["SparseGaussianProcessRegressor"]

_KERNELS = frozenset({"rbf", "matern32", "matern52"})


def _positive_finite(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a strictly positive finite scalar")
    array = np.asarray(value)
    if array.ndim != 0:
        raise ValueError(f"{name} must be a strictly positive finite scalar")
    try:
        scalar = float(array)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a strictly positive finite scalar") from exc
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be a strictly positive finite scalar")
    return scalar


def _positive_int(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _as2d(x: np.ndarray, *, name: str, feature_width: int | None = None) -> np.ndarray:
    """A 1-D array is ``n`` points in 1-D (shape (n, 1)); a 2-D array is ``n`` points in ``d`` dims."""
    try:
        x = np.asarray(x, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-empty finite one- or two-dimensional point array") from exc
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty finite one- or two-dimensional point array")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} must contain only finite coordinates")
    if feature_width is not None and x.shape[1] != feature_width:
        raise ValueError(f"{name} has {x.shape[1]} features; fitted data has {feature_width}")
    return x


class SparseGaussianProcessRegressor:
    """Sparse GP regression with ``m`` inducing points (FITC).

    Args:
        lengthscale, amplitude, noise: initial kernel/noise hyperparameters (all positive and finite).
        kernel: ``'rbf'``, ``'matern32'`` or ``'matern52'``.
        n_inducing: number of inducing points (placed by k-means++ over unique training inputs at ``fit``).
        jitter: strictly positive diagonal stabilization for the inducing covariance.
    """

    def __init__(
        self,
        lengthscale=1.0,
        amplitude=1.0,
        noise=0.1,
        kernel="rbf",
        n_inducing=50,
        jitter=1e-8,
    ):
        self.lengthscale = _positive_finite(lengthscale, "lengthscale")
        self.amplitude = _positive_finite(amplitude, "amplitude")
        self.noise = _positive_finite(noise, "noise")
        if not isinstance(kernel, str) or kernel.lower() not in _KERNELS:
            raise ValueError(f"kernel must be one of {sorted(_KERNELS)}")
        self.kernel = kernel.lower()
        self.n_inducing = _positive_int(n_inducing, "n_inducing")
        self.jitter = _positive_finite(jitter, "jitter")
        self.Z = None
        self.fit_receipt = None

    def _place_inducing(self, x, rng):
        """Choose a diverse k-means++ subset of the unique training inputs."""
        unique = np.unique(x, axis=0)
        m = min(self.n_inducing, len(unique))
        if m == len(unique):
            return unique.copy()

        chosen = np.empty(m, dtype=int)
        chosen[0] = int(rng.randint(len(unique)))
        with np.errstate(over="ignore", invalid="ignore"):
            delta = unique - unique[chosen[0]]
            nearest_d2 = np.einsum("ij,ij->i", delta, delta)
        if not np.all(np.isfinite(nearest_d2)):
            raise ValueError("x coordinate range is too large for finite inducing-point distances")
        for index in range(1, m):
            total = float(np.sum(nearest_d2))
            if not np.isfinite(total) or total <= 0.0:
                raise RuntimeError("could not place distinct inducing points from the unique training inputs")
            target = rng.random_sample() * total
            chosen[index] = min(int(np.searchsorted(np.cumsum(nearest_d2), target, side="right")), len(unique) - 1)
            with np.errstate(over="ignore", invalid="ignore"):
                delta = unique - unique[chosen[index]]
                candidate_d2 = np.einsum("ij,ij->i", delta, delta)
            if not np.all(np.isfinite(candidate_d2)):
                raise ValueError("x coordinate range is too large for finite inducing-point distances")
            nearest_d2 = np.minimum(nearest_d2, candidate_d2)
        return unique[chosen].copy()

    def _fitc_terms(self, x, y, ls, amp, noise, *, z=None):
        """Shared FITC quantities at given hyperparameters (Kuu chol, Sigma chol, the y-weighted vector)."""
        z = self.Z if z is None else z
        kuu = _kernel(z, z, ls, amp, self.kernel) + self.jitter * np.eye(len(z))
        kuf = _kernel(z, x, ls, amp, self.kernel)
        kff_diag = np.full(len(x), amp**2)
        luu = np.linalg.cholesky(kuu)
        v = np.linalg.solve(luu, kuf)  # m x n,  Kuu^{-1/2} Kuf
        qff_diag = np.sum(v**2, axis=0)  # diag(Kfu Kuu^-1 Kuf)
        lam = np.maximum(kff_diag - qff_diag, 0.0) + noise**2  # FITC diagonal
        v_lam = v / lam[None, :]
        a = np.eye(len(z)) + v_lam @ v.T  # I + Kuu^{-1/2} Kuf Lam^-1 Kfu Kuu^{-1/2}
        la = np.linalg.cholesky(a)
        return luu, la, v, v_lam, lam, kuf

    def _neg_log_marglik(self, x, y, ls, amp, noise, *, z=None, mean=None):
        luu, la, v, v_lam, lam, _ = self._fitc_terms(x, y, ls, amp, noise, z=z)
        n = len(x)
        ym = y - (self.mean if mean is None else mean)
        # log|Q_ff + Lam| = log|A| + sum log lam ; quadratic via Woodbury
        logdet = 2.0 * np.sum(np.log(np.diag(la))) + np.sum(np.log(lam))
        vy = v_lam @ ym  # m
        w = np.linalg.solve(la, vy)
        quad = np.sum(ym**2 / lam) - np.sum(w**2)
        return 0.5 * (logdet + quad + n * np.log(2.0 * np.pi))

    def fit(self, x, y, *, optimize=True, seed=0, max_iter=100):
        """Place inducing points and (optionally) fit hyperparameters by the FITC marginal likelihood."""
        x = _as2d(x, name="x")
        try:
            y = np.asarray(y, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("y must be a non-empty finite one-dimensional array") from exc
        if y.ndim != 1 or len(y) == 0:
            raise ValueError("y must be a non-empty finite one-dimensional array")
        if len(y) != len(x):
            raise ValueError(f"x and y must have the same number of rows, got {len(x)} and {len(y)}")
        if not np.all(np.isfinite(y)):
            raise ValueError("y must contain only finite values")
        if not isinstance(optimize, (bool, np.bool_)):
            raise ValueError("optimize must be a boolean")
        max_iter = _positive_int(max_iter, "max_iter")
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
            raise ValueError("seed must be an integer from 0 through 2**32 - 1")
        seed = int(seed)
        if not 0 <= seed <= np.iinfo(np.uint32).max:
            raise ValueError("seed must be an integer from 0 through 2**32 - 1")

        mean = float(y.mean())
        z = self._place_inducing(x, np.random.RandomState(seed))
        lengthscale, amplitude, noise = self.lengthscale, self.amplitude, self.noise
        optimizer_receipt = {
            "attempted": bool(optimize),
            "success": True,
            "status": 0,
            "message": "optimization disabled",
            "iterations": 0,
            "objective": None,
        }
        if optimize:
            theta0 = np.log([lengthscale, amplitude, noise])

            def obj(t):
                candidate = np.exp(t)
                if not np.all(np.isfinite(candidate)):
                    return np.inf
                try:
                    value = self._neg_log_marglik(x, y, *candidate, z=z, mean=mean)
                except (FloatingPointError, np.linalg.LinAlgError, ValueError):
                    return np.inf
                return float(value) if np.isfinite(value) else np.inf

            res = minimize(
                obj,
                theta0,
                method="L-BFGS-B",
                bounds=[(-30.0, 30.0)] * 3,
                options={"maxiter": max_iter, "ftol": 1e-9},
            )
            objective = float(res.fun) if np.ndim(res.fun) == 0 else np.nan
            optimizer_receipt = {
                "attempted": True,
                "success": bool(res.success),
                "status": int(res.status),
                "message": str(res.message),
                "iterations": int(getattr(res, "nit", 0)),
                "objective": objective,
            }
            if (
                not res.success
                or np.shape(res.x) != (3,)
                or not np.all(np.isfinite(res.x))
                or not np.isfinite(objective)
            ):
                raise RuntimeError(
                    "sparse GP hyperparameter optimization failed: "
                    f"status={res.status}, message={res.message!s}, objective={objective!r}"
                )
            lengthscale, amplitude, noise = (_positive_finite(value, name) for value, name in zip(
                np.exp(res.x),
                ("optimized lengthscale", "optimized amplitude", "optimized noise"),
                strict=True,
            ))

        try:
            self._fitc_terms(x, y, lengthscale, amplitude, noise, z=z)
        except (FloatingPointError, np.linalg.LinAlgError, ValueError) as exc:
            raise RuntimeError(
                "sparse GP final factorization failed for the fitted hyperparameters "
                f"(lengthscale={lengthscale}, amplitude={amplitude}, noise={noise}, jitter={self.jitter})"
            ) from exc

        self.mean = mean
        self.Z = z
        self._x, self._y = x.copy(), y.copy()
        self.lengthscale, self.amplitude, self.noise = lengthscale, amplitude, noise
        self.fit_receipt = {
            "training_rows": len(x),
            "feature_width": x.shape[1],
            "unique_training_rows": len(np.unique(x, axis=0)),
            "inducing_rows": len(z),
            "seed": seed,
            "optimizer": optimizer_receipt,
        }
        return self

    def predict(self, x_new, *, return_var=False):
        """Posterior mean (and optionally marginal variance) at ``x_new``. O(m^2) per query batch."""
        if self.Z is None:
            raise RuntimeError("call fit() before predict().")
        xs = _as2d(x_new, name="x_new", feature_width=self._x.shape[1])
        if not isinstance(return_var, (bool, np.bool_)):
            raise ValueError("return_var must be a boolean")
        ls, amp, noise = self.lengthscale, self.amplitude, self.noise
        luu, la, v, v_lam, lam, _ = self._fitc_terms(self._x, self._y, ls, amp, noise)
        ksu = _kernel(xs, self.Z, ls, amp, self.kernel)  # s x m
        b = np.linalg.solve(luu, ksu.T)  # m x s = Kuu^{-1/2} Kus
        c = np.linalg.solve(la, b)  # m x s = La^-1 Kuu^{-1/2} Kus  (Sigma = Kuu^{-1/2} La^-T La^-1 Kuu^{-1/2})
        w = np.linalg.solve(la, v_lam @ (self._y - self.mean))  # La^-1 Kuu^{-1/2} Kuf Lam^-1 (y - mean)
        mean = self.mean + c.T @ w  # mu* = Ksu Sigma Kuf Lam^-1 (y - mean)
        if not return_var:
            return mean
        qss = np.sum(b**2, axis=0)  # diag(Qss) = diag(Ksu Kuu^-1 Kus)
        var = amp**2 - qss + np.sum(c**2, axis=0)  # Kss_diag - Qss + diag(Ksu Sigma Kus)
        return mean, np.maximum(var, 1e-12)
