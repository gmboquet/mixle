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


def _as2d(x: np.ndarray) -> np.ndarray:
    """A 1-D array is ``n`` points in 1-D (shape (n, 1)); a 2-D array is ``n`` points in ``d`` dims."""
    x = np.asarray(x, dtype=float)
    return x[:, None] if x.ndim == 1 else x


class SparseGaussianProcessRegressor:
    """Sparse GP regression with ``m`` inducing points (FITC).

    Args:
        lengthscale, amplitude, noise: initial kernel/noise hyperparameters (all positive).
        kernel: ``'rbf'``, ``'matern32'`` or ``'matern52'``.
        n_inducing: number of inducing points (placed by k-means-lite over the training inputs at ``fit``).
    """

    def __init__(self, lengthscale=1.0, amplitude=1.0, noise=0.1, kernel="rbf", n_inducing=50):
        self.lengthscale = float(lengthscale)
        self.amplitude = float(amplitude)
        self.noise = float(noise)
        self.kernel = str(kernel).lower()
        self.n_inducing = int(n_inducing)
        self.Z = None

    def _place_inducing(self, x, rng):
        """Pick inducing inputs as a random subset of the unique training inputs."""
        x = np.atleast_2d(x)
        m = min(self.n_inducing, x.shape[0])
        idx = rng.choice(x.shape[0], size=m, replace=False)
        return x[idx].copy()

    def _fitc_terms(self, x, y, ls, amp, noise):
        """Shared FITC quantities at given hyperparameters (Kuu chol, Sigma chol, the y-weighted vector)."""
        z = self.Z
        kuu = _kernel(z, z, ls, amp, self.kernel) + 1e-8 * np.eye(len(z))
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

    def _neg_log_marglik(self, x, y, ls, amp, noise):
        luu, la, v, v_lam, lam, _ = self._fitc_terms(x, y, ls, amp, noise)
        n = len(x)
        ym = y - self.mean
        # log|Q_ff + Lam| = log|A| + sum log lam ; quadratic via Woodbury
        logdet = 2.0 * np.sum(np.log(np.diag(la))) + np.sum(np.log(lam))
        vy = v_lam @ ym  # m
        w = np.linalg.solve(la, vy)
        quad = np.sum(ym**2 / lam) - np.sum(w**2)
        return 0.5 * (logdet + quad + n * np.log(2.0 * np.pi))

    def fit(self, x, y, *, optimize=True, seed=0, max_iter=100):
        """Place inducing points and (optionally) fit hyperparameters by the FITC marginal likelihood."""
        x = _as2d(x)
        y = np.asarray(y, dtype=float).ravel()
        self.mean = float(y.mean())
        self.Z = self._place_inducing(x, np.random.RandomState(seed))
        self._x, self._y = x, y
        if optimize:
            theta0 = np.log([self.lengthscale, self.amplitude, self.noise])

            def obj(t):
                ls, amp, noise = np.exp(t)
                return self._neg_log_marglik(x, y, ls, amp, noise)

            res = minimize(obj, theta0, method="Nelder-Mead", options={"maxiter": max_iter, "xatol": 1e-3})
            self.lengthscale, self.amplitude, self.noise = np.exp(res.x)
        return self

    def predict(self, x_new, *, return_var=False):
        """Posterior mean (and optionally marginal variance) at ``x_new``. O(m^2) per query batch."""
        if self.Z is None:
            raise RuntimeError("call fit() before predict().")
        xs = _as2d(x_new)
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
