"""Small Torch-backed Gaussian-process regression model.

The implementation fits exact stationary-kernel GP regression with Gaussian
noise through Mixle's generic Torch objective optimizer and exposes prediction
and uncertainty helpers for examples and lightweight modeling workflows.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from mixle.inference.objectives import optimize_torch_objective

_KERNELS = {
    "rbf": "rbf",
    "se": "rbf",
    "squared_exponential": "rbf",
    "matern32": "matern32",
    "matern_3_2": "matern32",
    "matern52": "matern52",
    "matern_5_2": "matern52",
    "matern": "matern52",
}


class GaussianProcessRegressor:
    """Exact GP regression with a stationary kernel and Gaussian observation noise.

    The kernel is RBF (squared-exponential) by default; ``kernel="matern32"`` or ``"matern52"``
    selects the Matern-3/2 or Matern-5/2 covariance, whose less smooth sample paths often fit
    physical responses better than the very smooth RBF.
    """

    def __init__(
        self,
        lengthscale: float = 1.0,
        amplitude: float = 1.0,
        noise: float = 0.1,
        mean: float = 0.0,
        jitter: float = 1.0e-6,
        kernel: str = "rbf",
        engine: Any | None = None,
        precision: Any | None = None,
    ) -> None:
        self.kernel_name = _KERNELS.get(str(kernel).lower())
        if self.kernel_name is None:
            raise ValueError(f"unknown kernel {kernel!r}; choose from {sorted(set(_KERNELS))}.")
        lengthscale = _finite_positive(lengthscale, "lengthscale")
        amplitude = _finite_positive(amplitude, "amplitude")
        noise = _finite_positive(noise, "noise")
        jitter = _finite_positive(jitter, "jitter")
        mean = _finite_scalar(mean, "mean")
        torch, engine = _torch_engine(engine, precision=precision)
        self.torch = torch
        self.engine = engine
        self.log_lengthscale = _raw_positive(torch, engine, lengthscale)
        self.log_amplitude = _raw_positive(torch, engine, amplitude)
        self.log_noise = _raw_positive(torch, engine, noise)
        self.mean = engine.asarray(mean).clone().detach().requires_grad_(True)
        self.jitter = jitter

    def parameters(self):
        """Return trainable raw kernel/noise parameters and the mean."""
        return [self.log_lengthscale, self.log_amplitude, self.log_noise, self.mean]

    @property
    def lengthscale(self) -> float:
        """Return the fitted kernel lengthscale."""
        return float(self.log_lengthscale.detach().exp().cpu().item())

    @property
    def amplitude(self) -> float:
        """Return the fitted kernel amplitude."""
        return float(self.log_amplitude.detach().exp().cpu().item())

    @property
    def noise(self) -> float:
        """Return the fitted Gaussian observation-noise standard deviation."""
        return float(self.log_noise.detach().exp().cpu().item())

    def _xy(self, x: Any, y: Any) -> tuple[Any, Any]:
        xx = self._x(x, "x")
        yy = self.engine.asarray(y)
        if len(yy.shape) == 2 and yy.shape[1] == 1:
            yy = yy[:, 0]
        elif len(yy.shape) != 1:
            raise ValueError("y must be a one-dimensional vector or a two-dimensional single-column matrix")
        if yy.shape[0] != xx.shape[0]:
            raise ValueError(f"x and y must contain the same number of rows, got {xx.shape[0]} and {yy.shape[0]}")
        if yy.shape[0] == 0:
            raise ValueError("x and y must contain at least one observation")
        if not bool(self.torch.all(self.torch.isfinite(yy)).detach().cpu().item()):
            raise ValueError("y must contain only finite values")
        return xx, yy

    def _x(self, value: Any, name: str, *, n_features: int | None = None, allow_empty: bool = False) -> Any:
        xx = self.engine.asarray(value)
        if len(xx.shape) == 1:
            xx = xx[:, None]
        elif len(xx.shape) != 2:
            raise ValueError(f"{name} must be a one- or two-dimensional input array")
        if xx.shape[1] == 0:
            raise ValueError(f"{name} must contain at least one feature")
        if not allow_empty and xx.shape[0] == 0:
            raise ValueError(f"{name} must contain at least one row")
        if n_features is not None and xx.shape[1] != n_features:
            raise ValueError(f"{name} must contain exactly {n_features} features, got {xx.shape[1]}")
        if not bool(self.torch.all(self.torch.isfinite(xx)).detach().cpu().item()):
            raise ValueError(f"{name} must contain only finite values")
        return xx

    def _validate_parameters(self) -> None:
        for name, parameter in (
            ("lengthscale", self.log_lengthscale.exp()),
            ("amplitude", self.log_amplitude.exp()),
            ("noise", self.log_noise.exp()),
        ):
            value = parameter.detach()
            if not bool(self.torch.all(self.torch.isfinite(value)).cpu().item()) or not bool(
                self.torch.all(value > 0.0).cpu().item()
            ):
                raise RuntimeError(f"Gaussian-process {name} is not finite and positive")
        if not bool(self.torch.all(self.torch.isfinite(self.mean.detach())).cpu().item()):
            raise RuntimeError("Gaussian-process mean is not finite")

    def kernel(self, x1: Any, x2: Any) -> Any:
        """Return the covariance matrix between two input arrays under the configured kernel."""
        torch = self.torch
        self._validate_parameters()
        x1 = self._x(x1, "x1", allow_empty=True)
        x2 = self._x(x2, "x2", n_features=x1.shape[1], allow_empty=True)
        diff = (x1[:, None, :] - x2[None, :, :]) / self.log_lengthscale.exp()
        dist2 = torch.sum(diff * diff, dim=2)
        amp2 = self.log_amplitude.exp() ** 2
        if self.kernel_name == "rbf":
            return amp2 * torch.exp(-0.5 * dist2)
        # Matern kernels need the lengthscale-scaled Euclidean distance; the positive floor keeps the
        # sqrt subdifferentiable at zero separation.
        r = torch.sqrt(torch.clamp(dist2, min=0.0) + 1.0e-12)
        if self.kernel_name == "matern32":
            sqrt3 = 3.0**0.5
            return amp2 * (1.0 + sqrt3 * r) * torch.exp(-sqrt3 * r)
        sqrt5 = 5.0**0.5  # matern52
        return amp2 * (1.0 + sqrt5 * r + (5.0 / 3.0) * dist2) * torch.exp(-sqrt5 * r)

    def log_marginal_likelihood(self, x: Any, y: Any) -> Any:
        """Return the exact GP log marginal likelihood for training data."""
        torch = self.torch
        self._validate_parameters()
        xx, yy = self._xy(x, y)
        n = yy.shape[0]
        k = self.kernel(xx, xx)
        eye = torch.eye(n, dtype=yy.dtype, device=yy.device)
        noise2 = self.log_noise.exp() ** 2
        k = k + (noise2 + self.jitter) * eye
        centered = yy - self.mean
        chol = torch.linalg.cholesky(k)
        alpha = torch.cholesky_solve(centered[:, None], chol)[:, 0]
        quad = torch.dot(centered, alpha)
        logdet = 2.0 * torch.sum(torch.log(torch.diagonal(chol)))
        return -0.5 * (quad + logdet + n * np.log(2.0 * np.pi))

    def fit(
        self,
        x: Any,
        y: Any,
        max_its: int = 500,
        lr: float = 0.05,
        optimizer: str = "adam",
        tol: float = 1.0e-7,
        out: Any | None = None,
        print_iter: int = 100,
        return_result: bool = False,
        restore_best: bool = True,
    ) -> Any:
        """Maximize the GP log marginal likelihood and return ``self``.

        Returns the fitted model so ``model = gp.fit(x, y)`` works like every
        other ``Model.fit`` in ``mixle.models``. Set ``return_result=True``
        for the full objective diagnostics (an ``ObjectiveFitResult`` carrying
        the objective value, iteration count, and history).

        Compatibility note: before 0.8.0 the default return was the
        ``(value, iterations)`` tuple; those live behind ``return_result=True``
        now (``result.value`` / ``result.iterations``).
        """
        self._xy(x, y)
        result = optimize_torch_objective(
            self.parameters(),
            lambda: self.log_marginal_likelihood(x, y),
            engine=self.engine,
            max_its=max_its,
            lr=lr,
            optimizer=optimizer,
            tol=tol,
            maximize=True,
            out=out,
            print_iter=print_iter,
            return_result=return_result,
            restore_best=restore_best,
        )
        if return_result:
            return result
        return self

    def predict(self, x_train: Any, y_train: Any, x_new: Any, return_cov: bool = False) -> Any:
        """Return posterior predictive mean, and optionally covariance."""
        torch = self.torch
        with torch.no_grad():
            self._validate_parameters()
            x, y = self._xy(x_train, y_train)
            xs = self._x(x_new, "x_new", n_features=x.shape[1], allow_empty=True)
            n = y.shape[0]
            k = self.kernel(x, x)
            eye = torch.eye(n, dtype=y.dtype, device=y.device)
            k = k + (self.log_noise.exp() ** 2 + self.jitter) * eye
            chol = torch.linalg.cholesky(k)
            centered = y - self.mean
            alpha = torch.cholesky_solve(centered[:, None], chol)
            kxs = self.kernel(x, xs)
            mean = self.mean + kxs.T.matmul(alpha)[:, 0]
            if not return_cov:
                return mean.detach().cpu().numpy()
            v = torch.linalg.solve_triangular(chol, kxs, upper=False)
            cov = self.kernel(xs, xs) - v.T.matmul(v)
            cov = 0.5 * (cov + cov.T)
            if not bool(torch.all(torch.isfinite(cov)).cpu().item()):
                raise RuntimeError("Gaussian-process posterior covariance contains non-finite values")
            scale = (
                max(1.0, float(torch.linalg.matrix_norm(cov, ord=float("inf")).cpu().item()))
                if cov.numel()
                else 1.0
            )
            eps = torch.finfo(cov.dtype).eps
            tolerance = max(
                100.0 * eps * max(1, cov.shape[0]) * scale,
                math.sqrt(eps) * scale,
            )
            diagonal = torch.diagonal(cov)
            if bool(torch.any(diagonal < -tolerance).cpu().item()):
                raise RuntimeError("Gaussian-process posterior covariance has materially negative variance")
            if cov.numel():
                eigenvalues, eigenvectors = torch.linalg.eigh(cov)
                if bool(torch.any(eigenvalues < -tolerance).cpu().item()):
                    minimum = float(torch.min(eigenvalues).cpu().item())
                    raise RuntimeError(
                        "Gaussian-process posterior covariance is not positive semidefinite "
                        f"(minimum eigenvalue {minimum:.6g}, tolerance {tolerance:.6g})"
                    )
                # Subtraction in Kss - V.T@V can introduce tolerance-scale negative modes even though
                # the analytical posterior is PSD. Project only those certified roundoff modes to the
                # closed PSD cone; materially negative modes still fail above.
                cov = (eigenvectors * torch.clamp(eigenvalues, min=0.0)) @ eigenvectors.T
                cov = 0.5 * (cov + cov.T)
            return mean.detach().cpu().numpy(), cov.detach().cpu().numpy()

    def predict_monotone(self, x_train: Any, y_train: Any, x_new: Any, increasing: bool = True) -> np.ndarray:
        """Return the posterior-mean prediction projected to be monotone in scalar ``x_new``.

        Predicts the GP posterior mean at ``x_new`` and projects it onto the monotone cone
        (non-decreasing if ``increasing`` else non-increasing) by pool-adjacent-violators in
        ``x_new`` order -- the L2-closest monotone curve to the GP mean. Intended for scalar (1-D)
        inputs (e.g. monotone age-depth / dose-response fits); reduces to :meth:`predict` when the
        posterior mean is already monotone.
        """
        if not isinstance(increasing, (bool, np.bool_)):
            raise TypeError("increasing must be a boolean")
        train_coordinates = _scalar_coordinates(x_train, "x_train")
        x_sort_key = _scalar_coordinates(x_new, "x_new", allow_empty=True)
        mean = np.asarray(self.predict(train_coordinates, y_train, x_sort_key), dtype=float).reshape(-1)
        if x_sort_key.size == 0:
            return mean
        order = np.argsort(x_sort_key, kind="stable")
        sorted_x = x_sort_key[order]
        sorted_mean = mean[order]
        _, first, inverse, counts = np.unique(
            sorted_x, return_index=True, return_inverse=True, return_counts=True
        )
        group_mean = np.add.reduceat(sorted_mean, first) / counts
        fitted_groups = _pava(group_mean if increasing else -group_mean, weights=counts)
        if not increasing:
            fitted_groups = -fitted_groups
        fitted = fitted_groups[inverse]
        out = np.empty_like(mean)
        out[order] = fitted
        return out


def _pava(y: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Return the weighted-L2 closest non-decreasing sequence to ``y``."""
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or not np.all(np.isfinite(y)):
        raise ValueError("PAVA values must be a one-dimensional finite vector")
    n = y.size
    if weights is None:
        weights = np.ones(n, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (n,) or not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
            raise ValueError("PAVA weights must be a finite positive vector matching the values")
    if n <= 1:
        return y.astype(float).copy()
    vals: list[float] = []
    block_weights: list[float] = []
    block_counts: list[int] = []
    for yi, wi in zip(y, weights):
        vals.append(float(yi))
        block_weights.append(float(wi))
        block_counts.append(1)
        while len(vals) >= 2 and vals[-2] > vals[-1]:
            v2, w2, c2 = vals.pop(), block_weights.pop(), block_counts.pop()
            v1, w1, c1 = vals.pop(), block_weights.pop(), block_counts.pop()
            vals.append((v1 * w1 + v2 * w2) / (w1 + w2))
            block_weights.append(w1 + w2)
            block_counts.append(c1 + c2)
    out = np.empty(n, dtype=float)
    pos = 0
    for v, c in zip(vals, block_counts):
        out[pos : pos + c] = v
        pos += c
    return out


def _scalar_coordinates(value: Any, name: str, *, allow_empty: bool = False) -> np.ndarray:
    try:
        coordinates = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric scalar coordinates") from exc
    if coordinates.ndim == 2 and coordinates.shape[1] == 1:
        coordinates = coordinates[:, 0]
    elif coordinates.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional or a two-dimensional single-column matrix")
    if not allow_empty and coordinates.size == 0:
        raise ValueError(f"{name} must contain at least one coordinate")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError(f"{name} must contain only finite coordinates")
    return coordinates


def _torch_engine(engine: Any | None, precision: Any | None = None) -> tuple[Any, Any]:
    try:
        import torch
    except ImportError as e:  # pragma: no cover
        raise ImportError("GaussianProcessRegressor requires torch.") from e
    if engine is None:
        from mixle.engines import TorchEngine

        engine = TorchEngine(dtype=precision or torch.float64)
    elif precision is not None:
        from mixle.engines import engine_with_precision

        engine = engine_with_precision(engine, precision)
    return torch, engine


def _raw_positive(torch: Any, engine: Any, value: float) -> Any:
    return torch.log(engine.asarray(float(value))).clone().detach().requires_grad_(True)


def _finite_positive(value: Any, name: str) -> float:
    result = _finite_scalar(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _finite_scalar(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real scalar") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result
