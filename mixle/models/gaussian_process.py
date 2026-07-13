"""Small Torch-backed Gaussian-process regression model.

The implementation fits exact stationary-kernel GP regression with Gaussian
noise through Mixle's generic Torch objective optimizer and exposes prediction
and uncertainty helpers for examples and lightweight modeling workflows.
"""

from __future__ import annotations

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
        torch, engine = _torch_engine(engine, precision=precision)
        self.torch = torch
        self.engine = engine
        self.log_lengthscale = _raw_positive(torch, engine, lengthscale)
        self.log_amplitude = _raw_positive(torch, engine, amplitude)
        self.log_noise = _raw_positive(torch, engine, noise)
        self.mean = engine.asarray(float(mean)).clone().detach().requires_grad_(True)
        self.jitter = float(jitter)

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
        xx = self.engine.asarray(x)
        if len(xx.shape) == 1:
            xx = xx[:, None]
        yy = self.engine.asarray(y)
        if len(yy.shape) > 1:
            yy = yy.reshape((-1,))
        return xx, yy

    def kernel(self, x1: Any, x2: Any) -> Any:
        """Return the covariance matrix between two input arrays under the configured kernel."""
        torch = self.torch
        x1 = self.engine.asarray(x1)
        x2 = self.engine.asarray(x2)
        if len(x1.shape) == 1:
            x1 = x1[:, None]
        if len(x2.shape) == 1:
            x2 = x2[:, None]
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
            x, y = self._xy(x_train, y_train)
            xs = self.engine.asarray(x_new)
            if len(xs.shape) == 1:
                xs = xs[:, None]
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
            return mean.detach().cpu().numpy(), cov.detach().cpu().numpy()

    def predict_monotone(self, x_train: Any, y_train: Any, x_new: Any, increasing: bool = True) -> np.ndarray:
        """Return the posterior-mean prediction projected to be monotone in scalar ``x_new``.

        Predicts the GP posterior mean at ``x_new`` and projects it onto the monotone cone
        (non-decreasing if ``increasing`` else non-increasing) by pool-adjacent-violators in
        ``x_new`` order -- the L2-closest monotone curve to the GP mean. Intended for scalar (1-D)
        inputs (e.g. monotone age-depth / dose-response fits); reduces to :meth:`predict` when the
        posterior mean is already monotone.
        """
        x_sort_key = np.asarray(x_new, dtype=float).reshape(-1)
        mean = np.asarray(self.predict(x_train, y_train, x_new), dtype=float).reshape(-1)
        order = np.argsort(x_sort_key, kind="stable")
        fitted = _pava(mean[order] if increasing else -mean[order])
        if not increasing:
            fitted = -fitted
        out = np.empty_like(mean)
        out[order] = fitted
        return out


def _pava(y: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators: the L2-closest non-decreasing sequence to ``y`` (equal weights)."""
    y = np.asarray(y, dtype=float)
    n = y.size
    if n <= 1:
        return y.astype(float).copy()
    vals: list[float] = []
    counts: list[int] = []
    for yi in y:
        vals.append(float(yi))
        counts.append(1)
        while len(vals) >= 2 and vals[-2] > vals[-1]:
            v2, c2 = vals.pop(), counts.pop()
            v1, c1 = vals.pop(), counts.pop()
            vals.append((v1 * c1 + v2 * c2) / (c1 + c2))
            counts.append(c1 + c2)
    out = np.empty(n, dtype=float)
    pos = 0
    for v, c in zip(vals, counts):
        out[pos : pos + c] = v
        pos += c
    return out


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
