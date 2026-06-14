"""Small Torch-backed Gaussian-process regression model."""

from __future__ import annotations

from typing import Any

import numpy as np

from pysp.utils.objectives import optimize_torch_objective


class GaussianProcessRegressor:
    """Exact GP regression with an RBF kernel and Gaussian observation noise."""

    def __init__(
        self,
        lengthscale: float = 1.0,
        amplitude: float = 1.0,
        noise: float = 0.1,
        mean: float = 0.0,
        jitter: float = 1.0e-6,
        engine: Any | None = None,
        precision: Any | None = None,
    ) -> None:
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
        """Return the fitted RBF lengthscale."""
        return float(self.log_lengthscale.detach().exp().cpu().item())

    @property
    def amplitude(self) -> float:
        """Return the fitted RBF amplitude."""
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
        """Return the RBF covariance matrix between two input arrays."""
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
        return amp2 * torch.exp(-0.5 * dist2)

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
        """Maximize the GP log marginal likelihood.

        The default return shape is the historical ``(value, iterations)``
        tuple.  Set ``return_result=True`` for the full objective diagnostics.
        """
        return optimize_torch_objective(
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


def _torch_engine(engine: Any | None, precision: Any | None = None) -> tuple[Any, Any]:
    try:
        import torch
    except ImportError as e:  # pragma: no cover
        raise ImportError("GaussianProcessRegressor requires torch.") from e
    if engine is None:
        from pysp.engines import TorchEngine

        engine = TorchEngine(dtype=precision or torch.float64)
    elif precision is not None:
        from pysp.engines import engine_with_precision

        engine = engine_with_precision(engine, precision)
    return torch, engine


def _raw_positive(torch: Any, engine: Any, value: float) -> Any:
    return torch.log(engine.asarray(float(value))).clone().detach().requires_grad_(True)
