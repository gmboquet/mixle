"""Torch neural-network objective helpers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from mixle.inference.objectives import optimize_torch_objective


class GaussianRegressionNeuralNetwork:
    """A Torch module trained with a Gaussian regression log likelihood.

    The wrapped module predicts the response mean and this helper learns a
    scalar observation noise alongside module weights.  It uses the same
    generic Torch objective optimizer as the distribution objective helpers.
    """

    def __init__(
        self, module: Any, noise: float = 1.0, engine: Any | None = None, precision: Any | None = None
    ) -> None:
        torch, engine = _torch_engine(engine, precision=precision, owner="GaussianRegressionNeuralNetwork")
        self.torch = torch
        self.engine = engine
        self.module = module.to(device=engine.device, dtype=engine.dtype)
        self.log_noise = torch.log(engine.asarray(float(noise))).clone().detach().requires_grad_(True)

    def parameters(self) -> Iterable[Any]:
        """Return trainable module parameters plus the raw noise parameter."""
        return list(self.module.parameters()) + [self.log_noise]

    @property
    def noise(self) -> float:
        """Return the fitted observation standard deviation."""
        return float(self.log_noise.detach().exp().cpu().item())

    def _xy(self, x: Any, y: Any) -> tuple[Any, Any]:
        xx = self.engine.asarray(x)
        yy = self.engine.asarray(y)
        if len(xx.shape) == 1:
            xx = xx[:, None]
        if len(yy.shape) == 1:
            yy = yy[:, None]
        return xx, yy

    def predict_tensor(self, x: Any) -> Any:
        """Return module predictions as a Torch tensor on the configured engine."""
        xx = self.engine.asarray(x)
        if len(xx.shape) == 1:
            xx = xx[:, None]
        return self.module(xx)

    def log_likelihood(self, x: Any, y: Any) -> Any:
        """Return the summed Gaussian regression log likelihood."""
        torch = self.torch
        xx, yy = self._xy(x, y)
        pred = self.module(xx)
        noise2 = self.log_noise.exp() ** 2
        resid = yy - pred
        return -0.5 * torch.sum(resid * resid / noise2 + torch.log(2.0 * torch.pi * noise2))

    def fit(
        self,
        x: Any,
        y: Any,
        max_its: int = 500,
        lr: float = 0.01,
        optimizer: str = "adam",
        tol: float = 1.0e-7,
        out: Any | None = None,
        print_iter: int = 100,
        return_result: bool = False,
        restore_best: bool = True,
    ) -> Any:
        """Maximize the Gaussian regression log likelihood.

        The default return shape is the historical ``(value, iterations)``
        tuple.  Set ``return_result=True`` for the full objective diagnostics.
        """
        return optimize_torch_objective(
            self.parameters(),
            lambda: self.log_likelihood(x, y),
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

    def predict(self, x: Any) -> np.ndarray:
        """Return mean predictions as a NumPy array."""
        with self.torch.no_grad():
            return self.predict_tensor(x).detach().cpu().numpy()


class CategoricalClassificationNeuralNetwork:
    """A Torch classifier wrapper optimized by summed categorical log likelihood.

    The wrapped module must return one logits row per observation.  Fitting is
    delegated to ``optimize_torch_objective`` so classification examples get the
    same convergence diagnostics and best-state restoration as distribution
    objectives.
    """

    def __init__(self, module: Any, engine: Any | None = None, precision: Any | None = None) -> None:
        torch, engine = _torch_engine(engine, precision=precision, owner="CategoricalClassificationNeuralNetwork")
        self.torch = torch
        self.engine = engine
        self.module = module.to(device=engine.device, dtype=engine.dtype)

    def parameters(self) -> Iterable[Any]:
        """Return trainable parameters of the wrapped classification module."""
        return list(self.module.parameters())

    def _x(self, x: Any) -> Any:
        xx = self.engine.asarray(x)
        if len(xx.shape) == 1:
            xx = xx[:, None]
        return xx

    def _labels(self, y: Any) -> Any:
        labels = self.engine.asarray(y, dtype=self.torch.long)
        if len(labels.shape) != 1:
            labels = labels.reshape(-1)
        return labels

    def logits_tensor(self, x: Any) -> Any:
        """Return raw class logits for ``x`` as a Torch tensor."""
        return self.module(self._x(x))

    def log_likelihood(self, x: Any, y: Any) -> Any:
        """Return the summed categorical log likelihood for integer labels."""
        logits = self.logits_tensor(x)
        labels = self._labels(y)
        if len(logits.shape) != 2:
            raise ValueError("classification module must return a (n, classes) logits matrix.")
        if logits.shape[0] != labels.shape[0]:
            raise ValueError("classification labels must match row count.")
        return -self.torch.nn.functional.cross_entropy(logits, labels, reduction="sum")

    def fit(
        self,
        x: Any,
        y: Any,
        max_its: int = 500,
        lr: float = 0.01,
        optimizer: str = "adam",
        tol: float = 1.0e-7,
        out: Any | None = None,
        print_iter: int = 100,
        return_result: bool = False,
        restore_best: bool = True,
    ) -> Any:
        """Maximize the categorical classification log likelihood."""
        return optimize_torch_objective(
            self.parameters(),
            lambda: self.log_likelihood(x, y),
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

    def predict_proba_tensor(self, x: Any) -> Any:
        """Return class probabilities for ``x`` as a Torch tensor."""
        logits = self.logits_tensor(x)
        if len(logits.shape) != 2:
            raise ValueError("classification module must return a (n, classes) logits matrix.")
        return self.torch.softmax(logits, dim=1)

    def predict_proba(self, x: Any) -> np.ndarray:
        """Return class probabilities for ``x`` as a NumPy array."""
        with self.torch.no_grad():
            return self.predict_proba_tensor(x).detach().cpu().numpy()

    def predict(self, x: Any) -> np.ndarray:
        """Return maximum-probability class labels for ``x``."""
        return np.argmax(self.predict_proba(x), axis=1)


class PoissonRegressionNeuralNetwork:
    """A Torch count-regression wrapper optimized by Poisson log likelihood.

    The wrapped module predicts log rates.  Observed counts must be
    non-negative and match the module output shape after one-dimensional inputs
    are promoted to column vectors.
    """

    def __init__(self, module: Any, engine: Any | None = None, precision: Any | None = None) -> None:
        torch, engine = _torch_engine(engine, precision=precision, owner="PoissonRegressionNeuralNetwork")
        self.torch = torch
        self.engine = engine
        self.module = module.to(device=engine.device, dtype=engine.dtype)

    def parameters(self) -> Iterable[Any]:
        """Return trainable parameters of the wrapped log-rate module."""
        return list(self.module.parameters())

    def _x(self, x: Any) -> Any:
        xx = self.engine.asarray(x)
        if len(xx.shape) == 1:
            xx = xx[:, None]
        return xx

    def _counts_like(self, y: Any, log_rate: Any) -> Any:
        counts = self.engine.asarray(y)
        if len(counts.shape) == 1 and len(log_rate.shape) == 2 and log_rate.shape[1] == 1:
            counts = counts[:, None]
        if tuple(counts.shape) != tuple(log_rate.shape):
            raise ValueError("Poisson counts must match the module log-rate shape.")
        if bool(self.torch.any(counts < 0).detach().cpu().item()):
            raise ValueError("Poisson counts must be non-negative.")
        return counts

    def log_rate_tensor(self, x: Any) -> Any:
        """Return predicted log rates as a Torch tensor."""
        return self.module(self._x(x))

    def log_likelihood(self, x: Any, y: Any) -> Any:
        """Return the summed Poisson count log likelihood."""
        torch = self.torch
        log_rate = self.log_rate_tensor(x)
        counts = self._counts_like(y, log_rate)
        return torch.sum(counts * log_rate - torch.exp(log_rate) - torch.lgamma(counts + 1.0))

    def fit(
        self,
        x: Any,
        y: Any,
        max_its: int = 500,
        lr: float = 0.01,
        optimizer: str = "adam",
        tol: float = 1.0e-7,
        out: Any | None = None,
        print_iter: int = 100,
        return_result: bool = False,
        restore_best: bool = True,
    ) -> Any:
        """Maximize the Poisson count log likelihood."""
        return optimize_torch_objective(
            self.parameters(),
            lambda: self.log_likelihood(x, y),
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

    def predict_rate_tensor(self, x: Any) -> Any:
        """Return predicted Poisson rates as a Torch tensor."""
        return self.torch.exp(self.log_rate_tensor(x))

    def predict_rate(self, x: Any) -> np.ndarray:
        """Return predicted Poisson rates as a NumPy array."""
        with self.torch.no_grad():
            return self.predict_rate_tensor(x).detach().cpu().numpy()

    def predict(self, x: Any) -> np.ndarray:
        """Return rounded count predictions as integer NumPy values."""
        return np.rint(self.predict_rate(x)).astype(np.int64)


def make_mlp(input_dim: int, hidden_dims: Sequence[int], output_dim: int = 1, activation: str = "tanh") -> Any:
    """Create a simple fully connected Torch MLP."""
    try:
        import torch
    except ImportError as e:  # pragma: no cover
        raise ImportError("make_mlp requires torch.") from e
    activations = {
        "relu": torch.nn.ReLU,
        "tanh": torch.nn.Tanh,
        "gelu": torch.nn.GELU,
        "sigmoid": torch.nn.Sigmoid,
    }
    if activation not in activations:
        raise ValueError("Unknown activation %s. Expected one of %s." % (activation, ", ".join(sorted(activations))))
    if int(input_dim) <= 0 or int(output_dim) <= 0 or any(int(h) <= 0 for h in hidden_dims):
        raise ValueError(
            "make_mlp dims must be positive; got input_dim=%r hidden_dims=%r output_dim=%r"
            % (input_dim, list(hidden_dims), output_dim)
        )
    dims = [int(input_dim)] + [int(h) for h in hidden_dims] + [int(output_dim)]
    layers = []
    for i in range(len(dims) - 1):
        layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(activations[activation]())
    return torch.nn.Sequential(*layers)


def make_monotonic_mlp(
    input_dim: int, hidden_dims: Sequence[int], output_dim: int = 1, *, increasing: bool = True
) -> Any:
    """A fully connected Torch MLP that is monotonic in every input dimension jointly, BY CONSTRUCTION.

    Each layer's weight matrix is reparameterized through ``softplus`` before use, so every weight is
    strictly non-negative; composed with the (smooth, strictly increasing) ``Softplus`` activation, a
    non-negative-weight affine map followed by an increasing activation is itself increasing, and that
    property is closed under composition -- so the whole network is provably non-decreasing in every
    input coordinate, with no penalty term and no post-hoc check needed. ``increasing=False`` negates the
    output, giving a network non-increasing in every coordinate instead.

    This is a hard architectural constraint (unlike :class:`~mixle.models.pinn.PINNRegression`'s soft
    residual penalty): the guarantee holds at every point in input space, not just where training data
    landed. Drops into the same wrappers as :func:`make_mlp` -- :class:`~mixle.models.neural_leaf.NeuralGaussian`
    for regression, :class:`~mixle.models.softmax_leaf.NeuralCategorical` for classification -- no other
    changes needed. Only jointly monotonic in ALL inputs; a network monotonic in some coordinates and free
    in others needs a two-path (monotonic + unconstrained) variant, not built here.
    """
    try:
        import torch
    except ImportError as e:  # pragma: no cover
        raise ImportError("make_monotonic_mlp requires torch.") from e
    if int(input_dim) <= 0 or int(output_dim) <= 0 or any(int(h) <= 0 for h in hidden_dims):
        raise ValueError(
            "make_monotonic_mlp dims must be positive; got input_dim=%r hidden_dims=%r output_dim=%r"
            % (input_dim, list(hidden_dims), output_dim)
        )

    class _NonNegativeLinear(torch.nn.Module):
        def __init__(self, in_features: int, out_features: int) -> None:
            super().__init__()
            # softplus(0) = log(2) =~ 0.69, NOT ~0 -- a naive small-mean raw_weight init would put every
            # effective weight near 0.69 rather than near 0, exploding the signal through depth. Instead
            # init the EFFECTIVE weight at a normal fan-in scale, then invert softplus to get raw_weight.
            fan_in = max(in_features, 1)
            target = torch.empty(out_features, in_features).uniform_(1e-3, 1.0 / fan_in**0.5)
            self.raw_weight = torch.nn.Parameter(target + torch.log(-torch.expm1(-target)))  # softplus^-1
            self.bias = torch.nn.Parameter(torch.zeros(out_features))

        def forward(self, x: Any) -> Any:
            weight = torch.nn.functional.softplus(self.raw_weight)
            return torch.nn.functional.linear(x, weight, self.bias)

    class _NegateOutput(torch.nn.Module):
        def __init__(self, module: Any) -> None:
            super().__init__()
            self.module = module

        def forward(self, x: Any) -> Any:
            return -self.module(x)

    dims = [int(input_dim)] + [int(h) for h in hidden_dims] + [int(output_dim)]
    layers: list[Any] = []
    for i in range(len(dims) - 1):
        layers.append(_NonNegativeLinear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(torch.nn.Softplus())
    module = torch.nn.Sequential(*layers)
    return module if increasing else _NegateOutput(module)


def _torch_engine(
    engine: Any | None, precision: Any | None = None, owner: str = "GaussianRegressionNeuralNetwork"
) -> tuple[Any, Any]:
    try:
        import torch
    except ImportError as e:  # pragma: no cover
        raise ImportError("%s requires torch." % owner) from e
    if engine is None:
        from mixle.engines import TorchEngine

        engine = TorchEngine(dtype=precision or torch.float64)
    elif precision is not None:
        from mixle.engines import engine_with_precision

        engine = engine_with_precision(engine, precision)
    return torch, engine
