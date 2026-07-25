"""Torch neural-network wrappers trained through Mixle objective utilities.

The wrappers expose Gaussian regression and categorical classification models
with consistent log-likelihood objectives, convergence diagnostics, precision
handling, and prediction helpers.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from numbers import Integral, Real
from typing import Any

import numpy as np

from mixle.inference.objectives import optimize_torch_objective
from mixle.models.grad_leaf import _module_mode

try:
    import torch as _TORCH
except ImportError:  # pragma: no cover - Torch remains optional
    _TORCH = None

_ModuleBase = object if _TORCH is None else _TORCH.nn.Module


def _positive_finite(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a positive finite real number")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite real number")
    return result


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _module_output(module: Any, x: Any, rows: int, where: str) -> Any:
    value = module(x)
    if not isinstance(value, _TORCH.Tensor) or value.ndim != 2 or value.shape[0] != rows or value.shape[1] == 0:
        actual = getattr(value, "shape", None)
        raise ValueError(f"{where} module output must have shape (rows, outputs); got {actual}")
    if not bool(_TORCH.isfinite(value).all()):
        raise ValueError(f"{where} module output must be finite")
    return value


class NonNegativeLinear(_ModuleBase):
    """Pickle-stable linear layer with softplus-constrained non-negative weights."""

    def __init__(self, in_features: int, out_features: int) -> None:
        if _TORCH is None:  # pragma: no cover
            raise ImportError("NonNegativeLinear requires torch")
        super().__init__()
        self.in_features = _positive_int(in_features, "in_features")
        self.out_features = _positive_int(out_features, "out_features")
        target = _TORCH.empty(self.out_features, self.in_features).uniform_(1e-3, 1.0 / self.in_features**0.5)
        self.raw_weight = _TORCH.nn.Parameter(target + _TORCH.log(-_TORCH.expm1(-target)))
        self.bias = _TORCH.nn.Parameter(_TORCH.zeros(self.out_features))

    def forward(self, x: Any) -> Any:
        weight = _TORCH.nn.functional.softplus(self.raw_weight)
        return _TORCH.nn.functional.linear(x, weight, self.bias)


class NegateOutput(_ModuleBase):
    """Pickle-stable sign wrapper for decreasing monotone networks."""

    def __init__(self, module: Any) -> None:
        if _TORCH is None:  # pragma: no cover
            raise ImportError("NegateOutput requires torch")
        super().__init__()
        if not isinstance(module, _TORCH.nn.Module):
            raise TypeError("module must be a torch.nn.Module")
        self.module = module

    def forward(self, x: Any) -> Any:
        return -self.module(x)


class DeepSetNetwork(_ModuleBase):
    """Pickle-stable permutation-invariant network with validated set geometry."""

    def __init__(self, phi: Any, rho: Any, pooling: str, element_dim: int) -> None:
        if _TORCH is None:  # pragma: no cover
            raise ImportError("DeepSetNetwork requires torch")
        super().__init__()
        if not isinstance(phi, _TORCH.nn.Module) or not isinstance(rho, _TORCH.nn.Module):
            raise TypeError("phi and rho must be torch modules")
        if pooling not in ("mean", "sum", "max"):
            raise ValueError("pooling must be 'mean', 'sum', or 'max'")
        self.phi = phi
        self.rho = rho
        self.pooling = pooling
        self.element_dim = _positive_int(element_dim, "element_dim")

    def forward(self, x: Any) -> Any:
        if not isinstance(x, _TORCH.Tensor) or x.ndim < 2:
            raise ValueError("DeepSetNetwork input must have shape (..., set_size, element_dim)")
        if x.shape[-2] == 0 or x.shape[-1] != self.element_dim:
            raise ValueError(
                f"DeepSetNetwork requires non-empty sets with element_dim={self.element_dim}; got {tuple(x.shape)}"
            )
        if not bool(_TORCH.isfinite(x).all()):
            raise ValueError("DeepSetNetwork input must be finite")
        codes = self.phi(x)
        if self.pooling == "mean":
            pooled = codes.mean(dim=-2)
        elif self.pooling == "sum":
            pooled = codes.sum(dim=-2)
        else:
            pooled = codes.max(dim=-2).values
        result = self.rho(pooled)
        if not bool(_TORCH.isfinite(result).all()):
            raise ValueError("DeepSetNetwork output must be finite")
        return result


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
        if not isinstance(module, torch.nn.Module):
            raise TypeError("module must be a torch.nn.Module")
        noise = _positive_finite(noise, "noise")
        self.torch = torch
        self.engine = engine
        self.module = module.to(device=engine.device, dtype=engine.dtype)
        self.log_noise = torch.log(engine.asarray(noise)).clone().detach().requires_grad_(True)

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
        if len(xx.shape) != 2 or len(yy.shape) != 2 or xx.shape[0] == 0 or xx.shape[1] == 0 or yy.shape[1] == 0:
            raise ValueError("Gaussian regression x and y must be non-empty two-dimensional matrices")
        if xx.shape[0] != yy.shape[0]:
            raise ValueError("Gaussian regression x and y must have identical row counts")
        if not bool(self.torch.isfinite(xx).all()) or not bool(self.torch.isfinite(yy).all()):
            raise ValueError("Gaussian regression x and y must be finite")
        return xx, yy

    def predict_tensor(self, x: Any) -> Any:
        """Return module predictions as a Torch tensor on the configured engine."""
        xx = self.engine.asarray(x)
        if len(xx.shape) == 1:
            xx = xx[:, None]
        if len(xx.shape) != 2 or xx.shape[0] == 0 or xx.shape[1] == 0 or not bool(self.torch.isfinite(xx).all()):
            raise ValueError("Gaussian regression x must be a non-empty finite matrix")
        with _module_mode(self.module, train=False), self.torch.no_grad():
            return _module_output(self.module, xx, xx.shape[0], "Gaussian regression")

    def _objective(self, x: Any, y: Any) -> Any:
        torch = self.torch
        xx, yy = self._xy(x, y)
        pred = _module_output(self.module, xx, xx.shape[0], "Gaussian regression")
        if tuple(pred.shape) != tuple(yy.shape):
            raise ValueError(
                f"Gaussian target shape must exactly match predictions; got {tuple(yy.shape)} and {tuple(pred.shape)}"
            )
        noise2 = self.log_noise.exp() ** 2
        if not bool(torch.isfinite(noise2)) or bool(noise2 <= 0.0):
            raise ValueError("Gaussian regression noise must remain positive and finite")
        resid = yy - pred
        result = -0.5 * torch.sum(resid * resid / noise2 + torch.log(2.0 * torch.pi * noise2))
        if not bool(torch.isfinite(result)):
            raise ValueError("Gaussian regression log likelihood became non-finite")
        return result

    def log_likelihood(self, x: Any, y: Any) -> Any:
        """Return the summed Gaussian regression log likelihood."""
        with _module_mode(self.module, train=False):
            return self._objective(x, y)

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
        with _module_mode(self.module, train=True):
            return optimize_torch_objective(
                self.parameters(),
                lambda: self._objective(x, y),
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
        if not isinstance(module, torch.nn.Module):
            raise TypeError("module must be a torch.nn.Module")
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
        if len(xx.shape) != 2 or xx.shape[0] == 0 or xx.shape[1] == 0 or not bool(self.torch.isfinite(xx).all()):
            raise ValueError("classification x must be a non-empty finite matrix")
        return xx

    def _labels(self, y: Any, rows: int, classes: int) -> Any:
        raw = y.detach().cpu().numpy() if isinstance(y, self.torch.Tensor) else np.asarray(y)
        if raw.ndim != 1 or raw.shape != (rows,) or raw.dtype.kind not in {"i", "u"}:
            raise ValueError(f"classification labels must be {rows} exact integer class indices")
        if np.any(raw < 0) or np.any(raw >= classes):
            raise ValueError(f"classification labels must be in [0, {classes})")
        return self.engine.asarray(raw.astype(np.int64, copy=False), dtype=self.torch.long)

    def logits_tensor(self, x: Any) -> Any:
        """Return raw class logits for ``x`` as a Torch tensor."""
        xx = self._x(x)
        with _module_mode(self.module, train=False), self.torch.no_grad():
            return _module_output(self.module, xx, xx.shape[0], "classification")

    def _objective(self, x: Any, y: Any) -> Any:
        xx = self._x(x)
        logits = _module_output(self.module, xx, xx.shape[0], "classification")
        labels = self._labels(y, logits.shape[0], logits.shape[1])
        result = -self.torch.nn.functional.cross_entropy(logits, labels, reduction="sum")
        if not bool(self.torch.isfinite(result)):
            raise ValueError("classification log likelihood became non-finite")
        return result

    def log_likelihood(self, x: Any, y: Any) -> Any:
        """Return the summed categorical log likelihood for integer labels."""
        with _module_mode(self.module, train=False):
            return self._objective(x, y)

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
        with _module_mode(self.module, train=True):
            return optimize_torch_objective(
                self.parameters(),
                lambda: self._objective(x, y),
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
        probabilities = self.torch.softmax(logits, dim=1)
        if not bool(self.torch.isfinite(probabilities).all()):
            raise ValueError("classification probabilities became non-finite")
        return probabilities

    def predict_proba(self, x: Any) -> np.ndarray:
        """Return class probabilities for ``x`` as a NumPy array."""
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
        if not isinstance(module, torch.nn.Module):
            raise TypeError("module must be a torch.nn.Module")
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
        if len(xx.shape) != 2 or xx.shape[0] == 0 or xx.shape[1] == 0 or not bool(self.torch.isfinite(xx).all()):
            raise ValueError("Poisson regression x must be a non-empty finite matrix")
        return xx

    def _counts_like(self, y: Any, log_rate: Any) -> Any:
        counts = self.engine.asarray(y)
        if len(counts.shape) == 1 and len(log_rate.shape) == 2 and log_rate.shape[1] == 1:
            counts = counts[:, None]
        if tuple(counts.shape) != tuple(log_rate.shape):
            raise ValueError("Poisson counts must match the module log-rate shape.")
        if not bool(self.torch.isfinite(counts).all()):
            raise ValueError("Poisson counts must be finite")
        if bool(self.torch.any(counts < 0).detach().cpu().item()) or not bool(
            self.torch.all(counts == self.torch.round(counts))
        ):
            raise ValueError("Poisson counts must be non-negative integers.")
        return counts

    def log_rate_tensor(self, x: Any) -> Any:
        """Return predicted log rates as a Torch tensor."""
        xx = self._x(x)
        with _module_mode(self.module, train=False), self.torch.no_grad():
            return _module_output(self.module, xx, xx.shape[0], "Poisson regression")

    def _objective(self, x: Any, y: Any) -> Any:
        torch = self.torch
        xx = self._x(x)
        log_rate = _module_output(self.module, xx, xx.shape[0], "Poisson regression")
        counts = self._counts_like(y, log_rate)
        result = torch.sum(counts * log_rate - torch.exp(log_rate) - torch.lgamma(counts + 1.0))
        if not bool(torch.isfinite(result)):
            raise ValueError("Poisson log likelihood became non-finite")
        return result

    def log_likelihood(self, x: Any, y: Any) -> Any:
        """Return the summed Poisson count log likelihood."""
        with _module_mode(self.module, train=False):
            return self._objective(x, y)

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
        with _module_mode(self.module, train=True):
            return optimize_torch_objective(
                self.parameters(),
                lambda: self._objective(x, y),
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
        rates = self.torch.exp(self.log_rate_tensor(x))
        if not bool(self.torch.isfinite(rates).all()):
            raise ValueError("Poisson prediction produced a non-finite rate")
        return rates

    def predict_rate(self, x: Any) -> np.ndarray:
        """Return predicted Poisson rates as a NumPy array."""
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
    input_dim = _positive_int(input_dim, "input_dim")
    output_dim = _positive_int(output_dim, "output_dim")
    if not isinstance(hidden_dims, Sequence):
        raise TypeError("hidden_dims must be a sequence of positive integers")
    hidden_dims = [_positive_int(value, f"hidden_dims[{index}]") for index, value in enumerate(hidden_dims)]
    dims = [input_dim] + hidden_dims + [output_dim]
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
    input_dim = _positive_int(input_dim, "input_dim")
    output_dim = _positive_int(output_dim, "output_dim")
    if not isinstance(hidden_dims, Sequence):
        raise TypeError("hidden_dims must be a sequence of positive integers")
    hidden_dims = [_positive_int(value, f"hidden_dims[{index}]") for index, value in enumerate(hidden_dims)]
    if not isinstance(increasing, (bool, np.bool_)):
        raise TypeError("increasing must be a boolean")
    dims = [input_dim] + hidden_dims + [output_dim]
    layers: list[Any] = []
    for i in range(len(dims) - 1):
        layers.append(NonNegativeLinear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(torch.nn.Softplus())
    module = torch.nn.Sequential(*layers)
    return module if increasing else NegateOutput(module)


def make_deep_set(
    element_dim: int,
    phi_hidden: Sequence[int],
    latent_dim: int,
    rho_hidden: Sequence[int],
    output_dim: int = 1,
    *,
    pooling: str = "mean",
) -> Any:
    """A Deep Sets network (Zaheer et al. 2017): invariant to any permutation of the set axis, by construction.

    Input shape ``(..., set_size, element_dim)``: a per-element MLP ``phi`` (shared weights, applied
    identically to every element -- ``torch.nn.Linear`` already broadcasts over all leading dims, so
    reusing :func:`make_mlp` for ``phi`` gives exactly that) maps each element to a ``latent_dim`` code;
    a permutation-invariant pool (``pooling="mean"``/``"sum"``/``"max"``, taken over the set axis)
    aggregates the codes into one order-independent summary; a second MLP ``rho`` maps the summary to the
    output. Because ``phi`` is applied identically per element and the pool is a symmetric function, the
    output is exactly unchanged by any permutation of the set axis -- true for any weights, trained or not,
    unlike e.g. training on many random orderings and hoping the network learns invariance.

    The returned module is a plain ``torch.nn.Module``, trainable with any ordinary Torch optimizer loop
    over ``(set_size, element_dim)``-shaped inputs. Note: :class:`~mixle.models.neural_leaf.NeuralGaussian`'s
    accumulator flattens each observation to a 1-D feature vector (``reshape(n, -1)``) before the M-step,
    which destroys the set axis this module needs -- so it is not a drop-in wrapper for set-shaped data as
    :func:`make_mlp`/:func:`make_monotonic_mlp` are for flat feature vectors. Use this module directly with
    a custom training loop (or through a wrapper that preserves the set axis) for a fixed set size.
    """
    if _TORCH is None:  # pragma: no cover
        raise ImportError("make_deep_set requires torch.")
    if pooling not in ("mean", "sum", "max"):
        raise ValueError('pooling must be one of "mean", "sum", "max"; got %r' % (pooling,))
    element_dim = _positive_int(element_dim, "element_dim")
    latent_dim = _positive_int(latent_dim, "latent_dim")
    output_dim = _positive_int(output_dim, "output_dim")
    phi = make_mlp(element_dim, phi_hidden, latent_dim, activation="relu")
    rho = make_mlp(latent_dim, rho_hidden, output_dim, activation="relu")

    return DeepSetNetwork(phi, rho, pooling, element_dim)


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
