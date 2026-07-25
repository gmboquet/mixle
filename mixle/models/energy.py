"""``EnergyModel`` -- an energy-based density ``p(x) ∝ exp(-E(x))`` as a composable Mixle leaf.

The one neural density whose normalizer is *intractable*: ``p(x) = exp(-E(x)) / Z`` with ``Z = ∫ exp(-E(x)) dx``
unavailable in closed form. So unlike the flows (exact) it is trained and scored **approximately**, and this is
part of the model contract.

* **Training** is Noise-Contrastive Estimation (Gutmann & Hyvärinen 2010), not maximum likelihood: the model
  learns to tell data from samples of a known noise distribution, and in doing so learns a scalar additive
  normalization offset ``c`` alongside the energy net. NCE is *consistent* -- as data grow,
  ``c -> -log Z`` and ``-E(x) + c -> log p(x)`` -- so ``log_density(x) = -E(x) + c`` is an
  **approximately normalized** log-density, usable directly (no per-evaluation partition estimate). It
  composes in a mixture, but because it is only approximately normalized it can bias mixture weights
  against an exact leaf.
* **Sampling** is unnormalized-density MCMC: a few steps of Langevin dynamics ``x <- x - s ∇E(x) + sqrt(2s) ε``.

Its value over the flows is the inductive bias: an energy net imposes no ordering and no invertibility -- it scores
*compatibility*, so it captures undirected/symmetric structure a coupling or autoregressive flow parameterizes
awkwardly. :func:`build_energy_net` is a ready MLP energy to wrap.
"""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any

import numpy as np

from mixle.models._neural_serial import check_finite, decode_module, encode_module
from mixle.models.grad_leaf import _module_mode
from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)

try:
    import torch as _TORCH
except ImportError:  # pragma: no cover - Torch remains optional
    _TORCH = None

_ModuleBase = object if _TORCH is None else _TORCH.nn.Module


def _torch() -> Any:
    import torch

    return torch


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _positive_finite(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a positive finite real number")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite real number")
    return result


def _validate_module(module: Any) -> tuple[Any, int]:
    torch = _torch()
    if not isinstance(module, torch.nn.Module):
        raise TypeError("module must be a torch.nn.Module")
    if not callable(getattr(module, "energy", None)):
        raise TypeError("module must define energy(x)")
    if not hasattr(module, "log_norm"):
        raise TypeError("module must expose a scalar log_norm parameter")
    dim = _positive_int(getattr(module, "dim", None), "module.dim")
    return torch, dim


def _finite_matrix(x: Any, where: str, *, dim: int | None = None) -> np.ndarray:
    array = np.asarray(x)
    if np.iscomplexobj(array):
        raise TypeError(f"{where} must contain real values")
    try:
        array = np.asarray(array, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{where} must be a real numeric matrix") from exc
    array = np.atleast_2d(array)
    if array.ndim != 2 or not all(array.shape):
        raise ValueError(f"{where} must be a non-empty two-dimensional matrix")
    if dim is not None and array.shape[1] != dim:
        raise ValueError(f"{where} must have exactly {dim} features; got shape {array.shape}")
    return check_finite(array, where)


def _finite_nonnegative_weights(weights: Any, rows: int, where: str) -> np.ndarray:
    array = np.asarray(weights)
    if np.iscomplexobj(array):
        raise TypeError(f"{where} must contain real values")
    try:
        array = np.asarray(array, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{where} must be a real numeric vector") from exc
    if array.shape != (rows,):
        raise ValueError(f"{where} must have exact shape ({rows},); got {array.shape}")
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError(f"{where} must contain finite non-negative values")
    return array


def _energy_values(module: Any, x: Any, where: str) -> Any:
    torch = _torch()
    values = module.energy(x)
    if not isinstance(values, torch.Tensor) or tuple(values.shape) != (x.shape[0],):
        raise ValueError(f"{where} energy output must have exact shape ({x.shape[0]},)")
    log_norm = module.log_norm
    if not isinstance(log_norm, torch.Tensor) or log_norm.numel() != 1:
        raise ValueError(f"{where} log_norm must be a scalar tensor")
    if not bool(torch.isfinite(values).all()) or not bool(torch.isfinite(log_norm).all()):
        raise ValueError(f"{where} energy and log_norm must be finite")
    return values


class ICNNLayer(_ModuleBase):
    """Pickle-stable input-convex layer with a non-negative hidden-state path."""

    def __init__(self, z_dim: int | None, x_dim: int, out_dim: int) -> None:
        if _TORCH is None:  # pragma: no cover
            raise ImportError("ICNNLayer requires torch")
        super().__init__()
        x_dim = _positive_int(x_dim, "x_dim")
        out_dim = _positive_int(out_dim, "out_dim")
        if z_dim is not None:
            z_dim = _positive_int(z_dim, "z_dim")
        self.x_path = _TORCH.nn.Linear(x_dim, out_dim)
        self.raw_z_weight = _TORCH.nn.Parameter(_TORCH.randn(out_dim, z_dim) * 0.1) if z_dim is not None else None

    def forward(self, z: Any, x: Any) -> Any:
        out = self.x_path(x)
        if self.raw_z_weight is not None:
            out = out + _TORCH.nn.functional.linear(z, _TORCH.nn.functional.softplus(self.raw_z_weight))
        return out


class EnergyModel(SequenceEncodableProbabilityDistribution):
    """``log p(x) ≈ -E(x) + c`` for an energy module.

    The module exposes ``energy(x) -> (n,)`` and a learned scalar ``log_norm`` whose convention is
    ``c=-log Z``. Approximately normalized (trained by NCE); ``log_density`` returns ``-E(x) + c``.
    Composes like any leaf.
    """

    __pysp_serializable__ = True  # module persisted as bytes (see __pysp_getstate__); leaf round-trips in a mixture

    def __init__(
        self,
        module: Any,
        *,
        m_steps: int = 200,
        lr: float = 5e-3,
        noise_ratio: int = 1,
        langevin_steps: int = 40,
        langevin_step: float = 0.05,
        device: str = "cpu",
        name: str | None = None,
    ) -> None:
        _validate_module(module)
        self.module = module
        self.m_steps = _positive_int(m_steps, "m_steps")
        self.lr = _positive_finite(lr, "lr")
        self.noise_ratio = _positive_int(noise_ratio, "noise_ratio")
        self.langevin_steps = _positive_int(langevin_steps, "langevin_steps")
        self.langevin_step = _positive_finite(langevin_step, "langevin_step")
        self.device = device
        self.name = name

    def __str__(self) -> str:
        return f"EnergyModel({type(self.module).__name__})"

    def log_density(self, x: Any) -> float:
        """Return the approximate normalized log density for one observation."""
        values = self.seq_log_density(x)
        if values.shape != (1,):
            raise ValueError(f"EnergyModel.log_density expects one observation; got {values.shape[0]}")
        return float(values[0])

    def seq_log_density(self, x: Any) -> np.ndarray:
        """Return approximate normalized log densities for a batch of observations."""
        torch, dim = _validate_module(self.module)
        xx = _finite_matrix(x, "EnergyModel.seq_log_density", dim=dim)
        self.module.to(self.device)
        xt = torch.as_tensor(xx, dtype=torch.float32, device=self.device)
        with _module_mode(self.module, train=False), torch.no_grad():
            result = -_energy_values(self.module, xt, "EnergyModel.seq_log_density") + self.module.log_norm
            if not bool(torch.isfinite(result).all()):
                raise ValueError("EnergyModel.seq_log_density produced non-finite scores")
            return result.cpu().numpy()

    def sampler(self, seed: int | None = None) -> EnergyModelSampler:
        """Return a Langevin sampler for the learned energy model."""
        return EnergyModelSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> EnergyModelEstimator:
        """Return the NCE estimator used as the model's M-step."""
        return EnergyModelEstimator(
            self.module,
            m_steps=self.m_steps,
            lr=self.lr,
            noise_ratio=self.noise_ratio,
            langevin_steps=self.langevin_steps,
            langevin_step=self.langevin_step,
            device=self.device,
            name=self.name,
        )

    def dist_to_encoder(self) -> EnergyModelEncoder:
        """Return the encoder for vectorized energy-model scoring."""
        return EnergyModelEncoder()

    # Persist hparams and module bytes so a mixture holding this leaf can round-trip through serializers.
    def __pysp_getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["module"] = encode_module(self.module)
        return state

    def __pysp_setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.module = decode_module(state["module"])
        _validate_module(self.module)

    def to_dict(self) -> dict[str, Any]:
        """Serialize hyperparameters and module bytes for registry-based round trips."""
        return {
            "m_steps": self.m_steps,
            "lr": self.lr,
            "noise_ratio": self.noise_ratio,
            "langevin_steps": self.langevin_steps,
            "langevin_step": self.langevin_step,
            "device": self.device,
            "name": self.name,
            "module": encode_module(self.module),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EnergyModel:
        """Rebuild an :class:`EnergyModel` from :meth:`to_dict` output."""
        return cls(
            decode_module(payload["module"]),
            m_steps=payload["m_steps"],
            lr=payload["lr"],
            noise_ratio=payload["noise_ratio"],
            langevin_steps=payload["langevin_steps"],
            langevin_step=payload["langevin_step"],
            device=payload["device"],
            name=payload["name"],
        )


class EnergyModelSampler(DistributionSampler):
    """Langevin dynamics on the (unnormalized) energy: ``x <- x - s ∇E(x) + sqrt(2 s) ε``."""

    def __init__(self, dist: EnergyModel, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = np.random.RandomState(seed)

    def sample(self, size: int | None = None, *, batched: bool = True) -> Any:
        """Draw approximate samples with unadjusted Langevin dynamics."""
        torch, dim = _validate_module(self.dist.module)
        n = 1 if size is None else _positive_int(size, "size")
        if not isinstance(batched, (bool, np.bool_)):
            raise TypeError("batched must be a boolean")
        self.dist.module.to(self.dist.device)
        x = torch.as_tensor(self.rng.randn(n, dim), dtype=torch.float32, device=self.dist.device)
        s = self.dist.langevin_step
        with _module_mode(self.dist.module, train=False):
            for _ in range(self.dist.langevin_steps):
                x = x.detach().requires_grad_(True)
                energy = _energy_values(self.dist.module, x, "EnergyModelSampler")
                grad = torch.autograd.grad(energy.sum(), x)[0]
                if not bool(torch.isfinite(grad).all()):
                    raise ValueError("EnergyModelSampler produced non-finite energy gradients")
                noise = torch.as_tensor(self.rng.randn(n, dim), dtype=torch.float32, device=self.dist.device)
                x = x - s * grad + float(np.sqrt(2.0 * s)) * noise
                if not bool(torch.isfinite(x).all()):
                    raise ValueError("EnergyModelSampler produced non-finite samples")
        out = x.detach().cpu().numpy()
        return out if (size is not None) else out[0]


class EnergyModelEncoder(DataSequenceEncoder):
    """Encode observations for vectorized energy-model scoring and fitting."""

    def __str__(self) -> str:
        return "EnergyModelEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, EnergyModelEncoder)

    def seq_encode(self, data: list) -> np.ndarray:
        """Convert observations to a two-dimensional float array."""
        return np.array([np.atleast_1d(np.asarray(x, dtype=float)) for x in data])


class EnergyModelAccumulator(SequenceEncodableStatisticAccumulator):
    """Buffers responsibility-weighted data for the NCE M-step (weights = the E-step soft counts)."""

    def __init__(self) -> None:
        self.x: list = []
        self.w: list = []

    # Contiguous batch arrays concatenated once at value() (shape-preserving) rather than one ndarray per row.
    def update(self, x: Any, weight: float, estimate: Any) -> None:
        """Add one weighted observation to the NCE accumulator."""
        self.x.append(_finite_matrix(x, "EnergyModelAccumulator observation"))
        self.w.append(_finite_nonnegative_weights([weight], 1, "EnergyModelAccumulator weights"))

    def seq_update(self, enc: Any, weights: np.ndarray, estimate: Any) -> None:
        """Add an encoded batch and responsibility weights to the accumulator."""
        xb = _finite_matrix(enc, "EnergyModelAccumulator observations")
        self.x.append(xb)
        self.w.append(_finite_nonnegative_weights(weights, xb.shape[0], "EnergyModelAccumulator weights"))

    def initialize(self, x: Any, weight: float, rng: Any) -> None:
        """Initialize from one observation using the ordinary update path."""
        self.update(x, weight, None)

    def seq_initialize(self, enc: Any, weights: np.ndarray, rng: Any) -> None:
        """Initialize from an encoded batch using the ordinary batch update path."""
        self.seq_update(enc, weights, None)

    def combine(self, other: Any) -> EnergyModelAccumulator:
        """Merge the value tuple from another energy-model accumulator."""
        xs, ws = other
        if len(xs):
            xb = _finite_matrix(xs, "EnergyModelAccumulator combined observations")
            self.x.append(xb)
            self.w.append(_finite_nonnegative_weights(ws, xb.shape[0], "EnergyModelAccumulator combined weights"))
        return self

    def value(self) -> tuple:
        """Return contiguous ``(x, weights)`` arrays for the NCE M-step."""
        x = np.concatenate(self.x, axis=0) if self.x else np.zeros((0, 0))
        w = np.concatenate(self.w) if self.w else np.zeros((0,))
        return (x, w)

    def from_value(self, v: tuple) -> EnergyModelAccumulator:
        """Restore accumulator buffers from a value tuple."""
        x, w = v
        if len(x):
            xb = _finite_matrix(x, "EnergyModelAccumulator restored observations")
            self.x = [xb]
            self.w = [_finite_nonnegative_weights(w, xb.shape[0], "EnergyModelAccumulator restored weights")]
        else:
            if len(w):
                raise ValueError("empty restored observations require empty weights")
            self.x = []
            self.w = []
        return self

    def acc_to_encoder(self) -> EnergyModelEncoder:
        """Return the encoder expected by this accumulator."""
        return EnergyModelEncoder()


class EnergyModelAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for energy-model accumulators."""

    def make(self) -> EnergyModelAccumulator:
        """Create a fresh accumulator."""
        return EnergyModelAccumulator()


class EnergyModelEstimator(ParameterEstimator):
    """M-step: Noise-Contrastive Estimation against a Gaussian noise fit to the (weighted) data.

    Learns the energy net *and* scalar normalization offset ``log_norm = -log Z`` by logistic
    discrimination of data from noise, so ``-E(x) + log_norm`` is a consistent,
    approximately-normalized log-density.
    """

    # Before NCE learns log_norm, observed ``seq_log_density`` is not a comparable outer objective:
    # an unnormalized initial model can score arbitrarily well merely by shifting that constant.
    outer_objective_compatible = False

    def __init__(
        self,
        module: Any,
        *,
        m_steps: int = 200,
        lr: float = 5e-3,
        noise_ratio: int = 1,
        langevin_steps: int = 40,
        langevin_step: float = 0.05,
        device: str = "cpu",
        name: str | None = None,
    ) -> None:
        _validate_module(module)
        self.module = module
        self.m_steps = _positive_int(m_steps, "m_steps")
        self.lr = _positive_finite(lr, "lr")
        self.noise_ratio = _positive_int(noise_ratio, "noise_ratio")
        self.langevin_steps = _positive_int(langevin_steps, "langevin_steps")
        self.langevin_step = _positive_finite(langevin_step, "langevin_step")
        self.device = device
        self.name = name
        self._optimizer: Any | None = None

    def accumulator_factory(self) -> EnergyModelAccumulatorFactory:
        """Return an accumulator factory for weighted NCE batches."""
        return EnergyModelAccumulatorFactory()

    def _make(self) -> EnergyModel:
        return EnergyModel(
            self.module,
            m_steps=self.m_steps,
            lr=self.lr,
            noise_ratio=self.noise_ratio,
            langevin_steps=self.langevin_steps,
            langevin_step=self.langevin_step,
            device=self.device,
            name=self.name,
        )

    def estimate(self, nobs: float | None, suff_stat: tuple) -> EnergyModel:
        """Run the weighted NCE M-step and return the updated energy leaf."""
        torch, dim = _validate_module(self.module)
        xs, ws = suff_stat
        if len(xs) == 0:
            if len(ws):
                raise ValueError("empty energy observations require empty weights")
            return self._make()
        x_array = _finite_matrix(xs, "EnergyModelEstimator observations", dim=dim)
        weight_array = _finite_nonnegative_weights(
            ws,
            x_array.shape[0],
            "EnergyModelEstimator weights",
        )
        weight_sum = float(weight_array.sum())
        if not np.isfinite(weight_sum) or weight_sum <= 0.0:
            raise ValueError("EnergyModelEstimator weights must have positive finite total mass")
        x = torch.as_tensor(x_array, dtype=torch.float32, device=self.device)
        w = torch.as_tensor(weight_array / weight_sum, dtype=torch.float32, device=self.device)

        # noise distribution p_n = N(mu, diag var), matched to the weighted-data moments (a good NCE proposal)
        mu = (w[:, None] * x).sum(0)
        var = (w[:, None] * (x - mu) ** 2).sum(0) + 1e-3
        if not bool(torch.isfinite(mu).all()) or not bool(torch.isfinite(var).all()) or bool(torch.any(var <= 0.0)):
            raise ValueError("EnergyModelEstimator weighted moments must be finite with positive variance")
        d = dim
        log_nu = float(np.log(self.noise_ratio))
        const = -0.5 * float(d) * float(np.log(2.0 * np.pi)) - 0.5 * torch.log(var).sum()

        def log_pn(z: Any) -> Any:
            return const - 0.5 * (((z - mu) ** 2) / var).sum(1)

        def log_pm(z: Any) -> Any:
            return -_energy_values(self.module, z, "EnergyModelEstimator") + self.module.log_norm

        self.module.to(self.device)
        if self._optimizer is None:
            self._optimizer = torch.optim.Adam(self.module.parameters(), lr=self.lr)
        m = self.noise_ratio * x.shape[0]
        with _module_mode(self.module, train=True):
            for _ in range(self.m_steps):
                self._optimizer.zero_grad()
                y = mu + torch.sqrt(var) * torch.randn(m, d, device=self.device)  # noise draws
                # Population NCE: the data term is responsibility weighted (w sums to one), while the
                # noise expectation carries nu. The learned additive offset therefore approaches -log Z.
                loss_data = -(w * torch.nn.functional.logsigmoid(log_pm(x) - log_pn(x) - log_nu)).sum()
                loss_noise = (
                    -float(self.noise_ratio) * torch.nn.functional.logsigmoid(log_pn(y) + log_nu - log_pm(y)).mean()
                )
                loss = loss_data + loss_noise
                if not bool(torch.isfinite(loss)):
                    raise ValueError("EnergyModelEstimator NCE loss became non-finite")
                loss.backward()
                if any(
                    parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
                    for parameter in self.module.parameters()
                ):
                    raise ValueError("EnergyModelEstimator NCE gradients became non-finite")
                self._optimizer.step()
                if any(not bool(torch.isfinite(parameter).all()) for parameter in self.module.parameters()):
                    raise ValueError("EnergyModelEstimator parameters became non-finite")
        return self._make()


# --- a ready energy module to wrap: an MLP energy E(x) with a learned scalar normalization offset -----------
#
# EnergyNet is reachable at MODULE level (built on first use, resolved by name via __getattr__) so a wrapped leaf
# -- and any mixture holding one -- pickles for distributed EM.

_ENERGY_NET_CLASS: list[Any] = []


def _energy_net_class() -> Any:
    if _ENERGY_NET_CLASS:
        return _ENERGY_NET_CLASS[0]
    import torch
    import torch.nn as nn

    class EnergyNet(nn.Module):
        def __init__(self, dim: int, hidden: int = 64, layers: int = 3) -> None:
            super().__init__()
            self.dim = int(dim)
            self.hidden = int(hidden)
            self.layers = int(layers)
            body: list[nn.Module] = []
            prev = self.dim
            for _ in range(self.layers - 1):
                body += [nn.Linear(prev, self.hidden), nn.Softplus()]  # smooth => Langevin gradients well-behaved
                prev = self.hidden
            body += [nn.Linear(prev, 1)]
            self.net = nn.Sequential(*body)
            self.log_norm = nn.Parameter(torch.zeros(()))  # NCE learns the additive offset -log Z

        def energy(self, x: Any) -> Any:
            return self.net(x).squeeze(-1)

    EnergyNet.__module__ = __name__
    EnergyNet.__qualname__ = "EnergyNet"
    EnergyNet.__name__ = "EnergyNet"
    _ENERGY_NET_CLASS.append(EnergyNet)
    return EnergyNet


_CONVEX_ENERGY_NET_CLASS: list[Any] = []


def _convex_energy_net_class() -> Any:
    """An input-convex energy net (ICNN, Amos et al. 2017): ``E(x)`` is convex in ``x`` by construction.

    Each hidden layer takes the previous layer's activation ``z`` through a non-negative weight matrix
    (``softplus``-reparameterized, same trick as :func:`~mixle.models.neural.make_monotonic_mlp`) plus an
    unconstrained affine "skip" of the raw input ``x``, then a convex non-decreasing activation
    (``Softplus``). A non-negative-weight combination of convex functions, composed with a convex
    non-decreasing activation, is itself convex, and that property is closed under composition -- so the
    whole energy is provably convex everywhere, not just where training data landed. The unconstrained
    ``x``-skip at every layer is what makes this expressive (a purely non-negative-weight-in-``x`` network
    would be far too restricted); only the ``z``-path weights carry the non-negativity constraint.
    """
    if _CONVEX_ENERGY_NET_CLASS:
        return _CONVEX_ENERGY_NET_CLASS[0]
    import torch
    import torch.nn as nn

    class ConvexEnergyNet(nn.Module):
        def __init__(self, dim: int, hidden: int = 64, layers: int = 3) -> None:
            super().__init__()
            self.dim = int(dim)
            self.hidden = int(hidden)
            self.layers = int(layers)
            out_dims = [self.hidden] * (self.layers - 1) + [1]
            self.icnn_layers = nn.ModuleList()
            prev_dim: int | None = None
            for out_dim in out_dims:
                self.icnn_layers.append(ICNNLayer(prev_dim, self.dim, out_dim))
                prev_dim = out_dim
            self.log_norm = nn.Parameter(torch.zeros(()))  # NCE learns the additive offset -log Z

        def energy(self, x: Any) -> Any:
            z = None
            for i, layer in enumerate(self.icnn_layers):
                z = layer(z, x)
                if i < len(self.icnn_layers) - 1:
                    z = torch.nn.functional.softplus(z)
            return z.squeeze(-1)

    ConvexEnergyNet.__module__ = __name__
    ConvexEnergyNet.__qualname__ = "ConvexEnergyNet"
    ConvexEnergyNet.__name__ = "ConvexEnergyNet"
    _CONVEX_ENERGY_NET_CLASS.append(ConvexEnergyNet)
    return ConvexEnergyNet


_PRODUCT_ENERGY_NET_CLASS: list[Any] = []


def _product_energy_net_class() -> Any:
    """A product-of-experts energy: ``E(x) = sum_k E_k(x)``, so ``p(x) ∝ prod_k exp(-E_k(x)) = prod_k p_k(x)``.

    A mixture (:class:`~mixle.stats.latent.mixture.MixtureDistribution`) *adds* densities -- a disjunction,
    "x looks like expert A or expert B". A product of experts *multiplies* them -- a conjunction, "x is
    plausible under expert A and expert B" -- so each expert acts as a soft constraint and the
    product is their intersection (Hinton, "Training Products of Experts by Minimizing Contrastive
    Divergence", Neural Computation 2002). The normalizer of a product is intractable in general, which is
    exactly the problem the energy stack already solves: sum the expert energies into one energy module and
    fit the shared ``log_norm`` by NCE, sample by Langevin -- all inherited from :class:`EnergyModel` with
    no new machinery. Each expert stays a separately-specified, interpretable factor (``.experts``).
    """
    if _PRODUCT_ENERGY_NET_CLASS:
        return _PRODUCT_ENERGY_NET_CLASS[0]
    import torch
    import torch.nn as nn

    class ProductEnergyNet(nn.Module):
        def __init__(self, experts: Any) -> None:
            super().__init__()
            experts = list(experts)
            if len(experts) < 2:
                raise ValueError("ProductEnergyNet needs at least 2 experts; got %d" % len(experts))
            dims = {int(e.dim) for e in experts}
            if len(dims) != 1:
                raise ValueError("all experts must share one input dim; got %s" % sorted(dims))
            self.experts = nn.ModuleList(experts)
            self.dim = int(next(iter(dims)))
            self.log_norm = nn.Parameter(torch.zeros(()))  # the product's own NCE-learned -log Z offset

        def expert_energies(self, x: Any) -> Any:
            """``(n, K)`` per-expert energies -- the interpretable decomposition of the total energy."""
            import torch as _t

            return _t.stack([e.energy(x) for e in self.experts], dim=-1)

        def energy(self, x: Any) -> Any:
            # sum the experts' energies; their own log_norms are constants that only shift the (separate)
            # product log_norm, so they are harmless here and the product's log_norm absorbs the offset.
            return sum(e.energy(x) for e in self.experts)

    ProductEnergyNet.__module__ = __name__
    ProductEnergyNet.__qualname__ = "ProductEnergyNet"
    ProductEnergyNet.__name__ = "ProductEnergyNet"
    _PRODUCT_ENERGY_NET_CLASS.append(ProductEnergyNet)
    return ProductEnergyNet


def __getattr__(name: str) -> Any:  # PEP 562: lets ``pickle`` resolve the hoisted net classes by name
    if name == "EnergyNet":
        return _energy_net_class()
    if name == "ConvexEnergyNet":
        return _convex_energy_net_class()
    if name == "ProductEnergyNet":
        return _product_energy_net_class()
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


def build_energy_net(dim: int, *, hidden: int = 64, layers: int = 3) -> Any:
    """An MLP energy ``E(x): R^dim -> R`` plus a learned scalar normalization offset.

    Lower energy = higher unnormalized density. ``log_norm`` uses the ``-log Z`` convention, so the paired
    :class:`EnergyModel` scores ``-E(x) + log_norm``. Swap in any module exposing ``energy(x) -> (n,)``,
    a scalar ``log_norm`` parameter, and a ``dim`` attribute.
    """
    return _energy_net_class()(dim, hidden, layers)


def build_convex_energy_net(dim: int, *, hidden: int = 64, layers: int = 3) -> Any:
    """An input-convex MLP energy ``E(x): R^dim -> R``, convex in ``x`` BY CONSTRUCTION -- ready to wrap
    in an :class:`EnergyModel` exactly like :func:`build_energy_net`. A convex energy gives Langevin
    sampling (:class:`EnergyModelSampler`) a unimodal target with no spurious local minima to get stuck
    in, and gives any consumer of the fitted energy a certified-convex scalar-valued potential (e.g. a
    verified optimum for a downstream ``mixle.doe`` search over ``-E(x)``). See :func:`_convex_energy_net_class`
    for the construction.
    """
    return _convex_energy_net_class()(dim, hidden, layers)


def build_product_energy_net(experts: Any) -> Any:
    """Combine expert energy modules multiplicatively: one module with ``energy(x) = sum_k experts[k].energy(x)``.

    A product of experts, ``p(x) ∝ prod_k p_k(x)`` -- a *conjunction* (each expert a soft constraint, the
    product their intersection), as opposed to a mixture's disjunction. This is the ENERGY-BASED,
    arbitrary-density complement to :func:`mixle.ops.product_of_experts`, which pools *tractable* families
    (Categorical, Gaussian) in closed form but deliberately raises on the general continuous case because
    the product normalizer is then intractable. That intractable normalizer is exactly what the energy
    stack already handles: wrap the result here in an :class:`EnergyModel` to fit the shared ``log_norm``
    by NCE and sample by Langevin, no new machinery.

    Each expert must expose ``energy(x) -> (n,)`` and a ``dim`` attribute (e.g. any :func:`build_energy_net` /
    :func:`build_convex_energy_net` module, all sharing one input dim), and stays individually inspectable via
    the built module's ``.experts`` / ``.expert_energies(x)``. Fit it in one line::

        model = EnergyModel(build_product_energy_net([expert_a, expert_b]), m_steps=250)
    """
    return _product_energy_net_class()(experts)


def _register_serializable() -> None:
    # mixle.models classes aren't in the stats/analysis auto-walk, so opt in explicitly for to_json/from_json.
    try:
        from mixle.utils.serialization import register_serializable_class
    except Exception:  # pragma: no cover  # noqa: BLE001
        return
    register_serializable_class(EnergyModel)


_register_serializable()
