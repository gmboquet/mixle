"""``EnergyModel`` -- an energy-based density ``p(x) ∝ exp(-E(x))`` as a composable Mixle leaf.

The one neural density whose normalizer is *intractable*: ``p(x) = exp(-E(x)) / Z`` with ``Z = ∫ exp(-E(x)) dx``
unavailable in closed form. So unlike the flows (exact) it is trained and scored **approximately**, and this is
part of the model contract.

* **Training** is Noise-Contrastive Estimation (Gutmann & Hyvärinen 2010), not maximum likelihood: the model
  learns to tell data from samples of a known noise distribution, and in doing so learns a scalar log-normalizer
  ``c`` alongside the energy net. NCE is *consistent* -- as data grow, ``c -> log Z`` and ``-E(x) + c -> log p(x)``
  -- so ``log_density(x) = -E(x) + c`` is an **approximately normalized** log-density, usable directly (no
  per-evaluation partition estimate). It composes in a mixture, but because it is only approximately normalized
  it can bias mixture weights against an exact leaf.
* **Sampling** is unnormalized-density MCMC: a few steps of Langevin dynamics ``x <- x - s ∇E(x) + sqrt(2s) ε``.

Its value over the flows is the inductive bias: an energy net imposes no ordering and no invertibility -- it scores
*compatibility*, so it captures undirected/symmetric structure a coupling or autoregressive flow parameterizes
awkwardly. :func:`build_energy_net` is a ready MLP energy to wrap.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mixle.models._neural_serial import check_finite, decode_module, encode_module
from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)


def _torch() -> Any:
    import torch

    return torch


class EnergyModel(SequenceEncodableProbabilityDistribution):
    """``log p(x) ≈ -E(x) + c`` for an energy module (``module.energy(x) -> (n,)`` and a learned scalar ``log_norm``).

    Approximately normalized (trained by NCE); ``log_density`` returns ``-E(x) + c``. Composes like any leaf.
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
        self.module = module
        self.m_steps = int(m_steps)
        self.lr = float(lr)
        self.noise_ratio = int(noise_ratio)
        self.langevin_steps = int(langevin_steps)
        self.langevin_step = float(langevin_step)
        self.device = device
        self.name = name

    def __str__(self) -> str:
        return f"EnergyModel({type(self.module).__name__})"

    def log_density(self, x: Any) -> float:
        """Return the approximate normalized log density for one observation."""
        return float(self.seq_log_density(np.atleast_2d(np.asarray(x, dtype=float)))[0])

    def seq_log_density(self, x: Any) -> np.ndarray:
        """Return approximate normalized log densities for a batch of observations."""
        torch = _torch()
        xx = check_finite(np.atleast_2d(np.asarray(x, dtype=float)), "EnergyModel.seq_log_density")
        self.module.to(self.device).eval()
        xt = torch.as_tensor(xx, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return (-self.module.energy(xt) + self.module.log_norm).cpu().numpy().reshape(-1)

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
        torch = _torch()
        n = int(size or 1)
        self.dist.module.to(self.dist.device).eval()
        dim = int(self.dist.module.dim)
        x = torch.as_tensor(self.rng.randn(n, dim), dtype=torch.float32, device=self.dist.device)
        s = self.dist.langevin_step
        for _ in range(self.dist.langevin_steps):
            x = x.detach().requires_grad_(True)
            grad = torch.autograd.grad(self.dist.module.energy(x).sum(), x)[0]
            noise = torch.as_tensor(self.rng.randn(n, dim), dtype=torch.float32, device=self.dist.device)
            x = x - s * grad + float(np.sqrt(2.0 * s)) * noise
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
        self.x.append(np.atleast_1d(np.asarray(x, dtype=float))[None, ...])
        self.w.append(np.asarray([float(weight)], dtype=float))

    def seq_update(self, enc: Any, weights: np.ndarray, estimate: Any) -> None:
        """Add an encoded batch and responsibility weights to the accumulator."""
        xb = np.asarray(enc, dtype=float)
        self.x.append(xb.reshape(xb.shape[0], 1) if xb.ndim == 1 else xb)
        self.w.append(np.asarray(weights, dtype=float).ravel())

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
            self.x.append(np.asarray(xs, dtype=float))
            self.w.append(np.asarray(ws, dtype=float).ravel())
        return self

    def value(self) -> tuple:
        """Return contiguous ``(x, weights)`` arrays for the NCE M-step."""
        x = np.concatenate(self.x, axis=0) if self.x else np.zeros((0, 0))
        w = np.concatenate(self.w) if self.w else np.zeros((0,))
        return (x, w)

    def from_value(self, v: tuple) -> EnergyModelAccumulator:
        """Restore accumulator buffers from a value tuple."""
        x, w = v
        self.x = [np.asarray(x, dtype=float)] if len(x) else []
        self.w = [np.asarray(w, dtype=float).ravel()] if len(w) else []
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

    Learns the energy net *and* the scalar log-normalizer ``log_norm`` by logistic discrimination of data from
    noise -- so the resulting ``-E(x) + log_norm`` is a consistent, approximately-normalized log-density.
    """

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
        self.module = module
        self.m_steps = int(m_steps)
        self.lr = float(lr)
        self.noise_ratio = int(noise_ratio)
        self.langevin_steps = int(langevin_steps)
        self.langevin_step = float(langevin_step)
        self.device = device
        self.name = name

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
        torch = _torch()
        xs, ws = suff_stat
        if len(xs) == 0:
            return self._make()
        x = torch.as_tensor(np.asarray(xs, dtype=float), dtype=torch.float32, device=self.device)
        w = torch.as_tensor(np.asarray(ws, dtype=float), dtype=torch.float32, device=self.device)
        w = w / w.sum().clamp(min=1e-8)

        # noise distribution p_n = N(mu, diag var), matched to the weighted-data moments (a good NCE proposal)
        mu = (w[:, None] * x).sum(0)
        var = (w[:, None] * (x - mu) ** 2).sum(0) + 1e-3
        d = x.shape[1]
        log_nu = float(np.log(max(self.noise_ratio, 1)))
        const = -0.5 * float(d) * float(np.log(2.0 * np.pi)) - 0.5 * torch.log(var).sum()

        def log_pn(z: Any) -> Any:
            return const - 0.5 * (((z - mu) ** 2) / var).sum(1)

        def log_pm(z: Any) -> Any:
            return -self.module.energy(z) + self.module.log_norm

        self.module.to(self.device).train()
        opt = torch.optim.Adam(self.module.parameters(), lr=self.lr)
        m = int(self.noise_ratio) * x.shape[0]
        for _ in range(self.m_steps):
            opt.zero_grad()
            y = mu + torch.sqrt(var) * torch.randn(m, d, device=self.device)  # noise draws
            # posterior-that-it-is-data on each side; data weighted by responsibilities, noise averaged
            # population NCE (as empirical expectations): E_pd[log sig(G - log nu)] + nu * E_pn[log sig(log nu - G)].
            # The data term is the responsibility-weighted mean (w sums to 1); the noise term carries the nu factor,
            # so the learned log_norm converges to log Z independently of the noise ratio.
            loss_data = -(w * torch.nn.functional.logsigmoid(log_pm(x) - log_pn(x) - log_nu)).sum()
            loss_noise = (
                -float(self.noise_ratio) * torch.nn.functional.logsigmoid(log_pn(y) + log_nu - log_pm(y)).mean()
            )
            (loss_data + loss_noise).backward()
            opt.step()
        return self._make()


# --- a ready energy module to wrap: an MLP energy E(x) with a learned scalar log-normalizer -------------------
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
            self.log_norm = nn.Parameter(torch.zeros(()))  # the NCE-learned scalar log-normalizer

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

    class _ICNNLayer(nn.Module):
        def __init__(self, z_dim: int | None, x_dim: int, out_dim: int) -> None:
            super().__init__()
            self.x_path = nn.Linear(x_dim, out_dim)
            self.raw_z_weight = nn.Parameter(torch.randn(out_dim, z_dim) * 0.1) if z_dim is not None else None

        def forward(self, z: Any, x: Any) -> Any:
            out = self.x_path(x)
            if self.raw_z_weight is not None:
                out = out + torch.nn.functional.linear(z, torch.nn.functional.softplus(self.raw_z_weight))
            return out

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
                self.icnn_layers.append(_ICNNLayer(prev_dim, self.dim, out_dim))
                prev_dim = out_dim
            self.log_norm = nn.Parameter(torch.zeros(()))  # the NCE-learned scalar log-normalizer

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
            self.log_norm = nn.Parameter(torch.zeros(()))  # the product's own NCE-learned normalizer

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
    """An MLP energy ``E(x): R^dim -> R`` (plus a learned scalar ``log_norm``) -- ready to wrap in an EnergyModel.

    Lower energy = higher (unnormalized) density. ``log_norm`` is the NCE-learned normalizer, so the paired
    :class:`EnergyModel` scores ``-E(x) + log_norm``. Swap in any module exposing ``energy(x) -> (n,)``, a
    ``log_norm`` parameter and a ``dim`` attribute.
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
    except Exception:  # pragma: no cover
        return
    register_serializable_class(EnergyModel)


_register_serializable()
