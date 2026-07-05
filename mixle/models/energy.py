"""``EnergyModel`` -- an energy-based density ``p(x) ∝ exp(-E(x))`` as a composable mixle leaf.

The one neural density whose normalizer is *intractable*: ``p(x) = exp(-E(x)) / Z`` with ``Z = ∫ exp(-E(x)) dx``
unavailable in closed form. So unlike the flows (exact) it is trained and scored **approximately**, and this is
stated plainly -- it is the energy-model analogue of the VAE's ELBO caveat.

* **Training** is Noise-Contrastive Estimation (Gutmann & Hyvärinen 2010), not maximum likelihood: the model
  learns to tell data from samples of a known noise distribution, and in doing so learns a scalar log-normalizer
  ``c`` alongside the energy net. NCE is *consistent* -- as data grow, ``c -> log Z`` and ``-E(x) + c -> log p(x)``
  -- so ``log_density(x) = -E(x) + c`` is an **approximately normalized** log-density, usable directly (no
  per-evaluation partition estimate). It composes in a mixture, but being only approximately normalized it can
  bias mixture weights against an exact leaf (same honesty caveat as the VAE).
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
        return float(self.seq_log_density(np.atleast_2d(np.asarray(x, dtype=float)))[0])

    def seq_log_density(self, x: Any) -> np.ndarray:
        torch = _torch()
        xx = check_finite(np.atleast_2d(np.asarray(x, dtype=float)), "EnergyModel.seq_log_density")
        self.module.to(self.device).eval()
        xt = torch.as_tensor(xx, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return (-self.module.energy(xt) + self.module.log_norm).cpu().numpy().reshape(-1)

    def sampler(self, seed: int | None = None) -> EnergyModelSampler:
        return EnergyModelSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> EnergyModelEstimator:
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
        return EnergyModelEncoder()

    # --- serialization: persist hparams + the module (as portable bytes); registered below so a mixture holding
    # this leaf round-trips through to_dict/to_json/pickle as well. ---
    def __pysp_getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["module"] = encode_module(self.module)
        return state

    def __pysp_setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.module = decode_module(state["module"])

    def to_dict(self) -> dict[str, Any]:
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
    def __str__(self) -> str:
        return "EnergyModelEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, EnergyModelEncoder)

    def seq_encode(self, data: list) -> np.ndarray:
        return np.array([np.atleast_1d(np.asarray(x, dtype=float)) for x in data])


class EnergyModelAccumulator(SequenceEncodableStatisticAccumulator):
    """Buffers responsibility-weighted data for the NCE M-step (weights = the E-step soft counts)."""

    def __init__(self) -> None:
        self.x: list = []
        self.w: list = []

    # Contiguous batch arrays concatenated once at value() (shape-preserving) rather than one ndarray per row.
    def update(self, x: Any, weight: float, estimate: Any) -> None:
        self.x.append(np.atleast_1d(np.asarray(x, dtype=float))[None, ...])
        self.w.append(np.asarray([float(weight)], dtype=float))

    def seq_update(self, enc: Any, weights: np.ndarray, estimate: Any) -> None:
        xb = np.asarray(enc, dtype=float)
        self.x.append(xb.reshape(xb.shape[0], 1) if xb.ndim == 1 else xb)
        self.w.append(np.asarray(weights, dtype=float).ravel())

    def initialize(self, x: Any, weight: float, rng: Any) -> None:
        self.update(x, weight, None)

    def seq_initialize(self, enc: Any, weights: np.ndarray, rng: Any) -> None:
        self.seq_update(enc, weights, None)

    def combine(self, other: Any) -> EnergyModelAccumulator:
        xs, ws = other
        if len(xs):
            self.x.append(np.asarray(xs, dtype=float))
            self.w.append(np.asarray(ws, dtype=float).ravel())
        return self

    def value(self) -> tuple:
        x = np.concatenate(self.x, axis=0) if self.x else np.zeros((0, 0))
        w = np.concatenate(self.w) if self.w else np.zeros((0,))
        return (x, w)

    def from_value(self, v: tuple) -> EnergyModelAccumulator:
        x, w = v
        self.x = [np.asarray(x, dtype=float)] if len(x) else []
        self.w = [np.asarray(w, dtype=float).ravel()] if len(w) else []
        return self

    def acc_to_encoder(self) -> EnergyModelEncoder:
        return EnergyModelEncoder()


class EnergyModelAccumulatorFactory(StatisticAccumulatorFactory):
    def make(self) -> EnergyModelAccumulator:
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


def __getattr__(name: str) -> Any:  # PEP 562: lets ``pickle`` resolve the hoisted EnergyNet by name
    if name == "EnergyNet":
        return _energy_net_class()
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


def build_energy_net(dim: int, *, hidden: int = 64, layers: int = 3) -> Any:
    """An MLP energy ``E(x): R^dim -> R`` (plus a learned scalar ``log_norm``) -- ready to wrap in an EnergyModel.

    Lower energy = higher (unnormalized) density. ``log_norm`` is the NCE-learned normalizer, so the paired
    :class:`EnergyModel` scores ``-E(x) + log_norm``. Swap in any module exposing ``energy(x) -> (n,)``, a
    ``log_norm`` parameter and a ``dim`` attribute.
    """
    return _energy_net_class()(dim, hidden, layers)


def _register_serializable() -> None:
    # mixle.models classes aren't in the stats/analysis auto-walk, so opt in explicitly for to_json/from_json.
    try:
        from mixle.utils.serialization import register_serializable_class
    except Exception:  # pragma: no cover
        return
    register_serializable_class(EnergyModel)


_register_serializable()
