"""``NeuralDensityLeaf`` -- the adapter that turns ANY torch density module into a composable mixle distribution.

The point is not a specific architecture; it is the *wrapper*. ``NeuralLeaf`` already adapts a conditional net
(``p(y | x)``); this is its unconditional sibling: give it any torch module that exposes ``log_density(x) -> (n,)``
(and, to draw samples, ``sample(n) -> (n, d)``) and you get a full five-piece mixle ``Distribution`` -- so a
*flexible neural density* drops into a ``MixtureDistribution`` component, an HMM emission, or a
``CompositeDistribution`` field, and is fit **jointly with classical families** by EM. Its M-step is a
responsibility-weighted maximum-likelihood gradient ascent on the module, warm-started across EM iterations.

That is the thing no NN library offers: "a mixture of a normalizing flow and a Gamma", "an HMM whose emissions
are flows". :func:`build_coupling_flow` is a ready RealNVP-style module to wrap; any other exact density (a flow,
an autoregressive density, a normalized energy model) plugs in the same way.
"""

from __future__ import annotations

from typing import Any

import numpy as np

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


class NeuralDensityLeaf(SequenceEncodableProbabilityDistribution):
    """Wrap a torch density ``module`` (``module.log_density(x) -> (n,)``) as a composable mixle distribution."""

    def __init__(
        self, module: Any, *, m_steps: int = 60, lr: float = 5e-3, device: str = "cpu", name: str | None = None
    ) -> None:
        self.module = module
        self.m_steps = int(m_steps)
        self.lr = float(lr)
        self.device = device
        self.name = name

    def __str__(self) -> str:
        return f"NeuralDensityLeaf({type(self.module).__name__})"

    def log_density(self, x: Any) -> float:
        return float(self.seq_log_density(np.atleast_2d(np.asarray(x, dtype=float)))[0])

    def seq_log_density(self, x: Any) -> np.ndarray:
        torch = _torch()
        self.module.to(self.device).eval()
        xt = torch.as_tensor(np.atleast_2d(np.asarray(x, dtype=float)), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return self.module.log_density(xt).cpu().numpy().reshape(-1)

    def sampler(self, seed: int | None = None) -> NeuralDensitySampler:
        return NeuralDensitySampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> NeuralDensityEstimator:
        return NeuralDensityEstimator(self.module, m_steps=self.m_steps, lr=self.lr, device=self.device, name=self.name)

    def dist_to_encoder(self) -> NeuralDensityEncoder:
        return NeuralDensityEncoder()


class NeuralDensitySampler(DistributionSampler):
    def __init__(self, dist: NeuralDensityLeaf, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = np.random.RandomState(seed)

    def sample(self, size: int | None = None, *, batched: bool = True) -> Any:
        torch = _torch()
        n = int(size or 1)
        self.dist.module.to(self.dist.device).eval()
        torch.manual_seed(int(self.rng.randint(0, 2**31 - 1)))
        with torch.no_grad():
            out = self.dist.module.sample(n).cpu().numpy()
        return out if (size is not None) else out[0]


class NeuralDensityEncoder(DataSequenceEncoder):
    def __str__(self) -> str:
        return "NeuralDensityEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, NeuralDensityEncoder)

    def seq_encode(self, data: list) -> np.ndarray:
        return np.array([np.atleast_1d(np.asarray(x, dtype=float)) for x in data])


class NeuralDensityAccumulator(SequenceEncodableStatisticAccumulator):
    """Buffers the (responsibility-weighted) data for the M-step -- the weights are the E-step's soft counts."""

    def __init__(self) -> None:
        self.x: list = []
        self.w: list = []

    def update(self, x: Any, weight: float, estimate: Any) -> None:
        self.x.append(np.atleast_1d(np.asarray(x, dtype=float)))
        self.w.append(float(weight))

    def seq_update(self, enc: Any, weights: np.ndarray, estimate: Any) -> None:
        for i in range(len(enc)):
            self.x.append(np.atleast_1d(enc[i]))
            self.w.append(float(weights[i]))

    def initialize(self, x: Any, weight: float, rng: Any) -> None:
        self.update(x, weight, None)

    def seq_initialize(self, enc: Any, weights: np.ndarray, rng: Any) -> None:
        self.seq_update(enc, weights, None)

    def combine(self, other: Any) -> NeuralDensityAccumulator:
        xs, ws = other
        self.x.extend(xs)
        self.w.extend(ws)
        return self

    def value(self) -> tuple[list, list]:
        return (self.x, self.w)

    def from_value(self, v: tuple) -> NeuralDensityAccumulator:
        self.x, self.w = list(v[0]), list(v[1])
        return self

    def acc_to_encoder(self) -> NeuralDensityEncoder:
        return NeuralDensityEncoder()


class NeuralDensityAccumulatorFactory(StatisticAccumulatorFactory):
    def make(self) -> NeuralDensityAccumulator:
        return NeuralDensityAccumulator()


class NeuralDensityEstimator(ParameterEstimator):
    """M-step: responsibility-weighted MLE -- ``max sum_i w_i log p(x_i)`` by gradient ascent on the module (warm)."""

    def __init__(
        self, module: Any, *, m_steps: int = 60, lr: float = 5e-3, device: str = "cpu", name: str | None = None
    ) -> None:
        self.module = module
        self.m_steps = int(m_steps)
        self.lr = float(lr)
        self.device = device
        self.name = name

    def accumulator_factory(self) -> NeuralDensityAccumulatorFactory:
        return NeuralDensityAccumulatorFactory()

    def estimate(self, nobs: float | None, suff_stat: tuple) -> NeuralDensityLeaf:
        torch = _torch()
        xs, ws = suff_stat
        if not xs:
            return NeuralDensityLeaf(self.module, m_steps=self.m_steps, lr=self.lr, device=self.device, name=self.name)
        x = torch.as_tensor(np.stack(xs), dtype=torch.float32, device=self.device)
        w = torch.as_tensor(np.asarray(ws, dtype=float), dtype=torch.float32, device=self.device)
        w = w / w.sum().clamp(min=1e-8)
        self.module.to(self.device).train()
        opt = torch.optim.Adam(self.module.parameters(), lr=self.lr)
        for _ in range(self.m_steps):
            opt.zero_grad()
            loss = -(w * self.module.log_density(x)).sum()  # weighted negative log-likelihood
            loss.backward()
            opt.step()
        return NeuralDensityLeaf(self.module, m_steps=self.m_steps, lr=self.lr, device=self.device, name=self.name)


# --- a ready density module to wrap: a RealNVP-style coupling flow (exact log-density + sampling) ----------


def build_coupling_flow(dim: int, *, hidden: int = 32, layers: int = 4) -> Any:
    """A RealNVP coupling flow over ``R^dim`` with an exact ``log_density(x)`` and ``sample(n)`` -- ready to wrap.

    Alternating affine-coupling layers map data to a standard-normal base; ``log_density`` is the base log-prob
    plus the log-determinant of the (triangular) Jacobian. A minimal, correct instance of the density module a
    :class:`NeuralDensityLeaf` adapts -- swap in any other module with the same two methods.
    """
    import torch
    import torch.nn as nn

    class CouplingFlow(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dim = int(dim)
            masks = []
            for k in range(int(layers)):
                m = torch.zeros(self.dim)
                m[k % self.dim :: 2] = 1.0  # alternating coordinate masks
                masks.append(m)
            self.register_buffer("masks", torch.stack(masks))
            self.s = nn.ModuleList(
                [
                    nn.Sequential(nn.Linear(self.dim, hidden), nn.Tanh(), nn.Linear(hidden, self.dim))
                    for _ in range(int(layers))
                ]
            )
            self.t = nn.ModuleList(
                [
                    nn.Sequential(nn.Linear(self.dim, hidden), nn.Tanh(), nn.Linear(hidden, self.dim))
                    for _ in range(int(layers))
                ]
            )

        def _normalize(self, x: Any) -> tuple[Any, Any]:
            """x -> z (toward the base) and the accumulated log|det dz/dx|."""
            z = x
            logdet = torch.zeros(x.shape[0], device=x.device)
            for m, s_net, t_net in zip(self.masks, self.s, self.t):
                zm = z * m
                s = s_net(zm) * (1.0 - m)
                t = t_net(zm) * (1.0 - m)
                z = zm + (1.0 - m) * ((z - t) * torch.exp(-s))
                logdet = logdet - s.sum(dim=1)
            return z, logdet

        def log_density(self, x: Any) -> Any:
            z, logdet = self._normalize(x)
            base = -0.5 * (z**2).sum(dim=1) - 0.5 * self.dim * float(np.log(2.0 * np.pi))
            return base + logdet

        def sample(self, n: int) -> Any:
            z = torch.randn(int(n), self.dim, device=self.masks.device)
            x = z
            for m, s_net, t_net in zip(reversed(self.masks), reversed(list(self.s)), reversed(list(self.t))):
                xm = x * m
                s = s_net(xm) * (1.0 - m)
                t = t_net(xm) * (1.0 - m)
                x = xm + (1.0 - m) * (x * torch.exp(s) + t)  # inverse of _normalize
            return x

    return CouplingFlow()
