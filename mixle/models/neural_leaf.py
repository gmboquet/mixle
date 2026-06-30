"""A neural network as a mixle conditional-density leaf -- the bridge that makes nets *generative* components.

``NeuralLeaf(module)`` wraps a Torch module as a mixle distribution ``p(y | x) = N(y; module(x), noise^2 I)``
over observations ``(x, y)``. It implements the full ``SequenceEncodableProbabilityDistribution`` contract, so
it drops into ``MixtureDistribution`` / ``CompositeDistribution`` / HMM emissions like any leaf -- but its EM
**M-step is weighted-NLL gradient descent** on the module (warm-started across EM iterations => generalized EM).

A ``MixtureDistribution`` of ``NeuralLeaf`` components is therefore a **mixture of neural experts**: the E-step
computes responsibilities, the M-step trains each expert by responsibility-weighted regression. Combined with
the ``em`` move in :mod:`mixle.experimental.program`, the same model fits with EM where conjugate and gradient where neural::

    from mixle.stats import MixtureEstimator
    experts = MixtureEstimator([NeuralLeaf(mlp_a).estimator(), NeuralLeaf(mlp_b).estimator()])
    # ... run EM (estimate loop) -> each expert specializes, gated by the responsibilities.

Requires torch (the module). The leaf is conditional: ``sampler().sample_given(x)`` draws ``y``; ``sample()``
raises (there is no ``p(x)``).
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


class NeuralLeaf(SequenceEncodableProbabilityDistribution):
    """``p(y | x) = N(y; module(x), noise^2 I)`` as a mixle leaf. Observation is the pair ``(x, y)``."""

    def __init__(
        self, module: Any, noise: float = 1.0, m_steps: int = 40, lr: float = 0.01, name: str | None = None
    ) -> None:
        self.module = module
        self.noise = float(noise)
        self.m_steps = int(m_steps)
        self.lr = float(lr)
        self.name = name

    def __str__(self) -> str:
        return "NeuralLeaf(noise=%.3g)" % self.noise

    def _forward(self, x: np.ndarray) -> np.ndarray:
        torch = _torch()
        with torch.no_grad():
            mean = self.module(torch.as_tensor(np.atleast_2d(x), dtype=torch.float32))
        return np.atleast_2d(mean.detach().cpu().numpy())

    def log_density(self, xy: Any) -> float:
        x, y = xy
        return float(self.seq_log_density((np.atleast_2d(x), np.atleast_2d(y)))[0])

    def seq_log_density(self, enc: Any) -> np.ndarray:
        x, y = enc
        mean = self._forward(x)
        y = np.atleast_2d(np.asarray(y, dtype=float))
        d = y.shape[1]
        sq = ((y - mean) ** 2).sum(axis=1)
        return -0.5 * sq / (self.noise**2) - 0.5 * d * np.log(2.0 * np.pi * self.noise**2)

    def sampler(self, seed: int | None = None) -> NeuralLeafSampler:
        return NeuralLeafSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> NeuralLeafEstimator:
        return NeuralLeafEstimator(self.module, self.noise, self.m_steps, self.lr, self.name)

    def dist_to_encoder(self) -> NeuralLeafEncoder:
        return NeuralLeafEncoder()


class NeuralLeafSampler(DistributionSampler):
    def __init__(self, dist: NeuralLeaf, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = np.random.RandomState(seed)

    def sample(self, size: int | None = None, *, batched: bool = True) -> Any:
        raise NotImplementedError("NeuralLeaf is conditional p(y|x); use sampler().sample_given(x).")

    def sample_given(self, x: Any) -> np.ndarray:
        mean = self.dist._forward(x)[0]
        return mean + self.dist.noise * self.rng.randn(*mean.shape)


class NeuralLeafEncoder(DataSequenceEncoder):
    def __str__(self) -> str:
        return "NeuralLeafEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, NeuralLeafEncoder)

    def seq_encode(self, data: list) -> tuple[np.ndarray, np.ndarray]:
        x = np.array([np.atleast_1d(np.asarray(xy[0], dtype=float)) for xy in data])
        y = np.array([np.atleast_1d(np.asarray(xy[1], dtype=float)) for xy in data])
        return (x, y)


class NeuralLeafAccumulator(SequenceEncodableStatisticAccumulator):
    def __init__(self) -> None:
        self.x: list = []
        self.y: list = []
        self.w: list = []

    def update(self, xy: Any, weight: float, estimate: Any) -> None:
        self.x.append(np.atleast_1d(np.asarray(xy[0], dtype=float)))
        self.y.append(np.atleast_1d(np.asarray(xy[1], dtype=float)))
        self.w.append(float(weight))

    def seq_update(self, enc: Any, weights: np.ndarray, estimate: Any) -> None:
        x, y = enc
        for i in range(len(x)):
            self.x.append(np.atleast_1d(x[i]))
            self.y.append(np.atleast_1d(y[i]))
            self.w.append(float(weights[i]))

    def initialize(self, xy: Any, weight: float, rng: Any) -> None:
        self.update(xy, weight, None)

    def seq_initialize(self, enc: Any, weights: np.ndarray, rng: Any) -> None:
        self.seq_update(enc, weights, None)

    def combine(self, other: Any) -> NeuralLeafAccumulator:
        xo, yo, wo = other
        self.x.extend(xo)
        self.y.extend(yo)
        self.w.extend(wo)
        return self

    def value(self) -> tuple:
        return (list(self.x), list(self.y), list(self.w))

    def from_value(self, value: tuple) -> NeuralLeafAccumulator:
        self.x, self.y, self.w = list(value[0]), list(value[1]), list(value[2])
        return self

    def acc_to_encoder(self) -> NeuralLeafEncoder:
        return NeuralLeafEncoder()


class NeuralLeafAccumulatorFactory(StatisticAccumulatorFactory):
    def make(self) -> NeuralLeafAccumulator:
        return NeuralLeafAccumulator()


class NeuralLeafEstimator(ParameterEstimator):
    """EM estimator for a :class:`NeuralLeaf`: the M-step is ``m_steps`` of weighted-NLL gradient on the module.

    The module is held (and warm-started) across EM iterations, so each M-step is a *partial* maximization
    (generalized EM). The accumulator buffers responsibility-weighted ``(x, y)`` observations.
    """

    def __init__(
        self, module: Any, noise: float = 1.0, m_steps: int = 40, lr: float = 0.01, name: str | None = None
    ) -> None:
        self.module = module
        self.noise = float(noise)
        self.m_steps = int(m_steps)
        self.lr = float(lr)
        self.name = name

    def accumulator_factory(self) -> NeuralLeafAccumulatorFactory:
        return NeuralLeafAccumulatorFactory()

    def estimate(self, nobs: float | None, suff_stat: tuple) -> NeuralLeaf:
        torch = _torch()
        xs, ys, ws = suff_stat
        if not xs:
            return NeuralLeaf(self.module, self.noise, self.m_steps, self.lr, self.name)
        xt = torch.as_tensor(np.array(xs), dtype=torch.float32)
        yt = torch.as_tensor(np.array(ys), dtype=torch.float32)
        wt = torch.as_tensor(np.array(ws), dtype=torch.float32)
        wsum = float(wt.sum()) + 1e-8
        log_noise = torch.log(torch.tensor(float(self.noise))).clone().detach().requires_grad_(True)
        opt = torch.optim.Adam(list(self.module.parameters()) + [log_noise], lr=self.lr)
        d = yt.shape[1]
        for _ in range(self.m_steps):
            opt.zero_grad()
            mean = self.module(xt)
            sig2 = torch.exp(2.0 * log_noise)
            nll = (wt * (0.5 * ((yt - mean) ** 2).sum(1) / sig2 + 0.5 * d * torch.log(2.0 * np.pi * sig2))).sum() / wsum
            nll.backward()
            opt.step()
        self.noise = float(torch.exp(log_noise).detach())  # warm-start noise for the next EM iteration
        return NeuralLeaf(self.module, self.noise, self.m_steps, self.lr, self.name)
