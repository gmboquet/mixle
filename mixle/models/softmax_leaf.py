"""A neural classifier as a mixle conditional-density leaf: ``p(y | x) = softmax(module(x))``.

The discriminative sibling of :class:`~mixle.models.neural_leaf.NeuralLeaf`. ``SoftmaxNeuralLeaf(module)`` wraps
a Torch module that emits ``k`` logits as a mixle distribution over observations ``(x, y)`` with ``y`` an integer
class index. It implements the full ``SequenceEncodableProbabilityDistribution`` contract, so it drops into
``MixtureDistribution`` / ``CompositeDistribution`` / HMM emissions like any leaf -- and its EM **M-step is a
responsibility-weighted cross-entropy gradient step** on the module (warm-started across EM iterations =>
generalized EM). The model's ``seq_log_density`` IS ``-cross_entropy(module(x), y)``: the objective is the
leaf's log-density, never a user-supplied loss closure.

This is the leaf that the declarative ``Categorical(logits=Net(...))`` PPL slot lowers to, and the component
that makes a ``Mix([Categorical(logits=Net(...)), ...])`` a mixture of neural classifiers fit by ordinary EM.

Requires torch. The leaf is conditional: ``predict(x)`` / ``sampler().sample_given(x)`` work; ``sample()`` raises
(there is no ``p(x)``) -- the same honest limitation ``NeuralLeaf`` and ``RandomForestConditional`` carry.
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


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    m = logits.max(axis=1, keepdims=True)
    return logits - m - np.log(np.exp(logits - m).sum(axis=1, keepdims=True))


class SoftmaxNeuralLeaf(SequenceEncodableProbabilityDistribution):
    """``p(y | x) = softmax(module(x))`` as a mixle leaf. Observation is the pair ``(x, y)``, ``y`` an int class."""

    def __init__(self, module: Any, m_steps: int = 40, lr: float = 0.01, name: str | None = None) -> None:
        self.module = module
        self.m_steps = int(m_steps)
        self.lr = float(lr)
        self.name = name

    def __str__(self) -> str:
        return "SoftmaxNeuralLeaf()"

    def _logits(self, x: np.ndarray) -> np.ndarray:
        torch = _torch()
        with torch.no_grad():
            out = self.module(torch.as_tensor(np.atleast_2d(x), dtype=torch.float32))
        return np.atleast_2d(out.detach().cpu().numpy())

    def log_density(self, xy: Any) -> float:
        x, y = xy
        return float(self.seq_log_density((np.atleast_2d(x), np.array([int(y)])))[0])

    def seq_log_density(self, enc: Any) -> np.ndarray:
        x, y = enc
        logp = _log_softmax(self._logits(x))
        y = np.asarray(y, dtype=int)
        return logp[np.arange(len(y)), y]

    def predict(self, x: Any) -> np.ndarray:
        p = self._logits(x).argmax(axis=1)
        return int(p[0]) if np.ndim(x) == 1 else p

    def sampler(self, seed: int | None = None) -> SoftmaxNeuralLeafSampler:
        return SoftmaxNeuralLeafSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> SoftmaxNeuralLeafEstimator:
        return SoftmaxNeuralLeafEstimator(self.module, self.m_steps, self.lr, self.name)

    def dist_to_encoder(self) -> SoftmaxNeuralLeafEncoder:
        return SoftmaxNeuralLeafEncoder()


class SoftmaxNeuralLeafSampler(DistributionSampler):
    def __init__(self, dist: SoftmaxNeuralLeaf, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = np.random.RandomState(seed)

    def sample(self, size: int | None = None, *, batched: bool = True) -> Any:
        raise NotImplementedError("SoftmaxNeuralLeaf is conditional p(y|x); use sampler().sample_given(x).")

    def sample_given(self, x: Any) -> int:
        p = np.exp(_log_softmax(self.dist._logits(x))[0])
        return int(self.rng.choice(len(p), p=p / p.sum()))


class SoftmaxNeuralLeafEncoder(DataSequenceEncoder):
    def __str__(self) -> str:
        return "SoftmaxNeuralLeafEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SoftmaxNeuralLeafEncoder)

    def seq_encode(self, data: list) -> tuple[np.ndarray, np.ndarray]:
        x = np.array([np.atleast_1d(np.asarray(xy[0], dtype=float)) for xy in data])
        y = np.array([int(xy[1]) for xy in data], dtype=int)
        return (x, y)


class SoftmaxNeuralLeafAccumulator(SequenceEncodableStatisticAccumulator):
    def __init__(self) -> None:
        self.x: list = []
        self.y: list = []
        self.w: list = []

    def update(self, xy: Any, weight: float, estimate: Any) -> None:
        self.x.append(np.atleast_1d(np.asarray(xy[0], dtype=float)))
        self.y.append(int(xy[1]))
        self.w.append(float(weight))

    def seq_update(self, enc: Any, weights: np.ndarray, estimate: Any) -> None:
        x, y = enc
        for i in range(len(x)):
            self.x.append(np.atleast_1d(x[i]))
            self.y.append(int(y[i]))
            self.w.append(float(weights[i]))

    def initialize(self, xy: Any, weight: float, rng: Any) -> None:
        self.update(xy, weight, None)

    def seq_initialize(self, enc: Any, weights: np.ndarray, rng: Any) -> None:
        self.seq_update(enc, weights, None)

    def combine(self, other: Any) -> SoftmaxNeuralLeafAccumulator:
        xo, yo, wo = other
        self.x.extend(xo)
        self.y.extend(yo)
        self.w.extend(wo)
        return self

    def value(self) -> tuple:
        return (list(self.x), list(self.y), list(self.w))

    def from_value(self, value: tuple) -> SoftmaxNeuralLeafAccumulator:
        self.x, self.y, self.w = list(value[0]), list(value[1]), list(value[2])
        return self

    def acc_to_encoder(self) -> SoftmaxNeuralLeafEncoder:
        return SoftmaxNeuralLeafEncoder()


class SoftmaxNeuralLeafAccumulatorFactory(StatisticAccumulatorFactory):
    def make(self) -> SoftmaxNeuralLeafAccumulator:
        return SoftmaxNeuralLeafAccumulator()


class SoftmaxNeuralLeafEstimator(ParameterEstimator):
    """EM estimator for a :class:`SoftmaxNeuralLeaf`: the M-step is ``m_steps`` of responsibility-weighted
    cross-entropy gradient on the module (the module is warm-started across EM iterations => generalized EM).

    The weighted CE is normalized by the responsibility mass ``sum(w)`` so the M-step is scale-invariant to the
    responsibility magnitude (the easy bug: an unnormalized weighted loss makes the step size track cluster size).
    """

    def __init__(self, module: Any, m_steps: int = 40, lr: float = 0.01, name: str | None = None) -> None:
        self.module = module
        self.m_steps = int(m_steps)
        self.lr = float(lr)
        self.name = name

    def accumulator_factory(self) -> SoftmaxNeuralLeafAccumulatorFactory:
        return SoftmaxNeuralLeafAccumulatorFactory()

    def estimate(self, nobs: float | None, suff_stat: tuple) -> SoftmaxNeuralLeaf:
        torch = _torch()
        xs, ys, ws = suff_stat
        if not xs:
            return SoftmaxNeuralLeaf(self.module, self.m_steps, self.lr, self.name)
        xt = torch.as_tensor(np.array(xs), dtype=torch.float32)
        yt = torch.as_tensor(np.array(ys), dtype=torch.long)
        wt = torch.as_tensor(np.array(ws), dtype=torch.float32)
        wsum = float(wt.sum()) + 1e-8
        opt = torch.optim.Adam(self.module.parameters(), lr=self.lr)
        ce = torch.nn.CrossEntropyLoss(reduction="none")
        for _ in range(self.m_steps):
            opt.zero_grad()
            loss = (wt * ce(self.module(xt), yt)).sum() / wsum
            loss.backward()
            opt.step()
        return SoftmaxNeuralLeaf(self.module, self.m_steps, self.lr, self.name)
