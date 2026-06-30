"""A streaming, non-buffering transformer-LM leaf -- the keystone that removes the host-RAM materialization wall.

Where ``SoftmaxNeuralLeaf`` buffers the whole shard in the accumulator and news a fresh optimizer every M-step,
this **inverts the EM abstraction**: the M-step OWNS a long-lived module + optimizer, and the accumulator's
``seq_update`` IS one train step on a streamed micro-batch. ``value()`` returns ``(loss_sum, tokens)`` -- two
floats, telemetry only, NEVER the corpus -- and ``estimate()`` is a no-op that wraps the live module.

This deliberately voids the sufficient-statistic algebra (``value``/``combine`` are telemetry, not a foldable
statistic): a sanctioned non-leaf carve-out (like ``NeuralLeaf.sample()`` raising), NOT an ABC change. It is the
single-process prerequisite for the distributed (FSDP2) neural handle: each rank keeps its streamed shard
resident and the only cross-rank collective becomes the in-backward gradient reduce-scatter, never a
gather-suff-stats-to-root.
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


class StreamingTransformerLeaf(SequenceEncodableProbabilityDistribution):
    """Wraps a live, persistently-trained module. ``seq_log_density`` = next-token ``log p`` (eval/telemetry)."""

    def __init__(self, module: Any, device: str = "cpu") -> None:
        self.module = module
        self.device = device

    def __str__(self) -> str:
        return "StreamingTransformerLeaf()"

    def log_density(self, xy: Any) -> float:
        return float(self.seq_log_density((np.atleast_2d(xy[0]), [int(xy[1])]))[0])

    def predict(self, x: Any) -> np.ndarray:
        torch = _torch()
        self.module.to(self.device)
        with torch.no_grad():
            logits = self.module(torch.as_tensor(np.atleast_2d(x), dtype=torch.float32).to(self.device))
        return logits.argmax(1).cpu().numpy()

    def sampler(self, seed: int | None = None) -> StreamingTransformerLeafSampler:
        return StreamingTransformerLeafSampler(self, seed)

    def seq_log_density(self, enc: Any) -> np.ndarray:
        torch = _torch()
        x, y = enc
        self.module.to(self.device)
        out = []
        with torch.no_grad():
            xt = torch.as_tensor(np.atleast_2d(x), dtype=torch.float32)
            for k in range(0, xt.shape[0], 4096):
                out.append(self.module(xt[k : k + 4096].to(self.device)).cpu().numpy())
        logp = _log_softmax(np.concatenate(out))
        y = np.asarray(y, dtype=int)
        return logp[np.arange(len(y)), y]

    def estimator(self, pseudo_count: float | None = None) -> StreamingTransformerLeafEstimator:
        return StreamingTransformerLeafEstimator(self.module, device=self.device)

    def dist_to_encoder(self) -> StreamingTokenEncoder:
        return StreamingTokenEncoder()


class StreamingTransformerLeafSampler(DistributionSampler):
    def __init__(self, dist: StreamingTransformerLeaf, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = np.random.RandomState(seed)

    def sample(self, size: int | None = None, *, batched: bool = True) -> Any:
        raise NotImplementedError(
            "StreamingTransformerLeaf is a conditional next-token model; feed contexts to generate."
        )


class StreamingTokenEncoder(DataSequenceEncoder):
    def __str__(self) -> str:
        return "StreamingTokenEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, StreamingTokenEncoder)

    def seq_encode(self, data: list) -> tuple[np.ndarray, np.ndarray]:
        x = np.array([np.atleast_1d(np.asarray(d[0], dtype=float)) for d in data])
        y = np.array([int(d[1]) for d in data], dtype=int)
        return (x, y)


class StreamingTransformerLeafAccumulator(SequenceEncodableStatisticAccumulator):
    """``seq_update`` = ONE train step against the PERSISTENT module + optimizer; ``value()`` = telemetry only."""

    def __init__(self, module: Any, lr: float, device: str) -> None:
        torch = _torch()
        self.module = module.to(device)
        self.device = device
        self.opt = torch.optim.AdamW(self.module.parameters(), lr=float(lr))  # persistent across seq_update calls
        self.ce = torch.nn.CrossEntropyLoss()
        self.loss_sum = 0.0
        self.tokens = 0
        self.last_loss = float("nan")

    def seq_update(self, enc: Any, weights: Any, estimate: Any) -> None:
        torch = _torch()
        x, y = enc
        xt = torch.as_tensor(np.asarray(x), dtype=torch.float32).to(self.device)
        yt = torch.as_tensor(np.asarray(y), dtype=torch.long).to(self.device)
        self.opt.zero_grad()
        loss = self.ce(self.module(xt), yt)
        loss.backward()
        self.opt.step()
        self.last_loss = float(loss.detach())
        self.loss_sum += self.last_loss * len(yt)
        self.tokens += int(len(yt))

    def update(self, x: Any, weight: float, estimate: Any) -> None:
        self.seq_update((np.atleast_2d(x[0]), [int(x[1])]), None, estimate)

    def initialize(self, x: Any, weight: float, rng: Any) -> None:
        pass  # streaming: no separate initialization pass

    def seq_initialize(self, enc: Any, weights: Any, rng: Any) -> None:
        pass

    def combine(self, other: Any) -> StreamingTransformerLeafAccumulator:
        ls, t = other
        self.loss_sum += float(ls)
        self.tokens += int(t)
        return self

    def value(self) -> tuple[float, int]:
        return (self.loss_sum, self.tokens)  # two floats -- never the corpus

    def from_value(self, v: tuple) -> StreamingTransformerLeafAccumulator:
        self.loss_sum, self.tokens = float(v[0]), int(v[1])
        return self

    def acc_to_encoder(self) -> StreamingTokenEncoder:
        return StreamingTokenEncoder()


class StreamingTransformerLeafAccumulatorFactory(StatisticAccumulatorFactory):
    def __init__(self, module: Any, lr: float, device: str) -> None:
        self.module = module
        self.lr = lr
        self.device = device

    def make(self) -> StreamingTransformerLeafAccumulator:
        return StreamingTransformerLeafAccumulator(self.module, self.lr, self.device)


class StreamingTransformerLeafEstimator(ParameterEstimator):
    def __init__(self, module: Any, lr: float = 3e-3, device: str = "cpu") -> None:
        self.module = module
        self.lr = lr
        self.device = device

    def accumulator_factory(self) -> StreamingTransformerLeafAccumulatorFactory:
        return StreamingTransformerLeafAccumulatorFactory(self.module, self.lr, self.device)

    def estimate(self, nobs: float | None, suff_stat: tuple) -> StreamingTransformerLeaf:
        # suff_stat = (loss_sum, tokens) telemetry; the module was already trained in place by seq_update -- no-op.
        return StreamingTransformerLeaf(self.module, self.device)


def stream_fit(
    module: Any, token_source: Any, *, lr: float = 3e-3, device: str = "cpu", report_every: int = 200, log: Any = None
) -> tuple:
    """Train ``module`` by streaming micro-batches from ``token_source`` (a generator). The accumulator holds the
    PERSISTENT optimizer and trains incrementally; its payload stays ``(loss_sum, tokens)`` -- the corpus is never
    buffered. Returns ``(StreamingTransformerLeaf, (loss_sum, tokens))``."""
    est = StreamingTransformerLeafEstimator(module, lr=lr, device=device)
    acc = est.accumulator_factory().make()
    step = 0
    window_sum = 0.0
    window_n = 0
    for batch in token_source:
        acc.seq_update(batch, None, None)
        step += 1
        window_sum += acc.last_loss
        window_n += 1
        if log is not None and step % int(report_every) == 0:
            log(step, window_sum / window_n)
            window_sum, window_n = 0.0, 0
    return est.estimate(None, acc.value()), acc.value()
