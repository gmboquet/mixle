"""A streaming, non-buffering transformer-LM leaf for avoiding host-RAM materialization.

Where ``NeuralCategorical`` buffers the whole shard in the accumulator and creates a fresh optimizer every
M-step, this leaf keeps a long-lived module and optimizer in the M-step. The accumulator's ``seq_update``
is one train step on a streamed micro-batch. ``value()`` returns ``(loss_sum, tokens)`` -- two telemetry
floats rather than the corpus -- and ``estimate()`` is a no-op that wraps the live module.

This deliberately voids the sufficient-statistic algebra (``value``/``combine`` are telemetry, not a foldable
statistic): a sanctioned non-leaf carve-out (like ``NeuralGaussian.sample()`` raising), not an ABC change. It is the
single-process prerequisite for the distributed neural handle: each rank keeps its streamed shard
resident and the only cross-rank collective becomes the in-backward gradient reduce-scatter, never a
gather-suff-stats-to-root.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mixle.models._neural_serial import decode_module, encode_module
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


class StreamingTransformer(SequenceEncodableProbabilityDistribution):
    """Wraps a live, persistently-trained module. ``seq_log_density`` = next-token ``log p`` (eval/telemetry)."""

    __pysp_serializable__ = True  # module persisted as bytes (see __pysp_getstate__); leaf round-trips in a mixture

    def __init__(self, module: Any, device: str = "cpu") -> None:
        self.module = module
        self.device = device

    @classmethod
    def from_config(
        cls,
        vocab: int,
        *,
        d_model: int = 128,
        n_layer: int = 4,
        n_head: int = 4,
        block: int = 64,
        embedding: Any = None,
        device: str = "cpu",
    ) -> StreamingTransformer:
        """Build the leaf from hyperparameters (no hand-built torch module) -- the declarative estimator surface.

        ``embedding`` optionally ties a shared :class:`~mixle.models.embedding.CategoricalEmbedding` across leaves.
        """
        from mixle.models.transformer import build_causal_lm

        module = build_causal_lm(vocab, d_model, n_layer, n_head, block, embedding=embedding)
        return cls(module, device=device)

    def __str__(self) -> str:
        return "StreamingTransformer()"

    def log_density(self, xy: Any) -> float:
        """Return the next-token log probability for one ``(context, token)`` pair."""
        return float(self.seq_log_density((np.atleast_2d(xy[0]), [int(xy[1])]))[0])

    def predict(self, x: Any) -> np.ndarray:
        """Return argmax next-token predictions for one or more contexts."""
        torch = _torch()
        self.module.to(self.device)
        with torch.no_grad():
            logits = self.module(torch.as_tensor(np.atleast_2d(x), dtype=torch.float32).to(self.device))
        return logits.argmax(1).cpu().numpy()

    def sampler(self, seed: int | None = None) -> StreamingTransformerSampler:
        """Return the sampler for the conditional next-token model."""
        return StreamingTransformerSampler(self, seed)

    def seq_log_density(self, enc: Any) -> np.ndarray:
        """Return per-row next-token log probabilities for encoded context/token pairs."""
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

    def estimator(self, pseudo_count: float | None = None) -> StreamingTransformerEstimator:
        """Return the streaming estimator that trains the live module in accumulator updates."""
        return StreamingTransformerEstimator(self.module, device=self.device)

    def dist_to_encoder(self) -> StreamingTokenEncoder:
        """Return the encoder for context/token training pairs."""
        return StreamingTokenEncoder()

    # --- serialization: persist the module (as portable bytes); registered below so a mixture holding this
    # leaf round-trips through to_dict/to_json/pickle as well. ---
    def __pysp_getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["module"] = encode_module(self.module)
        return state

    def __pysp_setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.module = decode_module(state["module"])

    def to_dict(self) -> dict[str, Any]:
        """Serialize the module bytes and device for registry-based round trips."""
        return {"module": encode_module(self.module), "device": self.device}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> StreamingTransformer:
        """Rebuild a :class:`StreamingTransformer` from :meth:`to_dict` output."""
        return cls(decode_module(payload["module"]), device=payload["device"])


class StreamingTransformerSampler(DistributionSampler):
    """Sampler facade for a conditional next-token transformer leaf."""

    def __init__(self, dist: StreamingTransformer, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = np.random.RandomState(seed)

    def sample(self, size: int | None = None, *, batched: bool = True) -> Any:
        """Raise because contexts are required for transformer generation."""
        raise NotImplementedError("StreamingTransformer is a conditional next-token model; feed contexts to generate.")


class StreamingTokenEncoder(DataSequenceEncoder):
    """Encode context/token pairs for streaming transformer scoring and training."""

    def __str__(self) -> str:
        return "StreamingTokenEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, StreamingTokenEncoder)

    def seq_encode(self, data: list) -> tuple[np.ndarray, np.ndarray]:
        """Convert ``(context, token)`` pairs into batched context and integer-token arrays."""
        x = np.array([np.atleast_1d(np.asarray(d[0], dtype=float)) for d in data])
        y = np.array([int(d[1]) for d in data], dtype=int)
        return (x, y)


class StreamingTransformerAccumulator(SequenceEncodableStatisticAccumulator):
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
        """Run one optimizer step on an encoded micro-batch."""
        torch = _torch()
        x, y = enc
        xt = torch.as_tensor(np.asarray(x), dtype=torch.float32).to(self.device)
        yt = torch.as_tensor(np.asarray(y), dtype=torch.long).to(self.device)
        self.opt.zero_grad()
        logits = self.module(xt)
        if weights is None:
            loss = self.ce(logits, yt)  # pure-streaming path unchanged (uniform)
        else:
            # per-token responsibility (mixture expert / streaming decay / sample weight): weighted mean, so the
            # gradient scale matches the unweighted mean when the weights are uniform (bit-identical then).
            wt = torch.as_tensor(np.asarray(weights, dtype=float), dtype=torch.float32).to(self.device).ravel()
            per = torch.nn.functional.cross_entropy(logits, yt, reduction="none")
            loss = (wt * per).sum() / wt.sum().clamp(min=1e-8)
        loss.backward()
        self.opt.step()
        self.last_loss = float(loss.detach())
        self.loss_sum += self.last_loss * len(yt)
        self.tokens += int(len(yt))

    def update(self, x: Any, weight: float, estimate: Any) -> None:
        """Train on one weighted context/token pair through :meth:`seq_update`."""
        self.seq_update((np.atleast_2d(x[0]), [int(x[1])]), [float(weight)], estimate)

    def initialize(self, x: Any, weight: float, rng: Any) -> None:
        """No-op initialization hook for the streaming training path."""
        pass  # streaming: no separate initialization pass

    def seq_initialize(self, enc: Any, weights: Any, rng: Any) -> None:
        """No-op batch initialization hook for the streaming training path."""
        pass

    def combine(self, other: Any) -> StreamingTransformerAccumulator:
        """Merge telemetry from another streaming accumulator."""
        ls, t = other
        self.loss_sum += float(ls)
        self.tokens += int(t)
        return self

    def value(self) -> tuple[float, int]:
        """Return ``(loss_sum, token_count)`` telemetry without storing the corpus."""
        return (self.loss_sum, self.tokens)  # two floats -- never the corpus

    def from_value(self, v: tuple) -> StreamingTransformerAccumulator:
        """Restore telemetry counters from a value tuple."""
        self.loss_sum, self.tokens = float(v[0]), int(v[1])
        return self

    def acc_to_encoder(self) -> StreamingTokenEncoder:
        """Return the encoder expected by this accumulator."""
        return StreamingTokenEncoder()


class StreamingTransformerAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for streaming transformer accumulators sharing a live module."""

    def __init__(self, module: Any, lr: float, device: str) -> None:
        self.module = module
        self.lr = lr
        self.device = device

    def make(self) -> StreamingTransformerAccumulator:
        """Create a fresh accumulator around the shared live module."""
        return StreamingTransformerAccumulator(self.module, self.lr, self.device)


class StreamingTransformerEstimator(ParameterEstimator):
    """Estimator whose accumulator trains a live streaming transformer module in place."""

    def __init__(self, module: Any, lr: float = 3e-3, device: str = "cpu") -> None:
        self.module = module
        self.lr = lr
        self.device = device

    def accumulator_factory(self) -> StreamingTransformerAccumulatorFactory:
        """Return an accumulator factory for streamed context/token micro-batches."""
        return StreamingTransformerAccumulatorFactory(self.module, self.lr, self.device)

    def estimate(self, nobs: float | None, suff_stat: tuple) -> StreamingTransformer:
        """Return the live module wrapped as a fitted streaming transformer leaf."""
        # suff_stat = (loss_sum, tokens) telemetry; the module was already trained in place by seq_update -- no-op.
        return StreamingTransformer(self.module, self.device)


class TransformerLMEstimator(StreamingTransformerEstimator):
    """A Transformer language model as a fit-ready estimator: ``TransformerLMEstimator(vocab, d_model=..., ...)``.

    The clean, declarative surface -- no hand-built torch module, no ``Leaf(...).estimator()`` two-step. Drops into
    ``MixtureEstimator``/``CompositeEstimator`` like any other ``*Estimator``. ``embedding`` optionally ties a
    shared :class:`~mixle.models.embedding.CategoricalEmbedding` (e.g. one word embedding across a mixture's
    experts). ``TransformerLMEstimator(V, embedding=emb)`` and ``StreamingTransformer.from_config(V,
    embedding=emb).estimator()`` build the same thing.
    """

    def __init__(
        self,
        vocab: int,
        *,
        d_model: int = 128,
        n_layer: int = 4,
        n_head: int = 4,
        block: int = 64,
        embedding: Any = None,
        lr: float = 3e-3,
        device: str = "cpu",
    ) -> None:
        from mixle.models.transformer import build_causal_lm

        module = build_causal_lm(vocab, d_model, n_layer, n_head, block, embedding=embedding)
        super().__init__(module, lr=lr, device=device)


def stream_fit(
    module: Any, token_source: Any, *, lr: float = 3e-3, device: str = "cpu", report_every: int = 200, log: Any = None
) -> tuple:
    """Train ``module`` by streaming micro-batches from ``token_source`` (a generator). The accumulator holds the
    PERSISTENT optimizer and trains incrementally; its payload stays ``(loss_sum, tokens)`` -- the corpus is never
    buffered. Returns ``(StreamingTransformer, (loss_sum, tokens))``."""
    est = StreamingTransformerEstimator(module, lr=lr, device=device)
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


def _register_serializable() -> None:
    # mixle.models classes aren't in the stats/analysis auto-walk, so opt in explicitly for to_json/from_json.
    try:
        from mixle.utils.serialization import register_serializable_class
    except Exception:  # pragma: no cover  # noqa: BLE001
        return
    register_serializable_class(StreamingTransformer)


_register_serializable()


# --- back-compat aliases (the classes were renamed off the '...Leaf' suffix) ---
StreamingTransformerLeaf = StreamingTransformer
StreamingTransformerLeafEstimator = StreamingTransformerEstimator
