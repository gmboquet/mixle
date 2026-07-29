"""A streaming, non-buffering transformer-LM leaf for avoiding host-RAM materialization.

Where ``NeuralCategorical`` buffers a shard, this leaf keeps the corpus out of the accumulator. Each
``seq_update`` computes a weighted gradient numerator at a pinned base-model digest without changing the
module. ``value()`` returns model-sized gradients plus weighted telemetry; ``combine()`` sums only compatible
worker states, and ``estimate()`` applies one step through a persistent AdamW optimizer. Thus partitioned
and unpartitioned accumulation have explicit data-parallel semantics instead of silently discarding
worker-trained weights. :func:`stream_fit` preserves the familiar one-step-per-micro-batch behavior by
estimating after every streamed batch.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import io
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from mixle.models._neural_serial import (
    _require_trusted_deserialization,
    _serialization_error,
    decode_module,
    encode_module,
)
from mixle.models.grad_leaf import _module_mode
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
    if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] < 2:
        raise ValueError("transformer logits must have shape (n, actions) with n > 0 and actions >= 2")
    if not np.all(np.isfinite(logits)):
        raise ValueError("transformer logits must contain only finite values")
    m = logits.max(axis=1, keepdims=True)
    return logits - m - np.log(np.exp(logits - m).sum(axis=1, keepdims=True))


@dataclass
class StreamingGradientState:
    """Foldable weighted-gradient state computed at one exact model revision."""

    base_sha256: str
    parameter_names: tuple[str, ...]
    gradient_sums: tuple[np.ndarray, ...]
    loss_sum: float
    effective_weight: float
    rows: int
    batches: int

    @property
    def mean_loss(self) -> float:
        """Return the consistently weighted mean loss."""
        if self.effective_weight <= 0.0:
            raise ValueError("streaming gradient state has no positive effective weight")
        return self.loss_sum / self.effective_weight


class StreamingTransformer(SequenceEncodableProbabilityDistribution):
    """Wraps a live, persistently-trained module. ``seq_log_density`` = next-token ``log p`` (eval/telemetry)."""

    __pysp_serializable__ = True  # module persisted as bytes (see __pysp_getstate__); leaf round-trips in a mixture

    def __init__(
        self,
        module: Any,
        device: str = "cpu",
        *,
        lr: float = 3.0e-3,
        optimizer_state: dict[str, Any] | None = None,
    ) -> None:
        torch = _torch()
        if not isinstance(module, torch.nn.Module):
            raise TypeError("module must be a torch.nn.Module")
        try:
            device = str(torch.device(device))
        except (TypeError, RuntimeError) as exc:
            raise ValueError(f"invalid torch device {device!r}") from exc
        self.module = module
        self.device = device
        self.lr = _positive_finite(lr, "lr")
        if optimizer_state is not None and not isinstance(optimizer_state, dict):
            raise TypeError("optimizer_state must be a dictionary or None")
        self.optimizer_state = optimizer_state

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
        return float(self.seq_log_density((np.atleast_2d(xy[0]), [xy[1]]))[0])

    def predict(self, x: Any) -> np.ndarray:
        """Return argmax next-token predictions for one or more contexts."""
        torch = _torch()
        self.module.to(self.device)
        contexts = _contexts(x)
        _validate_model_contexts(contexts, self.module)
        dtype = _module_dtype(self.module, torch)
        with _module_mode(self.module, train=False), torch.no_grad():
            logits = self.module(torch.as_tensor(contexts, dtype=dtype, device=self.device))
        _validate_logits(logits, len(contexts), torch)
        return logits.argmax(1).cpu().numpy()

    def sampler(self, seed: int | None = None) -> StreamingTransformerSampler:
        """Return the sampler for the conditional next-token model."""
        return StreamingTransformerSampler(self, seed)

    def seq_log_density(self, enc: Any) -> np.ndarray:
        """Return per-row next-token log probabilities for encoded context/token pairs."""
        torch = _torch()
        x, y, _ = _training_batch(enc, None, self.module)
        self.module.to(self.device)
        dtype = _module_dtype(self.module, torch)
        out = []
        with _module_mode(self.module, train=False), torch.no_grad():
            xt = torch.as_tensor(x, dtype=dtype)
            for k in range(0, xt.shape[0], 4096):
                logits = self.module(xt[k : k + 4096].to(self.device))
                _validate_logits(logits, len(logits), torch)
                out.append(logits.cpu().numpy())
        logp = _log_softmax(np.concatenate(out))
        if np.any(y >= logp.shape[1]):
            raise ValueError(f"next-token actions must lie in [0, {logp.shape[1]})")
        return logp[np.arange(len(y)), y]

    def estimator(self, pseudo_count: float | None = None) -> StreamingTransformerEstimator:
        """Return the streaming estimator that trains the live module in accumulator updates."""
        return StreamingTransformerEstimator(
            self.module,
            lr=self.lr,
            device=self.device,
            optimizer_state=self.optimizer_state,
        )

    def dist_to_encoder(self) -> StreamingTokenEncoder:
        """Return the encoder for context/token training pairs."""
        return StreamingTokenEncoder()

    # --- serialization: persist the module (as portable bytes); registered below so a mixture holding this
    # leaf round-trips through to_dict/to_json/pickle as well. ---
    def __pysp_getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["module"] = encode_module(self.module)
        state["optimizer_state"] = _encode_optimizer_state(self.optimizer_state)
        return state

    def __pysp_setstate__(self, state: dict[str, Any]) -> None:
        restored = type(self)(
            decode_module(state["module"]),
            device=state["device"],
            lr=state.get("lr", 3.0e-3),
            optimizer_state=_decode_optimizer_state(state.get("optimizer_state")),
        )
        self.__dict__.update(restored.__dict__)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the module bytes and device for registry-based round trips."""
        return {
            "module": encode_module(self.module),
            "device": self.device,
            "lr": self.lr,
            "optimizer_state": _encode_optimizer_state(self.optimizer_state),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> StreamingTransformer:
        """Rebuild a :class:`StreamingTransformer` from :meth:`to_dict` output."""
        return cls(
            decode_module(payload["module"]),
            device=payload["device"],
            lr=payload.get("lr", 3.0e-3),
            optimizer_state=_decode_optimizer_state(payload.get("optimizer_state")),
        )


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
        if not isinstance(data, list):
            raise TypeError("StreamingTokenEncoder.seq_encode expects a list")
        if not data:
            return (np.zeros((0, 0)), np.zeros(0, dtype=int))
        if any(not isinstance(row, (tuple, list)) or len(row) != 2 for row in data):
            raise ValueError("every streaming observation must be a (context, token) pair")
        contexts, actions, _ = _training_batch(
            ([row[0] for row in data], [row[1] for row in data]),
            None,
        )
        return contexts, actions


class StreamingTransformerAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate foldable weighted gradients at one exact model revision."""

    def __init__(self, module: Any, lr: float, device: str) -> None:
        torch = _torch()
        self.module = module.to(device)
        self.device = device
        self.lr = _positive_finite(lr, "lr")
        self.base_sha256 = _model_digest(self.module)
        self.parameter_names = tuple(
            name for name, parameter in self.module.named_parameters() if parameter.requires_grad
        )
        if not self.parameter_names:
            raise ValueError("streaming transformer must contain trainable parameters")
        self.gradient_sums: list[np.ndarray | None] = [None] * len(self.parameter_names)
        self.loss_sum = 0.0
        self.effective_weight = 0.0
        self.rows = 0
        self.batches = 0
        self.last_loss = float("nan")

    def seq_update(self, enc: Any, weights: Any, estimate: Any) -> None:
        """Accumulate one weighted gradient numerator without mutating the shared model."""
        torch = _torch()
        if _model_digest(self.module) != self.base_sha256:
            raise RuntimeError("streaming accumulator's model changed after its base revision was captured")
        x, y, sample_weights = _training_batch(enc, weights, self.module)
        dtype = _module_dtype(self.module, torch)
        xt = torch.as_tensor(x, dtype=dtype, device=self.device)
        yt = torch.as_tensor(y, dtype=torch.long, device=self.device)
        wt = torch.as_tensor(sample_weights, dtype=dtype, device=self.device)
        parameters = dict(self.module.named_parameters())
        trainable = [parameters[name] for name in self.parameter_names]
        with _module_mode(self.module, train=True):
            logits = self.module(xt)
            _validate_logits(logits, len(x), torch)
            if torch.any(yt >= logits.shape[1]):
                raise ValueError(f"next-token actions must lie in [0, {logits.shape[1]})")
            per = torch.nn.functional.cross_entropy(logits, yt, reduction="none")
            numerator = (wt * per).sum()
            if not bool(torch.isfinite(numerator).detach().cpu().item()):
                raise RuntimeError("streaming transformer loss became non-finite")
            gradients = torch.autograd.grad(numerator, trainable, allow_unused=True)
        for index, (parameter, gradient) in enumerate(zip(trainable, gradients)):
            value = (
                np.zeros(tuple(parameter.shape), dtype=np.float64)
                if gradient is None
                else gradient.detach().cpu().numpy().astype(np.float64, copy=False)
            )
            if not np.all(np.isfinite(value)):
                raise RuntimeError(f"streaming gradient for parameter {self.parameter_names[index]!r} is non-finite")
            if self.gradient_sums[index] is None:
                self.gradient_sums[index] = value.copy()
            else:
                self.gradient_sums[index] += value
        batch_weight = float(np.sum(sample_weights))
        batch_loss_sum = float(numerator.detach().cpu().item())
        self.last_loss = batch_loss_sum / batch_weight
        self.loss_sum += batch_loss_sum
        self.effective_weight += batch_weight
        self.rows += len(y)
        self.batches += 1

    def update(self, x: Any, weight: float, estimate: Any) -> None:
        """Accumulate one weighted context/token gradient."""
        self.seq_update((np.atleast_2d(x[0]), [x[1]]), [weight], estimate)

    def initialize(self, x: Any, weight: float, rng: Any) -> None:
        """Initialize through the same gradient accumulation path."""
        self.update(x, weight, None)

    def seq_initialize(self, enc: Any, weights: Any, rng: Any) -> None:
        """Initialize a batch through the same gradient accumulation path."""
        self.seq_update(enc, weights, None)

    def combine(self, other: Any) -> StreamingTransformerAccumulator:
        """Sum worker gradients only when they were computed at the identical base revision."""
        state = _gradient_state(other)
        if state.base_sha256 != self.base_sha256:
            raise ValueError("cannot combine streaming gradients computed from different model revisions")
        if state.parameter_names != self.parameter_names or len(state.gradient_sums) != len(self.gradient_sums):
            raise ValueError("cannot combine streaming gradients with a different parameter schema")
        for index, gradient in enumerate(state.gradient_sums):
            if self.gradient_sums[index] is None:
                self.gradient_sums[index] = np.asarray(gradient, dtype=np.float64).copy()
            else:
                if self.gradient_sums[index].shape != np.asarray(gradient).shape:
                    raise ValueError("cannot combine streaming gradients with incompatible parameter shapes")
                self.gradient_sums[index] += gradient
        self.loss_sum += state.loss_sum
        self.effective_weight += state.effective_weight
        self.rows += state.rows
        self.batches += state.batches
        self.last_loss = self.loss_sum / self.effective_weight
        return self

    def value(self) -> StreamingGradientState:
        """Return model-sized gradients and weighted telemetry, never the corpus."""
        if (
            self.batches == 0
            or self.effective_weight <= 0.0
            or any(gradient is None for gradient in self.gradient_sums)
        ):
            raise ValueError("streaming accumulator has no positive-weight gradient batches")
        return StreamingGradientState(
            self.base_sha256,
            self.parameter_names,
            tuple(np.asarray(gradient).copy() for gradient in self.gradient_sums),
            self.loss_sum,
            self.effective_weight,
            self.rows,
            self.batches,
        )

    def from_value(self, v: Any) -> StreamingTransformerAccumulator:
        """Restore a complete foldable gradient state."""
        state = _gradient_state(v)
        if state.base_sha256 != self.base_sha256 or state.parameter_names != self.parameter_names:
            raise ValueError("streaming gradient state does not match this accumulator's model revision")
        self.gradient_sums = [np.asarray(gradient, dtype=np.float64).copy() for gradient in state.gradient_sums]
        self.loss_sum = state.loss_sum
        self.effective_weight = state.effective_weight
        self.rows = state.rows
        self.batches = state.batches
        self.last_loss = state.mean_loss
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
    """Persistent-optimizer estimator applying foldable worker gradient sums."""

    def __init__(
        self,
        module: Any,
        lr: float = 3e-3,
        device: str = "cpu",
        optimizer_state: dict[str, Any] | None = None,
    ) -> None:
        torch = _torch()
        if not isinstance(module, torch.nn.Module):
            raise TypeError("module must be a torch.nn.Module")
        try:
            self.device = str(torch.device(device))
        except (TypeError, RuntimeError) as exc:
            raise ValueError(f"invalid torch device {device!r}") from exc
        self.module = module.to(self.device)
        self.lr = _positive_finite(lr, "lr")
        self.optimizer = torch.optim.AdamW(self.module.parameters(), lr=self.lr)
        if optimizer_state is not None:
            if not isinstance(optimizer_state, dict):
                raise TypeError("optimizer_state must be a dictionary or None")
            try:
                self.optimizer.load_state_dict(optimizer_state)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("optimizer_state is incompatible with the streaming transformer") from exc
        self._pinned_digest: str | None = None

    def accumulator_factory(self) -> StreamingTransformerAccumulatorFactory:
        """Return workers pinned to the estimator's current model revision."""
        self._pinned_digest = _model_digest(self.module)
        return StreamingTransformerAccumulatorFactory(self.module, self.lr, self.device)

    def estimate(self, nobs: float | None, suff_stat: Any) -> StreamingTransformer:
        """Apply one persistent AdamW step from aggregated weighted gradient numerators."""
        torch = _torch()
        state = _gradient_state(suff_stat)
        # Check against the revision this estimator PINNED when it handed out accumulators, not
        # against the module as it stands now. Those differ exactly in the case the leaf exists to
        # support: several mixture components sharing one embedding tensor. Every component's
        # accumulators are built (and pinned) before any component estimates, so the first
        # component's optimizer step mutates the shared tensor -- and therefore the whole-module
        # digest -- underneath its still-unapplied siblings, whose gradients are perfectly valid
        # for the revision they were computed at. Comparing to the live digest rejected them and
        # made shared-parameter training impossible. A foreign accumulator still fails: its base
        # digest matches neither pin nor module. Without a pin (estimate called directly) the live
        # digest remains the reference.
        expected = self._pinned_digest if self._pinned_digest is not None else _model_digest(self.module)
        if state.base_sha256 != expected:
            raise ValueError("streaming gradients were not computed from the estimator's current model revision")
        named = dict(self.module.named_parameters())
        trainable_names = tuple(name for name, parameter in named.items() if parameter.requires_grad)
        if state.parameter_names != trainable_names or len(state.gradient_sums) != len(trainable_names):
            raise ValueError("streaming gradient parameter schema does not match the estimator")
        self.optimizer.zero_grad()
        for name, gradient in zip(state.parameter_names, state.gradient_sums):
            parameter = named[name]
            array = np.asarray(gradient, dtype=np.float64)
            if array.shape != tuple(parameter.shape) or not np.all(np.isfinite(array)):
                raise ValueError(f"invalid aggregated gradient for parameter {name!r}")
            parameter.grad = torch.as_tensor(
                array / state.effective_weight,
                dtype=parameter.dtype,
                device=parameter.device,
            )
        self.optimizer.step()
        if any(
            not bool(torch.all(torch.isfinite(parameter)).detach().cpu().item())
            for parameter in self.module.parameters()
        ):
            raise RuntimeError("streaming optimizer produced non-finite parameters")
        optimizer_state = _cpu_optimizer_state(self.optimizer.state_dict())
        return StreamingTransformer(
            self.module,
            self.device,
            lr=self.lr,
            optimizer_state=optimizer_state,
        )


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
    """Train one persistent-optimizer step per streamed micro-batch without buffering the corpus."""
    report_every = _positive_int(report_every, "report_every")
    if log is not None and not callable(log):
        raise TypeError("log must be callable or None")
    try:
        source = iter(token_source)
    except TypeError as exc:
        raise TypeError("token_source must be iterable") from exc
    est = StreamingTransformerEstimator(module, lr=lr, device=device)
    step = 0
    window_sum = 0.0
    window_weight = 0.0
    total_loss_sum = 0.0
    total_weight = 0.0
    leaf: StreamingTransformer | None = None
    for batch in source:
        acc = est.accumulator_factory().make()
        acc.seq_update(batch, None, None)
        state = acc.value()
        leaf = est.estimate(None, state)
        step += 1
        window_sum += state.loss_sum
        window_weight += state.effective_weight
        total_loss_sum += state.loss_sum
        total_weight += state.effective_weight
        if log is not None and step % report_every == 0:
            log(step, window_sum / window_weight)
            window_sum, window_weight = 0.0, 0.0
    if leaf is None:
        raise ValueError("token_source yielded no training batches")
    return leaf, (total_loss_sum, total_weight)


_MAX_OPTIMIZER_STATE_BYTES = 512 * 1024 * 1024
_OPTIMIZER_FORMAT = "torch-optimizer-state/v1"
_OPTIMIZER_FIELDS = frozenset({"__optimizer_state__", "format", "decoded_bytes", "sha256"})


def _training_batch(
    enc: Any,
    weights: Any,
    module: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(enc, (tuple, list)) or len(enc) != 2:
        raise ValueError("streaming data must be a (contexts, next_tokens) pair")
    contexts = _contexts(enc[0])
    if module is not None:
        _validate_model_contexts(contexts, module)
    actions = _exact_actions(enc[1], len(contexts))
    if weights is None:
        sample_weights = np.ones(len(contexts), dtype=np.float64)
    else:
        try:
            sample_weights = np.asarray(weights, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("streaming weights must be numeric") from exc
        if sample_weights.ndim != 1 or len(sample_weights) != len(contexts):
            raise ValueError(f"streaming weights must be a one-dimensional vector with {len(contexts)} rows")
        if not np.all(np.isfinite(sample_weights)) or np.any(sample_weights < 0.0):
            raise ValueError("streaming weights must contain only finite, non-negative values")
        if not np.any(sample_weights > 0.0):
            raise ValueError("streaming weights must contain positive effective weight")
    return contexts, actions, sample_weights


def _contexts(value: Any) -> np.ndarray:
    try:
        contexts = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("contexts must form a numeric two-dimensional matrix") from exc
    if contexts.ndim == 1:
        contexts = np.atleast_2d(contexts)
    if contexts.ndim != 2 or contexts.shape[0] == 0 or contexts.shape[1] == 0:
        raise ValueError("contexts must have non-empty shape (batch, sequence)")
    if not np.all(np.isfinite(contexts)):
        raise ValueError("contexts must contain only finite values")
    return contexts


def _validate_model_contexts(contexts: np.ndarray, module: Any) -> None:
    if hasattr(module, "block") and contexts.shape[1] > int(module.block):
        raise ValueError(f"context sequence length {contexts.shape[1]} exceeds configured block size {module.block}")
    if hasattr(module, "vocab"):
        if not np.all(contexts == np.round(contexts)):
            raise ValueError("token contexts must contain integer-valued entries")
        if np.any(contexts < 0) or np.any(contexts >= int(module.vocab)):
            raise ValueError(f"context token ids must lie in [0, {module.vocab})")


def _exact_actions(value: Any, n_rows: int) -> np.ndarray:
    actions = np.asarray(value)
    if actions.ndim != 1 or len(actions) != n_rows:
        raise ValueError(f"next-token actions must be a one-dimensional vector with {n_rows} rows")
    if actions.dtype.kind not in {"i", "u", "f"}:
        raise ValueError("next-token actions must be numeric integers")
    if actions.dtype.kind == "f" and (not np.all(np.isfinite(actions)) or not np.all(actions == np.round(actions))):
        raise ValueError("next-token actions must be finite integer values")
    if np.any(actions < 0) or np.any(actions > np.iinfo(np.intp).max):
        raise ValueError("next-token actions must be non-negative supported integer indices")
    return actions.astype(int, copy=False)


def _module_dtype(module: Any, torch: Any) -> Any:
    return next(
        (parameter.dtype for parameter in module.parameters() if parameter.dtype.is_floating_point),
        torch.float32,
    )


def _validate_logits(logits: Any, n_rows: int, torch: Any) -> None:
    if logits.ndim != 2 or logits.shape[0] != n_rows or logits.shape[1] < 2:
        raise ValueError("streaming module must return shape (batch, actions) with at least two actions")
    if not bool(torch.all(torch.isfinite(logits)).detach().cpu().item()):
        raise ValueError("streaming module returned non-finite logits")


def _gradient_state(value: Any) -> StreamingGradientState:
    if not isinstance(value, StreamingGradientState):
        raise TypeError("streaming sufficient statistics must be a StreamingGradientState")
    if (
        not isinstance(value.base_sha256, str)
        or len(value.base_sha256) != 64
        or not value.parameter_names
        or len(set(value.parameter_names)) != len(value.parameter_names)
        or len(value.gradient_sums) != len(value.parameter_names)
    ):
        raise ValueError("streaming gradient state has invalid identity or parameter schema")
    try:
        int(value.base_sha256, 16)
    except ValueError as exc:
        raise ValueError("streaming gradient base_sha256 must be hexadecimal") from exc
    if any(not isinstance(name, str) or not name for name in value.parameter_names):
        raise ValueError("streaming gradient parameter names must be non-empty strings")
    if (
        not np.isfinite(value.loss_sum)
        or value.loss_sum < 0.0
        or not np.isfinite(value.effective_weight)
        or value.effective_weight <= 0.0
    ):
        raise ValueError("streaming gradient telemetry must be finite with positive effective weight")
    for count, name in ((value.rows, "rows"), (value.batches, "batches")):
        if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or count <= 0:
            raise ValueError(f"streaming gradient {name} must be a positive integer")
    for gradient in value.gradient_sums:
        array = np.asarray(gradient)
        if not np.all(np.isfinite(array)):
            raise ValueError("streaming gradient state contains non-finite values")
    return value


def _model_digest(module: Any) -> str:
    torch = _torch()
    digest = hashlib.sha256()
    digest.update(f"{type(module).__module__}.{type(module).__qualname__}".encode())
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _cpu_optimizer_state(value: Any) -> Any:
    torch = _torch()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_optimizer_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_optimizer_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_optimizer_state(item) for item in value)
    return value


def _encode_optimizer_state(state: dict[str, Any] | None) -> dict[str, Any] | None:
    if state is None:
        return None
    torch = _torch()
    buffer = io.BytesIO()
    torch.save(_cpu_optimizer_state(state), buffer)
    data = buffer.getvalue()
    if not data or len(data) > _MAX_OPTIMIZER_STATE_BYTES:
        raise ValueError(f"optimizer state must contain 1 through {_MAX_OPTIMIZER_STATE_BYTES} decoded bytes")
    return {
        "__optimizer_state__": base64.b64encode(data).decode("ascii"),
        "format": _OPTIMIZER_FORMAT,
        "decoded_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _decode_optimizer_state(payload: Any) -> dict[str, Any] | None:
    if payload is None:
        return None
    _require_trusted_deserialization()
    if not isinstance(payload, Mapping) or set(payload) != _OPTIMIZER_FIELDS:
        raise _serialization_error("optimizer-state payload has invalid fields")
    if payload["format"] != _OPTIMIZER_FORMAT:
        raise _serialization_error("optimizer-state payload has an unsupported format")
    decoded_bytes = payload["decoded_bytes"]
    if type(decoded_bytes) is not int or not 0 < decoded_bytes <= _MAX_OPTIMIZER_STATE_BYTES:
        raise _serialization_error("optimizer-state decoded_bytes is outside the supported bound")
    encoded = payload["__optimizer_state__"]
    max_encoded = 4 * ((_MAX_OPTIMIZER_STATE_BYTES + 2) // 3)
    if not isinstance(encoded, str) or not encoded or len(encoded) > max_encoded:
        raise _serialization_error("optimizer state must be a bounded ASCII base64 string")
    try:
        data = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError, binascii.Error) as exc:
        raise _serialization_error("optimizer state is not strict ASCII base64") from exc
    if len(data) != decoded_bytes:
        raise _serialization_error("optimizer-state decoded length does not match decoded_bytes")
    expected_digest = payload["sha256"]
    if not isinstance(expected_digest, str) or len(expected_digest) != 64:
        raise _serialization_error("optimizer-state sha256 must be a 64-character digest")
    try:
        int(expected_digest, 16)
    except ValueError as exc:
        raise _serialization_error("optimizer-state sha256 must be hexadecimal") from exc
    if not hmac.compare_digest(hashlib.sha256(data).hexdigest(), expected_digest.lower()):
        raise _serialization_error("optimizer-state sha256 does not match decoded bytes")
    torch = _torch()
    try:
        state = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise _serialization_error("optimizer state could not be decoded safely") from exc
    if not isinstance(state, dict) or set(state) != {"state", "param_groups"}:
        raise _serialization_error("optimizer state must contain exactly state and param_groups")
    return state


def _positive_finite(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real scalar, not a boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real scalar") from exc
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


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
