"""A neural classifier as a Mixle conditional-density leaf: ``p(y | x) = softmax(module(x))``.

The discriminative sibling of :class:`~mixle.models.neural_leaf.NeuralGaussian`. ``NeuralCategorical(module)`` wraps
a Torch module that emits ``k`` logits as a mixle distribution over observations ``(x, y)`` with ``y`` an integer
class index. It implements the full ``SequenceEncodableProbabilityDistribution`` contract, so it drops into
``MixtureDistribution`` / ``CompositeDistribution`` / HMM emissions like any leaf -- and its EM **M-step is a
responsibility-weighted cross-entropy gradient step** on the module (warm-started across EM iterations =>
generalized EM). The model's ``seq_log_density`` IS ``-cross_entropy(module(x), y)``: the objective is the
leaf's log-density, never a user-supplied loss closure.

This is the leaf that the declarative ``Categorical(logits=Net(...))`` PPL slot lowers to, and the component
that makes a ``Mix([Categorical(logits=Net(...)), ...])`` a mixture of neural classifiers fit by ordinary EM.

Requires torch. The leaf is conditional: ``predict(x)`` and ``sampler().sample_given(x)`` work; ``sample()`` raises
because the model has no marginal ``p(x)``. This is the same conditional contract used by ``NeuralGaussian`` and
``RandomForestConditional``.
"""

from __future__ import annotations

from numbers import Real
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


def _torch() -> Any:
    import torch

    return torch


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    if logits.ndim != 2 or logits.shape[1] == 0:
        raise ValueError("categorical logits must have shape (rows, classes) with at least one class")
    if not np.all(np.isfinite(logits)):
        raise ValueError("categorical logits must contain only finite values")
    if logits.shape[0] == 0:
        return logits.copy()
    m = logits.max(axis=1, keepdims=True)
    return logits - m - np.log(np.exp(logits - m).sum(axis=1, keepdims=True))


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
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


def _features(value: Any, where: str, *, single: bool = False) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if single:
        if array.ndim == 0:
            array = array.reshape(1)
        array = array[None, ...]
    elif array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim < 2 or any(size == 0 for size in array.shape[1:]):
        raise ValueError(f"{where} features must have shape (rows, ...non-empty feature axes)")
    return check_finite(array, where)


def _labels(value: Any, rows: int, where: str, *, num_classes: int | None = None) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or array.shape != (rows,):
        raise ValueError(f"{where} labels must be one-dimensional with exactly {rows} entries")
    if array.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{where} labels must contain exact integer class indices")
    if np.any(array < 0) or np.any(array > np.iinfo(np.intp).max):
        raise ValueError(f"{where} labels must contain supported non-negative class indices")
    result = array.astype(np.int64, copy=False)
    if num_classes is not None and np.any(result >= num_classes):
        raise ValueError(f"{where} labels must be in [0, {num_classes})")
    return result


def _weights(value: Any, rows: int, where: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or array.shape != (rows,):
        raise ValueError(f"{where} weights must be one-dimensional with exactly {rows} entries")
    check_finite(array, where)
    if np.any(array < 0.0):
        raise ValueError(f"{where} weights must be non-negative")
    return array


def _validate_ewc(value: Any, module: Any) -> Any:
    if value is None:
        return None
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError("ewc must be an (anchor, fisher, lambda) triple")
    anchor, fisher, lam = value
    if isinstance(lam, (bool, np.bool_)) or not isinstance(lam, Real):
        raise TypeError("EWC lambda must be a finite non-negative real number")
    lam = float(lam)
    if not np.isfinite(lam) or lam < 0.0:
        raise ValueError("EWC lambda must be finite and non-negative")
    parameters = list(module.named_parameters())
    if hasattr(anchor, "names") or hasattr(fisher, "names"):
        expected_names = tuple(name for name, _ in parameters)
        if getattr(anchor, "names", None) != expected_names or getattr(fisher, "names", None) != expected_names:
            raise ValueError("EWC parameter identity manifests must exactly match the module")
    try:
        anchors = list(anchor)
        fishers = list(fisher)
    except TypeError as exc:
        raise TypeError("EWC anchors and Fisher values must be sequences") from exc
    if len(anchors) != len(parameters) or len(fishers) != len(parameters):
        raise ValueError("EWC anchors and Fisher values must cover every module parameter exactly once")
    torch = _torch()
    validated_anchors = []
    validated_fishers = []
    for (name, parameter), anchor_value, fisher_value in zip(parameters, anchors, fishers):
        if not isinstance(anchor_value, torch.Tensor) or not isinstance(fisher_value, torch.Tensor):
            raise TypeError(f"EWC parameter {name!r} anchor and Fisher values must be tensors")
        if anchor_value.shape != parameter.shape or fisher_value.shape != parameter.shape:
            raise ValueError(f"EWC parameter {name!r} shape must match {tuple(parameter.shape)}")
        if not bool(torch.isfinite(anchor_value).all()) or not bool(torch.isfinite(fisher_value).all()):
            raise ValueError(f"EWC parameter {name!r} must contain finite values")
        if bool(torch.any(fisher_value < 0.0)):
            raise ValueError(f"EWC Fisher values for parameter {name!r} must be non-negative")
        validated_anchors.append(anchor_value.detach().clone())
        validated_fishers.append(fisher_value.detach().clone())
    return validated_anchors, validated_fishers, lam


class NeuralCategorical(SequenceEncodableProbabilityDistribution):
    """``p(y | x) = softmax(module(x))`` as a mixle leaf. Observation is the pair ``(x, y)``, ``y`` an int class.

    ``batch_size`` (None = full batch) makes the M-step minibatch SGD over ``m_steps`` passes -- needed to train a
    real conv net on a large image set; ``max_optimizer_steps`` optionally caps updates independently of batch
    size, and ``device`` (e.g. ``"mps"``/``"cuda"``) runs them on the GPU.
    """

    __pysp_serializable__ = True  # module persisted as bytes (see __pysp_getstate__); leaf round-trips in a mixture

    def __init__(
        self,
        module: Any,
        m_steps: int = 40,
        lr: float = 0.01,
        name: str | None = None,
        batch_size: int | None = None,
        device: str = "cpu",
        max_optimizer_steps: int | None = None,
        optimizer_state: dict[str, Any] | None = None,
    ) -> None:
        torch = _torch()
        if not isinstance(module, torch.nn.Module):
            raise TypeError("module must be a torch.nn.Module")
        self.module = module
        self.m_steps = _positive_int(m_steps, "m_steps")
        self.lr = _positive_finite(lr, "lr")
        self.name = name
        self.batch_size = None if batch_size is None else _positive_int(batch_size, "batch_size")
        try:
            self.device = str(torch.device(device))
        except (TypeError, RuntimeError) as exc:
            raise ValueError(f"invalid torch device {device!r}") from exc
        self.max_optimizer_steps = (
            None if max_optimizer_steps is None else _positive_int(max_optimizer_steps, "max_optimizer_steps")
        )
        if optimizer_state is not None and not isinstance(optimizer_state, dict):
            raise TypeError("optimizer_state must be a dictionary or None")
        self.optimizer_state = optimizer_state

    def __str__(self) -> str:
        return "NeuralCategorical()"

    def _logits(self, x: np.ndarray) -> np.ndarray:
        torch = _torch()
        x = _features(x, "NeuralCategorical._logits")
        self.module.to(self.device)
        out = []
        with _module_mode(self.module, train=False), torch.no_grad():
            xt = torch.as_tensor(x, dtype=torch.float32)
            ranges = (
                [(0, 0)] if xt.shape[0] == 0 else [(k, min(k + 4096, xt.shape[0])) for k in range(0, xt.shape[0], 4096)]
            )
            for start, stop in ranges:
                value = self.module(xt[start:stop].to(self.device))
                if not isinstance(value, torch.Tensor):
                    raise TypeError("NeuralCategorical module must return a torch.Tensor")
                result = value.detach().cpu().numpy()
                if result.ndim != 2 or result.shape[0] != stop - start or result.shape[1] == 0:
                    raise ValueError(
                        "NeuralCategorical module output must have shape (rows, classes); "
                        f"got input rows={stop - start}, output shape={result.shape}"
                    )
                check_finite(result, "NeuralCategorical module output")
                out.append(result)
        widths = {value.shape[1] for value in out}
        if len(widths) != 1:
            raise ValueError("NeuralCategorical module output width changed between chunks")
        return np.concatenate(out, axis=0)

    def log_density(self, xy: Any) -> float:
        """Return ``log p(y | x)`` for one feature/class observation pair."""
        if not isinstance(xy, (tuple, list)) or len(xy) != 2:
            raise ValueError("NeuralCategorical.log_density expects an (x, label) pair")
        x = _features(xy[0], "NeuralCategorical.log_density", single=True)
        y = _labels([xy[1]], 1, "NeuralCategorical.log_density")
        return float(self.seq_log_density((x, y))[0])

    def seq_log_density(self, enc: Any) -> np.ndarray:
        """Return per-row categorical conditional log probabilities for encoded pairs."""
        if not isinstance(enc, (tuple, list)) or len(enc) != 2:
            raise ValueError("NeuralCategorical.seq_log_density expects an (x, labels) pair")
        x = _features(enc[0], "NeuralCategorical.seq_log_density")
        logp = _log_softmax(self._logits(x))
        y = _labels(enc[1], x.shape[0], "NeuralCategorical.seq_log_density", num_classes=logp.shape[1])
        return logp[np.arange(len(y)), y]

    def predict(self, x: Any) -> np.ndarray:
        """Return maximum-probability class predictions for one or more inputs."""
        raw = np.asarray(x)
        batch = _features(x, "NeuralCategorical.predict", single=raw.ndim < 2)
        if batch.shape[0] == 0:
            raise ValueError("NeuralCategorical.predict requires at least one row")
        p = self._logits(batch).argmax(axis=1)
        return int(p[0]) if raw.ndim < 2 else p

    def sampler(self, seed: int | None = None) -> NeuralCategoricalSampler:
        """Return a conditional sampler over labels given features."""
        return NeuralCategoricalSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> NeuralCategoricalEstimator:
        """Return the generalized-EM estimator for weighted cross-entropy training."""
        return NeuralCategoricalEstimator(
            self.module,
            self.m_steps,
            self.lr,
            self.name,
            self.batch_size,
            self.device,
            max_optimizer_steps=self.max_optimizer_steps,
            optimizer_state=self.optimizer_state,
        )

    def dist_to_encoder(self) -> NeuralCategoricalEncoder:
        """Return the encoder for ``(x, class)`` observation pairs."""
        return NeuralCategoricalEncoder()

    # --- serialization: persist hparams + the module (as portable bytes); registered below so a mixture holding
    # this leaf round-trips through to_dict/to_json/pickle as well. ---
    def __pysp_getstate__(self) -> dict[str, Any]:
        from mixle.models.streaming_transformer_leaf import _encode_optimizer_state

        state = dict(self.__dict__)
        state["module"] = encode_module(self.module)
        state["optimizer_state"] = _encode_optimizer_state(self.optimizer_state)
        return state

    def __pysp_setstate__(self, state: dict[str, Any]) -> None:
        from mixle.models.streaming_transformer_leaf import _decode_optimizer_state

        self.__dict__.update(state)
        self.module = decode_module(state["module"])
        self.optimizer_state = _decode_optimizer_state(state.get("optimizer_state"))

    def to_dict(self) -> dict[str, Any]:
        """Serialize hyperparameters and module bytes for registry-based round trips."""
        from mixle.models.streaming_transformer_leaf import _encode_optimizer_state

        return {
            "m_steps": self.m_steps,
            "lr": self.lr,
            "name": self.name,
            "batch_size": self.batch_size,
            "device": self.device,
            "max_optimizer_steps": self.max_optimizer_steps,
            "optimizer_state": _encode_optimizer_state(self.optimizer_state),
            "module": encode_module(self.module),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> NeuralCategorical:
        """Rebuild a :class:`NeuralCategorical` from :meth:`to_dict` output."""
        from mixle.models.streaming_transformer_leaf import _decode_optimizer_state

        return cls(
            decode_module(payload["module"]),
            m_steps=payload["m_steps"],
            lr=payload["lr"],
            name=payload["name"],
            batch_size=payload["batch_size"],
            device=payload["device"],
            max_optimizer_steps=payload.get("max_optimizer_steps"),
            optimizer_state=_decode_optimizer_state(payload.get("optimizer_state")),
        )


class NeuralCategoricalSampler(DistributionSampler):
    """Conditional sampler over class labels for :class:`NeuralCategorical`."""

    def __init__(self, dist: NeuralCategorical, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = np.random.RandomState(seed)

    def sample(self, size: int | None = None, *, batched: bool = True) -> Any:
        """Raise because the leaf defines ``p(y | x)`` and has no marginal ``p(x)``."""
        raise NotImplementedError("NeuralCategorical is conditional p(y|x); use sampler().sample_given(x).")

    def sample_given(self, x: Any) -> int | np.ndarray:
        """Draw class labels from ``p(y | x)``, preserving a supplied batch axis."""
        raw = np.asarray(x)
        batch = _features(x, "NeuralCategoricalSampler.sample_given", single=raw.ndim < 2)
        logp = _log_softmax(self.dist._logits(batch))
        samples = np.asarray(
            [self.rng.choice(logp.shape[1], p=np.exp(row)) for row in logp],
            dtype=np.int64,
        )
        return int(samples[0]) if raw.ndim < 2 else samples


class NeuralCategoricalEncoder(DataSequenceEncoder):
    """Encode feature/class pairs for neural-categorical scoring and fitting."""

    def __str__(self) -> str:
        return "NeuralCategoricalEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, NeuralCategoricalEncoder)

    def seq_encode(self, data: list) -> tuple[np.ndarray, np.ndarray]:
        """Convert ``(x, class)`` pairs into batched feature and integer-label arrays."""
        if not isinstance(data, list):
            raise TypeError("NeuralCategoricalEncoder.seq_encode requires a list")
        if not data:
            return np.zeros((0, 0), dtype=float), np.zeros((0,), dtype=np.int64)
        if any(not isinstance(item, (tuple, list)) or len(item) != 2 for item in data):
            raise ValueError("each neural-categorical observation must be an (x, label) pair")
        features = [np.atleast_1d(np.asarray(item[0], dtype=float)) for item in data]
        if any(value.size == 0 for value in features):
            raise ValueError("categorical features must have non-empty feature axes")
        try:
            x = np.stack(features)
        except ValueError as exc:
            raise ValueError("categorical features must all have the same shape") from exc
        check_finite(x, "NeuralCategoricalEncoder.seq_encode")
        y = _labels([item[1] for item in data], len(data), "NeuralCategoricalEncoder.seq_encode")
        return (x, y)


class NeuralCategoricalAccumulator(SequenceEncodableStatisticAccumulator):
    """Buffer weighted feature/class batches for the neural-categorical M-step."""

    def __init__(self) -> None:
        self.x: list = []
        self.y: list = []
        self.w: list = []

    # x/y/w hold contiguous batch arrays and concatenate once at value(), avoiding per-row ndarray buffering.
    # x batching is shape-preserving so conv/structured inputs survive; y stays an integer class index.
    def update(self, xy: Any, weight: float, estimate: Any) -> None:
        """Add one weighted feature/class pair to the accumulator."""
        if not isinstance(xy, (tuple, list)) or len(xy) != 2:
            raise ValueError("NeuralCategoricalAccumulator.update expects an (x, label) pair")
        self.x.append(_features(xy[0], "NeuralCategoricalAccumulator.update", single=True))
        self.y.append(_labels([xy[1]], 1, "NeuralCategoricalAccumulator.update"))
        self.w.append(_weights([weight], 1, "NeuralCategoricalAccumulator.update"))

    def seq_update(self, enc: Any, weights: np.ndarray, estimate: Any) -> None:
        """Add an encoded batch and responsibility weights to the accumulator."""
        if not isinstance(enc, (tuple, list)) or len(enc) != 2:
            raise ValueError("NeuralCategoricalAccumulator.seq_update expects an (x, labels) pair")
        x = _features(enc[0], "NeuralCategoricalAccumulator.seq_update")
        self.x.append(x)
        self.y.append(_labels(enc[1], x.shape[0], "NeuralCategoricalAccumulator.seq_update"))
        self.w.append(_weights(weights, x.shape[0], "NeuralCategoricalAccumulator.seq_update"))

    def initialize(self, xy: Any, weight: float, rng: Any) -> None:
        """Initialize from one observation using the ordinary update path."""
        self.update(xy, weight, None)

    def seq_initialize(self, enc: Any, weights: np.ndarray, rng: Any) -> None:
        """Initialize from an encoded batch using the ordinary batch update path."""
        self.seq_update(enc, weights, None)

    def combine(self, other: Any) -> NeuralCategoricalAccumulator:
        """Merge the value tuple from another categorical accumulator."""
        if not isinstance(other, (tuple, list)) or len(other) != 3:
            raise ValueError("categorical accumulator value must be an (x, labels, weights) triple")
        xo, yo, wo = other
        if len(xo):
            x = _features(xo, "NeuralCategoricalAccumulator.combine")
            self.x.append(x)
            self.y.append(_labels(yo, x.shape[0], "NeuralCategoricalAccumulator.combine"))
            self.w.append(_weights(wo, x.shape[0], "NeuralCategoricalAccumulator.combine"))
        elif len(yo) or len(wo):
            raise ValueError("empty categorical sufficient statistics require empty x, labels, and weights")
        return self

    def value(self) -> tuple:
        """Return contiguous ``(x, class, weights)`` arrays for the M-step."""
        if self.x and any(value.shape[1:] != self.x[0].shape[1:] for value in self.x[1:]):
            raise ValueError("categorical accumulator feature batches have incompatible shapes")
        x = np.concatenate(self.x, axis=0) if self.x else np.zeros((0, 0))
        y = np.concatenate(self.y) if self.y else np.zeros((0,), dtype=int)
        w = np.concatenate(self.w) if self.w else np.zeros((0,))
        return (x, y, w)

    def from_value(self, value: tuple) -> NeuralCategoricalAccumulator:
        """Restore accumulator buffers from a value tuple."""
        self.x = []
        self.y = []
        self.w = []
        self.combine(value)
        return self

    def acc_to_encoder(self) -> NeuralCategoricalEncoder:
        """Return the encoder expected by this accumulator."""
        return NeuralCategoricalEncoder()


class NeuralCategoricalAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for neural-categorical accumulators."""

    def make(self) -> NeuralCategoricalAccumulator:
        """Create a fresh accumulator."""
        return NeuralCategoricalAccumulator()


class NeuralCategoricalEstimator(ParameterEstimator):
    """EM estimator for a :class:`NeuralCategorical`: the M-step is ``m_steps`` of responsibility-weighted
    cross-entropy gradient on the module (the module is warm-started across EM iterations => generalized EM).

    A minibatch of size ``B`` uses ``N/B * sum_batch(w * CE) / sum_all(w)``. Its gradient is an unbiased
    estimate of the full responsibility-normalized objective, including when responsibility mass is unevenly
    distributed across batches. ``max_optimizer_steps`` gives batch-size comparisons a fixed update budget.
    """

    def __init__(
        self,
        module: Any,
        m_steps: int = 40,
        lr: float = 0.01,
        name: str | None = None,
        batch_size: int | None = None,
        device: str = "cpu",
        ewc: Any = None,
        max_optimizer_steps: int | None = None,
        optimizer_state: dict[str, Any] | None = None,
    ) -> None:
        torch = _torch()
        if not isinstance(module, torch.nn.Module):
            raise TypeError("module must be a torch.nn.Module")
        self.module = module
        self.m_steps = _positive_int(m_steps, "m_steps")
        self.lr = _positive_finite(lr, "lr")
        self.name = name
        self.batch_size = None if batch_size is None else _positive_int(batch_size, "batch_size")
        try:
            self.device = str(torch.device(device))
        except (TypeError, RuntimeError) as exc:
            raise ValueError(f"invalid torch device {device!r}") from exc
        # ewc = (anchor_params, fisher_diag, lambda): the EWC anti-forgetting penalty for continued pretraining
        self.ewc = _validate_ewc(ewc, module)
        self.max_optimizer_steps = (
            None if max_optimizer_steps is None else _positive_int(max_optimizer_steps, "max_optimizer_steps")
        )
        if optimizer_state is not None and not isinstance(optimizer_state, dict):
            raise TypeError("optimizer_state must be a dictionary or None")
        self.optimizer_state = optimizer_state

    def accumulator_factory(self) -> NeuralCategoricalAccumulatorFactory:
        """Return an accumulator factory for weighted classification batches."""
        return NeuralCategoricalAccumulatorFactory()

    def estimate(self, nobs: float | None, suff_stat: tuple) -> NeuralCategorical:
        """Run the weighted cross-entropy M-step and return the updated leaf."""
        torch = _torch()
        xs, ys, ws = suff_stat
        out = NeuralCategorical(
            self.module,
            self.m_steps,
            self.lr,
            self.name,
            self.batch_size,
            self.device,
            self.max_optimizer_steps,
            optimizer_state=self.optimizer_state,
        )
        if len(xs) == 0:
            if len(ys) or len(ws):
                raise ValueError("empty categorical sufficient statistics require empty x, labels, and weights")
            return out
        xs = _features(xs, "NeuralCategoricalEstimator.estimate")
        ys = _labels(ys, xs.shape[0], "NeuralCategoricalEstimator.estimate")
        ws = _weights(ws, xs.shape[0], "NeuralCategoricalEstimator.estimate")
        if not np.any(ws > 0.0):
            raise ValueError("NeuralCategoricalEstimator.estimate requires positive effective weight")
        dev = self.device
        self.module.to(dev)
        # data stays on CPU (a large image set won't fit on the GPU); each minibatch is moved to the device.
        # x arrives shape-preserving from the buffer so conv/structured inputs survive; the generic buffer
        # stores labels as a (n, 1) float64 column -- integral class indices cast to long exactly.
        xt = torch.as_tensor(xs, dtype=torch.float32)
        yt = torch.as_tensor(ys, dtype=torch.long)
        wt = torch.as_tensor(ws, dtype=torch.float32)
        n = xt.shape[0]
        bs = self.batch_size or n
        from mixle.models.optimizer_routing import resolve_neural_optimizer

        opt, optimizer_receipt = resolve_neural_optimizer(self.module, lr=self.lr, sign_stable=bs >= n)
        if self.optimizer_state is not None:
            try:
                opt.load_state_dict(self.optimizer_state)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("optimizer_state is incompatible with the categorical model") from exc
        ce = torch.nn.CrossEntropyLoss(reduction="none")
        total_weight = wt.sum()
        optimizer_steps = 0
        epochs_completed = 0
        ewc = None
        if self.ewc is not None:  # anchor + Fisher moved to the device once (continued-pretraining anti-forget)
            anchor, fisher, lam = self.ewc
            ewc = ([a.to(dev) for a in anchor], [f.to(dev) for f in fisher], float(lam))
        with _module_mode(self.module, train=True):
            for _ in range(self.m_steps):  # m_steps passes over the data (full-batch when batch_size is None)
                perm = torch.randperm(n) if bs < n else torch.arange(n)
                for k in range(0, n, bs):
                    if self.max_optimizer_steps is not None and optimizer_steps >= self.max_optimizer_steps:
                        break
                    idx = perm[k : k + bs]
                    xb, yb, wb = xt[idx].to(dev), yt[idx].to(dev), wt[idx].to(dev)
                    opt.zero_grad()
                    logits = self.module(xb)
                    if not isinstance(logits, torch.Tensor) or logits.ndim != 2 or logits.shape[0] != len(idx):
                        actual_shape = getattr(logits, "shape", None)
                        raise ValueError(
                            "NeuralCategorical module output must have shape (rows, classes); "
                            f"got rows={len(idx)}, output={actual_shape}"
                        )
                    if logits.shape[1] == 0 or not bool(torch.isfinite(logits).all()):
                        raise ValueError("NeuralCategorical module logits must be finite with at least one class")
                    if bool(torch.any(yb < 0)) or bool(torch.any(yb >= logits.shape[1])):
                        raise ValueError(f"categorical labels must be in [0, {logits.shape[1]})")
                    # Uniform minibatches estimate the full responsibility-normalized objective.
                    batch_scale = float(n) / float(len(idx))
                    loss = batch_scale * (wb * ce(logits, yb)).sum() / total_weight.to(dev)
                    if ewc is not None:  # + lambda * sum_i F_i (theta_i - theta*_i)^2 -- pull important weights back
                        anchor, fisher, lam = ewc
                        loss = loss + lam * sum(
                            (f * (p - a) ** 2).sum() for p, a, f in zip(self.module.parameters(), anchor, fisher)
                        )
                    if not bool(torch.isfinite(loss)):
                        raise ValueError("categorical weighted objective became non-finite")
                    loss.backward()
                    for parameter in self.module.parameters():
                        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                            raise ValueError("categorical optimization produced a non-finite gradient")
                    opt.step()
                    for parameter in self.module.parameters():
                        if not bool(torch.isfinite(parameter).all()):
                            raise ValueError("categorical optimization produced a non-finite parameter")
                    optimizer_steps += 1
                else:
                    epochs_completed += 1
                    continue
                break
        from mixle.models.streaming_transformer_leaf import _cpu_optimizer_state

        self.optimizer_state = _cpu_optimizer_state(opt.state_dict())
        out.optimizer_state = self.optimizer_state
        out.fit_receipt = {
            "nobs": int(n),
            "batch_size": int(min(bs, n)),
            "epochs_requested": self.m_steps,
            "epochs_completed": epochs_completed,
            "optimizer_steps": optimizer_steps,
            "max_optimizer_steps": self.max_optimizer_steps,
            "gradient_estimator": "N/B responsibility-weighted cross-entropy",
            "optimizer": optimizer_receipt["name"],
            "optimizer_plan": optimizer_receipt["plan"],
        }
        return out


def _register_serializable() -> None:
    # mixle.models classes aren't in the stats/analysis auto-walk, so opt in explicitly for to_json/from_json.
    try:
        from mixle.utils.serialization import register_serializable_class
    except Exception:  # pragma: no cover  # noqa: BLE001
        return
    register_serializable_class(NeuralCategorical)


_register_serializable()


# --- back-compat aliases (the classes were renamed off the '...Leaf' suffix) ---
SoftmaxNeuralLeaf = NeuralCategorical
SoftmaxNeuralLeafEstimator = NeuralCategoricalEstimator
