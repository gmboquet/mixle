"""``GradLeaf`` -- a torch module IS the model: the five-piece contract, manufactured.

The contract (Distribution / Sampler / Estimator / Accumulator / DataEncoder) earns its keep for
closed-form families: additive sufficient statistics are what make EM exact and distributable. A
GRADIENT leaf has no sufficient statistics -- its "accumulator" can only buffer the
responsibility-weighted data, its encoder is ``np.asarray``, and its M-step is SGD -- so per-family
contract code is pure ceremony (mixle.models grew nine hand-written buffer accumulators saying so).
This module writes that ceremony ONCE, generically. A neural family is now just a module:

    fitted = optimize(x, module)                     # a bare nn.Module coerces -- no wrapper at all
    leaf = GradLeaf(module)                          # or wrap explicitly to set knobs/hooks ...
    mix = MixtureDistribution([leaf, gamma], w)      # ... and compose with classical families

The module owns **forward and objective**; mixle owns the loop. The contract's requirements on the
module are two methods: ``log_density(x) -> (n,)`` (scoring; also the default M-step objective) and,
only if you draw samples, ``sample(n) -> (n, d)``. Control never leaves the caller:

* ``loss(module, x, w) -> scalar`` overrides the default responsibility-weighted NLL -- custom
  objectives are a hook, not a subclass tree;
* ``optimizer(params) -> torch.optim.Optimizer`` picks the optimizer; it receives only TRAINABLE
  parameters, so freezing submodules with ``requires_grad_(False)`` just works (train a projection
  head against a frozen encoder; a FULLY frozen module is a fixed distribution and the M-step is a
  no-op);
* ``fitted.module`` is the raw torch module -- nothing is trapped.

Serialization: the module round-trips as portable bytes (``mixle.models._neural_serial``); custom
``loss``/``optimizer`` hooks must be module-level functions to survive pickling, like any hook.
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

__all__ = ["DataBufferAccumulator", "DataBufferAccumulatorFactory", "GradEstimator", "GradLeaf"]


def _torch() -> Any:
    import torch

    return torch


def looks_like_torch_module(obj: Any) -> bool:
    """A bare torch density module: scores batches and carries parameters -- coercible to a leaf."""
    return (
        hasattr(obj, "log_density")
        and callable(getattr(obj, "parameters", None))
        and callable(getattr(obj, "state_dict", None))
        and not isinstance(obj, (SequenceEncodableProbabilityDistribution, ParameterEstimator))
    )


class GradLeaf(SequenceEncodableProbabilityDistribution):
    """Wrap a torch density ``module`` (``module.log_density(x) -> (n,)``) as a composable mixle
    distribution (see the module docstring). ``loss`` and ``optimizer`` are the M-step hooks."""

    __pysp_serializable__ = True  # module persisted as bytes (see __pysp_getstate__)

    def __init__(
        self,
        module: Any,
        *,
        m_steps: int = 60,
        lr: float = 5e-3,
        device: str = "cpu",
        name: str | None = None,
        loss: Any = None,
        optimizer: Any = None,
    ) -> None:
        self.module = module
        self.m_steps = int(m_steps)
        self.lr = float(lr)
        self.device = device
        self.name = name
        self.loss = loss
        self.optimizer = optimizer

    def __str__(self) -> str:
        return f"{type(self).__name__}({type(self.module).__name__})"

    def log_density(self, x: Any) -> float:
        return float(self.seq_log_density(np.atleast_2d(np.asarray(x, dtype=float)))[0])

    def seq_log_density(self, x: Any) -> np.ndarray:
        torch = _torch()
        xx = check_finite(np.atleast_2d(np.asarray(x, dtype=float)), f"{type(self).__name__}.seq_log_density")
        self.module.to(self.device).eval()
        xt = torch.as_tensor(xx, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return self.module.log_density(xt).cpu().numpy().reshape(-1)

    def sampler(self, seed: int | None = None) -> GradLeafSampler:
        return GradLeafSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> GradEstimator:
        return GradEstimator(
            self.module,
            m_steps=self.m_steps,
            lr=self.lr,
            device=self.device,
            name=self.name,
            loss=self.loss,
            optimizer=self.optimizer,
        )

    def dist_to_encoder(self) -> GradLeafEncoder:
        return GradLeafEncoder()

    def __pysp_getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["module"] = encode_module(self.module)
        return state

    def __pysp_setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.module = decode_module(state["module"])


class GradLeafSampler(DistributionSampler):
    def __init__(self, dist: GradLeaf, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = np.random.RandomState(seed)

    def sample(self, size: int | None = None, *, batched: bool = True) -> Any:
        if not callable(getattr(self.dist.module, "sample", None)):
            raise TypeError(
                f"{type(self.dist.module).__name__} has no sample(n); scoring and fitting need only "
                "log_density, but drawing samples needs the module to implement sample(n) -> (n, d)."
            )
        torch = _torch()
        n = int(size or 1)
        self.dist.module.to(self.dist.device).eval()
        torch.manual_seed(int(self.rng.randint(0, 2**31 - 1)))
        with torch.no_grad():
            out = self.dist.module.sample(n).cpu().numpy()
        return out if (size is not None) else out[0]


class GradLeafEncoder(DataSequenceEncoder):
    """The whole "encoding": rows to one contiguous float array."""

    def __str__(self) -> str:
        return "GradLeafEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, GradLeafEncoder)

    def seq_encode(self, data: list) -> np.ndarray:
        return np.array([np.atleast_1d(np.asarray(x, dtype=float)) for x in data])


class DataBufferAccumulator(SequenceEncodableStatisticAccumulator):
    """THE gradient-leaf "sufficient statistic": the encoded, responsibility-weighted data itself,
    buffered for the M-step (the weights are the E-step's soft counts). Generic over the encoding
    arity -- a single array for unconditional leaves, a tuple like ``(x, y)`` for conditional ones
    -- so every gradient family shares this one class instead of hand-writing its own buffer.
    Single observations route through the family's own encoder, so per-row quirks live in exactly
    one place."""

    def __init__(self, encoder: Any, n_fields: int = 1) -> None:
        self.encoder = encoder
        self.n_fields = int(n_fields)
        self.parts: list[list] = [[] for _ in range(self.n_fields)]
        self.w: list = []

    # Contiguous batch arrays concatenated once at value() (shape-preserving) rather than one ndarray per row.
    def _append(self, enc: Any, weights: np.ndarray) -> None:
        fields = enc if isinstance(enc, tuple) else (enc,)
        for buf, f in zip(self.parts, fields):
            fb = np.asarray(f, dtype=float)
            buf.append(fb.reshape(fb.shape[0], 1) if fb.ndim == 1 else fb)
        self.w.append(np.asarray(weights, dtype=float).ravel())

    def update(self, x: Any, weight: float, estimate: Any) -> None:
        self._append(self.encoder.seq_encode([x]), np.asarray([float(weight)]))

    def seq_update(self, enc: Any, weights: np.ndarray, estimate: Any) -> None:
        self._append(enc, weights)

    def initialize(self, x: Any, weight: float, rng: Any) -> None:
        self.update(x, weight, None)

    def seq_initialize(self, enc: Any, weights: np.ndarray, rng: Any) -> None:
        self.seq_update(enc, weights, None)

    def combine(self, other: Any) -> DataBufferAccumulator:
        *fields, ws = other
        if len(ws):
            for buf, f in zip(self.parts, fields):
                buf.append(np.asarray(f, dtype=float))
            self.w.append(np.asarray(ws, dtype=float).ravel())
        return self

    def value(self) -> tuple:
        fields = tuple(np.concatenate(buf, axis=0) if buf else np.zeros((0, 0)) for buf in self.parts)
        w = np.concatenate(self.w) if self.w else np.zeros((0,))
        return (*fields, w)

    def from_value(self, v: tuple) -> DataBufferAccumulator:
        *fields, w = v
        self.parts = [[np.asarray(f, dtype=float)] if len(f) else [] for f in fields]
        self.w = [np.asarray(w, dtype=float).ravel()] if len(w) else []
        return self

    def acc_to_encoder(self) -> Any:
        return self.encoder


class DataBufferAccumulatorFactory(StatisticAccumulatorFactory):
    def __init__(self, encoder: Any, n_fields: int = 1) -> None:
        self.encoder = encoder
        self.n_fields = int(n_fields)

    def make(self) -> DataBufferAccumulator:
        return DataBufferAccumulator(self.encoder, self.n_fields)


class GradEstimator(ParameterEstimator):
    """M-step: responsibility-weighted MLE -- ``max sum_i w_i log p(x_i)`` by gradient ascent on the
    module (warm-started across EM iterations). ``loss``/``optimizer`` are the caller's hooks; the
    optimizer only ever sees trainable parameters, so frozen submodules stay frozen and a fully
    frozen module makes the M-step a no-op (a fixed distribution)."""

    def __init__(
        self,
        module: Any,
        *,
        m_steps: int = 60,
        lr: float = 5e-3,
        device: str = "cpu",
        name: str | None = None,
        loss: Any = None,
        optimizer: Any = None,
    ) -> None:
        self.module = module
        self.m_steps = int(m_steps)
        self.lr = float(lr)
        self.device = device
        self.name = name
        self.loss = loss
        self.optimizer = optimizer

    def _leaf(self) -> GradLeaf:
        return GradLeaf(
            self.module,
            m_steps=self.m_steps,
            lr=self.lr,
            device=self.device,
            name=self.name,
            loss=self.loss,
            optimizer=self.optimizer,
        )

    def accumulator_factory(self) -> DataBufferAccumulatorFactory:
        return DataBufferAccumulatorFactory(GradLeafEncoder(), n_fields=1)

    def estimate(self, nobs: float | None, suff_stat: tuple) -> GradLeaf:
        torch = _torch()
        *fields, ws = suff_stat
        xs = fields[0]
        params = [p for p in self.module.parameters() if p.requires_grad]
        if len(xs) == 0 or not params:  # nothing to fit, or a fully frozen (fixed) module
            return self._leaf()
        x = torch.as_tensor(np.asarray(xs, dtype=float), dtype=torch.float32, device=self.device)
        w = torch.as_tensor(np.asarray(ws, dtype=float), dtype=torch.float32, device=self.device)
        w = w / w.sum().clamp(min=1e-8)
        self.module.to(self.device).train()
        opt = self.optimizer(params) if self.optimizer is not None else torch.optim.Adam(params, lr=self.lr)
        for _ in range(self.m_steps):
            opt.zero_grad()
            if self.loss is not None:
                loss = self.loss(self.module, x, w)
            else:
                loss = -(w * self.module.log_density(x)).sum()  # weighted negative log-likelihood
            loss.backward()
            opt.step()
        return self._leaf()


def _register_serializable() -> None:
    try:
        from mixle.utils.serialization import register_serializable_class
    except ImportError:  # pragma: no cover - serialization is core, but never block import on it
        return
    register_serializable_class(GradLeaf)


_register_serializable()
