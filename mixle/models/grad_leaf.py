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

from contextlib import contextmanager
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


def _resolve_device(device: Any, torch: Any) -> Any:
    """Where to run a leaf's module, in priority order (shared by every gradient-fit leaf --
    ``neural_leaf.py`` imports this rather than redefining it, so the priority order is one place):

    1. an explicit ``device=`` on the leaf/estimator (always wins);
    2. the device of the **active compute engine** -- so ``optimize(engine=TorchEngine(device="mps"))``
       (or ``"cuda"``) drives the M-step onto that device, matching mixle's engine philosophy
       (set the device once on the engine, the leaf follows);
    3. otherwise CUDA if available, else CPU -- the implicit default (note: not MPS, so existing local
       CPU behaviour and tests are unchanged; reach MPS explicitly or via the engine)."""
    if device is not None:
        return torch.device(device)
    from mixle.engines.base import active_engine

    eng_dev = getattr(active_engine(), "device", None)
    if eng_dev is not None and str(eng_dev) != "cpu":
        try:
            return torch.device(eng_dev)
        except (TypeError, RuntimeError):
            pass
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_dtype(torch: Any) -> Any:
    """The active compute engine's torch float dtype, or ``None`` when no torch precision policy applies.

    The precision twin of ``_resolve_device`` (one place for the engine-following policy): under
    ``TorchEngine(dtype=torch.float64)`` a neural leaf should evaluate its module in fp64 like the rest
    of the substrate math instead of silently dropping to fp32. Outside a torch engine -- including the
    NumPy default, whose ``dtype`` is a numpy dtype -- this returns ``None`` and callers keep their
    historical float32 behavior."""
    from mixle.engines.base import active_engine

    eng_dtype = getattr(active_engine(), "dtype", None)
    if isinstance(eng_dtype, torch.dtype) and eng_dtype.is_floating_point:
        return eng_dtype
    return None


@contextmanager
def _module_mode(module: Any, *, train: bool) -> Any:
    """Hold ``module`` in train/eval mode for the block, restoring every submodule's prior flag on exit.

    Scoring must be a pure read: without ``eval()`` a Dropout submodule scores stochastically, and a
    BatchNorm submodule both scores with batch statistics and MUTATES its running stats on a mere
    ``log_density`` call. The M-step is the converse -- a module the user pre-set to ``eval()`` must
    still optimize under train-mode semantics. The snapshot is per submodule (not just the root flag),
    so a deliberately eval-pinned submodule inside a train-mode net comes back exactly as the caller
    left it. Shared by every gradient-fit leaf, like ``_resolve_device`` above."""
    states = [(m, m.training) for m in module.modules()]
    module.train(train)
    try:
        yield module
    finally:
        for m, was_training in states:
            m.training = was_training


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

    Measured negative, so nobody re-derives it (2026-07-12, Apple M4, torch 2.12 CPU): wrapping the
    full-batch M-step loss in ``torch.compile`` is a LOSS here -- 0.93x at a 2x32 MLP / n=100k and
    0.79x at 2x256 / n=200k, plus ~6s compile overhead per module -- so there is deliberately no
    ``compile=`` flag. Re-measure before adding one (a CUDA build or a much larger module could
    flip it); the probe script pattern lives in the introducing PR.
    distribution (see the module docstring). ``loss`` and ``optimizer`` are the M-step hooks."""

    __pysp_serializable__ = True  # module persisted as bytes (see __pysp_getstate__)

    def __init__(
        self,
        module: Any,
        *,
        m_steps: int = 60,
        lr: float = 5e-3,
        device: Any = None,
        batch_size: int | None = None,
        max_optimizer_steps: int | None = None,
        precision: str = "fp32",
        name: str | None = None,
        loss: Any = None,
        optimizer: Any = None,
        lr_decay: float | None = None,
    ) -> None:
        self.module = module
        self.m_steps = int(m_steps)
        self.lr = float(lr)
        self.device = device  # None => active engine's device, else CUDA if available, else CPU (_resolve_device)
        self.batch_size = None if batch_size is None else int(batch_size)
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError("batch_size must be positive when supplied.")
        self.max_optimizer_steps = None if max_optimizer_steps is None else int(max_optimizer_steps)
        if self.max_optimizer_steps is not None and self.max_optimizer_steps <= 0:
            raise ValueError("max_optimizer_steps must be positive when supplied.")
        self.precision = precision
        self.name = name
        self.loss = loss
        self.optimizer = optimizer
        self.lr_decay = None if lr_decay is None else float(lr_decay)
        if self.lr_decay is not None and not 0.0 < self.lr_decay <= 1.0:
            raise ValueError("lr_decay must lie in (0, 1] when supplied.")
        if self.lr_decay is not None and optimizer is not None:
            raise ValueError(
                "lr_decay applies to the built-in Adam schedule; it cannot be combined with a custom optimizer hook."
            )
        self.outer_objective_compatible = loss is None

    def __str__(self) -> str:
        return f"{type(self).__name__}({type(self.module).__name__})"

    def log_density(self, x: Any) -> float:
        fields = x if isinstance(x, tuple) else (x,)
        rows = tuple(np.atleast_2d(np.asarray(f, dtype=float)) for f in fields)
        return float(self.seq_log_density(rows if isinstance(x, tuple) else rows[0])[0])

    def seq_log_density(self, x: Any) -> np.ndarray:
        torch = _torch()
        # a bare unconditional module sees one field (x); a bare CONDITIONAL module (log_density(x, y, ...))
        # sees a tuple of fields (GradLeafEncoder.seq_encode's arity-generalized output) -- unpack with
        # ``*`` either way, same tuple-default pattern GradEstimator.estimate uses for the M-step.
        fields = x if isinstance(x, tuple) else (x,)
        dev = _resolve_device(self.device, torch)
        self.module.to(dev).eval()
        xts = tuple(
            torch.as_tensor(
                check_finite(np.atleast_2d(np.asarray(f, dtype=float)), f"{type(self).__name__}.seq_log_density"),
                dtype=torch.float32,
                device=dev,
            )
            for f in fields
        )
        with torch.no_grad():
            return self.module.log_density(*xts).cpu().numpy().reshape(-1)

    def sampler(self, seed: int | None = None) -> GradLeafSampler:
        return GradLeafSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> GradEstimator:
        return GradEstimator(
            self.module,
            m_steps=self.m_steps,
            lr=self.lr,
            device=self.device,
            batch_size=self.batch_size,
            max_optimizer_steps=self.max_optimizer_steps,
            precision=self.precision,
            name=self.name,
            loss=self.loss,
            optimizer=self.optimizer,
            lr_decay=self.lr_decay,
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
        dev = _resolve_device(self.dist.device, torch)
        self.dist.module.to(dev).eval()
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

    def seq_encode(self, data: list) -> Any:
        # a bare unconditional module sees one field (x); a bare CONDITIONAL module (log_density(x, y, ...))
        # sees rows as tuples -- split into one contiguous array per position, same shape-preserving contract
        # DataBufferAccumulator already documents ("a single array ... a tuple like (x, y)"), just generalized
        # to arbitrary arity instead of hand-picking n_fields=2 per family.
        if len(data) and isinstance(data[0], tuple):
            n = len(data[0])
            return tuple(np.array([np.atleast_1d(np.asarray(row[i], dtype=float)) for row in data]) for i in range(n))
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
        # the declared n_fields is a default, not a ceiling: a generic bridge (GradLeaf) doesn't know a bare
        # module's arity until the first real batch arrives, so widen once, from empty, to match it.
        if len(fields) != len(self.parts) and not any(self.parts):
            self.parts = [[] for _ in fields]
            self.n_fields = len(fields)
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
            # same widen-once rule as _append: a combine()-fed root (the mp/mpi/dask/ray fan-in path)
            # starts at the declared default arity and must adopt the workers' true arity before
            # zipping, or every field past the first is silently dropped for conditional leaves.
            if len(fields) != len(self.parts) and not any(self.parts):
                self.parts = [[] for _ in fields]
                self.n_fields = len(fields)
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
        device: Any = None,
        batch_size: int | None = None,
        max_optimizer_steps: int | None = None,
        precision: str = "fp32",
        name: str | None = None,
        loss: Any = None,
        optimizer: Any = None,
        lr_decay: float | None = None,
    ) -> None:
        self.module = module
        self.m_steps = int(m_steps)
        self.lr = float(lr)
        self.device = device
        self.batch_size = None if batch_size is None else int(batch_size)
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError("batch_size must be positive when supplied.")
        self.max_optimizer_steps = None if max_optimizer_steps is None else int(max_optimizer_steps)
        if self.max_optimizer_steps is not None and self.max_optimizer_steps <= 0:
            raise ValueError("max_optimizer_steps must be positive when supplied.")
        self.precision = precision
        self.name = name
        self.loss = loss
        self.optimizer = optimizer
        self.lr_decay = None if lr_decay is None else float(lr_decay)
        if self.lr_decay is not None and not 0.0 < self.lr_decay <= 1.0:
            raise ValueError("lr_decay must lie in (0, 1] when supplied.")
        if self.lr_decay is not None and optimizer is not None:
            raise ValueError(
                "lr_decay applies to the built-in Adam schedule; it cannot be combined with a custom optimizer hook."
            )
        # Cumulative count of divergence recoveries across this estimator's M-steps (one optimize()
        # run shares one estimator tree, so this accumulates over EM rounds; see estimate()).
        self.nonfinite_recoveries = 0
        # 1-based count of M-step rounds this estimator has run; drives the lr_decay schedule.
        self._fit_rounds = 0

    def _leaf(self) -> GradLeaf:
        return GradLeaf(
            self.module,
            m_steps=self.m_steps,
            lr=self.lr,
            device=self.device,
            batch_size=self.batch_size,
            max_optimizer_steps=self.max_optimizer_steps,
            precision=self.precision,
            name=self.name,
            loss=self.loss,
            optimizer=self.optimizer,
            lr_decay=self.lr_decay,
        )

    def accumulator_factory(self) -> DataBufferAccumulatorFactory:
        return DataBufferAccumulatorFactory(GradLeafEncoder(), n_fields=1)

    def estimate(self, nobs: float | None, suff_stat: tuple) -> GradLeaf:
        torch = _torch()
        *fields, ws = suff_stat
        params = [p for p in self.module.parameters() if p.requires_grad]
        if not fields or len(fields[0]) == 0 or not params:  # nothing to fit, or a fully frozen (fixed) module
            return self._leaf()
        dev = _resolve_device(self.device, torch)
        # data stays on CPU (mirrors softmax_leaf.py) -- each minibatch is moved to the device, so a
        # larger-than-device-memory dataset still fits; batch_size=None keeps today's single full-batch pass.
        xs = tuple(torch.as_tensor(np.asarray(f, dtype=float), dtype=torch.float32) for f in fields)
        w = torch.as_tensor(np.asarray(ws, dtype=float), dtype=torch.float32)
        w = w / w.sum().clamp(min=1e-8)  # normalized once, up front -- batch_size=None reproduces the old math exactly
        n = xs[0].shape[0]
        bs = self.batch_size or n
        self.module.to(dev).train()
        self._fit_rounds += 1
        # SAEM window: a per-round Robbins--Monro schedule lr / t**a with a in (0.5, 1] satisfies
        # sum(step)=inf and sum(step^2)<inf -- the step-size conditions stochastic-approximation EM
        # analyses (SAEM, gradient-EM) require for almost-sure convergence to stationary points.
        # Constant lr (lr_decay=None, the default) keeps today's behavior and the weaker
        # best-visited-iterate guarantee provided by the outer loop.
        effective_lr = self.lr if self.lr_decay is None else self.lr / (self._fit_rounds**self.lr_decay)
        opt = self.optimizer(params) if self.optimizer is not None else torch.optim.Adam(params, lr=effective_lr)
        autocast_dev = "cuda" if str(dev).startswith("cuda") else "cpu"
        use_bf16 = self.precision == "bf16"
        # Divergence guard: an aggressive step can drive parameters non-finite, after which the
        # module's own log_density may RAISE (e.g. a torch.distributions constraint check) from
        # inside this M-step -- before the outer EM loop's non-finite acceptance gate or its
        # transaction restore can act, crashing the fit and leaving the shared module poisoned.
        # Snapshot the module state up front; on a non-finite loss/parameter or a raising module,
        # restore the snapshot and stop stepping. The round degrades to a no-op proposal the outer
        # loop gates normally, and the recovery is disclosed in the fit receipt.
        pre_step_state = {key: value.detach().clone() for key, value in self.module.state_dict().items()}
        recovered = False
        optimizer_steps = 0
        epochs_completed = 0
        stop = False
        for _ in range(self.m_steps):  # m_steps is epochs; max_optimizer_steps can make update budgets comparable
            perm = torch.randperm(n) if bs < n else torch.arange(n)
            for k in range(0, n, bs):
                idx = perm[k : k + bs]
                xb = tuple(xt[idx].to(dev) for xt in xs)
                wb = w[idx].to(dev)
                opt.zero_grad()
                step_healthy = True
                try:
                    with torch.autocast(device_type=autocast_dev, dtype=torch.bfloat16, enabled=use_bf16):
                        if self.loss is not None:
                            loss = self.loss(self.module, *xb, wb)
                        else:
                            # tuple default: log_density(*fields) -- a single field unpacks to log_density(x),
                            # identical to before; a conditional bare module's log_density(x, y, ...) just works.
                            loss = -(wb * self.module.log_density(*xb)).sum()
                            # ``w`` is normalized over the full M-step data. A uniform minibatch's raw
                            # weighted sum is smaller by E[batch_size / n]; rescale it so every optimizer
                            # step is an unbiased estimate of the same full responsibility-weighted Q
                            # objective. This stabilizes gradient scale across batch sizes without claiming
                            # identical Adam trajectories (their noise and moment estimates still differ).
                            if len(idx) < n:
                                loss = loss * (float(n) / float(len(idx)))
                    if not bool(torch.isfinite(loss)):
                        step_healthy = False
                    else:
                        loss.backward()
                        opt.step()
                        optimizer_steps += 1
                        if not all(bool(torch.isfinite(p).all()) for p in params):
                            step_healthy = False
                except (ValueError, RuntimeError):
                    step_healthy = False
                if not step_healthy:
                    with torch.no_grad():
                        self.module.load_state_dict(pre_step_state)
                    self.nonfinite_recoveries += 1
                    recovered = True
                    stop = True
                    break
                if self.max_optimizer_steps is not None and optimizer_steps >= self.max_optimizer_steps:
                    stop = True
                    break
            epochs_completed += 1
            if stop:
                break
        leaf = self._leaf()
        leaf.fit_receipt = {
            "nobs": int(n),
            "batch_size": int(bs),
            "epochs_requested": int(self.m_steps),
            "epochs_completed": int(epochs_completed),
            "optimizer_steps": int(optimizer_steps),
            "max_optimizer_steps": self.max_optimizer_steps,
            "gradient_estimator": "unbiased_full_weighted_objective" if self.loss is None else "custom_loss",
            "fit_round": int(self._fit_rounds),
            "lr_effective": float(effective_lr),
            "lr_decay": self.lr_decay,
            "saem_schedule": bool(self.lr_decay is not None and self.lr_decay > 0.5),
            "nonfinite_recovery": bool(recovered),
            "nonfinite_recoveries_total": int(self.nonfinite_recoveries),
        }
        return leaf


def _register_serializable() -> None:
    try:
        from mixle.utils.serialization import register_serializable_class
    except ImportError:  # pragma: no cover - serialization is core, but never block import on it
        return
    register_serializable_class(GradLeaf)


_register_serializable()
