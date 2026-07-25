"""``GradLeaf`` -- a torch module IS the model: the five-piece contract, manufactured.

The contract (Distribution / Sampler / Estimator / Accumulator / DataEncoder) earns its keep for
closed-form families: additive sufficient statistics are what make EM exact and distributable. A
GRADIENT leaf has no declared sufficient statistics -- its "accumulator" can only buffer the
responsibility-weighted data, its encoder is ``np.asarray``, and its generic M-step is gradient based -- so per-family
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
* ``module.mixle_analytic_m_step(*fields, weights=..., batch_size=...)`` can provide an exact or
  symbolically generated update, bypassing autograd and iterative optimization entirely;
* ``optimizer=None`` routes trainable parameters by role, geometry, and batch regime (AdaGrad for
  stochastic embeddings/routers, Rprop for small full-batch blocks, Muon for large matrices).
  Named optimizers and ``optimizer(params)`` remain explicit escape hatches;
* ``fitted.module`` is the raw torch module -- nothing is trapped.

Serialization: the module round-trips as portable bytes (``mixle.models._neural_serial``); custom
``loss``/``optimizer`` hooks must be module-level functions to survive pickling, like any hook.
"""

from __future__ import annotations

import inspect
import operator
from contextlib import contextmanager
from functools import wraps
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


def _build_optimizer(
    module: Any, params: list[Any], optimizer: Any, lr: float, torch: Any, *, sign_stable: bool
) -> tuple[Any, dict[str, Any]]:
    """Resolve the gradient tier while keeping Adam an explicit request."""

    del params, torch
    from mixle.models.optimizer_routing import resolve_neural_optimizer

    return resolve_neural_optimizer(module, optimizer, lr=lr, sign_stable=sign_stable)


def _run_analytic_m_step(module: Any, fields: tuple[Any, ...], weights: Any, batch_size: int | None) -> Any:
    """Run a module-owned exact update when the module declares one."""

    update = getattr(module, "mixle_analytic_m_step", None)
    if not callable(update):
        return None
    result = update(*fields, weights=weights, batch_size=batch_size)
    if result is None:
        return {}
    if not isinstance(result, dict):
        raise TypeError("mixle_analytic_m_step must return a receipt dictionary or None.")
    return dict(result)


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


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not a boolean.")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer.") from exc
    if result <= 0:
        raise ValueError(f"{name} must be positive.")
    return result


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a positive finite number, not a boolean.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a positive finite number.") from exc
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite number.")
    return result


def _validate_precision(precision: Any) -> str:
    if not isinstance(precision, str):
        raise TypeError("precision must be one of 'auto', 'fp32', 'fp64', or 'bf16'.")
    aliases = {
        "auto": "auto",
        "fp32": "fp32",
        "float32": "fp32",
        "fp64": "fp64",
        "float64": "fp64",
        "bf16": "bf16",
        "bfloat16": "bf16",
    }
    try:
        return aliases[precision.lower()]
    except KeyError as exc:
        raise ValueError("precision must be one of 'auto', 'fp32', 'fp64', or 'bf16'.") from exc


def _precision_policy(torch: Any, precision: str, device: Any, module: Any) -> tuple[Any, bool]:
    """Resolve active-engine precision before the leaf fallback, including module/data dtype."""
    normalized = _validate_precision(precision)
    engine_dtype = _resolve_dtype(torch)
    if engine_dtype is not None:
        chosen = engine_dtype
    elif normalized == "auto":
        chosen = next(
            (
                tensor.dtype
                for tensor in (*module.parameters(), *module.buffers())
                if tensor.dtype in {torch.float32, torch.float64}
            ),
            torch.float32,
        )
    else:
        chosen = {
            "fp32": torch.float32,
            "fp64": torch.float64,
            "bf16": torch.bfloat16,
        }[normalized]
    if chosen == torch.bfloat16:
        if device.type not in {"cpu", "cuda"}:
            raise ValueError(f"bf16 gradient leaves are not supported on {device.type} devices.")
        return torch.float32, True
    if chosen not in {torch.float32, torch.float64}:
        raise ValueError(f"unsupported active-engine Torch dtype for gradient leaves: {chosen}.")
    return chosen, False


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


def _sample_count(size: Any) -> int:
    """Validate the distribution-sampler size contract without lossy integer coercion."""
    if size is None:
        return 1
    if isinstance(size, (bool, np.bool_)):
        raise TypeError("sample size must be an integer, not a boolean.")
    try:
        count = operator.index(size)
    except TypeError as exc:
        raise TypeError("sample size must be an integer.") from exc
    if count <= 0:
        raise ValueError("sample size must be positive.")
    return count


def _accepts_generator(method: Any) -> bool:
    """Return whether a bound sampling method explicitly supports an isolated Torch generator."""
    try:
        parameters = inspect.signature(method).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.name == "generator"
        or parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def _call_sample_with_seed(method: Any, *args: Any, torch: Any, device: Any, seed: int) -> Any:
    """Call a sampling method reproducibly without changing the caller's Torch RNG state.

    New modules should accept ``generator=``. The state-preserving fallback keeps legacy ``sample(n)``
    modules working without mistaking an internal ``TypeError`` for an unsupported keyword.
    """
    if _accepts_generator(method):
        try:
            generator = torch.Generator(device=device)
        except RuntimeError:
            generator = None
        if generator is not None:
            generator.manual_seed(seed)
            return method(*args, generator=generator)

    cuda_devices = []
    if device.type == "cuda":
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
    mps_state = None
    if device.type == "mps" and hasattr(torch.mps, "get_rng_state"):
        mps_state = torch.mps.get_rng_state()
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            return method(*args)
    finally:
        if mps_state is not None:
            torch.mps.set_rng_state(mps_state)


def _preserve_fit_mode(method: Any) -> Any:
    """Run an estimator method in training mode and restore every caller-owned mode flag."""

    @wraps(method)
    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        with _module_mode(self.module, train=True):
            return method(self, *args, **kwargs)

    return wrapped


def _validate_scores(scores: Any, rows: int, torch: Any, context: str) -> Any:
    if not isinstance(scores, torch.Tensor):
        raise TypeError(f"{context} must return a Torch tensor.")
    if tuple(scores.shape) != (rows,):
        raise ValueError(f"{context} must return exactly one score per row; got shape {tuple(scores.shape)}.")
    if not bool(torch.isfinite(scores).all()):
        raise FloatingPointError(f"{context} returned a non-finite score.")
    return scores


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
        precision: str = "auto",
        name: str | None = None,
        loss: Any = None,
        optimizer: Any = None,
        lr_decay: float | None = None,
    ) -> None:
        self.module = module
        self.m_steps = _positive_int(m_steps, "m_steps")
        self.lr = _positive_float(lr, "lr")
        self.device = device  # None => active engine's device, else CUDA if available, else CPU (_resolve_device)
        self.batch_size = None if batch_size is None else _positive_int(batch_size, "batch_size")
        self.max_optimizer_steps = (
            None if max_optimizer_steps is None else _positive_int(max_optimizer_steps, "max_optimizer_steps")
        )
        self.precision = _validate_precision(precision)
        self.name = name
        self.loss = loss
        self.optimizer = optimizer
        self.lr_decay = None if lr_decay is None else _positive_float(lr_decay, "lr_decay")
        if self.lr_decay is not None and self.lr_decay > 1.0:
            raise ValueError("lr_decay must lie in (0, 1] when supplied.")
        if self.lr_decay is not None and callable(optimizer):
            raise ValueError(
                "lr_decay applies to mixle-managed optimizers; it cannot be combined with a custom optimizer hook."
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
        dtype, use_bf16 = _precision_policy(torch, self.precision, dev, self.module)
        self.module.to(device=dev, dtype=dtype)
        xts = tuple(
            torch.as_tensor(
                check_finite(np.atleast_2d(np.asarray(f, dtype=float)), f"{type(self).__name__}.seq_log_density"),
                dtype=dtype,
                device=dev,
            )
            for f in fields
        )
        autocast_dev = dev.type
        with (
            _module_mode(self.module, train=False),
            torch.no_grad(),
            torch.autocast(device_type=autocast_dev, dtype=torch.bfloat16, enabled=use_bf16),
        ):
            scores = _validate_scores(
                self.module.log_density(*xts), xts[0].shape[0], torch, f"{type(self.module).__name__}.log_density"
            )
            return scores.float().cpu().numpy() if scores.dtype == torch.bfloat16 else scores.cpu().numpy()

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
        n = _sample_count(size)
        dev = _resolve_device(self.dist.device, torch)
        self.dist.module.to(dev)
        seed = int(self.rng.randint(0, 2**31 - 1))
        with _module_mode(self.dist.module, train=False), torch.no_grad():
            out = _call_sample_with_seed(
                self.dist.module.sample, n, torch=torch, device=dev, seed=seed
            ).cpu().numpy()
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
        self.n_fields = _positive_int(n_fields, "n_fields")
        self.parts: list[list] = [[] for _ in range(self.n_fields)]
        self.w: list = []
        self._schema_bound = False

    # Contiguous batch arrays concatenated once at value() (shape-preserving) rather than one ndarray per row.
    def _append(self, enc: Any, weights: np.ndarray) -> None:
        fields = enc if isinstance(enc, tuple) else (enc,)
        if not fields:
            raise ValueError("gradient accumulator batches must contain at least one field.")
        # The declared arity is only an initial allocation hint: the first actual batch pins the schema.
        if not self._schema_bound:
            self.parts = [[] for _ in fields]
            self.n_fields = len(fields)
            self._schema_bound = True
        elif len(fields) != self.n_fields:
            raise ValueError(f"gradient accumulator expected {self.n_fields} fields, received {len(fields)}.")

        prepared = []
        rows = None
        for index, f in enumerate(fields):
            fb = np.asarray(f, dtype=float)
            if fb.ndim == 0:
                raise ValueError(f"gradient accumulator field {index} must have a row dimension.")
            fb = fb.reshape(fb.shape[0], 1) if fb.ndim == 1 else fb
            if rows is None:
                rows = fb.shape[0]
            elif fb.shape[0] != rows:
                raise ValueError("all gradient accumulator fields must have the same row count.")
            prepared.append(fb)
        wb = np.asarray(weights, dtype=float)
        if wb.ndim != 1:
            raise ValueError("gradient accumulator weights must be one-dimensional.")
        if wb.shape[0] != rows:
            raise ValueError(
                f"gradient accumulator received {wb.shape[0]} weights for {rows} rows."
            )
        for buf, fb in zip(self.parts, prepared):
            buf.append(fb)
        self.w.append(wb)

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
        if fields or len(ws):
            self._append(tuple(fields), np.asarray(ws, dtype=float))
        return self

    def value(self) -> tuple:
        fields = tuple(np.concatenate(buf, axis=0) if buf else np.zeros((0, 0)) for buf in self.parts)
        w = np.concatenate(self.w) if self.w else np.zeros((0,))
        return (*fields, w)

    def from_value(self, v: tuple) -> DataBufferAccumulator:
        *fields, w = v
        self.parts = [[] for _ in range(max(len(fields), 1))]
        self.w = []
        self._schema_bound = False
        self._append(tuple(fields), np.asarray(w, dtype=float))
        return self

    def acc_to_encoder(self) -> Any:
        return self.encoder


class DataBufferAccumulatorFactory(StatisticAccumulatorFactory):
    def __init__(self, encoder: Any, n_fields: int = 1) -> None:
        self.encoder = encoder
        self.n_fields = _positive_int(n_fields, "n_fields")

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
        precision: str = "auto",
        name: str | None = None,
        loss: Any = None,
        optimizer: Any = None,
        lr_decay: float | None = None,
    ) -> None:
        self.module = module
        self.m_steps = _positive_int(m_steps, "m_steps")
        self.lr = _positive_float(lr, "lr")
        self.device = device
        self.batch_size = None if batch_size is None else _positive_int(batch_size, "batch_size")
        self.max_optimizer_steps = (
            None if max_optimizer_steps is None else _positive_int(max_optimizer_steps, "max_optimizer_steps")
        )
        self.precision = _validate_precision(precision)
        self.name = name
        self.loss = loss
        self.optimizer = optimizer
        self.lr_decay = None if lr_decay is None else _positive_float(lr_decay, "lr_decay")
        if self.lr_decay is not None and self.lr_decay > 1.0:
            raise ValueError("lr_decay must lie in (0, 1] when supplied.")
        if self.lr_decay is not None and callable(optimizer):
            raise ValueError(
                "lr_decay applies to mixle-managed optimizers; it cannot be combined with a custom optimizer hook."
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

    @_preserve_fit_mode
    def estimate(self, nobs: float | None, suff_stat: tuple) -> GradLeaf:
        torch = _torch()
        *fields, ws = suff_stat
        params = [p for p in self.module.parameters() if p.requires_grad]
        if not fields:
            return self._leaf()
        arrays = tuple(np.asarray(field, dtype=float) for field in fields)
        if any(array.ndim == 0 for array in arrays):
            raise ValueError("gradient M-step fields must have a row dimension.")
        n = arrays[0].shape[0]
        if any(array.shape[0] != n for array in arrays):
            raise ValueError("gradient M-step fields must have identical row counts.")
        weights = np.asarray(ws, dtype=float)
        if weights.ndim != 1 or weights.shape[0] != n:
            raise ValueError(f"gradient M-step requires exactly one weight per row; got {weights.shape} for {n} rows.")
        if n == 0:
            return self._leaf()
        if not np.isfinite(weights).all():
            raise ValueError("gradient M-step weights must be finite.")
        if np.any(weights < 0.0):
            raise ValueError("gradient M-step weights must be non-negative.")
        weight_total = float(weights.sum())
        if not np.isfinite(weight_total) or weight_total <= 0.0:
            raise ValueError("gradient M-step weights must have positive finite total mass.")
        if not params:  # a fully frozen module is a fixed distribution, after validating its supplied batch
            return self._leaf()
        dev = _resolve_device(self.device, torch)
        dtype, use_bf16 = _precision_policy(torch, self.precision, dev, self.module)
        # data stays on CPU (mirrors softmax_leaf.py) -- each minibatch is moved to the device, so a
        # larger-than-device-memory dataset still fits; batch_size=None keeps today's single full-batch pass.
        xs = tuple(torch.as_tensor(array, dtype=dtype) for array in arrays)
        w = torch.as_tensor(weights / weight_total, dtype=dtype)
        bs = self.batch_size or n
        self.module.to(device=dev, dtype=dtype)
        self._fit_rounds += 1
        # SAEM window: a per-round Robbins--Monro schedule lr / t**a with a in (0.5, 1] satisfies
        # sum(step)=inf and sum(step^2)<inf -- the step-size conditions stochastic-approximation EM
        # analyses (SAEM, gradient-EM) require for almost-sure convergence to stationary points.
        # Constant lr (lr_decay=None, the default) keeps today's behavior and the weaker
        # best-visited-iterate guarantee provided by the outer loop.
        effective_lr = self.lr if self.lr_decay is None else self.lr / (self._fit_rounds**self.lr_decay)
        pre_step_state = {key: value.detach().clone() for key, value in self.module.state_dict().items()}
        try:
            analytic_receipt = _run_analytic_m_step(self.module, xs, w, self.batch_size)
        except Exception as exc:
            with torch.no_grad():
                self.module.load_state_dict(pre_step_state)
            raise RuntimeError(f"gradient analytic M-step failed; module state was restored: {exc}") from exc
        if analytic_receipt is not None:
            recovered = not all(bool(torch.isfinite(p).all()) for p in params)
            if recovered:
                with torch.no_grad():
                    self.module.load_state_dict(pre_step_state)
                self.nonfinite_recoveries += 1
            leaf = self._leaf()
            leaf.fit_receipt = {
                "nobs": int(n),
                "batch_size": int(bs),
                "epochs_requested": int(self.m_steps),
                "epochs_completed": 0,
                "optimizer_steps": 0,
                "max_optimizer_steps": self.max_optimizer_steps,
                "gradient_estimator": "not_used",
                "update_method": "analytic_m_step",
                "optimizer": "none",
                "optimizer_plan": None,
                "analytic_receipt": analytic_receipt,
                "fit_round": int(self._fit_rounds),
                "lr_effective": float(effective_lr),
                "lr_decay": self.lr_decay,
                "saem_schedule": False,
                "nonfinite_recovery": bool(recovered),
                "nonfinite_recoveries_total": int(self.nonfinite_recoveries),
            }
            return leaf
        opt, optimizer_receipt = _build_optimizer(
            self.module, params, self.optimizer, effective_lr, torch, sign_stable=bs >= n
        )
        autocast_dev = dev.type
        # Divergence guard: an aggressive step can drive parameters non-finite, after which the
        # module's own log_density may RAISE (e.g. a torch.distributions constraint check) from
        # inside this M-step -- before the outer EM loop's non-finite acceptance gate or its
        # transaction restore can act, crashing the fit and leaving the shared module poisoned.
        # Snapshot the module state up front; on a non-finite loss/parameter or a raising module,
        # restore the snapshot and stop stepping. The round degrades to a no-op proposal the outer
        # loop gates normally, and the recovery is disclosed in the fit receipt.
        recovered = False
        optimizer_steps = 0
        epochs_completed = 0
        stop = False
        for epoch in range(self.m_steps):  # m_steps is epochs; max_optimizer_steps can compare update budgets
            perm = torch.randperm(n) if bs < n else torch.arange(n)
            epoch_completed = True
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
                            if not isinstance(loss, torch.Tensor):
                                raise TypeError("custom gradient loss must return a Torch tensor.")
                            if loss.ndim != 0:
                                raise ValueError(
                                    f"custom gradient loss must return a scalar tensor; got shape {tuple(loss.shape)}."
                                )
                        else:
                            # tuple default: log_density(*fields) -- a single field unpacks to log_density(x),
                            # identical to before; a conditional bare module's log_density(x, y, ...) just works.
                            scores = _validate_scores(
                                self.module.log_density(*xb),
                                len(idx),
                                torch,
                                f"{type(self.module).__name__}.log_density",
                            )
                            loss = -(wb * scores).sum()
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
                except FloatingPointError:
                    step_healthy = False
                except Exception as exc:
                    with torch.no_grad():
                        self.module.load_state_dict(pre_step_state)
                    raise RuntimeError(
                        f"gradient M-step failed at epoch {epoch + 1}, batch starting at row {k}; "
                        f"module state was restored: {exc}"
                    ) from exc
                if not step_healthy:
                    with torch.no_grad():
                        self.module.load_state_dict(pre_step_state)
                    self.nonfinite_recoveries += 1
                    recovered = True
                    stop = True
                    epoch_completed = False
                    break
                if self.max_optimizer_steps is not None and optimizer_steps >= self.max_optimizer_steps:
                    stop = True
                    epoch_completed = k + bs >= n
                    break
            if epoch_completed:
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
            "update_method": "autograd",
            "optimizer": optimizer_receipt["name"],
            "optimizer_plan": optimizer_receipt["plan"],
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
