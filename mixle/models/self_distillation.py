"""Self-distillation during training (roadmap J3): EMA-teacher consistency + stochastic-depth targets,
wired into normal training as loss-hooks -- not a separate post-hoc distillation pass, and not a new
trainer either.

The idea, concretely
---------------------
Two self-consistency pressures applied *during* ordinary next-token training:

1. **EMA-teacher consistency** (:class:`EMATeacher`): maintain an exponential-moving-average copy of the
   model's own weights, updated every step (``teacher = decay * teacher + (1 - decay) * student``, the
   standard mean-teacher/BYOL/DINO pattern). The actively-trained (student) model is pushed to agree with
   this temporally-smoothed version of itself via :func:`consistency_loss` -- an implicit regularizer with
   no extra labels or extra model.
2. **Stochastic-depth consistency** (:func:`stochastic_depth_forward`): each step, run the SAME input
   through the model twice -- once at full depth, once with a random subset of blocks skipped entirely
   (the standard stochastic-depth / drop-path regularizer) -- and add a consistency term pulling the
   partial-depth output toward the full-depth output. This directly trains the model to tolerate missing
   blocks, which is exactly the redundancy G3's :mod:`mixle.models.coarsening` depth-merge exploits.

Why this belongs at the loss-hook level, not a new trainer
------------------------------------------------------------
:mod:`mixle.models.grad_leaf` already establishes the "compose via wrapping" pattern for this codebase's
M-step: a training loop is generic, and custom OBJECTIVES are a ``loss(module, x, w) -> scalar`` hook, not
a subclass tree (see ``GradLeaf``/``GradEstimator``). ``CausalLM`` doesn't fit ``GradLeaf`` directly (it has
no ``log_density``; its own dense-teacher-forcing loop lives in :mod:`mixle.models.language_model` and
:mod:`mixle.models.streaming_transformer_leaf`), so :func:`train_with_self_distillation` mirrors THOSE
loops' own conventions (``F.cross_entropy`` over ``(context, next_token)`` micro-batches from
:func:`mixle.data.stream_token_source.stream_token_source`, a plain ``torch.optim.Adam`` M-step) and adds
the two consistency terms as extra, addable loss components on top of the same per-step cross-entropy --
the loss-hook composition pattern, applied at the place this model family's training loop actually lives.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn.functional as F

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - torch is optional
    _HAS_TORCH = False

__all__ = [
    "EMATeacher",
    "TrainStats",
    "consistency_loss",
    "stochastic_depth_forward",
    "train_with_self_distillation",
]


def _require_torch() -> None:
    if not _HAS_TORCH:
        raise RuntimeError("mixle.models.self_distillation requires torch (mixle.models.transformer is torch-only).")


class EMATeacher:
    """An exponential-moving-average copy of a model's own weights: the standard mean-teacher/BYOL/DINO
    self-distillation teacher.

    ``update(student_model)`` applies ``teacher = decay * teacher + (1 - decay) * student`` to every
    tensor in the teacher's ``state_dict`` (parameters AND buffers, so e.g. non-trainable statistics stay
    consistent too) -- called once per training step, AFTER the optimizer step, so the teacher always
    tracks a temporally-smoothed trailing average of the student. The teacher is a real, independent,
    forward-passable module (``forward``/``predict``), held in eval mode with gradients disabled: it is a
    read-only distillation TARGET, never itself directly optimized.
    """

    def __init__(self, model: Any, decay: float = 0.999) -> None:
        _require_torch()
        if not (0.0 <= decay < 1.0):
            raise ValueError(f"decay must be in [0, 1); got {decay}")
        self.decay = float(decay)
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    def update(self, student_model: Any) -> None:
        """One EMA step: pull every teacher tensor toward the student's current value.

        ``CausalLM`` ties ``head.weight`` to ``tok.weight`` (weight tying), so several ``state_dict()``
        keys alias the SAME underlying storage -- updating each key naively would apply the EMA formula
        to that storage more than once per step (a double update). ``seen`` dedupes by storage identity
        (``data_ptr()``) so every real tensor is updated exactly once, however many names alias it.
        """
        # Not a @torch.no_grad() decorator: that evaluates torch.no_grad at class-definition (import)
        # time, which breaks this class's whole point of being importable without torch installed
        # (see _require_torch() in __init__). A `with` block defers the torch reference to call time.
        with torch.no_grad():
            self._update_impl(student_model)

    def _update_impl(self, student_model: Any) -> None:
        d = self.decay
        student_state = student_model.state_dict()
        seen: set = set()
        for name, teacher_tensor in self.ema_model.state_dict().items():
            ptr = teacher_tensor.data_ptr()
            if ptr in seen:
                continue
            seen.add(ptr)
            student_tensor = student_state[name]
            if torch.is_floating_point(teacher_tensor):
                teacher_tensor.mul_(d).add_(student_tensor.detach(), alpha=1.0 - d)
            else:  # integer/bool buffers (none in CausalLM today, but handled honestly): no EMA, just track
                teacher_tensor.copy_(student_tensor)

    def forward(self, x: Any) -> Any:
        """Run ``x`` through the EMA-teacher weights (eval mode, no grad)."""
        with torch.no_grad():
            return self.ema_model(x)

    predict = forward


def consistency_loss(student_output: Any, teacher_output: Any, mode: str = "mse") -> Any:
    """The self-distillation consistency term between a student prediction and a teacher/target
    prediction on the SAME input -- ``mode="mse"`` (default, plain squared-error between logits, the
    mean-teacher convention) or ``mode="kl"`` (``KL(teacher_softmax || student_log_softmax)``, the
    classic soft-target distillation loss).
    """
    _require_torch()
    if mode == "mse":
        return F.mse_loss(student_output, teacher_output)
    if mode == "kl":
        student_log_prob = F.log_softmax(student_output, dim=-1)
        teacher_prob = F.softmax(teacher_output, dim=-1)
        return F.kl_div(student_log_prob, teacher_prob, reduction="batchmean")
    raise ValueError(f"consistency_loss: unrecognized mode {mode!r}, expected 'mse' or 'kl'")


def _forward_with_keep_mask(model: Any, x: Any, keep_mask: list) -> Any:
    """Re-run :class:`~mixle.models.transformer.CausalLM`'s own forward, but skip any block whose
    ``keep_mask`` entry is ``False`` entirely (identity: the residual stream passes straight through) --
    the standard stochastic-depth / drop-path forward. Mirrors
    :func:`mixle.models.language_model._forward_all_positions`'s block-walking convention, restricted to
    the last position (``CausalLM.forward``'s own output shape) since that is what the acceptance-relevant
    cross-entropy and consistency losses score here.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch tensor")
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("x must have non-empty shape (batch, sequence)")
    if x.shape[1] > model.block:
        raise ValueError(f"x sequence length {x.shape[1]} exceeds configured block size {model.block}")
    if len(keep_mask) != len(model.blocks) or any(not isinstance(keep, bool) for keep in keep_mask):
        raise ValueError("keep_mask must contain exactly one boolean per model block")
    xt = model._validated_ids(
        x,
        name="x",
        expected_shapes=(tuple(x.shape),),
        upper_bound=model.vocab,
    )
    t = xt.shape[1]
    pos = torch.arange(t, device=xt.device)
    h = model.tok(xt) + model.pos(pos)[None, :, :]
    for keep, blk in zip(keep_mask, model.blocks):
        if keep:
            h = blk(h)
        # else: drop this block -- identity, the defining move of stochastic depth
    logits = model.head(model.ln(h)) * getattr(model, "mup_output_multiplier", 1.0)
    return logits[:, -1]


def stochastic_depth_forward(
    model: Any,
    x: Any,
    drop_prob: float,
    generator: Any = None,
    *,
    return_keep_mask: bool = False,
) -> tuple:
    """Run ``model`` on ``x`` twice: once at full depth, once with each block independently dropped with
    probability ``drop_prob`` (at least one block is always kept, so the partial pass never degenerates to
    the bare embedding/head). Returns ``(full_output, partial_output)`` -- the pair
    :func:`train_with_self_distillation` feeds to :func:`consistency_loss`.

    At ``drop_prob == 0`` both passes keep every block, so the two outputs are IDENTICAL (no dropout
    elsewhere in :class:`~mixle.models.transformer.Block`) -- the degenerate-case sanity check pinned in
    ``mixle/tests/self_distillation_test.py``.
    """
    _require_torch()
    drop_prob = _probability(drop_prob, "drop_prob", upper_inclusive=True)
    if not isinstance(return_keep_mask, bool):
        raise TypeError("return_keep_mask must be a boolean")
    if not hasattr(model, "blocks"):
        raise TypeError("model must expose a transformer block sequence")
    n = len(model.blocks)
    if n == 0 and drop_prob > 0.0:
        raise ValueError("a model with zero blocks cannot use positive stochastic-depth drop probability")
    full_mask = [True] * n
    full_output = _forward_with_keep_mask(model, x, full_mask)
    if drop_prob == 0.0:
        partial_output = _forward_with_keep_mask(model, x, full_mask)
        return (full_output, partial_output, full_mask) if return_keep_mask else (full_output, partial_output)

    if generator is not None:
        r = torch.rand(n, generator=generator)
        keep_idx_if_empty = int(torch.randint(0, n, (1,), generator=generator).item())
    else:
        r = torch.rand(n)
        keep_idx_if_empty = int(torch.randint(0, n, (1,)).item())
    keep_mask = (r >= float(drop_prob)).tolist()
    if not any(keep_mask):
        keep_mask[keep_idx_if_empty] = True
    partial_output = _forward_with_keep_mask(model, x, keep_mask)
    return (full_output, partial_output, keep_mask) if return_keep_mask else (full_output, partial_output)


@dataclass
class TrainStats:
    """Per-step telemetry from :func:`train_with_self_distillation` -- cross-entropy, stochastic-depth
    consistency, and EMA-teacher consistency losses, kept separately so a caller can see which pressure is
    doing what (and the combined total actually optimized)."""

    ce_loss: list[float] = field(default_factory=list)
    stochastic_depth_loss: list[float] = field(default_factory=list)
    ema_consistency_loss: list[float] = field(default_factory=list)
    total_loss: list[float] = field(default_factory=list)


def train_with_self_distillation(
    model: Any,
    data: Any,
    steps: int,
    *,
    ema_decay: float = 0.999,
    drop_prob: float = 0.1,
    consistency_weight: float = 1.0,
    ema_weight: float | None = None,
    stochastic_depth_weight: float | None = None,
    consistency_mode: str = "mse",
    lr: float = 3e-3,
    device: str = "cpu",
    optimizer: Any = None,
    seed: int = 0,
    log: Any = None,
) -> Any:
    """Train ``model`` (a :class:`~mixle.models.transformer.CausalLM`, trained in place and also
    returned) for ``steps`` next-token cross-entropy steps, with EMA-teacher consistency and
    stochastic-depth consistency added as extra loss terms on top of the SAME per-step batch -- both
    self-distillation pressures happen DURING training, not as a separate post-hoc pass.

    ``data`` yields ``(context, next_token)`` micro-batches shaped exactly like
    :func:`mixle.data.stream_token_source.stream_token_source` (``context: (batch, block)`` float ids,
    ``next_token: (batch,)`` int ids). ``data`` may be:

    * a zero-arg CALLABLE returning a fresh iterator each time (e.g.
      ``lambda: stream_token_source(ids, block=64, batch_size=32)``) -- restarted automatically whenever
      it runs dry before ``steps`` is reached, so training can outlast one epoch; or
    * a plain iterable/iterator (e.g. a list of batches, or a single generator object) -- consumed once,
      sized to yield at least ``steps`` batches (a bare generator can't be rewound).

    Per step: ``loss = cross_entropy(full_depth_logits, target) + stochastic_depth_weight *
    consistency(partial_depth_logits, full_depth_logits.detach()) + ema_weight *
    consistency(full_depth_logits, ema_teacher(context))``, then one optimizer step, then one EMA-teacher
    update. ``ema_weight``/``stochastic_depth_weight`` each default to ``consistency_weight`` when unset.
    """
    _require_torch()
    steps = _positive_int(steps, "steps")
    seed = _integer(seed, "seed")
    ema_decay = _probability(ema_decay, "ema_decay", upper_inclusive=False)
    drop_prob = _probability(drop_prob, "drop_prob", upper_inclusive=True)
    consistency_weight = _nonnegative_finite(consistency_weight, "consistency_weight")
    sd_weight = _nonnegative_finite(
        consistency_weight if stochastic_depth_weight is None else stochastic_depth_weight,
        "stochastic_depth_weight",
    )
    ema_w = _nonnegative_finite(
        consistency_weight if ema_weight is None else ema_weight,
        "ema_weight",
    )
    lr = _positive_finite(lr, "lr")
    if consistency_mode not in {"mse", "kl"}:
        raise ValueError("consistency_mode must be 'mse' or 'kl'")
    if log is not None and not callable(log):
        raise TypeError("log must be callable or None")
    if not callable(data):
        try:
            iter(data)
        except TypeError as exc:
            raise TypeError("data must be an iterable or a zero-argument callable returning an iterable") from exc
    if not hasattr(model, "blocks") or not hasattr(model, "vocab") or not hasattr(model, "block"):
        raise TypeError("model must provide transformer blocks, vocabulary size, and block size")
    if len(model.blocks) == 0 and drop_prob > 0.0:
        raise ValueError("a zero-block model cannot use positive stochastic-depth drop probability")
    try:
        target_device = torch.device(device)
    except (TypeError, RuntimeError) as exc:
        raise ValueError(f"invalid torch device {device!r}") from exc

    original_training = bool(model.training)
    initial_student_digest = _model_digest(model)
    teacher: EMATeacher | None = None
    stats = TrainStats()
    keep_schedule: list[list[bool]] = []
    completed_steps = 0
    try:
        torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed)
        model.to(target_device).train()
        teacher = EMATeacher(model, decay=ema_decay)
        initial_teacher_digest = _model_digest(teacher.ema_model)

        from mixle.models.optimizer_routing import resolve_neural_optimizer

        opt, optimizer_receipt = resolve_neural_optimizer(model, optimizer, lr=lr, sign_stable=False)

        restartable = callable(data)

        def _fresh_iter() -> Any:
            produced = data() if restartable else data
            try:
                return iter(produced)
            except TypeError as exc:
                raise TypeError("data callable must return an iterable") from exc

        data_iter = _fresh_iter()

        def _next_batch() -> Any:
            nonlocal data_iter
            try:
                return next(data_iter)
            except StopIteration:
                if not restartable:
                    raise ValueError(
                        f"data iterator was exhausted after {completed_steps} of {steps} requested steps"
                    ) from None
                data_iter = _fresh_iter()
                try:
                    return next(data_iter)
                except StopIteration:
                    raise ValueError("data callable returned an empty iterable") from None

        for step in range(steps):
            x, y = _validated_batch(_next_batch(), model, target_device, torch)

            full_out, partial_out, keep_mask = stochastic_depth_forward(
                model,
                x,
                drop_prob,
                generator,
                return_keep_mask=True,
            )
            if (
                full_out.shape != (len(x), model.vocab)
                or partial_out.shape != full_out.shape
                or not bool(torch.all(torch.isfinite(full_out)).detach().cpu().item())
                or not bool(torch.all(torch.isfinite(partial_out)).detach().cpu().item())
            ):
                raise RuntimeError("self-distillation model outputs must be finite (batch, vocab) logits")
            ce_loss = F.cross_entropy(full_out, y)
            sd_loss = consistency_loss(partial_out, full_out.detach(), mode=consistency_mode)
            with torch.no_grad():
                teacher_out = teacher.forward(x)
            ema_loss = consistency_loss(full_out, teacher_out, mode=consistency_mode)

            loss = ce_loss + sd_weight * sd_loss + ema_w * ema_loss
            if not bool(torch.isfinite(loss).detach().cpu().item()):
                raise RuntimeError("self-distillation objective became non-finite")
            opt.zero_grad()
            loss.backward()
            opt.step()
            if any(
                not bool(torch.all(torch.isfinite(parameter)).detach().cpu().item())
                for parameter in model.parameters()
            ):
                raise RuntimeError("self-distillation produced non-finite model parameters")
            teacher.update(model)

            stats.ce_loss.append(float(ce_loss.detach()))
            stats.stochastic_depth_loss.append(float(sd_loss.detach()))
            stats.ema_consistency_loss.append(float(ema_loss.detach()))
            stats.total_loss.append(float(loss.detach()))
            keep_schedule.append(list(keep_mask))
            completed_steps += 1
            if log is not None:
                log(step, stats)

        model.self_distillation_receipt = {
            "schema_version": "1.0.0",
            "student": {
                "class": f"{type(model).__module__}.{type(model).__qualname__}",
                "initial_sha256": initial_student_digest,
                "final_sha256": _model_digest(model),
            },
            "teacher": {
                "class": f"{type(teacher.ema_model).__module__}.{type(teacher.ema_model).__qualname__}",
                "initial_sha256": initial_teacher_digest,
                "final_sha256": _model_digest(teacher.ema_model),
                "decay": ema_decay,
            },
            "objective": {
                "mode": consistency_mode,
                "ema_weight": ema_w,
                "stochastic_depth_weight": sd_weight,
            },
            "budget": {"requested_steps": steps, "completed_steps": completed_steps},
            "seed": seed,
            "drop_prob": drop_prob,
            "keep_schedule": keep_schedule,
            "losses": {
                "cross_entropy": list(stats.ce_loss),
                "stochastic_depth": list(stats.stochastic_depth_loss),
                "ema_consistency": list(stats.ema_consistency_loss),
                "total": list(stats.total_loss),
            },
            "optimizer": optimizer_receipt,
        }
        return model
    finally:
        model.train(original_training)


def _validated_batch(batch: Any, model: Any, device: Any, torch_module: Any) -> tuple[Any, Any]:
    if not isinstance(batch, (tuple, list)) or len(batch) != 2:
        raise ValueError("each self-distillation batch must be a (context, next_token) pair")
    context, target = batch
    try:
        context_array = np.asarray(context)
        target_array = np.asarray(target)
    except Exception as exc:
        raise ValueError("context and next_token must be array-like") from exc
    if context_array.ndim != 2 or context_array.shape[0] == 0 or context_array.shape[1] == 0:
        raise ValueError("context must have non-empty shape (batch, sequence)")
    if context_array.shape[1] > model.block:
        raise ValueError(
            f"context sequence length {context_array.shape[1]} exceeds configured block size {model.block}"
        )
    if target_array.ndim != 1 or len(target_array) != len(context_array):
        raise ValueError("next_token must be a one-dimensional vector aligned with context rows")
    for values, name in ((context_array, "context"), (target_array, "next_token")):
        if values.dtype.kind not in {"i", "u", "f"}:
            raise ValueError(f"{name} must contain numeric token ids")
        if values.dtype.kind == "f" and (
            not np.all(np.isfinite(values)) or not np.all(values == np.round(values))
        ):
            raise ValueError(f"{name} must contain finite integer-valued token ids")
        if np.any(values < 0) or np.any(values >= model.vocab):
            raise ValueError(f"{name} token ids must lie in [0, {model.vocab})")
    return (
        torch_module.as_tensor(context_array, dtype=torch_module.long, device=device),
        torch_module.as_tensor(target_array, dtype=torch_module.long, device=device),
    )


def _model_digest(model: Any) -> str:
    digest = hashlib.sha256()
    digest.update(f"{type(model).__module__}.{type(model).__qualname__}".encode())
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _probability(value: Any, name: str, *, upper_inclusive: bool) -> float:
    result = _nonnegative_finite(value, name)
    if result > 1.0 or (not upper_inclusive and result == 1.0):
        boundary = "[0, 1]" if upper_inclusive else "[0, 1)"
        raise ValueError(f"{name} must lie in {boundary}")
    return result


def _positive_finite(value: Any, name: str) -> float:
    result = _nonnegative_finite(value, name)
    if result == 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_finite(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real scalar, not a boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real scalar") from exc
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _positive_int(value: Any, name: str) -> int:
    result = _integer(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    return int(value)
