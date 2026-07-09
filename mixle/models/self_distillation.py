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
    xt = x.long()
    t = xt.shape[1]
    pos = torch.arange(t, device=xt.device)
    h = model.tok(xt) + model.pos(pos)[None, :, :]
    for keep, blk in zip(keep_mask, model.blocks):
        if keep:
            h = blk(h)
        # else: drop this block -- identity, the defining move of stochastic depth
    return model.head(model.ln(h))[:, -1]


def stochastic_depth_forward(model: Any, x: Any, drop_prob: float, generator: Any = None) -> tuple:
    """Run ``model`` on ``x`` twice: once at full depth, once with each block independently dropped with
    probability ``drop_prob`` (at least one block is always kept, so the partial pass never degenerates to
    the bare embedding/head). Returns ``(full_output, partial_output)`` -- the pair
    :func:`train_with_self_distillation` feeds to :func:`consistency_loss`.

    At ``drop_prob == 0`` both passes keep every block, so the two outputs are IDENTICAL (no dropout
    elsewhere in :class:`~mixle.models.transformer.Block`) -- the degenerate-case sanity check pinned in
    ``mixle/tests/self_distillation_test.py``.
    """
    _require_torch()
    n = len(model.blocks)
    full_output = _forward_with_keep_mask(model, x, [True] * n)
    if drop_prob <= 0.0:
        partial_output = _forward_with_keep_mask(model, x, [True] * n)
        return full_output, partial_output

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
    return full_output, partial_output


@dataclass
class TrainStats:
    """Per-step telemetry from :func:`train_with_self_distillation` -- cross-entropy, stochastic-depth
    consistency, and EMA-teacher consistency losses, kept separately so a caller can see which pressure is
    doing what (and the combined total actually optimized)."""

    ce_loss: list = field(default_factory=list)
    stochastic_depth_loss: list = field(default_factory=list)
    ema_consistency_loss: list = field(default_factory=list)
    total_loss: list = field(default_factory=list)


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
    torch.manual_seed(int(seed))
    model.to(device).train()
    ema_teacher = EMATeacher(model, decay=ema_decay)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = optimizer(params) if optimizer is not None else torch.optim.Adam(params, lr=lr)

    sd_weight = float(consistency_weight if stochastic_depth_weight is None else stochastic_depth_weight)
    ema_w = float(consistency_weight if ema_weight is None else ema_weight)

    def _fresh_iter() -> Any:
        return iter(data()) if callable(data) else iter(data)

    data_iter = _fresh_iter()
    stats = TrainStats()

    for step in range(int(steps)):
        try:
            ctx, nxt = next(data_iter)
        except StopIteration:
            data_iter = _fresh_iter()
            ctx, nxt = next(data_iter)

        x = torch.as_tensor(np.asarray(ctx), dtype=torch.float32, device=device)
        y = torch.as_tensor(np.asarray(nxt), dtype=torch.long, device=device)

        full_out, partial_out = stochastic_depth_forward(model, x, drop_prob)
        ce_loss = F.cross_entropy(full_out, y)
        sd_loss = consistency_loss(partial_out, full_out.detach(), mode=consistency_mode)
        with torch.no_grad():
            teacher_out = ema_teacher.forward(x)
        ema_loss = consistency_loss(full_out, teacher_out, mode=consistency_mode)

        loss = ce_loss + sd_weight * sd_loss + ema_w * ema_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        ema_teacher.update(model)

        stats.ce_loss.append(float(ce_loss.detach()))
        stats.stochastic_depth_loss.append(float(sd_loss.detach()))
        stats.ema_consistency_loss.append(float(ema_loss.detach()))
        stats.total_loss.append(float(loss.detach()))
        if log is not None:
            log(step, stats)

    model.eval()
    return model
