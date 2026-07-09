"""Soft-label distillation from a teacher probability distribution.

:mod:`mixle.task.distill` distills hard teacher labels into a local student.
This module uses the richer case where the teacher exposes a probability or
top-k log-probability vector for each example. Matching that distribution
preserves runner-up class information and confidence structure that hard labels
discard.

This is the frontier-label analogue of temperature-softened Hinton
distillation in :mod:`mixle.task.distill_methods`, without requiring a torch
teacher that exposes logits. The teacher is any callable returning a per-example
probability vector. The student is the compact hashed-n-gram MLP used by
:mod:`mixle.task.distill`, trained against soft targets with temperature-scaled
KL and optionally mixed with hard-label loss. The result is a
:class:`~mixle.task.model.TaskModel` whose ``proba_batch`` approximates the
teacher's calibrated distribution and can be calibrated or routed like any
other student.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from mixle.task.model import HashedNGram, TaskModel, TextClassifierIO

_EPS = 1e-12


def _as_prob_matrix(rows: Sequence[Any], n_labels: int | None) -> np.ndarray:
    """Coerce teacher output to an ``(N, C)`` row-stochastic matrix; renormalize and clip small negatives."""
    p = np.atleast_2d(np.asarray(rows, dtype=np.float64))
    if p.ndim != 2:
        raise ValueError("teacher_probs must be a 2-D (N, C) array of per-example class probabilities.")
    if n_labels is not None and p.shape[1] != n_labels:
        raise ValueError(f"teacher_probs has {p.shape[1]} columns but {n_labels} labels were given.")
    p = np.clip(p, 0.0, None)
    sums = p.sum(axis=1, keepdims=True)
    if np.any(sums <= 0.0):
        raise ValueError("every teacher_probs row must have positive total mass.")
    return p / sums


def distill_from_soft_labels(
    texts: Sequence[str],
    teacher_probs: Sequence[Any],
    *,
    labels: Sequence[str],
    temperature: float = 2.0,
    hard_weight: float = 0.0,
    n: int = 3,
    dim: int = 256,
    hidden: Sequence[int] = (64,),
    epochs: int = 300,
    lr: float = 1e-2,
    seed: int = 0,
    task: str = "",
    device: str = "cpu",
) -> TaskModel:
    """Fit a student to per-example teacher probabilities over ``labels``.

    ``teacher_probs`` is ``(N, C)`` with rows summing to 1 (renormalized if not), column ``j`` the
    teacher's probability of ``labels[j]``. The student minimizes the temperature-softened
    ``T^2 * KL(teacher || student)`` (Hinton's scaling, so the soft gradients keep magnitude as ``T``
    grows), optionally mixed with ``hard_weight`` times the hard cross-entropy
    against the teacher's argmax. ``temperature > 1`` softens both sides so
    runner-up structure influences the fit. The result is deterministic given
    ``seed`` and returns a :class:`TaskModel` whose ``proba_batch``
    approximates the teacher's full distribution.
    """
    import torch

    from mixle.models.neural import make_mlp

    if not 0.0 <= hard_weight <= 1.0:
        raise ValueError("hard_weight must be in [0, 1].")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    label_list = [str(v) for v in labels]
    texts = [str(t) for t in texts]
    p_teacher = _as_prob_matrix(teacher_probs, len(label_list))
    if p_teacher.shape[0] != len(texts):
        raise ValueError("teacher_probs must have one row per text.")

    feat = HashedNGram(n=n, dim=dim, seed=seed)
    x = np.asarray(feat.transform(texts), dtype=np.float32)
    cfg = {
        "input_dim": int(x.shape[1]),
        "hidden_dims": [int(h) for h in hidden],
        "output_dim": len(label_list),
        "activation": "relu",
    }
    torch.manual_seed(int(seed))
    module = make_mlp(**cfg).to(device)

    xb = torch.as_tensor(x, device=device)
    pt = torch.as_tensor(p_teacher, dtype=torch.float32, device=device)
    pt_soft = torch.softmax(torch.log(pt.clamp_min(_EPS)) / temperature, dim=1)  # teacher at temperature T
    hard_idx = torch.as_tensor(np.argmax(p_teacher, axis=1), device=device)
    opt = torch.optim.Adam(module.parameters(), lr=float(lr))
    module.train()
    for _ in range(int(epochs)):
        opt.zero_grad()
        logits = module(xb)
        log_ps_T = torch.log_softmax(logits / temperature, dim=1)
        # KL(teacher || student) = sum pt_soft * (log pt_soft - log ps_T); the pt log-term is constant in
        # the student, so cross-entropy suffices -- scaled by T^2 to preserve soft-gradient magnitude.
        soft_loss = (temperature**2) * -(pt_soft * log_ps_T).sum(dim=1).mean()
        loss = (1.0 - hard_weight) * soft_loss
        if hard_weight > 0.0:
            loss = loss + hard_weight * torch.nn.functional.cross_entropy(logits, hard_idx)
        loss.backward()
        opt.step()
    module.eval()

    return TaskModel(
        module,
        TextClassifierIO(feat, label_list),
        builder="mixle.mlp",
        config=cfg,
        task=task or "soft-distilled text classifier",
        meta={
            "distilled": True,
            "soft": True,
            "temperature": float(temperature),
            "hard_weight": float(hard_weight),
            "n_examples": len(texts),
            "labels": label_list,
            "recipe": {"n": n, "dim": dim, "hidden": list(cfg["hidden_dims"]), "epochs": epochs, "lr": lr},
        },
    )


def distill_soft(
    teacher_proba: Callable[[list[str]], Any],
    texts: Sequence[str],
    *,
    labels: Sequence[str],
    **kwargs: Any,
) -> TaskModel:
    """Query a probability-returning teacher once over ``texts`` and soft-distill it (see
    :func:`distill_from_soft_labels`). ``teacher_proba(texts) -> (N, C)`` returns each example's class
    distribution over ``labels`` (e.g. an LLM's normalized top-k logprobs)."""
    probs = teacher_proba(list(str(t) for t in texts))
    return distill_from_soft_labels(texts, probs, labels=labels, **kwargs)


def soft_agreement(student: TaskModel, teacher_probs: Sequence[Any], texts: Sequence[str]) -> float:
    """Mean KL divergence ``KL(teacher || student)`` over ``texts`` -- how faithfully the student matches
    the teacher's full soft distribution (0 = identical), the soft-distillation analog of
    :func:`mixle.task.distill.agreement`. Lower is better; use it to compare soft vs hard students."""
    p_teacher = _as_prob_matrix(teacher_probs, None)
    p_student = np.asarray(student.adapter.proba_batch(student.model, [str(t) for t in texts]), dtype=np.float64)
    p_student = np.clip(p_student, _EPS, None)
    kl = np.sum(p_teacher * (np.log(np.clip(p_teacher, _EPS, None)) - np.log(p_student)), axis=1)
    return float(np.mean(kl))
