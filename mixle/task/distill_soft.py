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
    """Validate teacher output as a finite, non-negative, row-stochastic ``(N, C)`` matrix."""
    try:
        p = np.atleast_2d(np.asarray(rows, dtype=np.float64))
    except (TypeError, ValueError) as exc:
        raise ValueError("teacher_probs must be a numeric 2-D probability matrix") from exc
    if p.ndim != 2:
        raise ValueError("teacher_probs must be a 2-D (N, C) array of per-example class probabilities.")
    if p.shape[0] == 0 or p.shape[1] == 0:
        raise ValueError("teacher_probs must contain at least one row and one class")
    if n_labels is not None and p.shape[1] != n_labels:
        raise ValueError(f"teacher_probs has {p.shape[1]} columns but {n_labels} labels were given.")
    if not np.all(np.isfinite(p)):
        raise ValueError("teacher_probs must contain only finite probabilities")
    if np.any(p < 0.0):
        raise ValueError("teacher_probs must contain only non-negative probabilities")
    sums = p.sum(axis=1, keepdims=True)
    if not np.allclose(sums, 1.0, rtol=1e-7, atol=1e-8):
        raise ValueError("every teacher_probs row must sum to one")
    return p


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
    batch_size: int | None = None,
    optimizer: Any = "auto",
    analytic_ridge: float = 1.0e-6,
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
    from mixle.task.distill_methods import planned_response_distill

    try:
        hard_weight = float(hard_weight)
    except (TypeError, ValueError) as exc:
        raise ValueError("hard_weight must be finite and in [0, 1].") from exc
    if not np.isfinite(hard_weight) or not 0.0 <= hard_weight <= 1.0:
        raise ValueError("hard_weight must be in [0, 1].")
    try:
        temperature = float(temperature)
    except (TypeError, ValueError) as exc:
        raise ValueError("temperature must be finite and positive.") from exc
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    if isinstance(labels, (str, bytes)):
        raise ValueError("labels must be a sequence of unique class names")
    label_list = [str(v) for v in labels]
    if not label_list or len(set(label_list)) != len(label_list):
        raise ValueError("labels must be a nonempty sequence of unique class names")
    texts = [str(t) for t in texts]
    if not texts:
        raise ValueError("soft-label distillation requires at least one text")
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
    hard_idx = torch.as_tensor(np.argmax(p_teacher, axis=1), device=device)
    teacher_logits = torch.log(pt.clamp_min(_EPS))
    distilled = planned_response_distill(
        module,
        lambda _inputs: teacher_logits,
        xb,
        hard_idx,
        temperature=temperature,
        alpha=1.0 - hard_weight,
        ridge=analytic_ridge,
        refinement_epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        optimizer=optimizer,
        seed=seed,
    )
    module = distilled.student.eval()

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
            "recipe": {
                "n": n,
                "dim": dim,
                "hidden": list(cfg["hidden_dims"]),
                "epochs": epochs,
                "lr": lr,
                "batch_size": batch_size,
                "optimizer": distilled.extra["optimizer"],
                "analytic_projection": distilled.extra["analytic_projection"],
            },
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
    texts = [str(text) for text in texts]
    if not texts:
        raise ValueError("soft-label distillation requires at least one text")
    probs = teacher_proba(texts)
    return distill_from_soft_labels(texts, probs, labels=labels, **kwargs)


def soft_agreement(student: TaskModel, teacher_probs: Sequence[Any], texts: Sequence[str]) -> float:
    """Mean KL divergence ``KL(teacher || student)`` over ``texts`` -- how faithfully the student matches
    the teacher's full soft distribution (0 = identical), the soft-distillation analog of
    :func:`mixle.task.distill.agreement`. Lower is better; use it to compare soft vs hard students."""
    texts = [str(text) for text in texts]
    p_teacher = _as_prob_matrix(teacher_probs, None)
    if p_teacher.shape[0] != len(texts):
        raise ValueError("teacher_probs must have one row per text")
    raw_student = student.adapter.proba_batch(student.model, texts)
    p_student = _as_prob_matrix(raw_student, p_teacher.shape[1])
    if p_student.shape != p_teacher.shape:
        raise ValueError(
            f"student probabilities have shape {p_student.shape}, expected {p_teacher.shape}"
        )
    kl = np.sum(
        p_teacher
        * (
            np.log(np.clip(p_teacher, _EPS, None))
            - np.log(np.clip(p_student, _EPS, None))
        ),
        axis=1,
    )
    return float(np.mean(kl))
