"""Distill a big teacher into a tiny local TaskModel: ``teacher`` labels data, a small student learns to match.

The vision in one function. The teacher is *any* callable that labels text -- a frontier LM behind an endpoint,
a slow rule, a human-curated map -- exercised once over an unlabeled corpus. The student is a small classifier
over dependency-free hashed n-gram features (:class:`~mixle.task.model.HashedNGram`), trained to reproduce the
teacher's labels, and returned as a :class:`~mixle.task.model.TaskModel` you save and call locally at a fraction
of the teacher's cost. ``agreement`` measures how faithfully the student mimics the teacher on held-out text --
the number :func:`~mixle.task.tune.tune_recipe` optimizes when it searches student recipes with ``mixle.doe``.

Only the student fit needs torch; the teacher is opaque. ``distill`` is deterministic given ``seed``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from mixle.task.model import HashedNGram, TaskModel, TextClassifierIO


def _as_batched(teacher: Callable[..., Any]) -> Callable[[list[str]], list[Any]]:
    """Accept either a per-item ``teacher(text)`` or a batched ``teacher(list)`` and present a batched view."""

    def batched(texts: list[str]) -> list[Any]:
        out = teacher(texts)
        if isinstance(out, (list, tuple)) and len(out) == len(texts):
            return list(out)
        # teacher was per-item (returned one label for a list, or we guessed wrong): call element-wise
        return [teacher(t) for t in texts]

    return batched


def distill(
    teacher: Callable[..., Any],
    texts: Sequence[str],
    *,
    labels: Sequence[str] | None = None,
    n: int = 3,
    dim: int = 256,
    hidden: Sequence[int] = (64,),
    epochs: int = 200,
    lr: float = 1e-2,
    seed: int = 0,
    task: str = "",
    device: str = "cpu",
) -> TaskModel:
    """Label ``texts`` with ``teacher``, fit a small student to match, and return a callable :class:`TaskModel`.

    ``n``/``dim`` size the hashed n-gram featurizer; ``hidden`` the student MLP. ``labels`` fixes the label set
    (else inferred from the teacher's outputs). The student's train-set agreement with the teacher is recorded
    in ``meta``.
    """
    import torch

    from mixle.models.neural import make_mlp

    texts = [str(t) for t in texts]
    teacher_labels = _as_batched(teacher)(texts)
    label_list = list(labels) if labels is not None else sorted({str(y) for y in teacher_labels})
    label_index = {y: i for i, y in enumerate(label_list)}
    y = np.asarray([label_index[str(t)] for t in teacher_labels], dtype=np.int64)

    feat = HashedNGram(n=n, dim=dim, seed=seed)
    x = feat.transform(texts)

    cfg = {
        "input_dim": dim,
        "hidden_dims": [int(h) for h in hidden],
        "output_dim": len(label_list),
        "activation": "relu",
    }
    torch.manual_seed(seed)
    module = make_mlp(**cfg).to(device)
    xt = torch.from_numpy(x).to(device)
    yt = torch.from_numpy(y).to(device)
    opt = torch.optim.Adam(module.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    module.train()
    for _ in range(int(epochs)):
        opt.zero_grad()
        loss = loss_fn(module(xt), yt)
        loss.backward()
        opt.step()

    adapter = TextClassifierIO(feat, label_list)
    student = TaskModel(
        module,
        adapter,
        builder="mixle.mlp",
        config=cfg,
        task=task or "distilled text classifier",
        meta={
            "distilled": True,
            "n_examples": len(texts),
            "labels": label_list,
            "recipe": {"n": n, "dim": dim, "hidden": list(cfg["hidden_dims"]), "epochs": epochs, "lr": lr},
        },
    )
    student.meta["train_agreement"] = agreement(student, teacher_labels, texts)
    return student


def agreement(student: TaskModel, teacher_labels: Sequence[Any], texts: Sequence[str]) -> float:
    """Fraction of ``texts`` where the student's label matches the teacher's -- distillation fidelity."""
    pred = student.batch(list(texts))
    tl = [str(t) for t in teacher_labels]
    return float(np.mean([p == t for p, t in zip(pred, tl)])) if texts else 0.0
