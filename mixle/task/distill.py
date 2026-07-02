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

from mixle.task.model import (
    HashedNGram,
    HashedRecord,
    RecordClassifierIO,
    StructuredClassifierIO,
    TaskModel,
    TextClassifierIO,
)


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
    texts = [str(t) for t in texts]
    teacher_labels = _as_batched(teacher)(texts)
    return distill_from_labels(
        texts,
        teacher_labels,
        labels=labels,
        n=n,
        dim=dim,
        hidden=hidden,
        epochs=epochs,
        lr=lr,
        seed=seed,
        task=task,
        device=device,
    )


def distill_from_labels(
    texts: Sequence[str],
    teacher_labels: Sequence[Any],
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
    """Fit a student from already-labeled ``(texts, teacher_labels)`` -- the teacher-free training core of ``distill``.

    Active labeling (:mod:`mixle.task.active`) uses this to avoid re-querying the teacher: it controls exactly
    which examples were paid for and passes their labels straight in. ``labels`` fixes the label set so a student
    trained on a partial sample still spans every class.
    """
    texts = [str(t) for t in texts]
    label_list, y = _encode_labels(teacher_labels, labels)
    feat = HashedNGram(n=n, dim=dim, seed=seed)
    module, cfg = _fit_mlp(feat.transform(texts), y, len(label_list), hidden, epochs, lr, seed, device)
    student = _student(
        module,
        cfg,
        TextClassifierIO(feat, label_list),
        task or "distilled text classifier",
        len(texts),
        label_list,
        {"n": n, "dim": dim, "hidden": list(cfg["hidden_dims"]), "epochs": epochs, "lr": lr},
    )
    student.meta["train_agreement"] = agreement(student, teacher_labels, texts)
    return student


def distill_records(
    teacher: Callable[..., Any],
    records: Sequence[Any],
    *,
    labels: Sequence[str] | None = None,
    dim: int = 256,
    hidden: Sequence[int] = (64,),
    epochs: int = 200,
    lr: float = 1e-2,
    seed: int = 0,
    task: str = "",
    device: str = "cpu",
) -> TaskModel:
    """Distill a teacher into a record classifier (``record -> label`` over tuples/dicts of mixed fields).

    The structured-data sibling of :func:`distill`: classify a transaction, route a ticket, categorize a record.
    Uses the hashing-trick :class:`~mixle.task.model.HashedRecord` featurizer, so it needs no fitted encoder.
    """
    records = list(records)
    teacher_labels = _as_batched(teacher)(records)
    return distill_records_from_labels(
        records,
        teacher_labels,
        labels=labels,
        dim=dim,
        hidden=hidden,
        epochs=epochs,
        lr=lr,
        seed=seed,
        task=task,
        device=device,
    )


def distill_records_from_labels(
    records: Sequence[Any],
    teacher_labels: Sequence[Any],
    *,
    labels: Sequence[str] | None = None,
    dim: int = 256,
    hidden: Sequence[int] = (64,),
    epochs: int = 200,
    lr: float = 1e-2,
    seed: int = 0,
    task: str = "",
    device: str = "cpu",
) -> TaskModel:
    """Teacher-free record-classifier training core (mirrors :func:`distill_from_labels` for structured records)."""
    records = list(records)
    label_list, y = _encode_labels(teacher_labels, labels)
    feat = HashedRecord(dim=dim, seed=seed)
    module, cfg = _fit_mlp(feat.transform(records), y, len(label_list), hidden, epochs, lr, seed, device)
    student = _student(
        module,
        cfg,
        RecordClassifierIO(feat, label_list),
        task or "distilled record classifier",
        len(records),
        label_list,
        {"dim": dim, "hidden": list(cfg["hidden_dims"]), "epochs": epochs, "lr": lr},
    )
    student.meta["train_agreement"] = agreement(student, teacher_labels, records)
    return student


def distill_structured(
    teacher: Callable[..., Any],
    records: Sequence[Any],
    *,
    labels: Sequence[str] | None = None,
    n_components: int = 1,
    min_gain: float = 0.0,
    n_bins: int = 4,
    max_its: int = 30,
    seed: int = 0,
    task: str = "",
) -> TaskModel:
    """Distill a teacher into a tiny **structured probabilistic** classifier -- a learned Bayesian network, not an MLP.

    The teacher labels ``records``; :func:`mixle.inference.structure.learn_structure` then discovers the dependency
    forest over the joint ``(field_1, ..., field_m, label)`` and fits it. The student classifies by the generative
    rule ``argmax_label P(features, label)`` -- and because ``softmax_label log P(features, label) = P(label |
    features)`` exactly, its confidence is a real posterior the cascade/calibration stack can trust. Unlike
    :func:`distill_records` (a hashed-feature MLP), this student is *interpretable* (``model.edges()`` lists the
    discovered dependencies), a few kilobytes on disk, and needs no torch to run.

    ``n_components > 1`` fits a :class:`~mixle.inference.structure.MixtureOfDependencyTrees` -- a latent-cluster
    student whose sub-structures differ by regime. Assumes a fixed record schema (see :class:`StructuredClassifierIO`).
    """
    records = list(records)
    teacher_labels = [str(t) for t in _as_batched(teacher)(records)]
    return distill_structured_from_labels(
        records,
        teacher_labels,
        labels=labels,
        n_components=n_components,
        min_gain=min_gain,
        n_bins=n_bins,
        max_its=max_its,
        seed=seed,
        task=task,
    )


def distill_structured_from_labels(
    records: Sequence[Any],
    teacher_labels: Sequence[Any],
    *,
    labels: Sequence[str] | None = None,
    n_components: int = 1,
    min_gain: float = 0.0,
    n_bins: int = 4,
    max_its: int = 30,
    seed: int = 0,
    task: str = "",
) -> TaskModel:
    """Teacher-free core of :func:`distill_structured`: fit a structured classifier from labeled records."""
    from mixle.inference.structure import learn_mixture_structure, learn_structure

    records = list(records)
    teacher_labels = [str(t) for t in teacher_labels]
    label_list = list(labels) if labels is not None else sorted(set(teacher_labels))
    field_keys, values = _record_schema(records)
    label_index = len(field_keys) if field_keys is not None else len(values[0])
    augmented = [v + (lab,) for v, lab in zip(values, teacher_labels)]  # label as the last joint field

    if n_components > 1:
        model = learn_mixture_structure(
            augmented, n_components, seed=seed, min_gain=min_gain, n_bins=n_bins, max_its=max_its
        )
        edges = model.components[0].edges()
    else:
        model = learn_structure(augmented, min_gain=min_gain, n_bins=n_bins, max_its=max_its)
        edges = model.edges()

    adapter = StructuredClassifierIO(field_keys, label_index, label_list)
    student = TaskModel(
        model,
        adapter,
        payload="json",
        task=task or "distilled structured classifier",
        meta={
            "distilled": True,
            "structured": True,
            "n_examples": len(records),
            "labels": label_list,
            "recipe": {"n_components": n_components, "min_gain": min_gain, "n_bins": n_bins},
            "edges": edges,
        },
    )
    student.meta["train_agreement"] = agreement(student, teacher_labels, records)
    return student


def _record_schema(records: Sequence[Any]) -> tuple[list[str] | None, list[tuple]]:
    """Canonical (field_keys, per-record value tuples). Dict records key by sorted first-record keys (fixed schema);
    tuple/list records are positional (``field_keys=None``). Raises on a schema mismatch across dict records."""
    first = records[0]
    if isinstance(first, dict):
        field_keys = sorted(first)
        values = []
        for r in records:
            if not isinstance(r, dict) or sorted(r) != field_keys:
                raise ValueError("structured distillation needs a fixed dict schema; record keys differ")
            values.append(tuple(r[k] for k in field_keys))
        return field_keys, values
    values = [tuple(r) if isinstance(r, (list, tuple)) else (r,) for r in records]
    width = len(values[0])
    if any(len(v) != width for v in values):
        raise ValueError("structured distillation needs fixed-width tuple records")
    return None, values


def _encode_labels(teacher_labels: Sequence[Any], labels: Sequence[str] | None) -> tuple[list[str], np.ndarray]:
    label_list = list(labels) if labels is not None else sorted({str(y) for y in teacher_labels})
    index = {y: i for i, y in enumerate(label_list)}
    return label_list, np.asarray([index[str(t)] for t in teacher_labels], dtype=np.int64)


def _fit_mlp(x: np.ndarray, y: np.ndarray, n_labels: int, hidden, epochs, lr, seed, device):
    """Train a small MLP classifier on features ``x`` and integer labels ``y``; return ``(module, config)``."""
    import torch

    from mixle.models.neural import make_mlp

    cfg = {
        "input_dim": int(x.shape[1]),
        "hidden_dims": [int(h) for h in hidden],
        "output_dim": int(n_labels),
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
    return module, cfg


def _student(module, cfg, adapter, task, n_examples, label_list, recipe) -> TaskModel:
    return TaskModel(
        module,
        adapter,
        builder="mixle.mlp",
        config=cfg,
        task=task,
        meta={"distilled": True, "n_examples": n_examples, "labels": label_list, "recipe": recipe},
    )


def agreement(student: TaskModel, teacher_labels: Sequence[Any], texts: Sequence[str]) -> float:
    """Fraction of ``texts`` where the student's label matches the teacher's -- distillation fidelity."""
    pred = student.batch(list(texts))
    tl = [str(t) for t in teacher_labels]
    return float(np.mean([p == t for p, t in zip(pred, tl)])) if texts else 0.0
