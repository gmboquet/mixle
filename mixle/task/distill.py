"""Distill a teacher callable into a local :class:`TaskModel`.

The teacher can be any callable that labels text: an LLM endpoint, a slow rule,
or a human-reviewed map. The student is a compact classifier over
dependency-free hashed n-gram features (:class:`~mixle.task.model.HashedNGram`),
trained to reproduce the teacher's labels and returned as a durable
:class:`~mixle.task.model.TaskModel`. ``agreement`` measures held-out fidelity
to the teacher and is the objective :func:`~mixle.task.tune.tune_recipe`
optimizes when it searches student recipes with ``mixle.doe``.

Only the student fit needs torch; the teacher is opaque. ``distill`` is deterministic given ``seed``.

``distill_for_routing``/``distill_records_for_routing`` are the routing-ready siblings: they hold out a
calibration slice, fit the student on the rest, and return a :class:`~mixle.task.calibrate.CalibratedTaskModel`
-- ``decide()``-able out of the box, so it drops straight into :class:`~mixle.task.cascade.Cascade` or
:class:`~mixle.task.router.Router` with no separate calibration step to remember or get wrong.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from mixle.task.calibrate import CalibratedTaskModel
from mixle.task.density import DensityGate
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


def _teacher_labels(teacher: Callable[..., Any], items: list, n_jobs: int = 1) -> list[Any]:
    """Label ``items`` with ``teacher``, fanning contiguous blocks across ``n_jobs`` THREADS.

    Threads, not processes: teachers are typically network-bound LM calls (:mod:`mixle.task.llm`),
    where parallel in-flight requests are the whole win and the GIL never bites; a CPU-bound Python
    teacher gains nothing here and should batch internally instead. Labels come back in input
    order, and ``n_jobs=1`` is byte-for-byte the sequential batched call. Blocks are ~4 per worker
    so one slow straggler cannot idle the pool.
    """
    if n_jobs <= 1 or len(items) <= 1:
        return _as_batched(teacher)(items)
    from concurrent.futures import ThreadPoolExecutor

    n_blocks = min(len(items), 4 * n_jobs)
    edges = [round(i * len(items) / n_blocks) for i in range(n_blocks + 1)]
    blocks = [items[a:b] for a, b in zip(edges[:-1], edges[1:]) if b > a]
    batched = _as_batched(teacher)
    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        parts = list(pool.map(batched, blocks))
    return [label for part in parts for label in part]


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
    n_jobs: int = 1,
) -> TaskModel:
    """Label ``texts`` with ``teacher``, fit a local student, and return a callable :class:`TaskModel`.

    ``n``/``dim`` size the hashed n-gram featurizer; ``hidden`` the student MLP. ``labels`` fixes the label set
    (else inferred from the teacher's outputs). The student's train-set agreement with the teacher is recorded
    in ``meta``. ``n_jobs > 1`` fans teacher labeling across that many threads (order-preserving; the win is
    parallel in-flight requests against a network-bound teacher) -- every ``distill_*`` teacher entry point
    takes the same knob.
    """
    texts = [str(t) for t in texts]
    teacher_labels = _teacher_labels(teacher, texts, n_jobs=n_jobs)
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
    module, cfg, steps_run, optimizer_receipt = _fit_mlp(
        feat.transform(texts), y, len(label_list), hidden, epochs, lr, seed, device
    )
    student = _student(
        module,
        cfg,
        TextClassifierIO(feat, label_list),
        task or "distilled text classifier",
        len(texts),
        label_list,
        {
            "n": n,
            "dim": dim,
            "hidden": list(cfg["hidden_dims"]),
            "epochs": epochs,
            "epochs_run": steps_run,
            "lr": lr,
            "optimizer": optimizer_receipt,
        },
    )
    student.meta["train_agreement"] = agreement(student, teacher_labels, texts)
    return student


def distill_for_routing(
    teacher: Callable[..., Any],
    texts: Sequence[str],
    *,
    labels: Sequence[str] | None = None,
    calibration_frac: float = 0.2,
    alpha: float = 0.1,
    n: int = 3,
    dim: int = 256,
    hidden: Sequence[int] = (64,),
    epochs: int = 200,
    lr: float = 1e-2,
    seed: int = 0,
    task: str = "",
    device: str = "cpu",
    density_gate: bool = False,
    density_gate_alpha: float = 0.05,
    n_jobs: int = 1,
) -> CalibratedTaskModel:
    """Label ``texts`` with ``teacher``, fit a student, and calibrate it for routing -- all in one call.

    A ``calibration_frac`` slice of the (teacher-)labeled data is held out from training and used to set a
    conformal threshold, so the returned :class:`~mixle.task.calibrate.CalibratedTaskModel` is immediately
    ``decide()``-able: confident, in-distribution inputs get the student's label; everything else is
    ``ESCALATE``. Pass it straight to :class:`~mixle.task.cascade.Cascade` (with ``teacher``) or
    :class:`~mixle.task.router.Router` for tiered serving -- no separate calibration split to manage by hand.
    Deterministic given ``seed``; the calibration slice is disjoint from the student's training data.

    ``density_gate=True`` additionally escalates inputs a softmax cannot see are atypical: see
    :func:`distill_from_labels_for_routing`.
    """
    texts = [str(t) for t in texts]
    teacher_labels = _teacher_labels(teacher, texts, n_jobs=n_jobs)
    return distill_from_labels_for_routing(
        texts,
        teacher_labels,
        labels=labels,
        calibration_frac=calibration_frac,
        alpha=alpha,
        n=n,
        dim=dim,
        hidden=hidden,
        epochs=epochs,
        lr=lr,
        seed=seed,
        task=task,
        device=device,
        density_gate=density_gate,
        density_gate_alpha=density_gate_alpha,
    )


def distill_from_labels_for_routing(
    texts: Sequence[str],
    teacher_labels: Sequence[Any],
    *,
    labels: Sequence[str] | None = None,
    calibration_frac: float = 0.2,
    alpha: float = 0.1,
    n: int = 3,
    dim: int = 256,
    hidden: Sequence[int] = (64,),
    epochs: int = 200,
    lr: float = 1e-2,
    seed: int = 0,
    task: str = "",
    device: str = "cpu",
    density_gate: bool = False,
    density_gate_alpha: float = 0.05,
) -> CalibratedTaskModel:
    """Teacher-free training core of :func:`distill_for_routing`: fit + calibrate from labels already in hand.

    Splits ``(texts, teacher_labels)`` into a training slice and a held-out ``calibration_frac`` slice (fixed by
    ``seed``), trains the student on the former via :func:`distill_from_labels`, then calibrates
    (:meth:`~mixle.task.calibrate.CalibratedTaskModel.calibrate`) on the latter. ``labels`` (if given, else
    inferred from all of ``teacher_labels`` before the split) is shared by both slices so a class that lands
    entirely on one side of the split doesn't shrink the label set out from under the other.

    ``density_gate=True`` fits a :class:`~mixle.task.density.DensityGate` on the *training* slice's features
    (reusing the student's own featurizer, so there is no second feature space to keep in sync), calibrates its
    OOD floor (``density_gate_alpha``) on the disjoint *calibration* slice, and wires it into the returned
    model -- an input whose ``log p(x)`` falls below that floor escalates even if the conformal set is a
    confident singleton.
    """
    label_list = list(labels) if labels is not None else sorted({str(y) for y in teacher_labels})
    train_texts, train_labels, cal_texts, cal_labels = _split_for_calibration(
        texts, teacher_labels, calibration_frac, seed
    )
    student = distill_from_labels(
        train_texts,
        train_labels,
        labels=label_list,
        n=n,
        dim=dim,
        hidden=hidden,
        epochs=epochs,
        lr=lr,
        seed=seed,
        task=task,
        device=device,
    )
    gate = (
        _fit_density_gate(student, train_texts, cal_texts, alpha=density_gate_alpha, seed=seed)
        if density_gate
        else None
    )
    return CalibratedTaskModel(student, alpha=alpha, density_gate=gate).calibrate(cal_texts, cal_labels)


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
    n_jobs: int = 1,
) -> TaskModel:
    """Distill a teacher into a record classifier (``record -> label`` over tuples/dicts of mixed fields).

    The structured-data sibling of :func:`distill`: classify a transaction, route a ticket, categorize a record.
    Uses the hashing-trick :class:`~mixle.task.model.HashedRecord` featurizer, so it needs no fitted encoder.
    """
    records = list(records)
    teacher_labels = _teacher_labels(teacher, records, n_jobs=n_jobs)
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
    module, cfg, steps_run, optimizer_receipt = _fit_mlp(
        feat.transform(records), y, len(label_list), hidden, epochs, lr, seed, device
    )
    student = _student(
        module,
        cfg,
        RecordClassifierIO(feat, label_list),
        task or "distilled record classifier",
        len(records),
        label_list,
        {
            "dim": dim,
            "hidden": list(cfg["hidden_dims"]),
            "epochs": epochs,
            "epochs_run": steps_run,
            "lr": lr,
            "optimizer": optimizer_receipt,
        },
    )
    student.meta["train_agreement"] = agreement(student, teacher_labels, records)
    return student


def distill_records_for_routing(
    teacher: Callable[..., Any],
    records: Sequence[Any],
    *,
    labels: Sequence[str] | None = None,
    calibration_frac: float = 0.2,
    alpha: float = 0.1,
    dim: int = 256,
    hidden: Sequence[int] = (64,),
    epochs: int = 200,
    lr: float = 1e-2,
    seed: int = 0,
    task: str = "",
    device: str = "cpu",
    density_gate: bool = False,
    density_gate_alpha: float = 0.05,
    n_jobs: int = 1,
) -> CalibratedTaskModel:
    """The structured-record sibling of :func:`distill_for_routing`: fit + calibrate a record classifier
    in one call, returning a routing-ready :class:`~mixle.task.calibrate.CalibratedTaskModel`."""
    records = list(records)
    teacher_labels = _teacher_labels(teacher, records, n_jobs=n_jobs)
    return distill_records_from_labels_for_routing(
        records,
        teacher_labels,
        labels=labels,
        calibration_frac=calibration_frac,
        alpha=alpha,
        dim=dim,
        hidden=hidden,
        epochs=epochs,
        lr=lr,
        seed=seed,
        task=task,
        device=device,
        density_gate=density_gate,
        density_gate_alpha=density_gate_alpha,
    )


def distill_records_from_labels_for_routing(
    records: Sequence[Any],
    teacher_labels: Sequence[Any],
    *,
    labels: Sequence[str] | None = None,
    calibration_frac: float = 0.2,
    alpha: float = 0.1,
    dim: int = 256,
    hidden: Sequence[int] = (64,),
    epochs: int = 200,
    lr: float = 1e-2,
    seed: int = 0,
    task: str = "",
    device: str = "cpu",
    density_gate: bool = False,
    density_gate_alpha: float = 0.05,
) -> CalibratedTaskModel:
    """Teacher-free training core of :func:`distill_records_for_routing` (mirrors
    :func:`distill_from_labels_for_routing` for structured records). ``density_gate=True`` fits the OOD gate
    on the training slice's record features and calibrates it on the calibration slice, same as the text path."""
    label_list = list(labels) if labels is not None else sorted({str(y) for y in teacher_labels})
    train_records, train_labels, cal_records, cal_labels = _split_for_calibration(
        records, teacher_labels, calibration_frac, seed
    )
    student = distill_records_from_labels(
        train_records,
        train_labels,
        labels=label_list,
        dim=dim,
        hidden=hidden,
        epochs=epochs,
        lr=lr,
        seed=seed,
        task=task,
        device=device,
    )
    gate = (
        _fit_density_gate(student, train_records, cal_records, alpha=density_gate_alpha, seed=seed)
        if density_gate
        else None
    )
    return CalibratedTaskModel(student, alpha=alpha, density_gate=gate).calibrate(cal_records, cal_labels)


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
    n_jobs: int = 1,
) -> TaskModel:
    """Distill a teacher into a structured probabilistic classifier -- a learned Bayesian network, not an MLP.

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
    teacher_labels = [str(t) for t in _teacher_labels(teacher, records, n_jobs=n_jobs)]
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
        model = learn_structure(
            augmented, min_gain=min_gain, n_bins=n_bins, max_its=max_its, rng=np.random.RandomState(seed)
        )
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


def _split_for_calibration(
    items: Sequence[Any], teacher_labels: Sequence[Any], calibration_frac: float, seed: int
) -> tuple[list[Any], list[Any], list[Any], list[Any]]:
    """Split ``(items, teacher_labels)`` into disjoint ``(train_items, train_labels, cal_items, cal_labels)``.

    The calibration slice must be unseen by the student's training fit -- that disjointness is exactly what
    makes the conformal coverage guarantee (:mod:`mixle.task.calibrate`) real rather than optimistic. The split
    is a fixed permutation of ``seed``, so it is reproducible and shared across the text/record variants.
    """
    if not 0.0 < calibration_frac < 1.0:
        raise ValueError(f"calibration_frac must be in (0, 1), got {calibration_frac}")
    items, teacher_labels = list(items), list(teacher_labels)
    n_total = len(items)
    n_cal = max(1, int(round(n_total * calibration_frac)))
    if n_cal >= n_total:
        raise ValueError(
            f"calibration_frac={calibration_frac} leaves no training examples for {n_total} total example(s)"
        )
    perm = np.random.RandomState(seed).permutation(n_total)
    cal_idx, train_idx = perm[:n_cal], perm[n_cal:]
    return (
        [items[i] for i in train_idx],
        [teacher_labels[i] for i in train_idx],
        [items[i] for i in cal_idx],
        [teacher_labels[i] for i in cal_idx],
    )


def _fit_density_gate(
    student: TaskModel, train_items: Sequence[Any], cal_items: Sequence[Any], *, alpha: float, seed: int
) -> DensityGate:
    """Fit an OOD gate on the student's own featurizer: density from ``train_items``, floor from ``cal_items``.

    Fitting the density and its floor on the *same* slice would let the gate calibrate against the exact data
    it was fit on -- optimistic, the same mistake conformal calibration avoids. So the density is fit on
    ``train_items`` (disjoint from the calibration slice used to set the conformal threshold), then the floor is
    set from the ``alpha``-quantile of ``cal_items``' log-density under that fitted density: a genuinely
    held-out threshold, consistent with how :meth:`~mixle.task.calibrate.CalibratedTaskModel.calibrate` treats
    the same split.
    """
    gate = DensityGate(student.adapter.featurizer)
    gate.fit(train_items, seed=seed)
    gate.log_threshold = float(np.quantile(gate.log_density(cal_items), alpha))
    return gate


# Early-stopping schedule for _fit_mlp: check the training loss every _ES_CHECK_EVERY gradient steps, stop once
# it hasn't improved by _ES_MIN_DELTA for _ES_PATIENCE consecutive checks. Internal constants, not exposed on the
# public distill*() signatures -- epochs stays the one knob callers see; it now means "ceiling", not "exact count".
_ES_CHECK_EVERY = 10
_ES_PATIENCE = 3
_ES_MIN_DELTA = 1e-3


def _fit_mlp(x: np.ndarray, y: np.ndarray, n_labels: int, hidden, epochs, lr, seed, device):
    """Train an MLP classifier on features ``x`` and integer labels ``y``; return ``(module, config, steps_run)``.

    Wraps the module in a :class:`~mixle.models.NeuralCategorical` leaf and fits it through the ordinary
    :func:`~mixle.inference.optimize` entry point -- the same declare-a-leaf/call-optimize path every other
    mixle model goes through, rather than a bespoke torch loop. Training runs in ``_ES_CHECK_EVERY``-step
    chunks, each warm-started from the last (``optimize(..., prev_estimate=leaf, max_its=1)``), so ``epochs``
    is a CEILING: once the mean training cross-entropy stops improving, further chunks are skipped rather than
    always spending the full requested step count -- a task that converges early trains faster for it, and a
    harder one still gets up to ``epochs`` steps. ``steps_run`` (the actual gradient-step count) is recorded
    by the caller as ``recipe["epochs_run"]`` so the speedup is observable, not just internal. The fourth return
    value is the final chunk's optimizer receipt.
    """
    import torch

    from mixle.inference import optimize
    from mixle.models import NeuralCategorical
    from mixle.models.neural import make_mlp

    cfg = {
        "input_dim": int(x.shape[1]),
        "hidden_dims": [int(h) for h in hidden],
        "output_dim": int(n_labels),
        "activation": "relu",
    }
    torch.manual_seed(seed)
    module = make_mlp(**cfg).to(device)
    data = list(zip(x, y))
    enc = (np.asarray(x, dtype=float), np.asarray(y, dtype=int))

    remaining, best_loss, stall, fit_module, steps_run = int(epochs), float("inf"), 0, module, 0
    optimizer_receipt = None
    while remaining > 0:
        chunk = min(_ES_CHECK_EVERY, remaining)
        leaf = NeuralCategorical(fit_module, m_steps=chunk, lr=float(lr), device=device)
        # Each call is one checkpoint in a longer gradient trajectory. Returning the checkpoint's final
        # finite update lets the next chunk cross a shallow stochastic valley; early stopping below
        # still selects progress using the aligned full-data cross-entropy.
        fit = optimize(
            data,
            leaf.estimator(),
            prev_estimate=leaf,
            max_its=1,
            monotone=False,
            track_best=False,
            out=None,
        )
        fit_module = fit.module
        optimizer_receipt = fit.fit_receipt.get("optimizer_plan")
        remaining -= chunk
        steps_run += chunk
        loss = -float(np.mean(fit.seq_log_density(enc)))
        if best_loss - loss > _ES_MIN_DELTA:
            best_loss, stall = loss, 0
        else:
            stall += 1
            if stall >= _ES_PATIENCE:
                break
    return fit_module, cfg, steps_run, {"name": "auto", "plan": optimizer_receipt}


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
