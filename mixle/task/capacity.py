"""The capacity ladder: fit a student at each rung of increasing representation family, and report where
it stops matching the teacher.

Distillation (:mod:`mixle.task.distill`) and recipe search (:mod:`mixle.task.tune`) both assume the
student's representation *family* is fixed (hashed n-grams) and search knobs within it. Some teachers
need a richer family -- a rule that generalizes across synonyms a hashed n-gram featurizer cannot see, for
instance. :func:`capacity_ladder` climbs a small ordered set of representation families ("rungs"), measures
each rung's held-out agreement with the teacher, and returns the smallest rung that meets a target -- or a
measured "not capturable at these rungs" outcome with every rung's ceiling attached, never an exception.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np

from mixle.task.distill import _fit_mlp, _split_for_calibration, agreement, distill_from_labels
from mixle.task.model import HashedNGram, TaskModel, _ClassifierIO

#: the two rungs this module can fit; later rungs are recognized but may be unavailable in this environment.
DEFAULT_RUNGS: tuple[str, ...] = ("hashed_ngram", "embedding_head")

#: every rung name this module understands, in increasing-capacity order (used by :func:`climb_to`).
KNOWN_RUNGS: tuple[str, ...] = ("hashed_ngram", "embedding_head", "strong_encoder", "small_lm")

_BUILT_RUNGS = frozenset({"hashed_ngram", "embedding_head"})


def _exact_int(value: Any, name: str, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _finite_real(
    value: Any,
    name: str,
    *,
    minimum: float,
    maximum: float | None = None,
    lower_open: bool = False,
    upper_open: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    lower_bad = result <= minimum if lower_open else result < minimum
    upper_bad = maximum is not None and (result >= maximum if upper_open else result > maximum)
    if not np.isfinite(result) or lower_bad or upper_bad:
        upper = "inf" if maximum is None else str(maximum)
        interval = f"{'(' if lower_open else '['}{minimum}, {upper}{')' if upper_open else ']'}"
        raise ValueError(f"{name} must be finite and in {interval}")
    return result


def _materialize(value: Any, name: str, *, nonempty: bool = True) -> list[Any]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence, not a string")
    try:
        result = list(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a sequence") from exc
    if nonempty and not result:
        raise ValueError(f"{name} must be nonempty")
    return result


class WordEmbeddingFeaturizer:
    """Average per-word embedding vectors from a fixed lookup table -- a dependency-free "embedding head" featurizer.

    Unlike :class:`~mixle.task.model.HashedNGram` (which treats distinct surface tokens as unrelated hash buckets),
    two words given nearby vectors in ``vectors`` produce nearby features regardless of their spelling -- the
    property a synonym-generalizing rule needs. A word missing from ``vectors`` falls back to a deterministic
    hashed sub-vector, so out-of-vocabulary text still produces a valid feature; it just earns no semantic
    generalization it was never given a vector for.
    """

    def __init__(self, vectors: dict[str, Sequence[float]] | None, dim: int, seed: int = 0) -> None:
        self.dim = _exact_int(dim, "dim", minimum=1)
        self.seed = _exact_int(seed, "seed", minimum=0)
        if self.seed > np.iinfo(np.uint32).max:
            raise ValueError("seed must fit in an unsigned 32-bit integer")
        if vectors is not None and not isinstance(vectors, dict):
            raise TypeError("vectors must be a dictionary or None")
        self.vectors = {}
        for key, value in (vectors or {}).items():
            if not isinstance(key, str) or not key:
                raise ValueError("embedding keys must be non-empty strings")
            try:
                vector = np.asarray(value, dtype=np.float32)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"embedding vector {key!r} must be numeric") from exc
            if vector.shape != (self.dim,) or np.any(~np.isfinite(vector)):
                raise ValueError(f"embedding vector {key!r} must be finite with shape ({self.dim},).")
            vector = vector.copy()
            vector.setflags(write=False)
            self.vectors[key] = vector
        self._fallback = HashedNGram(n=3, dim=self.dim, seed=self.seed)

    def transform(self, texts: list[str]) -> np.ndarray:
        """Map texts to normalized embedding features with hashed fallback rows."""
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            words = str(t).lower().split()
            vecs = [self.vectors[w] for w in words if w in self.vectors]
            out[i] = np.mean(vecs, axis=0) if vecs else self._fallback.transform([t])[0]
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms > 0, norms, 1.0)

    def to_spec(self) -> dict[str, Any]:
        """Serialize embedding vectors and fallback hashing settings."""
        return {"vectors": {k: v.tolist() for k, v in self.vectors.items()}, "dim": self.dim, "seed": self.seed}

    @classmethod
    def from_spec(cls, spec: dict[str, Any]) -> WordEmbeddingFeaturizer:
        """Reconstruct the embedding featurizer from an artifact spec."""
        return cls(spec["vectors"], spec["dim"], spec["seed"])


class EmbeddingHeadIO(_ClassifierIO):
    """``str -> label`` classifier over :class:`WordEmbeddingFeaturizer` features -- the "embedding_head" rung."""

    kind = "embedding_head_classifier"
    _featurizer_cls = WordEmbeddingFeaturizer

    def __init__(self, featurizer: WordEmbeddingFeaturizer, labels: list[str]) -> None:
        super().__init__(featurizer, labels)


@dataclass
class RungResult:
    """One rung's measured outcome or explicit record that it was not evaluated."""

    rung: str
    score: float | None
    model: TaskModel | None
    note: str = ""
    status: str = "measured"


@dataclass
class LadderResult:
    """Every requested rung plus an outcome that distinguishes a measured ceiling from missing evaluation."""

    target: float
    rungs: list[RungResult]
    winner: str | None
    outcome: str

    def ceiling(self, rung: str) -> float | None:
        """The measured score of ``rung``, or ``None`` if that rung was unavailable in this environment."""
        for r in self.rungs:
            if r.rung == rung:
                return r.score
        return None

    def fully_evaluated(self) -> bool:
        """Whether every requested rung produced a held-out measurement."""
        return all(rung.status == "measured" for rung in self.rungs)


def _validate_word_vectors(word_vectors: Any) -> dict[str, Sequence[float]] | None:
    if word_vectors is None:
        return None
    if not isinstance(word_vectors, dict):
        raise TypeError("word_vectors must be a dictionary or None")
    if not word_vectors:
        return {}
    dimensions: set[int] = set()
    for key, value in word_vectors.items():
        if not isinstance(key, str) or not key:
            raise ValueError("word_vectors keys must be non-empty strings")
        try:
            vector = np.asarray(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"word vector {key!r} must be numeric") from exc
        if vector.ndim != 1 or vector.size == 0 or vector.dtype.kind not in {"i", "u", "f"}:
            raise ValueError(f"word vector {key!r} must be a non-empty real vector")
        try:
            finite = bool(np.all(np.isfinite(vector)))
        except TypeError as exc:
            raise ValueError(f"word vector {key!r} must be numeric") from exc
        if not finite:
            raise ValueError(f"word vector {key!r} must be finite")
        dimensions.add(int(vector.size))
    if len(dimensions) != 1:
        raise ValueError("all word vectors must have the same positive dimension")
    return word_vectors


def _validate_experiment(
    teacher_or_labels: Any,
    texts: Any,
    *,
    target: Any,
    rungs: Any,
    val_texts: Any,
    val_labels: Any,
    labels: Any,
    word_vectors: Any,
    calibration_frac: Any,
    n: Any,
    dim: Any,
    hidden: Any,
    epochs: Any,
    lr: Any,
    seed: Any,
    device: Any,
) -> tuple[
    Callable[..., Any] | list[Any],
    list[str],
    float,
    list[str],
    list[str] | None,
    list[Any] | None,
    list[str] | None,
    dict[str, Sequence[float]] | None,
    float,
    int,
    int,
    tuple[int, ...],
    int,
    float,
    int,
    str,
]:
    text_list = [str(text) for text in _materialize(texts, "texts")]
    target_value = _finite_real(target, "target", minimum=0.0, maximum=1.0)
    calibration_value = _finite_real(
        calibration_frac,
        "calibration_frac",
        minimum=0.0,
        maximum=1.0,
        lower_open=True,
        upper_open=True,
    )
    rung_list = _materialize(rungs, "rungs")
    if any(not isinstance(rung, str) or rung not in KNOWN_RUNGS for rung in rung_list):
        raise ValueError(f"rungs must contain only names from {KNOWN_RUNGS}")
    if len(set(rung_list)) != len(rung_list):
        raise ValueError("rungs must be unique")
    positions = [KNOWN_RUNGS.index(rung) for rung in rung_list]
    if positions != sorted(positions):
        raise ValueError("rungs must be ordered from lower to higher capacity")

    n_value = _exact_int(n, "n", minimum=1)
    dim_value = _exact_int(dim, "dim", minimum=1)
    hidden_list = tuple(_exact_int(width, "hidden width", minimum=1) for width in _materialize(hidden, "hidden"))
    epochs_value = _exact_int(epochs, "epochs", minimum=1)
    lr_value = _finite_real(lr, "lr", minimum=0.0, lower_open=True)
    seed_value = _exact_int(seed, "seed", minimum=0)
    if seed_value > np.iinfo(np.uint32).max:
        raise ValueError("seed must fit in an unsigned 32-bit integer")
    if not isinstance(device, str) or not device:
        raise ValueError("device must be a non-empty string")

    if callable(teacher_or_labels):
        source: Callable[..., Any] | list[Any] = teacher_or_labels
    else:
        source = _materialize(teacher_or_labels, "teacher_or_labels")
        if len(source) != len(text_list):
            raise ValueError("teacher labels and texts must have the same length")

    validation_texts = None if val_texts is None else [str(text) for text in _materialize(val_texts, "val_texts")]
    validation_labels = None if val_labels is None else _materialize(val_labels, "val_labels")
    if validation_texts is None and validation_labels is not None:
        raise ValueError("val_labels require val_texts")
    if validation_texts is not None:
        if validation_labels is None and not callable(source):
            raise ValueError("val_labels are required when teacher_or_labels is a label sequence")
        if validation_labels is not None and len(validation_labels) != len(validation_texts):
            raise ValueError("val_labels and val_texts must have the same length")

    declared_labels = None if labels is None else [str(label) for label in _materialize(labels, "labels")]
    if declared_labels is not None and len(set(declared_labels)) != len(declared_labels):
        raise ValueError("labels must be unique")
    vectors = _validate_word_vectors(word_vectors)
    return (
        source,
        text_list,
        target_value,
        rung_list,
        validation_texts,
        validation_labels,
        declared_labels,
        vectors,
        calibration_value,
        n_value,
        dim_value,
        hidden_list,
        epochs_value,
        lr_value,
        seed_value,
        device,
    )


def capacity_ladder(
    teacher_or_labels: Callable[..., Any] | Sequence[Any],
    texts: Sequence[str],
    *,
    target: float,
    rungs: Sequence[str] = DEFAULT_RUNGS,
    val_texts: Sequence[str] | None = None,
    val_labels: Sequence[Any] | None = None,
    labels: Sequence[str] | None = None,
    word_vectors: dict[str, Sequence[float]] | None = None,
    calibration_frac: float = 0.3,
    n: int = 3,
    dim: int = 256,
    hidden: Sequence[int] = (64,),
    epochs: int = 200,
    lr: float = 1e-2,
    seed: int = 0,
    device: str = "cpu",
) -> LadderResult:
    """Fit a student at each rung of ``rungs`` (increasing representation family) and measure held-out agreement.

    ``teacher_or_labels`` is either a callable teacher (labels ``texts`` and, if given separately, ``val_texts``)
    or a sequence of labels already aligned with ``texts`` -- mirroring the ``distill``/``distill_from_labels``
    duality. When ``val_texts``/``val_labels`` are not given, a ``calibration_frac`` held-out slice of
    ``(texts, teacher labels)`` is used (same split machinery as routing calibration), so a paraphrase/synonym
    generalization gap between train and held-out is measurable even with a single corpus.

    ``word_vectors`` (word -> dense vector) is the only thing that makes the ``"embedding_head"`` rung
    semantically richer than ``"hashed_ngram"`` -- without it, that rung still builds (never skipped, it is one
    of the two minimum rungs) but falls back to hashed features per out-of-vocabulary word, so it will not beat
    ``"hashed_ngram"``. ``"strong_encoder"``/``"small_lm"`` are recognized rung *names* with no estimator wired in
    this environment: they are recorded as ``status="not_evaluated"``.

    ``LadderResult.outcome`` is ``"target_met"``, ``"capacity_ceiling_measured"`` (all requested
    rungs were evaluated), or ``"not_evaluated"`` (the target was unmet but at least one requested
    family was unavailable). The latter must not be reported as a measured capacity ceiling.
    """
    (
        teacher_or_labels,
        texts,
        target,
        rungs,
        val_texts,
        val_labels,
        labels,
        word_vectors,
        calibration_frac,
        n,
        dim,
        hidden,
        epochs,
        lr,
        seed,
        device,
    ) = _validate_experiment(
        teacher_or_labels,
        texts,
        target=target,
        rungs=rungs,
        val_texts=val_texts,
        val_labels=val_labels,
        labels=labels,
        word_vectors=word_vectors,
        calibration_frac=calibration_frac,
        n=n,
        dim=dim,
        hidden=hidden,
        epochs=epochs,
        lr=lr,
        seed=seed,
        device=device,
    )
    label_list, train_texts, train_labels, hold_texts, hold_labels = _prepare_split(
        teacher_or_labels, texts, val_texts, val_labels, labels, calibration_frac, seed
    )

    results: list[RungResult] = []
    for rung in rungs:
        if rung not in KNOWN_RUNGS:
            raise ValueError(f"unknown rung {rung!r}; expected one of {KNOWN_RUNGS}")
        if rung not in _BUILT_RUNGS:
            results.append(
                RungResult(
                    rung,
                    None,
                    None,
                    note=f"rung {rung!r} not built in this environment",
                    status="not_evaluated",
                )
            )
            continue
        student = _fit_rung(
            rung,
            train_texts,
            train_labels,
            label_list,
            word_vectors=word_vectors,
            n=n,
            dim=dim,
            hidden=hidden,
            epochs=epochs,
            lr=lr,
            seed=seed,
            device=device,
        )
        score = agreement(student, hold_labels, hold_texts)
        results.append(RungResult(rung, score, student))

    winner = next((r.rung for r in results if r.status == "measured" and r.score >= target), None)
    if winner is not None:
        outcome = "target_met"
    elif all(result.status == "measured" for result in results):
        outcome = "capacity_ceiling_measured"
    else:
        outcome = "not_evaluated"
    return LadderResult(target=target, rungs=results, winner=winner, outcome=outcome)


def climb_to(fault: Any, *, rungs: Sequence[str] = KNOWN_RUNGS) -> str:
    """Given a refinement-loop fault localized to a saturated leaf's current rung, return the next rung up.

    ``fault`` is either a bare rung name or an object naming its current rung via a ``rung`` or ``dominant``
    attribute (the shape :func:`~mixle.inference.explain.diagnose`'s ``FaultReport`` will eventually carry) --
    this lets a caller climb straight to the next rung for the one saturated leaf, without re-running the whole
    ladder. Raises ``ValueError`` if the current rung is already the top of ``rungs``.
    """
    current = fault if isinstance(fault, str) else getattr(fault, "rung", None) or getattr(fault, "dominant", None)
    if current not in rungs:
        raise ValueError(f"unknown current rung {current!r}; expected one of {rungs}")
    idx = rungs.index(current)
    if idx + 1 >= len(rungs):
        raise ValueError(f"rung {current!r} is already the ceiling of {rungs}")
    return rungs[idx + 1]


def _prepare_split(
    teacher_or_labels: Callable[..., Any] | Sequence[Any],
    texts: list[str],
    val_texts: Sequence[str] | None,
    val_labels: Sequence[Any] | None,
    labels: Sequence[str] | None,
    calibration_frac: float,
    seed: int,
) -> tuple[list[str], list[str], list[Any], list[str], list[Any]]:
    teacher = teacher_or_labels if callable(teacher_or_labels) else None
    if teacher is not None:
        train_labels_all = _teacher_labels(teacher, texts)
    else:
        train_labels_all = list(teacher_or_labels)
    if len(train_labels_all) != len(texts):
        raise ValueError("teacher labels and texts must have the same length")

    if val_texts is not None:
        hold_texts = [str(t) for t in val_texts]
        if not hold_texts:
            raise ValueError("val_texts must be nonempty")
        if val_labels is None:
            if teacher is None:
                raise ValueError("val_labels are required when teacher_or_labels is a label sequence")
            hold_labels = _teacher_labels(teacher, hold_texts)
        else:
            hold_labels = list(val_labels)
        if len(hold_labels) != len(hold_texts):
            raise ValueError("val_labels and val_texts must have the same length")
        train_texts, train_labels = texts, train_labels_all
        label_list = list(labels) if labels is not None else sorted({str(y) for y in (*train_labels, *hold_labels)})
    else:
        train_texts, train_labels, hold_texts, hold_labels = _split_for_calibration(
            texts, train_labels_all, calibration_frac, seed
        )
        label_list = list(labels) if labels is not None else sorted({str(y) for y in train_labels_all})
    if not train_texts or not hold_texts:
        raise ValueError("capacity ladder requires nonempty training and holdout sets")
    if not label_list or len(set(label_list)) != len(label_list):
        raise ValueError("labels must be nonempty and unique")
    unknown = sorted({str(label) for label in (*train_labels, *hold_labels)} - set(label_list))
    if unknown:
        raise ValueError(f"observed labels are outside the declared label support: {unknown!r}")
    return label_list, train_texts, train_labels, hold_texts, hold_labels


def _teacher_labels(teacher: Callable[..., Any], texts: list[str]) -> list[Any]:
    if not callable(teacher):
        raise TypeError("teacher must be callable")
    out = teacher(texts)
    if not isinstance(out, (str, bytes)) and not np.isscalar(out):
        try:
            labels = list(out)
        except TypeError:
            labels = []
        else:
            if len(labels) != len(texts):
                raise ValueError(f"batch teacher returned {len(labels)} labels for {len(texts)} texts")
            return labels
    return [teacher(t) for t in texts]


def _fit_rung(
    rung: str,
    train_texts: list[str],
    train_labels: Sequence[Any],
    label_list: list[str],
    *,
    word_vectors: dict[str, Sequence[float]] | None,
    n: int,
    dim: int,
    hidden: Sequence[int],
    epochs: int,
    lr: float,
    seed: int,
    device: str,
) -> TaskModel:
    if rung == "hashed_ngram":
        return distill_from_labels(
            train_texts,
            train_labels,
            labels=label_list,
            n=n,
            dim=dim,
            hidden=hidden,
            epochs=epochs,
            lr=lr,
            seed=seed,
            task="capacity ladder: hashed_ngram",
            device=device,
        )
    if rung == "embedding_head":
        vec_dim = dim
        if word_vectors:
            vec_dim = len(next(iter(word_vectors.values())))
        label_index = {y: i for i, y in enumerate(label_list)}
        y = np.asarray([label_index[str(t)] for t in train_labels], dtype=np.int64)
        featurizer = WordEmbeddingFeaturizer(word_vectors, dim=vec_dim, seed=seed)
        # unpack ALL of _fit_mlp's returns: this call site missed a return-arity change TWICE (#47's
        # 2->3, then the optimizer-receipt 3->4) because it only runs where safetensors is installed
        # and CI skips it -- keep the arity in sync with distill.py's own call sites
        module, cfg, _steps_run, _optimizer_receipt = _fit_mlp(
            featurizer.transform(train_texts), y, len(label_list), hidden, epochs, lr, seed, device
        )
        student = TaskModel(
            module,
            EmbeddingHeadIO(featurizer, label_list),
            builder="mixle.mlp",
            config=cfg,
            task="capacity ladder: embedding_head",
            meta={"distilled": True, "n_examples": len(train_texts), "labels": label_list, "recipe": cfg},
        )
        student.meta["train_agreement"] = agreement(student, train_labels, train_texts)
        return student
    raise ValueError(f"rung {rung!r} has no fitting path wired")
