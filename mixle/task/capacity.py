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
from typing import Any

import numpy as np

from mixle.task.distill import _fit_mlp, _split_for_calibration, agreement, distill_from_labels
from mixle.task.model import HashedNGram, TaskModel, _ClassifierIO

#: the two rungs this module can fit; later rungs are recognized but may be unavailable in this environment.
DEFAULT_RUNGS: tuple[str, ...] = ("hashed_ngram", "embedding_head")

#: every rung name this module understands, in increasing-capacity order (used by :func:`climb_to`).
KNOWN_RUNGS: tuple[str, ...] = ("hashed_ngram", "embedding_head", "strong_encoder", "small_lm")

_BUILT_RUNGS = frozenset({"hashed_ngram", "embedding_head"})


class WordEmbeddingFeaturizer:
    """Average per-word embedding vectors from a fixed lookup table -- a dependency-free "embedding head" featurizer.

    Unlike :class:`~mixle.task.model.HashedNGram` (which treats distinct surface tokens as unrelated hash buckets),
    two words given nearby vectors in ``vectors`` produce nearby features regardless of their spelling -- the
    property a synonym-generalizing rule needs. A word missing from ``vectors`` falls back to a deterministic
    hashed sub-vector, so out-of-vocabulary text still produces a valid feature; it just earns no semantic
    generalization it was never given a vector for.
    """

    def __init__(self, vectors: dict[str, Sequence[float]] | None, dim: int, seed: int = 0) -> None:
        self.vectors = {str(k): np.asarray(v, dtype=np.float32) for k, v in (vectors or {}).items()}
        self.dim = int(dim)
        self.seed = int(seed)
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
    """One rung's measured outcome: its held-out agreement score, the fitted student (if built), and a note."""

    rung: str
    score: float | None
    model: TaskModel | None
    note: str = ""


@dataclass
class LadderResult:
    """The ladder's outcome: every rung's measured score, and the smallest rung meeting ``target`` (or ``None``)."""

    target: float
    rungs: list[RungResult]
    winner: str | None

    def ceiling(self, rung: str) -> float | None:
        """The measured score of ``rung``, or ``None`` if that rung was unavailable in this environment."""
        for r in self.rungs:
            if r.rung == rung:
                return r.score
        return None


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
    this environment: they are skipped with a note, never raised as an error.

    Returns a :class:`LadderResult` with every rung's measured score and either the smallest rung meeting
    ``target`` or ``winner=None`` with every built rung's ceiling attached -- "target unmet" is a valid
    result, never an exception.
    """
    texts = [str(t) for t in texts]
    label_list, train_texts, train_labels, hold_texts, hold_labels = _prepare_split(
        teacher_or_labels, texts, val_texts, val_labels, labels, calibration_frac, seed
    )

    results: list[RungResult] = []
    for rung in rungs:
        if rung not in KNOWN_RUNGS:
            raise ValueError(f"unknown rung {rung!r}; expected one of {KNOWN_RUNGS}")
        if rung not in _BUILT_RUNGS:
            results.append(RungResult(rung, None, None, note=f"rung {rung!r} not built in this environment"))
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

    winner = next((r.rung for r in results if r.score is not None and r.score >= target), None)
    return LadderResult(target=target, rungs=results, winner=winner)


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

    if val_texts is not None:
        hold_texts = [str(t) for t in val_texts]
        hold_labels = list(val_labels) if val_labels is not None else _teacher_labels(teacher, hold_texts)
        train_texts, train_labels = texts, train_labels_all
        label_list = list(labels) if labels is not None else sorted({str(y) for y in (*train_labels, *hold_labels)})
    else:
        train_texts, train_labels, hold_texts, hold_labels = _split_for_calibration(
            texts, train_labels_all, calibration_frac, seed
        )
        label_list = list(labels) if labels is not None else sorted({str(y) for y in train_labels_all})
    return label_list, train_texts, train_labels, hold_texts, hold_labels


def _teacher_labels(teacher: Callable[..., Any], texts: list[str]) -> list[Any]:
    out = teacher(texts)
    if isinstance(out, (list, tuple)) and len(out) == len(texts):
        return list(out)
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
        module, cfg, _steps_run = _fit_mlp(
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
