"""A GENERATIVE text student -- per-class token models, so the classifier owns a real ``p(x)``.

The moat meeting the product: instead of a discriminative hashed-feature net, the student is a set of
mixle generative models -- one multinomial ``p(tokens | class)`` per label (a token ``Categorical`` fit by
the ordinary estimator machinery; a document scores as the sum of its token logs) plus class log-priors. Classification is the exact posterior
``P(class | x) (softmax of the per-class log-joints)``, and -- the part a softmax net cannot offer --
``log p(x) = logsumexp_c log p(x, c)`` comes for free, so the same student scores how *typical* an input
is without a separate density gate.

Rare and unseen tokens clamp to ``<unk>`` (vocabulary = tokens seen at least ``min_count`` times), and
every class is Laplace-smoothed over the SHARED vocabulary — so a word the class never saw (or a novel
word) dims its likelihood smoothly instead of vetoing it to ``-inf``.

Drop-in with the rest of the spine: ``distill_text_generative(teacher, texts)`` returns a
:class:`~mixle.task.model.TaskModel` whose adapter exposes ``proba_batch``, so conformal calibration,
``solve(student="generative")``, cascades, and routers all work unchanged.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from mixle.task.extract import tokenize
from mixle.task.model import TaskModel, register_adapter

_UNK = "<unk>"


class GenerativeTextIO:
    """Adapter over ``{label: fitted p(tokens|label)}`` + log-priors: exact posteriors and ``log p(x)``."""

    kind = "generative_text"

    def __init__(self, labels: list[str], vocab: list[str], log_prior: list[float]) -> None:
        self.labels = [str(label) for label in labels]
        self._vocab_list = [str(token) for token in vocab]
        self.log_prior = [float(v) for v in log_prior]
        if not self.labels or any(not label for label in self.labels) or len(set(self.labels)) != len(self.labels):
            raise ValueError("generative text labels must be nonempty and unique")
        if (
            not self._vocab_list
            or _UNK not in self._vocab_list
            or len(set(self._vocab_list)) != len(self._vocab_list)
        ):
            raise ValueError("generative text vocabulary must be unique and include '<unk>'")
        if len(self.log_prior) != len(self.labels) or not np.all(np.isfinite(self.log_prior)):
            raise ValueError("generative text log priors must be finite and aligned with labels")
        prior_mass = float(np.exp(self.log_prior).sum())
        if not np.isfinite(prior_mass) or not np.isclose(prior_mass, 1.0, atol=1e-10):
            raise ValueError("generative text class priors must sum to one")
        self.vocab = set(self._vocab_list)

    def _tokens(self, text: str) -> list[str]:
        toks = [w.lower() for w, _s, _e in tokenize(str(text))]
        return [t if t in self.vocab else _UNK for t in toks] or [_UNK]

    def logits_batch(self, model: Any, raw_inputs: list[Any]) -> np.ndarray:
        """``log P(tokens, label)`` per label -- an ``(m, K)`` matrix (multinomial: sum of token logs)."""
        if not raw_inputs:  # empty batch: (0, K), skip the per-class encode/score
            return np.empty((0, len(self.labels)), dtype=np.float64)
        rows = [self._tokens(t) for t in raw_inputs]
        flat = [w for row in rows for w in row]
        doc = np.repeat(np.arange(len(rows)), [len(r) for r in rows])
        out = np.empty((len(rows), len(self.labels)), dtype=np.float64)
        for k, label in enumerate(self.labels):
            if label not in model:
                raise ValueError(f"generative text model is missing class {label!r}")
            dist = model[label]
            tok_logs = np.asarray(dist.seq_log_density(dist.dist_to_encoder().seq_encode(flat)), dtype=np.float64)
            if tok_logs.shape != (len(flat),) or not np.all(np.isfinite(tok_logs)):
                raise ValueError(f"generative text class {label!r} returned invalid token scores")
            out[:, k] = np.bincount(doc, weights=tok_logs, minlength=len(rows)) + self.log_prior[k]
        return out

    def proba_batch(self, model: Any, raw_inputs: list[Any]) -> np.ndarray:
        """The exact class posterior (softmax of log-joints; the shared evidence cancels)."""
        z = self.logits_batch(model, raw_inputs)
        if not len(z):
            return np.empty_like(z)
        z = np.where(np.isneginf(z).all(axis=1, keepdims=True), 0.0, z)
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        probabilities = e / e.sum(axis=1, keepdims=True)
        if not np.all(np.isfinite(probabilities)):
            raise ValueError("generative text posterior normalization failed")
        return probabilities

    def log_evidence(self, model: Any, raw_inputs: list[Any]) -> np.ndarray:
        """Per-token ``log p(x)`` (length-normalized) -- the built-in typicality/OOD score.

        Raw document evidence scales with length (a short gibberish string would outrank a long
        in-domain one), so typicality is reported per token: mean log-probability under the full
        generative model."""
        z = self.logits_batch(model, raw_inputs)
        if not len(z):
            return np.empty(0, dtype=np.float64)
        mx = z.max(axis=1, keepdims=True)
        doc = (mx + np.log(np.exp(z - mx).sum(axis=1, keepdims=True)))[:, 0]
        lens = np.asarray([len(self._tokens(t)) for t in raw_inputs], dtype=np.float64)
        return doc / np.maximum(lens, 1.0)

    def predict_batch(self, model: Any, raw_inputs: list[Any]) -> list[str]:
        """Return the highest-scoring generative class for each input."""
        idx = self.logits_batch(model, raw_inputs).argmax(axis=1)
        return [self.labels[i] for i in idx]

    def predict(self, model: Any, raw_input: Any) -> str:
        """Return the highest-scoring generative class for one input."""
        return self.predict_batch(model, [raw_input])[0]

    def to_spec(self) -> dict[str, Any]:
        """Serialize the generative text adapter."""
        return {"kind": self.kind, "labels": self.labels, "vocab": self._vocab_list, "log_prior": self.log_prior}

    @classmethod
    def from_spec(cls, spec: dict[str, Any]) -> GenerativeTextIO:
        """Reconstruct the generative text adapter from a spec."""
        return cls(spec["labels"], spec["vocab"], spec["log_prior"])


register_adapter("generative_text", GenerativeTextIO.from_spec)


def distill_text_generative_from_labels(
    texts: Sequence[str],
    teacher_labels: Sequence[Any],
    *,
    labels: Sequence[str] | None = None,
    pseudo_count: float = 1.0,
    min_count: int = 2,
    task: str = "",
) -> TaskModel:
    """Fit the per-class token models from already-labeled texts (the teacher-free training core)."""
    from mixle.inference import optimize
    from mixle.stats import CategoricalEstimator

    texts = [str(t) for t in texts]
    ys = [str(y) for y in teacher_labels]
    if not texts or len(texts) != len(ys):
        raise ValueError("texts and teacher_labels must be nonempty and have the same length")
    if not np.isfinite(pseudo_count) or pseudo_count <= 0.0:
        raise ValueError("pseudo_count must be finite and positive")
    if isinstance(min_count, bool) or not isinstance(min_count, int) or min_count <= 0:
        raise ValueError("min_count must be a positive integer")
    label_list = [str(label) for label in labels] if labels is not None else sorted(set(ys))
    if not label_list or any(not label for label in label_list) or len(set(label_list)) != len(label_list):
        raise ValueError("labels must be nonempty and unique")
    unknown = sorted(set(ys) - set(label_list))
    if unknown:
        raise ValueError(f"teacher labels are outside the declared support: {unknown}")

    counts = Counter(w.lower() for t in texts for w, _s, _e in tokenize(t))
    vocab = sorted([w for w, c in counts.items() if c >= min_count and w != _UNK]) + [_UNK]
    vset = set(vocab)

    def toks(t: str) -> list[str]:
        raw = [w.lower() for w, _s, _e in tokenize(t)]
        return [w if w in vset else _UNK for w in raw] or [_UNK]

    by_class: dict[str, list[str]] = {lab: [] for lab in label_list}
    n_docs: dict[str, int] = {lab: 0 for lab in label_list}
    for t, y in zip(texts, ys):
        by_class[y].extend(toks(t))
        n_docs[y] += 1

    n = len(texts)
    n_labels = len(label_list)
    models: dict[str, Any] = {}
    log_prior: list[float] = []
    smooth = {w: 1.0 / len(vocab) for w in vocab}
    for lab in label_list:
        # fractional Laplace over the SHARED vocabulary: pseudo_count total mass spreads uniformly over
        # the vocab (suff_stat), so a token this class never saw dims its likelihood (alpha/V) instead of
        # vetoing to -inf — and the smoothing mass stays small relative to the class's real counts
        est = CategoricalEstimator(pseudo_count=float(pseudo_count), suff_stat=smooth)
        models[lab] = optimize(by_class[lab] or [_UNK], est, max_its=2, out=None)
        prior = (n_docs[lab] + float(pseudo_count)) / (n + float(pseudo_count) * n_labels)
        log_prior.append(float(np.log(prior)))

    adapter = GenerativeTextIO(label_list, vocab, log_prior)
    return TaskModel(
        models,
        adapter,
        payload="json",
        task=task or "generative text classifier",
        meta={"distilled": True, "student": "generative_text", "n_examples": n, "vocab_size": len(vocab)},
    )


def distill_text_generative(
    teacher: Callable[..., Any],
    texts: Sequence[str],
    *,
    labels: Sequence[str] | None = None,
    pseudo_count: float = 0.5,
    min_count: int = 2,
    task: str = "",
) -> TaskModel:
    """Distill a teacher into the generative text student (the teacher labels; see module docstring)."""
    items = [str(t) for t in texts]
    if not items:
        raise ValueError("texts must contain at least one example")
    try:
        got = teacher(items)
    except TypeError:  # a per-item-only callable can reject the batch shape
        ys = [teacher(t) for t in items]
    else:
        if isinstance(got, (list, tuple)):
            if len(got) != len(items):
                raise ValueError("batched teacher must return exactly one label per text")
            ys = list(got)
        else:
            ys = [teacher(t) for t in items]
    return distill_text_generative_from_labels(
        items, ys, labels=labels, pseudo_count=pseudo_count, min_count=min_count, task=task
    )
