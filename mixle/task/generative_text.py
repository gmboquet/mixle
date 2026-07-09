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
        self.labels = list(labels)
        self.vocab = set(vocab)
        self._vocab_list = list(vocab)
        self.log_prior = [float(v) for v in log_prior]

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
            dist = model[label]
            tok_logs = np.asarray(dist.seq_log_density(dist.dist_to_encoder().seq_encode(flat)), dtype=np.float64)
            out[:, k] = np.bincount(doc, weights=tok_logs, minlength=len(rows)) + self.log_prior[k]
        return out

    def proba_batch(self, model: Any, raw_inputs: list[Any]) -> np.ndarray:
        """The exact class posterior (softmax of log-joints; the shared evidence cancels)."""
        z = self.logits_batch(model, raw_inputs)
        z = np.where(np.isneginf(z).all(axis=1, keepdims=True), 0.0, z)
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def log_evidence(self, model: Any, raw_inputs: list[Any]) -> np.ndarray:
        """Per-token ``log p(x)`` (length-normalized) -- the built-in typicality/OOD score.

        Raw document evidence scales with length (a short gibberish string would outrank a long
        in-domain one), so typicality is reported per token: mean log-probability under the full
        generative model."""
        z = self.logits_batch(model, raw_inputs)
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
    label_list = list(labels) if labels is not None else sorted(set(ys))

    counts = Counter(w.lower() for t in texts for w, _s, _e in tokenize(t))
    vocab = sorted([w for w, c in counts.items() if c >= int(min_count)]) + [_UNK]
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
    models: dict[str, Any] = {}
    log_prior: list[float] = []
    smooth = {w: 1.0 / len(vocab) for w in vocab}
    for lab in label_list:
        # fractional Laplace over the SHARED vocabulary: pseudo_count total mass spreads uniformly over
        # the vocab (suff_stat), so a token this class never saw dims its likelihood (alpha/V) instead of
        # vetoing to -inf — and the smoothing mass stays small relative to the class's real counts
        est = CategoricalEstimator(pseudo_count=float(pseudo_count), suff_stat=smooth)
        models[lab] = optimize(by_class[lab] or [_UNK], est, max_its=2, out=None)
        log_prior.append(float(np.log(max(n_docs[lab], 1) / max(n, 1))))

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
    try:
        got = teacher(items)
        ys = list(got) if isinstance(got, (list, tuple)) and len(got) == len(items) else [teacher(t) for t in items]
    except Exception:  # noqa: BLE001 - a per-item teacher raises on the list probe
        ys = [teacher(t) for t in items]
    return distill_text_generative_from_labels(
        items, ys, labels=labels, pseudo_count=pseudo_count, min_count=min_count, task=task
    )
