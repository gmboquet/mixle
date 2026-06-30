"""``CalibratedTaskModel`` -- wrap a task model in conformal sets so its escalate-or-answer decision is honest.

A distilled student classifies by argmax over a softmax that is *not* a describable random process: the numbers
sum to 1, but a confident-looking 0.97 carries no guarantee. Gating a cost-aware cascade on that number is
fiction. Conformal prediction fixes it without pretending the softmax is generative: on a held-out calibration
set it learns a score threshold (:func:`mixle.inference.conformal.conformal_label_threshold`) such that the
prediction *set* covers the true label with probability ``>= 1 - alpha``.

The decision rule the cascade and the cost model consume:

  * **singleton set** -> answer locally (covered at ``1 - alpha``);
  * **empty or multi-label set** -> escalate to the expensive teacher/frontier (genuinely ambiguous).

``escalation_rate`` is the empirical ``p_escalate`` -- the number that makes "expected \\$/request" real rather
than a vibe. Conformal coverage is *marginal*, and a softmax still can't see true OOD; a generative-density gate
(:mod:`mixle.task.density`) covers that residual. Calibration persists in the artifact, so a loaded model decides
identically in a fresh process.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from mixle.inference.conformal import conformal_label_sets, conformal_label_threshold
from mixle.task.model import TaskModel

ESCALATE = None  # the sentinel a decision returns when the conformal set is not a confident singleton


class CalibratedTaskModel:
    """A :class:`TaskModel` plus a conformal threshold: predicts label *sets* and decides answer-vs-escalate."""

    def __init__(self, task: TaskModel, *, alpha: float = 0.1, qhat: float | None = None) -> None:
        if not hasattr(task.adapter, "proba_batch"):
            raise TypeError("CalibratedTaskModel needs an adapter exposing proba_batch (e.g. TextClassifierIO)")
        self.task = task
        self.alpha = float(alpha)
        self.qhat = qhat

    @property
    def labels(self) -> list[str]:
        return self.task.adapter.labels

    def _proba(self, raw_inputs: list[Any]) -> np.ndarray:
        return self.task.adapter.proba_batch(self.task.model, list(raw_inputs))

    def calibrate(self, texts: Sequence[Any], teacher_labels: Sequence[Any]) -> CalibratedTaskModel:
        """Set the conformal threshold from held-out ``(texts, teacher_labels)`` for ``1 - alpha`` set coverage."""
        index = {label: i for i, label in enumerate(self.labels)}
        prob = self._proba(list(texts))
        true_idx = np.asarray([index[str(y)] for y in teacher_labels])
        cal_true = prob[np.arange(len(true_idx)), true_idx]
        self.qhat = conformal_label_threshold(cal_true, alpha=self.alpha)
        return self

    def predict_sets(self, texts: Sequence[Any]) -> list[list[str]]:
        """Conformal label set per input (the classes whose score clears the calibrated threshold)."""
        if self.qhat is None:
            raise RuntimeError("call calibrate(...) (or load a calibrated artifact) before predicting sets")
        sets, _ = conformal_label_sets(np.empty(0), self._proba(list(texts)), alpha=self.alpha, qhat=self.qhat)
        return [[self.labels[i] for i in np.flatnonzero(row)] for row in sets]

    def predict_set(self, text: Any) -> list[str]:
        return self.predict_sets([text])[0]

    def decide(self, text: Any) -> Any:
        """Return the label if the conformal set is a confident singleton, else ``ESCALATE`` (``None``)."""
        s = self.predict_set(text)
        return s[0] if len(s) == 1 else ESCALATE

    def batch_decide(self, texts: Sequence[Any]) -> list[Any]:
        return [s[0] if len(s) == 1 else ESCALATE for s in self.predict_sets(texts)]

    def escalation_rate(self, texts: Sequence[Any]) -> float:
        """Empirical ``p_escalate`` -- the fraction of inputs whose set is not a confident singleton."""
        sets = self.predict_sets(texts)
        return float(np.mean([len(s) != 1 for s in sets])) if len(sets) else 0.0

    def save(self, path: str) -> str:
        """Persist the underlying model plus the calibration (alpha, qhat) in the artifact metadata."""
        q = self.qhat if (self.qhat is not None and np.isfinite(self.qhat)) else None
        self.task.meta = {**self.task.meta, "calibration": {"alpha": self.alpha, "qhat": q}}
        return self.task.save(path)

    @classmethod
    def load(cls, path: str, *, device: str = "cpu") -> CalibratedTaskModel:
        """Rebuild a calibrated model from an artifact; decisions match the saving process exactly."""
        task = TaskModel.load(path, device=device)
        cal = task.meta.get("calibration", {})
        return cls(task, alpha=cal.get("alpha", 0.1), qhat=cal.get("qhat"))
