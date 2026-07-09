"""``CalibratedTaskModel`` wraps a task model in conformal answer sets.

A distilled student classifies by argmax over a softmax, but the softmax value
alone is not a coverage guarantee. Conformal prediction adds the serving
contract: on a held-out calibration set it learns a score threshold
(:func:`mixle.inference.conformal.conformal_label_threshold`) such that the
prediction *set* covers the true label with probability ``>= 1 - alpha`` under
the usual exchangeability assumption.

The decision rule the cascade and the cost model consume:

  * **singleton set** -> answer locally (covered at ``1 - alpha``);
  * **empty or multi-label set** -> escalate to the expensive teacher/frontier (genuinely ambiguous).

``escalation_rate`` is the empirical ``p_escalate`` used by the cost model.
Conformal coverage is *marginal*, and a softmax still cannot see true OOD; a generative-density gate
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


def _qhat_to_json(qhat: float | None) -> Any:
    """Serialize a threshold to a strict-JSON-safe value: ``+inf`` -> ``"inf"``, else the plain float/None."""
    if qhat is None:
        return None
    if not np.isfinite(qhat):
        return "inf"
    return float(qhat)


def _qhat_from_json(value: Any) -> float | None:
    """Inverse of :func:`_qhat_to_json`: ``"inf"`` (or a non-finite float) -> ``float('inf')``; else float/None."""
    if value is None:
        return None
    if isinstance(value, str):
        return float("inf") if value == "inf" else float(value)
    q = float(value)
    return float("inf") if not np.isfinite(q) else q


class CalibratedTaskModel:
    """A :class:`TaskModel` plus a conformal threshold: predicts label *sets* and decides answer-vs-escalate."""

    def __init__(
        self, task: TaskModel, *, alpha: float = 0.1, qhat: float | None = None, density_gate: Any = None
    ) -> None:
        if not hasattr(task.adapter, "proba_batch"):
            raise TypeError("CalibratedTaskModel needs an adapter exposing proba_batch (e.g. TextClassifierIO)")
        self.task = task
        self.alpha = float(alpha)
        self.qhat = qhat
        self.density_gate = density_gate  # optional p(x) OOD gate: escalate atypical inputs softmax can't see

    @property
    def labels(self) -> list[str]:
        """Return labels in the probability-vector order used by the adapter."""
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
        """Return the conformal label set for one input."""
        return self.predict_sets([text])[0]

    def _escalate_flags(self, texts: Sequence[Any], sets: list[list[str]]) -> np.ndarray:
        """Escalate when the conformal set is ambiguous OR (if a density gate is set) the input is OOD."""
        amb = np.asarray([len(s) != 1 for s in sets])
        if self.density_gate is None:
            return amb
        return amb | self.density_gate.ood_mask(list(texts))

    def decide(self, text: Any) -> Any:
        """Return the label if the input is a confident, in-distribution singleton, else ``ESCALATE`` (``None``)."""
        return self.batch_decide([text])[0]

    def batch_decide(self, texts: Sequence[Any]) -> list[Any]:
        """Return local labels or ``ESCALATE`` for a batch of inputs."""
        sets = self.predict_sets(texts)
        esc = self._escalate_flags(texts, sets)
        return [ESCALATE if e else s[0] for s, e in zip(sets, esc)]

    def escalation_rate(self, texts: Sequence[Any]) -> float:
        """Empirical ``p_escalate`` -- the fraction of inputs escalated (ambiguous set or, if gated, OOD)."""
        sets = self.predict_sets(texts)
        return float(np.mean(self._escalate_flags(texts, sets))) if len(sets) else 0.0

    def save(self, path: str) -> str:
        """Persist the underlying model, the calibration (alpha, qhat), and any density gate in the artifact.

        ``qhat`` can legitimately be ``+inf`` (a small calibration set / tight ``alpha``: too little data to
        admit any confident singleton, so every input escalates). That is a real, callable threshold, so it is
        persisted as the JSON-safe sentinel ``"inf"`` and reloads back to ``float('inf')`` -- a loaded model
        stays callable instead of raising "call calibrate".
        """
        cal: dict[str, Any] = {"alpha": self.alpha, "qhat": _qhat_to_json(self.qhat)}
        if self.density_gate is not None:
            cal["density_gate"] = self.density_gate.to_spec()
        self.task.meta = {**self.task.meta, "calibration": cal}
        return self.task.save(path)

    @classmethod
    def load(cls, path: str, *, device: str = "cpu") -> CalibratedTaskModel:
        """Rebuild a calibrated model (with its density gate, if any) from an artifact; decisions match exactly."""
        task = TaskModel.load(path, device=device)
        cal = task.meta.get("calibration", {})
        gate = None
        if cal.get("density_gate") is not None:
            from mixle.task.density import DensityGate

            gate = DensityGate.from_spec(cal["density_gate"])
        return cls(task, alpha=cal.get("alpha", 0.1), qhat=_qhat_from_json(cal.get("qhat")), density_gate=gate)
