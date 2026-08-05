"""``CalibratedTaskModel`` wraps a task model in conformal answer sets.

A distilled student classifies by argmax over a softmax, but the softmax value
alone is not a coverage guarantee. Conformal prediction adds the serving
contract: on a held-out calibration set it learns a score threshold
(:func:`mixle.inference.conformal.conformal_label_threshold`) such that the
prediction *set* covers the true label with probability ``>= 1 - alpha`` under
the usual exchangeability assumption.

The decision rule the cascade and the cost model consume:

  * **singleton set** -> answer locally. NOTE: the ``1 - alpha`` guarantee is MARGINAL --
    over the whole population, the set contains the truth with probability ``>= 1 - alpha``.
    It is NOT a conditional guarantee on the answered slice: error *among singleton answers*
    can be far higher (an adversarial review measured 9% marginal miscoverage alongside 47%
    answered-slice error at ``alpha=0.10``). Measure answered-slice agreement per deployment
    (``report()`` does) rather than reading it off ``alpha``;
  * **empty or multi-label set** -> escalate to the expensive teacher/frontier (genuinely ambiguous).

When the answered slice itself needs a risk guarantee, calibrate it directly:
:meth:`CalibratedTaskModel.calibrate_selective` runs fixed-sequence testing with exact
Clopper-Pearson binomial bounds over candidate confidence thresholds and keeps the most permissive
threshold whose ANSWERED-SLICE error bound is ``<= alpha`` at confidence ``1 - delta``. That is a
distribution-free selective-risk guarantee (Learn-Then-Test in its simplest fixed-sequence form):
with probability at least ``1 - delta`` over the calibration draw, the error rate among the
answers the model actually gives is at most ``alpha``. It exists because the marginal guarantee
above is NOT that statement -- the adversarial statistical review measured 9% marginal
miscoverage alongside 47.4% answered-slice error on a population built to split them.

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
from mixle.task.model import ImpossibleEvidenceError, TaskModel

ESCALATE = None  # the sentinel a decision returns when the conformal set is not a confident singleton


def _validated_alpha(alpha: Any) -> float:
    if (
        isinstance(alpha, (bool, np.bool_))
        or not isinstance(alpha, (int, float, np.integer, np.floating))
        or not np.isfinite(alpha)
        or not 0.0 <= float(alpha) <= 1.0
    ):
        raise ValueError("alpha must be a finite number in [0, 1]")
    return float(alpha)


def _validated_qhat(qhat: Any, *, allow_none: bool = True) -> float | None:
    if qhat is None and allow_none:
        return None
    if isinstance(qhat, (bool, np.bool_)) or not isinstance(qhat, (int, float, np.integer, np.floating)):
        raise ValueError("qhat must be a finite number in [0, 1], positive infinity, or None")
    result = float(qhat)
    if np.isnan(result) or np.isneginf(result) or result < 0.0 or (np.isfinite(result) and result > 1.0):
        raise ValueError("qhat must be a finite number in [0, 1], positive infinity, or None")
    return result


def _qhat_to_json(qhat: float | None) -> Any:
    """Serialize a threshold to a strict-JSON-safe value: ``+inf`` -> ``"inf"``, else the plain float/None."""
    validated = _validated_qhat(qhat)
    if validated is None:
        return None
    if np.isposinf(validated):
        return "inf"
    return validated


def _qhat_from_json(value: Any) -> float | None:
    """Inverse of :func:`_qhat_to_json`, rejecting corrupt/non-canonical thresholds."""
    if value is None:
        return None
    if isinstance(value, str):
        if value != "inf":
            raise ValueError("serialized qhat string must be the canonical 'inf' sentinel")
        return float("inf")
    return _validated_qhat(value, allow_none=False)


def selective_risk_threshold(
    top_probabilities: Any,
    correct: Any,
    *,
    alpha: float,
    delta: float = 0.05,
    min_answered: int = 10,
) -> float | None:
    """Choose a confidence threshold whose ANSWERED-SLICE error is controlled, not just marginal.

    Fixed-sequence testing (the simplest valid Learn-Then-Test schedule): candidate thresholds are
    walked from strictest to loosest, and for each the answered slice is the calibration rows whose
    top-class probability clears it. A threshold is accepted while the exact Clopper-Pearson upper
    bound (level ``delta``) on the slice's error rate stays ``<= alpha``; the walk stops at the
    first violation, which is what makes the sequence a valid multiple-testing schedule without
    any correction factor. Returns the loosest accepted threshold, or ``None`` when no threshold
    with at least ``min_answered`` answered rows passes -- ``None`` means "this model cannot
    guarantee that risk at that confidence; escalate everything".

    Guarantee, stated exactly: if calibration rows are exchangeable with deployment rows, then with
    probability at least ``1 - delta`` over the calibration draw, the deployment error rate among
    answered rows is at most ``alpha``. This is the statement the marginal conformal guarantee does
    not make.
    """
    from scipy.stats import beta as _beta

    alpha = _validated_alpha(alpha)
    if not isinstance(delta, (int, float)) or isinstance(delta, bool) or not 0.0 < float(delta) < 1.0:
        raise ValueError("delta must be a real number in (0, 1)")
    if isinstance(min_answered, bool) or not isinstance(min_answered, int) or min_answered < 1:
        raise ValueError("min_answered must be a positive integer")
    probs = np.asarray(top_probabilities, dtype=float)
    hits = np.asarray(correct, dtype=bool)
    if probs.ndim != 1 or probs.shape != hits.shape or probs.size == 0:
        raise ValueError("top_probabilities and correct must be equal-length non-empty 1-D arrays")
    if np.any(~np.isfinite(probs)) or np.any((probs < 0.0) | (probs > 1.0)):
        raise ValueError("top_probabilities must be finite values in [0, 1]")

    # The first slice size at which the bound CAN pass is deterministic in (alpha, delta): with
    # zero errors the Clopper-Pearson upper bound is 1 - delta**(1/n), so testing below
    # n >= log(delta)/log(1-alpha) is guaranteed to fail and would end a fixed-sequence walk
    # before it ever had a chance. Starting the schedule there is pre-specified independently of
    # the data, so the sequence's validity is untouched.
    import math as _math

    start_n = max(int(min_answered), int(_math.ceil(_math.log(float(delta)) / _math.log(1.0 - alpha))))
    order = np.argsort(-probs, kind="stable")
    errors = 0
    accepted: float | None = None
    n_seen = 0
    index = 0
    thresholds = probs[order]
    misses = ~hits[order]
    while index < thresholds.size:
        tau = float(thresholds[index])
        # admit every row tied at this threshold before testing it, so the slice is exactly
        # {top_probability >= tau}
        while index < thresholds.size and float(thresholds[index]) >= tau:
            errors += int(misses[index])
            n_seen += 1
            index += 1
        if n_seen < start_n:
            continue
        upper = float(_beta.ppf(1.0 - float(delta), errors + 1, n_seen - errors)) if errors < n_seen else 1.0
        if upper <= alpha:
            accepted = tau
        else:
            break
    return accepted


class CalibratedTaskModel:
    """A :class:`TaskModel` plus a conformal threshold: predicts label *sets* and decides answer-vs-escalate."""

    def __init__(
        self, task: TaskModel, *, alpha: float = 0.1, qhat: float | None = None, density_gate: Any = None
    ) -> None:
        if not hasattr(task.adapter, "proba_batch"):
            raise TypeError("CalibratedTaskModel needs an adapter exposing proba_batch (e.g. TextClassifierIO)")
        self.task = task
        self.alpha = _validated_alpha(alpha)
        self.qhat = _validated_qhat(qhat)
        self.density_gate = density_gate  # optional p(x) OOD gate: escalate atypical inputs softmax can't see
        # Selective-risk gate (calibrate_selective): when set, answering additionally requires the
        # top-class probability to clear this threshold, giving the answered slice its own
        # controlled error rate. None means the marginal-only contract documented above.
        self.tau: float | None = None

    @property
    def labels(self) -> list[str]:
        """Return labels in the probability-vector order used by the adapter."""
        return self.task.adapter.labels

    def _proba(self, raw_inputs: list[Any]) -> np.ndarray:
        return self.task.adapter.proba_batch(self.task.model, list(raw_inputs))

    def calibrate(self, texts: Sequence[Any], teacher_labels: Sequence[Any]) -> CalibratedTaskModel:
        """Set the conformal threshold from held-out ``(texts, teacher_labels)`` for ``1 - alpha`` set coverage."""
        if isinstance(texts, (str, bytes)) or isinstance(teacher_labels, (str, bytes)):
            raise TypeError("texts and teacher_labels must be sequences of rows, not scalar strings")
        rows = list(texts)
        observed = [str(label) for label in teacher_labels]
        if not rows:
            raise ValueError("calibration data must be non-empty")
        if len(rows) != len(observed):
            raise ValueError("texts and teacher_labels must have identical non-zero lengths")
        index = {label: i for i, label in enumerate(self.labels)}
        prob = np.asarray(self._proba(rows), dtype=float)
        if prob.shape != (len(rows), len(self.labels)):
            raise ValueError(
                f"adapter probabilities must have shape ({len(rows)}, {len(self.labels)}), got {prob.shape}"
            )
        if (
            np.any(~np.isfinite(prob))
            or np.any((prob < 0.0) | (prob > 1.0))
            or not np.allclose(prob.sum(axis=1), 1.0, rtol=1e-7, atol=1e-9)
        ):
            raise ValueError("adapter probabilities must be finite row-stochastic values in [0, 1]")
        # A teacher, or a realistic dataset split, can return a label the student's class set does not
        # contain. That label has no column in the probability vector, so its true-class score is 0 --
        # the student is guaranteed to miss it. Scoring the miss is what keeps the threshold sound: it
        # pushes qhat down, so the predictor becomes MORE conservative and the guaranteed miss is paid
        # for by wider sets elsewhere. Dropping such rows would overstate coverage, and rejecting them
        # outright makes calibration impossible on exactly the data conformal prediction is for.
        cal_true = np.array([prob[i, index[label]] if label in index else 0.0 for i, label in enumerate(observed)])
        qhat = conformal_label_threshold(cal_true, alpha=self.alpha)
        self.qhat = _validated_qhat(qhat, allow_none=False)
        return self

    def predict_sets(self, texts: Sequence[Any]) -> list[list[str]]:
        """Conformal label set per input (the classes whose score clears the calibrated threshold)."""
        if self.qhat is None:
            raise RuntimeError("call calibrate(...) (or load a calibrated artifact) before predicting sets")
        rows = list(texts)
        if not rows:
            return []
        try:
            probabilities = self._proba(rows)
            sets, _ = conformal_label_sets(np.empty(0), probabilities, alpha=self.alpha, qhat=self.qhat)
            return [[self.labels[i] for i in np.flatnonzero(row)] for row in sets]
        except ImpossibleEvidenceError:
            # Preserve impossible structured evidence as an empty prediction set so the calibrated
            # serving contract escalates it. Re-evaluate individually to keep valid rows in a mixed batch;
            # implementation failures other than the explicit impossible-evidence signal still propagate.
            result: list[list[str]] = []
            for row in rows:
                try:
                    probabilities = self._proba([row])
                except ImpossibleEvidenceError:
                    result.append([])
                    continue
                sets, _ = conformal_label_sets(
                    np.empty(0),
                    probabilities,
                    alpha=self.alpha,
                    qhat=self.qhat,
                )
                result.append([self.labels[i] for i in np.flatnonzero(sets[0])])
            return result

    def predict_set(self, text: Any) -> list[str]:
        """Return the conformal label set for one input."""
        return self.predict_sets([text])[0]

    def calibrate_selective(
        self,
        texts: Sequence[Any],
        teacher_labels: Sequence[Any],
        *,
        delta: float = 0.05,
        min_answered: int = 10,
    ) -> CalibratedTaskModel:
        """Calibrate the ANSWERED-SLICE risk gate (see :func:`selective_risk_threshold`).

        Runs after (or instead of) :meth:`calibrate`; sets ``tau`` so answering additionally
        requires the top-class probability to clear it. When no threshold can control the risk,
        ``tau`` becomes ``inf`` and everything escalates -- an honest refusal, never a silent
        downgrade to the marginal-only contract.
        """
        rows = list(texts)
        observed = [str(label) for label in teacher_labels]
        if not rows or len(rows) != len(observed):
            raise ValueError("texts and teacher_labels must have identical non-zero lengths")
        prob = np.asarray(self._proba(rows), dtype=float)
        top = prob.max(axis=1)
        predicted = [self.labels[i] for i in prob.argmax(axis=1)]
        hit = np.asarray([p == o for p, o in zip(predicted, observed)])
        tau = selective_risk_threshold(top, hit, alpha=self.alpha, delta=delta, min_answered=min_answered)
        self.tau = float("inf") if tau is None else float(tau)
        return self

    def _escalate_flags(self, texts: Sequence[Any], sets: list[list[str]]) -> np.ndarray:
        """Escalate on ambiguous conformal sets, OOD inputs (density gate), or a sub-tau confidence."""
        amb = np.asarray([len(s) != 1 for s in sets])
        if self.tau is not None:
            rows = list(texts)
            top = np.asarray(self._proba(rows), dtype=float).max(axis=1) if rows else np.empty(0)
            amb = amb | (top < self.tau)
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
        if not isinstance(cal, dict):
            raise ValueError("artifact calibration metadata must be a dictionary")
        gate = None
        if cal.get("density_gate") is not None:
            from mixle.task.density import DensityGate

            gate = DensityGate.from_spec(cal["density_gate"])
        return cls(task, alpha=cal.get("alpha", 0.1), qhat=_qhat_from_json(cal.get("qhat")), density_gate=gate)
