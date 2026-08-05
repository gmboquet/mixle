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
:meth:`CalibratedTaskModel.calibrate_selective` certifies thresholds from a PRE-SPECIFIED grid
(1001 evenly spaced values, fixed in code, never derived from the sample) with an exact
Clopper-Pearson binomial test per grid point at Bonferroni level ``delta / 1001``, and serves at
the loosest certified threshold. That is Learn-Then-Test risk control with explicit
multiple-testing correction: with probability at least ``1 - delta`` over an i.i.d. calibration
draw, the error rate among the answers the model actually gives is at most ``alpha``. It exists
because the marginal guarantee above is NOT that statement -- the adversarial statistical review
measured 9% marginal miscoverage alongside 47.4% answered-slice error on a population built to
split them.

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


def _validated_threshold(value: Any, *, name: str, allow_none: bool = True) -> float | None:
    """Shared domain for both serving thresholds (``qhat``, ``tau``): [0, 1], ``+inf``, or None."""
    if value is None and allow_none:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite number in [0, 1], positive infinity, or None")
    result = float(value)
    if np.isnan(result) or np.isneginf(result) or result < 0.0 or (np.isfinite(result) and result > 1.0):
        raise ValueError(f"{name} must be a finite number in [0, 1], positive infinity, or None")
    return result


def _validated_qhat(qhat: Any, *, allow_none: bool = True) -> float | None:
    return _validated_threshold(qhat, name="qhat", allow_none=allow_none)


def _threshold_to_json(value: float | None, *, name: str) -> Any:
    """Serialize a threshold to a strict-JSON-safe value: ``+inf`` -> ``"inf"``, else the plain float/None."""
    validated = _validated_threshold(value, name=name)
    if validated is None:
        return None
    if np.isposinf(validated):
        return "inf"
    return validated


def _threshold_from_json(value: Any, *, name: str) -> float | None:
    """Inverse of :func:`_threshold_to_json`, rejecting corrupt/non-canonical thresholds."""
    if value is None:
        return None
    if isinstance(value, str):
        if value != "inf":
            raise ValueError(f"serialized {name} string must be the canonical 'inf' sentinel")
        return float("inf")
    return _validated_threshold(value, name=name, allow_none=False)


def _qhat_to_json(qhat: float | None) -> Any:
    return _threshold_to_json(qhat, name="qhat")


def _qhat_from_json(value: Any) -> float | None:
    return _threshold_from_json(value, name="qhat")


# The candidate thresholds for selective_risk_threshold: 1001 evenly spaced values, fixed here in
# code before any data exists. Pre-specification is load-bearing -- testing data-derived
# thresholds voids the union-bound argument below (the re-review's STAT-R2: the cited
# Learn-Then-Test proofs assume the hypothesis family does not depend on the sample that tests
# it). The price is that tau is quantized to steps of 0.001.
_SELECTIVE_GRID = np.linspace(0.0, 1.0, 1001)


def _validated_correct(correct: Any) -> np.ndarray:
    """Correctness evidence, strictly Boolean: bools or 0/1 integers, never coerced strings/floats."""
    values = np.asarray(correct)
    if values.dtype == np.bool_:
        return values
    if np.issubdtype(values.dtype, np.integer) and bool(np.isin(values, (0, 1)).all()):
        return values.astype(bool)
    # np.asarray(["false"], dtype=bool) is [True]: silent coercion turns wrong answers into
    # correctness evidence, so anything but bools and 0/1 integers is rejected outright.
    raise ValueError("correct must contain only booleans or 0/1 integers")


def selective_risk_threshold(
    top_probabilities: Any,
    correct: Any,
    *,
    alpha: float,
    delta: float = 0.05,
    min_answered: int = 10,
) -> float | None:
    """Choose a confidence threshold whose ANSWERED-SLICE error is controlled, not just marginal.

    Learn-Then-Test risk control with explicit multiple-testing correction (Angelopoulos et al.,
    arXiv:2110.01052): every candidate threshold in the pre-specified grid ``_SELECTIVE_GRID`` is
    tested against the calibration sample with an exact Clopper-Pearson binomial upper bound on
    its answered-slice error rate, each test at Bonferroni level ``delta / len(grid)``. By the
    union bound, with probability at least ``1 - delta`` over the calibration draw, EVERY
    threshold that passes its test simultaneously has true answered-slice error at most ``alpha``
    -- which is what licenses returning the loosest certified threshold after seeing all the
    results. Returns ``None`` when no grid threshold with at least ``min_answered`` answered rows
    certifies: "this model cannot guarantee that risk; escalate everything".

    Guarantee, stated exactly: if calibration rows are drawn i.i.d. from the deployment
    distribution, then with probability at least ``1 - delta`` over the calibration draw, the
    deployment error rate among rows answered at the returned threshold (answer iff top-class
    probability ``>= tau``) is at most ``alpha``. The conditional-Binomial step needs i.i.d.
    sampling, not just exchangeability. The correction's price is conservatism: certifying a
    zero-error slice needs about ``log(delta/1001)/log(1-alpha)`` answered rows (~94 at
    ``alpha=0.10, delta=0.05``, versus ~29 uncorrected), and ``tau`` is quantized to 0.001.
    """
    from scipy.stats import beta as _beta

    if (
        isinstance(alpha, (bool, np.bool_))
        or not isinstance(alpha, (int, float, np.integer, np.floating))
        or not np.isfinite(alpha)
        or not 0.0 < float(alpha) < 1.0
    ):
        raise ValueError("alpha must be in the open interval (0, 1): risk control is unachievable at 0, vacuous at 1")
    if (
        isinstance(delta, (bool, np.bool_))
        or not isinstance(delta, (int, float, np.integer, np.floating))
        or not np.isfinite(delta)
        or not 0.0 < float(delta) < 1.0
    ):
        raise ValueError("delta must be a real number in (0, 1)")
    if isinstance(min_answered, (bool, np.bool_)) or not isinstance(min_answered, (int, np.integer)):
        raise ValueError("min_answered must be a positive integer")
    if min_answered < 1:
        raise ValueError("min_answered must be a positive integer")
    alpha = float(alpha)
    delta = float(delta)
    probs = np.asarray(top_probabilities, dtype=float)
    hits = _validated_correct(correct)
    if probs.ndim != 1 or probs.shape != hits.shape or probs.size == 0:
        raise ValueError("top_probabilities and correct must be equal-length non-empty 1-D arrays")
    if np.any(~np.isfinite(probs)) or np.any((probs < 0.0) | (probs > 1.0)):
        raise ValueError("top_probabilities must be finite values in [0, 1]")

    # One ascending sort makes every grid threshold's slice a pair of suffix lookups: the rows
    # with probability >= t start at searchsorted(t), and the error count is a suffix sum.
    order = np.argsort(probs, kind="stable")
    sorted_probs = probs[order]
    misses = (~hits[order]).astype(np.int64)
    suffix_errors = np.concatenate([np.cumsum(misses[::-1])[::-1], np.zeros(1, dtype=np.int64)])
    first = np.searchsorted(sorted_probs, _SELECTIVE_GRID, side="left")
    answered = probs.size - first
    errors = suffix_errors[first]

    level = delta / float(_SELECTIVE_GRID.size)
    with np.errstate(all="ignore"):
        upper = _beta.ppf(1.0 - level, errors + 1, answered - errors)
    # An all-error or empty slice has upper bound 1 (beta.ppf returns nan when its second shape
    # parameter is 0); either way that threshold cannot certify.
    upper = np.where(errors < answered, upper, 1.0)
    certified = (upper <= alpha) & (answered >= int(min_answered))
    if not bool(certified.any()):
        return None
    return float(_SELECTIVE_GRID[certified][0])


class CalibratedTaskModel:
    """A :class:`TaskModel` plus a conformal threshold: predicts label *sets* and decides answer-vs-escalate."""

    def __init__(
        self,
        task: TaskModel,
        *,
        alpha: float = 0.1,
        qhat: float | None = None,
        tau: float | None = None,
        density_gate: Any = None,
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
        self.tau: float | None = _validated_threshold(tau, name="tau")

    @property
    def labels(self) -> list[str]:
        """Return labels in the probability-vector order used by the adapter."""
        return self.task.adapter.labels

    def _proba(self, raw_inputs: list[Any]) -> np.ndarray:
        return self.task.adapter.proba_batch(self.task.model, list(raw_inputs))

    def _calibration_probabilities(
        self, texts: Sequence[Any], teacher_labels: Sequence[Any]
    ) -> tuple[list[Any], list[str], np.ndarray]:
        """Fail-closed input validation shared by BOTH calibration routes.

        The re-review's STAT-R4: ``calibrate_selective`` accepted a one-column matrix for two
        labels and rows summing to 1.8 because only ``calibrate`` validated. The contract is one
        method so the two routes cannot drift apart again.
        """
        if isinstance(texts, (str, bytes)) or isinstance(teacher_labels, (str, bytes)):
            raise TypeError("texts and teacher_labels must be sequences of rows, not scalar strings")
        rows = list(texts)
        observed = [str(label) for label in teacher_labels]
        if not rows:
            raise ValueError("calibration data must be non-empty")
        if len(rows) != len(observed):
            raise ValueError("texts and teacher_labels must have identical non-zero lengths")
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
        return rows, observed, prob

    def calibrate(self, texts: Sequence[Any], teacher_labels: Sequence[Any]) -> CalibratedTaskModel:
        """Set the conformal threshold from held-out ``(texts, teacher_labels)`` for ``1 - alpha`` set coverage."""
        _, observed, prob = self._calibration_probabilities(texts, teacher_labels)
        index = {label: i for i, label in enumerate(self.labels)}
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

        Runs after :meth:`calibrate` (both gates then apply) or instead of it (serving answers
        the argmax label whenever the top-class probability clears ``tau``), and applies the same
        fail-closed input validation as :meth:`calibrate`. When no threshold can control the
        risk, ``tau`` becomes ``inf`` and everything escalates -- an honest refusal, never a
        silent downgrade to the marginal-only contract.
        """
        _, observed, prob = self._calibration_probabilities(texts, teacher_labels)
        top = prob.max(axis=1)
        predicted = [self.labels[int(i)] for i in prob.argmax(axis=1)]
        hit = np.asarray([p == o for p, o in zip(predicted, observed)], dtype=bool)
        tau = selective_risk_threshold(top, hit, alpha=self.alpha, delta=delta, min_answered=min_answered)
        self.tau = float("inf") if tau is None else float(tau)
        return self

    def _argmax_sets(self, rows: list[Any]) -> list[list[str]]:
        """Selective-only serving candidates: the argmax label per row; impossible evidence -> empty set."""
        try:
            prob = np.asarray(self._proba(rows), dtype=float)
            return [[self.labels[int(i)]] for i in prob.argmax(axis=1)]
        except ImpossibleEvidenceError:
            result: list[list[str]] = []
            for row in rows:
                try:
                    prob = np.asarray(self._proba([row]), dtype=float)
                except ImpossibleEvidenceError:
                    result.append([])
                    continue
                result.append([self.labels[int(prob.argmax(axis=1)[0])]])
            return result

    def _serving_sets(self, rows: list[Any]) -> list[list[str]]:
        """Candidate sets for deciding: conformal sets when ``qhat`` is set, else argmax under the tau gate.

        The re-review's STAT-R3: the docstrings offer ``calibrate_selective`` INSTEAD of
        ``calibrate``, and serving used to call ``predict_sets`` first, which raises on
        ``qhat=None`` -- the documented mode could never run.
        """
        if self.qhat is not None:
            return self.predict_sets(rows)
        if self.tau is None:
            raise RuntimeError(
                "call calibrate(...) and/or calibrate_selective(...) (or load a calibrated artifact) before deciding"
            )
        if not rows:
            return []
        return self._argmax_sets(rows)

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
        rows = list(texts)
        sets = self._serving_sets(rows)
        esc = self._escalate_flags(rows, sets)
        return [ESCALATE if e else s[0] for s, e in zip(sets, esc)]

    def escalation_rate(self, texts: Sequence[Any]) -> float:
        """Empirical ``p_escalate`` -- the fraction of inputs escalated (ambiguous set or, if gated, OOD)."""
        rows = list(texts)
        sets = self._serving_sets(rows)
        return float(np.mean(self._escalate_flags(rows, sets))) if len(sets) else 0.0

    def save(self, path: str) -> str:
        """Persist the model, the full calibration (alpha, qhat, tau), and any density gate in the artifact.

        ``qhat`` and ``tau`` can each legitimately be ``+inf`` (qhat: too little calibration data to admit
        any confident singleton; tau: ``calibrate_selective`` could not control the risk). Those are real,
        callable thresholds, so both persist as the JSON-safe sentinel ``"inf"`` and reload back to
        ``float('inf')``. The persistence contract is decision identity INCLUDING abstentions: the
        re-review's STAT-R1 caught ``tau`` being dropped here, which silently downgraded a selectively
        calibrated artifact to the marginal-only rule and answered a row it had abstained on.
        """
        cal: dict[str, Any] = {
            "alpha": self.alpha,
            "qhat": _threshold_to_json(self.qhat, name="qhat"),
            "tau": _threshold_to_json(self.tau, name="tau"),
        }
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
        return cls(
            task,
            alpha=cal.get("alpha", 0.1),
            qhat=_threshold_from_json(cal.get("qhat"), name="qhat"),
            tau=_threshold_from_json(cal.get("tau"), name="tau"),
            density_gate=gate,
        )
