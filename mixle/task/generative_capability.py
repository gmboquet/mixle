"""Capture profiles for structured-output students: extraction and generative text.

:func:`~mixle.task.capability.capture_profile` scores agreement by exact match on a scalar label -- the
right notion for a classifier, but the wrong one for a field extractor: a student that gets 3 of 4 fields
right under a typo corruption is not "wrong", and exact-match agreement would report it identically to a
student that got every field wrong. This module extends the same :class:`~mixle.task.capability.CapabilitySuite`
machinery (corruptions, invariances -- reused, not reinvented) with executable verifiers suited to
structured output:

* extraction students (:func:`~mixle.task.extract.distill_extractor`, adapter ``ExtractionIO``): scored by
  micro-averaged field-level F1 against a fixed gold reference (the teacher's own extraction on the clean
  text -- corruption is a nuisance perturbation of the input, not a change to the true answer), plus a
  schema-validity check (every expected field present, every value actually grounded in -- a substring of
  -- the text it was extracted from). Both are checkable by code, never by eyeballing output.
* :func:`~mixle.task.generative_text.distill_text_generative` students predict a plain label (like any
  other classifier), so the base :func:`capture_profile` already scores them correctly -- no special
  handling needed here; this module documents that explicitly rather than silently duplicating it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from mixle.task.capability import CapabilitySuite, _decide, _escalation_rate, _predict


def validate_extraction_schema(record: dict[str, Any], source_text: str, fields: Sequence[str]) -> dict[str, Any]:
    """Executable schema check for one extracted record: complete (every expected field present) and
    grounded (every non-empty value is an actual substring of ``source_text``, not hallucinated).

    Returns a plain dict (``complete``, ``grounded``, ``missing``, ``ungrounded``) -- never a single
    pass/fail bit, so a caller can see exactly what failed.
    """
    fields = list(fields)
    missing = [f for f in fields if f not in record or not record.get(f)]
    ungrounded = [f for f, v in record.items() if f in fields and v and str(v) not in source_text]
    return {
        "complete": not missing,
        "grounded": not ungrounded,
        "missing": missing,
        "ungrounded": ungrounded,
    }


def _field_f1(pred: Sequence[dict[str, Any]], gold: Sequence[dict[str, Any]]) -> float:
    """Micro-averaged field-level F1 (same formula as :func:`mixle.task.extract.extraction_f1`, generalized
    to operate on already-computed prediction dicts instead of requiring a ``TaskModel.batch`` call, so it
    scores a bare-callable teacher the same way it scores a ``TaskModel`` student)."""
    tp = fp = fn = 0
    for p, g in zip(pred, gold):
        for field, value in g.items():
            if p.get(field) == value:
                tp += 1
            else:
                fn += 1
        for field in p:
            if field not in g:
                fp += 1
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom else 1.0


def _schema_validity_rate(records: Sequence[dict[str, Any]], texts: Sequence[str], fields: Sequence[str]) -> float:
    if not records:
        return 1.0
    checks = [validate_extraction_schema(r, t, fields) for r, t in zip(records, texts)]
    return float(np.mean([c["complete"] and c["grounded"] for c in checks]))


def extractive_capture_profile(
    student: Any, teacher: Any, texts: Sequence[str], suite: CapabilitySuite, *, fields: Sequence[str]
) -> dict[str, Any]:
    """The extraction-student capture profile: F1-against-gold and schema validity, not exact-match agreement.

    ``gold`` is the teacher's own extraction on the clean ``texts`` -- the true answer a corruption should
    not change. Reports, JSON-serializable:

    * ``"clean_f1"`` -- student F1 against gold on clean text (teacher's own clean F1 against its own gold
      is trivially 1.0 and omitted);
    * ``"corruptions"`` -- per corruption name, ``{"student_f1", "teacher_f1"}`` against the same fixed
      gold -- both sides scored against the same ground truth, so a comparison is meaningful;
    * ``"invariances"`` -- per invariance name, ``{"student_f1", "teacher_f1"}`` between each side's clean
      prediction and its prediction on the rewritten text (1.0 = perfectly invariant);
    * ``"schema_validity"`` -- ``{"student", "teacher"}`` fraction of clean-text extractions that are
      complete and grounded (:func:`validate_extraction_schema`) -- an executable check, never eyeballed;
    * ``"abstention"`` -- as in :func:`~mixle.task.capability.capture_profile`, if either side exposes a
      decision API.
    """
    texts = [str(t) for t in texts]
    fields = list(fields)
    gold = _predict(teacher, texts)

    student_clean = _predict(student, texts)
    profile: dict[str, Any] = {
        "clean_f1": _field_f1(student_clean, gold),
        "schema_validity": {
            "student": _schema_validity_rate(student_clean, texts, fields),
            "teacher": _schema_validity_rate(gold, texts, fields),
        },
    }

    corruptions: dict[str, dict[str, float]] = {}
    for name, corrupt in suite.corruptions.items():
        corrupted = [corrupt(t) for t in texts]
        corruptions[name] = {
            "student_f1": _field_f1(_predict(student, corrupted), gold),
            "teacher_f1": _field_f1(_predict(teacher, corrupted), gold),
        }
    profile["corruptions"] = corruptions

    teacher_clean = gold
    invariances: dict[str, dict[str, float]] = {}
    for name, rewrite in suite.invariances.items():
        rewritten = [rewrite(t) for t in texts]
        invariances[name] = {
            "student_f1": _field_f1(_predict(student, rewritten), student_clean),
            "teacher_f1": _field_f1(_predict(teacher, rewritten), teacher_clean),
        }
    profile["invariances"] = invariances

    student_decisions = _decide(student, texts)
    teacher_decisions = _decide(teacher, texts)
    if student_decisions is not None or teacher_decisions is not None:
        profile["abstention"] = {
            "student_escalation_rate": _escalation_rate(student_decisions) if student_decisions is not None else None,
            "teacher_escalation_rate": _escalation_rate(teacher_decisions) if teacher_decisions is not None else None,
        }

    return profile
