"""Behavioral capability profiles for distilled students.

Two students can share the same clean-holdout accuracy and still differ wildly on what they capture: one
degrades gracefully under typos, the other collapses; one honors the teacher's case-insensitivity, the other
doesn't. A capability suite names the input distribution's corruptions (severity levels), invariances
(meaning-preserving rewrites the teacher is expected to honor), and edge-case probes; :func:`capture_profile`
runs both the student and the teacher through all three and reports a JSON-serializable profile. The
profile intentionally leaves pass/fail policy to the caller.
"""

from __future__ import annotations

import random
import string
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class CapabilitySuite:
    """The behavioral spec an example distillation is checked against.

    ``corruptions`` maps a named severity level (e.g. ``"typo_10"``) to a text -> text corruption; insertion
    order is the intended severity order (mild first) so callers can read the profile's ordering directly.
    ``invariances`` maps a name to a meaning-preserving rewrite (case jitter, whitespace, a synonym swap) --
    a well-behaved model's prediction should not change under it. ``probes`` are fixed edge-case inputs whose
    raw predictions are recorded without assuming ground truth.
    """

    corruptions: dict[str, Callable[[str], str]] = field(default_factory=dict)
    invariances: dict[str, Callable[[str], str]] = field(default_factory=dict)
    probes: list[str] = field(default_factory=list)


def keyboard_typo_corruption(rate: float, *, seed: int = 0) -> Callable[[str], str]:
    """A corruption: replace each letter with a random lowercase letter independently with probability ``rate``.

    Deterministic given ``seed`` -- the same corruption function always maps the same text to the same output.
    """
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"rate must be in [0, 1], got {rate}")

    def corrupt(text: str) -> str:
        rng = random.Random(f"{seed}:{text}")
        return "".join(rng.choice(string.ascii_lowercase) if c.isalpha() and rng.random() < rate else c for c in text)

    return corrupt


def case_jitter_invariance(text: str) -> str:
    """A meaning-preserving rewrite: swap the case of every letter."""
    return text.swapcase()


def whitespace_invariance(text: str) -> str:
    """A meaning-preserving rewrite: collapse all whitespace runs to single spaces."""
    return " ".join(text.split())


def _predict(model: Any, texts: list[str]) -> list[Any]:
    """Batch-predict labels from ``model``, which may be a ``CalibratedTaskModel``, a ``TaskModel``-like object
    exposing ``batch``, or a bare ``teacher(texts) -> labels`` / ``teacher(text) -> label`` callable.

    Checks ``batch`` before ``task`` so bare ``TaskModel`` instances and
    ``CalibratedTaskModel`` wrappers both use the appropriate batch interface.
    """
    if hasattr(model, "batch"):
        return list(model.batch(texts))
    if hasattr(model, "task"):
        return list(model.task.batch(texts))
    out = model(texts)
    if isinstance(out, (list, tuple)) and len(out) == len(texts):
        return list(out)
    return [model(t) for t in texts]


def _decide(model: Any, texts: list[str]) -> list[Any] | None:
    """Batch decisions (label or ``ESCALATE``) if ``model`` exposes a decision API, else ``None``."""
    if hasattr(model, "batch_decide"):
        return list(model.batch_decide(texts))
    if hasattr(model, "decide"):
        return [model.decide(t) for t in texts]
    return None


def _agreement(a: Sequence[Any], b: Sequence[Any]) -> float:
    if not a:
        return 0.0
    return float(np.mean([str(x) == str(y) for x, y in zip(a, b)]))


def _violation_rate(before: Sequence[Any], after: Sequence[Any]) -> float:
    if not before:
        return 0.0
    return float(np.mean([str(x) != str(y) for x, y in zip(before, after)]))


def _escalation_rate(decisions: Sequence[Any]) -> float:
    if not decisions:
        return 0.0
    return float(np.mean([d is None for d in decisions]))


def capture_profile(student: Any, teacher: Any, texts: Sequence[str], suite: CapabilitySuite) -> dict[str, Any]:
    """Run ``student`` and ``teacher`` through ``suite`` and return a profile.

    Returns a plain, ``json.dumps``-safe dict:

    * ``"clean_agreement"`` -- student/teacher label agreement on the uncorrupted ``texts``;
    * ``"corruptions"`` -- per corruption name, student/teacher agreement on the corrupted texts (in the
      suite's insertion order, mild-to-severe by convention);
    * ``"invariances"`` -- per invariance name, ``{"student_violation_rate", "teacher_violation_rate"}``: how
      often each side's prediction changes under a rewrite that should not change it. A student must not be
      penalized for an invariance the teacher itself violates -- both rates are reported, never one diff;
    * ``"probes"`` -- ``{"student": [...], "teacher": [...]}`` raw predictions on the fixed probe inputs, or
      omitted if the suite has no probes;
    * ``"abstention"`` -- present only if ``student`` or ``teacher`` exposes a decision API (``decide`` /
      ``batch_decide``): each side's escalation rate on ``texts`` (``None`` for a side with no decision API).

    There is deliberately no single aggregate score field.
    """
    texts = [str(t) for t in texts]
    profile: dict[str, Any] = {"clean_agreement": _agreement(_predict(student, texts), _predict(teacher, texts))}

    corruptions: dict[str, float] = {}
    for name, corrupt in suite.corruptions.items():
        corrupted = [corrupt(t) for t in texts]
        corruptions[name] = _agreement(_predict(student, corrupted), _predict(teacher, corrupted))
    profile["corruptions"] = corruptions

    invariances: dict[str, dict[str, float]] = {}
    student_clean = _predict(student, texts)
    teacher_clean = _predict(teacher, texts)
    for name, rewrite in suite.invariances.items():
        rewritten = [rewrite(t) for t in texts]
        invariances[name] = {
            "student_violation_rate": _violation_rate(student_clean, _predict(student, rewritten)),
            "teacher_violation_rate": _violation_rate(teacher_clean, _predict(teacher, rewritten)),
        }
    profile["invariances"] = invariances

    if suite.probes:
        profile["probes"] = {
            "student": _predict(student, list(suite.probes)),
            "teacher": _predict(teacher, list(suite.probes)),
        }

    student_decisions = _decide(student, texts)
    teacher_decisions = _decide(teacher, texts)
    if student_decisions is not None or teacher_decisions is not None:
        profile["abstention"] = {
            "student_escalation_rate": _escalation_rate(student_decisions) if student_decisions is not None else None,
            "teacher_escalation_rate": _escalation_rate(teacher_decisions) if teacher_decisions is not None else None,
        }

    return profile
