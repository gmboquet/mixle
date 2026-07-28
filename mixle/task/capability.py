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
from collections.abc import Callable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from mixle.task.calibrate import ESCALATE


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


class CapabilityAbstention(StrEnum):
    """Typed decision value used inside capability evaluation."""

    ESCALATE = "escalate"


CAPABILITY_ESCALATE = CapabilityAbstention.ESCALATE


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


def _exact_outputs(outputs: Any, n: int, *, context: str) -> list[Any]:
    """Validate one batch result as exactly ``n`` per-input outputs, and materialize it as a list.

    Accepts any indexable, sized batch result -- a list or tuple, a NumPy prediction vector, a pandas
    Series, a tensor (MXR-080-1600). Requiring ``collections.abc.Sequence`` rejected all of those,
    even though a NumPy label vector of exactly the right length is the single most ordinary thing a
    batch callable returns. Strings/bytes and mappings/sets are still rejected: the first is a single
    output that merely happens to be iterable, and the latter two carry no positional correspondence
    to the inputs, so neither can be aligned row-for-row.
    """
    if isinstance(outputs, (str, bytes, bytearray, Mapping, AbstractSet)):
        raise ValueError(f"{context} must return one indexable output per input")
    if isinstance(outputs, np.ndarray):
        if outputs.ndim != 1:
            raise ValueError(f"{context} must return a 1-D batch of outputs, got shape {outputs.shape}")
    elif not isinstance(outputs, Sequence) and not (hasattr(outputs, "__len__") and hasattr(outputs, "__getitem__")):
        raise ValueError(f"{context} must return a sequence with exactly one output per input")
    result = list(outputs)
    if len(result) != n:
        raise ValueError(f"{context} returned {len(result)} outputs for {n} inputs")
    return result


def _predict(model: Any, texts: list[str], *, callable_mode: str = "batch") -> list[Any]:
    """Batch-predict labels from ``model``, which may be a ``CalibratedTaskModel``, a ``TaskModel``-like object
    exposing ``batch``, or a bare ``teacher(texts) -> labels`` / ``teacher(text) -> label`` callable.

    Checks ``batch`` before ``task`` so bare ``TaskModel`` instances and
    ``CalibratedTaskModel`` wrappers both use the appropriate batch interface.
    """
    if callable_mode not in {"batch", "item"}:
        raise ValueError("callable_mode must be 'batch' or 'item'")
    if callable(getattr(model, "batch", None)):
        return _exact_outputs(model.batch(texts), len(texts), context="model.batch")
    task = getattr(model, "task", None)
    if callable(getattr(task, "batch", None)):
        return _exact_outputs(task.batch(texts), len(texts), context="model.task.batch")
    if not callable(model):
        raise TypeError("profile model must expose batch/task.batch or be callable")
    if callable_mode == "item":
        return [model(text) for text in texts]
    return _exact_outputs(model(texts), len(texts), context="batch callable")


def _decide(model: Any, texts: list[str]) -> list[Any] | None:
    """Batch decisions (label or ``ESCALATE``) if ``model`` exposes a decision API, else ``None``."""
    if callable(getattr(model, "batch_decide", None)):
        decisions = _exact_outputs(
            model.batch_decide(texts),
            len(texts),
            context="model.batch_decide",
        )
    elif callable(getattr(model, "decide", None)):
        decisions = [model.decide(text) for text in texts]
    else:
        return None
    return [CAPABILITY_ESCALATE if decision is ESCALATE else decision for decision in decisions]


def _agreement(a: Sequence[Any], b: Sequence[Any]) -> float:
    if len(a) != len(b):
        raise ValueError("agreement inputs must have identical cardinality")
    if len(a) == 0:
        raise ValueError("agreement requires at least one evaluated row")
    return float(np.mean([str(x) == str(y) for x, y in zip(a, b, strict=True)]))


def _violation_rate(before: Sequence[Any], after: Sequence[Any]) -> float:
    if len(before) != len(after):
        raise ValueError("invariance inputs must have identical cardinality")
    if len(before) == 0:
        raise ValueError("invariance evaluation requires at least one row")
    return float(np.mean([str(x) != str(y) for x, y in zip(before, after, strict=True)]))


def _escalation_rate(decisions: Sequence[Any]) -> float:
    if len(decisions) == 0:
        raise ValueError("abstention evaluation requires at least one decision")
    return float(np.mean([decision is CAPABILITY_ESCALATE for decision in decisions]))


def capture_profile(
    student: Any,
    teacher: Any,
    texts: Sequence[str],
    suite: CapabilitySuite,
    *,
    student_callable_mode: str = "batch",
    teacher_callable_mode: str = "batch",
) -> dict[str, Any]:
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
    if not isinstance(suite, CapabilitySuite):
        raise TypeError("suite must be a CapabilitySuite")
    if isinstance(texts, (str, bytes)):
        raise TypeError("texts must be a sequence of evaluation strings")
    texts = [str(t) for t in texts]
    if not texts or any(not text for text in texts):
        raise ValueError("capability profiles require non-empty evaluation text")
    for kind, transforms in (
        ("corruption", suite.corruptions),
        ("invariance", suite.invariances),
    ):
        if any(
            not isinstance(name, str) or not name or not callable(transform) for name, transform in transforms.items()
        ):
            raise ValueError(f"{kind} entries require non-empty names and callable transforms")
    if any(not isinstance(probe, str) or not probe for probe in suite.probes):
        raise ValueError("probes must contain non-empty strings")

    def student_predictions(rows: list[str]) -> list[Any]:
        return _predict(student, rows, callable_mode=student_callable_mode)

    def teacher_predictions(rows: list[str]) -> list[Any]:
        return _predict(teacher, rows, callable_mode=teacher_callable_mode)

    student_clean = student_predictions(texts)
    teacher_clean = teacher_predictions(texts)
    profile: dict[str, Any] = {
        "n_evaluated": len(texts),
        "clean_agreement": _agreement(student_clean, teacher_clean),
    }

    corruptions: dict[str, float] = {}
    for name, corrupt in suite.corruptions.items():
        corrupted = [corrupt(t) for t in texts]
        corruptions[name] = _agreement(
            student_predictions(corrupted),
            teacher_predictions(corrupted),
        )
    profile["corruptions"] = corruptions

    invariances: dict[str, dict[str, float]] = {}
    for name, rewrite in suite.invariances.items():
        rewritten = [rewrite(t) for t in texts]
        invariances[name] = {
            "student_violation_rate": _violation_rate(
                student_clean,
                student_predictions(rewritten),
            ),
            "teacher_violation_rate": _violation_rate(
                teacher_clean,
                teacher_predictions(rewritten),
            ),
        }
    profile["invariances"] = invariances

    if suite.probes:
        profile["probes"] = {
            "student": student_predictions(list(suite.probes)),
            "teacher": teacher_predictions(list(suite.probes)),
        }

    student_decisions = _decide(student, texts)
    teacher_decisions = _decide(teacher, texts)
    if student_decisions is not None or teacher_decisions is not None:
        profile["abstention"] = {
            "student_escalation_rate": _escalation_rate(student_decisions) if student_decisions is not None else None,
            "teacher_escalation_rate": _escalation_rate(teacher_decisions) if teacher_decisions is not None else None,
        }

    return profile
