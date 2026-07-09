"""Measure whether accumulated calibrated knowledge improves held-out answers.

The knowledge-accumulation flywheel measures whether adding calibrated
knowledge to the belief store (:mod:`mixle.substrate.belief`) improves answers
on a held-out question set, with no model retraining. Distillation improves the
student model; accumulation improves what the student or teacher can retrieve.

Two guards keep the measurement grounded:

  * **attribution** -- the improvement must disappear when the newly-assimilated items are withheld
    from retrieval, proving the gain came from the store growing rather than anything else (timing,
    caching, a lucky ``answer_fn``).
  * **credence weighting** -- retrieval goes through
    :func:`mixle.substrate.belief.retrieve_beliefs`, which ranks by ``relevance * credence`` and can be
    hard-thresholded with ``min_credence``; a batch of low-credence (e.g. pure ``MODEL_ASSERTION``)
    knowledge must not inflate the measured improvement -- it is down-weighted, never treated as
    ground truth.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from mixle.substrate.belief import retrieve_beliefs
from mixle.substrate.core import Substrate


@dataclass
class QAItem:
    """One held-out question: ``answer_fn`` is judged correct on it if it produces ``answer`` from the
    retrieved context alone."""

    question: str
    answer: str


@dataclass
class FlywheelMeasurement:
    """Held-out answer quality and grounding rate for one flywheel measurement."""

    solve_rate: float
    grounded_fraction: float  # fraction of questions where retrieval returned at least one belief


@dataclass
class FlywheelReport:
    """Before/after/withheld flywheel measurements with an attribution check."""

    before: FlywheelMeasurement
    after: FlywheelMeasurement
    withheld: FlywheelMeasurement  # measured after assimilation, but with the new items excluded from retrieval
    attribution_confirmed: bool


def _measure(
    sub: Substrate,
    questions: Sequence[QAItem],
    answer_fn: Callable[[str, list[str]], str],
    *,
    k: int,
    min_credence: float | None,
    exclude_ids: set[str],
) -> FlywheelMeasurement:
    n_correct = 0
    n_grounded = 0
    for qa in questions:
        beliefs = retrieve_beliefs(sub, qa.question, k=k + len(exclude_ids), min_credence=min_credence)
        beliefs = [b for b in beliefs if b.id not in exclude_ids][:k]
        if beliefs:
            n_grounded += 1
        context = [b.claim.text for b in beliefs]
        if answer_fn(qa.question, context) == qa.answer:
            n_correct += 1
    n = len(questions) or 1
    return FlywheelMeasurement(solve_rate=n_correct / n, grounded_fraction=n_grounded / n)


def measure_flywheel(
    sub: Substrate,
    questions: Sequence[QAItem],
    answer_fn: Callable[[str, list[str]], str],
    assimilate_batch: Callable[[Substrate], list[str]],
    *,
    k: int = 5,
    min_credence: float | None = None,
) -> FlywheelReport:
    """Measure ``answer_fn`` against ``questions`` before and after ``assimilate_batch(sub)`` adds a
    batch of calibrated beliefs (returning the ids it touched), with a THIRD measurement that excludes
    exactly those ids from retrieval -- the attribution control. ``answer_fn`` and every other piece of
    the system stay fixed throughout: only the store's content (and what it makes retrievable) changes.
    """
    before = _measure(sub, questions, answer_fn, k=k, min_credence=min_credence, exclude_ids=set())
    added_ids = set(assimilate_batch(sub))
    after = _measure(sub, questions, answer_fn, k=k, min_credence=min_credence, exclude_ids=set())
    withheld = _measure(sub, questions, answer_fn, k=k, min_credence=min_credence, exclude_ids=added_ids)

    attribution_confirmed = after.solve_rate > before.solve_rate and withheld.solve_rate <= before.solve_rate + 1e-9
    return FlywheelReport(before=before, after=after, withheld=withheld, attribution_confirmed=attribution_confirmed)
