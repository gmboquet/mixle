"""Measure whether accumulated calibrated knowledge improves held-out answers.

The knowledge-accumulation flywheel measures whether adding calibrated
knowledge to the belief store (:mod:`mixle.substrate.belief`) improves answers
on a held-out question set, with no model retraining. Distillation improves the
student model; accumulation improves what the student or teacher can retrieve.

Two guards keep the measurement grounded:

  * **attribution** -- the improvement must disappear when the newly-assimilated items are withheld
    from retrieval, proving the gain came from the store growing rather than anything else (timing,
    caching, a lucky ``answer_fn``). Withholding is delta-level, not belief-level (MXR-080-0246): a
    belief ``assimilate_batch`` UPDATED (rather than created from nothing) already existed for the
    ``before`` pass, so the withheld measurement rolls it back to exactly its pre-batch snapshot for
    that one measurement rather than hiding it outright -- otherwise pre-existing evidence that had
    nothing to do with the batch would vanish from the counterfactual too, driving withheld quality
    below the true baseline and fabricating or inflating the measured attribution.
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
from mixle.substrate.core import Substrate, SubstrateItem


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
    withheld: FlywheelMeasurement  # measured after assimilation, with only the batch's OWN contribution removed
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
    batch of calibrated beliefs (returning the ids it touched), with a THIRD measurement that isolates
    exactly the batch's own contribution -- the attribution control. ``answer_fn`` and every other piece
    of the system stay fixed throughout: only the store's content (and what it makes retrievable)
    changes.

    Withheld isolation is delta-level, not belief-level (MXR-080-0246): ``assimilate_batch`` may UPDATE
    a belief that already existed -- and was already visible to the ``before`` pass -- rather than
    create one from nothing. An id it returns that was already present pre-batch is rolled back, for the
    duration of the withheld measurement only, to exactly its pre-batch snapshot (its pre-existing
    evidence, unmodified); an id that did not exist pre-batch is excluded from retrieval outright, same
    as before, since there is no pre-batch state for it to roll back to. Either way ``sub`` is restored
    to its true post-assimilation content before this function returns -- the rollback is an internal,
    temporary device for taking one measurement, never an observable side effect of calling this
    function.
    """
    before = _measure(sub, questions, answer_fn, k=k, min_credence=min_credence, exclude_ids=set())

    # MXR-080-0246: a full pre-batch snapshot of every record (belief or not -- retrieve_beliefs itself
    # scans the whole "record" kind, so a non-belief record could in principle collide on retrieval
    # ranking too), taken right before the one call to assimilate_batch that is allowed to mutate `sub`,
    # so an UPDATED belief's pre-existing state can be replayed rather than just erased. `Substrate.all`/
    # `.get`/`.put` all cross a deep-copy boundary (see core.py's immutability contract), so these
    # snapshots are genuinely independent of whatever assimilate_batch does to `sub` afterward.
    pre_batch: dict[str, SubstrateItem] = {item.id: item for item in sub.all(kind="record")}
    added_ids = set(assimilate_batch(sub))
    after = _measure(sub, questions, answer_fn, k=k, min_credence=min_credence, exclude_ids=set())

    created_ids = added_ids - pre_batch.keys()  # genuinely new -- no pre-batch state to roll back to
    updated_ids = added_ids - created_ids  # pre-existing -- roll back to exactly what they looked like before
    post_batch_updated = {i: sub.get(i) for i in updated_ids}
    for i in updated_ids:
        sub.put(pre_batch[i])
    try:
        withheld = _measure(sub, questions, answer_fn, k=k, min_credence=min_credence, exclude_ids=created_ids)
    finally:
        # Restore the true post-assimilation state no matter how the measurement above went, so the
        # rollback above is never an observable side effect of calling this function.
        for i, item in post_batch_updated.items():
            if item is not None:
                sub.put(item)

    attribution_confirmed = after.solve_rate > before.solve_rate and withheld.solve_rate <= before.solve_rate + 1e-9
    return FlywheelReport(before=before, after=after, withheld=withheld, attribution_confirmed=attribution_confirmed)
