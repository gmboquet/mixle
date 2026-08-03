"""Measure whether accumulated calibrated knowledge improves held-out answers.

The knowledge-accumulation flywheel measures whether adding calibrated
knowledge to the belief store (:mod:`mixle.substrate.belief`) improves answers
on a held-out question set, with no model retraining. Distillation improves the
student model; accumulation improves what the student or teacher can retrieve.

Three guards keep the measurement grounded:

  * **attribution** -- the improvement must disappear when the newly-assimilated items are withheld
    from retrieval, proving the gain came from the store growing rather than anything else (timing,
    caching, a lucky ``answer_fn``). Withholding is delta-level, not belief-level (MXR-080-0246): a
    belief ``assimilate_batch`` UPDATED (rather than created from nothing) already existed for the
    ``before`` pass, so the withheld measurement rolls it back to exactly its pre-batch snapshot for
    that one measurement rather than hiding it outright -- otherwise pre-existing evidence that had
    nothing to do with the batch would vanish from the counterfactual too, driving withheld quality
    below the true baseline and fabricating or inflating the measured attribution.
  * **evaluator control** (MXR-080-0247) -- ``answer_fn`` is reset (via ``reset_answer_fn``) to an
    identical starting state before every one of the before/after/withheld passes, so a stateful
    evaluator -- a cache, a call counter, a seeded sampler -- cannot drift across passes and masquerade
    as a store-content effect. ``trials > 1`` repeats each pass that many times (reset before every
    replicate) and keeps every individual rate, not just the mean, so attribution becomes a paired
    comparison across matched replicates with visible spread instead of one unrepeated scalar.
  * **credence weighting** -- retrieval goes through
    :func:`mixle.substrate.belief.retrieve_beliefs`, which ranks by ``relevance * credence`` and can be
    hard-thresholded with ``min_credence``; a batch of low-credence (e.g. pure ``MODEL_ASSERTION``)
    knowledge must not inflate the measured improvement -- it is down-weighted, never treated as
    ground truth.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from statistics import mean

from mixle.substrate.belief import retrieve_beliefs
from mixle.substrate.core import Substrate, SubstrateItem


@dataclass
class QAItem:
    """One held-out question: ``answer_fn`` is judged correct on it if it produces ``answer`` from the
    retrieved context alone."""

    question: str
    answer: str


@dataclass(frozen=True)
class FlywheelMeasurement:
    """Held-out answer quality and grounding rate for one flywheel measurement.

    ``trial_solve_rates`` / ``trial_grounded_fractions`` carry one entry per replicate trial
    (MXR-080-0247): a 1-tuple equal to ``solve_rate`` / ``grounded_fraction`` for the default
    ``trials=1``, longer when :func:`measure_flywheel` is asked to repeat each pass. Exposed raw --
    rather than collapsed into one particular spread statistic -- so a caller can compute whatever
    uncertainty measure (stderr, a confidence interval, a paired significance test) actually fits its
    own ``answer_fn``, instead of this module asserting one test is correct for an arbitrary evaluator.
    """

    solve_rate: float
    grounded_fraction: float  # fraction of questions where retrieval returned at least one belief
    trial_solve_rates: tuple[float, ...] = field(default_factory=tuple)
    trial_grounded_fractions: tuple[float, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
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
    rate, grounded = n_correct / n, n_grounded / n
    return FlywheelMeasurement(
        solve_rate=rate, grounded_fraction=grounded, trial_solve_rates=(rate,), trial_grounded_fractions=(grounded,)
    )


def _aggregate(samples: Sequence[FlywheelMeasurement]) -> FlywheelMeasurement:
    """Combine ``trials`` independent :func:`_measure` replicates -- each taken with ``answer_fn`` reset
    to the same starting state first -- into one measurement: ``solve_rate`` / ``grounded_fraction`` are
    the mean, and every replicate's own rate survives on ``trial_solve_rates`` / ``trial_grounded_
    fractions`` for the paired comparison in :func:`_attribution_confirmed` (MXR-080-0247)."""
    solve_rates = tuple(s.solve_rate for s in samples)
    grounded_fractions = tuple(s.grounded_fraction for s in samples)
    return FlywheelMeasurement(
        solve_rate=mean(solve_rates),
        grounded_fraction=mean(grounded_fractions),
        trial_solve_rates=solve_rates,
        trial_grounded_fractions=grounded_fractions,
    )


def _attribution_confirmed(
    before: FlywheelMeasurement, after: FlywheelMeasurement, withheld: FlywheelMeasurement
) -> bool:
    """Whether the after-vs-before gain is attributable to the batch: a genuine improvement that
    (on average) disappears once exactly the batch's own contribution is withheld.

    With a single trial this is the original single-scalar comparison, unchanged. With more than one
    (MXR-080-0247), it is a PAIRED comparison over matched trial indices instead of comparing the three
    passes' overall means independently: trial ``i``'s before/after/withheld all started from the same
    freshly reset evaluator state, so differencing at matched ``i`` cancels whatever per-replicate
    evaluator variation the reset alone doesn't -- the same reason a paired t-test controls for
    subject-level variation that a two-sample test would otherwise fold into the noise.
    """
    if len(before.trial_solve_rates) <= 1:
        return after.solve_rate > before.solve_rate and withheld.solve_rate <= before.solve_rate + 1e-9
    d_after = [a - b for a, b in zip(after.trial_solve_rates, before.trial_solve_rates, strict=True)]
    d_withheld = [w - b for w, b in zip(withheld.trial_solve_rates, before.trial_solve_rates, strict=True)]
    return mean(d_after) > 0 and mean(d_withheld) <= 1e-9


def measure_flywheel(
    sub: Substrate,
    questions: Sequence[QAItem],
    answer_fn: Callable[[str, list[str]], str],
    assimilate_batch: Callable[[Substrate], list[str]],
    *,
    k: int = 5,
    min_credence: float | None = None,
    reset_answer_fn: Callable[[], None] | None = None,
    trials: int = 1,
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

    ``answer_fn`` must be a pure function of ``(question, context)``, OR ``reset_answer_fn`` must rewind
    whatever internal state it has to an identical starting point (MXR-080-0247): when given, it is
    called immediately before every individual measurement -- every trial of every one of the three
    passes -- so a stateful evaluator (a cache, a call counter, a seeded sampler) cannot drift across
    before/after/withheld and masquerade as a store-content effect. ``trials`` (requires
    ``reset_answer_fn`` when greater than 1 -- without a reset, repeated calls just accumulate whatever
    state ``answer_fn`` carries between them rather than sampling independent replicates) repeats each
    pass that many times and keeps every individual rate alongside the mean, turning the attribution
    check into a paired comparison across matched replicates instead of one unrepeated scalar.
    """
    if trials < 1:
        raise ValueError(f"trials must be >= 1, got {trials}")
    if trials > 1 and reset_answer_fn is None:
        raise ValueError(
            "trials > 1 requires reset_answer_fn (MXR-080-0247): replicate measurements are only a "
            "valid paired sample when answer_fn is rewound to the same starting state before each one; "
            "otherwise repeated calls just accumulate whatever state answer_fn carries between them, "
            "which is not an independent replicate of anything."
        )

    def run(exclude_ids: set[str]) -> FlywheelMeasurement:
        samples = []
        for _ in range(trials):
            if reset_answer_fn is not None:
                reset_answer_fn()
            samples.append(_measure(sub, questions, answer_fn, k=k, min_credence=min_credence, exclude_ids=exclude_ids))
        return _aggregate(samples)

    before = run(set())

    # MXR-080-0246: a full pre-batch snapshot of every record (belief or not -- retrieve_beliefs itself
    # scans the whole "record" kind, so a non-belief record could in principle collide on retrieval
    # ranking too), taken right before the one call to assimilate_batch that is allowed to mutate `sub`,
    # so an UPDATED belief's pre-existing state can be replayed rather than just erased. `Substrate.all`/
    # `.get`/`.put` all cross a deep-copy boundary (see core.py's immutability contract), so these
    # snapshots are genuinely independent of whatever assimilate_batch does to `sub` afterward.
    pre_batch: dict[str, SubstrateItem] = {item.id: item for item in sub.all(kind="record")}
    added_ids = set(assimilate_batch(sub))
    after = run(set())

    created_ids = added_ids - pre_batch.keys()  # genuinely new -- no pre-batch state to roll back to
    updated_ids = added_ids - created_ids  # pre-existing -- roll back to exactly what they looked like before
    post_batch_updated = {i: sub.get(i) for i in updated_ids}
    for i in updated_ids:
        sub.put(pre_batch[i])
    try:
        withheld = run(created_ids)
    finally:
        # Restore the true post-assimilation state no matter how the measurement above went, so the
        # rollback above is never an observable side effect of calling this function.
        for i, item in post_batch_updated.items():
            if item is not None:
                sub.put(item)

    attribution_confirmed = _attribution_confirmed(before, after, withheld)
    return FlywheelReport(before=before, after=after, withheld=withheld, attribution_confirmed=attribution_confirmed)
