"""Information-gain retrieval over substrate items.

Score substrate items by how much they would actually move a belief, not by how
textually similar they are to the query.

:func:`~mixle.substrate.retrieve.retrieve` ranks by cosine/lexical similarity -- a sound default, but many
similar-looking items can carry the *same* evidence (redundant), while a single differently-worded item can
be decisive. :func:`eig_retrieve` instead scores each candidate by the entropy it would actually remove from a
given :class:`~mixle.inference.belief.BeliefState` if assimilated, and greedily picks the highest-gain item
each round, updating the running belief before scoring what remains -- so a second item redundant with the
first correctly scores near zero the next round. Experiment-design workflows can reuse the same greedy EIG
scorer; it is written once, here.

**Expected, not realized, gain (MXR-080-0250).** Most substrate items are already fully in hand -- their
content, and therefore their evidence, is deterministically known before scoring (``evidence_fn`` just reads
it off, e.g. a precomputed log-likelihood payload). A genuine experiment-design candidate is different: what
it would show is not known until it is actually run. For those, ``evidence_fn`` returns
:class:`EvidenceOutcomes` -- the candidate's own predictive distribution over what its evidence could turn
out to be -- and the score becomes the true expected posterior-entropy reduction, averaged over that
distribution, rather than the entropy change from one arbitrarily-picked realization (which can diverge
sharply from the real expectation: entropy is a concave functional of the belief, so "entropy after updating
on the average evidence" is not "the average of the entropy after each possible update"). A deterministic
evidence value is the degenerate, single-outcome case of the same formula, so both cases share one code
path. A candidate is also never picked with a non-positive expected gain: once the best remaining candidate
cannot clear a strictly-positive bar, :func:`eig_retrieve` stops rather than present a harmful
(entropy-increasing) or useless (zero-information) pick as if it were beneficial evidence.

**Copy-on-update, narrow failure handling (MXR-080-0251).** :meth:`~mixle.inference.belief.BeliefState.update`
is documented to return a *new* belief, but nothing enforces that a given realization actually honors it --
so every hypothetical update here runs against a fresh clone of the running belief, never the running belief
itself, and the caller's own ``belief`` argument is never mutated by this function no matter how
``update()`` is actually implemented. Only a declared evidence-incompatibility (``ValueError``, the same
type :class:`~mixle.inference.belief.BeliefState` realizations in this codebase raise for
malformed/incompatible evidence) is treated as "this candidate has no usable evidence"; any other exception
is a genuine bug and propagates. Every skipped candidate is recorded, with why, on the returned
:class:`EigRetrieval` -- never silently dropped.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np
from numpy.random import RandomState

from mixle.inference.belief import BeliefState
from mixle.substrate.core import Substrate, SubstrateItem
from mixle.substrate.retrieve import Retrieval


@dataclass(frozen=True)
class EvidenceOutcome:
    """One possible realization of a candidate's evidence, weighted by its probability of occurring."""

    probability: float
    evidence: Any

    def __post_init__(self) -> None:
        """Canonicalize the probability to a builtin float (MXR-080-1884).

        ``EvidenceOutcomes`` validated ``float(o.probability)`` and then discarded the conversion, so
        the ORIGINAL value stayed on the outcome and was what scoring later multiplied. A string
        ``"0.5"`` passed validation and failed several frames away inside the entropy computation --
        the exact confusion the eager validation was written to prevent -- and a bool sailed through
        as 0/1. Converting here means what was checked is what is used.
        """
        value = self.probability
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise TypeError(f"EvidenceOutcome probability must be a real number, got {value!r}")
        numeric = float(value)
        if not np.isfinite(numeric) or numeric < 0.0:
            raise ValueError(f"EvidenceOutcome probability must be finite and non-negative, got {value!r}")
        object.__setattr__(self, "probability", numeric)


@dataclass(frozen=True)
class EvidenceOutcomes:
    """A candidate's predictive distribution over what its evidence could turn out to be (MXR-080-0250).

    Return this from ``evidence_fn`` instead of a bare evidence value when a candidate's true evidence is
    NOT yet known at scoring time -- e.g. a not-yet-run experiment or a diagnostic whose result is
    uncertain until performed. ``outcomes`` is the finite set of what it could resolve to, each paired
    with its probability under the current belief's own predictive distribution; :func:`eig_retrieve`
    scores the candidate by the true expectation, ``H[belief] - E[H[belief | outcome]]``, over this set,
    rather than the entropy change from a single realized update. Validated eagerly, the same discipline
    :class:`~mixle.inference.belief.CategoricalBelief` applies to its own probabilities: every
    ``probability`` must be finite and non-negative, and they must sum to 1 (within floating-point
    tolerance) -- a caller that gets this wrong finds out immediately, at the point they built the
    outcome set, not several stack frames away inside a confusing entropy computation.
    """

    outcomes: Sequence[EvidenceOutcome]

    def __post_init__(self) -> None:
        if not self.outcomes:
            raise ValueError("EvidenceOutcomes requires at least one outcome")
        probs = [float(o.probability) for o in self.outcomes]
        if any(not np.isfinite(p) or p < 0.0 for p in probs):
            raise ValueError(f"EvidenceOutcomes probabilities must be finite and non-negative, got {probs}")
        total = sum(probs)
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(f"EvidenceOutcomes probabilities must sum to 1.0, got {total}")


@dataclass
class SkippedCandidate:
    """A substrate item scoring could not use, and why (MXR-080-0251) -- surfaced, never just dropped."""

    item: SubstrateItem
    reason: str


class EigRetrieval(Retrieval):
    """A :class:`~mixle.substrate.retrieve.Retrieval` from :func:`eig_retrieve`, plus the diagnostics a
    similarity retrieval has no use for: which candidates scoring skipped (and why), and whether the run
    stopped early because no remaining candidate cleared the admissible-gain bar (MXR-080-0250/0251).

    Deliberately a plain subclass with its own ``__init__`` rather than a second ``@dataclass`` layer:
    :class:`Retrieval` itself is ``@dataclass(init=False)`` with a hand-written constructor that builds
    its validated ``hits`` tuple from ``items``/``scores`` (MXR-080-0248, e.g. rejecting a length
    mismatch or a non-finite score) -- re-declaring ``EigRetrieval`` as its own ``@dataclass`` would let
    dataclass field inheritance synthesize a DIFFERENT ``__init__`` over the raw field list instead
    (``query, hits=(), skipped=..., stop_reason=...``), silently bypassing that validation. Delegating to
    ``Retrieval.__init__`` via ``super()`` keeps construction going through the one real validation path
    regardless of how ``Retrieval`` represents itself internally.
    """

    def __init__(
        self,
        query: str,
        items: list[SubstrateItem] | None = None,
        scores: list[float] | None = None,
        *,
        skipped: list[SkippedCandidate] | None = None,
        stop_reason: str | None = None,
    ) -> None:
        super().__init__(query, items, scores)
        self.skipped: list[SkippedCandidate] = list(skipped) if skipped is not None else []
        self.stop_reason = stop_reason


def _as_rng(seed: int | RandomState | None) -> RandomState:
    return seed if isinstance(seed, RandomState) else RandomState(seed)


def _clone(belief: BeliefState) -> BeliefState:
    """A defensive copy of ``belief`` before a hypothetical candidate update (MXR-080-0251).

    Prefers the belief's own ``copy()``/``clone()`` when it defines one; falls back to ``copy.deepcopy``
    for any :class:`BeliefState` realization that doesn't. :meth:`BeliefState.update` is documented to
    return a *new* belief, but nothing enforces that a given realization actually honors it -- cloning
    before every hypothetical update means even a non-compliant ``update()`` that mutates ``self`` in
    place can never contaminate ``current``, the baseline every candidate in the round is scored against.
    """
    for attr in ("copy", "clone"):
        fn = getattr(belief, attr, None)
        if callable(fn):
            duplicate = fn()
            # A copy() that returns self defeats the whole defence: the "clone" IS current, so a
            # non-compliant update() mutating in place contaminates the baseline every sibling
            # candidate is scored against -- which is the one thing this function exists to prevent
            # (MXR-080-1884). Verified rather than assumed, and deepcopy is tried before giving up.
            if duplicate is not belief:
                return duplicate
            break
    duplicate = copy.deepcopy(belief)
    if duplicate is belief:
        raise ValueError(
            f"{type(belief).__name__}.copy()/clone() and copy.deepcopy both returned the SAME object, "
            "so a hypothetical candidate update cannot be isolated from the current belief. Scoring "
            "would mutate the baseline every candidate is compared against. Give the belief a copy() "
            "that returns a new object."
        )
    return duplicate


def _as_outcomes(raw: Any) -> Sequence[EvidenceOutcome]:
    """Normalize ``evidence_fn``'s return value to a sequence of outcomes: itself, or a single point mass."""
    if isinstance(raw, EvidenceOutcomes):
        return raw.outcomes
    return (EvidenceOutcome(1.0, raw),)


def _score_candidate(
    current: BeliefState,
    current_entropy: float,
    evidence_fn: Callable[[SubstrateItem], Any],
    item: SubstrateItem,
) -> tuple[float, list[tuple[float, BeliefState]]]:
    """Expected posterior-entropy reduction for ``item``, and the belief choosing it would move to.

    Every outcome's hypothetical update runs against its own fresh clone of ``current`` (MXR-080-0251), so
    scoring this candidate can never mutate ``current`` itself -- the same baseline every sibling candidate
    in this round is scored against. When ``item``'s evidence is a single deterministic value (the common
    case), the returned belief is the exact post-update state. When it is a genuine
    :class:`EvidenceOutcomes`, the true outcome cannot be known without actually acquiring it -- the
    returned belief is the post-update state for ONE outcome, drawn from ``rng`` per the candidate's own
    outcome probabilities (an honest simulation of what the next round would actually see, consistent with
    the nested-Monte-Carlo EIG estimators elsewhere in this codebase, e.g.
    :func:`mixle.doe.active.expected_information_gain_nmc`, never cherry-picked favorably).

    Raises ``ValueError`` -- and only ``ValueError`` -- when ``item`` has no usable evidence: either
    ``evidence_fn`` raises it directly, or an outcome fails ``current.update()``'s own validation. Any
    other exception is a genuine bug (in ``evidence_fn``, in ``update``, or elsewhere) and propagates.
    """
    outcomes = _as_outcomes(evidence_fn(item))
    expected_next_entropy = 0.0
    posteriors: list[tuple[float, BeliefState]] = []
    for outcome in outcomes:
        nxt = _clone(current).update(outcome.evidence)
        expected_next_entropy += outcome.probability * nxt.entropy()
        posteriors.append((outcome.probability, nxt))
    return float(current_entropy - expected_next_entropy), posteriors


def _realize(posteriors: list[tuple[float, BeliefState]], rng: RandomState) -> BeliefState:
    """Draw the outcome that actually occurred, for the SELECTED candidate only (MXR-080-1884).

    This draw used to happen inside scoring, once per candidate. Scoring is a hypothetical -- nothing
    is acquired, and every candidate but one is discarded -- so every rejected candidate consumed
    draws from the shared stream and shifted it for the winner. The carried posterior therefore
    depended on how many *other* items happened to be scored first: with the same seed and the same
    winner, adding an unrelated candidate to the pool changed the belief the run continued from
    (the audit measured 0.0 versus 0.4 at seed=1).

    Acquisition happens once, after selection, so the stream is consumed only by outcomes that were
    actually realized.
    """
    if len(posteriors) == 1:
        return posteriors[0][1]
    probs = np.array([p for p, _ in posteriors], dtype=np.float64)
    probs = probs / probs.sum()  # re-normalize: EvidenceOutcomes only guarantees close to 1, not exact
    return posteriors[int(rng.choice(len(posteriors), p=probs))][1]


def eig_retrieve(
    substrate: Substrate,
    belief: BeliefState,
    evidence_fn: Callable[[SubstrateItem], Any],
    *,
    k: int = 8,
    kind: str | None = None,
    scope: str | None = None,
    seed: int | RandomState | None = None,
) -> EigRetrieval:
    """Greedily pick up to ``k`` substrate items by expected posterior-entropy reduction against ``belief``.

    ``evidence_fn(item)`` turns a candidate item into whatever ``belief.update(...)`` expects (e.g. a
    per-hypothesis log-likelihood vector for a :class:`~mixle.inference.belief.CategoricalBelief``), OR --
    when the item's true evidence is not yet known -- an :class:`EvidenceOutcomes` describing what it
    could turn out to be. Each round, every remaining candidate is scored by the expected entropy
    reduction ``current.entropy() - E[updated.entropy()]`` (MXR-080-0250; the exact realized reduction in
    the common deterministic-evidence case, since that is the same formula's single-outcome degenerate
    case); the best-scoring item is taken -- but only if its gain clears zero, otherwise
    :func:`eig_retrieve` stops rather than hand back a pick that would increase uncertainty (or add none)
    dressed as evidence -- the running belief moves to its post-update state, and scoring repeats against
    the shrunk pool, so an item whose evidence is redundant with an already-picked item scores near zero on
    its next look, the direct fix for similarity retrieval pulling in near-duplicates.

    ``belief`` itself is never mutated: every hypothetical update runs against a fresh clone
    (MXR-080-0251), so a candidate that turns out not to be chosen -- or a
    :class:`~mixle.inference.belief.BeliefState` realization whose ``update()`` does not honor the
    documented "returns a new belief" contract -- can never contaminate scoring for the rest of the round.
    ``evidence_fn`` should raise ``ValueError`` to signal "no usable evidence for this item" (mirroring
    what ``belief.update()`` itself raises for incompatible evidence); that is the only exception type
    treated as a skip -- every skip is recorded, with its reason, on the returned :class:`EigRetrieval`
    rather than silently dropped, and any other exception is a genuine bug and propagates. ``seed`` (an
    ``int`` or ``numpy.random.RandomState``) is used only when a chosen candidate carries genuine outcome
    uncertainty, to draw which outcome is treated as realized for the next round's baseline.

    Returned as an :class:`EigRetrieval` (``query`` is a fixed marker, not a text query) so it composes
    with the same ``to_context``/``by_kind`` surface as cosine retrieval, plus ``.skipped`` and
    ``.stop_reason`` diagnostics cosine retrieval has no equivalent of.
    """
    pool = [it for it in substrate.all(scope=scope) if kind is None or it.kind == kind]
    chosen: list[SubstrateItem] = []
    scores: list[float] = []
    skipped: list[SkippedCandidate] = []
    current = belief
    remaining = list(pool)
    rng = _as_rng(seed)
    stop_reason: str | None = None
    while remaining and len(chosen) < k:
        current_entropy = current.entropy()
        evaluated: list[tuple[SubstrateItem, float, list[tuple[float, BeliefState]]]] = []
        for item in remaining:
            try:
                gain, posteriors = _score_candidate(current, current_entropy, evidence_fn, item)
            except ValueError as exc:
                skipped.append(SkippedCandidate(item=item, reason=str(exc)))
                continue
            evaluated.append((item, gain, posteriors))
        if not evaluated:
            stop_reason = "no_usable_candidates"
            break
        best_item, best_gain, best_posteriors = max(evaluated, key=lambda t: t[1])
        if best_gain <= 0.0:
            stop_reason = "no_admissible_positive_gain"
            break
        chosen.append(best_item)
        scores.append(best_gain)
        # Acquire only what was selected: the draw is made here, after the winner is known, so
        # rejected candidates no longer consume the stream (MXR-080-1884).
        current = _realize(best_posteriors, rng)
        remaining = [item for item, _, _ in evaluated if item is not best_item]
    return EigRetrieval(
        query="<information-gain>", items=chosen, scores=scores, skipped=skipped, stop_reason=stop_reason
    )
