"""System-level scorecard for held-out question sets.

Named ``SystemScorecard`` (not ``Scorecard``) to avoid colliding with
:class:`mixle.task.scorecard.Scorecard` -- a different, narrower comparison (one student solution vs its
teacher on a task); this one evaluates a whole :class:`~mixle.system.core.System` (teacher, captured cache,
degraded modes, budget, all of it) end to end.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from mixle.data.hashing import dataset_hash
from mixle.system.core import Query, System

#: Bumped whenever :func:`_default_scorer`'s judgement changes, so cards judged by different versions
#: of the default judge get different :func:`question_set_identity` values and cannot be compared.
DEFAULT_SCORER_VERSION = "substring/v2"


def _require_reference(expected: str) -> str:
    """A non-empty reference answer, or ``ValueError``.

    An empty ``expected`` made the substring test vacuously true, so every non-``None`` reply scored
    correct and the card reported ``quality=1.0`` while measuring nothing at all. A reference that
    cannot discriminate is invalid scoring data, not a lenient one.
    """
    if not isinstance(expected, str) or not expected.strip():
        raise ValueError(f"expected answer must be a non-empty reference string, got {expected!r}")
    return expected


def _default_scorer(reply: str | None, expected: str) -> bool:
    """Lexical baseline judge: is the (validated, non-empty) reference a substring of the reply?

    This is a weak judge and is documented as one. It is purely lexical: it cannot detect negation
    ("Paris is not the answer" contains "Paris" and scores correct for expected "Paris"), paraphrase,
    or semantic equivalence, so a scorecard built on it bounds quality from *above* and
    :func:`~mixle.system.meta.improve_by_regret`, which allocates effort by measured quality, is only
    as trustworthy as this judge. Pass a task-specific ``scorer`` to :func:`evaluate` for anything
    where being wrong about correctness matters; that scorer's identity is folded into
    :func:`question_set_identity`, so cards from different judges are never compared to each other.

    What it now refuses is scoring against a reference that cannot discriminate at all (see
    :func:`_require_reference`).
    """
    _require_reference(expected)
    return reply is not None and expected.strip().lower() in reply.strip().lower()


def _scorer_identity(scorer: Callable[[str | None, str], bool]) -> str:
    """A stable name for the judge used to produce a scorecard (part of the card's held-out identity).

    ``repr(scorer)`` is NOT an acceptable fallback (MXR-080-1902). An ordinary callable object's repr
    is ``<pkg.Judge object at 0x10c3f5a90>`` -- a memory address, so two structurally identical judges
    in one process got two different identities, and the same judge got a new one in every process.
    That does not weaken the gate, it inverts it: :func:`detect_regression` reads a mismatched
    identity as ``comparable=False, regressed=True``, so an improved round measured by the same judge
    in a fresh process is reported as a regression it never had.

    A judge that exposes ``__qualname__`` (a function, a lambda, a bound method) has a genuine,
    content-determined name and is used as-is. Anything else is refused rather than fingerprinted by
    process-dependent text -- the same closed-schema discipline
    :func:`mixle.data.hashing._canonical` applies to provenance digests.

    What this deliberately does NOT do is derive an identity from a callable object's attributes.
    ``vars(judge)`` does not determine what a judge decides: a closure it holds, a model it calls, or
    module state it reads are all invisible there, so a digest over them would assert that two judges
    are the same evidence on grounds that cannot support the claim -- and a false MATCH lets
    :func:`detect_regression` compare cards from different judges, which is the failure this identity
    exists to prevent. Wrapping the judge in a module-level ``def`` (which the error names) is the
    caller-side fix, and it is the caller who knows whether two configurations are the same judge.
    """
    qualname = getattr(scorer, "__qualname__", None)
    if not isinstance(qualname, str) or not qualname:
        raise ValueError(
            f"scorer {type(scorer).__name__} has no stable identity: a scorecard is bound to the judge "
            "that produced it, and this one exposes no __qualname__, leaving only repr() -- which for a "
            "callable object or a functools.partial embeds a memory address and so names a different "
            "judge in every process. Wrap it in a module-level function (def my_judge(reply, expected): "
            "...) and pass that instead."
        )
    name = f"{getattr(scorer, '__module__', '?')}.{qualname}"
    return f"{name}@{DEFAULT_SCORER_VERSION}" if scorer is _default_scorer else name


def question_set_identity(
    question_set: Sequence[tuple[Query, str]], *, scorer: Callable[[str | None, str], bool] | None = None
) -> str:
    """The identity two scorecards must share before :func:`detect_regression` may compare them.

    A regression comparison is only meaningful between cards measured on the SAME held-out set with
    the SAME judge: a "perfect" card over a different (or empty) question set, or the same set scored
    by a laxer scorer, is not evidence that a round did not regress. Digesting the ``(query, expected)``
    pairs in order plus the scorer's identity turns that documented precondition into something
    :func:`detect_regression` can actually check, instead of silently accepting mismatched evidence.

    The pairs are digested by CONTENT (:func:`mixle.data.hashing.dataset_hash`, order-sensitive),
    never by ``repr`` (MXR-080-1902). ``Query.fingerprint`` is typed ``Any``, and ``repr`` of an
    ordinary object embeds its memory address, so two equal question sets built from equal
    fingerprint objects digested differently -- and :func:`detect_regression` then reported an
    improved round against the same held-out set as ``comparable=False, regressed=True``, a
    fabricated regression on the exact gate this identity exists to make trustworthy.
    ``dataset_hash`` encodes a ``Query`` structurally by its declared dataclass fields and refuses
    (rather than repr-ing) any value it cannot encode canonically, which is what makes an identity
    computed in one process match one computed in another.

    Note that this changes the digest of every question set relative to the ``repr``-based scheme, so
    an identity persisted by an older build will not match one computed here. That is unavoidable and
    correct: the old digest's whole problem is that it did not identify what it claimed to.
    """
    h = hashlib.sha256()
    h.update(_scorer_identity(scorer or _default_scorer).encode())
    h.update(b"\x00")
    try:
        pairs_digest = dataset_hash([(query, expected) for query, expected in question_set])
    except TypeError as exc:
        raise ValueError(
            f"this question set has no stable identity: {exc}. A scorecard is bound to the held-out "
            "set it was measured on, and a value with no content-determined encoding (typically a "
            "custom object in Query.fingerprint or Query.expected_output) can only be identified by "
            "repr(), whose memory address makes the same question set look like a different one in "
            "the next process."
        ) from exc
    h.update(pairs_digest.encode())
    return h.hexdigest()


def _finite_fraction(name: str, value: float) -> float:
    v = float(value)
    if not math.isfinite(v) or not 0.0 <= v <= 1.0:
        raise ValueError(f"{name} must be a finite fraction in [0, 1], got {value!r}")
    return v


@dataclass
class SystemScorecard:
    """One evaluation of a :class:`~mixle.system.core.System` against a fixed held-out question set.

    Every metric is validated on construction: a card carrying NaN/out-of-range numbers or a negative
    case count is not a weaker measurement, it is not a measurement at all, and
    :func:`detect_regression` silently treated such a card as "no regression" (fail-open on the exact
    gate meant to catch a worsening round). ``question_set_id`` binds the card to the held-out set and
    judge it was measured with (see :func:`question_set_identity`).
    """

    quality: float
    realized_cost: float
    grounded_fraction: float
    n: int
    question_set_id: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.n, bool) or not isinstance(self.n, int) or self.n < 0:
            raise ValueError(f"n must be a nonnegative integer case count, got {self.n!r}")
        self.quality = _finite_fraction("quality", self.quality)
        self.grounded_fraction = _finite_fraction("grounded_fraction", self.grounded_fraction)
        cost = float(self.realized_cost)
        if not math.isfinite(cost) or cost < 0.0:
            raise ValueError(f"realized_cost must be finite and nonnegative, got {self.realized_cost!r}")
        self.realized_cost = cost
        if not isinstance(self.question_set_id, str):
            raise ValueError(f"question_set_id must be a string identity, got {self.question_set_id!r}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the scorecard into primitive fields."""
        return dict(self.__dict__)


def evaluate(
    system: System,
    question_set: Sequence[tuple[Query, str]],
    *,
    scorer: Callable[[str | None, str], bool] | None = None,
) -> SystemScorecard:
    """Evaluate ``system`` over a fixed ``[(query, expected_answer), ...]`` set.

    Every ``expected`` is validated up front: a blank reference cannot discriminate, and a question set
    containing one is rejected rather than scored (it used to make every non-``None`` reply correct and
    report ``quality=1.0``).

    * ``quality`` -- fraction of questions ``scorer`` judges correct. The default judge is
      :func:`_default_scorer`, a deliberately weak lexical baseline (case-insensitive substring match)
      that cannot see negation or paraphrase; read its docstring before treating ``quality`` as
      authoritative, and pass a task-specific ``scorer`` where correctness actually matters.
    * ``grounded_fraction`` -- fraction answered WITHOUT a degraded mode (teacher or captured, not a
      store-only fallback / refusal / failure): the fraction of answers you can actually trust came
      from a real answer path, not a fault-boundary guess.
    * ``realized_cost`` -- total spend units (:meth:`~mixle.spend.Spend.total_units`) across the set.
    * ``question_set_id`` -- the identity of the held-out set and judge this card was measured with,
      so :func:`detect_regression` can refuse to compare cards from different evidence.

    There is deliberately NO ``calibration`` field. One used to exist and was assigned ``quality``
    verbatim: a deterministic system that emits no confidence at all scored a perfect "calibration",
    and :func:`detect_regression` never compared the field anyway, so a drop from 1.0 to 0.0 was not a
    regression. :meth:`~mixle.system.core.System.answer` carries no per-answer confidence to calibrate
    against, so there is nothing here to compute; when it does, add a real calibration statistic to
    THIS function and a matching axis to :func:`detect_regression` in the same change, rather than
    re-adding a copied number that reads as evidence of something never measured.

    This is a genuinely read-only pass: every call goes through :meth:`~mixle.system.core.System.answer`
    with ``read_only=True``, so evaluating a held-out ``question_set`` can never itself teach ``system``
    the held-out answers (no promotion into the harvest a later ``improve()`` could capture, no change to
    ``system.total_spend``). Without that, a held-out set would stop being held-out the moment it was
    first evaluated: a later ``improve()`` could promote its answers into the captured cache, and
    re-evaluating the "held-out" set afterward would be trivially free and perfect. Never call
    ``system.answer`` without ``read_only=True`` from this function.
    """
    for _query, expected in question_set:
        _require_reference(expected)
    set_id = question_set_identity(question_set, scorer=scorer)
    n = len(question_set)
    if n == 0:
        return SystemScorecard(quality=0.0, realized_cost=0.0, grounded_fraction=0.0, n=0, question_set_id=set_id)
    scorer = scorer or _default_scorer
    correct = 0
    grounded = 0
    cost = 0.0
    for query, expected in question_set:
        reply, receipt = system.answer(query, read_only=True)
        if scorer(reply, expected):
            correct += 1
        if receipt.get("status") == "answered" and receipt.get("degraded_mode") is None:
            grounded += 1
        spend = receipt.get("spend") or {}
        cost += float(spend.get("frontier_calls", 0)) + float(spend.get("oracle_calls", 0))
    return SystemScorecard(
        quality=correct / n,
        realized_cost=cost,
        grounded_fraction=grounded / n,
        n=n,
        question_set_id=set_id,
    )


@dataclass
class RegressionReport:
    """Whether ``current`` is worse than ``baseline`` on any tracked axis, and exactly why.

    ``comparable`` is False when the two cards are not evidence about the same thing (different
    held-out set/judge, different case counts, an unidentified card). An incomparable pair is reported
    as ``regressed=True``: a comparison that could not be made is not a passed comparison.
    """

    regressed: bool
    reasons: list[str] = field(default_factory=list)
    comparable: bool = True


def detect_regression(
    baseline: SystemScorecard, current: SystemScorecard, *, tolerance: float = 1e-9
) -> RegressionReport:
    """Compare two scorecards from the SAME held-out set across improve-rounds.

    Never silently accepts a round that answers worse, less groundedly, or for more cost than the round
    before it -- each tracked axis that got worse (beyond ``tolerance``) is named in ``reasons``.

    The "same held-out set" precondition is enforced, not merely documented: both cards must carry the
    same non-empty :func:`question_set_identity` and the same case count. Comparing a card against
    evidence from a different (or empty) question set cannot show that a round did not regress, so
    such a pair comes back ``comparable=False, regressed=True`` rather than a clean bill of health.
    ``tolerance`` must itself be finite and nonnegative -- a NaN tolerance makes every ordered
    comparison below false, which silently turned this gate into an unconditional "no regression".
    """
    if not isinstance(baseline, SystemScorecard) or not isinstance(current, SystemScorecard):
        raise TypeError("detect_regression compares two SystemScorecard instances")
    tol = float(tolerance)
    if not math.isfinite(tol) or tol < 0.0:
        raise ValueError(f"tolerance must be finite and nonnegative, got {tolerance!r}")

    incomparable: list[str] = []
    if not baseline.question_set_id or not current.question_set_id:
        incomparable.append("scorecards are not bound to a held-out question set identity")
    elif baseline.question_set_id != current.question_set_id:
        incomparable.append(
            f"scorecards cover different held-out sets: {baseline.question_set_id[:12]} != "
            f"{current.question_set_id[:12]}"
        )
    if baseline.n != current.n:
        incomparable.append(f"scorecards cover different case counts: {baseline.n} != {current.n}")
    if incomparable:
        return RegressionReport(regressed=True, reasons=incomparable, comparable=False)

    reasons: list[str] = []
    if current.quality < baseline.quality - tol:
        reasons.append(f"quality regressed: {baseline.quality:.3f} -> {current.quality:.3f}")
    if current.grounded_fraction < baseline.grounded_fraction - tol:
        reasons.append(
            f"grounded_fraction regressed: {baseline.grounded_fraction:.3f} -> {current.grounded_fraction:.3f}"
        )
    if current.realized_cost > baseline.realized_cost + tol:
        reasons.append(f"realized_cost increased: {baseline.realized_cost:.3f} -> {current.realized_cost:.3f}")
    return RegressionReport(regressed=bool(reasons), reasons=reasons)
