"""Action-based investigation over retrieve, compute, simulate, and create steps.

:func:`investigate` accepts a question and a set of :class:`Action` objects. It
scores actions by relevance per unit cost, executes them under an optional
budget, accumulates evidence fragments, and returns an
:class:`Investigation` containing either an answer or an abstention.

Each action is a small adapter around ``run(question) -> list[str]`` plus cost
and description metadata. Helper builders adapt substrate retrieval, registered
skills, simulators, and creators into the same action protocol.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

_WORD = re.compile(r"[a-z0-9]+")
# a minimal stoplist so common function words don't manufacture spurious overlap (e.g. matching an
# action to a question purely on "the"/"is"); relevance should reflect content words, not glue.
_STOP = frozenset(
    "a an and are as at be by do does for from how in is of on or the to was what when where which who "
    "will with you your this that".split()
)


def _tokens(text: str) -> set[str]:
    return {t for t in _WORD.findall(text.lower()) if t not in _STOP}


@dataclass
class Action:
    """One evidence-acquiring move: run it on a question, get back evidence fragments, at a cost."""

    name: str
    kind: str  # retrieve | compute | simulate | create
    run: Callable[[str], list[str]]
    cost: float = 1.0
    description: str = ""
    base_score: float = 0.0  # a floor added before division (RETRIEVE is always weakly informative)


@dataclass
class Step:
    """A fired action and what it yielded -- the audit trail behind an investigated answer."""

    action: str
    kind: str
    fragments: list[str]
    cost: float
    score: float  # ordering priority: relevance / cost (why this action was tried when)
    relevance: float = 0.0  # on-topic-ness independent of cost (what confidence is earned from)


@dataclass
class Investigation:
    """A cited answer (or abstention) plus the sequence of actions that acquired its evidence."""

    question: str
    answer: str | None
    abstained: bool
    confidence: float
    steps: list[Step] = field(default_factory=list)
    note: str = ""
    factuality: Any = None  # optional FactualityReceipt when the answer was verified against a substrate
    proposal: Any = None  # optional ResearchProposal attached to an abstention ("here is how to find out")

    @property
    def evidence(self) -> list[str]:
        """Flatten all evidence fragments collected by the investigation."""
        return [f for s in self.steps for f in s.fragments]

    @property
    def spent(self) -> float:
        """Total action cost spent by the investigation."""
        return sum(s.cost for s in self.steps)

    def trace(self) -> list[dict[str, Any]]:
        """The actions taken, in order -- the provenance the answer must be checkable against."""
        return [{"action": s.action, "kind": s.kind, "n_fragments": len(s.fragments)} for s in self.steps]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable investigation summary."""
        return {
            "question": self.question,
            "answer": self.answer,
            "abstained": self.abstained,
            "confidence": round(self.confidence, 4),
            "note": self.note,
            "trace": self.trace(),
        }


def _n_fragments(steps: list[Step]) -> int:
    return sum(len(s.fragments) for s in steps)


def _confidence(steps: list[Step], min_evidence: int) -> float:
    """Best productive action's RELEVANCE, damped by how much evidence was gathered vs required.

    Confidence keys on relevance, not the cost-discounted ordering score -- an expensive but on-topic
    action (a priced delegate that actually answers) earns full confidence, it is merely tried last."""
    productive = [s.relevance for s in steps if s.fragments]
    top = max(productive, default=0.0)
    n = _n_fragments(steps)
    return round(min(1.0, top) * min(1.0, n / max(min_evidence, 1)), 4)


def relevance_of(action: Action, question: str) -> float:
    """How on-topic an action is for a question (lexical overlap + its base floor), ignoring cost."""
    q = _tokens(question)
    overlap = len(q & _tokens(action.description)) / len(q) if q else 0.0
    return action.base_score + overlap


def score_action(action: Action, question: str) -> float:
    """EIG-per-cost proxy: lexical relevance of the action to the question, divided by its cost.

    This heuristic can be replaced with a learned or calibrated
    expected-information-gain estimate. Retrieval-style actions carry a
    ``base_score`` floor because retrieval is always at least weakly
    informative.
    """
    q = _tokens(question)
    overlap = len(q & _tokens(action.description)) / len(q) if q else 0.0
    return (action.base_score + overlap) / max(action.cost, 1e-9)


def investigate(
    question: str,
    actions: list[Action],
    answerer: Callable[[str, str], str],
    *,
    budget_cost: float | None = None,
    min_evidence: int = 1,
    min_confidence: float = 0.15,
    target_confidence: float | None = None,
    max_actions: int | None = None,
    scorer: Callable[[Action, str], float] | None = None,
    telemetry: Any = None,
) -> Investigation:
    """Answer ``question`` by firing evidence-acquiring ``actions`` under a cost budget, or abstain.

    Actions are ordered by ``scorer`` (default :func:`score_action`, EIG-per-cost) and fired
    highest-first. The loop stops early once it holds at least ``min_evidence``
    fragments and confidence clears ``target_confidence`` (default:
    ``min_confidence``). It also stops at the cost budget or ``max_actions``.
    ``scorer`` can be replaced with a learned acquisition policy such as
    :func:`mixle.inference.learn_action_policy`. The ``answerer`` is called only
    when the evidence clears the bar. The returned :class:`Investigation`
    carries the ordered action trace as provenance, and each fired action can
    emit telemetry for later policy learning.
    """
    score = scorer or score_action
    stop_at = target_confidence if target_confidence is not None else min_confidence
    ranked = sorted(actions, key=lambda a: score(a, question), reverse=True)
    if max_actions is not None:
        ranked = ranked[:max_actions]

    steps: list[Step] = []
    spent = 0.0
    for action in ranked:
        if budget_cost is not None and spent + action.cost > budget_cost:
            continue
        sc = score(action, question)
        if sc <= 0.0:
            continue
        try:
            fragments = [str(f) for f in action.run(question) if str(f).strip()]
        except Exception:  # noqa: BLE001 - one failed action must not stop the whole investigation
            fragments = []
        spent += action.cost
        steps.append(
            Step(
                action=action.name,
                kind=action.kind,
                fragments=fragments,
                cost=action.cost,
                score=sc,
                relevance=relevance_of(action, question),
            )
        )
        _emit_route(telemetry, question, action, fragments)
        # early stop: enough evidence above the bar means the remaining (costlier) actions are wasted spend
        if _n_fragments(steps) >= min_evidence and _confidence(steps, min_evidence) >= stop_at:
            break

    evidence = [f for s in steps for f in s.fragments]
    confidence = _confidence(steps, min_evidence)

    if len(evidence) < min_evidence or confidence < min_confidence:
        inv = Investigation(
            question=question,
            answer=None,
            abstained=True,
            confidence=confidence,
            steps=steps,
            note=(
                f"abstained: {len(evidence)} fragment(s) at confidence {confidence:.2f} from "
                f"{len(steps)} action(s) -- escalate rather than guess"
            ),
        )
        _emit(telemetry, inv)
        return inv

    text = answerer(question, "\n".join(evidence))
    inv = Investigation(
        question=question,
        answer=text,
        abstained=False,
        confidence=confidence,
        steps=steps,
        note=f"answered from {len(evidence)} fragment(s) across {len(steps)} action(s)",
    )
    _emit(telemetry, inv)
    return inv


# -- action builders: turn the ecosystem's verbs into reasoner actions --------------------------------


def retrieve_action(
    substrate: Any,
    *,
    name: str = "retrieve",
    k: int = 6,
    scope: str | None = None,
    cost: float = 1.0,
    min_score: float = 0.0,
) -> Action:
    """A retrieve action over a :class:`~mixle.substrate.Substrate` (the always-available floor action).

    ``min_score`` filters out weak matches: a small embedder returns a result for every query, so a
    positive floor keeps genuinely-irrelevant items from becoming false evidence. It defaults to 0.0
    (keep everything) but a small positive value makes retrieval conservative on a noisy index."""

    def _run(question: str) -> list[str]:
        from mixle.substrate.retrieve import retrieve

        r = retrieve(substrate, question, k=k, scope=scope)
        return [it.text for it, sc in zip(r.items, r.scores) if it.text and sc >= min_score]

    return Action(name=name, kind="retrieve", run=_run, cost=cost, description="", base_score=0.35)


def compute_action(skill: Any, *, name: str | None = None, cost: float = 1.0, description: str | None = None) -> Action:
    """A COMPUTE action that runs a :class:`~mixle.inference.skill.Skill` and reports its result."""
    nm = name or getattr(skill, "name", "compute")
    desc = description if description is not None else getattr(skill, "description", "")

    def _run(question: str) -> list[str]:
        try:
            result = skill(question)
        except TypeError:
            result = skill()
        return [f"{nm} => {result}"]

    return Action(name=nm, kind="compute", run=_run, cost=cost, description=desc)


def simulate_action(
    simulator: Any,
    field_index: int,
    scenario: str,
    *,
    name: str | None = None,
    cost: float = 2.0,
    description: str = "",
) -> Action:
    """A SIMULATE action that runs a what-if scenario and reports the simulated outcome mean."""
    nm = name or f"simulate:{scenario}"

    def _run(question: str) -> list[str]:
        mean = simulator.outcome_mean(field_index, scenario=scenario)
        return [f"under scenario '{scenario}', mean of field {field_index} = {mean:.3f}"]

    return Action(name=nm, kind="simulate", run=_run, cost=cost, description=description)


def create_action(
    build: Callable[[str], Any],
    *,
    name: str = "create",
    cost: float = 4.0,
    description: str = "",
    report: Callable[[Any], str] | None = None,
) -> Action:
    """A CREATE action that builds a model / dataset on demand and reports what it made.

    ``build(question) -> artifact`` fits/synthesizes something (e.g. via :func:`mixle.inference.create`
    or :func:`mixle.inference.synthesize`); ``report`` renders the artifact to an evidence fragment
    (default: a certificate/guarantee summary when present, else ``repr``). Creation is the most
    expensive action, so it defaults to ``cost=4`` -- the reasoner reaches for it only when cheaper
    retrieve/compute/simulate actions cannot answer."""

    def _default_report(artifact: Any) -> str:
        guarantee = getattr(artifact, "guarantee", None)
        if guarantee is not None:
            why = artifact.why() if hasattr(artifact, "why") else ""
            return f"{name} => built a model (guarantee {guarantee}). {why}".strip()
        n = len(artifact) if hasattr(artifact, "__len__") else "?"
        return f"{name} => built an artifact ({n} rows/items)"

    render = report or _default_report

    def _run(question: str) -> list[str]:
        return [render(build(question))]

    return Action(name=name, kind="create", run=_run, cost=cost, description=description)


def delegate_action(
    delegate: Callable[[str], Any],
    *,
    name: str = "delegate",
    cost: float = 8.0,
    description: str = "",
    priced: bool = True,
) -> Action:
    """A DELEGATE action that hands the question to an external worker (pool job / remote tool / agent).

    ``delegate(question) -> answer`` is any priced external capability. This is the reasoner's most
    expensive move (default ``cost=8``) and the escalation of last resort: it fires only when nothing
    local clears the bar, honoring the 99%-local topology. ``priced=True`` records that the call incurs
    real spend (the pool/interop layers own the actual budget-reject + confirm rails)."""

    def _run(question: str) -> list[str]:
        result = delegate(question)
        tag = "delegate (priced)" if priced else "delegate"
        return [f"{tag} => {result}"]

    return Action(name=name, kind="delegate", run=_run, cost=cost, description=description)


def action_features(action: Action, question: str) -> dict[str, Any]:
    """The features a learned acquisition policy keys on: the action's kind, cost, and query overlap."""
    q = _tokens(question)
    overlap = len(q & _tokens(action.description)) / len(q) if q else 0.0
    return {"kind": action.kind, "cost": float(action.cost), "overlap": round(overlap, 4)}


def _emit_route(telemetry: Any, question: str, action: Action, fragments: list[str]) -> None:
    """Emit one ``route`` row per fired action: which action, on what query features, and what it yielded.

    This is the training data for the learned acquisition policy (J3): the ``value`` outcome is whether
    the action produced usable evidence, so a policy can learn which actions pay off where."""
    try:
        from mixle.telemetry import record

        rec = telemetry.record if telemetry is not None else record
        rec(
            "route",
            features=action_features(action, question),
            choice=action.kind,
            outcome={"value": 1.0 if fragments else 0.0, "n_fragments": len(fragments)},
        )
    except Exception:  # noqa: BLE001 - telemetry must never break the reasoner
        pass


def _emit(telemetry: Any, inv: Investigation) -> None:
    try:
        from mixle.telemetry import record

        rec = telemetry.record if telemetry is not None else record
        rec(
            "reason",
            features={
                "action": "investigate",
                "n_actions": len(inv.steps),
                "kinds": sorted({s.kind for s in inv.steps}),
            },
            choice="abstain" if inv.abstained else "answer",
            outcome={"confidence": inv.confidence, "spent": inv.spent},
        )
    except Exception:  # noqa: BLE001 - telemetry must never break the reasoner
        pass
