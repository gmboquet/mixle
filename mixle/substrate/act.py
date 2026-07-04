"""``investigate()`` -- the reasoner's WIDENED action space: retrieve / compute / simulate / create (S3).

:func:`~mixle.substrate.answer.answer_from_substrate` wired one action -- RETRIEVE. A real reasoner buys
evidence with computation: it can also RUN a model (COMPUTE), run a what-if (SIMULATE), or build a model
/ dataset on the fly (CREATE). :func:`investigate` is that loop. You hand it a question and a set of
:class:`Action` s; it orders them by expected information gain per unit cost, fires them under a cost
budget, accumulates the evidence fragments each returns, and then answers FROM that evidence or ABSTAINS
-- same no-answer-without-provenance rule as the RETRIEVE-only seed, now over a plural action space.

Each action is a thin adapter: a ``run(question) -> list[str]`` plus a ``cost`` and a ``description``
(what it can answer, used to score relevance). The creation verbs (:func:`mixle.inference.skill`,
:func:`mixle.inference.simulate`, :func:`mixle.inference.create`) drop straight in via the
:func:`compute_action` / :func:`simulate_action` / :func:`retrieve_action` builders, so the things the
ecosystem can *make* become the things the reasoner can *do*.

EIG-per-cost is a v1 proxy: an action's score is the lexical overlap of its description with the question
(RETRIEVE gets a base floor because it is always at least weakly informative), divided by its cost. The
seam to a learned acquisition model is :func:`score_action` -- swap it for a calibrated EIG estimate and
the loop is unchanged. The discipline mirrors the rest of the stack: cheap, honest, never fabricates.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


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
    score: float


@dataclass
class Investigation:
    """A cited answer (or abstention) plus the sequence of actions that acquired its evidence."""

    question: str
    answer: str | None
    abstained: bool
    confidence: float
    steps: list[Step] = field(default_factory=list)
    note: str = ""

    @property
    def evidence(self) -> list[str]:
        return [f for s in self.steps for f in s.fragments]

    @property
    def spent(self) -> float:
        return sum(s.cost for s in self.steps)

    def trace(self) -> list[dict[str, Any]]:
        """The actions taken, in order -- the provenance the answer must be checkable against."""
        return [{"action": s.action, "kind": s.kind, "n_fragments": len(s.fragments)} for s in self.steps]

    def as_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "abstained": self.abstained,
            "confidence": round(self.confidence, 4),
            "note": self.note,
            "trace": self.trace(),
        }


def score_action(action: Action, question: str) -> float:
    """EIG-per-cost proxy: lexical relevance of the action to the question, divided by its cost.

    v1 heuristic -- the seam to a learned/calibrated expected-information-gain estimate. RETRIEVE-style
    actions carry a ``base_score`` floor because retrieval is always at least weakly informative."""
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
    max_actions: int | None = None,
    telemetry: Any = None,
) -> Investigation:
    """Answer ``question`` by firing evidence-acquiring ``actions`` under a cost budget, or abstain.

    Actions are ordered by :func:`score_action` (EIG-per-cost) and fired highest-first until the cost
    budget or ``max_actions`` is exhausted, or enough evidence is gathered. The ``answerer``
    (``(question, evidence_text) -> str``) is called ONLY when at least ``min_evidence`` fragments were
    acquired and confidence clears ``min_confidence`` -- otherwise it abstains rather than guess. The
    returned :class:`Investigation` carries the ordered action trace as provenance.
    """
    ranked = sorted(actions, key=lambda a: score_action(a, question), reverse=True)
    if max_actions is not None:
        ranked = ranked[:max_actions]

    steps: list[Step] = []
    spent = 0.0
    for action in ranked:
        if budget_cost is not None and spent + action.cost > budget_cost:
            continue
        sc = score_action(action, question)
        if sc <= 0.0:
            continue
        try:
            fragments = [str(f) for f in action.run(question) if str(f).strip()]
        except Exception:  # noqa: BLE001 - one broken action must not sink the whole investigation
            fragments = []
        spent += action.cost
        steps.append(Step(action=action.name, kind=action.kind, fragments=fragments, cost=action.cost, score=sc))

    evidence = [f for s in steps for f in s.fragments]
    # confidence: best action score that actually returned evidence, damped by how much we gathered
    productive = [s.score for s in steps if s.fragments]
    top = max(productive, default=0.0)
    confidence = round(min(1.0, top) * min(1.0, len(evidence) / max(min_evidence, 1)), 4)

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
    substrate: Any, *, name: str = "retrieve", k: int = 6, scope: str | None = None, cost: float = 1.0
) -> Action:
    """A RETRIEVE action over a :class:`~mixle.substrate.Substrate` (the always-available floor action)."""

    def _run(question: str) -> list[str]:
        from mixle.substrate.retrieve import retrieve

        r = retrieve(substrate, question, k=k, scope=scope)
        return [it.text for it in r.items if it.text]

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
