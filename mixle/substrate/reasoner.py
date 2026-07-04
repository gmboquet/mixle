"""``Reasoner`` -- a deployable shell that bundles a substrate, skills, and actions behind ``.ask`` (R).

:func:`investigate` is the loop; :class:`Reasoner` is the *product* around it -- the tiny-agent harness a
consumer actually deploys. You hand it a knowledge store and a skill registry, it wires the standard
action space (RETRIEVE over the store, one COMPUTE per registered skill, plus any simulators / creators /
delegates you attach), and then ``ask(question)`` runs the whole evidence-buying loop and returns a cited
:class:`~mixle.substrate.act.Investigation`. Attach a learned acquisition policy and the same object routes
by learned expected-gain instead of the lexical prior -- the never-worse upgrade, transparent to the caller.

This is the shell the workplan's Harness product slots into: a fixed answerer (a 99%-local student), a
whitelisted set of actions, an escalation cost budget, and one method. Everything it does is provenanced
(every answer carries its action trace) and honest (it abstains rather than guess).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from mixle.substrate.act import (
    Action,
    Investigation,
    compute_action,
    investigate,
    retrieve_action,
)
from mixle.substrate.core import Substrate


class Reasoner:
    """A configured reasoner: a knowledge store + skills + actions, asked questions through one method."""

    def __init__(
        self,
        answerer: Callable[[str, str], str],
        *,
        substrate: Substrate | None = None,
        skills: Any = None,
        actions: list[Action] | None = None,
        budget_cost: float | None = None,
        min_confidence: float = 0.15,
        retrieve_min_score: float = 0.0,
        scorer: Callable[[Action, str], float] | None = None,
        telemetry: Any = None,
    ) -> None:
        self.answerer = answerer
        self.substrate = substrate
        self.budget_cost = budget_cost
        self.min_confidence = min_confidence
        self.scorer = scorer
        self.telemetry = telemetry
        self._actions: list[Action] = []
        if substrate is not None:
            self._actions.append(retrieve_action(substrate, min_score=retrieve_min_score))
        if skills is not None:
            for sk in skills.all() if hasattr(skills, "all") else skills:
                self._actions.append(compute_action(sk))
        if actions:
            self._actions.extend(actions)

    @property
    def actions(self) -> list[Action]:
        """The current action space (retrieve + one compute per skill + attached simulators/creators)."""
        return list(self._actions)

    def add_action(self, action: Action) -> Reasoner:
        """Attach an extra action (a simulator, creator, or delegate); returns self for chaining."""
        self._actions.append(action)
        return self

    def use_policy(self, scorer: Callable[[Action, str], float]) -> Reasoner:
        """Route by a learned acquisition policy instead of the lexical prior (never-worse). Chainable."""
        self.scorer = scorer
        return self

    def ask(self, question: str, **overrides: Any) -> Investigation:
        """Answer ``question`` over the configured action space, or abstain. ``overrides`` pass through to
        :func:`investigate` (e.g. ``budget_cost``, ``min_confidence``, ``target_confidence``, ``max_actions``)."""
        kw: dict[str, Any] = {
            "budget_cost": self.budget_cost,
            "min_confidence": self.min_confidence,
            "scorer": self.scorer,
            "telemetry": self.telemetry,
        }
        kw.update(overrides)
        return investigate(question, self._actions, self.answerer, **kw)
