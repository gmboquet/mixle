"""Reasoner facade over a substrate, skills, and configured actions.

:func:`investigate` runs the action loop. :class:`Reasoner` packages that loop
behind ``ask(question)`` by wiring retrieval over a substrate, compute actions
for registered skills, and any additional simulator, creator, or delegate
actions supplied by the caller.

The result of ``ask`` is an :class:`~mixle.substrate.act.Investigation` carrying
the selected evidence, action trace, answer, and abstention state.
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
        """Route by a learned acquisition policy instead of the lexical prior. Chainable."""
        self.scorer = scorer
        return self

    def ask(self, question: str, *, verify: bool = False, **overrides: Any) -> Investigation:
        """Answer ``question`` over the configured action space, or abstain. ``overrides`` pass through to
        :func:`investigate` (e.g. ``budget_cost``, ``min_confidence``, ``target_confidence``, ``max_actions``).

        With ``verify=True`` and a substrate configured, the (non-abstained) answer is run back through
        :func:`~mixle.substrate.factuality.check_factuality` and the :class:`FactualityReceipt` is attached
        to ``Investigation.factuality`` -- the reasoner grounds its own answer's claims and reports which it
        can cite. It does not suppress the answer; the receipt is there for the caller to gate on."""
        kw: dict[str, Any] = {
            "budget_cost": self.budget_cost,
            "min_confidence": self.min_confidence,
            "scorer": self.scorer,
            "telemetry": self.telemetry,
        }
        kw.update(overrides)
        inv = investigate(question, self._actions, self.answerer, **kw)
        if verify and inv.answer is not None and self.substrate is not None:
            from mixle.substrate.factuality import check_factuality

            inv.factuality = check_factuality(self.substrate, inv.answer)
        return inv
