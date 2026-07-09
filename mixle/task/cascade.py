"""Cascade serving with realized cost tracking and targeted retraining data.

Each request is answered locally when the
:class:`~mixle.task.calibrate.CalibratedTaskModel` is confident and
in-distribution, and escalated to the teacher otherwise. The cascade tracks
actual spend against a :class:`~mixle.task.economics.CostModel`, so
``report()`` returns observed cost and savings relative to a teacher-only route.

Every escalated request marks a case where the local model deferred and the
teacher supplied a targeted label. ``harvested()`` returns those
``(text, label)`` pairs for the next distillation run.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from mixle.task.calibrate import ESCALATE, CalibratedTaskModel
from mixle.task.economics import CostModel, RoutePlan, recommend_route


@dataclass
class CascadeStats:
    """Running tally of how a cascade served traffic -- the basis for realized cost and the harvest."""

    n_requests: int = 0
    n_escalated: int = 0
    escalated_texts: list[Any] = field(default_factory=list)
    escalated_labels: list[Any] = field(default_factory=list)

    @property
    def realized_escalation_rate(self) -> float:
        """Return the observed fraction of requests escalated to the teacher."""
        return self.n_escalated / self.n_requests if self.n_requests else 0.0


class Cascade:
    """Serve ``text -> label`` through a confident local model, escalating to the teacher when needed."""

    def __init__(
        self, model: CalibratedTaskModel, teacher: Callable[..., Any], *, cost: CostModel | None = None
    ) -> None:
        self.model = model
        self.teacher = teacher
        self.cost = cost
        self.stats = CascadeStats()

    def _teacher_label(self, text: Any) -> Any:
        out = self.teacher([text])
        return out[0] if isinstance(out, (list, tuple)) else out

    def __call__(self, text: Any) -> Any:
        """Answer one request, escalating to the teacher only when the local model defers; updates stats."""
        self.stats.n_requests += 1
        local = self.model.decide(text)
        if local is not ESCALATE:
            return local
        label = self._teacher_label(text)
        self.stats.n_escalated += 1
        self.stats.escalated_texts.append(text)
        self.stats.escalated_labels.append(label)
        return label

    def serve(self, texts: Sequence[Any]) -> list[Any]:
        """Serve a batch of requests through the cascade."""
        return [self(t) for t in texts]

    def harvested(self) -> tuple[list[Any], list[Any]]:
        """Return escalated ``(texts, teacher_labels)`` as targeted retraining data."""
        return list(self.stats.escalated_texts), list(self.stats.escalated_labels)

    def realized_cost(self) -> float:
        """Actual spend so far: ``c_local`` per request plus ``c_frontier`` per escalation (requires a CostModel)."""
        if self.cost is None:
            raise RuntimeError("Cascade needs a CostModel to report cost")
        return self.stats.n_requests * self.cost.c_local + self.stats.n_escalated * self.cost.c_frontier

    def report(self) -> dict[str, Any]:
        """Realized economics: requests, escalation rate, spend, and savings vs serving everything on the frontier."""
        out: dict[str, Any] = {
            "n_requests": self.stats.n_requests,
            "n_escalated": self.stats.n_escalated,
            "realized_escalation_rate": self.stats.realized_escalation_rate,
        }
        if self.cost is not None:
            spent = self.realized_cost()
            frontier_only = self.stats.n_requests * self.cost.c_frontier
            out["realized_cost"] = spent
            out["frontier_only_cost"] = frontier_only
            out["savings_vs_frontier"] = frontier_only - spent
        return out

    def plan(self, *, volume: int, n_label: int, max_escalation: float | None = None) -> RoutePlan:
        """Project the lowest-cost route at ``volume`` using the realized escalation rate."""
        if self.cost is None:
            raise RuntimeError("Cascade needs a CostModel to plan a route")
        return recommend_route(
            self.cost,
            volume=volume,
            n_label=n_label,
            p_escalate=self.stats.realized_escalation_rate,
            max_escalation=max_escalation,
        )
