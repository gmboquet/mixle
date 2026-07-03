"""``Router`` -- calibrated N-tier model routing: tiny models first, the frontier only when necessary.

The multi-tier generalization of :class:`~mixle.task.cascade.Cascade`, and the honest version of "LLM
routing": each tier is a calibrated task model that answers **only** when its conformal set is a
confident singleton and (if gated) the input is in-distribution — otherwise the request falls through
to the next tier, ending at the frontier/teacher, which always answers. Routing decisions carry
coverage guarantees, not learned vibes; the report carries realized cost, not projections::

    router = Router.from_solutions([tiny, small], teacher=frontier, costs=[0.0001, 0.001, 0.03])
    router(x)                     # answered by the cheapest tier that is SURE
    router.report()               # per-tier traffic, realized $/req, savings vs frontier-only
    router.harvested()            # the frontier's answers on hard inputs = training data for the tiers

Every request the frontier answers is a teacher-labeled example exactly where the local tiers were
unsure — feed ``harvested()`` back through ``solve(prelabeled=...)`` and the routing gets cheaper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mixle.task.calibrate import ESCALATE


@dataclass
class TierStats:
    name: str
    cost_per_request: float
    answered: int = 0


@dataclass
class RouterStats:
    tiers: list[TierStats] = field(default_factory=list)
    harvested_inputs: list[Any] = field(default_factory=list)
    harvested_labels: list[Any] = field(default_factory=list)

    @property
    def n_requests(self) -> int:
        return int(sum(t.answered for t in self.tiers))


class Router:
    """Route each request to the cheapest tier whose calibrated model is confident; the last tier always answers."""

    def __init__(self, tiers: list[tuple[str, Any, float]]) -> None:
        """``tiers``: ``(name, model_or_callable, cost_per_request)`` cheapest-first. Every tier except the
        last must expose ``decide(x)`` (a :class:`CalibratedTaskModel` / loaded Solution model) returning a
        label or ``ESCALATE``; the last tier is the fallback answerer (any callable — the frontier/teacher)."""
        if len(tiers) < 2:
            raise ValueError("Router needs at least one calibrated tier plus the final fallback tier")
        for name, model, _ in tiers[:-1]:
            if not hasattr(model, "decide"):
                raise TypeError(f"tier {name!r} must expose decide(x) (a calibrated task model)")
        if not callable(tiers[-1][1]):
            raise TypeError("the final tier must be a callable answerer (the frontier/teacher)")
        self.tiers = list(tiers)
        self.stats = RouterStats(tiers=[TierStats(name, float(c)) for name, _, c in tiers])

    @classmethod
    def from_solutions(
        cls, solutions: list, teacher: Any, *, costs: list[float], names: list[str] | None = None
    ) -> Router:
        """Build from :class:`~mixle.task.solve.Solution` objects (cheapest-first) + the frontier callable.

        ``costs`` has one entry per solution plus one for the teacher (per-request)."""
        if len(costs) != len(solutions) + 1:
            raise ValueError("costs needs one entry per solution plus one for the teacher")
        names = names or [f"tier{i}" for i in range(len(solutions))] + ["frontier"]
        tiers: list[tuple[str, Any, float]] = [
            (names[i], sol.cascade.model, float(costs[i])) for i, sol in enumerate(solutions)
        ]
        tiers.append((names[-1], teacher, float(costs[-1])))
        return cls(tiers)

    def __call__(self, x: Any) -> Any:
        """Answer with the cheapest confident tier; the final tier's answers are harvested as labels."""
        for i, (name, model, _) in enumerate(self.tiers[:-1]):
            label = model.decide(x)
            if label is not ESCALATE:
                self.stats.tiers[i].answered += 1
                return label
        _, teacher, _ = self.tiers[-1]
        label = teacher(x)
        self.stats.tiers[-1].answered += 1
        self.stats.harvested_inputs.append(x)
        self.stats.harvested_labels.append(label)
        return label

    def serve(self, xs: Any) -> list[Any]:
        return [self(x) for x in xs]

    def harvested(self) -> tuple[list[Any], list[Any]]:
        """The frontier-answered ``(inputs, labels)`` — targeted training data for the cheaper tiers."""
        return list(self.stats.harvested_inputs), list(self.stats.harvested_labels)

    def report(self) -> dict[str, Any]:
        """Per-tier traffic and REALIZED economics vs sending everything to the final tier."""
        n = self.stats.n_requests
        frontier_cost = self.tiers[-1][2]
        realized = float(sum(t.answered * t.cost_per_request for t in self.stats.tiers))
        per_tier = [
            {
                "tier": t.name,
                "answered": t.answered,
                "share": (t.answered / n) if n else 0.0,
                "cost_per_request": t.cost_per_request,
            }
            for t in self.stats.tiers
        ]
        return {
            "requests": n,
            "tiers": per_tier,
            "realized_cost": realized,
            "frontier_only_cost": float(n * frontier_cost),
            "savings": float(n * frontier_cost - realized),
            "cost_per_request": (realized / n) if n else 0.0,
            "harvested_labels": len(self.stats.harvested_labels),
        }

    def summary(self) -> str:
        r = self.report()
        lines = [
            f"routed {r['requests']} requests @ ${r['cost_per_request']:.5f}/req "
            f"(frontier-only ${self.tiers[-1][2]:.5f}/req; saved ${r['savings']:.2f})"
        ]
        lines += [
            f"  {t['tier']}: {t['answered']} ({t['share']:.0%}) @ ${t['cost_per_request']:.5f}" for t in r["tiers"]
        ]
        lines.append(f"  harvested {r['harvested_labels']} frontier labels for the next re-solve")
        return "\n".join(lines)


def _sorted_by_cost(tiers: list[tuple[str, Any, float]]) -> list[tuple[str, Any, float]]:
    return sorted(tiers, key=lambda t: t[2])


def route_stack(solutions: list, teacher: Any, *, costs: list[float]) -> Router:
    """Convenience: :meth:`Router.from_solutions` with tiers sorted cheapest-first by cost."""
    order = np.argsort(np.asarray(costs[:-1], dtype=np.float64))
    sols = [solutions[i] for i in order]
    cs = [float(costs[i]) for i in order] + [float(costs[-1])]
    return Router.from_solutions(sols, teacher, costs=cs)
