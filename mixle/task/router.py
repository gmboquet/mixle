"""Calibrated N-tier model routing.

``Router`` generalizes :class:`~mixle.task.cascade.Cascade` from one local tier
plus a teacher to several calibrated tiers. Each local tier answers only when
its conformal set is a confident singleton and, if a density gate is configured,
the input is in distribution. Otherwise the request falls through to the next
tier, ending at a teacher/frontier callable that always answers. Reports carry
realized traffic and cost::

    router = Router.from_solutions([fast, accurate], teacher=frontier, costs=[0.0001, 0.001, 0.03])
    router(x)                     # answered by the lowest-cost confident tier
    router.report()               # per-tier traffic, realized $/req, savings vs all-teacher serving
    router.harvested()            # the frontier's answers on hard inputs = training data for the tiers

Every request the teacher answers is harvested as targeted training data for
the lower-cost tiers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mixle.fault import DegradedResult
from mixle.task.calibrate import ESCALATE, CalibratedTaskModel


@dataclass
class TierStats:
    """Traffic counter and request cost for one router tier."""

    name: str
    cost_per_request: float
    answered: int = 0


@dataclass
class RouterStats:
    """Mutable accounting for routed requests, harvested labels, and degraded tier calls."""

    tiers: list[TierStats] = field(default_factory=list)
    harvested_inputs: list[Any] = field(default_factory=list)
    harvested_labels: list[Any] = field(default_factory=list)
    degraded: list[DegradedResult] = field(default_factory=list)  # model_error events, in order

    @property
    def n_requests(self) -> int:
        """Return the total number of requests answered across all tiers."""
        return int(sum(t.answered for t in self.tiers))


class Router:
    """Route each request to the lowest-cost tier whose calibrated model is confident."""

    def __init__(self, tiers: list[tuple[str, Any, float]]) -> None:
        """``tiers``: ``(name, model_or_callable, cost_per_request)`` in ascending cost order. Every tier except the
        last must expose ``decide(x)`` returning a label or ``ESCALATE``; the
        last tier is the fallback teacher/frontier callable."""
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
        """Build from :class:`~mixle.task.solve.Solution` objects ordered by cost plus the teacher callable.

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
        """Answer with the lowest-cost confident tier; the final tier's answers are harvested as labels.

        If a tier's ``decide(x)`` raises, the router records a ``model_error``
        in ``stats.degraded`` and gives the next tier a chance to answer.
        """
        for i, (name, model, _) in enumerate(self.tiers[:-1]):
            try:
                label = model.decide(x)
            except Exception as exc:  # noqa: BLE001 -- route past this tier to the next, whatever it raised
                self.stats.degraded.append(
                    DegradedResult(value=None, degraded=True, mode="model_error", reason=f"{name}: {exc}")
                )
                continue
            if label is not ESCALATE:
                self.stats.tiers[i].answered += 1
                return label
        _, teacher, _ = self.tiers[-1]
        # The frontier/teacher is a BATCHED callable (`texts -> [label]`, e.g. llm_labeler's shape) --
        # calling it with a bare `x` (a single string) would iterate over its characters instead of
        # treating it as one request. Wrap-and-unwrap the same way Cascade._teacher_label already does.
        out = teacher([x])
        label = out[0] if isinstance(out, (list, tuple)) else out
        self.stats.tiers[-1].answered += 1
        self.stats.harvested_inputs.append(x)
        self.stats.harvested_labels.append(label)
        return label

    def serve(self, xs: Any) -> list[Any]:
        """Route a batch of requests and return the tier-selected answers."""
        return [self(x) for x in xs]

    def harvested(self) -> tuple[list[Any], list[Any]]:
        """Return teacher-answered ``(inputs, labels)`` for retraining lower-cost tiers."""
        return list(self.stats.harvested_inputs), list(self.stats.harvested_labels)

    def report(self) -> dict[str, Any]:
        """Return per-tier traffic and realized economics."""
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
        """Render a compact human-readable traffic and cost summary."""
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
    """Convenience: :meth:`Router.from_solutions` with tiers sorted by ascending cost."""
    order = np.argsort(np.asarray(costs[:-1], dtype=np.float64))
    sols = [solutions[i] for i in order]
    cs = [float(costs[i]) for i in order] + [float(costs[-1])]
    return Router.from_solutions(sols, teacher, costs=cs)


# Below this many calibration points, escalation rate has too little resolution
# to distinguish a real drop from a random train/calibration split artifact.
_MIN_CAL_FOR_MEANINGFUL_MEASUREMENT = 10


@dataclass
class HarvestResolveResult:
    """Receipt from :func:`resolve_from_harvest`.

    ``escalation_before`` is exactly 1.0: every harvested input, by definition, escalated all the way
    to the teacher under the current router. ``escalation_after`` is the new tier's own calibrated
    escalation rate on a held-out split of that same harvested set; ``escalation_drop`` is the
    difference. ``router`` is the new stack with the tier inserted (``None`` when nothing was
    accepted because there is too little harvested data to fit/calibrate or the new tier does
    not escalate measurably less often than always-escalate, in which case it buys nothing and is
    rejected).
    """

    accepted: bool
    n_harvested: int
    escalation_before: float
    escalation_after: float
    escalation_drop: float
    agreement: float
    router: Router | None = None
    tier_name: str = ""


def resolve_from_harvest(
    router: Router,
    *,
    cost_per_request: float,
    name: str = "resolved",
    alpha: float = 0.1,
    holdout: float = 0.25,
    min_drop: float = 0.05,
    distill_kw: dict[str, Any] | None = None,
    seed: int = 0,
) -> HarvestResolveResult:
    """Train a new router tier from harvested teacher labels.

    Every harvested input escalated through the existing tiers, so the baseline
    escalation rate on that set is 1.0. A new tier is fit and calibrated on a
    held-out split of the same harvested set without re-calling the teacher. It
    is inserted only if its calibrated escalation rate drops by at least
    ``min_drop`` below 1.0 on that split.
    """
    from mixle.task.distill import agreement
    from mixle.task.solve import _fit_gate, _fit_student

    inputs, labels = router.harvested()
    n_harvested = len(inputs)
    if n_harvested < 4 + _MIN_CAL_FOR_MEANINGFUL_MEASUREMENT:
        return HarvestResolveResult(
            accepted=False,
            n_harvested=n_harvested,
            escalation_before=1.0,
            escalation_after=1.0,
            escalation_drop=0.0,
            agreement=0.0,
        )

    kind = "text" if isinstance(inputs[0], str) else "record"
    str_labels = [str(y) for y in labels]

    rng = np.random.RandomState(seed)
    order = rng.permutation(n_harvested)
    n_cal = max(_MIN_CAL_FOR_MEANINGFUL_MEASUREMENT, int(round(n_harvested * holdout)))
    cal_idx, train_idx = order[:n_cal], order[n_cal:]
    train_in, train_lab = [inputs[i] for i in train_idx], [str_labels[i] for i in train_idx]
    cal_in, cal_lab = [inputs[i] for i in cal_idx], [str_labels[i] for i in cal_idx]
    if len(train_in) < 4 or len(cal_in) < _MIN_CAL_FOR_MEANINGFUL_MEASUREMENT:
        return HarvestResolveResult(
            accepted=False,
            n_harvested=n_harvested,
            escalation_before=1.0,
            escalation_after=1.0,
            escalation_drop=0.0,
            agreement=0.0,
        )

    kw = dict(distill_kw or {})
    kw.setdefault("seed", seed)
    student = _fit_student(kind, train_in, train_lab, kw)
    gate = _fit_gate(kind, train_in, 0.02, seed)
    cal = CalibratedTaskModel(student, alpha=alpha, density_gate=gate).calibrate(cal_in, cal_lab)

    agree = agreement(student, cal_lab, cal_in)
    esc_after = cal.escalation_rate(cal_in)
    drop = 1.0 - esc_after

    if drop < min_drop:
        return HarvestResolveResult(
            accepted=False,
            n_harvested=n_harvested,
            escalation_before=1.0,
            escalation_after=float(esc_after),
            escalation_drop=float(drop),
            agreement=float(agree),
        )

    new_tiers = list(router.tiers[:-1]) + [(name, cal, float(cost_per_request)), router.tiers[-1]]
    # the input router's harvest is now consumed into the new tier -- clear it (mirrors
    # Solution.improve()'s escalated_texts/labels.clear() after promoting) so a caller that keeps
    # using `router` for observability, or calls resolve_from_harvest again after more traffic, does
    # not double-count these same escalations as still-unresolved.
    router.stats.harvested_inputs.clear()
    router.stats.harvested_labels.clear()
    return HarvestResolveResult(
        accepted=True,
        n_harvested=n_harvested,
        escalation_before=1.0,
        escalation_after=float(esc_after),
        escalation_drop=float(drop),
        agreement=float(agree),
        router=Router(new_tiers),
        tier_name=name,
    )
