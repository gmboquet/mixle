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
from scipy.stats import beta as beta_distribution

from mixle.system.fault import DegradedResult
from mixle.task.calibrate import ESCALATE, CalibratedTaskModel


@dataclass
class TierStats:
    """Traffic counter and request cost for one router tier."""

    name: str
    cost_per_request: float
    attempted: int = 0
    answered: int = 0
    failed: int = 0


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

    @property
    def n_attempts(self) -> int:
        """Return every paid tier invocation, including escalation and failure."""
        return int(sum(t.attempted for t in self.tiers))


class Router:
    """Route each request to the lowest-cost tier whose calibrated model is confident."""

    def __init__(self, tiers: list[tuple[str, Any, float]]) -> None:
        """``tiers``: ``(name, model_or_callable, cost_per_request)`` in ascending cost order. Every tier except the
        last must expose ``decide(x)`` returning a label or ``ESCALATE``; the
        last tier is the fallback teacher/frontier callable."""
        if not isinstance(tiers, list) or len(tiers) < 2:
            raise ValueError("Router needs at least one calibrated tier plus the final fallback tier")
        normalized: list[tuple[str, Any, float]] = []
        names: set[str] = set()
        prior_cost = float("-inf")
        for i, tier in enumerate(tiers):
            if not isinstance(tier, tuple) or len(tier) != 3:
                raise TypeError(f"tier {i} must be a (name, model, cost_per_request) tuple")
            name, model, cost = tier
            if not isinstance(name, str) or not name:
                raise ValueError(f"tier {i} name must be a non-empty string")
            if name in names:
                raise ValueError(f"duplicate tier name {name!r}")
            names.add(name)
            if (
                isinstance(cost, (bool, np.bool_))
                or not isinstance(cost, (int, float, np.integer, np.floating))
                or not np.isfinite(cost)
                or cost < 0.0
            ):
                raise ValueError(f"tier {name!r} cost must be a finite non-negative number")
            cost = float(cost)
            if cost < prior_cost:
                raise ValueError("router tiers must be ordered by non-decreasing cost")
            prior_cost = cost
            normalized.append((name, model, cost))
        for name, model, _ in normalized[:-1]:
            if not hasattr(model, "decide"):
                raise TypeError(f"tier {name!r} must expose decide(x) (a calibrated task model)")
        if not callable(normalized[-1][1]):
            raise TypeError("the final tier must be a callable answerer (the frontier/teacher)")
        self.tiers = normalized
        self.stats = RouterStats(tiers=[TierStats(name, cost) for name, _, cost in normalized])

    @classmethod
    def from_solutions(
        cls, solutions: list, teacher: Any, *, costs: list[float], names: list[str] | None = None
    ) -> Router:
        """Build from :class:`~mixle.task.solve.Solution` objects ordered by cost plus the teacher callable.

        ``costs`` has one entry per solution plus one for the teacher (per-request)."""
        if len(costs) != len(solutions) + 1:
            raise ValueError("costs needs one entry per solution plus one for the teacher")
        names = names or [f"tier{i}" for i in range(len(solutions))] + ["frontier"]
        if len(names) != len(solutions) + 1:
            raise ValueError("names needs one entry per solution plus one for the teacher")
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
            self.stats.tiers[i].attempted += 1
            try:
                label = model.decide(x)
            except Exception as exc:  # noqa: BLE001 -- route past this tier to the next, whatever it raised
                self.stats.tiers[i].failed += 1
                self.stats.degraded.append(
                    DegradedResult(value=None, degraded=True, mode="model_error", reason=f"{name}: {exc}")
                )
                continue
            if label is not ESCALATE:
                self.stats.tiers[i].answered += 1
                return label
        _, teacher, _ = self.tiers[-1]
        frontier_stats = self.stats.tiers[-1]
        frontier_stats.attempted += 1
        # The frontier/teacher is a BATCHED callable (`texts -> [label]`, e.g. llm_labeler's shape) --
        # calling it with a bare `x` (a single string) would iterate over its characters instead of
        # treating it as one request. Wrap-and-unwrap the same way Cascade._teacher_label already does.
        try:
            out = teacher([x])
            if isinstance(out, (list, tuple)):
                if len(out) != 1:
                    raise ValueError("frontier batch response must contain exactly one answer")
                label = out[0]
            else:
                label = out
        except Exception:
            frontier_stats.failed += 1
            raise
        frontier_stats.answered += 1
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
        realized = float(sum(t.attempted * t.cost_per_request for t in self.stats.tiers))
        per_tier = [
            {
                "tier": t.name,
                "attempted": t.attempted,
                "answered": t.answered,
                "failed": t.failed,
                "share": (t.answered / n) if n else 0.0,
                "attempt_share": (t.attempted / n) if n else 0.0,
                "cost_per_request": t.cost_per_request,
                "realized_cost": t.attempted * t.cost_per_request,
            }
            for t in self.stats.tiers
        ]
        return {
            "requests": n,
            "attempts": self.stats.n_attempts,
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
            f"  {t['tier']}: {t['answered']} answered / {t['attempted']} attempted"
            f" ({t['failed']} failed) @ ${t['cost_per_request']:.5f}"
            for t in r["tiers"]
        ]
        lines.append(f"  harvested {r['harvested_labels']} frontier labels for the next re-solve")
        return "\n".join(lines)


def _sorted_by_cost(tiers: list[tuple[str, Any, float]]) -> list[tuple[str, Any, float]]:
    return sorted(tiers, key=lambda t: t[2])


def route_stack(solutions: list, teacher: Any, *, costs: list[float]) -> Router:
    """Convenience: :meth:`Router.from_solutions` with tiers sorted by ascending cost."""
    if len(costs) != len(solutions) + 1:
        raise ValueError("costs needs one entry per solution plus one for the teacher")
    order = np.argsort(np.asarray(costs[:-1], dtype=np.float64))
    sols = [solutions[i] for i in order]
    cs = [float(costs[i]) for i in order] + [float(costs[-1])]
    return Router.from_solutions(sols, teacher, costs=cs)


# Below this many calibration points, escalation rate has too little resolution
# to distinguish a real drop from a random train/calibration split artifact.
_MIN_CAL_FOR_MEANINGFUL_MEASUREMENT = 10
_MIN_EVAL_FOR_MEANINGFUL_MEASUREMENT = 20


def _binomial_lower(successes: int, total: int, tail_probability: float) -> float:
    """One-sided Clopper-Pearson lower bound."""
    if total <= 0 or not 0 <= successes <= total:
        return 0.0
    if successes == 0:
        return 0.0
    return float(beta_distribution.ppf(tail_probability, successes, total - successes + 1))


@dataclass
class HarvestResolveResult:
    """Receipt from :func:`resolve_from_harvest`.

    ``escalation_before`` is one on the reserved evaluation subset because
    every member is a recorded frontier escalation under the current router.
    ``escalation_after`` and ``local_accuracy`` are measured only after the
    candidate tier is frozen. Their one-sided simultaneous lower bounds gate
    insertion. ``router`` remains ``None`` when either bound is insufficient.
    """

    accepted: bool
    n_harvested: int
    escalation_before: float
    escalation_after: float
    escalation_drop: float
    agreement: float
    router: Router | None = None
    tier_name: str = ""
    n_train: int = 0
    n_calibration: int = 0
    n_evaluation: int = 0
    escalation_drop_lower: float = 0.0
    local_accuracy: float = 0.0
    local_accuracy_lower: float = 0.0
    evaluation_confidence: float = 0.0
    reason: str = ""


def resolve_from_harvest(
    router: Router,
    *,
    cost_per_request: float,
    name: str = "resolved",
    alpha: float = 0.1,
    holdout: float = 0.25,
    evaluation: float = 0.25,
    evaluation_confidence: float = 0.95,
    min_drop: float = 0.05,
    distill_kw: dict[str, Any] | None = None,
    seed: int = 0,
) -> HarvestResolveResult:
    """Train a new router tier from harvested teacher labels.

    The harvest is split *before fitting* into training, calibration, and an
    untouched post-selection evaluation set. The new tier is selected using
    only training/calibration data. It is inserted only when simultaneous
    one-sided exact-binomial lower bounds on evaluation interception and local
    accuracy clear ``min_drop`` and ``1 - alpha`` respectively.
    """
    from mixle.task.distill import agreement
    from mixle.task.solve import _fit_gate, _fit_student

    inputs, labels = router.harvested()
    n_harvested = len(inputs)
    if (
        not isinstance(name, str)
        or not name
        or any(existing_name == name for existing_name, _, _ in router.tiers)
    ):
        raise ValueError("name must be non-empty and unique within the router")
    if (
        isinstance(cost_per_request, (bool, np.bool_))
        or not isinstance(cost_per_request, (int, float, np.integer, np.floating))
        or not np.isfinite(cost_per_request)
        or cost_per_request < 0.0
        or cost_per_request > router.tiers[-1][2]
    ):
        raise ValueError("cost_per_request must be finite, non-negative, and no greater than frontier cost")
    for value, label in (
        (alpha, "alpha"),
        (holdout, "holdout"),
        (evaluation, "evaluation"),
        (evaluation_confidence, "evaluation_confidence"),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not np.isfinite(value)
            or not 0.0 < float(value) < 1.0
        ):
            raise ValueError(f"{label} must be finite and strictly between 0 and 1")
    if (
        isinstance(min_drop, (bool, np.bool_))
        or not isinstance(min_drop, (int, float, np.integer, np.floating))
        or not np.isfinite(min_drop)
        or not 0.0 <= float(min_drop) <= 1.0
    ):
        raise ValueError("min_drop must be finite and in [0, 1]")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an exact integer")

    minimum = 4 + _MIN_CAL_FOR_MEANINGFUL_MEASUREMENT + _MIN_EVAL_FOR_MEANINGFUL_MEASUREMENT
    if n_harvested < minimum:
        return HarvestResolveResult(
            accepted=False,
            n_harvested=n_harvested,
            escalation_before=1.0,
            escalation_after=1.0,
            escalation_drop=0.0,
            agreement=0.0,
            reason=f"need at least {minimum} harvested examples for train/calibration/evaluation separation",
        )

    kind = "text" if isinstance(inputs[0], str) else "record"
    str_labels = [str(y) for y in labels]

    rng = np.random.RandomState(seed)
    order = rng.permutation(n_harvested)
    n_eval = max(_MIN_EVAL_FOR_MEANINGFUL_MEASUREMENT, int(round(n_harvested * evaluation)))
    candidate_count = n_harvested - n_eval
    n_cal = max(_MIN_CAL_FOR_MEANINGFUL_MEASUREMENT, int(round(candidate_count * holdout)))
    eval_idx = order[:n_eval]
    cal_idx = order[n_eval : n_eval + n_cal]
    train_idx = order[n_eval + n_cal :]
    train_in, train_lab = [inputs[i] for i in train_idx], [str_labels[i] for i in train_idx]
    cal_in, cal_lab = [inputs[i] for i in cal_idx], [str_labels[i] for i in cal_idx]
    eval_in, eval_lab = [inputs[i] for i in eval_idx], [str_labels[i] for i in eval_idx]
    if (
        len(train_in) < 4
        or len(cal_in) < _MIN_CAL_FOR_MEANINGFUL_MEASUREMENT
        or len(eval_in) < _MIN_EVAL_FOR_MEANINGFUL_MEASUREMENT
    ):
        return HarvestResolveResult(
            accepted=False,
            n_harvested=n_harvested,
            escalation_before=1.0,
            escalation_after=1.0,
            escalation_drop=0.0,
            agreement=0.0,
            n_train=len(train_in),
            n_calibration=len(cal_in),
            n_evaluation=len(eval_in),
            reason="split geometry leaves insufficient independent evidence",
        )

    kw = dict(distill_kw or {})
    kw.setdefault("seed", seed)
    student = _fit_student(kind, train_in, train_lab, kw)
    gate = _fit_gate(kind, train_in, 0.02, seed)
    cal = CalibratedTaskModel(student, alpha=alpha, density_gate=gate).calibrate(cal_in, cal_lab)

    # The evaluation split has not influenced fitting, gate construction,
    # conformal calibration, or any threshold choice above.
    decisions = [cal.decide(value) for value in eval_in]
    intercepted = np.asarray([decision is not ESCALATE for decision in decisions], dtype=bool)
    n_intercepted = int(intercepted.sum())
    drop = n_intercepted / len(eval_in)
    esc_after = 1.0 - drop
    correct = int(
        sum(
            str(decisions[i]) == eval_lab[i]
            for i in range(len(eval_in))
            if intercepted[i]
        )
    )
    local_accuracy = correct / n_intercepted if n_intercepted else 0.0
    # Two simultaneous release claims: Bonferroni-split the failure budget.
    tail = (1.0 - float(evaluation_confidence)) / 2.0
    drop_lower = _binomial_lower(n_intercepted, len(eval_in), tail)
    accuracy_lower = _binomial_lower(correct, n_intercepted, tail)
    agree = agreement(student, eval_lab, eval_in)

    accepted = drop_lower >= min_drop and accuracy_lower >= 1.0 - alpha
    if not accepted:
        return HarvestResolveResult(
            accepted=False,
            n_harvested=n_harvested,
            escalation_before=1.0,
            escalation_after=float(esc_after),
            escalation_drop=float(drop),
            agreement=float(agree),
            n_train=len(train_in),
            n_calibration=len(cal_in),
            n_evaluation=len(eval_in),
            escalation_drop_lower=float(drop_lower),
            local_accuracy=float(local_accuracy),
            local_accuracy_lower=float(accuracy_lower),
            evaluation_confidence=float(evaluation_confidence),
            reason=(
                "independent evaluation lower bounds did not clear both "
                f"drop>={min_drop} and accuracy>={1.0 - alpha}"
            ),
        )

    new_locals = list(router.tiers[:-1]) + [(name, cal, float(cost_per_request))]
    new_tiers = sorted(new_locals, key=lambda tier: tier[2]) + [router.tiers[-1]]
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
        n_train=len(train_in),
        n_calibration=len(cal_in),
        n_evaluation=len(eval_in),
        escalation_drop_lower=float(drop_lower),
        local_accuracy=float(local_accuracy),
        local_accuracy_lower=float(accuracy_lower),
        evaluation_confidence=float(evaluation_confidence),
        reason="independent evaluation lower bounds cleared release gates",
    )
