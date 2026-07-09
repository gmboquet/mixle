"""Experimental design for cross-modal reasoning -- which evidence to acquire, in what batch, under a budget.

:meth:`~mixle.reason.store.CrossModalStore.next_evidence` picks the single most informative item (expected
information gain). But real evidence gathering is a *budgeted batch* over *fidelities*: acquire the set of
``(item, fidelity)`` observations that most sharpens the answer per unit cost, without wasting budget on evidence
that is redundant given what you have already chosen. This is cost-aware multi-fidelity experimental design --
the discrete-corpus analogue of ``mixle.doe.multi_fidelity_minimize`` -- and it is *adaptive*: after each pick it
re-scores every candidate against the *updated* belief, so overlapping evidence is naturally avoided (the
near-optimal greedy for submodular information gain).

:func:`select_evidence_batch` plans the acquisition; the returned :class:`AcquisitionPlan` carries the chosen
``(index, fidelity, gain, cost)`` trail, the total nats gained, and the assimilated belief.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from mixle.reason.store import CrossModalStore, _apply, _query_entropy


@dataclass
class AcquisitionPlan:
    """A budgeted evidence-acquisition plan: the chosen items, the nats they bought, and the final belief."""

    items: list[tuple[int, str, float, float]] = field(default_factory=list)  # (index, fidelity, gain_nats, cost)
    total_cost: float = 0.0
    total_gain: float = 0.0  # prior query entropy - final query entropy (nats)
    belief: Any = None

    @property
    def indices(self) -> list[int]:
        """Return selected evidence item indices."""
        return [i for i, _f, _g, _c in self.items]


def select_evidence_batch(
    store: CrossModalStore,
    belief: Any,
    *,
    budget: float,
    query: Any = None,
    fine_cost: float = 1.0,
    coarse_cost: float = 0.2,
    fidelities: Sequence[str] = ("coarse", "fine"),
    candidates: Sequence[int] | None = None,
    max_items: int | None = None,
    min_gain: float = 1e-9,
) -> AcquisitionPlan:
    """Greedily acquire the most-informative-per-cost ``(item, fidelity)`` evidence under a total ``budget``.

    At each step every remaining candidate is scored -- at each allowed fidelity -- by the entropy it would remove
    from the query *given the belief so far*, divided by its cost; the best affordable one is folded in. Adaptive
    re-scoring means a batch never double-counts overlapping evidence. Stops when nothing affordable helps.
    """
    cost_of = {"coarse": float(coarse_cost), "fine": float(fine_cost)}
    build_of: dict[str, Callable[[Any], Any]] = {"coarse": store.coarse, "fine": store.fine}
    fids = [f for f in fidelities if f in build_of]

    pool = list(range(len(store.payloads))) if candidates is None else list(candidates)
    prior_entropy = _query_entropy(belief, query)
    plan = AcquisitionPlan(belief=belief)
    remaining = set(pool)

    while remaining and (max_items is None or len(plan.items) < max_items):
        before = _query_entropy(plan.belief, query)
        best = None  # (gain_per_cost, idx, fidelity, evidence, gain, cost)
        for idx in remaining:
            payload = store.payloads[idx]
            for fidelity in fids:
                cost = cost_of[fidelity]
                if plan.total_cost + cost > budget:
                    continue
                ev = build_of[fidelity](payload)
                gain = before - _query_entropy(_apply(plan.belief, ev), query)
                if gain <= min_gain:
                    continue
                gpc = gain / max(cost, 1e-12)
                if best is None or gpc > best[0]:
                    best = (gpc, idx, fidelity, ev, gain, cost)
        if best is None:
            break
        _, idx, fidelity, ev, gain, cost = best
        plan.belief = _apply(plan.belief, ev)
        plan.total_cost += cost
        plan.items.append((idx, fidelity, float(gain), cost))
        remaining.discard(idx)

    plan.total_gain = float(prior_entropy - _query_entropy(plan.belief, query))
    return plan
