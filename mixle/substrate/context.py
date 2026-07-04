"""ContextPacket + assembly-on-route -- a budgeted, provenanced view of the substrate for a target (O2).

When the orchestrator routes work to a target -- a tiny student, an LLM, a pool job, another agent, a
human -- it builds the CONTEXT that target needs at the size it can afford. A :class:`ContextPacket`
is that view: the task, the substrate items selected for it (in relevance order), the budget, and the
provenance of every included piece. Targets declare a :class:`ContextBudget` the way devices declare a
DeviceSpec: how many characters/items they can take and in what shape (passages for an LLM, a brief
for a human, features for a student).

Assembly is retrieval (:meth:`Substrate.search`) + greedy budgeted selection: the most relevant items
are packed until the budget is hit, so the packet is always the best-affordable view rather than a
blind top-k. Every assembly emits a ``context`` telemetry event (what budget, how much was used, how
many items) -- the history the learned context-assembly policy (workstream J) will train on. Nothing
is included without provenance: the reasoner can cite where every piece came from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mixle.substrate.core import Substrate, SubstrateItem


@dataclass
class ContextBudget:
    """What a target can take -- the DeviceSpec of context. ``shape`` hints the rendering style."""

    max_chars: int = 2000
    max_items: int = 20
    shape: str = "passages"  # 'passages' (LLM) | 'brief' (human) | 'features' (student)


@dataclass
class ContextPacket:
    """A budgeted, provenanced view of the substrate assembled for one target + task."""

    task: str
    items: list[SubstrateItem] = field(default_factory=list)  # selected, in descending relevance
    scores: list[float] = field(default_factory=list)
    budget: ContextBudget = field(default_factory=ContextBudget)
    used_chars: int = 0
    n_candidates: int = 0  # how many the retriever surfaced before budgeting

    def render(self, *, header: bool = True) -> str:
        """The assembled context string the target consumes (respecting the budget shape)."""
        head = f"# Context for: {self.task}\n" if header else ""
        if self.budget.shape == "brief":
            body = "\n".join(f"- {_one_line(i.text)}" for i in self.items)
        else:  # passages / features: full item surfaces, provenance-tagged
            body = "\n\n".join(f"[{i.kind}:{i.id}] {i.text}" for i in self.items)
        return head + body

    def provenance(self) -> list[dict[str, Any]]:
        """Where every included piece came from -- ids, kinds, sources, relevance scores."""
        return [
            {
                "id": i.id,
                "kind": i.kind,
                "source": i.provenance.get("source") or i.provenance.get("path"),
                "score": round(float(s), 4),
            }
            for i, s in zip(self.items, self.scores)
        ]

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "n_items": len(self.items),
            "n_candidates": self.n_candidates,
            "used_chars": self.used_chars,
            "budget_chars": self.budget.max_chars,
            "shape": self.budget.shape,
            "provenance": self.provenance(),
        }

    def __len__(self) -> int:
        return len(self.items)


def _one_line(text: str, limit: int = 160) -> str:
    t = " ".join(str(text).split())
    return t if len(t) <= limit else t[: limit - 1] + "…"


def _render_len(item: SubstrateItem, shape: str) -> int:
    return len(_one_line(item.text)) if shape == "brief" else len(item.text) + len(item.kind) + len(item.id) + 6


def assemble_context(
    substrate: Substrate,
    task: str,
    *,
    budget: ContextBudget | None = None,
    kind: str | None = None,
    scope: str | None = None,
    telemetry: Any = None,
) -> ContextPacket:
    """Assemble the best-affordable :class:`ContextPacket` for ``task`` from ``substrate``.

    Retrieves relevant items (:meth:`Substrate.search`), then packs them in descending relevance until
    the character budget or item cap is reached -- always keeping at least the single most relevant
    item so a tiny budget still yields something. Emits a ``context`` telemetry event.
    """
    budget = budget or ContextBudget()
    hits = substrate.search(task, k=max(budget.max_items * 2, 8), kind=kind, scope=scope)

    selected: list[SubstrateItem] = []
    scores: list[float] = []
    used = 0
    for item, score in hits:
        piece = _render_len(item, budget.shape)
        if selected and (used + piece > budget.max_chars or len(selected) >= budget.max_items):
            break
        selected.append(item)
        scores.append(score)
        used += piece

    packet = ContextPacket(
        task=task,
        items=selected,
        scores=scores,
        budget=budget,
        used_chars=used,
        n_candidates=len(hits),
    )
    _emit(telemetry, packet)
    return packet


def _emit(telemetry: Any, packet: ContextPacket) -> None:
    try:
        from mixle.telemetry import record

        rec = telemetry.record if telemetry is not None else record
        rec(
            "context",
            features={
                "budget_chars": packet.budget.max_chars,
                "max_items": packet.budget.max_items,
                "shape": packet.budget.shape,
                "n_candidates": packet.n_candidates,
            },
            choice=[i.id for i in packet.items],
            outcome={"n_selected": len(packet.items), "used_chars": packet.used_chars},
        )
    except Exception:  # noqa: BLE001 - telemetry must never break assembly
        pass
