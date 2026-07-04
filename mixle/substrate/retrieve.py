"""All-data retrieval -- one planned query surface over every kind in the substrate (S1).

:meth:`Substrate.search` ranks all items together with one score, so whichever kind ranks highest
(usually embedding-scored text) crowds out the rest. But a question is often answered by knowledge
spread ACROSS kinds: a document explains the concept, an artifact is the model that computes it, a
trace shows how it was handled before. :func:`retrieve` is the planned front door: it queries the
kinds that matter, weights them, and DIVERSIFIES the result so it spans modalities instead of
returning the top-k of a single kind. The result is a typed :class:`Retrieval` with per-kind grouping
and provenance, one hop from a :class:`~mixle.substrate.context.ContextPacket`.

This is the S1 seed for all-data RAG; planned MULTI-HOP acquisition across typed indices (S2) and the
reasoner's evidence-buying action space (S3) build on this surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mixle.substrate.core import Substrate, SubstrateItem


@dataclass
class Retrieval:
    """A planned, cross-kind retrieval result: items in merged relevance order, grouped by kind."""

    query: str
    items: list[SubstrateItem] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)

    def by_kind(self) -> dict[str, list[SubstrateItem]]:
        out: dict[str, list[SubstrateItem]] = {}
        for it in self.items:
            out.setdefault(it.kind, []).append(it)
        return out

    def kinds(self) -> list[str]:
        return sorted(self.by_kind())

    def top(self, n: int) -> list[SubstrateItem]:
        return self.items[: int(n)]

    def provenance(self) -> list[dict[str, Any]]:
        return [
            {
                "id": i.id,
                "kind": i.kind,
                "source": i.provenance.get("source") or i.provenance.get("path"),
                "score": round(float(s), 4),
            }
            for i, s in zip(self.items, self.scores)
        ]

    def to_context(self, task: str | None = None, **assemble_kw: Any) -> Any:
        """Assemble a :class:`ContextPacket` from this retrieval (over an in-memory shard of its items)."""
        from mixle.substrate.context import assemble_context

        shard = Substrate()
        for it in self.items:
            shard.put(it)
        return assemble_context(shard, task or self.query, **assemble_kw)

    def __len__(self) -> int:
        return len(self.items)


def retrieve(
    substrate: Substrate,
    query: str,
    *,
    k: int = 8,
    kinds: list[str] | None = None,
    weights: dict[str, float] | None = None,
    diversify: bool = True,
    scope: str | None = None,
    telemetry: Any = None,
) -> Retrieval:
    """Plan a cross-kind retrieval for ``query`` (see module docstring).

    Args:
        k: total items to return.
        kinds: restrict to these substrate kinds (default: every kind present).
        weights: per-kind score multipliers (e.g. ``{"artifact": 1.2}`` to favor deployable models).
        diversify: when True (default), interleave the top hits of each kind so the result spans
            modalities; when False, take a flat merged top-k (whichever kind scores highest wins).
        scope: restrict to a team/access scope.
    """
    present = kinds if kinds is not None else sorted({i.kind for i in substrate.all(scope=scope)})
    weights = weights or {}

    per_kind: dict[str, list[tuple[SubstrateItem, float]]] = {}
    for kd in present:
        hits = substrate.search(query, k=k, kind=kd, scope=scope)
        w = float(weights.get(kd, 1.0))
        per_kind[kd] = [(it, sc * w) for it, sc in hits]

    if diversify:
        merged: list[tuple[SubstrateItem, float]] = []
        seen: set[str] = set()
        # round-robin across kinds (each kind's hits already in descending order) so the result set
        # spans modalities; ties in a round are broken by weighted score.
        rank = 0
        while len(merged) < k and any(rank < len(v) for v in per_kind.values()):
            layer = [(kd, per_kind[kd][rank]) for kd in present if rank < len(per_kind[kd])]
            layer.sort(key=lambda t: -t[1][1])
            for _kd, (it, sc) in layer:
                if it.id not in seen and len(merged) < k:
                    merged.append((it, sc))
                    seen.add(it.id)
            rank += 1
    else:
        merged = sorted((p for v in per_kind.values() for p in v), key=lambda t: -t[1])[:k]

    result = Retrieval(query=query, items=[it for it, _ in merged], scores=[sc for _, sc in merged])
    _emit(telemetry, result, present, diversify)
    return result


def _emit(telemetry: Any, result: Retrieval, kinds: list[str], diversify: bool) -> None:
    try:
        from mixle.telemetry import record

        rec = telemetry.record if telemetry is not None else record
        rec(
            "reason",
            features={"queried_kinds": kinds, "diversify": diversify, "action": "retrieve"},
            choice=[i.id for i in result.items],
            outcome={"n": len(result.items), "kinds_covered": len(result.by_kind())},
        )
    except Exception:  # noqa: BLE001 - telemetry must never break retrieval
        pass
