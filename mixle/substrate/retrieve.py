"""Cross-kind retrieval over substrate items.

:func:`retrieve` queries the substrate across selected item kinds, applies
per-kind weights, and diversifies results so evidence can span documents,
records, artifacts, traces, and other modalities. The returned
:class:`Retrieval` preserves merged relevance order, per-kind grouping, scores,
and provenance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from mixle.substrate.core import Substrate, SubstrateItem, _require_count
from mixle.utils.exact import require_exact_bool


def _require_finite_weight(value: Any, kind: str) -> float:
    """Validate a per-kind retrieval weight is finite and non-negative (MXR-080-0248).

    A weight multiplies every hit's score for its kind (``sc * w``). A negative weight silently
    reverses that kind's relevance ordering -- the worst match for that kind would sort first, the
    best last. A NaN weight makes every ordering comparison involving it false, so the merged sort
    order stops being well-defined. An infinite weight swamps every other kind's score
    unconditionally, regardless of actual relevance. All three used to reach
    ``float(weights.get(kd, 1.0))`` uncaught; rejected here, before any score is multiplied, instead
    of silently corrupting the merged order.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"weights[{kind!r}] must be a real number, got {type(value).__name__}: {value!r}")
    fvalue = float(value)
    if not math.isfinite(fvalue):
        raise ValueError(f"weights[{kind!r}] must be finite, got {fvalue!r}")
    if fvalue < 0:
        raise ValueError(f"weights[{kind!r}] must be non-negative, got {fvalue!r}")
    return fvalue


@dataclass(frozen=True)
class RetrievedHit:
    """One retrieved item immutably paired with its relevance score (MXR-080-0248).

    :class:`Retrieval` used to carry ``items``/``scores`` as two independently-settable parallel
    lists, re-paired on demand via ``zip(self.items, self.scores)`` wherever a caller needed both --
    nothing enforced they stayed the same length, so :meth:`Retrieval.provenance`'s zip could silently
    drop whichever tail was longer instead of raising, quietly omitting real evidence from a citation
    list. A ``RetrievedHit`` is the unit of construction instead: an item can never exist in a
    :class:`Retrieval` without a validated, finite score attached, and -- being frozen -- the pairing
    itself cannot drift out of sync after the fact.
    """

    item: SubstrateItem
    score: float

    def __post_init__(self) -> None:
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
            raise TypeError(
                f"RetrievedHit.score must be a real number, got {type(self.score).__name__}: {self.score!r}"
            )
        fscore = float(self.score)
        if not math.isfinite(fscore):
            raise ValueError(f"RetrievedHit.score must be finite, got {fscore!r}")
        object.__setattr__(self, "score", fscore)


_ELLIPSIS = "…"


def _truncate_for_budget(text: str, max_chars: int) -> str:
    """Cut ``text`` to at most ``max_chars`` (``max_chars >= 0``), preferring to mark the cut with an
    ellipsis when there is room for one.

    A small local twin of :mod:`mixle.substrate.context`'s own hard-truncation helper -- duplicated
    rather than imported so :meth:`Retrieval.to_context`'s single-item budget fallback does not reach
    into another module's private helper for one five-line primitive.
    """
    if len(text) <= max_chars:
        return text
    keep = max(max_chars - len(_ELLIPSIS), 0)
    return text[:keep] + (_ELLIPSIS if keep < max_chars else "")


@dataclass(init=False)
class Retrieval:
    """A planned, cross-kind retrieval result: items in merged relevance order, grouped by kind.

    Construction contract (MXR-080-0248): built from aligned, validated ``(item, score)`` pairs (see
    :class:`RetrievedHit`) -- not independently-settable parallel lists. ``items``/``scores`` stay
    available as read-only derived views for every existing reader (``r.items``, ``r.scores``,
    ``zip(r.items, r.scores)``, ``r.scores[0]``); what changed is that a ``Retrieval`` can no longer be
    constructed into a state where they disagree on length, or where a score is NaN/inf.
    """

    query: str
    hits: tuple[RetrievedHit, ...] = ()

    def __init__(
        self,
        query: str,
        items: list[SubstrateItem] | None = None,
        scores: list[float] | None = None,
    ) -> None:
        items = list(items) if items is not None else []
        scores = list(scores) if scores is not None else []
        if len(items) != len(scores):
            raise ValueError(
                "Retrieval requires items and scores of equal length; got "
                f"{len(items)} item(s) and {len(scores)} score(s)"
            )
        self.query = query
        self.hits = tuple(RetrievedHit(item=it, score=sc) for it, sc in zip(items, scores, strict=True))

    @property
    def items(self) -> list[SubstrateItem]:
        """Retrieved items in merged relevance order (a read-only derived view of :attr:`hits`)."""
        return [h.item for h in self.hits]

    @property
    def scores(self) -> list[float]:
        """Scores aligned 1:1 with :attr:`items` (a read-only derived view of :attr:`hits`)."""
        return [h.score for h in self.hits]

    def by_kind(self) -> dict[str, list[SubstrateItem]]:
        """Group retrieved items by substrate kind."""
        out: dict[str, list[SubstrateItem]] = {}
        for h in self.hits:
            out.setdefault(h.item.kind, []).append(h.item)
        return out

    def kinds(self) -> list[str]:
        """Return the sorted substrate kinds present in the result."""
        return sorted(self.by_kind())

    def top(self, n: int) -> list[SubstrateItem]:
        """Return the top ``n`` retrieved items.

        ``n`` shares :func:`Substrate.search`'s exact-non-negative-integer contract (MXR-080-0236) --
        no bool, no fractional value, no negative-indexing trick.
        """
        return [h.item for h in self.hits[: _require_count(n, "n")]]

    def provenance(self) -> list[dict[str, Any]]:
        """Return compact provenance records for retrieved items."""
        return [
            {
                "id": h.item.id,
                "kind": h.item.kind,
                "source": h.item.provenance.get("source") or h.item.provenance.get("path"),
                "score": round(h.score, 4),
            }
            for h in self.hits
        ]

    def to_context(self, task: str | None = None, **assemble_kw: Any) -> Any:
        """Budget THIS retrieval's own ordered, weighted hits into a ``ContextPacket`` (MXR-080-0249).

        Previously this copied ``self.items`` into a throwaway
        :class:`~mixle.substrate.core.Substrate` and called
        :func:`~mixle.substrate.context.assemble_context`, which ran ITS OWN fresh, unweighted search
        over that tiny shard -- discarding this retrieval's cross-kind weights, diversification order,
        and scores entirely, and substituting whatever a brand-new lexical/cosine fit over just these
        few items happened to produce instead. A weighted retrieval that had ranked an artifact above
        a text item (scores 50 and 1) came back from that fresh search reordered text-before-artifact,
        rescored 1 and 0.5 -- the weighting had silently vanished.

        This instead packs ``self.hits`` -- already in this retrieval's own decided order, at their
        own decided scores -- directly against the budget, with the same greedy, fit-checked packing
        :func:`~mixle.substrate.context.assemble_context` itself uses (built on the same public
        ``ContextPacket``/``ContextBudget``/``compress_text`` surface, so the fit guarantee -- the
        default rendering is always checked against its TRUE length, never an estimate -- holds here
        too), but never re-scoring, re-ordering, or re-searching.

        Accepts the same ``budget``/``compress``/``telemetry`` keywords ``assemble_context`` did.
        ``kind``/``scope`` are no longer accepted: there is no fresh search left for them to filter --
        restrict ``kinds=``/``scope=`` on the original :func:`retrieve` call instead.
        """
        from mixle.substrate.context import ContextBudget, ContextPacket, compress_text

        budget = assemble_kw.pop("budget", None) or ContextBudget()
        # Truthiness: compress="false" summarized anyway, silently replacing retrieved item text with
        # a lossy summary for a caller who asked for the text (MXR-080-1885).
        compress = require_exact_bool(assemble_kw.pop("compress", False), "pack compress")
        telemetry = assemble_kw.pop("telemetry", None)
        if assemble_kw:
            raise TypeError(
                "Retrieval.to_context() got unexpected keyword argument(s): "
                f"{sorted(assemble_kw)}; kind/scope no longer apply here (there is no fresh search to "
                "filter) -- pass kinds=/scope= to retrieve() instead"
            )
        task_text = task or self.query

        empty_len = ContextPacket(task=task_text, budget=budget).used_chars
        if empty_len > budget.max_chars:
            raise ValueError(
                f"ContextBudget.max_chars={budget.max_chars} cannot fit even an empty context for "
                f"this task (the header alone renders to {empty_len} characters); use a larger "
                "max_chars or a shorter task string"
            )

        hits = self.hits
        overhead = 0 if budget.shape == "brief" else 24
        selected: list[SubstrateItem] = []
        scores: list[float] = []
        texts: list[str] = []

        if compress and hits:
            # give each of up to max_items sources a fair share of the budget and summarize each to
            # fit, so several relevant sources are covered instead of one full document crowding out
            # the rest -- mirrors assemble_context's own compress path exactly.
            n_target = min(budget.max_items, len(hits))
            per_item = budget.max_chars // n_target - overhead
            for h in hits[:n_target]:
                if per_item < 1:
                    continue
                summary = compress_text(h.item.text, task_text, per_item)
                trial_items, trial_scores, trial_texts = [*selected, h.item], [*scores, h.score], [*texts, summary]
                trial = ContextPacket(
                    task=task_text, items=trial_items, scores=trial_scores, budget=budget, texts=trial_texts
                )
                if trial.used_chars > budget.max_chars:
                    break
                selected, scores, texts = trial_items, trial_scores, trial_texts
        else:
            for h in hits:
                if len(selected) >= budget.max_items:
                    break
                trial_items, trial_scores, trial_texts = [*selected, h.item], [*scores, h.score], [*texts, h.item.text]
                trial = ContextPacket(
                    task=task_text, items=trial_items, scores=trial_scores, budget=budget, texts=trial_texts
                )
                if trial.used_chars > budget.max_chars:
                    break
                selected, scores, texts = trial_items, trial_scores, trial_texts

        if not selected and hits:
            # a small-but-feasible budget still yields the single most relevant hit, shrunk to
            # genuinely fit -- never the full (possibly overflowing) text, never nothing when
            # something can fit.
            h = hits[0]
            item_overhead = ContextPacket(
                task=task_text, items=[h.item], scores=[h.score], budget=budget, texts=[""]
            ).used_chars
            available = budget.max_chars - item_overhead
            if available > 0:
                source = (
                    compress_text(h.item.text, task_text, available)
                    if compress and len(h.item.text) > available
                    else h.item.text
                )
                selected, scores, texts = [h.item], [h.score], [_truncate_for_budget(source, available)]

        packet = ContextPacket(
            task=task_text,
            items=selected,
            scores=scores,
            budget=budget,
            n_candidates=len(hits),
            texts=texts,
            compressed=compress and any(len(t) < len(i.text) for i, t in zip(selected, texts, strict=True)),
        )
        _emit_context(telemetry, packet)
        return packet

    def __len__(self) -> int:
        return len(self.hits)


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
        k: total items to return. Must be an exact, non-negative :class:`int` (MXR-080-0236, see
            :func:`~mixle.substrate.core._require_count`) -- ``bool`` and fractional values are
            rejected, and a negative ``k`` raises rather than silently slicing from the end. Validated
            once, up front, before any per-kind search runs, and used identically by both the
            diversified and flat merge paths below so a given ``k`` means the same thing regardless of
            ``diversify``.
        kinds: restrict to these substrate kinds (default: every kind present).
        weights: per-kind score multipliers (e.g. ``{"artifact": 1.2}`` to favor deployable models).
            Every weight must be finite and non-negative (MXR-080-0248) -- a negative weight would
            silently reverse that kind's relevance ordering, and a NaN/infinite weight would
            destabilize the merged order; both are rejected up front, before any score is multiplied.
        diversify: when True (default), interleave the top hits of each kind so the result spans
            modalities; when False, take a flat merged top-k (whichever kind scores highest wins).
        scope: restrict to a team/access scope.
    """
    k = _require_count(k, "k")
    present = kinds if kinds is not None else sorted({i.kind for i in substrate.all(scope=scope)})
    validated_weights = {kd: _require_finite_weight(w, kd) for kd, w in (weights or {}).items()}

    per_kind: dict[str, list[tuple[SubstrateItem, float]]] = {}
    for kd in present:
        hits = substrate.search(query, k=k, kind=kd, scope=scope)
        w = validated_weights.get(kd, 1.0)
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


def _emit_context(telemetry: Any, packet: Any) -> None:
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
            outcome={
                "n_selected": len(packet.items),
                "used_chars": packet.used_chars,
                "compressed": packet.compressed,
                "compression_ratio": packet.compression_ratio,
            },
        )
    except Exception:  # noqa: BLE001 - telemetry must never break assembly
        pass
