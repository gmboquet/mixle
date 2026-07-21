"""Budgeted, provenanced context packets assembled from the substrate.

A :class:`ContextPacket` is a task-specific view of selected substrate items:
the task, items in relevance order, rendered text, budget, and provenance for
the included evidence. A :class:`ContextBudget` describes how much context a
target can accept and in what shape.

Assembly combines substrate retrieval with greedy budgeted selection. The most
relevant items are packed until the budget is reached, and an optional telemetry
event records the budget, usage, and number of selected items.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from mixle.substrate.core import Substrate, SubstrateItem

# --- IC-13 compatibility bridge (M0b) --------------------------------------------------------------
#
# Core substrate items are dependency-free dataclasses; `mixle_knowledge.contracts.KnowledgeItem` /
# `KnowledgeBundle` (IC-13) are the frozen, validated wire shapes the rest of the platform (mixle-knowledge,
# mixle-mlops, tools, models) exchanges. `substrate_item_to_knowledge_dict` and
# `ContextPacket.to_knowledge_bundle_dict` translate one into the other as PLAIN, JSON-native dicts --
# core never imports `mixle_knowledge`, so constructing/validating the pydantic model is the receiving
# package's (or a test's) responsibility.

GENERIC_KNOWLEDGE_SCHEMA = "mixle://schema/substrate-item/1"
PROPERTY_GRAPH_SCHEMA = "mixle://schema/property-graph/1"
TYPED_TABLE_SCHEMA = "mixle://schema/typed-table/1"
SPATIAL_MEDIA_SCHEMA = "mixle://schema/spatial-media/1"
TENSOR_SCHEMA = "mixle://schema/tensor/1"
SIGNAL_SCHEMA = "mixle://schema/signal/1"
MESH_SCHEMA = "mixle://schema/mesh/1"

# substrate kind -> (IC-13 ResourceKind value, IC-13 Modality value). An unmapped/future kind falls back
# to a generic artifact/structured pair rather than raising, so the bridge degrades instead of breaking.
_RESOURCE_KIND_BY_SUBSTRATE_KIND: dict[str, tuple[str, str]] = {
    "text": ("document", "text"),
    "record": ("table", "table"),
    "image": ("image", "image"),
    "signal": ("timeseries", "timeseries"),
    "graph": ("artifact", "graph"),
    "field": ("geospatial_layer", "raster"),
    "mesh": ("mesh", "mesh"),
    "tensor": ("tensor", "tensor"),
    "volume": ("tensor", "volume"),
    "spectrum": ("signal", "spectrum"),
    "event_stream": ("dataset", "timeseries"),
    "artifact": ("artifact", "structured"),
    "trace": ("trace", "structured"),
    "context": ("context_packet", "structured"),
}


def _canonical_json(obj: Any) -> str:
    """A stable JSON encoding (sorted keys, no incidental whitespace) so equal content hashes equal."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _canonical_item_hash(
    *,
    schema_uri: str,
    schema_version: str,
    payload: Any,
    artifact_ref: str | None,
    metadata: dict[str, Any],
) -> str:
    """IC-13's frozen ``content_hash`` recipe: sha256 over ``{schema_uri,schema_version,payload,
    artifact_ref,metadata}`` -- never over a text summary alone, so structurally-equal items dedupe and
    silent drift is detectable. A separate artifact-byte digest (see :func:`_provenance_refs`/metadata)
    verifies the referenced bytes independently and is never substituted for this hash."""
    envelope = {
        "schema_uri": schema_uri,
        "schema_version": schema_version,
        "payload": payload,
        "artifact_ref": artifact_ref,
        "metadata": metadata,
    }
    return hashlib.sha256(_canonical_json(envelope).encode("utf-8")).hexdigest()


def _as_artifact_ref(ref: Any) -> str:
    """Make a bare path/id URI-like (``artifact_ref`` convention); an already-URI value passes through."""
    text = str(ref)
    return text if "://" in text else f"substrate://artifact/{text.lstrip('/')}"


def _looks_like_typed_table(payload: dict[str, Any]) -> bool:
    return {"primary_key", "columns"} <= set(payload)


def _infer_schema_uri(item: SubstrateItem, raw_payload: dict[str, Any]) -> str:
    """Best-effort default schema for an item with no explicit override (a heuristic bridge, not a
    validator -- core does not check the payload actually conforms; the receiving package does)."""
    if item.kind == "graph":
        return PROPERTY_GRAPH_SCHEMA
    if item.kind in ("image", "field"):
        return SPATIAL_MEDIA_SCHEMA
    if item.kind in ("tensor", "volume"):
        return TENSOR_SCHEMA
    if item.kind in ("signal", "spectrum"):
        return SIGNAL_SCHEMA
    if item.kind == "mesh":
        return MESH_SCHEMA
    if item.kind == "record" and _looks_like_typed_table(raw_payload):
        return TYPED_TABLE_SCHEMA
    return GENERIC_KNOWLEDGE_SCHEMA


def _access_policy_dict(scope: str) -> dict[str, Any]:
    """``SubstrateItem.scope`` is ``"local"`` or a team id; IC-13's ``AccessPolicy`` names an explicit scope."""
    if scope == "local":
        return {"scope": "private"}
    return {"scope": "team", "teams": [scope]}


def _provenance_refs(item: SubstrateItem) -> list[dict[str, Any]]:
    """Summarize ``item.provenance`` as one IC-13 ``SourceRef``-shaped dict (``uri`` is the only
    required field there). The substrate's own file-byte digest (``provenance["content_hash"]``, a
    truncated sha256 -- see ``freshness.content_hash``) is NOT put in ``sha256`` here: that field is
    64-hex only, while the substrate's is a 32-hex digest of different provenance; it is preserved
    verbatim in ``metadata["substrate_provenance"]`` instead so it is never lost nor misrepresented."""
    prov = item.provenance or {}
    uri = prov.get("source") or prov.get("path") or prov.get("uri") or f"substrate:{item.id}"
    ref: dict[str, Any] = {"uri": str(uri)}
    if prov.get("media_type"):
        ref["media_type"] = prov["media_type"]
    if prov.get("version") is not None:
        ref["version"] = str(prov["version"])
    if prov.get("license"):
        ref["license"] = prov["license"]
    return [ref]


def _iso(epoch_seconds: float) -> str:
    return datetime.fromtimestamp(float(epoch_seconds), tz=UTC).isoformat()


def substrate_item_to_knowledge_dict(item: SubstrateItem, *, schema_uri: str | None = None) -> dict[str, Any]:
    """Bridge one :class:`SubstrateItem` to an IC-13 ``KnowledgeItem``-shaped dict (M0b).

    Core never imports ``mixle_knowledge``: this returns a plain, JSON-native dict with exactly the
    frozen field names, so a caller (or test) can validate it with
    ``mixle_knowledge.contracts.KnowledgeItem.model_validate`` without core depending on that package.

    ``item.payload`` is the canonical structured payload UNLESS it carries a ``"ref"``/``"path"``
    pointer (the substrate's own artifact convention -- see ``ingest_artifacts``), in which case that
    pointer becomes ``artifact_ref`` and any remaining payload keys (e.g. spatial metadata) stay in
    ``payload``. ``schema_uri`` defaults from ``item.kind`` (and, for a ``record`` item whose payload
    already looks like a typed table, from its shape) but an explicit override always wins. The item
    hash is always computed fresh over the canonical envelope; it never hashes ``item.text`` alone.
    """
    raw_payload: dict[str, Any] = dict(item.payload) if item.payload else {}
    resolved_schema_uri = schema_uri or _infer_schema_uri(item, raw_payload)

    payload_out: dict[str, Any] | None = dict(raw_payload)
    artifact_ref: str | None = None
    for ref_key in ("ref", "path"):
        if ref_key in payload_out:
            artifact_ref = _as_artifact_ref(payload_out.pop(ref_key))
            break
    if not payload_out and artifact_ref is not None:
        payload_out = None
    if payload_out is None and artifact_ref is None:
        payload_out = {}  # IC-13 requires payload or artifact_ref; an empty canonical payload satisfies it

    resource_kind, modality = _RESOURCE_KIND_BY_SUBSTRATE_KIND.get(item.kind, ("artifact", "structured"))

    metadata: dict[str, Any] = {"tags": list(item.tags)}
    if item.provenance:
        metadata["substrate_provenance"] = dict(item.provenance)

    schema_version = "1.0.0"
    item_hash = _canonical_item_hash(
        schema_uri=resolved_schema_uri,
        schema_version=schema_version,
        payload=payload_out,
        artifact_ref=artifact_ref,
        metadata=metadata,
    )

    return {
        "id": item.id,
        "kind": resource_kind,
        "modality": modality,
        "schema_uri": resolved_schema_uri,
        "schema_version": schema_version,
        "media_type": None,
        "content_hash": item_hash,
        "payload": payload_out,
        "artifact_ref": artifact_ref,
        "text_surface": item.text or None,
        "provenance": _provenance_refs(item),
        "relations": [{"predicate": "related_to", "target_id": link} for link in item.links],
        "uncertainty": None,
        "metadata": metadata,
        "access": _access_policy_dict(item.scope),
        "revision": 1,
        "supersedes": [],
        "created_at": _iso(item.created_at),
    }


@dataclass
class ContextBudget:
    """What a target can take -- the DeviceSpec of context. ``shape`` hints the rendering style."""

    max_chars: int = 2000
    max_items: int = 20
    shape: str = "passages"  # 'passages' (LLM) | 'brief' (human) | 'features' (student)


@dataclass
class ContextPacket:
    """A budgeted, provenanced view of the substrate assembled for one target + task.

    ``texts`` holds the text actually used per item -- the full item surface, or (when the packet was
    compressed) an extractive summary that keeps only the query-relevant sentences. ``preservation``
    receipts how much of each item's query-relevant content survived, so compression is measured, not
    trusted.
    """

    task: str
    items: list[SubstrateItem] = field(default_factory=list)  # selected, in descending relevance
    scores: list[float] = field(default_factory=list)
    budget: ContextBudget = field(default_factory=ContextBudget)
    used_chars: int = 0
    n_candidates: int = 0  # how many the retriever surfaced before budgeting
    texts: list[str] = field(default_factory=list)  # the text actually used per item (full or compressed)
    compressed: bool = False

    def __post_init__(self) -> None:
        if not self.texts:  # default: use the items' full surfaces
            self.texts = [i.text for i in self.items]

    def render(self, *, header: bool = True) -> str:
        """The assembled context string the target consumes (respecting the budget shape)."""
        head = f"# Context for: {self.task}\n" if header else ""
        if self.budget.shape == "brief":
            body = "\n".join(f"- {_one_line(t)}" for t in self.texts)
        else:  # passages / features: full/compressed item surfaces, provenance-tagged
            body = "\n\n".join(f"[{i.kind}:{i.id}] {t}" for i, t in zip(self.items, self.texts))
        return head + body

    def preservation(self) -> list[float]:
        """Per item, the fraction of the task's query terms retained in the used text (1.0 = all kept).

        The receipt for compression: a value near 1.0 means the summary kept what the query cares
        about; a low value flags an item whose relevant content was squeezed out.
        """
        return [_query_coverage(used, self.task, full=i.text) for i, used in zip(self.items, self.texts)]

    @property
    def compression_ratio(self) -> float:
        """Used chars / full chars over the selected items (1.0 = uncompressed)."""
        full = sum(len(i.text) for i in self.items)
        used = sum(len(t) for t in self.texts)
        return round(used / full, 4) if full else 1.0

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
        """Return a JSON-serializable context-packet summary."""
        return {
            "task": self.task,
            "n_items": len(self.items),
            "n_candidates": self.n_candidates,
            "used_chars": self.used_chars,
            "budget_chars": self.budget.max_chars,
            "shape": self.budget.shape,
            "compressed": self.compressed,
            "compression_ratio": self.compression_ratio,
            "provenance": self.provenance(),
        }

    def to_knowledge_dict(
        self,
        *,
        id: str,  # noqa: A002 - matches the mixle-knowledge ContextPacket field name exactly
        project_id: str,
        target_kind: str,
        target_id: str | None = None,
        expected_output_schema: dict[str, Any] | None = None,
        factuality: Any = None,
    ) -> dict[str, Any]:
        """Return a plain dict shaped like ``mixle_knowledge.contracts.ContextPacket``.

        The exported fields cover ``id``, ``project_id``, ``task``, ``target_kind``, ``target_id``,
        token and byte budgets, evidence item identifiers, constraints, citations,
        ``expected_output_schema``, and ``payload``. Constructing a validated pydantic object is the
        receiving package's responsibility; core mixle intentionally keeps this as a dependency-free
        dictionary so platform contract packages can depend on core rather than the reverse.

        When ``factuality`` is a :class:`~mixle.substrate.factuality.FactualityReceipt`, it is included
        in ``payload["factuality"]`` so receivers can inspect grounding metadata before trusting the
        packet.
        """
        citations = [{"uri": p["source"] or f"substrate:{p['id']}", "media_type": p["kind"]} for p in self.provenance()]
        payload: dict[str, Any] = {
            "rendered": self.render(),
            "shape": self.budget.shape,
            "compressed": self.compressed,
            "compression_ratio": self.compression_ratio,
            "preservation": self.preservation(),
        }
        if factuality is not None:
            payload["factuality"] = factuality.as_dict()
        return {
            "id": id,
            "project_id": project_id,
            "task": self.task,
            "target_kind": target_kind,
            "target_id": target_id,
            "token_budget": None,
            "byte_budget": self.budget.max_chars,
            "evidence_item_ids": [i.id for i in self.items],
            "constraints": [],
            "citations": citations,
            "expected_output_schema": expected_output_schema or {},
            "payload": payload,
        }

    def to_knowledge_bundle_dict(
        self,
        *,
        id: str,  # noqa: A002 - matches the mixle-knowledge KnowledgeBundle field name exactly
        project_id: str,
        target_kind: str,
        target_id: str | None = None,
        expected_output_schema: dict[str, Any] | None = None,
        gaps: list[dict[str, Any]] | None = None,
        required_capability_ids: list[str] | None = None,
        handoff_policy: dict[str, Any] | None = None,
        continuation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return a plain dict shaped like ``mixle_knowledge.contracts.KnowledgeBundle`` (IC-13, M0b).

        Unlike :meth:`to_knowledge_dict` (kept, unchanged, as a deprecated compatibility view -- M1c
        must never use it as canonical state), this bridge carries each selected item's OWN canonical
        structured payload/artifact_ref/relations (:func:`substrate_item_to_knowledge_dict`) instead of
        flattening everything into one rendered text string: a graph stays a graph, a typed table stays
        a typed table, an image keeps its artifact ref. ``self.render()`` and the compression receipt
        are only a disposable legacy view, kept under ``renderings["legacy_text"]``; they may be
        recomputed or dropped at will without touching any item's canonical payload or hash.

        As with :func:`substrate_item_to_knowledge_dict`, core never imports ``mixle_knowledge`` --
        the caller validates the result (e.g. with ``KnowledgeBundle.model_validate``).
        """
        citations = [{"uri": p["source"] or f"substrate:{p['id']}", "media_type": p["kind"]} for p in self.provenance()]
        return {
            "id": id,
            "project_id": project_id,
            "task": self.task,
            "target_kind": target_kind,
            "target_id": target_id,
            "revision": 1,
            "items": [substrate_item_to_knowledge_dict(item) for item in self.items],
            "gaps": gaps or [],
            "constraints": [],
            "citations": citations,
            "expected_output_schema": expected_output_schema or {},
            "token_budget": None,
            "byte_budget": self.budget.max_chars,
            "lineage": [],
            "required_capability_ids": required_capability_ids or [],
            "handoff_policy": handoff_policy or {},
            "continuation": continuation,
            "renderings": {
                "legacy_text": {
                    "text": self.render(),
                    "shape": self.budget.shape,
                    "compressed": self.compressed,
                    "compression_ratio": self.compression_ratio,
                    "preservation": self.preservation(),
                }
            },
        }

    def __len__(self) -> int:
        return len(self.items)


def _one_line(text: str, limit: int = 160) -> str:
    t = " ".join(str(text).split())
    return t if len(t) <= limit else t[: limit - 1] + "…"


def _render_len(item: SubstrateItem, shape: str, text: str | None = None) -> int:
    body = item.text if text is None else text
    return len(_one_line(body)) if shape == "brief" else len(body) + len(item.kind) + len(item.id) + 6


def _sentences(text: str) -> list[str]:
    import re

    parts = re.split(r"(?<=[.!?])\s+|\n+", str(text).strip())
    return [p.strip() for p in parts if p.strip()]


def _q_tokens(task: str) -> set[str]:
    return {w for w in str(task).lower().split() if len(w) > 2}


def _covered(q: set[str], text: str) -> set[str]:
    """Query tokens matched in ``text`` by prefix overlap (so 'refund' matches 'refunds', 'refunded')."""
    toks = str(text).lower().split()
    hit = set()
    for w in q:
        stem = w[:-1] if len(w) > 4 and w.endswith("s") else w
        if any(t == w or t.startswith(stem) or w.startswith(t[:-1] if len(t) > 4 else t) for t in toks):
            hit.add(w)
    return hit


def _query_coverage(text: str, task: str, *, full: str | None = None) -> float:
    """Fraction of the task's query tokens present in ``text`` (by prefix match). With ``full``,
    normalize by the terms the full item actually had, so an item that never mentioned a query term
    is not penalized for a summary that also lacks it."""
    q = _q_tokens(task)
    if not q:
        return 1.0
    if full is not None:
        present = _covered(q, full)
        if not present:
            return 1.0
        return len(_covered(q, text) & present) / len(present)
    return len(_covered(q, text)) / len(q)


def _compress(text: str, task: str, max_chars: int) -> str:
    """Extractive summary: keep the highest query-relevant sentences (in original order) within budget.

    Deterministic and torch-free -- sentences are ranked by query-token overlap, the top ones packed
    until ``max_chars``, then re-emitted in their original order so the summary reads coherently.
    """
    if len(text) <= max_chars:
        return text
    sents = _sentences(text)
    if len(sents) <= 1:
        return text[: max(max_chars - 1, 1)] + "…"
    q = _q_tokens(task)
    scored = sorted(
        range(len(sents)),
        key=lambda i: (-len(_covered(q, sents[i])), len(sents[i])),
    )
    keep: set[int] = set()
    used = 0
    for i in scored:
        add = len(sents[i]) + 1
        if used + add > max_chars and keep:
            break
        keep.add(i)
        used += add
    summary = " ".join(sents[i] for i in sorted(keep))
    return summary if summary else text[: max(max_chars - 1, 1)] + "…"


def assemble_context(
    substrate: Substrate,
    task: str,
    *,
    budget: ContextBudget | None = None,
    kind: str | None = None,
    scope: str | None = None,
    compress: bool = False,
    telemetry: Any = None,
) -> ContextPacket:
    """Assemble the best-affordable :class:`ContextPacket` for ``task`` from ``substrate``.

    Retrieves relevant items (:meth:`Substrate.search`), then packs them in descending relevance until
    the character budget or item cap is reached -- always keeping at least the single most relevant
    item so a small budget still yields something. With ``compress=True``, an
    item too large to fit whole is extractively summarized to its query-relevant
    sentences instead of dropped; ``packet.preservation()`` reports what was
    kept. Emits a ``context`` event when telemetry is supplied.
    """
    budget = budget or ContextBudget()
    hits = substrate.search(task, k=max(budget.max_items * 2, 8), kind=kind, scope=scope)

    selected: list[SubstrateItem] = []
    scores: list[float] = []
    texts: list[str] = []
    used = 0
    overhead = 0 if budget.shape == "brief" else 24  # per-item provenance-tag overhead estimate

    if compress and hits:
        # give each of up to max_items sources a fair share of the budget and summarize each to fit,
        # so several relevant sources are covered instead of one full document crowding out the rest.
        n_target = min(budget.max_items, len(hits))
        per_item = max(budget.max_chars // n_target - overhead, 40)
        for item, score in hits[:n_target]:
            summary = _compress(item.text, task, per_item)
            piece = _render_len(item, budget.shape, text=summary)
            if selected and used + piece > budget.max_chars:
                break
            selected.append(item)
            scores.append(score)
            texts.append(summary)
            used += piece
    else:
        for item, score in hits:
            piece = _render_len(item, budget.shape)
            if selected and (used + piece > budget.max_chars or len(selected) >= budget.max_items):
                break
            selected.append(item)
            scores.append(score)
            texts.append(item.text)
            used += piece

    packet = ContextPacket(
        task=task,
        items=selected,
        scores=scores,
        budget=budget,
        used_chars=used,
        n_candidates=len(hits),
        texts=texts,
        compressed=compress and any(len(t) < len(i.text) for i, t in zip(selected, texts)),
    )
    _emit(telemetry, packet)
    return packet


def compress_text(text: str, task: str, max_chars: int) -> str:
    """Extractive, torch-free summary of ``text`` keeping the sentences most relevant to ``task``,
    within ``max_chars`` (the standalone compressor used by :func:`assemble_context` with ``compress=True``)."""
    return _compress(text, task, int(max_chars))


@dataclass
class ReceiverProfile:
    """A named receiver's capacity -- what :func:`assemble_for_receivers` budgets and shapes for it.

    A frontier LM and a local student are not the same target: the LM affords a large, prose-shaped
    context; the student needs a small, feature-shaped one. ``ReceiverProfile`` names that difference
    so it is set once per receiver, not re-derived ad hoc at every call site."""

    name: str
    max_chars: int = 2000
    max_items: int = 20
    shape: str = "passages"  # 'passages' (LLM) | 'brief' (human) | 'features' (student)
    compress: bool = False

    def to_budget(self) -> ContextBudget:
        """Convert this receiver profile to a context budget."""
        return ContextBudget(max_chars=self.max_chars, max_items=self.max_items, shape=self.shape)


def assemble_for_receivers(
    substrate: Substrate,
    task: str,
    receivers: Sequence[ReceiverProfile],
    *,
    kind: str | None = None,
    scope: str | None = None,
    telemetry: Any = None,
) -> dict[str, ContextPacket]:
    """Assemble ONE task-conditioned :class:`ContextPacket` per named receiver -- the concrete
    receiver-conditioned compression path.

    Two receivers reading the same substrate for the same task get genuinely different renderings:
    budget, shape, and, via ``compress``, which sentences survive. The result is not the same blob
    truncated to fit each consumer.

        packets = assemble_for_receivers(substrate, task, [
            ReceiverProfile("frontier_llm", max_chars=2000, shape="passages"),
            ReceiverProfile("local_student", max_chars=200, shape="features", compress=True),
        ])
        packets["frontier_llm"].render(), packets["local_student"].render()
    """
    return {
        r.name: assemble_context(
            substrate, task, budget=r.to_budget(), kind=kind, scope=scope, compress=r.compress, telemetry=telemetry
        )
        for r in receivers
    }


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
            outcome={
                "n_selected": len(packet.items),
                "used_chars": packet.used_chars,
                "compressed": packet.compressed,
                "compression_ratio": packet.compression_ratio,
            },
        )
    except Exception:  # noqa: BLE001 - telemetry must never break assembly
        pass
