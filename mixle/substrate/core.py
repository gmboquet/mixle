"""Core substrate store and item model.

The substrate is a filesystem-backed, queryable surface for typed items with
provenance, scope, freshness metadata, tags, and links. Raw data, documents,
model artifacts, traces, simulation outputs, ontology triples, and context
packets can all be represented as :class:`SubstrateItem` records.

Text and document items rank by cosine similarity over a learned embedding when
available. Structured records and other items fall back to lexical, tag, and
provenance matching. Higher-level retrieval and context assembly build on this
single local store.
"""

from __future__ import annotations

import copy
import json
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

# The modality types a substrate item (and a ProbabilisticModule interface) can carry. Kept as plain
# strings so items serialize trivially and new modalities need no code change here.
MODALITIES = (
    "text",
    "record",
    "image",
    "signal",
    "graph",
    "field",
    "mesh",
    "tensor",
    "volume",
    "spectrum",
    "event_stream",
    "artifact",  # a fitted model / dataset / simulator artifact (payload is a path + manifest)
    "trace",  # a harvested agent/interaction trace
    "context",  # a stored ContextPacket
)


@dataclass
class SubstrateItem:
    """One typed, provenanced, scoped item in the substrate."""

    kind: str  # one of MODALITIES
    text: str = ""  # a retrievable text surface (the document, a summary, a serialized record)
    payload: dict[str, Any] = field(default_factory=dict)  # the structured content or a {"ref": path}
    provenance: dict[str, Any] = field(default_factory=dict)  # where it came from (source, hashes, parent ids)
    scope: str = "local"  # access scope: "local" or a team id
    tags: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)  # ids of related items (KG edges, lineage)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if self.kind not in MODALITIES:
            raise ValueError(f"unknown modality {self.kind!r}; expected one of {MODALITIES}")

    def to_json(self) -> dict[str, Any]:
        """Return this item as a JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> SubstrateItem:
        """Build a substrate item from its serialized dictionary form."""
        return cls(**d)


class Substrate:
    """A local shard of the knowledge substrate: a filesystem-backed store with typed retrieval.

    ``put`` / ``get`` / ``remove`` / ``all`` manage items; ``search`` retrieves the ``k`` most
    relevant items for a query, filtered by kind and scope, ranking text items semantically (a learned
    embedding over the current text corpus) and everything else lexically. ``save`` / ``load`` persist
    the shard as one ``items.jsonl`` under ``root``.

    Immutability contract: every item that crosses the store boundary is a defensive copy. ``put``
    copies the object you hand it before storing it; ``get`` / ``all`` (and therefore ``search``)
    return copies of what is stored. Mutating an object you obtained from -- or are about to pass to --
    this store never affects it: the store's own state, and everything derived from it (the semantic
    index), is independent of what any caller does with its own reference afterward. The only way to
    change a stored item is :meth:`put` (replace wholesale) or :meth:`update` (change named fields);
    both revalidate the result and correctly invalidate the semantic index, exactly like a fresh write.
    """

    def __init__(self, root: str | None = None) -> None:
        self._items: dict[str, SubstrateItem] = {}
        self.root = Path(root) if root is not None else None
        self._embedder: Any = None
        self._embed_ids: list[str] = []  # the text-item ids the current embedder index covers
        self._dirty = True  # the embedding index needs a rebuild
        if self.root is not None and (self.root / "items.jsonl").exists():
            self.load()

    # -- CRUD --------------------------------------------------------------------------------------
    def put(self, item: SubstrateItem) -> str:
        """Add or replace an item; returns its id and schedules semantic-index rebuilds for text items.

        Stores a defensive copy of ``item`` (see the class docstring's immutability contract) -- the
        caller's own object is never aliased into the store, so mutating it after this call has no
        effect. Dirty whenever this put could change what :meth:`_text_items` (the embedding index's
        real inclusion rule -- ANY kind with a truthy ``.text``, not just a fixed subset) would return:
        the new item itself carries text, OR it replaces an item that used to. That second half
        matters as much as the first -- clearing an existing item's text (or replacing it with a
        no-text item) shrinks the indexed corpus exactly as much as adding text grows it, and without
        it the stale embedding for the old text would keep matching queries after the text is gone.
        """
        previous = self._items.get(item.id)
        stored = copy.deepcopy(item)
        self._items[stored.id] = stored
        if stored.text or (previous is not None and previous.text):
            self._dirty = True
        return stored.id

    def add(self, kind: str, text: str = "", **kw: Any) -> str:
        """Convenience: build a :class:`SubstrateItem` and :meth:`put` it."""
        return self.put(SubstrateItem(kind=kind, text=text, **kw))

    def get(self, item_id: str) -> SubstrateItem | None:
        """Return a defensive copy of the item stored as ``item_id``, or ``None`` when it is absent.

        Mutating the returned object does not change what is stored -- call :meth:`update` (or
        :meth:`put` a full replacement) to make a change land, so index invalidation stays correct.
        """
        stored = self._items.get(item_id)
        return copy.deepcopy(stored) if stored is not None else None

    def remove(self, item_id: str) -> bool:
        """Remove an item by id and return whether anything was deleted."""
        existed = self._items.pop(item_id, None) is not None
        if existed:
            self._dirty = True
        return existed

    def all(self, *, kind: str | None = None, scope: str | None = None) -> list[SubstrateItem]:
        """Return defensive copies of stored items, optionally filtered by kind and scope.

        As with :meth:`get`, mutating a returned item never affects the store; use :meth:`update`.
        """
        out = list(self._items.values())
        if kind is not None:
            out = [i for i in out if i.kind == kind]
        if scope is not None:
            out = [i for i in out if i.scope == scope]
        return [copy.deepcopy(i) for i in out]

    def update(self, item_id: str, **fields: Any) -> SubstrateItem:
        """Change named fields on the stored item ``item_id``, the supported way to edit in place.

        Mutating an object returned by :meth:`get`/:meth:`all` never reaches the store -- both return
        defensive copies, so the store's own state (and anything derived from it, like the semantic
        index) is untouched by that. ``update`` is how a change actually lands: it rebuilds the stored
        item with ``fields`` applied, re-runs :class:`SubstrateItem`'s own validation
        (``__post_init__``, e.g. rejecting an unknown ``kind``) via :func:`dataclasses.replace`, and
        routes the result back through :meth:`put` so the same dirty/index-invalidation bookkeeping
        fires as it would for any other write -- a change that affects what :meth:`_text_items` covers
        (new text, cleared text, a different scope for a text item) correctly schedules a reindex
        instead of leaving the semantic index silently stale.

        Returns a defensive copy of the updated item (consistent with :meth:`get`/:meth:`all`).

        Raises:
            KeyError: ``item_id`` is not stored.
            ValueError: ``fields`` tries to change ``id`` -- update changes an item's content, not its
                identity; ``put`` a new item and ``remove`` the old one for that.
        """
        current = self._items.get(item_id)
        if current is None:
            raise KeyError(f"Substrate.update: no item with id {item_id!r}")
        if "id" in fields and fields["id"] != item_id:
            raise ValueError(
                f"Substrate.update cannot change id ({item_id!r} -> {fields['id']!r}); "
                "put() a new item and remove() the old one instead"
            )
        updated = replace(current, **fields)
        self.put(updated)
        return copy.deepcopy(updated)

    def __len__(self) -> int:
        return len(self._items)

    # -- retrieval ---------------------------------------------------------------------------------
    def _text_items(self, scope: str | None) -> list[SubstrateItem]:
        return [i for i in self._items.values() if i.text and (scope is None or i.scope == scope)]

    def reindex(self) -> None:
        """(Re)fit the embedding index over the current text-bearing items. Idempotent, lazy-called."""
        items = self._text_items(scope=None)
        if len(items) < 8:  # small corpus: a learned embedder can over-rank unsupported queries
            # (an out-of-vocabulary query lands close to SOMETHING when there are only a handful of
            # vectors), so retrieval stays on the deterministic lexical path until the corpus can
            # actually support an embedding; small corpora stay on lexical retrieval.
            self._embedder, self._embed_ids, self._dirty = None, [i.id for i in items], False
            return
        from mixle.represent import fit_embedder

        self._embed_ids = [i.id for i in items]
        self._embedder = fit_embedder([i.text for i in items], kind="text", dim=16, epochs=80, seed=0)
        self._dirty = False

    def search(
        self, query: str, k: int = 5, *, kind: str | None = None, scope: str | None = None
    ) -> list[tuple[SubstrateItem, float]]:
        """The ``k`` most relevant items to ``query`` as ``(item, score)``, filtered by kind/scope.

        Text-bearing items rank by cosine similarity in the learned embedding space; when there are too
        few items to learn one (or for a non-text query), ranking falls back to a lexical token overlap.
        Structured items with no text always rank lexically over their serialized payload + tags.
        """
        if self._dirty:
            self.reindex()
        candidates = self.all(kind=kind, scope=scope)
        if not candidates:
            return []

        scored: list[tuple[SubstrateItem, float]] = []
        if self._embedder is not None:
            qv = self._embedder.transform(query)
            id_to_row = {iid: r for r, iid in enumerate(self._embed_ids)}
            vecs = self._embedder.corpus_vectors
            for item in candidates:
                if item.id in id_to_row:
                    scored.append((item, float(vecs[id_to_row[item.id]] @ qv)))
                else:  # a candidate outside the text index (structured/no-text) -> lexical
                    scored.append((item, _lexical_score(query, item)))
        else:
            scored = [(item, _lexical_score(query, item)) for item in candidates]

        scored.sort(key=lambda t: -t[1])
        return scored[: int(k)]

    # -- persistence -------------------------------------------------------------------------------
    def save(self, root: str | None = None) -> str:
        """Persist the shard to ``{root}/items.jsonl`` (one item per line)."""
        target = Path(root) if root is not None else self.root
        if target is None:
            raise ValueError("Substrate.save needs a root (none was set at construction)")
        target.mkdir(parents=True, exist_ok=True)
        with open(target / "items.jsonl", "w") as f:
            for item in self._items.values():
                f.write(json.dumps(item.to_json()) + "\n")
        self.root = target
        return str(target)

    def load(self, root: str | None = None) -> None:
        """Load items from ``{root}/items.jsonl`` into this shard."""
        target = Path(root) if root is not None else self.root
        if target is None:
            raise ValueError("Substrate.load needs a root")
        self._items.clear()
        with open(target / "items.jsonl") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = SubstrateItem.from_json(json.loads(line))
                    self._items[item.id] = item
        self.root, self._dirty = target, True


# a minimal stoplist so shared function words can't manufacture relevance ("what is the ..." must not
# match a document on "is"/"the" alone) -- the same discipline the reasoner's action scorer applies.
_STOPWORDS = frozenset(
    "a an and are as at be by do does for from how in is of on or the to was what when where which who "
    "will with you your this that it its my".split()
)


def _content_tokens(text: str) -> set[str]:
    return {t for t in text.lower().split() if t not in _STOPWORDS}


def _token_matches(q_tok: str, toks: set[str]) -> bool:
    """Exact or prefix-morphology match ('refund' ~ 'refunds' ~ 'refund-router'), min stem length 4."""
    if q_tok in toks:
        return True
    if len(q_tok) < 4:
        return False
    return any(t.startswith(q_tok) or (len(t) >= 4 and q_tok.startswith(t)) for t in toks)


def _lexical_score(query: str, item: SubstrateItem) -> float:
    """Content-token overlap over an item's text + serialized payload + tags (the no-embedding path).

    Stopwords are excluded on BOTH sides, so relevance reflects content words (a query of only
    stopwords scores 0 everywhere); tokens match exactly or by prefix morphology, the same
    discipline the O3 compressor uses ('refund' ~ 'refunds')."""
    q = _content_tokens(str(query))
    if not q:
        return 0.0
    surface = " ".join([item.text, json.dumps(item.payload), " ".join(item.tags)])
    toks = _content_tokens(surface)
    if not toks:
        return 0.0
    return sum(1.0 for t in q if _token_matches(t, toks)) / len(q)
