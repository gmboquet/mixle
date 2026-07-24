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
import os
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from mixle.substrate.security import SECRET_POLICIES, SecretPolicy, enforce_secret_policy

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
    """One typed, provenanced, scoped item in the substrate.

    Edge typing (MXR-080-0261): ``links`` and ``derived_from`` are both lists of item ids, but they
    are NOT interchangeable. ``links`` is the generic, untyped KG-relation surface -- "related to",
    "mentions", "co-occurs with", anything :mod:`~mixle.substrate.kg_rag`/:mod:`~mixle.substrate.multihop`
    want to associate two items by. ``derived_from`` is narrower and load-bearing: it is the ONLY place
    genuine provenance/ancestry ("this item was derived from that one") is recorded, and it is the only
    field :func:`~mixle.substrate.trust.verify_lineage` ever traverses. Before this field existed, lineage
    verification walked ``links`` itself, so a merely-related entity was indistinguishable from a true
    derivation parent -- putting a citation-worthy relation into ``links`` silently made it certifiable
    ancestry. Keeping the two lists separate (rather than tagging each entry with a type) is a deliberate
    minimal-diff choice: every existing reader of ``links`` (:mod:`kg_rag`, :mod:`multihop`,
    :mod:`freshness`, :mod:`context`, :mod:`governance`, :mod:`spaces`) keeps reading exactly the same
    untyped list it always has, with no change to its shape or meaning.
    """

    kind: str  # one of MODALITIES
    text: str = ""  # a retrievable text surface (the document, a summary, a serialized record)
    payload: dict[str, Any] = field(default_factory=dict)  # the structured content or a {"ref": path}
    provenance: dict[str, Any] = field(default_factory=dict)  # where it came from (source, hashes, parent ids)
    scope: str = "local"  # access scope: "local" or a team id
    tags: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)  # ids of RELATED items (generic KG edges) -- not ancestry
    # ids this item genuinely DERIVES FROM (true provenance/ancestry parents) -- the only edges
    # mixle.substrate.trust.verify_lineage() traverses (MXR-080-0261). Deliberately a separate list
    # rather than a type tag on `links`'s entries: every pre-existing reader of `links` is unaffected,
    # and an item with no recorded ancestry simply has an empty `derived_from`, not a guess.
    derived_from: list[str] = field(default_factory=list)
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

    Scope isolation contract: ``search(..., scope=X)`` ranks using an embedding index fit ONLY on
    scope ``X``'s own text items (see :meth:`reindex`) -- a scoped query's results, and the model that
    produced them, never depend on any other scope's content, so the presence or content of items in a
    scope you cannot pass never leaks through a scope you can. ``scope=None`` (the default) is the one
    deliberate exception: it asks for every item across every scope, so its index is fit over
    everything, same as asking for all of it directly through ``all()``/``get()`` would show you.

    Secret-handling contract (MXR-080-0262): every write -- :meth:`put`, and :meth:`update`, which
    routes through it -- scans ``item.text``, ``item.payload`` (recursively), and ``item.tags`` for
    known secret shapes (:mod:`mixle.substrate.security`) using exactly the surface
    :func:`_lexical_score` serializes for search, BEFORE anything reaches ``self._items`` or either
    index. This is unconditional: there is no path into the store that skips it, so the
    redact-before-store guard :func:`~mixle.substrate.security.safe_text` always offered is no longer
    opt-in. ``secret_policy`` (set at construction, or overridden per call) picks what happens when a
    secret is found: ``"redact"`` (the default) masks it in place so only the masked form is ever
    stored, embedded, or served; ``"reject"`` raises
    :class:`~mixle.substrate.security.SecretPolicyError` and stores nothing.
    """

    def __init__(self, root: str | None = None, *, secret_policy: SecretPolicy = "redact") -> None:
        if secret_policy not in SECRET_POLICIES:
            raise ValueError(f"unknown secret_policy {secret_policy!r}; expected one of {SECRET_POLICIES}")
        self.secret_policy = secret_policy
        self._items: dict[str, SubstrateItem] = {}
        self.root = Path(root) if root is not None else None
        # One embedder PER visibility domain (see reindex()): _embedders[scope] / _embed_ids[scope] is
        # fit ONLY on that scope's own text items, so a scoped query's transform and ranking never
        # depend on another scope's content. Key None is the unrestricted (scope=None) index, fit over
        # every item -- the caller asking for that already gets everything, so pooling is not a leak.
        self._embedders: dict[str | None, Any] = {}
        self._embed_ids: dict[str | None, list[str]] = {}  # per-index: the text-item ids it covers
        self._dirty = True  # the embedding index needs a rebuild
        if self.root is not None and (self.root / "items.jsonl").exists():
            self.load()

    # -- CRUD --------------------------------------------------------------------------------------
    def _store(self, item: SubstrateItem, *, secret_policy: SecretPolicy | None = None) -> SubstrateItem:
        """Sanitize ``item`` per the active secret policy, deep-copy it into the store, and return the
        stored (already-deep-copied) item. The one place an item crosses into ``self._items`` --
        :meth:`put` and :meth:`update` (via :meth:`put`) both route through this, so every write gets
        the same secret guard and the same dirty/index-invalidation bookkeeping unconditionally.
        """
        policy = secret_policy if secret_policy is not None else self.secret_policy
        sanitized, _scan = enforce_secret_policy(item, policy=policy)
        previous = self._items.get(sanitized.id)
        stored = copy.deepcopy(sanitized)
        self._items[stored.id] = stored
        if stored.text or (previous is not None and previous.text):
            self._dirty = True
        return stored

    def put(self, item: SubstrateItem, *, secret_policy: SecretPolicy | None = None) -> str:
        """Add or replace an item; returns its id and schedules semantic-index rebuilds for text items.

        Stores a defensive copy of ``item`` (see the class docstring's immutability contract) -- the
        caller's own object is never aliased into the store, so mutating it after this call has no
        effect. Dirty whenever this put could change what :meth:`_text_items` (the embedding index's
        real inclusion rule -- ANY kind with a truthy ``.text``, not just a fixed subset) would return:
        the new item itself carries text, OR it replaces an item that used to. That second half
        matters as much as the first -- clearing an existing item's text (or replacing it with a
        no-text item) shrinks the indexed corpus exactly as much as adding text grows it, and without
        it the stale embedding for the old text would keep matching queries after the text is gone.

        Secret-handling (MXR-080-0262, see the class docstring's secret-handling contract): before
        anything is stored or indexed, this scans ``item.text``, ``item.payload`` (recursively), and
        ``item.tags`` -- the same surface :func:`_lexical_score` serializes for search -- for known
        secret shapes, and acts per ``secret_policy`` (this call's override, else ``self.secret_policy``,
        default ``"redact"``): ``"redact"`` masks detected secrets in place before storing;
        ``"reject"`` raises :class:`~mixle.substrate.security.SecretPolicyError` and stores nothing.
        This is unconditional -- there is no way to reach ``self._items`` that skips it.
        """
        return self._store(item, secret_policy=secret_policy).id

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

    def update(self, item_id: str, *, secret_policy: SecretPolicy | None = None, **fields: Any) -> SubstrateItem:
        """Change named fields on the stored item ``item_id``, the supported way to edit in place.

        Mutating an object returned by :meth:`get`/:meth:`all` never reaches the store -- both return
        defensive copies, so the store's own state (and anything derived from it, like the semantic
        index) is untouched by that. ``update`` is how a change actually lands: it rebuilds the stored
        item with ``fields`` applied, re-runs :class:`SubstrateItem`'s own validation
        (``__post_init__``, e.g. rejecting an unknown ``kind``) via :func:`dataclasses.replace`, and
        routes the result back through :meth:`put`'s storage path so the same dirty/index-invalidation
        bookkeeping AND the same secret-handling guard (see the class docstring's secret-handling
        contract; ``secret_policy`` here overrides ``self.secret_policy`` for this call only) fire as
        they would for any other write -- a change that affects what :meth:`_text_items` covers (new
        text, cleared text, a different scope for a text item) correctly schedules a reindex instead of
        leaving the semantic index silently stale, and a change that introduces a secret into any field
        is masked (or rejected) exactly like a fresh :meth:`put` would.

        Returns a defensive copy of the item as actually stored (post-redaction, consistent with
        :meth:`get`/:meth:`all` -- NOT the raw pre-policy ``fields``-applied object).

        Raises:
            KeyError: ``item_id`` is not stored.
            ValueError: ``fields`` tries to change ``id`` -- update changes an item's content, not its
                identity; ``put`` a new item and ``remove`` the old one for that.
            SecretPolicyError: ``secret_policy="reject"`` (or ``self.secret_policy`` is) and the update
                introduces a secret anywhere in the surface -- nothing is changed.
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
        stored = self._store(updated, secret_policy=secret_policy)
        return copy.deepcopy(stored)

    def __len__(self) -> int:
        return len(self._items)

    # -- retrieval ---------------------------------------------------------------------------------
    def _text_items(self, scope: str | None) -> list[SubstrateItem]:
        return [i for i in self._items.values() if i.text and (scope is None or i.scope == scope)]

    def reindex(self) -> None:
        """(Re)fit one embedding index PER visibility domain. Idempotent, lazy-called.

        MXR-080-0237: a single embedder fit over every scope's text pooled together makes a scoped
        query's transform and ranking depend on inaccessible scopes' content -- a membership side
        channel (whether some distinctive text exists in a scope you can never read is inferable from
        how it shifts your OWN scoped query's scores). Fixed by fitting one embedder per distinct scope
        value present, each ONLY on that scope's own text items, plus one unrestricted index (key
        ``None``) fit over everything for callers that pass ``scope=None`` -- that caller is asking for
        every scope by construction, so pooling there asserts no isolation boundary to violate.
        :meth:`search` picks the index matching its own ``scope`` argument, so a scope's results and the
        model that produced them never see a single byte of another scope's text.
        """
        self._embedders, self._embed_ids = {}, {}
        scopes_present = {i.scope for i in self._items.values() if i.text}
        for scope_key in (*scopes_present, None):
            self._fit_index(scope_key)
        self._dirty = False

    def _fit_index(self, scope_key: str | None) -> None:
        """Fit (or fall back to lexical-only) the embedding index for one visibility domain."""
        items = self._text_items(scope=scope_key)
        if len(items) < 8:  # small corpus: a learned embedder can over-rank unsupported queries
            # (an out-of-vocabulary query lands close to SOMETHING when there are only a handful of
            # vectors), so retrieval stays on the deterministic lexical path until the corpus can
            # actually support an embedding; small corpora stay on lexical retrieval. This threshold now
            # applies PER scope: a scope can't borrow another scope's items to clear it, by design.
            self._embedders[scope_key] = None
            self._embed_ids[scope_key] = [i.id for i in items]
            return
        from mixle.represent import fit_embedder

        self._embed_ids[scope_key] = [i.id for i in items]
        self._embedders[scope_key] = fit_embedder([i.text for i in items], kind="text", dim=16, epochs=80, seed=0)

    def search(
        self, query: str, k: int = 5, *, kind: str | None = None, scope: str | None = None
    ) -> list[tuple[SubstrateItem, float]]:
        """The ``k`` most relevant items to ``query`` as ``(item, score)``, filtered by kind/scope.

        Text-bearing items rank by cosine similarity in the learned embedding space fit for THIS
        ``scope`` alone (see :meth:`reindex`); when there are too few same-scope items to learn one (or
        for a non-text query), ranking falls back to a lexical token overlap. Structured items with no
        text always rank lexically over their serialized payload + tags.
        """
        if self._dirty:
            self.reindex()
        candidates = self.all(kind=kind, scope=scope)
        if not candidates:
            return []

        embedder = self._embedders.get(scope)
        scored: list[tuple[SubstrateItem, float]] = []
        if embedder is not None:
            qv = embedder.transform(query)
            id_to_row = {iid: r for r, iid in enumerate(self._embed_ids.get(scope, []))}
            vecs = embedder.corpus_vectors
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
        """Persist the shard to ``{root}/items.jsonl`` (one item per line), atomically.

        The full snapshot is written to a sibling temp file in ``target`` first, fsynced, then
        published with a single ``os.replace`` -- mirrors :func:`mixle.task.artifact._atomic_json_dump`
        and :meth:`mixle.system.registry.Registry._write_index`'s temp-file-plus-``os.replace``
        convention. A plain ``open(path, "w")`` truncates ``items.jsonl`` BEFORE the new content is
        written, so any failure partway through this method (a crash, a disk-full write, a
        non-serializable item further down the store) would destroy the last good shard instead of
        just failing the save; staging the new snapshot off to the side first means the write is
        all-or-nothing and the previous ``items.jsonl`` is untouched by a failed attempt.
        """
        target = Path(root) if root is not None else self.root
        if target is None:
            raise ValueError("Substrate.save needs a root (none was set at construction)")
        target.mkdir(parents=True, exist_ok=True)
        dst = target / "items.jsonl"
        fd, tmp = tempfile.mkstemp(dir=str(target), prefix=".tmp-substrate-", suffix=".jsonl")
        try:
            with os.fdopen(fd, "w") as f:
                for item in self._items.values():
                    f.write(json.dumps(item.to_json()) + "\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, dst)
        except BaseException:
            # Serialization or the write failed partway through: drop the temp snapshot, leave the
            # previously-published items.jsonl (if any) exactly as it was.
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        self.root = target
        return str(target)

    def load(self, root: str | None = None) -> None:
        """Load items from ``{root}/items.jsonl`` into this shard, all-or-nothing.

        Every row is parsed into a private, freshly-built dict first; only once the ENTIRE file has
        parsed and validated cleanly does that dict become ``self._items``. A malformed or invalid row
        anywhere in the file raises -- naming the file, the 1-based line number, and a preview of the
        offending row -- and leaves whatever this shard held before the call completely untouched:
        never a partial store holding only the rows that happened to precede the bad one.
        """
        target = Path(root) if root is not None else self.root
        if target is None:
            raise ValueError("Substrate.load needs a root")
        path = target / "items.jsonl"
        loaded: dict[str, SubstrateItem] = {}
        with open(path) as f:
            for lineno, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    item = SubstrateItem.from_json(json.loads(line))
                except Exception as e:
                    raise ValueError(
                        f"Substrate.load: malformed row at {path}:{lineno}: {e} (row starts: {line[:80]!r})"
                    ) from e
                loaded[item.id] = item
        # Only now, with the whole file known-good, does it replace the live store.
        self._items = loaded
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
