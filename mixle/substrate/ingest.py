"""Ingest adapters: pull the stores the ecosystem already has into the knowledge substrate.

The substrate does not want copies of everything -- it wants TYPED, PROVENANCED, RETRIEVABLE entries
pointing at what already exists. These adapters turn the three stores mixle already keeps into
:class:`~mixle.substrate.core.SubstrateItem` s:

  * ``ingest_documents`` -- raw text / passages -> ``kind="text"`` items (the RAG corpus).
  * ``ingest_artifacts`` -- a registry directory of deployed model/dataset artifacts (each a
    ``manifest.json``) -> ``kind="artifact"`` items whose text surface is the manifest summary and
    whose payload references the artifact path (so lineage + retrieval work without copying weights).
  * ``ingest_traces`` -- a harvested ``.jsonl`` (the ``/feedback`` / agent-trace format) ->
    ``kind="trace"`` items (input->answer pairs for retrieval and curriculum).

Every item carries provenance (source path, kind, ingest time) so the reasoner can cite where a piece
of knowledge came from.

Integrity, identity, and failure visibility (MXR-080-0268/0269): every function below returns an
:class:`IngestReceipt` rather than a bare ``list[str]`` of successful ids. Three problems this fixes
together, because they share one root cause -- nothing observed a per-input failure or recorded what
would let a LATER reader detect one:

* **Missing integrity data.** When a doc/record/trace's ``source`` (or an artifact's ``manifest.json``)
  names a real, readable file, its provenance now carries a full, algorithm-labelled ``content_hash``
  -- byte-for-byte the same digest :func:`~mixle.substrate.freshness.content_hash` produces, so
  :func:`~mixle.substrate.freshness.check_freshness` can actually detect a later byte change instead of
  silently having nothing recorded to compare against.
* **Duplicate records instead of revisions.** Every item's id is now derived deterministically from
  ``(kind, source, position)`` (see :func:`_stable_id`) rather than a fresh random UUID every call.
  Re-ingesting the SAME source reuses the SAME ids, so :meth:`~mixle.substrate.core.Substrate.put`'s
  replace-by-id semantics turn a repeat ingestion into a REVISION of the existing records (a bumped
  ``provenance["revision"]`` counter, see :func:`_put_revisioned`) instead of an ever-growing pile of
  duplicates that all claim to be the current copy.
* **Silent data loss.** A malformed line, a manifest that is valid JSON of the wrong shape (a list where
  an object was expected), or any other per-input failure used to be swallowed by a bare ``continue``
  (or, worse, allowed to raise and abort the WHOLE batch after earlier inputs had already been stored --
  MXR-080-0269's partial-write-then-crash). Every ``ingest_*`` function now validates each input's shape
  BEFORE constructing/storing anything for it, and wraps that input's handling so nothing it does can
  abort its siblings: one bad input becomes one named entry on the returned receipt's ``.failed``, and
  every other input in the same batch still lands. A caller can always tell "N succeeded, M silently
  failed" apart from "N+M succeeded" -- and can always tell exactly what happened to every single input,
  never just "the call didn't raise."
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mixle.substrate.core import Substrate, SubstrateItem
from mixle.substrate.freshness import content_hash


@dataclass
class IngestFailure:
    """One batch input that did NOT make it into the substrate (MXR-080-0268/0269): its position in
    the batch, the source it came from, and why. Every ``ingest_*`` function reports these on its
    returned :class:`IngestReceipt` instead of silently skipping the input -- "N succeeded, M silently
    failed" must never be indistinguishable from "N+M succeeded"."""

    index: int
    source: str
    error: str


class IngestReceipt(list):
    """The ids of every item actually ingested, in order -- so ``len(receipt)``, ``receipt[0]``,
    ``if receipt:``, ``receipt == [...]``, and ``for item_id in receipt:`` all mean exactly what they
    meant back when every ``ingest_*`` function returned a bare ``list[str]``. That backward
    compatibility is deliberate, not an oversight: unlike a fresh ``@dataclass`` receipt (e.g.
    :class:`~mixle.substrate.factuality.FactualityReceipt`), this one has EXISTING external callers
    that already index/measure it as a plain list of ids --
    :func:`mixle.reason.receipt._ingest_posterior`'s ``ids[0] if ids else ""``,
    :meth:`mixle.scientist.Scientist.learn`'s ``len(ingest_documents(...))`` -- and both keep working
    completely unchanged against this type.

    PLUS the structured per-input accounting a bare list could never carry (MXR-080-0268): ``.failed``
    names every input that did NOT make it in, and why (see :class:`IngestFailure`); ``.ok`` is
    ``True`` iff nothing failed. Before this fix, a malformed line or manifest was silently
    ``continue``-d past (or, for a wrong-shaped-but-valid JSON value, allowed to raise and abort
    everything after it -- MXR-080-0269) and only the ids of what DID succeed were ever returned. A
    caller could not tell "3 succeeded, 2 silently failed" from "5 succeeded, nothing lost." Now it
    always can, without losing any of the old call sites' simplicity.
    """

    def __init__(self, ids: Sequence[str] = (), failed: Sequence[IngestFailure] = ()) -> None:
        super().__init__(ids)
        self.failed: list[IngestFailure] = list(failed)

    @property
    def ok(self) -> bool:
        """``True`` iff every input in this batch was ingested -- no failures at all."""
        return not self.failed

    def merge(self, other: IngestReceipt) -> IngestReceipt:
        """Combine two receipts (:func:`ingest_file`'s ``.jsonl`` branch fans one file out to both
        :func:`ingest_documents` and :func:`ingest_records`): ids from both, in order, AND both
        ``.failed`` lists. Deliberately not done via ``+``/``+=`` -- plain ``list`` concatenation only
        extends the id storage, silently dropping whichever side's ``.failed`` entries were not
        ``self`` -- exactly the kind of silent loss this whole fix exists to prevent.
        """
        return IngestReceipt([*self, *other], [*self.failed, *other.failed])

    def __repr__(self) -> str:
        base = super().__repr__()
        if self.ok:
            return f"IngestReceipt({base})"
        return f"IngestReceipt({base}, failed={[f.error for f in self.failed]!r})"


def _stable_id(*parts: Any) -> str:
    """A deterministic 16-hex-char id derived from ``parts`` -- the same shape as
    :class:`~mixle.substrate.core.SubstrateItem`'s own default id (``uuid.uuid4().hex[:16]``, also 64
    bits), so this is not a weaker id namespace than the one already in use everywhere else in the
    substrate.

    The SAME ``parts`` always yield the SAME id, so re-ingesting the same ``(kind, source, position)``
    updates/revises the existing record (via :func:`_put_revisioned`, which relies on
    :meth:`~mixle.substrate.core.Substrate.put`'s replace-by-id semantics) instead of accumulating a
    fresh random-id duplicate every time (MXR-080-0268). Deliberately independent of the item's
    CONTENT: a changed byte range is exactly what a *revision* of the same logical item should capture
    (a new ``content_hash`` under the SAME id), not a new identity.
    """
    digest = hashlib.sha256("\x1f".join(str(p) for p in parts).encode()).hexdigest()
    return digest[:16]


def _source_content_hash(source: str) -> str | None:
    """The recorded ingest-time ``content_hash`` for ``source`` when it names a real, readable file --
    ``None`` when ``source`` is just a label (e.g. the ``"documents"``/``"records"`` defaults, or a
    caller-chosen name with no backing file) with nothing to hash. Delegates to
    :func:`~mixle.substrate.freshness.content_hash` so the digest this records is byte-for-byte the
    same algorithm/format :func:`~mixle.substrate.freshness.check_freshness` will later compare against
    (MXR-080-0268 completing the MXR-080-0267 contract end to end)."""
    try:
        if not Path(source).is_file():
            return None
    except (OSError, ValueError):
        return None
    return content_hash(source)


def _put_revisioned(
    substrate: Substrate,
    *,
    id_: str,
    kind: str,
    text: str,
    payload: dict[str, Any],
    provenance: dict[str, Any],
    scope: str,
    tags: list[str] | None = None,
    links: list[str] | None = None,
) -> str:
    """Store one item under the STABLE id ``id_`` (see :func:`_stable_id`), reusing
    :meth:`~mixle.substrate.core.Substrate.put`'s add-or-replace-by-id semantics to make repeat
    ingestion a REVISION rather than a duplicate (MXR-080-0268).

    If ``id_`` already names a stored item, ``provenance["revision"]`` is the existing item's revision
    plus one, and ``provenance["first_ingested_at"]`` carries forward the ORIGINAL ingest time (this
    call's own ``provenance["ingested_at"]``, if present, records when THIS revision landed). A fresh
    id starts at ``revision=1`` with ``first_ingested_at`` equal to this call's ``ingested_at``.
    """
    existing = substrate.get(id_)
    revision = 1
    first_ingested_at = provenance.get("ingested_at")
    if existing is not None:
        revision = int(existing.provenance.get("revision", 1)) + 1
        first_ingested_at = existing.provenance.get("first_ingested_at", first_ingested_at)
    full_provenance = {**provenance, "revision": revision, "first_ingested_at": first_ingested_at}
    item = SubstrateItem(
        id=id_,
        kind=kind,
        text=text,
        payload=payload,
        tags=tags or [],
        scope=scope,
        provenance=full_provenance,
        links=links or [],
    )
    return substrate.put(item)


def ingest_documents(
    substrate: Substrate,
    docs: Sequence[str | dict[str, Any]],
    *,
    source: str = "documents",
    scope: str = "local",
    indices: Sequence[int] | None = None,
) -> IngestReceipt:
    """Add text passages to the substrate as ``kind="text"`` items. Returns an :class:`IngestReceipt`
    (the new/revised item ids, plus any per-doc failures -- see its docstring; behaves exactly like the
    historical bare ``list[str]`` for every existing caller).

    Each doc is a string, or a ``{"text": ..., "tags": [...], "payload": {...}}`` dict for metadata.
    Anything else is reported on ``.failed`` rather than crashing the whole batch (MXR-080-0269).

    Stable, revisioned identity and integrity (MXR-080-0268): each doc's id is derived from
    ``(source, its position)`` -- ``indices[i]`` when the caller supplies one (so :func:`ingest_file`
    can pass through a ``.jsonl`` file's true line numbers even after splitting mixed content into
    docs/records), else the plain ``enumerate`` position. Re-ingesting the SAME ``source`` reuses the
    SAME ids, turning a repeat call into a revision of the existing records (see :func:`_put_revisioned`)
    rather than a fresh random-id duplicate every time. When ``source`` names a real, readable file,
    every doc's provenance also carries that file's full, algorithm-labelled ``content_hash`` (the exact
    digest :func:`~mixle.substrate.freshness.check_freshness` later compares against) -- ``source`` is
    often just a label with nothing to hash (e.g. the ``"documents"`` default), so no ``content_hash``
    is recorded for those.
    """
    ids: list[str] = []
    failed: list[IngestFailure] = []
    file_hash = _source_content_hash(source)
    for i, d in enumerate(docs):
        pos = indices[i] if indices is not None else i
        try:
            if isinstance(d, str):
                text, tags, payload = d, [], {}
            elif isinstance(d, dict):
                text, tags, payload = str(d.get("text", "")), list(d.get("tags", [])), dict(d.get("payload", {}))
            else:
                raise TypeError(f"doc must be a str or dict, got {type(d).__name__}")
            provenance: dict[str, Any] = {"source": source, "index": pos, "ingested_at": time.time()}
            if file_hash is not None:
                provenance["content_hash"] = file_hash
            item_id = _put_revisioned(
                substrate,
                id_=_stable_id("text", source, pos),
                kind="text",
                text=text,
                payload=payload,
                provenance=provenance,
                scope=scope,
                tags=tags,
            )
            ids.append(item_id)
        except Exception as e:  # noqa: BLE001 - one bad doc must never abort the batch (MXR-080-0269)
            failed.append(IngestFailure(index=pos, source=source, error=f"{type(e).__name__}: {e}"))
    return IngestReceipt(ids, failed)


def ingest_artifacts(substrate: Substrate, registry_root: str, *, scope: str = "local") -> IngestReceipt:
    """Index every deployed artifact under ``registry_root`` (dirs containing a ``manifest.json``).
    Returns an :class:`IngestReceipt` (behaves exactly like the historical bare ``list[str]``).

    The item's text surface is a human summary of the manifest (kind, io, meta); its payload REFERENCES
    the artifact directory (``{"ref": path}``) rather than copying it -- unchanged, since other readers
    (e.g. :func:`mixle.reason.receipt.decision_receipt`) depend on ``ref`` naming the directory, not the
    manifest file -- and provenance carries the manifest's lineage fields when present.

    Integrity, identity, and validation (MXR-080-0268/0269): provenance also carries the manifest
    FILE's full, algorithm-labelled ``content_hash`` (:func:`~mixle.substrate.freshness.content_hash`)
    and its own ``manifest_path``, for anyone who wants to verify it directly -- note that because
    ``payload["ref"]`` (a directory) is what
    :func:`~mixle.substrate.freshness.check_freshness` actually re-hashes, an artifact item's freshness
    verdict is UNVERIFIABLE by construction (a directory cannot be read as file bytes), which is the
    documented, intentional behavior of that check, not a gap in this one; recording the hash here is
    still correct and useful (forensic verification, and a future reader can compare it against a fresh
    hash of the same manifest file by hand). Each artifact's id is derived from its directory path (see
    :func:`_stable_id`), so re-ingesting the SAME artifact directory revises the existing record instead
    of duplicating it. A manifest that fails to read, fails to parse, or parses to valid JSON of the
    wrong shape (anything other than a JSON object) is reported on ``.failed`` -- checked BEFORE any
    ``.get()`` is called on it and BEFORE anything is stored for it -- rather than raising and aborting
    every artifact discovered after it (MXR-080-0269).
    """
    root = Path(registry_root)
    if not root.is_dir():
        return IngestReceipt()
    ids: list[str] = []
    failed: list[IngestFailure] = []
    manifest_paths = sorted(root.rglob("manifest.json"))
    for i, manifest_path in enumerate(manifest_paths):
        try:
            try:
                manifest = json.loads(manifest_path.read_text())
            except OSError as e:
                raise ValueError(f"could not read {manifest_path}: {e}") from e
            except json.JSONDecodeError as e:
                raise ValueError(f"malformed JSON in {manifest_path}: {e}") from e
            if not isinstance(manifest, dict):
                raise ValueError(f"{manifest_path} must contain a JSON object, got {type(manifest).__name__}")

            adir = manifest_path.parent
            meta = manifest.get("meta", {})
            meta = meta if isinstance(meta, dict) else {}
            summary = _manifest_summary(adir.name, manifest, meta)
            digest = content_hash(str(manifest_path))
            provenance = {
                "source": "registry",
                "path": str(adir),
                "manifest_path": str(manifest_path),
                "artifact_kind": manifest.get("mixle_artifact") or manifest.get("kind"),
                "parent": manifest.get("parent") or (meta.get("parent") if isinstance(meta, dict) else None),
                "ingested_at": time.time(),
            }
            if digest is not None:
                provenance["content_hash"] = digest
            item_id = _put_revisioned(
                substrate,
                id_=_stable_id("artifact", str(adir)),
                kind="artifact",
                text=summary,
                payload={"ref": str(adir), "manifest": manifest},
                provenance=provenance,
                scope=scope,
                tags=[str(k) for k in meta] if isinstance(meta, dict) else [],
            )
            ids.append(item_id)
        except Exception as e:  # noqa: BLE001 - one bad manifest must never abort the batch (MXR-080-0269)
            failed.append(IngestFailure(index=i, source=str(manifest_path), error=f"{type(e).__name__}: {e}"))
    return IngestReceipt(ids, failed)


def ingest_traces(
    substrate: Substrate, jsonl_path: str, *, source: str | None = None, scope: str = "local"
) -> IngestReceipt:
    """Index a harvested ``.jsonl`` of ``{"input": ..., "answer"/"label"/"call": ...}`` rows as traces.
    Returns an :class:`IngestReceipt` (behaves exactly like the historical bare ``list[str]``).

    Integrity, identity, and validation (MXR-080-0268/0269): every row's provenance carries the source
    file's full, algorithm-labelled ``content_hash`` (one read of the whole file up front, reused for
    every row -- all rows from the same file share the same recorded hash, since a byte anywhere in the
    file invalidates every row's "as ingested" guarantee equally). Each row's id is derived from
    ``(source, its line number)``, so re-ingesting the SAME file revises the existing per-row records
    instead of duplicating them. A line that fails to parse, or parses to valid JSON that is not a JSON
    object (the documented row shape requires ``.get()``-able fields), is reported on ``.failed`` --
    checked BEFORE any ``.get()`` is called on it -- rather than raising and aborting every row after it
    in the file (MXR-080-0269, the exact "trace code assumes every row is a mapping" bug).
    """
    path = Path(jsonl_path)
    if not path.exists():
        return IngestReceipt()
    src = source or str(path)
    digest = content_hash(str(path))
    ids: list[str] = []
    failed: list[IngestFailure] = []
    with open(path) as f:
        for i, raw_line in enumerate(f):
            line = raw_line.strip()
            if not line:
                continue
            try:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"malformed JSON at line {i + 1}: {e}") from e
                if not isinstance(row, dict):
                    raise ValueError(f"line {i + 1} must be a JSON object, got {type(row).__name__}")

                answer = row.get("answer", row.get("label", row.get("call")))
                provenance: dict[str, Any] = {"source": src, "row": i, "ingested_at": time.time()}
                if digest is not None:
                    provenance["content_hash"] = digest
                item_id = _put_revisioned(
                    substrate,
                    id_=_stable_id("trace", src, i),
                    kind="trace",
                    text=f"{_stringify(row.get('input'))} => {_stringify(answer)}",
                    payload=row,
                    provenance=provenance,
                    scope=scope,
                )
                ids.append(item_id)
            except Exception as e:  # noqa: BLE001 - one bad row must never abort the batch (MXR-080-0269)
                failed.append(IngestFailure(index=i, source=src, error=f"{type(e).__name__}: {e}"))
    return IngestReceipt(ids, failed)


def ingest_records(
    substrate: Substrate,
    records: Sequence[Any],
    *,
    source: str = "records",
    scope: str = "local",
    text_fields: Sequence[str] | None = None,
    indices: Sequence[int] | None = None,
) -> IngestReceipt:
    """Add a sequence of records (dicts or tuples) to the substrate as ``kind="record"`` items. Returns
    an :class:`IngestReceipt` (behaves exactly like the historical bare ``list[str]``).

    Each record's payload is stored structured; its retrievable text surface is the ``text_fields``
    (for dict records) joined, else the whole serialized record -- so records are searchable by content.

    Stable, revisioned identity and integrity (MXR-080-0268), mirroring :func:`ingest_documents`: each
    record's id is derived from ``(source, its position)`` -- ``indices[i]`` when supplied (see
    :func:`ingest_file`), else the plain position -- so re-ingesting the SAME source revises the
    existing records instead of duplicating them, and a real, readable ``source`` file's
    ``content_hash`` is recorded on every record's provenance. One record's storage failure (e.g. a
    rejected secret under ``secret_policy="reject"``) is reported on ``.failed`` rather than aborting
    the rest of the batch.
    """
    ids: list[str] = []
    failed: list[IngestFailure] = []
    file_hash = _source_content_hash(source)
    for i, rec in enumerate(records):
        pos = indices[i] if indices is not None else i
        try:
            if isinstance(rec, dict):
                payload = dict(rec)
                surface = " ".join(str(rec[f]) for f in (text_fields or rec) if f in rec)
            else:
                payload = {"values": list(rec) if isinstance(rec, (list, tuple)) else [rec]}
                surface = " ".join(_stringify(v) for v in payload["values"])
            provenance: dict[str, Any] = {"source": source, "index": pos, "ingested_at": time.time()}
            if file_hash is not None:
                provenance["content_hash"] = file_hash
            item_id = _put_revisioned(
                substrate,
                id_=_stable_id("record", source, pos),
                kind="record",
                text=surface,
                payload=payload,
                provenance=provenance,
                scope=scope,
            )
            ids.append(item_id)
        except Exception as e:  # noqa: BLE001 - one bad record must never abort the batch (MXR-080-0269)
            failed.append(IngestFailure(index=pos, source=source, error=f"{type(e).__name__}: {e}"))
    return IngestReceipt(ids, failed)


def ingest_file(
    substrate: Substrate,
    path: str,
    *,
    kind: str | None = None,
    source: str | None = None,
    scope: str = "local",
) -> IngestReceipt:
    """Ingest a data file into the substrate. Format inferred from the extension unless ``kind`` forces
    it. Returns an :class:`IngestReceipt` (behaves exactly like the historical bare ``list[str]``).

    ``.txt``/``.md`` -> one ``text`` item per non-empty line; ``.jsonl`` -> one item per JSON line
    (a string / ``{"text": ...}`` becomes a text item, any other object a record item); ``.csv`` ->
    one ``record`` item per row keyed by the header. ``source`` defaults to the file path, so every
    resulting item picks up that file's ``content_hash`` and a stable, revisioned identity (see
    :func:`ingest_documents`/:func:`ingest_records`) automatically.

    A malformed ``.jsonl`` line is reported on the returned receipt's ``.failed`` (naming its 1-based
    line number) rather than silently dropped (MXR-080-0268) or allowed to abort every line after it
    (MXR-080-0269); the ``.jsonl`` branch preserves each row's TRUE original line number as its stable
    identity even after splitting mixed content into a docs sub-batch and a records sub-batch (via
    :func:`IngestReceipt.merge`, which -- unlike plain list concatenation -- keeps both sub-batches'
    ``.failed`` entries instead of silently dropping one side's).
    """
    p = Path(path)
    if not p.exists():
        return IngestReceipt()
    src = source or str(p)
    fmt = (kind or p.suffix.lstrip(".")).lower()

    if fmt in ("txt", "md", "text"):
        lines = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
        return ingest_documents(substrate, lines, source=src, scope=scope)

    if fmt in ("jsonl", "ndjson"):
        docs: list[str | dict[str, Any]] = []
        doc_lines: list[int] = []
        recs: list[dict[str, Any]] = []
        rec_lines: list[int] = []
        failed: list[IngestFailure] = []
        with open(p) as f:
            for lineno, raw_line in enumerate(f):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    failed.append(
                        IngestFailure(index=lineno, source=src, error=f"malformed JSON at line {lineno + 1}: {e}")
                    )
                    continue
                if isinstance(row, str) or (isinstance(row, dict) and set(row) <= {"text", "tags", "payload"}):
                    docs.append(row)
                    doc_lines.append(lineno)
                elif isinstance(row, dict):
                    recs.append(row)
                    rec_lines.append(lineno)
                else:
                    docs.append(_stringify(row))
                    doc_lines.append(lineno)
        receipt = IngestReceipt(failed=failed)
        if docs:
            receipt = receipt.merge(ingest_documents(substrate, docs, source=src, scope=scope, indices=doc_lines))
        if recs:
            receipt = receipt.merge(ingest_records(substrate, recs, source=src, scope=scope, indices=rec_lines))
        return receipt

    if fmt == "csv":
        import csv

        with open(p, newline="") as f:
            rows = list(csv.DictReader(f))
        return ingest_records(substrate, rows, source=src, scope=scope)

    raise ValueError(f"unsupported file format {fmt!r}; use txt/md, jsonl, or csv (or pass kind=)")


def _manifest_summary(name: str, manifest: dict[str, Any], meta: dict[str, Any]) -> str:
    parts = [name]
    kind = manifest.get("mixle_artifact") or manifest.get("kind")
    if kind:
        parts.append(str(kind))
    if isinstance(meta, dict):
        for key in ("solve", "regress", "multilabel", "structured", "task"):
            if key in meta:
                parts.append(key)
    io = manifest.get("io")
    if isinstance(io, dict) and io.get("kind"):
        parts.append(str(io["kind"]))
    return " ".join(parts)


def _stringify(v: Any) -> str:
    if isinstance(v, str):
        return v
    try:
        return json.dumps(v)
    except Exception:  # noqa: BLE001
        return str(v)
