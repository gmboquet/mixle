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
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from mixle.substrate.core import Substrate, SubstrateItem


def ingest_documents(
    substrate: Substrate, docs: Sequence[str | dict[str, Any]], *, source: str = "documents", scope: str = "local"
) -> list[str]:
    """Add text passages to the substrate as ``kind="text"`` items. Returns the new item ids.

    Each doc is a string, or a ``{"text": ..., "tags": [...], "payload": {...}}`` dict for metadata.
    """
    ids = []
    for i, d in enumerate(docs):
        if isinstance(d, str):
            text, tags, payload = d, [], {}
        else:
            text, tags, payload = str(d.get("text", "")), list(d.get("tags", [])), dict(d.get("payload", {}))
        item = SubstrateItem(
            kind="text",
            text=text,
            payload=payload,
            tags=tags,
            scope=scope,
            provenance={"source": source, "index": i, "ingested_at": time.time()},
        )
        ids.append(substrate.put(item))
    return ids


def ingest_artifacts(substrate: Substrate, registry_root: str, *, scope: str = "local") -> list[str]:
    """Index every deployed artifact under ``registry_root`` (dirs containing a ``manifest.json``).

    The item's text surface is a human summary of the manifest (kind, io, meta); its payload REFERENCES
    the artifact directory (``{"ref": path}``) rather than copying it, and provenance carries the
    manifest's lineage fields when present.
    """
    root = Path(registry_root)
    ids: list[str] = []
    if not root.is_dir():
        return ids
    for manifest_path in sorted(root.rglob("manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception:  # noqa: BLE001 - an unreadable manifest is skipped, not fatal
            continue
        adir = manifest_path.parent
        meta = manifest.get("meta", {}) if isinstance(manifest, dict) else {}
        summary = _manifest_summary(adir.name, manifest, meta)
        item = SubstrateItem(
            kind="artifact",
            text=summary,
            payload={"ref": str(adir), "manifest": manifest},
            tags=[str(k) for k in meta] if isinstance(meta, dict) else [],
            scope=scope,
            provenance={
                "source": "registry",
                "path": str(adir),
                "artifact_kind": manifest.get("mixle_artifact") or manifest.get("kind"),
                "parent": manifest.get("parent") or (meta.get("parent") if isinstance(meta, dict) else None),
                "ingested_at": time.time(),
            },
        )
        ids.append(substrate.put(item))
    return ids


def ingest_traces(
    substrate: Substrate, jsonl_path: str, *, source: str | None = None, scope: str = "local"
) -> list[str]:
    """Index a harvested ``.jsonl`` of ``{"input": ..., "answer"/"label"/"call": ...}`` rows as traces."""
    path = Path(jsonl_path)
    ids: list[str] = []
    if not path.exists():
        return ids
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            answer = row.get("answer", row.get("label", row.get("call")))
            item = SubstrateItem(
                kind="trace",
                text=f"{_stringify(row.get('input'))} => {_stringify(answer)}",
                payload=row,
                scope=scope,
                provenance={"source": source or str(path), "row": i, "ingested_at": time.time()},
            )
            ids.append(substrate.put(item))
    return ids


def ingest_records(
    substrate: Substrate,
    records: Sequence[Any],
    *,
    source: str = "records",
    scope: str = "local",
    text_fields: Sequence[str] | None = None,
) -> list[str]:
    """Add a sequence of records (dicts or tuples) to the substrate as ``kind="record"`` items.

    Each record's payload is stored structured; its retrievable text surface is the ``text_fields``
    (for dict records) joined, else the whole serialized record -- so records are searchable by content.
    """
    ids: list[str] = []
    for i, rec in enumerate(records):
        if isinstance(rec, dict):
            payload = dict(rec)
            surface = " ".join(str(rec[f]) for f in (text_fields or rec) if f in rec)
        else:
            payload = {"values": list(rec) if isinstance(rec, (list, tuple)) else [rec]}
            surface = " ".join(_stringify(v) for v in payload["values"])
        item = SubstrateItem(
            kind="record",
            text=surface,
            payload=payload,
            scope=scope,
            provenance={"source": source, "index": i, "ingested_at": time.time()},
        )
        ids.append(substrate.put(item))
    return ids


def ingest_file(
    substrate: Substrate,
    path: str,
    *,
    kind: str | None = None,
    source: str | None = None,
    scope: str = "local",
) -> list[str]:
    """Ingest a data file into the substrate. Format inferred from the extension unless ``kind`` forces it.

    ``.txt``/``.md`` -> one ``text`` item per non-empty line; ``.jsonl`` -> one item per JSON line
    (a string / ``{"text": ...}`` becomes a text item, any other object a record item); ``.csv`` ->
    one ``record`` item per row keyed by the header. ``source`` defaults to the file path.
    """
    p = Path(path)
    if not p.exists():
        return []
    src = source or str(p)
    fmt = (kind or p.suffix.lstrip(".")).lower()

    if fmt in ("txt", "md", "text"):
        lines = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
        return ingest_documents(substrate, lines, source=src, scope=scope)

    if fmt in ("jsonl", "ndjson"):
        docs: list[str | dict[str, Any]] = []
        recs: list[dict[str, Any]] = []
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:  # noqa: BLE001 - skip a malformed line, do not fail the file
                    continue
                if isinstance(row, str) or (isinstance(row, dict) and set(row) <= {"text", "tags", "payload"}):
                    docs.append(row)
                elif isinstance(row, dict):
                    recs.append(row)
                else:
                    docs.append(_stringify(row))
        ids = ingest_documents(substrate, docs, source=src, scope=scope) if docs else []
        ids += ingest_records(substrate, recs, source=src, scope=scope) if recs else []
        return ids

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
