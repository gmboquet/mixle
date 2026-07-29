"""E7 — cross-chain provenance receipt (work-plan §5, IC-2 / IC-5 / IC-13).

``decision_receipt`` walks a data -> posterior -> claim -> decision chain and emits an
:mod:`mixle.task.trace_record` (IC-5) whose every step carries a content hash for its own output and
its parent's hash as ``result["parent_hash"]`` -- a hashed, re-derivable lineage edge from raw data
through an inversion, an interpretation, and a decision. The posterior is *ingested* into a
:class:`~mixle.substrate.core.Substrate` (via :func:`~mixle.substrate.ingest.ingest_artifacts`) rather
than copied: the substrate item's payload references the artifact (an IC-13-shaped
``{artifact_ref, schema, grid, crs, units}`` record), the array bytes stay wherever the IC-2 artifact
already lives.

Repo-boundary note (see the PR body for the full explanation): this PR only touches the core ``mixle``
repository. ``mixle-pde`` (E2's ``io/artifacts.py`` -- IC-2) and ``mixle-mlops`` (E3/E4/E5) had not
landed on ``release/0.8.0`` as of this PR, so ``decision_receipt`` does not import either package.
Instead it verifies an on-disk posterior artifact directly and, when present,
checks its ``{posterior_ref}.json`` sidecar against a streaming SHA-256 of the
artifact bytes. ``claim`` and ``decision`` accept a plain dictionary or
dataclass and are encoded through a strict versioned canonical schema.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mixle.inference.production.provenance import build_header
from mixle.semantics import canonical_json
from mixle.substrate.core import Substrate
from mixle.substrate.ingest import ingest_artifacts
from mixle.task.trace_record import validate_trace_record

__all__ = ["ContentDigest", "decision_receipt", "content_edge_hash"]

_HEX64 = re.compile(r"^[0-9a-f]{64}$")

# Fallback schema tag used only when the posterior artifact carries no header of its own (e.g. a bare
# in-memory reference in a test/demo chain, with no IC-2 `{path}.json` sibling to read).
_FALLBACK_ARTIFACT_SCHEMA = "mixle_pde.field_posterior/v1"


@dataclass(frozen=True)
class ContentDigest:
    """A validated digest value, distinct from an ordinary 64-character string."""

    algorithm: str
    hexdigest: str

    def __post_init__(self) -> None:
        if self.algorithm != "sha256" or not _HEX64.fullmatch(self.hexdigest):
            raise ValueError("ContentDigest currently requires sha256 and 64 lowercase hexadecimal characters.")


def content_edge_hash(value: Any) -> str:
    """Hash a strict versioned canonical envelope for one inline lineage value."""
    if isinstance(value, ContentDigest):
        return value.hexdigest
    normalized = _to_canonical_value(value)
    payload = canonical_json({"schema": "mixle.content-edge/v1", "value": normalized})
    return hashlib.sha256(payload).hexdigest()


def _stringify_ref(ref: Any) -> str:
    if isinstance(ref, str):
        return ref
    if isinstance(ref, ContentDigest):
        return f"{ref.algorithm}:{ref.hexdigest}"
    return canonical_json(_to_canonical_value(ref)).decode("utf-8")


def _to_plain(value: Any) -> dict[str, Any]:
    """Normalize a claim/decision through the strict canonical receipt schema."""
    if isinstance(value, dict):
        out = dict(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        out = dict(dataclasses.asdict(value))
    elif not isinstance(value, dict):
        raise TypeError("claim and decision must be dictionaries or dataclass instances.")
    canonical_json(_to_canonical_value(out))
    return out


def _posterior_reference(posterior_ref: Any) -> tuple[str, dict[str, Any], str]:
    """Resolve and verify a posterior artifact, returning location, metadata, and byte digest."""
    if isinstance(posterior_ref, dict):
        ref = posterior_ref.get("artifact_ref") or posterior_ref.get("ref") or posterior_ref.get("path")
        if not isinstance(ref, (str, os.PathLike)):
            raise ValueError("posterior metadata must include a filesystem artifact_ref, ref, or path.")
        meta = dict(posterior_ref)
    elif isinstance(posterior_ref, (str, os.PathLike)):
        ref = posterior_ref
        meta = {}
    else:
        raise TypeError("posterior_ref must be a filesystem path or a metadata dictionary containing one.")

    path = Path(ref)
    if not path.is_file():
        raise FileNotFoundError(f"posterior artifact does not exist or is not a regular file: {path}")
    header_path = Path(f"{path}.json")
    if header_path.exists():
        try:
            sidecar = json.loads(header_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"posterior sidecar is unreadable or invalid JSON: {header_path}") from exc
        if not isinstance(sidecar, dict):
            raise ValueError(f"posterior sidecar must contain a JSON object: {header_path}")
        overlap = set(meta) & set(sidecar)
        mismatched = {key for key in overlap if meta[key] != sidecar[key]}
        if mismatched:
            raise ValueError(f"posterior metadata conflicts with sidecar fields: {sorted(mismatched)!r}")
        meta = {**sidecar, **meta}

    digest = _stream_sha256(path)
    declared = meta.get("content_hash")
    if declared is not None and _declared_hexdigest(declared) != digest:
        raise ValueError("posterior content_hash does not match the artifact bytes.")
    meta["content_hash"] = digest
    meta["digest_algorithm"] = "sha256"
    canonical_json(_to_canonical_value(meta))
    return str(path), meta, digest


def _registry_dir_for(substrate: Substrate, digest: str) -> Path:
    """A stable, per-digest micro-directory to host one ``manifest.json`` for `ingest_artifacts` --
    under the substrate's own root when it has one (persistent), else a shared scratch location. Never
    holds array bytes: only the small JSON manifest referencing the real artifact."""
    if not _HEX64.fullmatch(digest):
        raise ValueError("registry digest must be 64 lowercase hexadecimal characters.")
    if substrate.root is None:
        raise ValueError("rootless substrates require an isolated temporary registry.")
    d = Path(substrate.root) / "receipt_registry" / "field_posterior" / digest
    d.mkdir(mode=0o700, parents=True, exist_ok=True)
    return d


def _ingest_posterior(
    substrate: Substrate, ref_str: str, meta: dict[str, Any], digest: str
) -> tuple[str, dict[str, Any]]:
    """Ingest a reference to the posterior artifact via :func:`ingest_artifacts` (arrays stay behind the
    ref); returns ``(substrate_item_id, artifact_record)`` where ``artifact_record`` is the IC-13-shaped
    ``{artifact_ref, schema, grid, crs, units, content_hash}`` payload."""
    record = {
        "artifact_ref": ref_str,
        "schema": meta.get("schema", _FALLBACK_ARTIFACT_SCHEMA),
        "grid": meta.get("grid"),
        "crs": meta.get("crs"),
        "units": meta.get("units"),
        "content_hash": digest,
        "digest_algorithm": "sha256",
    }
    manifest = {
        "mixle_artifact": "field_posterior",
        "kind": "field_posterior",
        "parent": meta.get("provenance", {}).get("parent") if isinstance(meta.get("provenance"), dict) else None,
        "meta": record,
    }
    if substrate.root is None:
        with tempfile.TemporaryDirectory(prefix=f"mixle-receipt-{os.getpid()}-") as temporary_root:
            registry_dir = Path(temporary_root) / "field_posterior" / digest
            registry_dir.mkdir(mode=0o700, parents=True)
            _write_manifest_once(registry_dir / "manifest.json", manifest)
            ids = ingest_artifacts(substrate, str(registry_dir))
    else:
        registry_dir = _registry_dir_for(substrate, digest)
        _write_manifest_once(registry_dir / "manifest.json", manifest)
        ids = ingest_artifacts(substrate, str(registry_dir))
    if not ids:
        raise RuntimeError("posterior artifact manifest was not ingested.")
    item_id = ids[0] if ids else ""
    return item_id, record


def decision_receipt(
    *, dataset_ref: Any, posterior_ref: Any, claim: Any, decision: Any, substrate: Substrate
) -> dict[str, Any]:
    """Build an IC-5 trace record for one data -> posterior -> claim -> decision chain.

    ``dataset_ref`` is canonically fingerprinted. ``posterior_ref`` must resolve
    to an accessible file whose bytes are hashed and checked against any
    declared sidecar digest. ``claim`` and ``decision`` are canonical
    dictionaries or dataclasses. The posterior is ingested into ``substrate``
    as a referenced (not copied) artifact.

    Returns a dict that satisfies :func:`mixle.task.trace_record.validate_trace_record`: every scalar
    output (the posterior, the claim, the decision) resolves to a hashed lineage edge whose ``content_hash``
    is independently re-derivable via :func:`content_edge_hash`, and whose ``parent_hash`` chains back to
    the previous edge.
    """
    data_hash = content_edge_hash(dataset_ref)

    ref_str, posterior_meta, posterior_hash = _posterior_reference(posterior_ref)
    substrate_item_id, artifact_record = _ingest_posterior(substrate, ref_str, posterior_meta, posterior_hash)

    claim_record = _to_plain(claim)
    claim_hash = content_edge_hash(claim_record)

    decision_record = _to_plain(decision)
    decision_hash = content_edge_hash(decision_record)

    lineage = [
        {"stage": "data", "content_hash": data_hash, "parent_hash": None},
        {
            "stage": "posterior",
            "content_hash": posterior_hash,
            "parent_hash": data_hash,
            "substrate_item_id": substrate_item_id,
            "artifact": artifact_record,
        },
        {"stage": "claim", "content_hash": claim_hash, "parent_hash": posterior_hash},
        {"stage": "decision", "content_hash": decision_hash, "parent_hash": claim_hash},
    ]

    header = build_header(decision_record, [data_hash, posterior_hash, claim_hash, decision_hash], final_loglik=None)

    steps: list[dict[str, Any]] = [
        {
            "tool": "dataset",
            "args": {"dataset_ref": _stringify_ref(dataset_ref)},
            "result": {"content_hash": data_hash},
            "model": None,
            "verdict": None,
        },
        {
            "tool": "run_inversion",
            "args": {"dataset_ref": _stringify_ref(dataset_ref)},
            "result": {
                "content_hash": posterior_hash,
                "parent_hash": data_hash,
                "posterior_ref": ref_str,
                "substrate_item_id": substrate_item_id,
                **artifact_record,
            },
            "model": posterior_meta.get("model"),
            "verdict": None,
        },
        {
            "tool": "interpret",
            "args": {"posterior_ref": ref_str},
            "result": {"content_hash": claim_hash, "parent_hash": posterior_hash, "claim": claim_record},
            "model": claim_record.get("model"),
            "verdict": None,
        },
        {
            "tool": "decide",
            "args": {"claim_hash": claim_hash},
            "result": {"content_hash": decision_hash, "parent_hash": claim_hash, "decision": decision_record},
            "model": decision_record.get("model"),
            "verdict": None,
        },
    ]

    receipt: dict[str, Any] = {
        "prompt": f"decision receipt for posterior {ref_str}",
        "steps": steps,
        "outcome": decision_record,
        "provenance": {
            "lineage": lineage,
            "header": header.to_dict(),
            "posterior_substrate_item": substrate_item_id,
            "created_at": time.time(),
        },
    }
    validate_trace_record(receipt)
    return receipt


def _stream_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    after = path.stat()
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        raise RuntimeError(f"posterior artifact changed while it was being hashed: {path}")
    return digest.hexdigest()


def _declared_hexdigest(value: Any) -> str:
    if isinstance(value, ContentDigest):
        return value.hexdigest
    if not isinstance(value, str):
        raise TypeError("declared content_hash must be a hexadecimal string or ContentDigest.")
    candidate = value.removeprefix("sha256:")
    if not _HEX64.fullmatch(candidate):
        raise ValueError("declared content_hash must be 64 lowercase hexadecimal characters.")
    return candidate


def _write_manifest_once(path: Path, manifest: dict[str, Any]) -> None:
    payload = canonical_json(_to_canonical_value(manifest))
    if path.exists():
        try:
            existing = path.read_bytes()
        except OSError as exc:
            raise RuntimeError(f"cannot verify existing receipt manifest: {path}") from exc
        if existing != payload:
            raise RuntimeError(f"receipt manifest collision for immutable digest directory: {path.parent.name}")
        return
    temporary = path.with_name(f".manifest.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise RuntimeError(f"receipt manifest collision for immutable digest directory: {path.parent.name}")
    finally:
        temporary.unlink(missing_ok=True)


def _to_canonical_value(value: Any) -> Any:
    if isinstance(value, ContentDigest):
        return {"algorithm": value.algorithm, "hexdigest": value.hexdigest}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _to_canonical_value(dataclasses.asdict(value))
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("receipt mappings require string keys.")
        return {key: _to_canonical_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_to_canonical_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"value of type {type(value).__name__} is not supported by the canonical receipt schema.")
