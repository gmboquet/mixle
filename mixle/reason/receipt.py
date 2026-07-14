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
Instead it accepts already-content-hashed ``*_ref`` handles (the same opaque-string convention IC-3's
``run_inversion``/``query_posterior`` use) and, when ``posterior_ref`` names an on-disk IC-2 artifact
(a ``{posterior_ref}.json`` sibling exists, carrying the frozen ``HEADER_KEYS``), reads that header
directly (plain ``json.load`` -- no ``mixle_pde`` import) rather than depending on a real
``load_posterior``. ``claim``/``decision`` accept a plain dict, any dataclass, or an ad hoc object
(e.g. a ``mixle.reason.language_bridge.Claim``); each is normalized to a dict before hashing.
"""

from __future__ import annotations

import dataclasses
import json
import re
import tempfile
import time
from pathlib import Path
from typing import Any

from mixle.data.hashing import dataset_hash
from mixle.inference.production.provenance import build_header
from mixle.substrate.core import Substrate
from mixle.substrate.ingest import ingest_artifacts
from mixle.task.trace_record import validate_trace_record

__all__ = ["decision_receipt", "content_edge_hash"]

_HEX64 = re.compile(r"^[0-9a-f]{64}$")

# Fallback schema tag used only when the posterior artifact carries no header of its own (e.g. a bare
# in-memory reference in a test/demo chain, with no IC-2 `{path}.json` sibling to read).
_FALLBACK_ARTIFACT_SCHEMA = "mixle_pde.field_posterior/v1"


def content_edge_hash(value: Any) -> str:
    """The hash for one lineage edge: pass an already-hashed hex digest through unchanged (the IC-2/IC-3
    ``*_ref`` convention), else fingerprint ``value`` with :func:`mixle.data.hashing.dataset_hash` (a
    stable sha256 over a canonical byte encoding) so the edge is deterministic and independently
    re-derivable from the same inline payload."""
    if isinstance(value, str) and _HEX64.match(value):
        return value
    return dataset_hash([value])


def _stringify_ref(ref: Any) -> str:
    if isinstance(ref, str):
        return ref
    return json.dumps(ref, sort_keys=True, default=str)


def _to_plain(value: Any) -> dict[str, Any]:
    """Normalize a claim/decision payload into a JSON-friendly dict so it can be hashed and embedded."""
    if isinstance(value, dict):
        return dict(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        out = dict(dataclasses.asdict(value))
        text_fn = getattr(value, "text", None)
        if callable(text_fn):
            try:
                out.setdefault("text", text_fn())
            except Exception:  # noqa: BLE001 - best-effort text surface only
                pass
        return out
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return {"value": value}


def _posterior_reference(posterior_ref: Any) -> tuple[str, dict[str, Any]]:
    """Resolve ``posterior_ref`` (a bare string/path/hash, an IC-2-header-shaped dict, or an ad hoc
    object) into ``(ref_str, header_meta)``. When ``posterior_ref`` is a path with an on-disk IC-2
    sibling ``{posterior_ref}.json``, that header (frozen ``HEADER_KEYS``: schema/content_hash/crs/
    grid/units/provenance/created) is read directly -- no ``mixle_pde`` import required."""
    if isinstance(posterior_ref, dict):
        ref = (
            posterior_ref.get("artifact_ref")
            or posterior_ref.get("ref")
            or posterior_ref.get("path")
            or posterior_ref.get("content_hash")
        )
        return (str(ref) if ref is not None else _stringify_ref(posterior_ref)), dict(posterior_ref)
    if isinstance(posterior_ref, str):
        header_path = Path(f"{posterior_ref}.json")
        if header_path.is_file():
            try:
                meta = json.loads(header_path.read_text())
            except Exception:  # noqa: BLE001 - an unreadable/corrupt header falls back to a bare ref
                meta = {}
            if isinstance(meta, dict):
                return posterior_ref, meta
        return posterior_ref, {}
    meta = {
        k: getattr(posterior_ref, k)
        for k in ("schema", "grid", "crs", "units", "content_hash", "provenance")
        if hasattr(posterior_ref, k)
    }
    ref = getattr(posterior_ref, "path", None) or getattr(posterior_ref, "ref", None) or posterior_ref
    return _stringify_ref(ref), meta


def _registry_dir_for(substrate: Substrate, digest: str) -> Path:
    """A stable, per-digest micro-directory to host one ``manifest.json`` for `ingest_artifacts` --
    under the substrate's own root when it has one (persistent), else a shared scratch location. Never
    holds array bytes: only the small JSON manifest referencing the real artifact."""
    base = substrate.root if substrate.root is not None else Path(tempfile.gettempdir()) / "mixle_receipt_registry"
    d = Path(base) / "field_posterior" / digest[:16]
    d.mkdir(parents=True, exist_ok=True)
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
    }
    registry_dir = _registry_dir_for(substrate, digest)
    manifest = {
        "mixle_artifact": "field_posterior",
        "kind": "field_posterior",
        "parent": meta.get("provenance", {}).get("parent") if isinstance(meta.get("provenance"), dict) else None,
        "meta": record,
    }
    (registry_dir / "manifest.json").write_text(json.dumps(manifest))
    ids = ingest_artifacts(substrate, str(registry_dir))
    item_id = ids[0] if ids else ""
    return item_id, record


def decision_receipt(
    *, dataset_ref: Any, posterior_ref: Any, claim: Any, decision: Any, substrate: Substrate
) -> dict[str, Any]:
    """Build an IC-5 trace record for one data -> posterior -> claim -> decision chain.

    ``dataset_ref`` / ``posterior_ref`` are content-hashed handles (the IC-2/IC-3 ``*_ref`` convention;
    a bare string is treated as already-hashed, anything else is fingerprinted). ``claim`` / ``decision``
    are the interpretation/decision payloads (a dict, a dataclass, or an ad hoc object such as a
    ``language_bridge.Claim``), normalized to a dict before hashing. The posterior is ingested into
    ``substrate`` as a referenced (not copied) artifact.

    Returns a dict that satisfies :func:`mixle.task.trace_record.validate_trace_record`: every scalar
    output (the posterior, the claim, the decision) resolves to a hashed lineage edge whose ``content_hash``
    is independently re-derivable via :func:`content_edge_hash`, and whose ``parent_hash`` chains back to
    the previous edge.
    """
    data_hash = content_edge_hash(dataset_ref)

    ref_str, posterior_meta = _posterior_reference(posterior_ref)
    posterior_hash = posterior_meta.get("content_hash") or content_edge_hash(
        {"ref": ref_str, **{k: v for k, v in posterior_meta.items() if k != "content_hash"}}
    )
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
