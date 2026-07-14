"""M0b: core substrate -> IC-13 bundle compatibility bridge (notes/exec/workstream-M.md).

Exercises `substrate_item_to_knowledge_dict` and `ContextPacket.to_knowledge_bundle_dict` against
`mixle_knowledge.contracts.KnowledgeBundle` (IC-13). Core itself never imports `mixle_knowledge` --
only this test does, to validate the dependency-free dicts the bridge produces.
"""

import pytest

pydantic = pytest.importorskip("pydantic")
from pydantic import ValidationError  # noqa: E402

knowledge_contracts = pytest.importorskip("mixle_knowledge.contracts")
KnowledgeBundle = knowledge_contracts.KnowledgeBundle

from mixle.substrate.context import (  # noqa: E402
    PROPERTY_GRAPH_SCHEMA,
    SPATIAL_MEDIA_SCHEMA,
    TYPED_TABLE_SCHEMA,
    ContextPacket,
    substrate_item_to_knowledge_dict,
)
from mixle.substrate.core import SubstrateItem  # noqa: E402


def _graph_item():
    return SubstrateItem(
        id="graph-1",
        kind="graph",
        text="deposit graph",
        payload={
            "nodes": [{"id": "n1", "type": "deposit"}, {"id": "n2", "type": "assay"}],
            "edges": [{"id": "e1", "source": "n1", "target": "n2", "type": "refines"}],
        },
        provenance={"source": "geo-model"},
        scope="local",
        tags=["geology"],
        links=["table-1"],
    )


def _table_item():
    return SubstrateItem(
        id="table-1",
        kind="record",
        text="assay table",
        payload={
            "primary_key": ["sample_id"],
            "columns": [
                {"name": "sample_id", "type": "string"},
                {"name": "cu_pct", "type": "float", "unit": "%"},
            ],
            "rows": [{"sample_id": "A-1", "cu_pct": 1.8}],
        },
        provenance={"source": "lab"},
        scope="team-geo",
        tags=["assay"],
        links=["graph-1"],
    )


def _image_item():
    return SubstrateItem(
        id="image-1",
        kind="image",
        text="site raster",
        payload={
            "ref": "blob/sha256/" + "3" * 64,
            "crs": "EPSG:32611",
            "extent": [500000, 4100000, 501000, 4101000],
            "pixel_to_crs": [1, 0, 500000, 0, -1, 4101000],
        },
        provenance={"source": "survey"},
        scope="local",
        tags=[],
        links=[],
    )


def test_graph_item_round_trips_through_knowledge_item_validation():
    d = substrate_item_to_knowledge_dict(_graph_item())
    assert d["schema_uri"] == PROPERTY_GRAPH_SCHEMA
    assert d["kind"] == "artifact" and d["modality"] == "graph"
    item = knowledge_contracts.KnowledgeItem.model_validate(d)
    assert item.payload == _graph_item().payload
    assert item.relations[0].target_id == "table-1"
    assert item.access.scope == "private"
    assert len(item.content_hash) == 64


def test_table_item_infers_typed_table_schema_from_shape():
    d = substrate_item_to_knowledge_dict(_table_item())
    assert d["schema_uri"] == TYPED_TABLE_SCHEMA
    item = knowledge_contracts.KnowledgeItem.model_validate(d)
    assert item.payload["rows"][0]["cu_pct"] == 1.8
    assert item.access.scope == "team" and item.access.teams == ["team-geo"]


def test_image_item_splits_ref_into_artifact_ref_and_keeps_spatial_payload():
    d = substrate_item_to_knowledge_dict(_image_item())
    assert d["schema_uri"] == SPATIAL_MEDIA_SCHEMA
    assert d["artifact_ref"] == "substrate://artifact/blob/sha256/" + "3" * 64
    assert "ref" not in d["payload"]
    item = knowledge_contracts.KnowledgeItem.model_validate(d)
    assert item.payload["crs"] == "EPSG:32611"
    assert item.artifact_ref == d["artifact_ref"]


def test_explicit_schema_uri_override_wins():
    d = substrate_item_to_knowledge_dict(_graph_item(), schema_uri="mixle://schema/substrate-item/1")
    assert d["schema_uri"] == "mixle://schema/substrate-item/1"


def test_bundle_round_trips_graph_table_image_without_flattening():
    packet = ContextPacket(
        task="rank targets",
        items=[_graph_item(), _table_item(), _image_item()],
        scores=[0.9, 0.8, 0.7],
    )
    gap = {
        "id": "gap-assay",
        "question": "Find the missing Cu assay for A-2",
        "required_schema": {"type": "number"},
        "acceptance_criteria": ["verified lab result"],
    }
    bundle_dict = packet.to_knowledge_bundle_dict(
        id="bundle-1",
        project_id="p",
        target_kind="model",
        target_id="model-a",
        expected_output_schema={"type": "object"},
        gaps=[gap],
    )

    bundle = KnowledgeBundle.model_validate(bundle_dict)
    assert len(bundle.items) == 3
    by_id = {item.id: item for item in bundle.items}

    # exact payload/links/scope preserved -- nothing flattened into rendered text
    assert by_id["graph-1"].payload["edges"][0]["id"] == "e1"
    assert by_id["graph-1"].relations[0].target_id == "table-1"
    assert by_id["table-1"].payload["columns"][1]["type"] == "float"
    assert by_id["table-1"].access.teams == ["team-geo"]
    assert by_id["image-1"].artifact_ref.endswith("3" * 64)
    assert by_id["image-1"].payload["extent"] == [500000, 4100000, 501000, 4101000]

    assert bundle.gaps[0].id == "gap-assay"
    assert bundle.gaps[0].status.value == "open"

    # renderings carry the legacy text view only -- it may differ freely without moving item hashes
    hashes_before = {item.id: item.content_hash for item in bundle.items}
    legacy_text_first = bundle.renderings["legacy_text"]["text"]
    bundle_dict_again = packet.to_knowledge_bundle_dict(
        id="bundle-1", project_id="p", target_kind="model", target_id="model-a"
    )
    bundle_again = KnowledgeBundle.model_validate(bundle_dict_again)
    hashes_after = {item.id: item.content_hash for item in bundle_again.items}
    assert hashes_after == hashes_before
    assert isinstance(legacy_text_first, str) and legacy_text_first


def test_bundle_omits_legacy_to_knowledge_dict_from_canonical_items():
    # to_knowledge_dict (legacy) stays available and unchanged, but is not what feeds the bundle.
    packet = ContextPacket(task="t", items=[_graph_item()], scores=[1.0])
    legacy = packet.to_knowledge_dict(id="x", project_id="p", target_kind="model")
    assert "items" not in legacy  # the flattened legacy view has no canonical per-item structure
    bundle_dict = packet.to_knowledge_bundle_dict(id="x", project_id="p", target_kind="model")
    assert bundle_dict["items"][0]["payload"]["nodes"][0]["id"] == "n1"


def test_item_with_neither_payload_nor_ref_still_validates_with_empty_payload():
    item = SubstrateItem(id="bare", kind="text", text="just text", payload={})
    d = substrate_item_to_knowledge_dict(item)
    assert d["payload"] == {} and d["artifact_ref"] is None
    knowledge_contracts.KnowledgeItem.model_validate(d)  # does not raise


def test_bad_graph_payload_fails_downstream_validation_not_silently():
    item = SubstrateItem(
        id="bad-graph",
        kind="graph",
        payload={
            "nodes": [{"id": "n1", "type": "deposit"}],
            "edges": [{"id": "e", "source": "n1", "target": "missing", "type": "near"}],
        },
    )
    d = substrate_item_to_knowledge_dict(item)
    with pytest.raises(ValidationError):
        knowledge_contracts.KnowledgeItem.model_validate(d)
