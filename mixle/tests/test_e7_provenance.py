"""E7 -- cross-chain provenance receipt.

Repo-boundary note: the work order's literal DoD command targets
``$MX/mixle-mlops/tests/test_e7_provenance.py`` (a different repository, gmboquet/mixle-mlops), which
had not landed E3/E4/E5 on ``release/0.8.0`` as of this PR and is out of scope for a PR against
``gmboquet/mixle``. This test exercises the exact DoD scenario -- a synthetic data -> invert ->
interpret -> decide chain, every scalar resolving to a hashed, re-derivable lineage edge, and
``validate_trace_record`` passing -- entirely with core-``mixle`` fixtures (a real
``mixle.reason.language_bridge.Claim`` for the interpretation step, a hand-built IC-2-header-shaped
dict for the posterior step, standing in for the not-yet-landed ``mixle_pde`` artifact).
"""

import hashlib
import json

from mixle.reason.language_bridge import Claim
from mixle.reason.receipt import content_edge_hash, decision_receipt
from mixle.substrate.core import Substrate
from mixle.task.trace_record import STEP_KEYS, validate_trace_record


def _synthetic_dataset_ref():
    return {"kind": "gravity", "n_stations": 5, "values": [1.1, 2.2, 3.3, 4.4, 5.5]}


def _synthetic_posterior_header(tmp_path):
    """A hand-built header shaped exactly like IC-2's frozen ``{schema, content_hash, crs, grid, units,
    provenance, created}`` so `decision_receipt` reads it as an on-disk artifact sibling, the same way
    it would read a real `mixle_pde.io.artifacts.save_posterior` output."""
    array_digest = hashlib.sha256(b"mean|cov|synthetic-field-posterior-arrays").hexdigest()
    header = {
        "schema": "mixle_pde.field_posterior/v1",
        "content_hash": array_digest,
        "crs": "EPSG:32611",
        "grid": {"shape": [4, 4, 4], "origin": [0.0, 0.0, 0.0], "spacing": [10.0, 10.0, 10.0]},
        "units": "kg/m^3",
        "provenance": {"modality": "gravity", "prior": "smooth"},
        "created": "2026-07-14T00:00:00Z",
    }
    path_prefix = tmp_path / "posterior_run_001"
    (tmp_path / "posterior_run_001.json").write_text(json.dumps(header))
    return str(path_prefix), array_digest


def test_decision_receipt_chain_is_hashed_and_validates(tmp_path):
    substrate = Substrate()

    dataset_ref = _synthetic_dataset_ref()
    posterior_ref, array_digest = _synthetic_posterior_header(tmp_path)
    claim = Claim(field="density", lo=2650.0, hi=2800.0)
    decision = {"action": "drill", "region": "target-7", "tonnage_ci": [120_000.0, 180_000.0]}

    receipt = decision_receipt(
        dataset_ref=dataset_ref, posterior_ref=posterior_ref, claim=claim, decision=decision, substrate=substrate
    )

    # the record shape is IC-5-valid
    validate_trace_record(receipt)
    assert len(receipt["steps"]) == 4
    for step in receipt["steps"]:
        assert set(STEP_KEYS) <= set(step)

    lineage = {edge["stage"]: edge for edge in receipt["provenance"]["lineage"]}
    assert set(lineage) == {"data", "posterior", "claim", "decision"}

    # every edge carries a content_hash, and it is re-derivable: re-running the exact same chain
    # yields byte-identical hashes at every stage (determinism, not just presence).
    for edge in lineage.values():
        assert isinstance(edge["content_hash"], str) and len(edge["content_hash"]) == 64

    receipt_again = decision_receipt(
        dataset_ref=dataset_ref, posterior_ref=posterior_ref, claim=claim, decision=decision, substrate=Substrate()
    )
    lineage_again = {edge["stage"]: edge for edge in receipt_again["provenance"]["lineage"]}
    for stage in ("data", "posterior", "claim", "decision"):
        assert lineage[stage]["content_hash"] == lineage_again[stage]["content_hash"]

    # the data hash is directly re-derivable from the raw input via the public helper
    assert lineage["data"]["content_hash"] == content_edge_hash(dataset_ref)

    # the posterior's IC-2 array digest is PRESERVED as the artifact digest, never recomputed
    assert lineage["posterior"]["content_hash"] == array_digest

    # each edge's parent_hash chains to the previous edge -- data -> posterior -> claim -> decision
    assert lineage["data"]["parent_hash"] is None
    assert lineage["posterior"]["parent_hash"] == lineage["data"]["content_hash"]
    assert lineage["claim"]["parent_hash"] == lineage["posterior"]["content_hash"]
    assert lineage["decision"]["parent_hash"] == lineage["claim"]["content_hash"]

    # the posterior was ingested into the substrate (referenced, not copied): the substrate item exists,
    # is a structured IC-13-shaped {artifact_ref, schema, grid, crs, units} record, and carries no raw
    # arrays -- only the reference and metadata.
    item_id = receipt["provenance"]["posterior_substrate_item"]
    item = substrate.get(item_id)
    assert item is not None and item.kind == "artifact"
    meta = item.payload["manifest"]["meta"]
    assert meta["artifact_ref"] == posterior_ref
    assert meta["content_hash"] == array_digest
    assert meta["crs"] == "EPSG:32611"
    assert meta["grid"]["shape"] == [4, 4, 4]
    assert "mean" not in meta and "cov" not in meta and "samples" not in meta

    # the receipt carries a content-hash header (reused from inference.production.provenance.build_header)
    header = receipt["provenance"]["header"]
    assert isinstance(header["dataset_hash"], str) and len(header["dataset_hash"]) == 64

    # the outcome is the decision itself
    assert receipt["outcome"]["action"] == "drill"
