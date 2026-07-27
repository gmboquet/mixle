"""Typed exact, bounded-stale, and corrected-eventual proposal tests."""

import math

import numpy as np
import pytest

from mixle.experimental.typed_runtime import (
    ConsistencyRequirement,
    CorrectionResult,
    CorrectionSemantics,
    MergeLaw,
    ObjectiveKind,
    ProposalPacket,
    RuntimeVersions,
    StalenessAction,
    StalenessPolicy,
    UpdateContract,
    UpdateKind,
    assess_staleness,
    shrink_proposal,
)

pytestmark = [pytest.mark.experimental]


def _contract(consistency, *, exact=False):
    return UpdateContract(
        objective_kind=ObjectiveKind.MLE,
        update_kind=UpdateKind.FIRST_ORDER,
        merge_law=MergeLaw.LOW_RANK,
        consistency=consistency,
        exact=exact,
    )


def _proposal(*, base=3, node_version=2):
    return ProposalPacket(
        proposal_id="proposal",
        run_id="run",
        model_id="model",
        node_id="node",
        shard_id="remote-island",
        base_model_version=base,
        dependency_versions={"node": node_version},
        update_kind=UpdateKind.FIRST_ORDER,
        objective_kind=ObjectiveKind.MLE,
        payload={"delta": np.array([2.0, -4.0])},
        predicted_gain=2.0,
    )


def _versions():
    return RuntimeVersions(5, {"node": 4})


def _correction(proposal, payload, *, semantics=CorrectionSemantics.STATISTICAL_APPROXIMATION):
    return CorrectionResult(
        source_proposal_id=proposal.proposal_id,
        source_payload_hash=proposal.payload_hash,
        run_id=proposal.run_id,
        model_id=proposal.model_id,
        node_id=proposal.node_id,
        target_model_version=5,
        target_dependency_versions={"node": 4},
        semantics=semantics,
        method="held-out-gradient-probe",
        evidence_id="evidence-1",
        payload=payload,
    )


def test_exact_and_strict_updates_reject_any_staleness():
    policy = StalenessPolicy(max_model_lag=3, max_node_lag=3)
    exact = assess_staleness(
        _proposal(),
        _contract(ConsistencyRequirement.BOUNDED_STALE, exact=True),
        _versions(),
        policy,
    )
    strict = assess_staleness(
        _proposal(),
        _contract(ConsistencyRequirement.STRICT_SYNCHRONOUS),
        _versions(),
        policy,
    )
    assert exact.reason == "exact-update-requires-current-version"
    assert strict.reason == "consistency-requires-current-version"
    assert not exact.accepted and not strict.accepted


def test_bounded_stale_delta_is_shrunk_rebased_and_refingerprinted():
    proposal = _proposal()
    receipt = assess_staleness(
        proposal,
        _contract(ConsistencyRequirement.BOUNDED_STALE),
        _versions(),
        StalenessPolicy(max_model_lag=2, max_node_lag=2, shrink_decay=0.5),
    )
    assert receipt.action is StalenessAction.SHRINK
    assert receipt.scale == pytest.approx(math.exp(-2.0))

    transformed = shrink_proposal(proposal, receipt, proposal_id="proposal-rebased")
    np.testing.assert_allclose(transformed.payload["delta"], proposal.payload["delta"] * receipt.scale)
    assert transformed.base_model_version == 5
    assert transformed.dependency_version_map == {"node": 4}
    assert transformed.payload_hash != proposal.payload_hash
    assert transformed.predicted_gain == pytest.approx(2.0 * receipt.scale)
    assert transformed.staleness_semantics == "bounded_stale_approximation"


def test_lag_over_bound_and_future_version_are_rejected():
    bounded = _contract(ConsistencyRequirement.BOUNDED_STALE)
    too_stale = assess_staleness(_proposal(), bounded, _versions(), StalenessPolicy(max_model_lag=1, max_node_lag=2))
    future = assess_staleness(
        _proposal(base=6, node_version=5),
        bounded,
        _versions(),
        StalenessPolicy(max_model_lag=2, max_node_lag=2),
    )
    assert too_stale.reason == "lag-bound-exceeded"
    assert future.reason == "future-version"


def test_corrected_eventual_binds_and_labels_statistical_correction():
    proposal = _proposal()
    contract = _contract(ConsistencyRequirement.CORRECTED_EVENTUAL)
    policy = StalenessPolicy(max_model_lag=2, max_node_lag=2)
    missing = assess_staleness(proposal, contract, _versions(), policy)
    correction = _correction(proposal, {"delta": np.array([1.0, -1.0])})
    corrected = assess_staleness(
        proposal,
        contract,
        _versions(),
        policy,
        correction=correction,
    )
    assert missing.reason == "missing-drift-correction"
    assert corrected.action is StalenessAction.CORRECT
    with pytest.raises(ValueError, match="bound correction"):
        shrink_proposal(proposal, corrected, proposal_id="corrected")

    transformed = shrink_proposal(
        proposal,
        corrected,
        proposal_id="corrected",
        correction=correction,
    )
    np.testing.assert_allclose(
        transformed.payload["delta"],
        np.array([1.0, -1.0]) * corrected.scale,
    )
    assert transformed.staleness_semantics == "corrected_statistical_approximation"
    assert transformed.correction_fingerprint == correction.fingerprint
    assert transformed.predicted_gain is None


def test_exact_rebase_is_not_shrunk_and_correction_payload_cannot_be_substituted():
    proposal = _proposal()
    contract = _contract(ConsistencyRequirement.CORRECTED_EVENTUAL)
    policy = StalenessPolicy(max_model_lag=2, max_node_lag=2)
    correction = _correction(
        proposal,
        {"delta": np.array([1.0, -1.0])},
        semantics=CorrectionSemantics.EXACT_REBASE,
    )
    receipt = assess_staleness(proposal, contract, _versions(), policy, correction=correction)
    assert receipt.scale == 1.0
    transformed = shrink_proposal(proposal, receipt, proposal_id="exact", correction=correction)
    np.testing.assert_array_equal(transformed.payload["delta"], [1.0, -1.0])
    assert transformed.staleness_semantics == "exact_rebase"

    substituted = _correction(
        proposal,
        {"delta": np.array([9.0, 9.0])},
        semantics=CorrectionSemantics.EXACT_REBASE,
    )
    with pytest.raises(ValueError, match="does not bind"):
        shrink_proposal(proposal, receipt, proposal_id="substituted", correction=substituted)


def test_correction_source_target_and_post_evidence_mutation_are_rejected():
    proposal = _proposal()
    contract = _contract(ConsistencyRequirement.CORRECTED_EVENTUAL)
    policy = StalenessPolicy(max_model_lag=2, max_node_lag=2)
    wrong_target = _correction(proposal, {"delta": np.ones(2)})
    object.__setattr__(wrong_target, "target_model_version", 4)
    with pytest.raises(ValueError, match="target version vector"):
        assess_staleness(proposal, contract, _versions(), policy, correction=wrong_target)

    mutable_payload = {"delta": np.ones(2)}
    correction = _correction(proposal, mutable_payload)
    mutable_payload["delta"][0] = 7.0
    with pytest.raises(ValueError, match="changed after"):
        assess_staleness(proposal, contract, _versions(), policy, correction=correction)


def test_staleness_decay_must_apply_real_shrinkage():
    with pytest.raises(ValueError, match="positive"):
        StalenessPolicy(shrink_decay=0.0)
