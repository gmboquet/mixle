"""Typed exact, bounded-stale, and corrected-eventual proposal tests."""

import math

import numpy as np
import pytest

from mixle.experimental.typed_runtime import (
    ConsistencyRequirement,
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

pytestmark = [pytest.mark.experimental, pytest.mark.fast]


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


def test_corrected_eventual_requires_correction_receipt_and_corrected_payload():
    proposal = _proposal()
    contract = _contract(ConsistencyRequirement.CORRECTED_EVENTUAL)
    policy = StalenessPolicy(max_model_lag=2, max_node_lag=2)
    missing = assess_staleness(proposal, contract, _versions(), policy)
    corrected = assess_staleness(
        proposal,
        contract,
        _versions(),
        policy,
        correction_fingerprint="probe-gradient-hash",
    )
    assert missing.reason == "missing-drift-correction"
    assert corrected.action is StalenessAction.CORRECT
    with pytest.raises(ValueError, match="corrected_payload"):
        shrink_proposal(proposal, corrected, proposal_id="corrected")

    transformed = shrink_proposal(
        proposal,
        corrected,
        proposal_id="corrected",
        corrected_payload={"delta": np.array([1.0, -1.0])},
    )
    np.testing.assert_allclose(
        transformed.payload["delta"],
        np.array([1.0, -1.0]) * corrected.scale,
    )
