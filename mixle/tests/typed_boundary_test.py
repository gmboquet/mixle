"""Version, ordering, duplicate, and approximation tests for shard messages."""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import pytest

from mixle.experimental.typed_runtime import (
    BoundaryInbox,
    BoundaryMessage,
    BoundaryMessageKind,
    RuntimeVersions,
    compile_update_graph,
)
from mixle.stats import GaussianDistribution, GaussianEstimator, MixtureDistribution, MixtureEstimator

pytestmark = [pytest.mark.experimental, pytest.mark.fast]


def _setup():
    graph = compile_update_graph(GaussianDistribution(0.0, 1.0), GaussianEstimator())
    return graph, RuntimeVersions.for_graph(graph), BoundaryInbox(graph, run_id="run", model_id="model")


def _message(
    message_id="m0",
    sequence=0,
    *,
    run_id="run",
    model_id="model",
    model_version=0,
    node_version=0,
    dependency_versions=None,
    approximate=False,
):
    return BoundaryMessage(
        message_id=message_id,
        run_id=run_id,
        model_id=model_id,
        node_id="n0000",
        source_shard="left",
        target_shard="right",
        model_version=model_version,
        node_version=node_version,
        dependency_versions={"n0000": node_version} if dependency_versions is None else dependency_versions,
        sequence_number=sequence,
        kind=BoundaryMessageKind.SUFFICIENT_STATISTICS,
        payload={"sum": np.array([1.0, 2.0]), "count": 2},
        approximate=approximate,
        error_bound=0.01 if approximate else None,
    )


def test_exactly_once_ordered_delivery():
    _, versions, inbox = _setup()
    first = _message()
    assert inbox.receive(first, versions).accepted

    duplicate = inbox.receive(first, versions)
    stale = inbox.receive(_message("m1", 0), versions)
    gap = inbox.receive(_message("m2", 2), versions)
    second = inbox.receive(_message("m3", 1), versions)

    assert duplicate.reason == "duplicate-message-id"
    assert stale.reason == "stale-sequence"
    assert gap.reason == "sequence-gap"
    assert second.accepted
    assert inbox.next_sequence(first) == 2
    json.dumps(inbox.as_dict(), allow_nan=False)


def test_version_mismatch_never_advances_stream():
    _, versions, inbox = _setup()
    wrong_model = inbox.receive(_message(model_version=1), versions)
    wrong_node = inbox.receive(_message("m1", node_version=1), versions)
    assert wrong_model.reason == "model-version-mismatch"
    assert wrong_node.reason == "node-version-mismatch"
    assert inbox.receive(_message("m2"), versions).accepted


def test_mutated_payload_is_rejected_before_consumption():
    _, versions, inbox = _setup()
    message = _message()
    message.payload["sum"][0] = 999.0
    receipt = inbox.receive(message, versions)
    assert receipt.reason == "payload-mutated"
    assert inbox.receive(_message("clean"), versions).accepted


def test_approximation_is_rejected_for_an_exact_node_without_consuming_sequence():
    _, versions, inbox = _setup()
    approximate = inbox.receive(_message(approximate=True), versions)
    assert approximate.reason == "approximation-for-exact-node"
    assert inbox.receive(_message("exact"), versions).accepted


def test_message_constructor_checks_declared_hash_and_error_bound():
    with pytest.raises(ValueError, match="error_bound"):
        replace(_message(), approximate=True, error_bound=None)
    with pytest.raises(ValueError, match="does not match"):
        replace(_message(), payload_hash="wrong")


def test_run_model_and_message_identity_do_not_share_sequence_or_receipt_domains():
    graph, versions, inbox = _setup()
    accepted = inbox.receive(_message(), versions)
    assert (accepted.run_id, accepted.model_id) == ("run", "model")

    wrong_run = inbox.receive(_message("other-message", run_id="other-run"), versions)
    wrong_model = inbox.receive(_message("other-model-message", model_id="other-model"), versions)
    assert wrong_run.reason == "run-id-mismatch"
    assert wrong_model.reason == "model-id-mismatch"
    assert _message().stream_key != _message(run_id="other-run").stream_key

    other_inbox = BoundaryInbox(graph, run_id="other-run", model_id="model")
    assert other_inbox.receive(_message(run_id="other-run"), versions).accepted


def test_dependency_vector_is_complete_and_compared_to_each_declared_dependency():
    model = MixtureDistribution(
        [GaussianDistribution(-1.0, 1.0), GaussianDistribution(1.0, 1.0)],
        [0.5, 0.5],
    )
    graph = compile_update_graph(model, MixtureEstimator([GaussianEstimator(), GaussianEstimator()]))
    versions = RuntimeVersions(0, {"n0000": 0, "n0001": 3, "n0002": 4})
    inbox = BoundaryInbox(graph, run_id="run", model_id="model")
    base = _message(
        node_version=0,
        dependency_versions={"n0000": 0, "n0001": 3, "n0002": 4},
    )
    assert inbox.receive(replace(base, dependency_versions=(("n0000", 0),)), versions).reason == (
        "dependency-vector-mismatch"
    )
    stale_dependency = replace(
        base,
        message_id="stale-child",
        dependency_versions=(("n0000", 0), ("n0001", 2), ("n0002", 4)),
    )
    assert inbox.receive(stale_dependency, versions).reason == "dependency-version-mismatch"
    assert inbox.receive(replace(base, message_id="complete"), versions).accepted


def test_concurrent_receive_checks_and_sequence_commit_are_atomic():
    _, versions, inbox = _setup()
    messages = (_message("left", 0), _message("right", 0))
    with ThreadPoolExecutor(max_workers=2) as pool:
        receipts = tuple(pool.map(lambda message: inbox.receive(message, versions), messages))
    assert sum(receipt.accepted for receipt in receipts) == 1
    assert {receipt.reason for receipt in receipts} == {"accepted", "stale-sequence"}
    assert inbox.next_sequence(messages[0]) == 1
