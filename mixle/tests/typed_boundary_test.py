"""Version, ordering, duplicate, and approximation tests for shard messages."""

import json
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
from mixle.stats import GaussianDistribution, GaussianEstimator

pytestmark = [pytest.mark.experimental, pytest.mark.fast]


def _setup():
    graph = compile_update_graph(GaussianDistribution(0.0, 1.0), GaussianEstimator())
    return graph, RuntimeVersions.for_graph(graph), BoundaryInbox(graph)


def _message(message_id="m0", sequence=0, *, model_version=0, node_version=0, approximate=False):
    return BoundaryMessage(
        message_id=message_id,
        run_id="run",
        model_id="model",
        node_id="n0000",
        source_shard="left",
        target_shard="right",
        model_version=model_version,
        node_version=node_version,
        target_dependency_version=node_version,
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
