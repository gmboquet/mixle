"""Reproducible distributed delivery and worker-failure tests."""

import json

import numpy as np
import pytest

from mixle.experimental.typed_runtime import (
    BoundaryFaultInjector,
    BoundaryInbox,
    BoundaryMessage,
    BoundaryMessageKind,
    FaultEvent,
    FaultKind,
    RuntimeVersions,
    compile_update_graph,
)
from mixle.stats import GaussianDistribution, GaussianEstimator

pytestmark = [pytest.mark.experimental, pytest.mark.fast]


def _runtime():
    graph = compile_update_graph(GaussianDistribution(0.0, 1.0), GaussianEstimator())
    return BoundaryInbox(graph), RuntimeVersions(2, {"n0000": 0})


def _message(message_id="message", sequence=0):
    return BoundaryMessage(
        message_id,
        "run",
        "model",
        "n0000",
        "worker-a",
        "coordinator",
        2,
        0,
        0,
        sequence,
        BoundaryMessageKind.SUFFICIENT_STATISTICS,
        {"sum": np.array([1.0, 2.0])},
    )


def test_duplicate_is_delivered_twice_but_consumed_once():
    inbox, versions = _runtime()
    injector = BoundaryFaultInjector((FaultEvent("message", FaultKind.DUPLICATE),))
    deliveries = injector.intercept(_message(), step=0)
    receipts = [inbox.receive(message, versions) for message in deliveries]
    assert len(deliveries) == 2
    assert receipts[0].accepted
    assert receipts[1].reason == "duplicate-message-id"


def test_corruption_is_caught_and_clean_retry_can_commit_same_sequence():
    inbox, versions = _runtime()
    injector = BoundaryFaultInjector((FaultEvent("message", FaultKind.CORRUPT),))
    corrupted = injector.intercept(_message(), step=0)[0]
    assert inbox.receive(corrupted, versions).reason == "payload-mutated"

    clean_retry = injector.intercept(_message(), step=1)[0]
    assert inbox.receive(clean_retry, versions).accepted


def test_delay_does_not_deliver_before_release_step():
    inbox, versions = _runtime()
    injector = BoundaryFaultInjector((FaultEvent("message", FaultKind.DELAY, release_step=3),))
    assert injector.intercept(_message(), step=0) == ()
    assert injector.release(step=2) == ()
    released = injector.release(step=3)
    assert len(released) == 1
    assert inbox.receive(released[0], versions).accepted


def test_stale_version_is_rejected_and_retry_is_current():
    inbox, versions = _runtime()
    injector = BoundaryFaultInjector((FaultEvent("message", FaultKind.STALE_VERSION),))
    stale = injector.intercept(_message(), step=0)[0]
    assert inbox.receive(stale, versions).reason == "model-version-mismatch"
    assert inbox.receive(injector.intercept(_message(), step=1)[0], versions).accepted


def test_worker_loss_blocks_other_messages_until_recovery_without_double_counting():
    inbox, versions = _runtime()
    injector = BoundaryFaultInjector((FaultEvent("message", FaultKind.WORKER_LOSS),))
    assert injector.intercept(_message(), step=0) == ()
    assert injector.intercept(_message("other"), step=1) == ()
    injector.recover("worker-a")
    retry = injector.intercept(_message(), step=2)[0]
    assert inbox.receive(retry, versions).accepted
    assert len(inbox.as_dict()["seen_message_ids"]) == 1
    json.dumps(injector.as_dict(), allow_nan=False)
