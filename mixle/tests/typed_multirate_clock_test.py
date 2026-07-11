"""Tests for independent step/token/observation update clocks."""

import json

import pytest

from mixle.experimental.typed_runtime import (
    ClockProgress,
    ClockTrigger,
    MultiRateUpdateClocks,
    UpdateCadence,
)

pytestmark = [pytest.mark.experimental, pytest.mark.fast]


def test_nodes_become_due_on_different_progress_axes():
    clocks = MultiRateUpdateClocks(
        {
            "gate": UpdateCadence(every_steps=1),
            "retriever": UpdateCadence(every_tokens=1_000, max_staleness_steps=10),
            "calibrator": UpdateCadence(every_observations=500),
        }
    )
    initial = clocks.evaluate(ClockProgress())
    assert all(decision.triggers == (ClockTrigger.NEVER_COMMITTED,) for decision in initial)
    clocks.mark_committed(tuple(decision.node_id for decision in initial), ClockProgress())

    decisions = {decision.node_id: decision for decision in clocks.evaluate(ClockProgress(1, 400, 200))}
    assert decisions["gate"].due
    assert decisions["gate"].triggers == (ClockTrigger.STEP,)
    assert not decisions["retriever"].due
    assert not decisions["calibrator"].due

    decisions = {decision.node_id: decision for decision in clocks.evaluate(ClockProgress(2, 1_200, 600))}
    assert decisions["retriever"].triggers == (ClockTrigger.TOKENS,)
    assert decisions["calibrator"].triggers == (ClockTrigger.OBSERVATIONS,)


def test_staleness_bound_forces_a_slow_clock_and_only_commit_advances_it():
    clocks = MultiRateUpdateClocks({"slow": UpdateCadence(every_tokens=1_000_000, max_staleness_steps=3)})
    clocks.mark_committed(("slow",), ClockProgress())

    decision = clocks.evaluate(ClockProgress(3, 10, 1.0))[0]
    assert decision.triggers == (ClockTrigger.STALENESS_BOUND,)
    assert decision.commit_count == 1
    assert clocks.evaluate(ClockProgress(4, 20, 2.0))[0].due

    clocks.mark_committed(("slow",), ClockProgress(4, 20, 2.0))
    assert not clocks.evaluate(ClockProgress(5, 21, 3.0))[0].due
    json.dumps(clocks.as_dict(), allow_nan=False)


def test_progress_rewind_and_unknown_clock_fail():
    clocks = MultiRateUpdateClocks({"node": UpdateCadence(every_steps=1)})
    clocks.evaluate(ClockProgress(2, 10, 5.0))
    with pytest.raises(ValueError, match="backward"):
        clocks.evaluate(ClockProgress(1, 10, 5.0))
    with pytest.raises(KeyError, match="unknown update clocks"):
        clocks.evaluate(ClockProgress(3, 10, 5.0), ("missing",))
