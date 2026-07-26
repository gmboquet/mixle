"""Focused accounting contracts for frontier-to-native evaluation receipts."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from mixle.task.cascade import Cascade
from mixle.task.economics import CostModel
from mixle.task.frontier_to_native import measure_cascade_receipt


class _Student:
    def batch(self, rows):
        return ["student" for _ in rows]


class _Calibrated:
    def __init__(self):
        self.task = _Student()

    def batch_decide(self, rows):
        return ["student" if row == "local" else None for row in rows]


class _StatefulTeacher:
    def __init__(self):
        self.calls = 0

    def __call__(self, rows):
        self.calls += 1
        label = "teacher" if self.calls == 1 else "changed"
        return [label for _ in rows]


def test_receipt_uses_batch_deltas_and_one_teacher_snapshot():
    teacher = _StatefulTeacher()
    cascade = Cascade(
        _Calibrated(),
        teacher,
        cost=CostModel(c_frontier=2.0, c_local=0.1),
    )
    cascade.stats.n_requests = 7
    cascade.stats.n_escalated = 3

    with patch(
        "mixle.task.frontier_to_native.footprint",
        return_value=SimpleNamespace(bytes=10),
    ):
        receipt = measure_cascade_receipt(
            cascade,
            ["local", "remote"],
            ["student", "teacher"],
            teacher_bytes=100,
        )

    assert teacher.calls == 1
    assert receipt.n_requests == 2
    assert receipt.n_escalated == 1
    assert receipt.cascade_quality == 1.0
    assert receipt.cascade_cost_per_request == pytest.approx(1.1)
    assert cascade.stats.n_requests == 9
    assert cascade.stats.n_escalated == 4
    assert cascade.stats.escalated_texts[-1] == "remote"
    assert cascade.stats.escalated_labels[-1] == "teacher"


def test_receipt_rejects_empty_evaluation_batch():
    cascade = Cascade(
        _Calibrated(),
        _StatefulTeacher(),
        cost=CostModel(c_frontier=2.0, c_local=0.1),
    )
    with pytest.raises(ValueError, match="non-empty"):
        measure_cascade_receipt(cascade, [], [])
