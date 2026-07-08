"""Frontier -> mixle-native students (mixle.task.frontier_to_native): the J4 integration loop, end to end.

A "frontier" teacher (an oversized torch MLP, standing in for a big general-purpose model) is distilled
into a small, task-specific, LNS-structured student; the student is calibrated and composed with the
teacher into a served :class:`~mixle.task.cascade.Cascade`. This exercises the acceptance criteria
directly: a real served cascade cost/quality receipt (cascade cost near the cheap student's, quality
closer to the expensive teacher's), the LNS student's real compressed footprint versus the frontier's,
and the real student-teacher agreement rate on held-out data.
"""

import unittest

import numpy as np
import pytest

pytest.importorskip("torch")

from mixle.task.distill import distill_records_from_labels  # noqa: E402
from mixle.task.economics import CostModel  # noqa: E402
from mixle.task.edge import footprint  # noqa: E402
from mixle.task.frontier_to_native import (  # noqa: E402
    build_served_cascade,
    distill_to_lns_student,
    measure_cascade_receipt,
)


def _truth(records):
    """The ground-truth rule a frontier model (and, hopefully, its distilled student) should recover."""
    out = []
    for r in records:
        score = (1.5 if r["region"] == "west" else -0.5) + 0.4 * r["spend"] + 0.3 * r["visits"]
        out.append("churn" if score < 1.0 else "retain")
    return out


def _gen(n, seed):
    rng = np.random.RandomState(seed)
    return [
        {
            "region": rng.choice(["west", "east"]),
            "spend": float(rng.normal(2.0, 1.5)),
            "visits": int(rng.poisson(3)),
        }
        for _ in range(n)
    ]


def _build_frontier():
    """An oversized torch MLP distilled from the ground-truth rule -- the expensive 'frontier' teacher.

    Big on purpose (wide hidden layers, high-dim hashed features) so its measured byte footprint
    (:func:`~mixle.task.edge.footprint`) is genuinely large relative to the LNS student -- the frontier
    is a real mixle artifact here, not a hand-waved constant, so the compression ratio is real too.
    """
    records = _gen(1500, 0)
    labels = _truth(records)
    frontier = distill_records_from_labels(
        records, labels, dim=1024, hidden=[256, 256], epochs=200, lr=1e-2, seed=0, task="frontier"
    )
    return frontier


class _Teacher:
    """Wraps the frontier TaskModel as a plain batched callable -- the ``teacher`` contract distill/cascade expect."""

    def __init__(self, frontier):
        self.frontier = frontier
        self.calls = 0

    def __call__(self, records):
        self.calls += len(records)
        return self.frontier.batch(list(records))


class FrontierToNativeReceiptTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frontier = _build_frontier()
        cls.teacher = _Teacher(cls.frontier)
        cls.teacher_bytes = footprint(cls.frontier).bytes

        train = _gen(700, 1)
        cls.lns_student = distill_to_lns_student(cls.teacher, train, min_gain=1.0, seed=0, task="lns_edge_student")

        cal = _gen(200, 2)
        cost = CostModel(c_frontier=1.0, c_local=0.001)
        cls.cascade = build_served_cascade(cls.lns_student, cls.teacher, cal, alpha=0.1, cost=cost)

        cls.test_records = _gen(400, 3)
        cls.truth = _truth(cls.test_records)
        cls.receipt = measure_cascade_receipt(cls.cascade, cls.test_records, cls.truth, teacher_bytes=cls.teacher_bytes)

    def test_student_is_lns_structured_and_torch_free(self):
        from mixle.task.quantize import LNSStructuredClassifierIO

        self.assertIsInstance(self.lns_student.adapter, LNSStructuredClassifierIO)
        self.assertEqual(self.lns_student.payload, "json")  # no torch weights on disk
        fp = footprint(self.lns_student)
        self.assertTrue(fp.torch_free)

    def test_served_cascade_cost_quality_receipt(self):
        r = self.receipt
        print("\n" + r.summary())

        # cost: the cascade is cheap -- close to the student, well under always-escalate-to-frontier
        self.assertLessEqual(r.cascade_cost_per_request, r.teacher_cost_per_request)
        self.assertLess(r.cascade_cost_per_request, (r.student_cost_per_request + r.teacher_cost_per_request) / 2)

        # quality: cascade should not be worse than the student alone, and the frontier is genuinely better
        self.assertGreaterEqual(r.cascade_quality, r.student_quality - 1e-9)
        self.assertGreater(r.teacher_quality, 0.8)

        self.assertTrue(r.earns_its_complexity())
        self.assertGreater(r.n_requests, 0)
        self.assertLessEqual(r.n_escalated, r.n_requests)

    def test_edge_student_footprint_and_agreement(self):
        r = self.receipt
        # a real, measured compression ratio: the structured/LNS student is dramatically smaller
        self.assertIsNotNone(r.compression_ratio)
        self.assertGreater(r.student_bytes, 0)
        self.assertLess(r.student_bytes, self.teacher_bytes)
        self.assertGreater(r.compression_ratio, 5.0)

        # a real student-teacher agreement rate on held-out data
        self.assertGreaterEqual(r.agreement_rate, 0.0)
        self.assertLessEqual(r.agreement_rate, 1.0)
        manual_student_preds = [str(p) for p in self.lns_student.batch(self.test_records)]
        manual_teacher_preds = [str(p) for p in self.teacher(self.test_records)]
        manual_agreement = float(np.mean([s == t for s, t in zip(manual_student_preds, manual_teacher_preds)]))
        self.assertAlmostEqual(r.agreement_rate, manual_agreement, places=9)
        # a genuinely useful edge student agrees with its teacher well above chance (2 classes here)
        self.assertGreater(r.agreement_rate, 0.6)

    def test_teacher_only_touched_on_escalation(self):
        # the cascade only calls the teacher for escalated requests, not every request
        calls_before = self.teacher.calls
        cascade = build_served_cascade(
            self.lns_student, self.teacher, _gen(150, 20), alpha=0.1, cost=CostModel(c_frontier=1.0, c_local=0.0)
        )
        served = _gen(120, 21)
        cascade.serve(served)
        self.assertEqual(cascade.stats.n_escalated, self.teacher.calls - calls_before - 150)
        self.assertLess(cascade.stats.n_escalated, len(served))


if __name__ == "__main__":
    unittest.main()
