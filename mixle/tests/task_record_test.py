"""Record/tabular tasks (mixle.task: HashedRecord + RecordClassifierIO + distill_records).

The spine must work for structured records (tuples/dicts of mixed fields), not just free text: distill a record
classifier from a teacher, calibrate it, persist it, and run a cascade -- the shape of classify-a-transaction /
route-a-ticket business tasks.
"""

import os
import tempfile
import unittest

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("safetensors")

from mixle.task.calibrate import CalibratedTaskModel  # noqa: E402
from mixle.task.cascade import Cascade  # noqa: E402
from mixle.task.distill import agreement, distill_records  # noqa: E402
from mixle.task.economics import CostModel  # noqa: E402
from mixle.task.model import HashedRecord, TaskModel  # noqa: E402


def records(seed, n=400):
    """Transactions: {amount, country, is_weekend} -> 'fraud' if (amount high AND foreign) else 'ok'."""
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(n):
        amount = float(rng.exponential(120))
        country = str(rng.choice(["US", "GB", "NG", "RU", "DE"]))
        is_weekend = bool(rng.rand() < 0.3)
        out.append({"amount": amount, "country": country, "is_weekend": is_weekend})
    return out


def teacher(recs):
    return ["fraud" if (r["amount"] > 200 and r["country"] in ("NG", "RU")) else "ok" for r in recs]


class HashedRecordTest(unittest.TestCase):
    def test_deterministic_and_handles_tuple_and_dict(self):
        f = HashedRecord(dim=64, seed=1)
        a = f.transform([{"x": 1.0, "c": "US"}])
        b = f.transform([{"x": 1.0, "c": "US"}])
        self.assertTrue(np.array_equal(a, b))
        self.assertEqual(f.transform([(1.0, "US", True)]).shape, (1, 64))

    def test_spec_round_trip(self):
        f = HashedRecord(dim=128, seed=5)
        g = HashedRecord.from_spec(f.to_spec())
        self.assertTrue(np.array_equal(f.transform([{"a": 2.0}]), g.transform([{"a": 2.0}])))


class DistillRecordsTest(unittest.TestCase):
    def test_student_recovers_record_rule(self):
        train = records(seed=1)
        student = distill_records(teacher, train, dim=512, hidden=[64], epochs=300, seed=0)
        self.assertGreaterEqual(student.meta["train_agreement"], 0.9)
        test = records(seed=99)
        self.assertGreaterEqual(agreement(student, teacher(test), test), 0.85)

    def test_save_load_and_call(self):
        train = records(seed=2)
        student = distill_records(teacher, train, dim=512, hidden=[64], epochs=250, seed=0)
        test = records(seed=7)
        before = student.batch(test)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "txn")
            student.save(path)
            loaded = TaskModel.load(path)
            self.assertEqual(loaded.adapter.kind, "record_classifier")
            after = loaded.batch(test)
        self.assertEqual(before, after)
        self.assertIn(student({"amount": 9999.0, "country": "RU", "is_weekend": False}), ("fraud", "ok"))


class RecordCascadeTest(unittest.TestCase):
    def test_calibrated_cascade_on_records(self):
        train, cal = records(seed=3), records(seed=53)
        student = distill_records(teacher, train, dim=512, hidden=[64], epochs=300, seed=0)
        model = CalibratedTaskModel(student, alpha=0.1).calibrate(cal, teacher(cal))
        casc = Cascade(model, teacher, cost=CostModel(c_frontier=1.0, c_local=0.0))
        served = records(seed=900)
        preds = casc.serve(served)
        acc = np.mean([p == t for p, t in zip(preds, teacher(served))])
        self.assertGreaterEqual(acc, 0.85)
        self.assertGreater(casc.report()["savings_vs_frontier"], 0.0)


if __name__ == "__main__":
    unittest.main()
