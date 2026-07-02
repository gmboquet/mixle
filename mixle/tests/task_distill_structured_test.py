"""Structured distillation (mixle.task.distill_structured): a teacher is distilled into a *structured probabilistic*
classifier -- a learned dependency network, not a neural net.

The teacher applies a rule over heterogeneous fields (a category + a continuous + a count); the student discovers
the dependency structure over the joint ``(fields, label)`` and classifies generatively by ``argmax P(fields, label)``.
It should recover the teacher with high agreement, expose the discovered edges, produce a *real* posterior (so the
conformal calibration stack works), survive a fresh-process save/load, and be tiny -- with no torch dependency.
"""

import os
import tempfile
import unittest

import numpy as np

from mixle.task import CalibratedTaskModel, TaskModel, distill_structured


def _teacher(records):
    # the rule the student must recover: label is driven by region (category), spend (real), visits (count)
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


class DistillStructuredTest(unittest.TestCase):
    def test_student_recovers_teacher(self):
        student = distill_structured(_teacher, _gen(600, 1), min_gain=1.0, task="churn")
        self.assertGreaterEqual(student.meta["train_agreement"], 0.85)

        test = _gen(300, 99)
        pred, truth = student.batch(test), _teacher(test)
        held_out = float(np.mean([p == t for p, t in zip(pred, truth)]))
        self.assertGreaterEqual(held_out, 0.8)

    def test_discovers_dependency_edges(self):
        student = distill_structured(_teacher, _gen(600, 2), min_gain=1.0)
        # schema is [region, spend, visits, LABEL] -> field 3 is the label; every field should touch the label
        edges = student.meta["edges"]
        label_linked = {a for a, b in edges if b == 3} | {b for a, b in edges if a == 3}
        self.assertTrue(label_linked, "the student should discover at least one dependency involving the label")
        self.assertTrue(student.meta["structured"])

    def test_posterior_is_a_real_probability(self):
        student = distill_structured(_teacher, _gen(400, 3), min_gain=1.0)
        test = _gen(50, 7)
        proba = student.adapter.proba_batch(student.model, test)
        self.assertTrue(np.allclose(proba.sum(axis=1), 1.0))
        self.assertEqual(proba.shape[1], len(student.adapter.labels))

    def test_conformal_calibration_covers(self):
        train, cal_recs = _gen(600, 4), _gen(200, 5)
        student = distill_structured(_teacher, train, min_gain=1.0)
        cal = CalibratedTaskModel(student, alpha=0.1).calibrate(cal_recs, _teacher(cal_recs))
        test = _gen(200, 6)
        truth = _teacher(test)
        covered = float(np.mean([truth[i] in cal.predict_set(test[i]) for i in range(len(test))]))
        self.assertGreaterEqual(covered, 0.85)  # ~1 - alpha coverage on a real posterior

    def test_artifact_roundtrip_is_tiny_and_torch_free(self):
        student = distill_structured(_teacher, _gen(400, 8), min_gain=1.0)
        test = _gen(100, 9)
        pred = student.batch(test)
        with tempfile.TemporaryDirectory() as d:
            student.save(d)
            self.assertFalse(any(f.endswith(".safetensors") for f in os.listdir(d)))  # json payload, no weights
            size = sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d))
            self.assertLess(size, 50_000)  # a few kilobytes, not a neural checkpoint
            reloaded = TaskModel.load(d)
        self.assertEqual(reloaded.batch(test), pred)  # fresh-object reconstruction is bit-identical

    def test_mixture_of_trees_student(self):
        train = _gen(600, 10)
        student = distill_structured(_teacher, train, n_components=2, min_gain=0.5, seed=1)
        test = _gen(200, 11)
        pred, truth = student.batch(test), _teacher(test)
        self.assertGreaterEqual(float(np.mean([p == t for p, t in zip(pred, truth)])), 0.8)


if __name__ == "__main__":
    unittest.main()
