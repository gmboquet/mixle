"""Fail-closed task-adapter evidence contracts."""

import unittest

import numpy as np

from mixle.task.calibrate import ESCALATE, CalibratedTaskModel
from mixle.task.capacity import WordEmbeddingFeaturizer
from mixle.task.model import (
    ImpossibleEvidenceError,
    StructuredClassifierIO,
    TaskModel,
)


class _Encoder:
    def seq_encode(self, rows):
        return rows


class _ImpossibleModel:
    def dist_to_encoder(self):
        return _Encoder()

    def seq_log_density(self, rows):
        return np.full(len(rows), -np.inf)


class _BrokenModel(_ImpossibleModel):
    def seq_log_density(self, rows):
        raise RuntimeError("model implementation failed")


class StructuredEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.adapter = StructuredClassifierIO(["value"], 1, ["a", "b"])

    def test_impossible_evidence_is_explicit_and_calibrated_serving_escalates(self):
        with self.assertRaises(ImpossibleEvidenceError):
            self.adapter.proba_batch(_ImpossibleModel(), [{"value": "unseen"}])
        with self.assertRaises(ImpossibleEvidenceError):
            self.adapter.predict_batch(_ImpossibleModel(), [{"value": "unseen"}])

        task = TaskModel(_ImpossibleModel(), self.adapter, payload="json")
        calibrated = CalibratedTaskModel(task, alpha=0.1, qhat=1.0)
        self.assertEqual(calibrated.predict_set({"value": "unseen"}), [])
        self.assertIs(calibrated.decide({"value": "unseen"}), ESCALATE)

    def test_model_implementation_failure_is_not_converted_to_impossible_support(self):
        with self.assertRaisesRegex(RuntimeError, "implementation failed"):
            self.adapter.proba_batch(_BrokenModel(), [{"value": "x"}])

    def test_structured_schema_and_label_position_are_validated(self):
        with self.assertRaises(ValueError):
            StructuredClassifierIO(["x", "x"], 1, ["a", "b"])
        with self.assertRaises(ValueError):
            StructuredClassifierIO(["x"], 2, ["a", "b"])
        with self.assertRaisesRegex(ValueError, "exactly the keys"):
            self.adapter.logits_batch(_ImpossibleModel(), [{"other": 1}])


class EmbeddingGeometryTest(unittest.TestCase):
    def test_every_embedding_must_match_declared_finite_width(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            WordEmbeddingFeaturizer({"x": [1.0, 2.0]}, dim=3)
        with self.assertRaisesRegex(ValueError, "finite"):
            WordEmbeddingFeaturizer({"x": [1.0, np.nan]}, dim=2)


if __name__ == "__main__":
    unittest.main()
