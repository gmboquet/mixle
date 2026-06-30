"""Model-shape recommendation (mixle.task.recommend): point at data, get a model shape with confidence.

Wraps mixle's generative structure analysis into a decision-oriented advisor: the recommended estimator, the
per-field family + confidence, the low-confidence fields (where more data helps), and a fit shortcut.
Torch-free -- runs in the base suite.
"""

import unittest

import numpy as np

from mixle.task.recommend import ModelRecommendation, recommend_model


def _hetero_records(n=800, seed=0):
    rng = np.random.RandomState(seed)
    return [("a" if rng.rand() < 0.5 else "b", float(rng.randn()), int(rng.poisson(4))) for _ in range(n)]


class RecommendTest(unittest.TestCase):
    def test_recommends_composite_for_heterogeneous_record(self):
        rec = recommend_model(_hetero_records())
        self.assertIsInstance(rec, ModelRecommendation)
        self.assertEqual(type(rec.estimator).__name__, "CompositeEstimator")
        self.assertEqual(len(rec.fields), 3)
        kinds = [c.kind for c in rec.fields]
        self.assertIn("string", kinds)
        self.assertIn("numeric", kinds)
        self.assertIn("integer", kinds)

    def test_string_field_is_type_determined_confident(self):
        rec = recommend_model(_hetero_records())
        string_field = next(c for c in rec.fields if c.kind == "string")
        self.assertEqual(string_field.family, "categorical")
        self.assertTrue(string_field.confident)  # no numeric contender -> type-determined

    def test_low_confidence_fields_are_a_subset(self):
        rec = recommend_model(_hetero_records())
        low = rec.low_confidence_fields()
        self.assertTrue(set(c.path for c in low).issubset(c.path for c in rec.fields))
        for c in low:
            self.assertFalse(c.confident)

    def test_fit_returns_a_model_that_scores(self):
        data = _hetero_records()
        rec = recommend_model(data)
        model = rec.fit(data, out=None)
        enc = model.dist_to_encoder().seq_encode(data)
        self.assertEqual(len(model.seq_log_density(enc)), len(data))

    def test_fit_flag_attaches_model(self):
        data = _hetero_records(n=400, seed=3)
        rec = recommend_model(data, fit=True)
        self.assertTrue(hasattr(rec, "model"))
        self.assertTrue(np.isfinite(rec.model.log_density(data[0])))

    def test_explain_returns_lines(self):
        rec = recommend_model(_hetero_records(n=300, seed=5))
        lines = rec.explain()
        self.assertTrue(all(isinstance(s, str) for s in lines))
        self.assertGreater(len(lines), 0)


if __name__ == "__main__":
    unittest.main()
