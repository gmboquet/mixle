"""Tests for the WS-I model-writing training-data generator (free verifiable labels)."""

import unittest

import numpy as np

from pysp.ppl.training_data import (
    ModelingExample,
    build_model_from_code,
    families,
    fit_example,
    generate_examples,
)


class TrainingDataTest(unittest.TestCase):
    def test_generation_is_deterministic_and_covers_families(self):
        a = generate_examples(10, seed=0)
        b = generate_examples(10, seed=0)
        self.assertEqual([e.code for e in a], [e.code for e in b])
        self.assertEqual([e.family for e in a], [e.family for e in b])
        # cycling through templates covers every family within one pass
        self.assertEqual(set(e.family for e in generate_examples(len(families()), seed=1)), set(families()))

    def test_every_example_roundtrips(self):
        # The emitted code must build a model that fits its own sampled data to finite log-densities.
        for ex in generate_examples(15, seed=3):
            with self.subTest(family=ex.family):
                self.assertIsInstance(ex, ModelingExample)
                self.assertTrue(len(ex.data) >= 50)
                fitted = fit_example(ex)
                lp = np.asarray(fitted.dist.seq_log_density(fitted.dist.dist_to_encoder().seq_encode(ex.data)))
                self.assertTrue(np.all(np.isfinite(lp)))

    def test_build_model_requires_model_variable(self):
        with self.assertRaises(ValueError):
            build_model_from_code("from pysp.ppl import Normal, free\nx = Normal(free, free)")

    def test_recovers_a_known_gaussian_mean(self):
        # Sanity: fitting the generated Gaussian code recovers a mean near the data mean.
        ex = next(e for e in generate_examples(5, seed=7) if e.family == "gaussian")
        fitted = fit_example(ex)
        self.assertAlmostEqual(float(fitted.dist.mu), float(np.mean(ex.data)), delta=0.3)


if __name__ == "__main__":
    unittest.main()
