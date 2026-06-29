"""Tests for LOO stacking weights / model averaging (WS-F)."""

import unittest

import numpy as np

from mixle.ppl.diagnostics import loo_stack, loo_stacking_weights


class LooStackingTest(unittest.TestCase):
    def test_weights_on_simplex(self):
        rng = np.random.RandomState(0)
        lpd = rng.normal(0.0, 1.0, size=(200, 3))
        w = loo_stacking_weights(lpd)
        self.assertEqual(w.shape, (3,))
        self.assertAlmostEqual(float(w.sum()), 1.0, places=8)
        self.assertTrue(np.all(w >= -1.0e-12))

    def test_dominant_model_gets_all_weight(self):
        # Model 0 is uniformly better per observation -> stacking concentrates on it.
        rng = np.random.RandomState(1)
        base = rng.normal(-1.0, 0.5, size=200)
        lpd = np.column_stack([base + 2.0, base, base - 1.0])
        w = loo_stacking_weights(lpd)
        self.assertGreater(w[0], 0.95)

    def test_complementary_models_blend_and_beat_each(self):
        # Two models, each better on a disjoint half of the data -> stacking blends them and the
        # stacked LOO log-score exceeds either single model's.
        n = 300
        a = np.full((n, 2), -3.0)
        a[: n // 2, 0] = -0.1  # model 0 great on first half
        a[n // 2 :, 1] = -0.1  # model 1 great on second half
        result = loo_stack([a[:, 0][None, :], a[:, 1][None, :]])
        w = result["weights"]
        self.assertTrue(0.2 < w[0] < 0.8)
        self.assertGreaterEqual(result["stacked_elpd_loo"], max(result["model_elpd_loo"]) - 1.0e-6)

    def test_single_model_weight_is_one(self):
        self.assertTrue(np.allclose(loo_stacking_weights(np.zeros((10, 1))), [1.0]))


if __name__ == "__main__":
    unittest.main()
