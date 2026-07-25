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
        result = loo_stack([np.tile(a[:, 0], (2, 1)), np.tile(a[:, 1], (2, 1))])
        w = result["weights"]
        self.assertTrue(0.2 < w[0] < 0.8)
        self.assertGreaterEqual(result["stacked_elpd_loo"], max(result["model_elpd_loo"]) - 1.0e-6)
        self.assertTrue(result["converged"])
        self.assertAlmostEqual(result["objective"], result["stacked_elpd_loo"])

    def test_single_model_weight_is_one(self):
        self.assertTrue(np.allclose(loo_stacking_weights(np.zeros((10, 1))), [1.0]))

    def test_zero_iters_raises(self):
        # w starts at the uniform simplex point np.full(k, 1/k) and is only updated inside the
        # `for _ in range(int(iters))` loop -- iters=0 used to fall through as a no-op and return that
        # untouched uniform guess as though it were the fitted stacking solution, indistinguishable
        # from a real (if coincidentally uniform) answer. Matches optimize(max_its < 1)'s rejection.
        rng = np.random.RandomState(0)
        lpd = rng.normal(0.0, 1.0, size=(200, 3))
        with self.assertRaises(ValueError):
            loo_stacking_weights(lpd, iters=0)

    def test_negative_iters_raises(self):
        rng = np.random.RandomState(0)
        lpd = rng.normal(0.0, 1.0, size=(200, 3))
        with self.assertRaises(ValueError):
            loo_stacking_weights(lpd, iters=-5)

    def test_nan_iters_raises(self):
        # `iters >= 1` is False for NaN too (a NaN comparison is always False), so `not (iters >= 1)`
        # catches it -- unlike a plain `iters < 1` check, which would miss it.
        rng = np.random.RandomState(0)
        lpd = rng.normal(0.0, 1.0, size=(200, 3))
        with self.assertRaises(ValueError):
            loo_stacking_weights(lpd, iters=float("nan"))

    def test_one_iter_still_moves_off_uniform(self):
        # Sanity check that a positive iters count still performs real optimization (as opposed to
        # the guard becoming so strict it breaks legitimate small-but-positive values).
        base = np.random.RandomState(2).normal(-1.0, 0.5, size=200)
        lpd = np.column_stack([base + 2.0, base, base - 1.0])
        w = loo_stacking_weights(lpd, iters=1)
        self.assertFalse(np.allclose(w, 1.0 / 3.0))

    def test_invalid_matrix_and_controls_are_rejected(self):
        for lpd in (np.array([]), np.ones(3), np.empty((0, 2)), np.empty((2, 0))):
            with self.assertRaises(ValueError):
                loo_stacking_weights(lpd)
        nonfinite = np.ones((5, 2))
        nonfinite[0, 0] = np.inf
        with self.assertRaises(ValueError):
            loo_stacking_weights(nonfinite)
        for iters in (True, 1.5):
            with self.assertRaises(ValueError):
                loo_stacking_weights(np.ones((5, 2)), iters=iters)
        for tol in (-1.0, float("nan"), True):
            with self.assertRaises(ValueError):
                loo_stacking_weights(np.ones((5, 2)), tol=tol)

    def test_result_receipt_reports_iteration_limit(self):
        base = np.linspace(-2.0, 0.0, 30)
        lpd = np.column_stack([base + 1.0, base])
        result = loo_stacking_weights(lpd, iters=1, tol=0.0, return_result=True)
        self.assertEqual(result["iterations"], 1)
        self.assertFalse(result["converged"])
        self.assertEqual(result["reason"], "iteration_limit")
        expected = np.log(
            np.exp(lpd[:, 0]) * result["weights"][0] + np.exp(lpd[:, 1]) * result["weights"][1]
        ).sum()
        self.assertAlmostEqual(result["objective"], expected)

    def test_stack_rejects_empty_misaligned_and_single_draw_models(self):
        with self.assertRaises(ValueError):
            loo_stack([])
        with self.assertRaises(ValueError):
            loo_stack([np.ones((2, 3)), np.ones((2, 4))])
        with self.assertRaises(ValueError):
            loo_stack([np.ones((1, 3))])


if __name__ == "__main__":
    unittest.main()
