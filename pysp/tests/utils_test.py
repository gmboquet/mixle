"""Regression tests for small pysp.utils helpers."""

import io
import unittest

import numpy as np

from pysp.inference.estimation import best_of
from pysp.stats import GaussianEstimator
from pysp.utils.metrics import roc_auc, roc_curve
from pysp.utils.optsutil import least_occurring
from pysp.utils.vector import sorted_dict_merge_add


class UtilsTestCase(unittest.TestCase):
    def test_least_occurring_modern_dict_items(self):
        rare = list(least_occurring(["a", "a", "b", "c", "c", "d"], count=2, keep_freq=True))
        self.assertEqual(set(rare), {"b", "d"})

        rare_values = least_occurring(["a", "a", "b", "c", "c", "d"], count=2, keep_freq=False)
        self.assertEqual(set(rare_values), {"b", "d"})

    def test_sorted_dict_merge_add_keeps_counts_aligned(self):
        keys, counts = sorted_dict_merge_add(
            np.asarray([1, 3, 5]),
            np.asarray([10, 30, 50]),
            np.asarray([2, 3]),
            np.asarray([20, 300]),
        )
        np.testing.assert_array_equal(keys, np.asarray([1, 2, 3, 5]))
        np.testing.assert_array_equal(counts, np.asarray([10, 20, 330, 50]))

        keys, counts = sorted_dict_merge_add(
            np.asarray([1, 2]),
            np.asarray([1, 2]),
            np.asarray([3]),
            np.asarray([3]),
        )
        np.testing.assert_array_equal(keys, np.asarray([1, 2, 3]))
        np.testing.assert_array_equal(counts, np.asarray([1, 2, 3]))

    def test_roc_curve_and_auc(self):
        pd, fa = roc_curve([0.9, 0.8], [0.7, 0.1])
        np.testing.assert_allclose(pd[[0, -1]], [0.0, 1.0])
        np.testing.assert_allclose(fa[[0, -1]], [0.0, 1.0])
        self.assertAlmostEqual(roc_auc([0.9, 0.8], [0.7, 0.1]), 1.0)

    def test_best_of_without_validation_uses_training_score(self):
        data = [-1.0, 0.0, 1.0, 2.0]
        ll, model = best_of(
            data,
            None,
            GaussianEstimator(),
            trials=1,
            max_its=2,
            init_p=1.0,
            delta=1.0e-9,
            rng=np.random.RandomState(1),
            out=io.StringIO(),
        )
        self.assertTrue(np.isfinite(ll))
        self.assertTrue(np.isfinite(model.log_density(0.0)))


if __name__ == "__main__":
    unittest.main()
