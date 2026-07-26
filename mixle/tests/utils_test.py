"""Regression tests for small mixle.utils helpers."""

import io
import unittest
from unittest import mock

import numpy as np

from mixle.inference.estimation import best_of
from mixle.stats import GaussianEstimator
from mixle.utils.metrics import roc_auc, roc_curve
from mixle.utils.optsutil import get_inv_map, least_occurring, text_file
from mixle.utils.vector import sorted_dict_merge_add


class UtilsTestCase(unittest.TestCase):
    def test_text_file_closes_the_descriptor_it_opens(self):
        opened = mock.mock_open(read_data="a\nb\n")
        with mock.patch("builtins.open", opened):
            self.assertEqual(text_file("observations.txt"), ["a", "b"])
        opened.assert_called_once_with("observations.txt", encoding="utf-8")
        opened.return_value.__enter__.assert_called_once_with()
        opened.return_value.__exit__.assert_called_once()

    def test_inverse_map_rejects_or_preserves_collisions_explicitly(self):
        with self.assertRaisesRegex(ValueError, "not invertible"):
            get_inv_map({"first": 1, "second": 1})
        self.assertEqual(get_inv_map({"first": 1, "second": 1}, multi=True), {1: ["first", "second"]})
        self.assertEqual(get_inv_map({"first": 1, "second": 2}), {1: "first", 2: "second"})

    def test_least_occurring_modern_dict_items(self):
        rare = list(least_occurring(["a", "a", "b", "c", "c", "d"], count=2, keep_freq=True))
        self.assertEqual(set(rare), {"b", "d"})

        rare_values = least_occurring(["a", "a", "b", "c", "c", "d"], count=2, keep_freq=False)
        self.assertEqual(set(rare_values), {"b", "d"})

    def test_least_occurring_validates_selection_controls(self):
        values = ["a", "b"]
        for count in (-1, 1.5, True):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "nonnegative integer"):
                least_occurring(values, count=count)
        for percent in (-0.1, 1.1, float("nan"), float("inf"), True):
            with self.subTest(percent=percent), self.assertRaisesRegex(ValueError, "between 0 and 1"):
                least_occurring(values, percent=percent)
        with self.assertRaisesRegex(ValueError, "only one"):
            least_occurring(values, count=1, percent=0.5)
        self.assertEqual(least_occurring(values, percent=0.0), [])

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
