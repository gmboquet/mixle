"""DOE distillation design helpers."""

import unittest

import numpy as np

from mixle.doe import (
    cross_modal_distillation_design,
    distillation_design,
    multitask_distillation_design,
)


class DistillationDesignTest(unittest.TestCase):
    def test_multitask_design_balances_tasks_despite_uncertainty_skew(self):
        x = np.arange(9, dtype=np.float64).reshape(-1, 1)
        tasks = ["vision"] * 3 + ["text"] * 3 + ["audio"] * 3
        uncertainty = np.array([10.0, 9.0, 8.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        result = multitask_distillation_design(
            x,
            3,
            task_labels=tasks,
            uncertainty=uncertainty,
            uncertainty_weight=0.1,
            diversity_weight=0.0,
            task_coverage_weight=5.0,
            seed=0,
        )

        picked_tasks = {tasks[i] for i in result.indices}
        self.assertEqual(picked_tasks, {"vision", "text", "audio"})
        self.assertEqual(result.task_counts, {"vision": 1, "text": 1, "audio": 1})

    def test_cost_aware_design_avoids_expensive_tie(self):
        x = np.array([[0.0], [0.1], [4.0], [5.0]])
        uncertainty = np.array([1.0, 1.0, 0.0, 0.0])
        cost = np.array([10.0, 1.0, 1.0, 1.0])

        result = distillation_design(
            x,
            1,
            uncertainty=uncertainty,
            cost=cost,
            diversity_weight=0.0,
            cost_weight=1.0,
            seed=1,
        )

        self.assertEqual(result.indices.tolist(), [1])

    def test_reference_features_push_design_into_new_region(self):
        x = np.array([[0.0], [0.1], [0.2], [5.0]])
        reference = np.array([[0.0], [0.15]])

        result = distillation_design(
            x,
            1,
            reference_features=reference,
            uncertainty_weight=0.0,
            diversity_weight=1.0,
            seed=0,
        )

        self.assertEqual(result.indices.tolist(), [3])

    def test_equal_length_multi_tag_rows_are_tags_not_numeric_incidence(self):
        x = np.arange(4, dtype=np.float64).reshape(-1, 1)
        modalities = [["text", "image"], ["text", "signal"], ["image", "signal"], ["text", "image"]]

        result = distillation_design(
            x,
            2,
            modalities=modalities,
            uncertainty_weight=0.0,
            diversity_weight=0.0,
            modality_coverage_weight=1.0,
            seed=0,
        )

        self.assertGreaterEqual(result.modality_counts["text"], 1)
        self.assertGreaterEqual(result.modality_counts["image"], 1)

    def test_cross_modal_design_requires_pairs_and_prefers_alignment_gap(self):
        text = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [1.0, 1.0],
                [np.nan, np.nan],
                [0.2, 0.1],
            ]
        )
        image = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.1],
                [-1.0, -1.0],
                [1.0, 1.0],
                [np.nan, np.nan],
            ]
        )
        tasks = ["caption", "caption", "retrieval", "retrieval", "caption"]

        result = cross_modal_distillation_design(
            {"text": text, "image": image},
            2,
            task_labels=tasks,
            required_modalities=["text", "image"],
            uncertainty_weight=1.0,
            diversity_weight=0.1,
            task_coverage_weight=1.0,
            modality_coverage_weight=1.0,
            alignment_weight=2.0,
            seed=0,
        )

        self.assertTrue(set(result.indices).issubset({0, 1, 2}))
        self.assertIn(2, result.indices.tolist())
        self.assertEqual(result.modality_counts, {"text": 2, "image": 2})
        self.assertFalse(np.isnan(result.scores).any())

    def test_deterministic_given_seed(self):
        x = np.random.RandomState(0).normal(size=(30, 4))
        unc = np.random.RandomState(1).uniform(size=30)
        a = distillation_design(x, 6, uncertainty=unc, seed=7)
        b = distillation_design(x, 6, uncertainty=unc, seed=7)
        np.testing.assert_array_equal(a.indices, b.indices)
        np.testing.assert_allclose(a.scores, b.scores)

    def test_eligible_mask_restricts_selection(self):
        x = np.arange(6, dtype=np.float64).reshape(-1, 1)
        eligible = np.array([False, True, True, False, True, False])
        result = distillation_design(x, 2, eligible=eligible, diversity_weight=1.0, seed=0)
        self.assertTrue(set(result.indices).issubset({1, 2, 4}))

    def test_no_selection_larger_than_the_eligible_pool(self):
        x = np.arange(4, dtype=np.float64).reshape(-1, 1)
        with self.assertRaises(ValueError):
            distillation_design(x, 5)  # more than the whole pool
        with self.assertRaises(ValueError):
            distillation_design(x, 3, eligible=np.array([True, True, False, False]))  # only 2 eligible

    def test_invalid_inputs_are_rejected(self):
        x = np.arange(4, dtype=np.float64).reshape(-1, 1)
        with self.assertRaises(ValueError):
            distillation_design(x, 0)  # non-positive n
        with self.assertRaises(ValueError):
            distillation_design(np.empty((0, 2)), 1)  # empty pool
        with self.assertRaises(ValueError):
            distillation_design(x, 1, cost=np.array([1.0, 0.0, 1.0, 1.0]))  # non-positive cost

    def test_selection_is_unique(self):
        x = np.random.RandomState(2).normal(size=(20, 3))
        result = distillation_design(x, 8, seed=3)
        self.assertEqual(len(set(result.indices.tolist())), 8)  # no candidate picked twice


if __name__ == "__main__":
    unittest.main()
