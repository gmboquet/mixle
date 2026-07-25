"""Holdout isolation, overlap, and uncertainty contracts for learned policy promotion."""

import unittest

import numpy as np

from mixle.inference.orchestration import learn_placement_policy, meta_improve


def _cost(size, choice):
    return 1.0 + 2.0 * size if choice == "local" else 5.0 + 0.2 * size


def _paired_training():
    return [
        ({"size": float(size)}, choice, {"cost": _cost(size, choice)})
        for size in np.linspace(0.0, 10.0, 40)
        for choice in ("local", "pool")
    ]


def _always_local(_features):
    return "local"


class RealizedEvaluationContractsTest(unittest.TestCase):
    def test_policy_evaluation_never_imputes_missing_heldout_actions(self):
        policy = learn_placement_policy(_paired_training(), _always_local)
        logged_only = [
            ({"size": 20.0 + index / 100.0}, ["local", "pool"][index % 2], {"cost": float(index + 1)})
            for index in range(20)
        ]
        evaluation = policy.evaluate(logged_only)
        self.assertEqual(evaluation["n"], 0)
        self.assertEqual(evaluation["overlap_fraction"], 0.0)
        self.assertIsNone(evaluation["learned_mean_cost"])
        self.assertIsNone(evaluation["static_mean_cost"])

    def test_separate_matched_subsets_cannot_trigger_promotion(self):
        # At every held-out context the learned policy selects pool and the
        # static policy selects local, but only one action was logged. The old
        # evaluator compared cheap pool rows with expensive local rows and
        # promoted from selection imbalance.
        logged_only = [
            (
                {"size": 20.0 + index / 100.0},
                "pool" if index % 2 == 0 else "local",
                {"cost": 1.0 if index % 2 == 0 else 100.0},
            )
            for index in range(40)
        ]
        result = meta_improve(_paired_training(), _always_local, holdout_rows=logged_only)
        self.assertFalse(result["promoted"])
        self.assertEqual(result["receipt"]["n_evaluated"], 0)
        self.assertIn("insufficient common", result["receipt"]["reason"])

    def test_propensity_logged_holdout_uses_same_rows_for_both_policies(self):
        randomized_holdout = [
            (
                {"size": 20.0 + index / 100.0},
                "pool" if index % 2 == 0 else "local",
                {
                    "cost": 1.0 if index % 2 == 0 else 100.0,
                    "propensities": {"local": 0.5, "pool": 0.5},
                },
            )
            for index in range(100)
        ]
        result = meta_improve(_paired_training(), _always_local, holdout_rows=randomized_holdout)
        self.assertTrue(result["promoted"])
        receipt = result["receipt"]
        self.assertEqual(receipt["evaluation_method"], "inverse_propensity_weighted")
        self.assertEqual(receipt["overlap_fraction"], 1.0)
        self.assertLess(receipt["upper_confidence_bound"], 0.0)

    def test_improvement_and_uncertainty_thresholds_gate_promotion(self):
        paired_holdout = [
            ({"size": 20.0 + index / 100.0}, choice, {"cost": _cost(20.0 + index / 100.0, choice)})
            for index in range(30)
            for choice in ("local", "pool")
        ]
        result = meta_improve(
            _paired_training(),
            _always_local,
            holdout_rows=paired_holdout,
            min_improvement=100.0,
        )
        self.assertFalse(result["promoted"])
        self.assertEqual(result["receipt"]["evaluation_method"], "paired_realized_outcomes")
        self.assertGreater(result["receipt"]["upper_confidence_bound"], -100.0)

    def test_external_holdout_must_be_context_disjoint(self):
        with self.assertRaises(ValueError):
            meta_improve(
                _paired_training(),
                _always_local,
                holdout_rows=[({"size": 0.0}, "local", {"cost": 1.0})],
            )

    def test_controls_and_required_outcomes_are_validated(self):
        with self.assertRaises(ValueError):
            learn_placement_policy([({"x": 1.0}, "a", {})], _always_local)
        with self.assertRaises(ValueError):
            meta_improve(_paired_training(), _always_local, holdout_frac=1.0)
        with self.assertRaises(ValueError):
            meta_improve(_paired_training(), _always_local, min_overlap=0.0)
        with self.assertRaises(ValueError):
            meta_improve(_paired_training(), _always_local, confidence_level=0.5)


if __name__ == "__main__":
    unittest.main()
