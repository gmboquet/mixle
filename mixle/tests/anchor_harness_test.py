"""Independent train/calibration/test contract for the two-modality anchor."""

import unittest

from mixle.reason.anchor_harness import AnchorHarnessReport, run_anchor_harness


class AnchorHarnessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = run_anchor_harness(n_train=800, n_calibration=80, n_test=100, seed=0)

    def test_runs_end_to_end_with_disjoint_declared_splits(self):
        report = self.report
        self.assertIsInstance(report, AnchorHarnessReport)
        self.assertEqual((report.training_rows, report.calibration_rows, report.test_rows), (800, 80, 100))
        self.assertTrue(any("training data only" in note for note in report.notes))
        self.assertTrue(any("calibration data only" in note for note in report.notes))

    def test_both_modalities_enter_the_measured_prediction(self):
        report = self.report
        self.assertEqual(set(report.modalities), {"geochemistry", "gravity"})
        self.assertEqual(report.hop_names, ["gravity_to_density", "density_and_geochemistry_to_grade"])
        self.assertGreater(report.gravity_ablation_penalty, 0.0)
        self.assertLess(report.both_modalities_mae, report.gravity_only_mae)

    def test_baseline_is_learned_and_reported_as_an_ablation(self):
        report = self.report
        self.assertGreater(report.gravity_only_mae, 0.0)
        self.assertTrue(any("baselines use training data only" in note for note in report.notes))
        self.assertFalse(hasattr(report, "frontier_mae"))

    def test_calibration_is_a_measured_confidence_bound_not_a_p_value_label(self):
        report = self.report
        self.assertEqual(set(report.coverage_by_hop), {1, 2})
        self.assertGreaterEqual(report.test_coverage, 0.0)
        self.assertLessEqual(report.test_coverage, 1.0)
        self.assertGreaterEqual(report.coverage_lower_bound, 0.0)
        self.assertLessEqual(report.coverage_lower_bound, report.test_coverage)
        self.assertEqual(report.coverage_target, 0.9)
        self.assertIsInstance(report.walk_is_calibrated, bool)

    def test_abstention_is_operational_and_selective_risk_is_reported(self):
        report = self.report
        self.assertGreater(len(report.abstained_site_ids), 0)
        self.assertGreater(report.answer_rate, 0.0)
        self.assertLess(report.answer_rate, 1.0)
        self.assertAlmostEqual(report.answer_rate + report.abstain_rate, 1.0)
        self.assertGreaterEqual(report.accepted_mae, 0.0)
        self.assertGreaterEqual(report.all_answer_mae, 0.0)
        self.assertLess(report.accepted_mae, report.all_answer_mae)

    def test_task_projections_are_of_the_inferred_test_posteriors(self):
        report = self.report
        self.assertGreater(report.driller_projection_components, report.scout_projection_components)
        self.assertGreaterEqual(report.projection_decision_agreement, 0.99)
        self.assertIsInstance(report.driller_readout, str)
        self.assertIsInstance(report.scout_readout, str)

    def test_deterministic_given_seed(self):
        first = run_anchor_harness(n_train=300, n_calibration=40, n_test=40, seed=3)
        second = run_anchor_harness(n_train=300, n_calibration=40, n_test=40, seed=3)
        self.assertEqual(first.coverage_by_hop, second.coverage_by_hop)
        self.assertEqual(first.abstained_site_ids, second.abstained_site_ids)
        self.assertEqual(first.both_modalities_mae, second.both_modalities_mae)
        self.assertEqual(first.projection_decision_agreement, second.projection_decision_agreement)

    def test_too_little_data_fails_before_fitting(self):
        with self.assertRaisesRegex(ValueError, "at least"):
            run_anchor_harness(n_train=8, n_calibration=30, n_test=30, seed=0)


if __name__ == "__main__":
    unittest.main()
