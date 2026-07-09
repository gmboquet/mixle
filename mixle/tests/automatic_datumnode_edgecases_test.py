"""DatumNode edge cases found in a follow-up audit of mixle.utils.automatic (beyond factories.py's
crash-on-empty-vdict and detectors' dropped-fit/pseudo_count, fixed separately):

1. A sequence-typed field whose every observed value is EMPTY (e.g. an always-empty tags/history
   list) crashed with IndexError -- DatumNode._merged_child() indexed self.children[0] with zero
   children, since no element was ever added to merge a type from.
2. The modality-fingerprint checks (_fixed_numeric_vector_dim/_fixed_numeric_matrix_shape) never
   checked dict_count, so a field mixing dict rows with fixed-length numeric-vector rows could still
   report a vector/matrix dimension and emit a "routed to a hybrid neural density" warning -- while
   the actual estimator (correctly) drops the field entirely as unmodelable (mixed container kinds).
"""

import unittest

import numpy as np

from mixle.inference.estimation import fit
from mixle.utils.automatic import analyze_structure, get_estimator


class AlwaysEmptySequenceTest(unittest.TestCase):
    def test_bare_list_of_empty_lists_does_not_crash(self):
        est = get_estimator([[], [], []])
        self.assertEqual(type(est).__name__, "SequenceEstimator")

    def test_nested_always_empty_list_field_does_not_crash(self):
        data = [{"tags": []}, {"tags": []}, {"tags": []}]
        est = get_estimator(data)
        self.assertEqual(type(est).__name__, "RecordEstimator")

    def test_the_length_model_still_correctly_says_always_empty(self):
        data = [[], [], [], []]
        model = fit(data, get_estimator(data), max_its=5, out=None)
        # every observed length was 0 -- the fitted length distribution must say so, and the model
        # must not crash scoring the (empty) sequences it was fit on.
        for x in data:
            ll = model.log_density(x)
            self.assertTrue(ll == float("-inf") or ll <= 0.0)

    def test_a_mix_of_empty_and_non_empty_sequences_is_unaffected(self):
        # sanity: the fix only changes behavior when self.children is EMPTY; a field with at least
        # one non-empty observation must still merge element types exactly as before.
        data = [[], [1.0, 2.0], [], [3.0]]
        est = get_estimator(data)
        self.assertEqual(type(est).__name__, "SequenceEstimator")


class ModalityFingerprintDictGuardTest(unittest.TestCase):
    def test_mixed_dict_and_vector_rows_does_not_claim_an_embedding_dimension(self):
        data = [{"a": 1}, [0.0] * 16, {"a": 2}, [0.0] * 16] * 5
        profile = analyze_structure(data, pairwise=False)
        self.assertFalse(any("modality fingerprint" in w for w in profile.warnings))
        # the actual estimator (mixed container kinds) drops the field -- the diagnostic must agree.
        self.assertEqual(type(get_estimator(data)).__name__, "IgnoredEstimator")

    def test_mixed_dict_and_matrix_rows_does_not_claim_a_matrix_shape(self):
        row = [[0.0] * 4 for _ in range(4)]
        data = [{"a": 1}, row, {"a": 2}, row] * 5
        profile = analyze_structure(data, pairwise=False)
        self.assertFalse(any("modality fingerprint" in w for w in profile.warnings))

    def test_pure_embedding_field_without_dict_rows_is_unaffected(self):
        # sanity: the dict_count guard must not suppress the warning when there are genuinely no
        # dict rows -- only the mixed case should change.
        data = [[float(i)] * 16 for i in range(20)]
        profile = analyze_structure(data, pairwise=False)
        self.assertTrue(any("modality fingerprint" in w for w in profile.warnings))


class InfiniteFloatValueTest(unittest.TestCase):
    """Regression: DatumNode tracked infinite float values (inf_count) but, unlike None/NaN
    (none_count/nan_count), never wrapped the resulting estimator to account for them -- add_datum
    excludes non-finite floats from vdict, so get_estimator() returned a plain estimator (e.g.
    GaussianEstimator) that had never seen the infinities, and fitting that estimator on the SAME
    unfiltered raw data then crashed (GaussianDistribution requires finite support)."""

    def test_mixed_finite_and_positive_infinity_does_not_crash_optimize(self):
        from mixle.inference.estimation import optimize

        data = [1.0, 2.0, float("inf"), 3.0]
        est = get_estimator(data)
        self.assertEqual(type(est).__name__, "OptionalEstimator")  # not a bare GaussianEstimator
        model = optimize(data, out=None)
        self.assertTrue(np.isfinite(model.log_density(2.0)))
        self.assertEqual(model.log_density(float("inf")), 0.0)

    def test_mixed_finite_and_negative_infinity_does_not_crash_optimize(self):
        from mixle.inference.estimation import optimize

        data = [1.0, 2.0, -float("inf"), 3.0]
        model = optimize(data, out=None)
        self.assertTrue(np.isfinite(model.log_density(2.0)))
        self.assertEqual(model.log_density(-float("inf")), 0.0)

    def test_both_signs_of_infinity_present_are_both_handled(self):
        from mixle.inference.estimation import optimize

        data = [1.0, -float("inf"), 2.0, float("inf"), 3.0]
        model = optimize(data, out=None)
        self.assertTrue(np.isfinite(model.log_density(2.0)))
        self.assertEqual(model.log_density(float("inf")), 0.0)
        self.assertEqual(model.log_density(-float("inf")), 0.0)

    def test_all_infinite_field_does_not_silently_disappear(self):
        from mixle.inference.estimation import optimize

        data = [float("inf")] * 5
        est = get_estimator(data)
        self.assertEqual(type(est).__name__, "OptionalEstimator")  # not a bare IgnoredEstimator
        model = optimize(data, out=None)
        self.assertEqual(model.log_density(float("inf")), 0.0)

    def test_finite_only_data_is_unaffected(self):
        est = get_estimator([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(type(est).__name__, "GaussianEstimator")


if __name__ == "__main__":
    unittest.main()
