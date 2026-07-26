"""Regression contracts for shared automatic-input normalization."""

import unittest

import numpy as np

from mixle.utils.automatic import analyze_structure, get_estimator
from mixle.utils.automatic.profiling import _MappingObservation, _SequenceObservation, normalize_input


class AutomaticObservationNormalizationTest(unittest.TestCase):
    def test_generator_valued_fields_are_materialized_once(self):
        rows = [(index, (value for value in (index, index + 1))) for index in range(40)]
        normalized = normalize_input(rows)
        self.assertIsInstance(normalized[0][1], _SequenceObservation)
        self.assertEqual(tuple(normalized[0][1]), (0, 1))

        profile = analyze_structure(normalized, pairwise=False)
        estimator = get_estimator(normalized)
        self.assertEqual(type(profile.recommend()), type(estimator))

    def test_nested_normalized_values_are_immutable(self):
        normalized = normalize_input([{"values": [1, 2, 3]}])[0]
        self.assertIsInstance(normalized, _MappingObservation)
        self.assertIsInstance(normalized["values"], _SequenceObservation)
        with self.assertRaises((AttributeError, TypeError)):
            normalized["values"].values = ()

    def test_nonfinite_values_do_not_change_numeric_type_inference(self):
        data = [1.0, 2.0, 3.0, np.inf, -np.inf, np.nan] * 20
        profile = analyze_structure(data, pairwise=False)
        field = profile.fields[0]
        self.assertEqual(field.kind, "numeric")
        self.assertEqual(field.missing_count, 60)
        self.assertNotEqual(field.recommendation, "ignored")
        self.assertIn("non-finite", " ".join(field.notes))

    def test_mixed_string_integer_values_are_consistently_categorical(self):
        data = ["low", 2, "high", 3] * 40
        profile = analyze_structure(data, pairwise=False)
        self.assertEqual(profile.fields[0].kind, "mixed_categorical")
        self.assertEqual(profile.fields[0].recommendation, "categorical")
        self.assertIn("Categorical", type(profile.recommend()).__name__)

    def test_boolean_numeric_collision_fails_closed(self):
        data = [True, 1, False, 0] * 40
        profile = analyze_structure(data, pairwise=False)
        self.assertEqual(profile.fields[0].recommendation, "ignored")
        self.assertIn("Ignored", type(profile.recommend()).__name__)
        self.assertIn("True == 1", " ".join(profile.fields[0].notes))


if __name__ == "__main__":
    unittest.main()
