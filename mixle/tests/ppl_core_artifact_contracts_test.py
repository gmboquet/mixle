"""Release contracts for PPL estimator construction, model criteria, and artifacts."""

import pickle
import unittest

import numpy as np

from mixle.ppl import Field, Normal, compare, free
from mixle.ppl.core import Family


class EstimatorConstructionContractTest(unittest.TestCase):
    @staticmethod
    def _family(estimator):
        return Family("Dummy", object, estimator, lambda: {}, 0)

    def test_internal_type_error_is_not_retried(self):
        class BrokenEstimator:
            calls = 0

            def __init__(self, name=None):
                type(self).calls += 1
                raise TypeError("internal constructor defect")

        with self.assertRaisesRegex(TypeError, "internal constructor defect"):
            self._family(BrokenEstimator).make_estimator("named", "key")
        self.assertEqual(BrokenEstimator.calls, 1)

    def test_unsupported_metadata_is_filtered_before_one_call(self):
        class BareEstimator:
            calls = 0

            def __init__(self):
                type(self).calls += 1

        estimator = self._family(BareEstimator).make_estimator("named", "key")
        self.assertIsInstance(estimator, BareEstimator)
        self.assertEqual(BareEstimator.calls, 1)


class ModelCriterionContractTest(unittest.TestCase):
    def setUp(self):
        self.data = list(np.random.RandomState(4).normal(1.5, 0.8, 120))

    def test_aic_bic_use_declared_free_dimension_only(self):
        fixed_scale = Normal(free, 2.0).fit(self.data)
        self.assertEqual(fixed_scale._cache["_free_parameter_count"], 1)
        self.assertEqual(fixed_scale.parameter_dimension, 1)
        ll = fixed_scale.log_likelihood(self.data)
        self.assertAlmostEqual(fixed_scale.aic(self.data), 2.0 - 2.0 * ll)
        self.assertAlmostEqual(fixed_scale.bic(self.data), np.log(len(self.data)) - 2.0 * ll)

        free_scale = Normal(free, free).fit(self.data)
        self.assertEqual(free_scale._cache["_free_parameter_count"], 2)

    def test_compare_materializes_inputs_once_and_validates_before_scoring(self):
        first = Normal(free, 1.0).fit(self.data)
        second = Normal(free, 2.0).fit(self.data)
        rows = compare((model for model in (first, second)), (value for value in self.data), by="aic")
        self.assertEqual(len(rows), 2)
        with self.assertRaisesRegex(ValueError, "by must be"):
            compare([first], self.data, by="unknown")
        with self.assertRaisesRegex(ValueError, "at least one"):
            compare([], self.data)
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            compare([first], [])


class RandomVariableArtifactContractTest(unittest.TestCase):
    def test_regression_result_and_prediction_survive_pickle(self):
        x = np.linspace(-1.0, 1.0, 40)
        y = 0.75 + 1.5 * x
        fitted = Normal(free * Field("x") + free, free).fit(y, given={"x": x})
        restored = pickle.loads(pickle.dumps(fitted))
        expected = fitted.result.predict({"x": [-0.5, 0.5]})
        np.testing.assert_allclose(restored.result.predict({"x": [-0.5, 0.5]}), expected)
        self.assertEqual(restored.summary(), fitted.summary())
        self.assertEqual(restored._cache["_free_parameter_count"], 3)
        self.assertEqual(restored.explain_fit(), fitted.explain_fit())

    def test_indexed_group_key_is_structural_and_serialized(self):
        grouped = Normal(0.0, 1.0).each(by="group")
        restored = pickle.loads(pickle.dumps(grouped))
        self.assertEqual(restored.scope, "grouped")
        self.assertEqual(restored._group_by, "group")
        with self.assertRaises(ValueError):
            Normal(0.0, 1.0).each(by="")


if __name__ == "__main__":
    unittest.main()
