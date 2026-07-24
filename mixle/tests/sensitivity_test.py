"""Global sensitivity analysis: Sobol indices vs the Ishigami analytic values, Morris screening (Phase 4).

MXR-080-0192 (shared bounds/count/name/model-output validation contract) is covered below the
original Phase 4 coverage.
"""

import unittest
import warnings

import numpy as np

from mixle.doe import dgsm, fast_indices, morris_screening, sobol_indices


def ishigami(x, a=7.0, b=0.1):
    return np.sin(x[:, 0]) + a * np.sin(x[:, 1]) ** 2 + b * x[:, 2] ** 4 * np.sin(x[:, 0])


BOUNDS = [(-np.pi, np.pi)] * 3


class SobolTest(unittest.TestCase):
    def setUp(self):
        self.res = sobol_indices(ishigami, BOUNDS, n=16384, seed=0, names=["x1", "x2", "x3"])

    def test_first_order_matches_analytic_ishigami(self):
        np.testing.assert_allclose(self.res["S1"], [0.314, 0.442, 0.0], atol=0.03)

    def test_total_order_matches_analytic_ishigami(self):
        np.testing.assert_allclose(self.res["ST"], [0.557, 0.442, 0.244], atol=0.03)

    def test_x3_is_pure_interaction(self):
        self.assertLess(self.res["S1"][2], 0.05)  # no main effect
        self.assertGreater(self.res["ST"][2] - self.res["S1"][2], 0.1)  # but interacts (with x1)

    def test_additive_linear_model(self):
        res = sobol_indices(lambda x: x[:, 0] + 2 * x[:, 1] + 3 * x[:, 2], [(0, 1)] * 3, n=8192)
        self.assertAlmostEqual(res["S1"].sum(), 1.0, delta=0.02)  # additive -> first orders partition
        np.testing.assert_allclose(res["S1"], res["ST"], atol=0.02)  # no interactions
        np.testing.assert_allclose(res["S1"] / res["S1"][0], [1.0, 4.0, 9.0], atol=0.2)  # variance ~ coef^2

    def test_constant_output_is_all_zero(self):
        res = sobol_indices(lambda x: np.ones(len(x)), [(0, 1)] * 2, n=512)
        np.testing.assert_array_equal(res["S1"], [0.0, 0.0])
        np.testing.assert_array_equal(res["ST"], [0.0, 0.0])


class MorrisTest(unittest.TestCase):
    def test_ranks_influential_inputs(self):
        m = morris_screening(ishigami, BOUNDS, trajectories=60, seed=1, names=["x1", "x2", "x3"])
        self.assertEqual(m["mu_star"].shape, (3,))
        self.assertTrue(np.all(m["mu_star"] > 0))  # all three move the output (x3 via interaction)


def _linear(x):
    return x[:, 0] + x[:, 1]


def _wrong_cardinality(_x):
    return np.array([1.0, 2.0, 3.0])  # always 3 outputs, regardless of how many input rows were given


def _non_finite_output(x):
    return x[:, 0] + np.nan


class SharedValidationContractTest(unittest.TestCase):
    """MXR-080-0192: bounds/count/name/model-output validation is centralized (``_as_bounds``,
    ``_require_exact_positive_int`` shared with :mod:`mixle.doe.designs`, plus this module's own
    ``_validate_names``/``_eval_model``) and applied identically by every estimator here -- not
    reimplemented (and silently diverging) per function.
    """

    # (estimator, kwargs that make it a cheap-but-otherwise-well-posed call) for a 2D box.
    ESTIMATORS = (
        (sobol_indices, {"n": 32}),
        (morris_screening, {"trajectories": 4}),
        (fast_indices, {"n": 32, "harmonics": 2}),
        (dgsm, {"n": 32}),
    )

    def test_bad_bounds_shape_is_rejected(self):
        for func, kwargs in self.ESTIMATORS:
            with self.subTest(func=func.__name__), self.assertRaises(ValueError):
                func(_linear, [(0, 1, 2), (0, 1, 2)], **kwargs)  # (d, 3), not (d, 2)

    def test_non_finite_bounds_are_rejected(self):
        for func, kwargs in self.ESTIMATORS:
            with self.subTest(func=func.__name__), self.assertRaises(ValueError):
                func(_linear, [(0, np.inf), (0, 1)], **kwargs)

    def test_reversed_bounds_are_rejected(self):
        for func, kwargs in self.ESTIMATORS:
            with self.subTest(func=func.__name__), self.assertRaises(ValueError):
                func(_linear, [(1, 0), (0, 1)], **kwargs)  # lower > upper

    def test_fractional_count_is_rejected(self):
        cases = (
            (sobol_indices, {"n": 8.5}),
            (morris_screening, {"trajectories": 4.5}),
            (fast_indices, {"n": 32.5, "harmonics": 2}),
            (dgsm, {"n": 32.5}),
        )
        for func, kwargs in cases:
            with self.subTest(func=func.__name__), self.assertRaises(ValueError):
                func(_linear, [(0, 1), (0, 1)], **kwargs)

    def test_fractional_harmonics_and_levels_are_also_rejected(self):
        # the estimator-specific extra counts (not just the shared "n") route through the same check.
        with self.assertRaises(ValueError):
            fast_indices(_linear, [(0, 1), (0, 1)], n=32, harmonics=2.5)
        with self.assertRaises(ValueError):
            morris_screening(_linear, [(0, 1), (0, 1)], trajectories=4, levels=3.5)

    def test_wrong_length_names_are_rejected(self):
        for func, kwargs in self.ESTIMATORS:
            with self.subTest(func=func.__name__), self.assertRaises(ValueError):
                func(_linear, [(0, 1), (0, 1)], names=["only_one"], **kwargs)

    def test_wrong_cardinality_model_output_is_rejected(self):
        for func, kwargs in self.ESTIMATORS:
            with self.subTest(func=func.__name__), self.assertRaises(ValueError):
                func(_wrong_cardinality, [(0, 1), (0, 1)], **kwargs)

    def test_non_finite_model_output_is_rejected(self):
        for func, kwargs in self.ESTIMATORS:
            with self.subTest(func=func.__name__), self.assertRaises(ValueError):
                func(_non_finite_output, [(0, 1), (0, 1)], **kwargs)

    def test_zero_count_is_rejected_before_any_computation_not_nan(self):
        # must raise a clean ValueError *before* touching func/np.var -- not warn-and-NaN. Promoting
        # warnings to errors here means any leftover "Mean of empty slice"/"invalid value encountered"
        # RuntimeWarning (the old NaN-index symptom) would itself fail the test.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with self.assertRaises(ValueError):
                sobol_indices(_linear, [(0, 1), (0, 1)], n=0)
            with self.assertRaises(ValueError):
                morris_screening(_linear, [(0, 1), (0, 1)], trajectories=0)
            with self.assertRaises(ValueError):
                fast_indices(_linear, [(0, 1), (0, 1)], n=0, harmonics=2)
            with self.assertRaises(ValueError):
                dgsm(_linear, [(0, 1), (0, 1)], n=0)

    def test_negative_count_is_also_rejected(self):
        with self.assertRaises(ValueError):
            sobol_indices(_linear, [(0, 1), (0, 1)], n=-8)

    def test_well_formed_inputs_still_work_for_every_estimator(self):
        # Negative control: the new validation must not reject a normal, well-posed call.
        for func, kwargs in self.ESTIMATORS:
            with self.subTest(func=func.__name__):
                res = func(_linear, [(0, 1), (0, 1)], seed=0, **kwargs)
                self.assertEqual(len(res["names"]), 2)
                self.assertEqual(res["names"], ["x0", "x1"])

    def test_well_formed_named_inputs_still_work(self):
        res = sobol_indices(_linear, [(0, 1), (0, 1)], n=32, names=["alpha", "beta"])
        self.assertEqual(res["names"], ["alpha", "beta"])


if __name__ == "__main__":
    unittest.main()
