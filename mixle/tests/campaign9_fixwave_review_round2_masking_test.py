"""Campaign nine, fix-wave review ROUND 2: regression tests for the per-component masking
granularity fix (D-0209, findings #2/#7/#15 in mixle/stats/compute/declarations.py, and #8 in
four hand-rolled backend_stacked_sufficient_statistics hooks).

The round-1-review fix (campaign9_fixwave_review_test.py's StackedKernelSquareBeforeWeightTest and
GeneratedStatisticsZeroWeightTest) masked a row's raw statistic to 0.0 only when NO stacked
component wants it at all. A row wanted by SOME but not ALL components kept its true, unmasked
value, so a component with EXACTLY zero weight on that row could still be poisoned via
0.0 * inf = nan when another component's nonzero weight left the row's overflow unmasked. Fixed
by masking per (row, component) pair instead of per row, in both declarations.py's generic
_weighted_component_sum (matmul and broadcast-sum branches) and the four families whose
backend_stacked_sufficient_statistics hooks hand-roll the same reduction outside that machinery.
"""

import unittest

import numpy as np

from mixle.engines import NUMPY_ENGINE
from mixle.stats.compute.declarations import StatisticSpec, _weighted_component_sum
from mixle.stats.univariate.continuous.generalized_pareto import GeneralizedParetoDistribution
from mixle.stats.univariate.continuous.gumbel import GumbelDistribution
from mixle.stats.univariate.continuous.nakagami import NakagamiDistribution
from mixle.stats.univariate.continuous.rician import RicianDistribution


class WeightedComponentSumMixedWeightTest(unittest.TestCase):
    """Findings #2/#7/#15: _weighted_component_sum must mask per (row, component), not per row."""

    def test_matmul_branch_zero_weight_component_immune_to_other_components_poison(self):
        n, d = 5, 3
        arr = np.random.RandomState(0).normal(size=(n, d))
        arr[2, :] = -np.inf  # component 0 wants this row; component 1 does not
        weights = np.array(
            [[1.0, 1.0], [1.0, 1.0], [0.6, 0.0], [1.0, 1.0], [1.0, 1.0]],
        )
        spec = StatisticSpec("x", kind="vector_moment")
        out = _weighted_component_sum(arr, spec, weights, NUMPY_ENGINE)
        self.assertFalse(np.any(np.isnan(out[1])), out[1])
        self.assertTrue(np.all(np.isneginf(out[0])), out[0])
        expected_component1 = np.sum(np.delete(arr, 2, axis=0), axis=0)
        np.testing.assert_allclose(out[1], expected_component1)

    def test_broadcast_branch_zero_weight_component_immune_to_other_components_poison(self):
        n = 6
        scalar_stat = np.random.RandomState(1).normal(size=n)
        scalar_stat[4] = -np.inf  # component 1 wants this row; components 0 and 2 do not
        weights = np.array(
            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [0.0, 0.7, 0.0], [1.0, 1.0, 1.0]]
        )
        spec = StatisticSpec("s", kind="sum")
        out = _weighted_component_sum(scalar_stat, spec, weights, NUMPY_ENGINE)
        self.assertFalse(np.isnan(out[0]), out)
        self.assertFalse(np.isnan(out[2]), out)
        self.assertTrue(np.isneginf(out[1]), out)
        expected = np.sum(np.delete(scalar_stat, 4))
        self.assertAlmostEqual(out[0], expected)
        self.assertAlmostEqual(out[2], expected)

    def test_fast_path_is_unaffected_when_nothing_is_poisoned(self):
        rng = np.random.RandomState(2)
        arr = rng.normal(size=(20, 4))
        weights = rng.dirichlet(np.ones(3), size=20)
        spec = StatisticSpec("x", kind="vector_moment")
        out = _weighted_component_sum(arr, spec, weights, NUMPY_ENGINE)
        expected = weights.T @ arr
        np.testing.assert_allclose(out, expected)


class HandRolledStackedHooksMixedWeightTest(unittest.TestCase):
    """Finding #8: the four hand-rolled backend_stacked_sufficient_statistics hooks independently
    reimplemented the same per-row-only masking flaw; each must mask per (row, component)."""

    def _check(self, dist_cls, params, x, extreme_idx, power_indices):
        xx = np.array(x, dtype=float)
        xx[extreme_idx] = 1.0e200
        n = len(xx)
        weights = np.ones((n, 2))
        weights[extreme_idx, 1] = 0.0
        stats = dist_cls.backend_stacked_sufficient_statistics(xx, weights, params, NUMPY_ENGINE)
        for idx in power_indices:
            s = np.asarray(stats[idx])
            self.assertFalse(np.isnan(s[1]), "%s stat[%d] component 1: %r" % (dist_cls.__name__, idx, s))
            self.assertTrue(
                np.isinf(s[0]), "%s stat[%d] component 0 should still see the overflow" % (dist_cls.__name__, idx)
            )
        good = np.delete(xx, extreme_idx)
        good_weights = np.delete(weights[:, 1], extreme_idx)
        return good, good_weights, stats

    def test_gumbel_stacked_stats(self):
        good, good_weights, stats = self._check(
            GumbelDistribution,
            {"loc": np.array([0.0, 0.0]), "scale": np.array([1.0, 1.0])},
            np.random.RandomState(0).normal(size=8),
            3,
            power_indices=(1,),
        )
        np.testing.assert_allclose(np.asarray(stats[1])[1], np.sum(good_weights * good * good))

    def test_generalized_pareto_stacked_stats(self):
        good, good_weights, stats = self._check(
            GeneralizedParetoDistribution,
            {"loc": np.array([0.0, 0.0]), "scale": np.array([1.0, 1.0]), "shape": np.array([0.1, 0.1])},
            np.abs(np.random.RandomState(0).normal(size=8)) + 0.1,
            3,
            power_indices=(1,),
        )
        np.testing.assert_allclose(np.asarray(stats[1])[1], np.sum(good_weights * good * good))

    def test_nakagami_stacked_stats(self):
        good, good_weights, stats = self._check(
            NakagamiDistribution,
            {"m": np.array([1.0, 1.0]), "omega": np.array([1.0, 1.0])},
            np.abs(np.random.RandomState(0).normal(size=8)) + 0.1,
            3,
            power_indices=(1, 2),
        )
        np.testing.assert_allclose(np.asarray(stats[1])[1], np.sum(good_weights * good * good))
        np.testing.assert_allclose(np.asarray(stats[2])[1], np.sum(good_weights * good**4))

    def test_rician_stacked_stats(self):
        good, good_weights, stats = self._check(
            RicianDistribution,
            {"nu": np.array([1.0, 1.0]), "sigma": np.array([1.0, 1.0])},
            np.abs(np.random.RandomState(0).normal(size=8)) + 0.1,
            3,
            power_indices=(1, 2),
        )
        np.testing.assert_allclose(np.asarray(stats[1])[1], np.sum(good_weights * good * good))
        np.testing.assert_allclose(np.asarray(stats[2])[1], np.sum(good_weights * good**4))


if __name__ == "__main__":
    unittest.main()
