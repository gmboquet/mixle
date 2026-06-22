"""Regression tests for HiddenAssociationAccumulator keyed-statistic delegation.

key_merge/key_replace previously delegated only to size_accumulator, so keyed
statistics configured on the conditional (and given) child estimators were
silently never pooled through the shared stats_dict.
"""

import unittest

import numpy as np

from pysp.stats.base.categorical import CategoricalEstimator
from pysp.stats.combinator.conditional import ConditionalDistributionEstimator
from pysp.stats.latent.hidden_association import HiddenAssociationEstimator


def make_accumulator(cond_key="cond_suff_stats", keys=(None, None)):
    """Build a HiddenAssociationAccumulator whose conditional children share a key."""
    cond_est = ConditionalDistributionEstimator(
        {"a": CategoricalEstimator(keys=cond_key), "b": CategoricalEstimator(keys=cond_key)}
    )
    est = HiddenAssociationEstimator(cond_est, keys=keys)
    return est.accumulator_factory().make()


class HiddenAssociationKeysTestCase(unittest.TestCase):
    def test_cond_children_keyed_stats_are_pooled(self):
        rng = np.random.RandomState(1)
        acc1 = make_accumulator()
        acc2 = make_accumulator()

        # Single given value makes the Dirichlet assignment weights degenerate (1.0),
        # so the conditional child counts are deterministic.
        acc1.initialize(([("a", 1.0)], [("x", 1.0), ("y", 2.0)]), 1.0, rng)
        acc2.initialize(([("b", 1.0)], [("z", 3.0)]), 1.0, rng)

        stats_dict = dict()
        acc1.key_merge(stats_dict)
        acc2.key_merge(stats_dict)
        acc1.key_replace(stats_dict)
        acc2.key_replace(stats_dict)

        expected = {"x": 1.0, "y": 2.0, "z": 3.0}
        for acc in (acc1, acc2):
            for given_value in ("a", "b"):
                child_value = acc.cond_accumulator.accumulator_map[given_value].value()
                self.assertEqual(set(child_value.keys()), set(expected.keys()))
                for k, v in expected.items():
                    self.assertAlmostEqual(child_value[k], v)

        self.assertEqual(acc1.value()[0], acc2.value()[0])

    def test_size_accumulator_keyed_stats_still_pooled(self):
        rng = np.random.RandomState(1)
        cond_est = ConditionalDistributionEstimator({"a": CategoricalEstimator()})
        est1 = HiddenAssociationEstimator(cond_est, len_estimator=CategoricalEstimator(keys="size_key"))
        est2 = HiddenAssociationEstimator(cond_est, len_estimator=CategoricalEstimator(keys="size_key"))
        acc1 = est1.accumulator_factory().make()
        acc2 = est2.accumulator_factory().make()

        acc1.initialize(([("a", 1.0)], [("x", 1.0)]), 1.0, rng)
        acc2.initialize(([("a", 1.0)], [("x", 1.0), ("y", 1.0)]), 1.0, rng)

        stats_dict = dict()
        acc1.key_merge(stats_dict)
        acc2.key_merge(stats_dict)
        acc1.key_replace(stats_dict)
        acc2.key_replace(stats_dict)

        expected = {1.0: 1.0, 2.0: 1.0}
        self.assertEqual(acc1.size_accumulator.value(), expected)
        self.assertEqual(acc2.size_accumulator.value(), expected)

    def test_keys_none_construction(self):
        acc = make_accumulator(keys=None)
        self.assertIsNone(acc.init_key)
        self.assertIsNone(acc.trans_key)

        # key_merge/key_replace with no keys configured anywhere is a no-op.
        rng = np.random.RandomState(1)
        acc_no_keys = make_accumulator(cond_key=None, keys=None)
        acc_no_keys.initialize(([("a", 1.0)], [("x", 1.0)]), 1.0, rng)
        before = acc_no_keys.value()

        stats_dict = dict()
        acc_no_keys.key_merge(stats_dict)
        self.assertEqual(stats_dict, dict())
        acc_no_keys.key_replace(stats_dict)
        self.assertEqual(acc_no_keys.value(), before)


if __name__ == "__main__":
    unittest.main()
