"""Regression test for HiddenMarkovAccumulatorFactory keys handling.

The factory previously replaced any caller-supplied keys tuple with (None, None, None)
due to an inverted None check, silently disabling key-based suff-stat merging for the
HMM initial/transition/emission statistics.
"""

import unittest

from pysp.stats.categorical import CategoricalEstimator
from pysp.stats.hidden_markov import HiddenMarkovEstimator


class HiddenMarkovKeysTestCase(unittest.TestCase):
    def test_factory_preserves_keys(self):
        keys = ("init_k", "trans_k", "emis_k")
        est = HiddenMarkovEstimator([CategoricalEstimator(), CategoricalEstimator()], keys=keys)
        factory = est.accumulator_factory()
        self.assertEqual(factory.keys, keys)
        acc = factory.make()
        self.assertEqual(acc.init_key, "init_k")
        self.assertEqual(acc.trans_key, "trans_k")
        self.assertEqual(acc.state_key, "emis_k")

    def test_factory_defaults_keys_when_none(self):
        est = HiddenMarkovEstimator([CategoricalEstimator(), CategoricalEstimator()], keys=None)
        factory = est.accumulator_factory()
        self.assertEqual(factory.keys, (None, None, None))
        acc = factory.make()
        self.assertIsNone(acc.init_key)
        self.assertIsNone(acc.trans_key)
        self.assertIsNone(acc.state_key)


if __name__ == "__main__":
    unittest.main()
