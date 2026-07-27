"""Regression test: MarkovChainEstimator.estimate1() must not crash when called directly with its own
default pseudo_count=None.

estimate()'s dispatcher only ever routes to estimate1() when self.pseudo_count is not None, so the bug
was invisible through the normal .estimate() path -- but estimate1() is itself a public method with no
guard of its own, and its docstring never says it requires pseudo_count to be set.
"""

import unittest

from mixle.stats.sequences.markov_chain import MarkovChainAccumulatorFactory, MarkovChainEstimator


class MarkovChainEstimate1DirectCallTestCase(unittest.TestCase):
    def test_estimate1_with_default_pseudo_count_does_not_raise(self):
        est = MarkovChainEstimator()  # pseudo_count defaults to None
        self.assertIsNone(est.pseudo_count)
        acc = MarkovChainAccumulatorFactory().make()
        acc.update(["a", "b", "a", "b", "a"], 1.0, None)
        fit = est.estimate1(None, acc.value())
        self.assertAlmostEqual(sum(fit.init_prob_map.values()), 1.0, places=10)

    def test_estimate1_agrees_with_estimate_on_pseudo_count_free_probabilities(self):
        # estimate() falls back to estimate0() (a sparse, only-observed-keys MLE) when pseudo_count is
        # None; estimate1() instead densifies over every key it has seen (including e.g. "c", which
        # never starts a sequence, at explicit probability 0.0) -- a real, intentional design
        # difference for levels unification, not something these two paths need to agree on. What they
        # DO need to agree on: the actual probability mass for keys estimate0() reports at all.
        est = MarkovChainEstimator()
        acc = MarkovChainAccumulatorFactory().make()
        acc.update(["a", "b", "a", "b", "a", "c", "a"], 1.0, None)
        acc.update(["b", "a", "c", "a"], 1.0, None)
        via_estimate1 = est.estimate1(None, acc.value())
        via_estimate = est.estimate(None, acc.value())
        for k, v in via_estimate.init_prob_map.items():
            self.assertAlmostEqual(via_estimate1.init_prob_map[k], v, places=10)
        for k1, row in via_estimate.transition_map.items():
            for k2, v in row.items():
                self.assertAlmostEqual(via_estimate1.transition_map[k1][k2], v, places=10)


if __name__ == "__main__":
    unittest.main()
