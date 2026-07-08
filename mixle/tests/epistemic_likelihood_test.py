"""mixle.epistemic.likelihood: pluggable reweighting strategies at a declared verifiability tier (Card E3)."""

import unittest

import numpy as np

from mixle.epistemic.likelihood import CallableLikelihood, DiscrepancyLikelihood, LikelihoodStrategy
from mixle.epistemic.portfolio import Hypothesis, HypothesisPortfolio


class CallableLikelihoodTest(unittest.TestCase):
    def test_round_trips_tier_and_computes_the_wrapped_function(self):
        strategy = CallableLikelihood(lambda h, o: float(h.payload == o), tier="executable")
        self.assertEqual(strategy.tier, "executable")
        self.assertEqual(strategy(Hypothesis("h", "x"), "x"), 1.0)
        self.assertEqual(strategy(Hypothesis("h", "x"), "y"), 0.0)

    def test_isinstance_protocol_conformance(self):
        strategy = CallableLikelihood(lambda h, o: 1.0, tier="executable")
        self.assertIsInstance(strategy, LikelihoodStrategy)

    def test_unknown_tier_raises(self):
        with self.assertRaises(ValueError):
            CallableLikelihood(lambda h, o: 1.0, tier="self_graded")


class DiscrepancyLikelihoodTest(unittest.TestCase):
    def test_is_a_drop_in_for_portfolio_reweight(self):
        hyps = [Hypothesis("h0", 0.0), Hypothesis("h1", 2.0), Hypothesis("h2", 5.0)]
        weights = np.array([1 / 3, 1 / 3, 1 / 3])
        portfolio = HypothesisPortfolio(hyps, weights, w_open=0.0)

        class Predicted:
            def __init__(self, loc):
                self.loc = loc

            def log_density(self, x):
                return float(-0.5 * (x - self.loc) ** 2)

            def sample(self, n):
                return np.random.RandomState(0).normal(loc=self.loc, size=n)

        strategy = DiscrepancyLikelihood(lambda h: Predicted(h.payload), tier="simulation", temperature=0.5)
        self.assertIsInstance(strategy, LikelihoodStrategy)
        self.assertEqual(strategy.tier, "simulation")

        # A plain float observation (no log_density) routes discrepancy_report's auto-dispatch to the
        # "predicted is a distribution, observed is a concrete measurement" mmd-over-samples branch.
        rng = np.random.RandomState(0)
        for _ in range(10):
            observation = float(rng.normal(loc=2.0, scale=0.3))
            portfolio = portfolio.reweight(observation, strategy)
        idx = {h.id: i for i, h in enumerate(portfolio.hypotheses)}
        self.assertGreater(portfolio.weights[idx["h1"]], 0.9)

    def test_unknown_tier_raises(self):
        with self.assertRaises(ValueError):
            DiscrepancyLikelihood(lambda h: h, tier="not_a_tier")


if __name__ == "__main__":
    unittest.main()
