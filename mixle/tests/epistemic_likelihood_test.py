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

    def test_out_of_domain_wrapped_output_is_rejected(self):
        # MXR-080-1745: a wrapped function returning a negative/NaN/infinite "likelihood" used to be
        # passed straight through, where it moved belief mass and produced out-of-range surprise.
        for bad in (-1.0, -2.0, float("nan"), float("inf")):
            strategy = CallableLikelihood(lambda h, o, v=bad: v, tier="executable")
            with self.subTest(value=repr(bad)), self.assertRaises(ValueError):
                strategy(Hypothesis("h", None), "o")

    def test_zero_is_a_legitimate_wrapped_likelihood(self):
        strategy = CallableLikelihood(lambda h, o: 0.0, tier="executable")
        self.assertEqual(strategy(Hypothesis("h", None), "o"), 0.0)

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

    def test_zero_temperature_rejected_at_construction(self):
        # temperature=0 must not build: exp(-discrepancy / 0) would otherwise crash with
        # ZeroDivisionError inside __call__, far from the constructor call that caused it.
        with self.assertRaises(ValueError):
            DiscrepancyLikelihood(lambda h: h, tier="simulation", temperature=0.0)

    def test_negative_temperature_rejected_at_construction(self):
        # A negative temperature flips the sign in the exponent and reverses the discrepancy
        # ordering (a worse match would score a *higher* likelihood); reject it up front.
        with self.assertRaises(ValueError):
            DiscrepancyLikelihood(lambda h: h, tier="simulation", temperature=-1.0)

    def test_nan_temperature_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            DiscrepancyLikelihood(lambda h: h, tier="simulation", temperature=float("nan"))

    def test_infinite_temperature_rejected_at_construction(self):
        # temperature=inf would otherwise make exp(-discrepancy / inf) collapse to exactly 1 for
        # any finite discrepancy, silently discarding all discriminating power.
        with self.assertRaises(ValueError):
            DiscrepancyLikelihood(lambda h: h, tier="simulation", temperature=float("inf"))

    def test_valid_temperature_preserves_discrepancy_ordering(self):
        # Negative control: a valid positive finite temperature still constructs and scores fine,
        # and a worse (larger) discrepancy must score a strictly lower likelihood than a better one
        # -- the ordering a negative temperature would otherwise reverse.
        class Predicted:
            def __init__(self, loc):
                self.loc = loc

            def log_density(self, x):
                return float(-0.5 * (x - self.loc) ** 2)

            def sample(self, n):
                return np.random.RandomState(0).normal(loc=self.loc, size=n)

        strategy = DiscrepancyLikelihood(lambda h: Predicted(h.payload), tier="simulation", temperature=1.0)
        observation = 2.0
        close_likelihood = strategy(Hypothesis("h_close", 2.1), observation)
        far_likelihood = strategy(Hypothesis("h_far", 50.0), observation)
        self.assertGreater(close_likelihood, far_likelihood)

    def test_non_finite_scoring_output_is_rejected(self):
        # Internal-consistency check on the class's own output: a NaN discrepancy (e.g. leaking in
        # from a predict_fn bug) must not silently become a NaN "likelihood" that then corrupts
        # portfolio normalization -- it should raise where it happened instead.
        strategy = DiscrepancyLikelihood(lambda h: float("nan"), tier="simulation", temperature=1.0)
        with self.assertRaises(ValueError):
            strategy(Hypothesis("h", None), 5.0)


if __name__ == "__main__":
    unittest.main()
