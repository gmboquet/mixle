"""mixle.epistemic.portfolio: typed weighted hypotheses + open-world mass (Card E2)."""

import unittest

import numpy as np

from mixle.epistemic.portfolio import Hypothesis, HypothesisPortfolio


def _gaussian_likelihood(hypothesis, observation):
    return float(np.exp(-0.5 * (observation - hypothesis.payload) ** 2))


def _toy_portfolio(w_open=0.0):
    hyps = [Hypothesis("h0", 0.0), Hypothesis("h1", 2.0), Hypothesis("h2", 5.0)]
    weights = np.array([(1.0 - w_open) / 3] * 3)
    return HypothesisPortfolio(hyps, weights, w_open=w_open)


class ReweightConvergenceTest(unittest.TestCase):
    def test_weight_converges_to_the_true_generating_hypothesis(self):
        portfolio = _toy_portfolio()
        rng = np.random.RandomState(0)
        for _ in range(15):
            observation = rng.normal(loc=2.0, scale=1.0)
            portfolio = portfolio.reweight(observation, _gaussian_likelihood)
        idx = {h.id: i for i, h in enumerate(portfolio.hypotheses)}
        self.assertGreater(portfolio.weights[idx["h1"]], 0.95)

    def test_all_zero_likelihood_moves_everything_to_open_world(self):
        portfolio = _toy_portfolio(w_open=0.1)
        result = portfolio.reweight(1e9, lambda h, o: 0.0, open_world_likelihood=lambda o: 0.0)
        self.assertEqual(result.w_open, 1.0)
        self.assertTrue(np.allclose(result.weights, 0.0))


class InvariantTest(unittest.TestCase):
    def test_mass_conservation_across_operations(self):
        portfolio = _toy_portfolio(w_open=0.05)

        def total(p):
            return float(p.weights.sum()) + p.w_open

        self.assertAlmostEqual(total(portfolio), 1.0, places=8)
        portfolio = portfolio.reweight(2.0, _gaussian_likelihood)
        self.assertAlmostEqual(total(portfolio), 1.0, places=8)
        portfolio = portfolio.resample(rng=np.random.RandomState(1))
        self.assertAlmostEqual(total(portfolio), 1.0, places=8)
        portfolio = portfolio.prune(min_weight=0.5)
        self.assertAlmostEqual(total(portfolio), 1.0, places=8)
        active_ids = [h.id for h in portfolio.hypotheses if not h.active]
        if active_ids:
            portfolio = portfolio.resurrect(active_ids[0])
            self.assertAlmostEqual(total(portfolio), 1.0, places=8)

    def test_constructor_rejects_a_broken_invariant(self):
        hyps = [Hypothesis("a", 1), Hypothesis("b", 2)]
        with self.assertRaises(ValueError):
            HypothesisPortfolio(hyps, np.array([0.5, 0.6]), w_open=0.0)

    def test_constructor_rejects_duplicate_hypothesis_ids(self):
        # resample()'s and resurrect()'s docstrings both promise every hypothesis id stays unique;
        # a duplicate would make resurrect() silently reactivate only the first match.
        hyps = [Hypothesis("a", 1), Hypothesis("a", 2), Hypothesis("b", 3)]
        with self.assertRaises(ValueError):
            HypothesisPortfolio(hyps, np.array([0.2, 0.3, 0.5]), w_open=0.0)

    def test_constructor_rejects_nan_weight(self):
        # `nan < -1e-9` is always False, and the sum-to-1 check (`abs(total - 1.0) > 1e-6`) is also
        # always False once a NaN reaches it, so a NaN weight used to construct silently.
        hyps = [Hypothesis("a", 1)]
        with self.assertRaises(ValueError):
            HypothesisPortfolio(hyps, np.array([float("nan")]), w_open=0.0)

    def test_constructor_rejects_nan_w_open(self):
        # Same blind spot as the weight case, on the w_open range check instead.
        hyps = [Hypothesis("a", 1)]
        with self.assertRaises(ValueError):
            HypothesisPortfolio(hyps, np.array([0.5]), w_open=float("nan"))

    def test_constructor_rejects_infinite_weight(self):
        hyps = [Hypothesis("a", 1)]
        with self.assertRaises(ValueError):
            HypothesisPortfolio(hyps, np.array([float("inf")]), w_open=0.0)

    def test_constructor_rejects_infinite_w_open(self):
        hyps = [Hypothesis("a", 1)]
        with self.assertRaises(ValueError):
            HypothesisPortfolio(hyps, np.array([0.5]), w_open=float("inf"))

    def test_reweight_rejects_a_nan_producing_likelihood_fn(self):
        # reweight() has no validation of its own -- it always returns HypothesisPortfolio(...), so a
        # buggy likelihood_fn that returns NaN must be caught by the constructor it delegates to,
        # same as every other mutating method (resample/prune/resurrect).
        portfolio = _toy_portfolio()
        with self.assertRaises(ValueError):
            portfolio.reweight(0.0, lambda h, o: float("nan"))


class PruneResurrectRoundTripTest(unittest.TestCase):
    def test_pruned_mass_folds_into_open_world_and_resurrect_reverses_it(self):
        hyps = [Hypothesis("a", 1), Hypothesis("b", 2), Hypothesis("c", 3)]
        portfolio = HypothesisPortfolio(hyps, np.array([0.02, 0.5, 0.48]), w_open=0.0)
        pruned = portfolio.prune(min_weight=0.05)
        a = next(h for h in pruned.hypotheses if h.id == "a")
        self.assertFalse(a.active)
        self.assertAlmostEqual(pruned.w_open, 0.02, places=8)

        resurrected = pruned.resurrect("a", floor_weight=0.02)
        a2 = next(h for h in resurrected.hypotheses if h.id == "a")
        self.assertTrue(a2.active)
        idx = {h.id: i for i, h in enumerate(resurrected.hypotheses)}
        self.assertAlmostEqual(resurrected.weights[idx["a"]], 0.02, places=8)
        self.assertAlmostEqual(resurrected.w_open, 0.0, places=8)

    def test_resurrecting_an_unknown_id_raises(self):
        portfolio = _toy_portfolio()
        with self.assertRaises(KeyError):
            portfolio.resurrect("does-not-exist")


class SurpriseScoreTest(unittest.TestCase):
    def test_high_for_an_observation_far_from_every_hypothesis(self):
        portfolio = _toy_portfolio()
        self.assertGreater(portfolio.surprise_score(1000.0, _gaussian_likelihood), 0.99)

    def test_lower_for_an_observation_central_to_a_hypothesis_than_for_a_far_one(self):
        portfolio = _toy_portfolio()
        central = portfolio.surprise_score(2.0, _gaussian_likelihood)
        far = portfolio.surprise_score(1000.0, _gaussian_likelihood)
        self.assertLess(central, far)
        self.assertLess(central, 0.8)


class WeightImmutabilityTest(unittest.TestCase):
    """MXR-080-1744: constructor checks run once, so validated weights must not stay writable."""

    def test_public_weights_cannot_be_mutated_in_place(self):
        portfolio = HypothesisPortfolio([Hypothesis("a", None), Hypothesis("b", None)], np.array([0.5, 0.5]))
        with self.assertRaises(ValueError):
            portfolio.weights[:] = [2.0, -1.0]
        self.assertTrue(np.allclose(portfolio.weights, [0.5, 0.5]))

    def test_mutating_the_callers_array_does_not_change_the_portfolio(self):
        weights = np.array([0.5, 0.5])
        portfolio = HypothesisPortfolio([Hypothesis("a", None), Hypothesis("b", None)], weights)
        weights[:] = [2.0, -1.0]
        self.assertTrue(np.allclose(portfolio.weights, [0.5, 0.5]))
        self.assertAlmostEqual(float(portfolio.weights.sum()) + portfolio.w_open, 1.0, places=12)

    def test_derived_portfolios_are_still_constructible_and_frozen(self):
        # Every mutating method rebuilds weights, so freezing must not break the update path.
        portfolio = _toy_portfolio(w_open=0.05).reweight(2.0, _gaussian_likelihood).prune(min_weight=0.01)
        self.assertFalse(portfolio.weights.flags.writeable)
        self.assertAlmostEqual(float(portfolio.weights.sum()) + portfolio.w_open, 1.0, places=10)
        self.assertTrue(portfolio.weights.copy().flags.writeable)  # scratch copies still work


class LikelihoodDomainTest(unittest.TestCase):
    """MXR-080-1745: invalid likelihoods must fail closed, not become plausible belief/surprise."""

    def test_negative_likelihood_is_not_reinterpreted_as_no_support(self):
        portfolio = HypothesisPortfolio([Hypothesis("a", None), Hypothesis("b", None)], np.array([0.5, 0.5]))
        with self.assertRaises(ValueError):
            portfolio.reweight("o", lambda h, o: -1.0)

    def test_negative_open_world_likelihood_is_rejected(self):
        portfolio = _toy_portfolio(w_open=0.1)
        with self.assertRaises(ValueError):
            portfolio.reweight(1.0, _gaussian_likelihood, open_world_likelihood=lambda o: -1.0)

    def test_zero_likelihood_remains_legitimate(self):
        # The documented all-mass-to-w_open outcome must survive the new validation.
        result = _toy_portfolio(w_open=0.1).reweight(1e9, lambda h, o: 0.0, open_world_likelihood=lambda o: 0.0)
        self.assertEqual(result.w_open, 1.0)

    def test_surprise_score_refuses_out_of_range_producing_likelihoods(self):
        portfolio = _toy_portfolio()
        for bad in (-1.0, -2.0, float("nan"), float("inf")):
            with self.subTest(likelihood=bad), self.assertRaises(ValueError):
                portfolio.surprise_score("o", lambda h, o, v=bad: v)

    def test_surprise_score_stays_in_range_for_valid_likelihoods(self):
        portfolio = _toy_portfolio()
        for good in (0.0, 1e-12, 1.0, 1e6):
            score = portfolio.surprise_score("o", lambda h, o, v=good: v)
            self.assertGreaterEqual(score, 0.0)
            self.assertLess(score, 1.0 + 1e-12)


class ResampleValidationTest(unittest.TestCase):
    """MXR-080-1759 -- `resample`'s contract must not depend on the current weights.

    `method` was only inspected after the ESS early return, and `ess_threshold` was never validated
    at all, so the same call raised or silently no-opped depending on the data it was handed.
    """

    def test_unknown_method_raises_even_when_ess_clears_the_threshold(self):
        # Uniform weights => ESS == n, so the ESS branch returns early and never reaches the method
        # dispatch; the invalid method must still be reported.
        portfolio = _toy_portfolio()
        with self.assertRaises(ValueError):
            portfolio.resample(method="bogus", rng=np.random.RandomState(0))

    def test_unknown_method_still_raises_when_a_resample_would_happen(self):
        portfolio = HypothesisPortfolio([Hypothesis("h0", 0.0), Hypothesis("h1", 1.0)], np.array([0.99, 0.01]))
        with self.assertRaises(ValueError):
            portfolio.resample(method="bogus", rng=np.random.RandomState(0))

    def test_out_of_domain_thresholds_are_rejected(self):
        portfolio = _toy_portfolio()
        for bad in (-0.5, 1.5, float("nan"), float("inf"), -float("inf")):
            with self.subTest(ess_threshold=bad), self.assertRaises(ValueError):
                portfolio.resample(ess_threshold=bad, rng=np.random.RandomState(0))

    def test_boundary_thresholds_are_accepted(self):
        portfolio = _toy_portfolio()
        for good in (0.0, 0.5, 1.0):
            with self.subTest(ess_threshold=good):
                out = portfolio.resample(ess_threshold=good, rng=np.random.RandomState(0))
                self.assertAlmostEqual(float(out.weights.sum()) + out.w_open, 1.0, places=8)

    def test_zero_threshold_disables_resampling_instead_of_a_negative_one(self):
        # A negative threshold used to be the (silent) way to disable resampling; 0.0 is the
        # in-domain way to ask for the same thing and must keep working.
        portfolio = HypothesisPortfolio([Hypothesis("h0", 0.0), Hypothesis("h1", 1.0)], np.array([0.99, 0.01]))
        out = portfolio.resample(ess_threshold=0.0, rng=np.random.RandomState(0))
        self.assertIs(out, portfolio)


class SerializationRoundTripTest(unittest.TestCase):
    def test_to_dict_from_dict_round_trips_exactly(self):
        portfolio = _toy_portfolio(w_open=0.1).prune(min_weight=0.5)
        restored = HypothesisPortfolio.from_dict(portfolio.to_dict())
        self.assertTrue(np.allclose(restored.weights, portfolio.weights))
        self.assertAlmostEqual(restored.w_open, portfolio.w_open, places=10)
        self.assertEqual([h.id for h in restored.hypotheses], [h.id for h in portfolio.hypotheses])
        self.assertEqual([h.active for h in restored.hypotheses], [h.active for h in portfolio.hypotheses])
        self.assertEqual([h.payload for h in restored.hypotheses], [h.payload for h in portfolio.hypotheses])


if __name__ == "__main__":
    unittest.main()
