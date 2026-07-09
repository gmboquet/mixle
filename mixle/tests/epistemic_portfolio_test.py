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
