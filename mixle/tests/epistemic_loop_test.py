"""mixle.epistemic.loop: one step of OBSERVE -> UPDATE -> ABDUCE -> ACT (Card E4)."""

import unittest

import numpy as np

from mixle.epistemic.loop import step
from mixle.epistemic.portfolio import Hypothesis, HypothesisPortfolio


def _gaussian_likelihood(hypothesis, observation):
    return float(np.exp(-0.5 * (observation - hypothesis.payload) ** 2))


def _toy_portfolio(w_open=0.0):
    hyps = [Hypothesis("h0", 0.0), Hypothesis("h1", 2.0), Hypothesis("h2", 5.0)]
    weights = np.array([(1.0 - w_open) / 3] * 3)
    return HypothesisPortfolio(hyps, weights, w_open=w_open)


class UpdateOnlyStepTest(unittest.TestCase):
    def test_repeated_steps_converge_like_direct_reweight(self):
        portfolio = _toy_portfolio()
        rng = np.random.RandomState(0)
        for _ in range(15):
            observation = rng.normal(loc=2.0, scale=1.0)
            outcome = step(portfolio, observation, _gaussian_likelihood)
            portfolio = outcome.portfolio_after
        idx = {h.id: i for i, h in enumerate(portfolio.hypotheses)}
        self.assertGreater(portfolio.weights[idx["h1"]], 0.95)

    def test_no_action_space_returns_none_action(self):
        portfolio = _toy_portfolio()
        outcome = step(portfolio, 2.0, _gaussian_likelihood)
        self.assertIsNone(outcome.next_action)
        self.assertIsNone(outcome.next_action_eig)


class ActionSelectionTest(unittest.TestCase):
    def test_picks_the_action_with_higher_eig_near_the_dominant_hypothesis(self):
        # Two candidate "probe locations"; probing near the already-dominant hypothesis (2.0) is more
        # informative than probing somewhere no hypothesis predicts anything (100.0): the simulator
        # returns near-flat, uninformative noise for the latter, and a peaked, discriminating signal
        # for the former.
        hyps = [Hypothesis("h0", 0.0), Hypothesis("h1", 2.0)]
        weights = np.array([0.05, 0.95])
        portfolio = HypothesisPortfolio(hyps, weights, w_open=0.0)

        def simulate_fn(hypothesis, action, rng):
            if abs(action - 2.0) < 1e-6:
                return float(hypothesis.payload + rng.normal(scale=0.05))
            return float(rng.normal(scale=0.05))  # uninformative regardless of hypothesis

        def likelihood(hypothesis, observation):
            target = hypothesis.payload
            return float(np.exp(-0.5 * ((observation - target) / 0.05) ** 2))

        outcome = step(
            portfolio,
            2.0,
            likelihood,
            action_space=[2.0, 100.0],
            simulate_fn=simulate_fn,
            n_outer=64,
            n_inner=32,
            rng=0,
        )
        self.assertEqual(outcome.next_action, 2.0)
        self.assertIsNotNone(outcome.next_action_eig)

    def test_action_space_without_simulate_fn_raises(self):
        portfolio = _toy_portfolio()
        with self.assertRaises(ValueError):
            step(portfolio, 2.0, _gaussian_likelihood, action_space=[1.0, 2.0])


class SurpriseAbductionTest(unittest.TestCase):
    def test_a_surprising_observation_triggers_propose_fn_and_grows_the_new_hypothesis(self):
        portfolio = _toy_portfolio(w_open=0.3)
        proposed = {"called": False}

        def propose_fn(current_portfolio):
            proposed["called"] = True
            return Hypothesis("h_new", 1000.0)

        outcome = step(
            portfolio,
            1000.0,
            _gaussian_likelihood,
            surprise_threshold=0.5,
            propose_fn=propose_fn,
        )
        self.assertTrue(proposed["called"])
        ids = [h.id for h in outcome.portfolio_after.hypotheses]
        self.assertIn("h_new", ids)

    def test_below_threshold_does_not_call_propose_fn(self):
        portfolio = _toy_portfolio()
        proposed = {"called": False}

        def propose_fn(current_portfolio):
            proposed["called"] = True
            return Hypothesis("h_new", 1000.0)

        step(portfolio, 2.0, _gaussian_likelihood, surprise_threshold=0.999, propose_fn=propose_fn)
        self.assertFalse(proposed["called"])


if __name__ == "__main__":
    unittest.main()
