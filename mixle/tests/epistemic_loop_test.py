"""mixle.epistemic.loop: one step of OBSERVE -> UPDATE -> ABDUCE -> ACT (Card E4)."""

import math
import unittest
import warnings

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


class ZeroWeightActiveHypothesisEIGTest(unittest.TestCase):
    """An ``active=True`` hypothesis may legally carry weight exactly ``0.0`` --
    :class:`~mixle.epistemic.portfolio.HypothesisPortfolio`'s constructor only forces weight ``0.0``
    onto *inactive* hypotheses, not the reverse (e.g. right after
    :meth:`~mixle.epistemic.portfolio.HypothesisPortfolio.reweight` collapses every likelihood and
    moves all mass to ``w_open``). Filtering the EIG routine's "in play" set by ``active`` alone let
    such a hypothesis through: its lone weight renormalized as ``0 / 0``, and the resulting NaN
    sampling distribution crashed ``numpy``'s categorical sampler with ``ValueError: probabilities
    contain NaN``.
    """

    @staticmethod
    def _likelihood(hypothesis, observation):
        return float(np.exp(-0.5 * (observation - hypothesis.payload) ** 2))

    @staticmethod
    def _simulate_fn(hypothesis, action, rng):
        return float(hypothesis.payload + rng.normal(scale=0.2))

    def test_one_active_zero_weight_hypothesis_with_full_open_world_mass_does_not_crash(self):
        # The exact reported shape: a valid portfolio (passes its own constructor validation) with
        # exactly one active hypothesis whose weight is 0.0, and w_open=1.0 (all mass reserved for
        # "none of the above"). Before the fix, this raised ValueError: probabilities contain NaN
        # from inside rng.choice.
        portfolio = HypothesisPortfolio([Hypothesis("h0", 0.0, active=True)], np.array([0.0]), w_open=1.0)

        with warnings.catch_warnings():
            # A 0/0 divide must not even warn -- the fix excludes the zero-weight hypothesis before
            # the sum is taken, removing the NaN at its source rather than just catching the eventual
            # crash it causes.
            warnings.simplefilter("error")
            outcome = step(
                portfolio,
                0.0,
                self._likelihood,
                action_space=[1.0, -1.0],
                simulate_fn=self._simulate_fn,
                n_outer=16,
                n_inner=8,
                rng=0,
            )

        # Nothing tracked is credible (no active hypothesis carries positive weight), so there is
        # nothing to discriminate between: EIG abstains to a clean 0.0 -- the same "nothing to
        # compare" contract _portfolio_eig_nmc already used for an empty active set -- instead of a
        # NaN or a crash.
        self.assertEqual(outcome.next_action_eig, 0.0)
        self.assertFalse(math.isnan(outcome.next_action_eig))
        # ACT still completes and picks a (degenerate but well-defined) action.
        self.assertIn(outcome.next_action, [1.0, -1.0])

    def test_two_active_zero_weight_hypotheses_with_full_open_world_mass_does_not_crash(self):
        # Same degenerate shape with two zero-weight active hypotheses rather than one, so the fix
        # isn't accidentally keyed to "exactly one hypothesis in the portfolio".
        hyps = [Hypothesis("h0", 0.0, active=True), Hypothesis("h1", 5.0, active=True)]
        portfolio = HypothesisPortfolio(hyps, np.array([0.0, 0.0]), w_open=1.0)

        outcome = step(
            portfolio,
            0.0,
            self._likelihood,
            action_space=[1.0],
            simulate_fn=self._simulate_fn,
            n_outer=16,
            n_inner=8,
            rng=1,
        )
        self.assertEqual(outcome.next_action_eig, 0.0)

    def test_normal_positive_weight_case_still_computes_informative_eig(self):
        # Regression guard: two ACTIVE hypotheses that both keep positive weight through the UPDATE
        # step (observation 1.5 is equidistant from payloads 0.0 and 3.0, so the symmetric Gaussian
        # likelihood leaves the prior 0.5/0.5 split untouched) must still yield a real, clearly
        # positive EIG -- the fix must not turn every case into the degenerate 0.0 abstain value.
        hyps = [Hypothesis("h0", 0.0), Hypothesis("h1", 3.0)]
        portfolio = HypothesisPortfolio(hyps, np.array([0.5, 0.5]), w_open=0.0)

        outcome = step(
            portfolio,
            1.5,
            self._likelihood,
            action_space=[0.0],
            simulate_fn=self._simulate_fn,
            n_outer=64,
            n_inner=32,
            rng=0,
        )
        # Both hypotheses are still genuinely "in play" post-update -- this is exercising the normal
        # (unaffected) path, not a collapsed one.
        self.assertTrue(np.allclose(outcome.portfolio_after.weights, [0.5, 0.5]))
        # A well-separated, well-sampled two-hypothesis case should land near log(2) ~= 0.693 nats;
        # a generous margin absorbs Monte Carlo noise without accepting the degenerate 0.0/NaN values
        # this test guards against.
        self.assertGreater(outcome.next_action_eig, 0.4)
        self.assertFalse(math.isnan(outcome.next_action_eig))


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
