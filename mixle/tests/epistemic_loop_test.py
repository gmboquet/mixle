"""mixle.epistemic.loop: one step of OBSERVE -> UPDATE -> ABDUCE -> ACT (Card E4)."""

import math
import unittest
import warnings

import numpy as np

from mixle.epistemic.loop import _add_hypothesis, _portfolio_eig_nmc, step
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


class SingleLikelihoodEvaluationTest(unittest.TestCase):
    """MXR-080-1752: surprise and the posterior must come from the SAME evidence."""

    def test_each_active_likelihood_is_evaluated_once_per_step(self):
        calls = []

        def counting_likelihood(hypothesis, observation):
            calls.append(hypothesis.id)
            return _gaussian_likelihood(hypothesis, observation)

        step(_toy_portfolio(), 2.0, counting_likelihood)
        self.assertEqual(sorted(calls), ["h0", "h1", "h2"])

    def test_a_stateful_likelihood_cannot_report_two_different_beliefs(self):
        # Alternating values previously fed surprise one pair and the posterior the opposite pair.
        state = {"n": 0}

        def alternating(hypothesis, observation):
            state["n"] += 1
            return 1.0 if state["n"] % 2 else 2.0

        hyps = [Hypothesis("a", None), Hypothesis("b", None)]
        portfolio = HypothesisPortfolio(hyps, np.array([0.5, 0.5]))
        outcome = step(portfolio, "o", alternating)
        self.assertEqual(state["n"], 2)  # two hypotheses, one evaluation each
        # The values actually used were 1.0 and 2.0, so the posterior is exactly [1/3, 2/3] AND the
        # surprise is the score of that same pair -- consistent, not two different experiments.
        self.assertTrue(np.allclose(outcome.portfolio_after.weights, [1 / 3, 2 / 3]))
        self.assertAlmostEqual(outcome.surprise, 1.0 / (1.0 + 1.5), places=12)


class OpenWorldEIGTest(unittest.TestCase):
    """MXR-080-1753: EIG must not silently condition away the open-world mass."""

    @staticmethod
    def _likelihood(hypothesis, observation):
        return float(np.exp(-0.5 * ((observation - hypothesis.payload) / 0.2) ** 2))

    @staticmethod
    def _simulate_fn(hypothesis, action, rng):
        return float(hypothesis.payload + rng.normal(scale=0.2))

    def _eig(self, w_open):
        # Two portfolios identical apart from w_open (the exact reported shape): before the fix both
        # renormalized the known hypotheses to one and returned the very same number.
        hyps = [Hypothesis("h0", 0.0), Hypothesis("h1", 3.0)]
        weights = np.array([(1.0 - w_open) / 2] * 2)
        return _portfolio_eig_nmc(
            HypothesisPortfolio(hyps, weights, w_open=w_open),
            0.0,
            self._likelihood,
            self._simulate_fn,
            np.random.RandomState(0),
            n_outer=64,
            n_inner=32,
        )

    def test_open_world_mass_lowers_the_reported_eig(self):
        closed = self._eig(0.0)
        mostly_open = self._eig(0.99)
        self.assertGreater(closed, 0.4)
        self.assertLess(mostly_open, closed)
        # The conditional estimate is weighted by the probability the model set is complete at all.
        self.assertAlmostEqual(mostly_open, 0.01 * closed, places=10)

    def test_all_open_world_mass_yields_no_expected_gain(self):
        self.assertEqual(self._eig(1.0), 0.0)


class AbductionFundingTest(unittest.TestCase):
    """MXR-080-1754: an abduced hypothesis must get a prior multiplicative updates can revive."""

    def test_a_closed_portfolio_still_funds_the_new_hypothesis(self):
        portfolio = HypothesisPortfolio([Hypothesis("incumbent", 0.0)], np.array([1.0]), w_open=0.0)
        outcome = step(
            portfolio,
            1000.0,
            lambda h, o: 1e-9,
            surprise_threshold=0.5,
            propose_fn=lambda p: Hypothesis("h_new", 1000.0),
        )
        after = outcome.portfolio_after
        idx = {h.id: i for i, h in enumerate(after.hypotheses)}
        self.assertGreater(after.weights[idx["h_new"]], 0.0)
        self.assertLess(after.weights[idx["incumbent"]], 1.0)
        self.assertAlmostEqual(float(after.weights.sum()) + after.w_open, 1.0, places=12)

    def test_a_revivable_prior_actually_revives_under_evidence(self):
        # The whole point of abduction: the new hypothesis must be able to overtake the incumbent it
        # was proposed against. With a weight of 0.0 multiplicative updates could never do that.
        portfolio = HypothesisPortfolio([Hypothesis("incumbent", 0.0)], np.array([1.0]), w_open=0.0)
        after = step(
            portfolio,
            1000.0,
            lambda h, o: 1e-9,
            surprise_threshold=0.5,
            propose_fn=lambda p: Hypothesis("h_new", 1000.0),
        ).portfolio_after
        after = after.reweight(1000.0, _gaussian_likelihood)
        idx = {h.id: i for i, h in enumerate(after.hypotheses)}
        self.assertGreater(after.weights[idx["h_new"]], 0.9)

    def test_open_world_mass_is_the_preferred_funding_source(self):
        hyps = [Hypothesis("a", None), Hypothesis("b", None)]
        portfolio = HypothesisPortfolio(hyps, np.array([0.4, 0.3]), w_open=0.3)
        grown = _add_hypothesis(portfolio, Hypothesis("h_new", None), floor_weight=0.1)
        self.assertAlmostEqual(grown.w_open, 0.2, places=12)
        self.assertTrue(np.allclose(grown.weights[:2], [0.4, 0.3]))  # incumbents untouched

    def test_shortfall_dilutes_incumbents_proportionally(self):
        hyps = [Hypothesis("a", None), Hypothesis("b", None)]
        portfolio = HypothesisPortfolio(hyps, np.array([0.6, 0.3]), w_open=0.1)
        grown = _add_hypothesis(portfolio, Hypothesis("h_new", None), floor_weight=0.3)
        self.assertAlmostEqual(grown.w_open, 0.0, places=12)
        self.assertAlmostEqual(grown.weights[-1], 0.3, places=12)
        # 0.1 came from w_open; the 0.2 shortfall scales the 0.9 of active mass down to 0.7.
        self.assertTrue(np.allclose(grown.weights[:2], np.array([0.6, 0.3]) * (0.7 / 0.9)))
        self.assertAlmostEqual(float(grown.weights.sum()) + grown.w_open, 1.0, places=12)

    def test_a_floor_weight_outside_the_open_interval_is_rejected(self):
        portfolio = HypothesisPortfolio([Hypothesis("a", None)], np.array([1.0]))
        for bad in (0.0, -0.1, 1.0, 2.0, float("nan")):
            with self.subTest(floor_weight=repr(bad)), self.assertRaises(ValueError):
                _add_hypothesis(portfolio, Hypothesis("h_new", None), floor_weight=bad)


class ActionEvidenceValidationTest(unittest.TestCase):
    """MXR-080-1755 / MXR-080-1756: invalid budgets and economics must not silently erase actions."""

    @staticmethod
    def _simulate_fn(hypothesis, action, rng):
        return float(hypothesis.payload + rng.normal(scale=0.2))

    def _step(self, **kwargs):
        return step(
            _toy_portfolio(),
            2.0,
            _gaussian_likelihood,
            action_space=[1.0],
            simulate_fn=self._simulate_fn,
            rng=0,
            **kwargs,
        )

    def test_invalid_sample_budgets_are_rejected(self):
        for budget in (0, -5, 2.5, float("nan"), float("inf"), None, True):
            with self.subTest(n_outer=repr(budget)), self.assertRaises(ValueError):
                self._step(n_outer=budget, n_inner=8)
            with self.subTest(n_inner=repr(budget)), self.assertRaises(ValueError):
                self._step(n_outer=8, n_inner=budget)

    def test_invalid_costs_do_not_make_the_only_candidate_vanish(self):
        for bad in (float("nan"), float("inf"), -1.0):
            with self.subTest(cost=repr(bad)), self.assertRaises(ValueError):
                self._step(n_outer=8, n_inner=8, cost_fn=lambda a, v=bad: v)

    def test_invalid_lam_is_rejected(self):
        for bad in (float("nan"), float("inf"), -1.0):
            with self.subTest(lam=repr(bad)), self.assertRaises(ValueError):
                self._step(n_outer=8, n_inner=8, lam=bad, cost_fn=lambda a: 1.0)

    def test_valid_economics_still_select_an_action(self):
        outcome = self._step(n_outer=8, n_inner=8, lam=0.5, cost_fn=lambda a: 1.0)
        self.assertEqual(outcome.next_action, 1.0)
        self.assertTrue(math.isfinite(outcome.next_action_eig))


if __name__ == "__main__":
    unittest.main()
