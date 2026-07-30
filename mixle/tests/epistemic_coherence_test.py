"""mixle.epistemic.coherence: exchangeability / martingale / evidence-conservation checks (Card E6)."""

import unittest

import numpy as np

from mixle.epistemic.coherence import (
    CONSERVATION_ATOL,
    evidence_conservation_deviation,
    evidence_conservation_violation,
    exchangeability_violation,
    martingale_violation,
)
from mixle.epistemic.portfolio import Hypothesis, HypothesisPortfolio


def _gaussian_likelihood(hypothesis, observation):
    return float(np.exp(-0.5 * (observation - hypothesis.payload) ** 2))


def _toy_portfolio():
    hyps = [Hypothesis("h0", 0.0), Hypothesis("h1", 2.0), Hypothesis("h2", 5.0)]
    return HypothesisPortfolio(hyps, np.array([1 / 3, 1 / 3, 1 / 3]), w_open=0.0)


class ExchangeabilityTest(unittest.TestCase):
    def test_well_behaved_likelihood_shows_no_violation(self):
        rng = np.random.RandomState(0)
        observations = [rng.normal(loc=2.0, scale=1.0) for _ in range(6)]
        violation = exchangeability_violation(
            _toy_portfolio(), observations, _gaussian_likelihood, n_permutations=15, rng=1
        )
        self.assertLess(violation, 1e-8)

    def test_order_dependent_likelihood_is_caught(self):
        # A per-POSITION boost (e.g. "every other call") is a red herring here: multiplication is
        # commutative, so a fixed per-position multiplier factors out of the full product regardless
        # of which value lands in which position, and never actually shows up as a permutation
        # violation. A genuinely order-dependent likelihood has to condition on the *relative order of
        # observation values themselves* -- e.g. "boost h1 whenever this observation is larger than
        # the previous one" -- which is what real hidden-state incoherence looks like.
        state = {"prev": None}

        def wide_gaussian_likelihood(hypothesis, observation, sigma=3.0):
            # A wider likelihood than the other tests' -- keeps the posterior from saturating near 0/1
            # after only 6 observations, so the order-dependent perturbation stays visible in the raw
            # weights instead of being swamped by an already near-certain posterior.
            return float(np.exp(-0.5 * ((observation - hypothesis.payload) / sigma) ** 2))

        def order_dependent_likelihood(hypothesis, observation):
            base = wide_gaussian_likelihood(hypothesis, observation)
            if hypothesis.id == "h1":
                if state["prev"] is not None and observation > state["prev"]:
                    base *= 50.0
                state["prev"] = observation
            return base

        rng = np.random.RandomState(0)
        observations = [rng.normal(loc=2.0, scale=1.0) for _ in range(6)]
        violation = exchangeability_violation(
            _toy_portfolio(), observations, order_dependent_likelihood, n_permutations=40, rng=0
        )
        self.assertGreater(violation, 1e-3)


class MartingaleTest(unittest.TestCase):
    def test_self_consistent_predictive_has_small_violation(self):
        portfolio = _toy_portfolio()

        def observation_sampler(rng):
            idx = rng.choice(3, p=portfolio.weights)
            return float(portfolio.hypotheses[idx].payload + rng.normal(scale=1.0))

        violation = martingale_violation(portfolio, observation_sampler, _gaussian_likelihood, n=2000, rng=0)
        self.assertLess(violation, 0.05)

    def test_a_biased_predictive_has_a_larger_violation(self):
        portfolio = _toy_portfolio()

        def biased_sampler(rng):
            # Always draws near h1 regardless of the portfolio's actual weights -- not the model's own
            # predictive distribution, so the martingale property should NOT hold.
            return float(2.0 + rng.normal(scale=0.1))

        violation = martingale_violation(portfolio, biased_sampler, _gaussian_likelihood, n=2000, rng=0)
        self.assertGreater(violation, 0.2)


class EvidenceConservationTest(unittest.TestCase):
    def test_naive_double_ingestion_without_dedup_is_a_violation(self):
        violation = evidence_conservation_violation(_toy_portfolio(), 2.0, _gaussian_likelihood)
        self.assertTrue(violation)

    def test_dedup_aware_likelihood_shows_no_violation(self):
        # A whole reweight() call evaluates the likelihood once per hypothesis, so "have I already
        # fully scored this observation" only becomes true after that many calls -- tracking dedup by
        # raw call count (rather than a naive "seen the value once" flag) survives that without
        # accidentally deduping mid-way through the FIRST, real ingestion.
        portfolio = _toy_portfolio()
        n_hypotheses = len(portfolio)
        counts = {}

        def dedup_likelihood(hypothesis, observation):
            counts[observation] = counts.get(observation, 0) + 1
            if counts[observation] > n_hypotheses:
                return 1.0
            return _gaussian_likelihood(hypothesis, observation)

        violation = evidence_conservation_violation(portfolio, 2.0, dedup_likelihood)
        self.assertFalse(violation)

    def test_a_small_double_ingestion_is_not_absorbed_by_a_relative_tolerance(self):
        """MXR-080-1757: np.allclose was given atol=1e-9 but kept its default rtol=1e-5, so the
        effective threshold was ~250x the documented absolute scale and a repeat update that moved a
        weight by 2.5e-7 -- real double counting -- was reported as conserved."""
        portfolio = _toy_portfolio()

        def barely_informative(hypothesis, observation):
            return 1.0 + (1e-6 if hypothesis.id == "h0" else 0.0)

        deviation = evidence_conservation_deviation(portfolio, 2.0, barely_informative)
        self.assertGreater(deviation, 1e-7)
        self.assertLess(deviation, 1e-5)  # inside the old default rtol band, outside the stated atol
        self.assertTrue(evidence_conservation_violation(portfolio, 2.0, barely_informative))

    def test_a_genuinely_conserved_update_still_reports_no_violation(self):
        portfolio = _toy_portfolio()
        deviation = evidence_conservation_deviation(portfolio, 2.0, lambda h, o: 1.0)
        self.assertLessEqual(deviation, CONSERVATION_ATOL)
        self.assertFalse(evidence_conservation_violation(portfolio, 2.0, lambda h, o: 1.0))


class BudgetAndEmptyPortfolioTest(unittest.TestCase):
    """MXR-080-1758: a zero sample budget reported assurance without measuring anything, and a valid
    open-world-only portfolio raised out of np.max on its empty weight vector."""

    def test_a_zero_or_invalid_sample_budget_is_refused(self):
        portfolio = _toy_portfolio()
        for budget in (0, -1, 2.5, True):
            with self.assertRaises(ValueError):
                exchangeability_violation(portfolio, [1.0, 2.0], _gaussian_likelihood, n_permutations=budget)
            with self.assertRaises(ValueError):
                martingale_violation(portfolio, lambda rng: 1.0, _gaussian_likelihood, n=budget)

    def test_an_open_world_only_portfolio_is_trivially_coherent(self):
        empty = HypothesisPortfolio([], np.array([]), w_open=1.0)
        self.assertEqual(
            exchangeability_violation(empty, [1.0, 2.0], _gaussian_likelihood, n_permutations=3, rng=0), 0.0
        )
        self.assertEqual(martingale_violation(empty, lambda rng: 1.0, _gaussian_likelihood, n=3, rng=0), 0.0)
        self.assertFalse(evidence_conservation_violation(empty, 1.0, _gaussian_likelihood))

    def test_a_sequence_with_only_one_ordering_is_refused_not_certified(self):
        """MXR-080-1758: 0 or 1 observations cannot exhibit order dependence, so 0.0 would lie.

        The zero-``n_permutations`` case was already refused for exactly this reason. A sequence of
        length 0 or 1 has a single ordering, so every permutation is the base order and the deviation
        is identically zero -- "no violation" from a test that never ran.
        """
        portfolio = _toy_portfolio()
        for observations in ([], [1.0]):
            with self.subTest(n_observations=len(observations)):
                with self.assertRaisesRegex(ValueError, "at least two observations"):
                    exchangeability_violation(portfolio, observations, _gaussian_likelihood, n_permutations=3, rng=0)
        # Two is the smallest sequence with a second ordering, so it must still be accepted.
        self.assertIsInstance(
            exchangeability_violation(portfolio, [1.0, 2.0], _gaussian_likelihood, n_permutations=3, rng=0), float
        )


if __name__ == "__main__":
    unittest.main()
