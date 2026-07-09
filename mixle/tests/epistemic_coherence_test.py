"""mixle.epistemic.coherence: exchangeability / martingale / evidence-conservation checks (Card E6)."""

import unittest

import numpy as np

from mixle.epistemic.coherence import (
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


if __name__ == "__main__":
    unittest.main()
