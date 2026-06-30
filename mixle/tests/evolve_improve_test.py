"""The improve() driver, auto_select, the ledger, and decision-regret (mixle.evolve)."""

import json
import unittest

import numpy as np

from mixle.evolve import (
    EvolutionLedger,
    auto_select,
    crps_objective,
    decision_regret_objective,
    improve,
    nll_objective,
)
from mixle.inference import bayes_action, posterior
from mixle.inference.estimation import optimize
from mixle.stats import GaussianDistribution


def _fit(data, mu=0.0, sigma2=1.0):
    return optimize(list(data), GaussianDistribution(mu, sigma2).estimator(), out=None)


class ImproveTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.data = list(rng.normal(3.0, 2.0, 500))
        self.nll = nll_objective()

    def test_improve_beats_a_bad_champion(self):
        champion = GaussianDistribution(0.0, 1.0)
        ledger = EvolutionLedger()
        result = improve(champion, self.data, objective=self.nll, ledger=ledger, seed=1)
        self.assertTrue(result.verified)
        self.assertGreater(result.delta, 0.0)
        self.assertLessEqual(self.nll.scalar(result.model, self.data), self.nll.scalar(champion, self.data))
        self.assertTrue(len(ledger) >= 1)
        # the ledger is JSON-serializable.
        json.loads(ledger.to_json())

    def test_anti_regression_never_returns_worse_model(self):
        # an already-MLE champion: no operator may produce a verified worse model; the returned model
        # is the unchanged champion (verified=False) or a model that is no worse on the objective.
        champion = _fit(self.data, 3.0, 2.0)
        for seed in range(4):
            result = improve(champion, self.data, objective=self.nll, seed=seed)
            self.assertLessEqual(
                self.nll.scalar(result.model, self.data),
                self.nll.scalar(champion, self.data) + 1e-6,
            )
            if not result.verified:
                self.assertIs(result.model, champion)

    def test_improve_records_every_attempt(self):
        champion = GaussianDistribution(0.0, 1.0)
        ledger = EvolutionLedger()
        improve(champion, self.data, objective=self.nll, ledger=ledger, seed=2, parent_hash="abc")
        self.assertTrue(all(row["parent_hash"] == "abc" for row in ledger))
        self.assertTrue(all("operator" in row and "delta" in row for row in ledger))

    def test_budget_skips_expensive_operators(self):
        champion = GaussianDistribution(0.0, 1.0)
        ledger = EvolutionLedger()
        # budget below AutoSelect.cost_hint (3.0) -> AutoSelect must be skipped (no ledger row for it).
        improve(champion, self.data, objective=self.nll, ledger=ledger, seed=3, budget=1.5)
        self.assertNotIn("auto_select", [row["operator"] for row in ledger])


class AutoSelectTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(1)
        self.data = list(rng.normal(2.0, 1.5, 600))

    def test_bic_picks_a_sensible_family(self):
        result = auto_select(self.data, criterion="bic")
        # a single Gaussian field should be recovered as a univariate continuous model that scores well.
        ld = nll_objective().scalar(result.model, self.data)
        ref = nll_objective().scalar(GaussianDistribution(2.0, 1.5**2), self.data)
        self.assertLess(ld, ref + 0.5)

    def test_proper_score_gate_runs(self):
        result = auto_select(self.data, criterion=crps_objective(seed=0), seed=0)
        # returns a fitted model and a verdict either way; never raises.
        self.assertIsNotNone(result.model)
        self.assertIn("family", result.evidence)

    def test_space_is_phase2(self):
        with self.assertRaises(NotImplementedError):
            auto_select(self.data, space=object())


class DecisionRegretTest(unittest.TestCase):
    def test_bayes_action_picks_known_optimum(self):
        rng = np.random.RandomState(0)
        data = list(rng.normal(5.0, 1.0, 500))
        model = _fit(data, 0.0, 1.0)
        actions = list(np.linspace(0.0, 10.0, 21))

        def sq_loss(a, draw):
            return (np.asarray(draw, dtype=float) - a) ** 2

        res = bayes_action(posterior(model, over="predictive"), sq_loss, actions, n=4000, seed=0)
        # the squared-error Bayes action is the predictive mean ~ 5.
        self.assertAlmostEqual(res["action"], 5.0, delta=0.6)

    def test_decision_regret_lower_for_better_model(self):
        rng = np.random.RandomState(0)
        data = list(rng.normal(5.0, 1.0, 500))
        good = _fit(data, 0.0, 1.0)
        bad = GaussianDistribution(0.0, 1.0)
        actions = list(np.linspace(0.0, 10.0, 21))

        def sq_loss(a, draw):
            return (np.asarray(draw, dtype=float) - a) ** 2

        obj = decision_regret_objective(sq_loss, actions, seed=0)
        self.assertLess(obj.scalar(good, data), obj.scalar(bad, data))

    def test_bayes_action_requires_samples_contract(self):
        from mixle.capability import CapabilityError

        with self.assertRaises(CapabilityError):
            bayes_action(object(), lambda a, d: 0.0, [1, 2])


if __name__ == "__main__":
    unittest.main()
