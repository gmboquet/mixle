"""Regression coverage for two bugs in mixle.evolve.objective:

* ``log_score_objective`` used to exponentiate a log-density to a plain probability before scoring it.
  ``exp(x)`` underflows to exactly ``0.0`` for any ``x`` past about ``-745`` (the float64 underflow
  boundary), so every sufficiently bad fit collapsed to the identical clipped score instead of staying
  distinguishable. Fixed to compute ``-log_density`` directly (see ``LogScoreObjectiveTest``).
* ``decision_regret_objective`` used to ignore its own ``data`` argument: each candidate model chose a
  Bayes-optimal action under its OWN predictive posterior and had that action's "regret" measured
  against draws from that SAME posterior -- a pure self-consistency check a confidently wrong model
  could pass perfectly. Fixed to measure the chosen action's realized loss against the actual ``data``
  (see ``DecisionRegretObjectiveTest``).
"""

import unittest

import numpy as np

from mixle.evolve import (
    calibration_objective,
    crps_objective,
    decision_regret_objective,
    interval_objective,
    log_score_objective,
    nll_objective,
)
from mixle.evolve.objective import sample_ensemble
from mixle.inference import bayes_action, log_score, posterior
from mixle.inference.estimation import optimize
from mixle.stats import GaussianDistribution


def _fit(data, mu=0.0, sigma2=1.0):
    return optimize(list(data), GaussianDistribution(mu, sigma2).estimator(), out=None)


class _FixedLogDensityModel:
    """Stand-in model whose per-observation log-density is an exact, pre-chosen value.

    Lets the Finding-1 reproducer pin log-densities that no real distribution's tail can hit precisely,
    through ``pointwise_log_density``'s split-safe ``seq_log_density_raw(rows)`` path.
    """

    def __init__(self, log_densities):
        self._log_densities = np.asarray(log_densities, dtype=float)

    def seq_log_density_raw(self, rows):
        assert len(rows) == self._log_densities.shape[0]
        return self._log_densities


class LogScoreObjectiveTest(unittest.TestCase):
    def test_distinguishes_deeply_negative_log_densities(self):
        # Bug reproducer: log-densities [-1000, -800, -700] used to all pass through exp() first;
        # exp(-1000) and exp(-800) both underflow to exactly 0.0 (the float64 underflow boundary is
        # ~-745), so the old objective scored them identically at -log(np.finfo(float).tiny) ==
        # 708.3964..., indistinguishable from each other despite being genuinely -- if both poor --
        # different fits. The fix must recover the exact, distinguishable negated log-densities.
        model = _FixedLogDensityModel([-1000.0, -800.0, -700.0])
        data = [0.0, 0.0, 0.0]
        obj = log_score_objective()

        result = obj.pointwise(model, data)

        np.testing.assert_array_equal(result, [1000.0, 800.0, 700.0])
        self.assertNotEqual(result[0], result[1])  # no more collapsing distinct fits together
        self.assertEqual(obj.scalar(model, data), float(np.mean([1000.0, 800.0, 700.0])))

    def test_negative_control_naive_exp_then_relog_really_does_collapse(self):
        # Sanity check that the reproducer above is a genuine bug scenario, not a coincidence: the
        # naive exp-then-relog path (mixle.inference.scoring.log_score on a plain probability -- exactly
        # what the objective used to do) really does collapse -1000 and -800 together.
        log_d = np.array([-1000.0, -800.0, -700.0])
        naive = log_score(np.exp(log_d), mean=False)
        self.assertEqual(naive[0], naive[1])
        self.assertAlmostEqual(float(naive[0]), -np.log(np.finfo(float).tiny))
        self.assertAlmostEqual(float(naive[2]), 700.0)  # -700 alone doesn't underflow, stays correct

    def test_matches_nll_objective_exactly_on_a_real_model(self):
        # log_score and nll are the same quantity (-log p(y_i)); once log_score_objective stops
        # round-tripping through exp/log it should agree with nll_objective bit for bit, not just
        # approximately (a real exp/log round trip loses a few ULPs even without underflow).
        rng = np.random.RandomState(0)
        data = list(rng.normal(3.0, 2.0, 200))
        model = GaussianDistribution(3.0, 4.0)

        log_score_pw = log_score_objective().pointwise(model, data)
        nll_pw = nll_objective().pointwise(model, data)

        np.testing.assert_array_equal(log_score_pw, nll_pw)

    def test_ordinary_log_densities_unaffected(self):
        # Negative control: well-behaved (non-underflowing) log-densities must score as exactly
        # -log_density -- the fix changes the broken deep-tail behavior, not the ordinary case.
        model = _FixedLogDensityModel([-1.0, -5.0, -10.0, 0.0, 2.0])
        data = [0.0] * 5
        result = log_score_objective().pointwise(model, data)
        np.testing.assert_array_equal(result, [1.0, 5.0, 10.0, 0.0, -2.0])


class DecisionRegretObjectiveTest(unittest.TestCase):
    @staticmethod
    def _sq_loss(a, draw):
        return (np.asarray(draw, dtype=float) - a) ** 2

    def test_good_model_beats_confidently_wrong_model_against_shared_reference_data(self):
        # Bug reproducer: a model fit to the real reference data must score a clearly better (lower)
        # regret than a model that is confidently wrong (concentrated far outside where the data
        # actually lives), when both are measured against the SAME shared reference `data`.
        rng = np.random.RandomState(0)
        data = list(rng.normal(5.0, 1.0, 500))  # the real/held-out reference outcomes
        good = _fit(data, mu=0.0, sigma2=1.0)  # fit to the real data -> recovers ~N(5, 1)
        wrong = GaussianDistribution(-20.0, 1.0)  # confidently wrong: concentrated on the wrong region
        actions = list(np.linspace(-30.0, 20.0, 51))

        obj = decision_regret_objective(self._sq_loss, actions, seed=0)
        good_score = obj.scalar(good, data)
        wrong_score = obj.scalar(wrong, data)

        self.assertLess(good_score, wrong_score)
        # not just "a bit better": the wrong model's chosen action (~-20) is ~25 units from where the
        # real data lives, so its realized squared-error loss must be enormous, not a near-tie.
        self.assertGreater(wrong_score, 100.0 * good_score)

    def test_negative_control_self_consistent_scoring_does_not_discriminate(self):
        # Sanity check / bug reproducer: the OLD scoring -- bayes_action's own `expected_loss`, i.e.
        # each model's chosen action measured against draws from that SAME model's own posterior, never
        # against real data -- can't tell the confidently wrong model from the good one. Both collapse
        # to roughly their own predictive variance (~1.0) regardless of how far off their mean is,
        # because the Bayes action always re-centers on the model's own (possibly wrong) belief.
        rng = np.random.RandomState(0)
        data = list(rng.normal(5.0, 1.0, 500))
        good = _fit(data, mu=0.0, sigma2=1.0)
        wrong = GaussianDistribution(-20.0, 1.0)
        actions = list(np.linspace(-30.0, 20.0, 51))

        good_self = bayes_action(posterior(good, over="predictive"), self._sq_loss, actions, n=2000, seed=0)[
            "expected_loss"
        ]
        wrong_self = bayes_action(posterior(wrong, over="predictive"), self._sq_loss, actions, n=2000, seed=0)[
            "expected_loss"
        ]

        # near-indistinguishable self-consistency scores despite a catastrophic difference in fit quality.
        self.assertAlmostEqual(good_self, wrong_self, delta=0.5)

    def test_data_argument_is_actually_used(self):
        # Most direct regression test for "ignores its own data argument": for a FIXED model, moving
        # the reference data must change the score. Under the old code the score depended only on
        # `model` (via its own posterior), so this would have been an exact tie.
        model = GaussianDistribution(0.0, 1.0)
        actions = list(np.linspace(-10.0, 10.0, 41))
        obj = decision_regret_objective(self._sq_loss, actions, seed=0)

        near_data = [0.1, -0.2, 0.05, 0.3, -0.1]
        far_data = [9.8, 10.1, 9.95, 10.2, 9.9]

        self.assertLess(obj.scalar(model, near_data), obj.scalar(model, far_data))

    def test_pointwise_is_still_none_scalar_only(self):
        obj = decision_regret_objective(self._sq_loss, [0.0, 1.0])
        self.assertIsNone(obj.pointwise(GaussianDistribution(0.0, 1.0), [0.0, 1.0]))


class EnsembleBudgetTest(unittest.TestCase):
    """MXR-080-1767: a sampled objective must run the ensemble its signature advertises."""

    class _ShortSampler:
        def sample(self, m):
            del m
            return np.array([1.0, 2.0])  # always two draws, whatever was asked for

    class _ShortModel:
        def sampler(self, seed):
            del seed
            return EnsembleBudgetTest._ShortSampler()

    def test_a_sampler_returning_fewer_draws_than_requested_is_rejected(self):
        with self.assertRaises(ValueError):
            sample_ensemble(self._ShortModel(), 3, 10, seed=0)

    def test_a_realized_ensemble_matching_the_request_is_accepted(self):
        model = GaussianDistribution(0.0, 1.0)
        self.assertEqual(sample_ensemble(model, 4, 7, seed=0).shape, (4, 7))

    def test_invalid_ensemble_and_row_counts_are_rejected(self):
        model = GaussianDistribution(0.0, 1.0)
        for bad in (0, -1, 2.5, float("nan"), True):
            with self.subTest(m=bad), self.assertRaises(ValueError):
                sample_ensemble(model, 3, bad, seed=0)
            with self.subTest(n=bad), self.assertRaises(ValueError):
                sample_ensemble(model, bad, 3, seed=0)

    def test_objective_builders_validate_their_advertised_budgets(self):
        for bad in (0, -1, 2.5):
            with self.subTest(ensemble=bad):
                with self.assertRaises(ValueError):
                    crps_objective(ensemble=bad)
                with self.assertRaises(ValueError):
                    interval_objective(ensemble=bad)
                with self.assertRaises(ValueError):
                    calibration_objective(ensemble=bad)
                with self.assertRaises(ValueError):
                    calibration_objective(bins=bad)


if __name__ == "__main__":
    unittest.main()
