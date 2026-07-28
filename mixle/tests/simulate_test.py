"""simulate() (F1): a fitted model as a runtime data generator with intervention scenarios."""

import unittest

import numpy as np

import mixle.stats as st
from mixle.inference import learn_bayesian_network, optimize, simulate
from mixle.inference.simulate import IncompleteSimulationError, Simulator


def _plan_spend(n, seed):
    r = np.random.RandomState(seed)
    rows = []
    for _ in range(n):
        plan = ["free", "pro"][r.randint(2)]
        spend = float({"free": 20.0, "pro": 100.0}[plan] + 5.0 * r.randn())
        rows.append((plan, spend))
    return rows


class BaselineTest(unittest.TestCase):
    def test_simulate_any_model_baseline(self):
        model = optimize(
            [float(x) for x in np.random.RandomState(0).normal(5, 2, 300)], st.GaussianEstimator(), out=None
        )
        sim = simulate(model)
        self.assertIsInstance(sim, Simulator)
        rows = sim.run(50, seed=1)
        self.assertEqual(len(rows), 50)

    def test_deterministic_by_seed(self):
        net = learn_bayesian_network(_plan_spend(400, 0))
        sim = simulate(net)
        self.assertEqual(sim.run(10, seed=3), sim.run(10, seed=3))


class InterventionTest(unittest.TestCase):
    def test_do_scenarios_shift_the_outcome(self):
        net = learn_bayesian_network(_plan_spend(600, 0), max_parents=1)
        sim = simulate(net).scenario("free", {0: "free"}).scenario("pro", {0: "pro"})
        free = sim.outcome_mean(1, scenario="free")
        pro = sim.outcome_mean(1, scenario="pro")
        self.assertLess(free, pro)  # forcing pro raises spend
        effect = sim.compare(1, "pro", "free")
        self.assertGreater(effect, 60.0)  # true effect ~ 80
        self.assertLess(effect, 100.0)

    def test_ad_hoc_interventions(self):
        net = learn_bayesian_network(_plan_spend(600, 1), max_parents=1)
        sim = simulate(net)
        rows = sim.run(20, interventions={0: "pro"}, seed=2)
        self.assertTrue(all(r[0] == "pro" for r in rows))  # the clamped field holds

    def test_unknown_scenario_raises(self):
        sim = simulate(learn_bayesian_network(_plan_spend(200, 0)))
        with self.assertRaises(KeyError):
            sim.run(5, scenario="nope")


class _ShortSampler:
    """A generator that always yields one fewer record than asked for."""

    def sample(self, n):
        return [(1.0, 2.0)] * max(0, int(n) - 1)


class _ShortModel:
    def sampler(self, seed=0):
        return _ShortSampler()


class ExperimentSizeTest(unittest.TestCase):
    """MXR-080-1651: a simulation must run the size it was asked for, or say that it did not."""

    def setUp(self):
        self.sim = simulate(_ShortModel())

    def test_underproducing_generator_raises_instead_of_shrinking_the_experiment(self):
        with self.assertRaises(IncompleteSimulationError) as ctx:
            self.sim.run(4)
        self.assertEqual(ctx.exception.requested, 4)
        self.assertEqual(len(ctx.exception.records), 3)

    def test_outcome_mean_does_not_summarize_a_short_run(self):
        with self.assertRaises(IncompleteSimulationError):
            self.sim.outcome_mean(0, n=4)

    def test_zero_and_negative_draws_are_rejected_rather_than_returning_nan(self):
        for bad in (0, -5):
            with self.assertRaises(ValueError):
                self.sim.run(bad)
            with self.assertRaises(ValueError):
                self.sim.outcome_mean(0, n=bad)

    def test_fractional_and_boolean_draw_counts_are_rejected(self):
        for bad in (2.9, True):
            with self.assertRaises(TypeError):
                self.sim.run(bad)

    def test_exact_generator_still_runs(self):
        model = optimize(
            [float(x) for x in np.random.RandomState(0).normal(5, 2, 300)], st.GaussianEstimator(), out=None
        )
        self.assertEqual(len(simulate(model).run(7, seed=1)), 7)


class NonGraphTest(unittest.TestCase):
    def test_interventions_need_a_bayesian_network(self):
        model = optimize(
            [float(x) for x in np.random.RandomState(0).normal(0, 1, 200)], st.GaussianEstimator(), out=None
        )
        sim = simulate(model)
        with self.assertRaises(TypeError):
            sim.scenario("x", {0: 1.0})  # a scalar Gaussian has no do-operator


if __name__ == "__main__":
    unittest.main()
