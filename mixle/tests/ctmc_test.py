"""CTMC (E3): continuous-time Markov chain over trajectories — closed-form generator MLE, GLOBAL_UNIQUE."""

import unittest

import numpy as np

import mixle.stats as st
from mixle.inference import Guarantee, certify, optimize


def _true():
    return np.array([[0.0, 2.0, 0.5], [1.0, 0.0, 1.5], [0.3, 0.7, 0.0]])


class CTMCTest(unittest.TestCase):
    def test_generator_recovered_by_closed_form(self):
        true = _true()
        d = st.ContinuousTimeMarkovChainDistribution(true, horizon=50.0)
        traj = d.sampler(seed=0).sample(400)
        fit = optimize(traj, st.ContinuousTimeMarkovChainEstimator(3), out=None, max_its=1)
        self.assertLess(float(np.abs(fit.rates - true).max()), 0.15)  # recovered from one pass

    def test_certifies_global_unique(self):
        d = st.ContinuousTimeMarkovChainDistribution(_true(), horizon=40.0)
        traj = d.sampler(seed=1).sample(200)
        fit = optimize(traj, st.ContinuousTimeMarkovChainEstimator(3), out=None, max_its=1)
        cert = certify(fit)
        self.assertEqual(cert.guarantee, Guarantee.GLOBAL_UNIQUE)  # closed-form Poisson rates, unique
        self.assertEqual(len(cert.gradient_blocks), 0)

    def test_impossible_transition_is_minus_inf(self):
        d = st.ContinuousTimeMarkovChainDistribution(
            np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]), horizon=10.0
        )
        self.assertEqual(d.log_density((0, [(0.1, 2)])), -np.inf)  # 0->2 has rate 0

    def test_generator_rows_sum_to_zero(self):
        d = st.ContinuousTimeMarkovChainDistribution(_true())
        self.assertTrue(np.allclose(d.generator.sum(axis=1), 0.0))  # Q is a valid generator

    def test_serialization_round_trip(self):
        from mixle.utils.serialization import from_json, to_json

        d = st.ContinuousTimeMarkovChainDistribution(_true(), horizon=25.0)
        d2 = from_json(to_json(d))
        self.assertTrue(np.allclose(d2.rates, d.rates))
        self.assertEqual(d2.horizon, 25.0)

    def test_log_density_matches_hand_computation(self):
        # a two-state chain, one trajectory: 0 ->(dt=2) 1, likelihood = q01 * exp(-q0*2)
        d = st.ContinuousTimeMarkovChainDistribution(np.array([[0.0, 3.0], [1.0, 0.0]]))
        traj = (0, [(2.0, 1)])
        expected = np.log(3.0) - 3.0 * 2.0  # log q01 - q0 * T0
        self.assertAlmostEqual(d.log_density(traj), expected, places=10)

    def test_bad_rate_matrix_rejected(self):
        with self.assertRaises(ValueError):
            st.ContinuousTimeMarkovChainDistribution(np.array([[0.0, -1.0], [1.0, 0.0]]))  # negative rate
        with self.assertRaises(ValueError):
            st.ContinuousTimeMarkovChainDistribution(np.array([1.0, 2.0]))  # not square


if __name__ == "__main__":
    unittest.main()
