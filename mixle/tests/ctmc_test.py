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
        cert = certify(fit, data=traj)
        self.assertEqual(cert.guarantee, Guarantee.GLOBAL_UNIQUE)  # closed-form Poisson rates, unique
        self.assertEqual(len(cert.gradient_blocks), 0)

    def test_impossible_transition_is_minus_inf(self):
        d = st.ContinuousTimeMarkovChainDistribution(
            np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]), horizon=10.0
        )
        self.assertEqual(d.log_density((0, 10.0, [(0.1, 2)])), -np.inf)  # 0->2 has rate 0

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
        # a two-state chain, one trajectory: 0 ->(dt=2) 1, observed to horizon=6 (so state 1 is dwelt in,
        # right-censored, for 6-2=4 more time units after the jump); likelihood is
        # q01 * exp(-q0*2) * exp(-q1*4) -- the pre-jump exposure in state 0 AND the post-jump, censored
        # exposure in state 1 both contribute.
        d = st.ContinuousTimeMarkovChainDistribution(np.array([[0.0, 3.0], [1.0, 0.0]]), horizon=6.0)
        traj = (0, 6.0, [(2.0, 1)])
        expected = np.log(3.0) - 3.0 * 2.0 - 1.0 * 4.0  # log q01 - q0*T0 - q1*T1, T1 = 6 - 2 (censored)
        self.assertAlmostEqual(d.log_density(traj), expected, places=10)

    def test_estimator_round_trip_preserves_initial_state_and_horizon(self):
        # .estimator() previously dropped initial_state/horizon entirely (ContinuousTimeMarkovChainEstimator
        # had no such params to receive them), so a fitted result silently reset to the class defaults
        # (initial_state=0, horizon=10.0) regardless of what the original distribution had -- a real
        # difference for anyone then sampling new trajectories from the fitted result.
        d = st.ContinuousTimeMarkovChainDistribution(_true(), initial_state=2, horizon=77.0)
        traj = d.sampler(seed=0).sample(50)
        fit = optimize(traj, d.estimator(), out=None, max_its=1)
        self.assertEqual(fit.initial_state, 2)
        self.assertEqual(fit.horizon, 77.0)

    def test_bad_rate_matrix_rejected(self):
        with self.assertRaises(ValueError):
            st.ContinuousTimeMarkovChainDistribution(np.array([[0.0, -1.0], [1.0, 0.0]]))  # negative rate
        with self.assertRaises(ValueError):
            st.ContinuousTimeMarkovChainDistribution(np.array([1.0, 2.0]))  # not square

    # -- regression: the final right-censored dwell was never counted -------------------------------
    #
    # _trajectory_stats only added dwell time when a jump fired to trigger the addition, so the LAST
    # dwell interval of every trajectory (from the last jump, or from t=0 if there were no jumps, out to
    # the observation horizon) was silently dropped from T_i. The old 2-element (s0, jumps) format had no
    # field to even represent a horizon, so this was not a patchable accumulation bug: the fix adds a
    # mandatory `horizon` element to the trajectory format. These two cases are exact numeric findings
    # from an external adversarial review, reproduced against the pre-fix code before this fix landed
    # (empty path scored 0.0 instead of -10.0; one-jump path scored -1.30685 instead of -13.30685).

    def test_empty_trajectory_scores_full_censored_dwell(self):
        # A chain that never jumps still accrues exposure: starting in state 0 (exit rate q0=2) and
        # observed for the whole horizon=5 without a single jump must score -q0*horizon = -10, not 0.
        d = st.ContinuousTimeMarkovChainDistribution(np.array([[0.0, 2.0], [3.0, 0.0]]), horizon=5.0)
        traj = (0, 5.0, [])
        self.assertAlmostEqual(d.log_density(traj), -10.0, places=10)

    def test_one_jump_trajectory_includes_post_jump_censored_dwell(self):
        # One jump 0->1 at dt=1, observed to horizon=5: the pre-fix code only counted the pre-jump
        # exposure (log(q01) - q0*1); the fix also charges the post-jump censored exposure in state 1
        # out to the horizon (-q1*(5-1)).
        d = st.ContinuousTimeMarkovChainDistribution(np.array([[0.0, 2.0], [3.0, 0.0]]), horizon=5.0)
        traj = (0, 5.0, [(1.0, 1)])
        expected = np.log(2.0) - 2.0 * 1.0 - 3.0 * 4.0  # log q01 - q0*T0 - q1*T1, T1 = 5 - 1 (censored)
        self.assertAlmostEqual(d.log_density(traj), expected, places=10)
        self.assertAlmostEqual(expected, -13.306852819440055, places=9)  # pins the reviewer's -13.30685

    def test_estimator_uses_censored_dwell_for_rate_mle(self):
        # The missing final dwell corrupted fitting, not just scoring: q_ij = n_ij / T_i systematically
        # over-estimated every rate because T_i was missing each trajectory's last interval. A single,
        # never-jumping trajectory contributes n=0 transitions and (post-fix) T_0=horizon of dwell; fit
        # on nothing else, the closed-form rate must come out at (approximately) 0, not undefined/blown
        # up from a T_i of 0.
        from mixle.stats.processes.ctmc import ContinuousTimeMarkovChainAccumulator

        d = st.ContinuousTimeMarkovChainDistribution(np.array([[0.0, 2.0], [3.0, 0.0]]), horizon=5.0)
        acc = ContinuousTimeMarkovChainAccumulator(num_states=2)
        acc.update((0, 5.0, []), 1.0, None)
        counts, dwell = acc.value()
        np.testing.assert_array_equal(counts, np.zeros((2, 2)))
        np.testing.assert_allclose(dwell, [5.0, 0.0])  # all-censored dwell recovered, not [0.0, 0.0]
        fit = d.estimator().estimate(None, (counts, dwell))
        self.assertAlmostEqual(fit.rates[0, 1], 0.0, places=8)  # 0 observed transitions / 5.0 dwell

    def test_sampler_output_round_trips_through_log_density(self):
        # sample() must emit the same (s0, horizon, jumps) shape log_density/the encoder now require.
        d = st.ContinuousTimeMarkovChainDistribution(_true(), horizon=8.0)
        traj = d.sampler(seed=7).sample()
        self.assertEqual(len(traj), 3)
        self.assertEqual(traj[1], 8.0)
        self.assertTrue(np.isfinite(d.log_density(traj)))

    # -- initial_state is a sampling-time default, not a scoring constraint -------------------------
    #
    # initial_state has no other use anywhere in the codebase besides sampler()'s starting point, and
    # the closed-form rate MLE q_ij = n_ij/T_i does not depend on it either. Constraining log_density to
    # -inf whenever a trajectory's observed s0 disagreed with the configured default would penalize the
    # ordinary case of fitting one CTMC to trajectories that were legitimately observed starting from
    # different states (e.g. a cohort entering a study at different stages) -- so scoring uses whatever
    # s0 (and horizon) each trajectory itself declares, exactly like horizon already does.

    def test_log_density_scores_trajectory_own_initial_state(self):
        rates = np.array([[0.0, 2.0], [3.0, 0.0]])
        traj_from_1 = (1, 5.0, [])  # starts in state 1, never observed to jump
        d_default = st.ContinuousTimeMarkovChainDistribution(rates, initial_state=0, horizon=5.0)
        d_matching = st.ContinuousTimeMarkovChainDistribution(rates, initial_state=1, horizon=5.0)
        # scoring is identical regardless of the configured initial_state -- only the data's own s0 and
        # horizon matter
        self.assertEqual(d_default.log_density(traj_from_1), d_matching.log_density(traj_from_1))
        self.assertAlmostEqual(d_default.log_density(traj_from_1), -3.0 * 5.0, places=10)  # -q1 * horizon

    # -- trajectory validation ------------------------------------------------------------------------

    def test_old_two_element_format_rejected(self):
        # The 2-element (s0, jumps) format is no longer accepted: it has no field for the trajectory's
        # horizon and cannot represent the final censored dwell time at all.
        d = st.ContinuousTimeMarkovChainDistribution(np.array([[0.0, 2.0], [3.0, 0.0]]), horizon=5.0)
        with self.assertRaises(ValueError):
            d.log_density((0, [(1.0, 1)]))

    def test_negative_dwell_time_rejected(self):
        d = st.ContinuousTimeMarkovChainDistribution(np.array([[0.0, 2.0], [3.0, 0.0]]), horizon=5.0)
        with self.assertRaises(ValueError):
            d.log_density((0, 5.0, [(-1.0, 1)]))

    def test_nonfinite_dwell_time_rejected(self):
        d = st.ContinuousTimeMarkovChainDistribution(np.array([[0.0, 2.0], [3.0, 0.0]]), horizon=5.0)
        with self.assertRaises(ValueError):
            d.log_density((0, 5.0, [(float("nan"), 1)]))
        with self.assertRaises(ValueError):
            d.log_density((0, 5.0, [(float("inf"), 1)]))

    def test_over_horizon_dwell_time_rejected(self):
        d = st.ContinuousTimeMarkovChainDistribution(np.array([[0.0, 2.0], [3.0, 0.0]]), horizon=5.0)
        with self.assertRaises(ValueError):
            d.log_density((0, 5.0, [(3.0, 1), (4.0, 0)]))  # dwell times sum to 7 > horizon 5

    def test_negative_or_nonfinite_horizon_rejected(self):
        d = st.ContinuousTimeMarkovChainDistribution(np.array([[0.0, 2.0], [3.0, 0.0]]))
        with self.assertRaises(ValueError):
            d.log_density((0, -1.0, []))
        with self.assertRaises(ValueError):
            d.log_density((0, float("nan"), []))
        with self.assertRaises(ValueError):
            d.log_density((0, float("inf"), []))

    def test_out_of_range_state_index_rejected(self):
        d = st.ContinuousTimeMarkovChainDistribution(np.array([[0.0, 2.0], [3.0, 0.0]]), horizon=5.0)
        with self.assertRaises(ValueError):
            d.log_density((5, 5.0, []))  # bad initial state
        with self.assertRaises(ValueError):
            d.log_density((0, 5.0, [(1.0, 5)]))  # bad jump target

    def test_configuration_requires_exact_states_and_sampling_bounds(self):
        rates = np.array([[0.0, 2.0], [3.0, 0.0]])
        for initial_state in (True, 0.5, -1, 2):
            with self.subTest(initial_state=initial_state), self.assertRaises((TypeError, ValueError)):
                st.ContinuousTimeMarkovChainDistribution(
                    rates,
                    initial_state=initial_state,
                )
        for horizon in (True, -1.0, np.nan, np.inf):
            with self.subTest(horizon=horizon), self.assertRaises((TypeError, ValueError)):
                st.ContinuousTimeMarkovChainDistribution(
                    rates,
                    horizon=horizon,
                )
        with self.assertRaises(ValueError):
            st.ContinuousTimeMarkovChainDistribution(np.empty((0, 0)))

    def test_trajectory_states_are_exact_and_jumps_change_state(self):
        dist = st.ContinuousTimeMarkovChainDistribution(_true())
        malformed = (
            (0.5, 1.0, []),
            (0, 1.0, [(0.5, 1.5)]),
            (0, 1.0, [(0.0, 1)]),
            (0, 1.0, [(0.5, 0)]),
            (0, 1.0, [(0.5,)]),
        )
        for trajectory in malformed:
            with self.subTest(trajectory=trajectory), self.assertRaises((TypeError, ValueError)):
                dist.log_density(trajectory)

    def test_encoded_statistics_fail_closed_before_accumulation(self):
        from mixle.stats.processes.ctmc import ContinuousTimeMarkovChainAccumulator

        dist = st.ContinuousTimeMarkovChainDistribution(np.array([[0.0, 2.0], [3.0, 0.0]]))
        valid = (np.array([[0.0, 1.0], [0.0, 0.0]]), np.array([1.0, 0.0]))
        malformed = (
            (np.zeros((3, 3)), np.zeros(3)),
            (np.array([[0.0, -1.0], [0.0, 0.0]]), np.ones(2)),
            (np.array([[1.0, 0.0], [0.0, 0.0]]), np.ones(2)),
            (np.array([[0.0, 1.0], [0.0, 0.0]]), np.zeros(2)),
            (np.zeros((2, 2)), np.array([np.nan, 0.0])),
        )
        for statistic in malformed:
            with self.subTest(statistic=statistic):
                with self.assertRaises(ValueError):
                    dist.seq_log_density([statistic])
                with self.assertRaises(ValueError):
                    dist.estimator().estimate(None, statistic)
                with self.assertRaises(ValueError):
                    ContinuousTimeMarkovChainAccumulator(2).combine(statistic)

        accumulator = ContinuousTimeMarkovChainAccumulator(2)
        before = accumulator.value()
        for weights in ([], [-1.0], [np.nan]):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                accumulator.seq_update([valid], weights, None)
            np.testing.assert_array_equal(
                accumulator.counts,
                before.transition_counts,
            )
            np.testing.assert_array_equal(
                accumulator.dwell,
                before.dwell_times,
            )

    def test_statistics_restore_defensively_and_preserve_schema(self):
        from mixle.stats.processes.ctmc import ContinuousTimeMarkovChainAccumulator

        source = ContinuousTimeMarkovChainAccumulator(2)
        source.update((0, 2.0, [(1.0, 1)]), 1.0, None)
        value = source.value()
        self.assertEqual(value.schema_version, 1)
        restored = ContinuousTimeMarkovChainAccumulator(2).from_value(value)
        value.transition_counts[0, 1] = 99.0
        value.dwell_times[0] = 99.0
        self.assertEqual(restored.counts[0, 1], 1.0)
        self.assertEqual(restored.dwell[0], 1.0)

    def test_explicit_gamma_prior_has_coherent_edgewise_map_update(self):
        prior_shape = np.array([[1.0, 3.0], [2.0, 1.0]])
        prior_rate = np.array([[0.0, 6.0], [7.0, 0.0]])
        estimator = st.ContinuousTimeMarkovChainEstimator(
            2,
            prior_shape=prior_shape,
            prior_rate=prior_rate,
            initial_state=1,
            horizon=12.0,
            keys="shared",
            name="ctmc",
        )
        counts = np.array([[0.0, 2.0], [3.0, 0.0]])
        dwell = np.array([4.0, 5.0])
        model = estimator.estimate(None, (counts, dwell))
        self.assertAlmostEqual(model.rates[0, 1], 0.4)
        self.assertAlmostEqual(model.rates[1, 0], 1.0 / 3.0)
        self.assertEqual(model.initial_state, 1)
        self.assertEqual(model.horizon, 12.0)
        self.assertEqual(model.keys, "shared")
        round_trip = model.estimator()
        np.testing.assert_array_equal(round_trip.prior_shape, prior_shape)
        np.testing.assert_array_equal(round_trip.prior_rate, prior_rate)

    def test_pseudo_count_maps_to_a_documented_gamma_prior(self):
        from mixle.utils.serialization import from_json, to_json

        estimator = st.ContinuousTimeMarkovChainEstimator(2, pseudo_count=2.0)
        counts = np.array([[0.0, 2.0], [0.0, 0.0]])
        dwell = np.array([4.0, 0.0])
        model = estimator.estimate(None, (counts, dwell))
        self.assertAlmostEqual(model.rates[0, 1], 4.0 / 6.0)
        self.assertAlmostEqual(model.rates[1, 0], 1.0)
        loaded = from_json(to_json(estimator))
        np.testing.assert_array_equal(loaded.prior_shape, estimator.prior_shape)
        np.testing.assert_array_equal(loaded.prior_rate, estimator.prior_rate)
        with self.assertRaises(ValueError):
            st.ContinuousTimeMarkovChainEstimator(
                2,
                pseudo_count=1.0,
                prior_shape=2.0,
            )


if __name__ == "__main__":
    unittest.main()
