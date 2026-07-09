"""M1's own acceptance receipts: the EIG agent vs an oracle and vs random probing at matched
budget, the streaming belief's credible-interval coverage, and deterministic replay -- see
notes/designs/M1.md for how each threshold/parameter here was picked (calibration sweep for
GaussianStreamingBelief.belief_pseudo_count; the drill-only oracle formula)."""

import unittest

import numpy as np

from mixle.task import ExplorationEnvironment, GaussianStreamingBelief, InteractionLog, interact
from mixle.task.environment import Environment
from mixle.task.explore_world import DRILL_COST, random_policy


class EnvironmentProtocolTest(unittest.TestCase):
    def test_exploration_environment_satisfies_the_protocol(self):
        env = ExplorationEnvironment(n_cells=5, n_targets=1, budget=10)
        self.assertIsInstance(env, Environment)

    def test_reset_returns_a_json_safe_observation_and_populates_world(self):
        env = ExplorationEnvironment(n_cells=5, n_targets=1, budget=10)
        obs = env.reset(seed=0)
        self.assertIsInstance(obs, dict)
        self.assertIsNotNone(env.world)

    def test_step_reports_the_action_cost(self):
        env = ExplorationEnvironment(n_cells=5, n_targets=1, budget=10)
        env.reset(seed=0)
        _, cost = env.step({"type": "survey", "cell": 0})
        self.assertEqual(cost, 1.0)
        _, cost = env.step({"type": "drill", "cell": 1})
        self.assertEqual(cost, 5.0)


def _random_policy_fn(env, belief, menu):
    return random_policy(env.world)


class EigVsOracleVsRandomTest(unittest.TestCase):
    """Card acceptance: the EIG agent reaches >= 80% of a computable oracle's information gain
    at matched budget, and beats random probing. The oracle here is exact and cheap: a
    drill-only policy that already knows the true targets can identify at most
    min(n_targets, budget // DRILL_COST) of them (each identification costs one drill; no
    survey is needed once targets are known) -- that IS the achievable ceiling under the
    world's own cost model, not an approximation."""

    n_cells = 30
    n_targets = 4
    budget = 60
    seeds = range(30)

    def _mean_score(self, policy):
        scores = []
        for s in self.seeds:
            env = ExplorationEnvironment(n_cells=self.n_cells, n_targets=self.n_targets, budget=self.budget)
            belief = GaussianStreamingBelief()
            interact(env, belief, policy=policy, budget=self.budget, seed=s)
            scores.append(env.world.score())
        return float(np.mean(scores))

    def test_eig_reaches_at_least_80pct_of_oracle_and_beats_random(self):
        oracle_score = min(self.n_targets, self.budget // DRILL_COST)
        eig_mean = self._mean_score("eig")
        random_mean = self._mean_score(_random_policy_fn)

        self.assertGreaterEqual(eig_mean, 0.8 * oracle_score)
        self.assertGreater(eig_mean, random_mean)


class CalibratedBeliefTest(unittest.TestCase):
    """Card acceptance: the belief posterior is calibrated -- credible-interval coverage sits
    within finite-sample bounds of its nominal level, checked against ground truth ONLY
    available because the synthetic world exposes it (``world._geology``)."""

    def test_ninety_percent_credible_intervals_cover_true_geology_near_nominal_rate(self):
        n_cells, n_targets, budget = 30, 4, 200
        level = 0.9
        rng = np.random.RandomState(0)
        hits = 0
        total = 0
        for s in range(60):

            def survey_heavy_policy(env, belief, menu, _rng=rng):
                survey = [a for a in menu if a["type"] == "survey"]
                if survey and _rng.rand() < 0.85:
                    return survey[_rng.randint(0, len(survey))]
                drills = [a for a in menu if a["type"] == "drill"]
                return drills[_rng.randint(0, len(drills))] if drills else None

            env = ExplorationEnvironment(n_cells=n_cells, n_targets=n_targets, budget=budget)
            belief = GaussianStreamingBelief()
            interact(env, belief, policy=survey_heavy_policy, budget=budget, seed=s)
            for c in range(n_cells):
                if belief.n(c) >= 1:
                    lo, hi = belief.credible_interval(c, level=level)
                    true_geology = float(env.world._geology[c])
                    total += 1
                    if lo <= true_geology <= hi:
                        hits += 1

        self.assertGreater(total, 500)  # enough draws that the rate below is meaningful
        coverage = hits / total
        # deterministic given the fixed seed loop above -- measured ~0.93 in the design-note
        # sweep; a wide but real band around the 90% nominal level, not the raw (undercovering,
        # ~0.80) sample-variance-only estimate this replaced.
        self.assertGreaterEqual(coverage, 0.85)
        self.assertLessEqual(coverage, 0.98)


class DeterministicReplayTest(unittest.TestCase):
    """Card acceptance: InteractionLog replays deterministically, reusing mixle.task.replay."""

    def test_same_seed_reproduces_the_same_trace(self):
        env1 = ExplorationEnvironment(n_cells=10, n_targets=2, budget=30)
        log1 = interact(env1, GaussianStreamingBelief(), policy="eig", budget=30, seed=3)

        env2 = ExplorationEnvironment(n_cells=10, n_targets=2, budget=30)
        log2 = interact(env2, GaussianStreamingBelief(), policy="eig", budget=30, seed=3)

        self.assertEqual(log1.trace.dumps(), log2.trace.dumps())
        self.assertEqual(log1.n_actions, log2.n_actions)

    def test_log_replays_bit_identically_against_a_reset_environment(self):
        env = ExplorationEnvironment(n_cells=10, n_targets=2, budget=30)
        log = interact(env, GaussianStreamingBelief(), policy="eig", budget=30, seed=5)

        self.assertIsInstance(log, InteractionLog)
        self.assertTrue(log.is_deterministic(env, GaussianStreamingBelief()))


if __name__ == "__main__":
    unittest.main()
