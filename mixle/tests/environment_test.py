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

    def test_action_cost_quotes_without_mutating_the_world(self):
        env = ExplorationEnvironment(n_cells=5, n_targets=1, budget=10)
        env.reset(seed=0)
        before = env.world.remaining_budget
        self.assertEqual(env.action_cost({"type": "survey", "cell": 0}), 1.0)
        self.assertEqual(env.action_cost({"type": "drill", "cell": 0}), 5.0)
        self.assertEqual(env.world.remaining_budget, before)


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


class _NoopBelief:
    def __init__(self):
        self.observations = []

    def update(self, observation):
        self.observations.append(observation)


class _CostEnvironment:
    def __init__(self, quote=1.0, actual=1.0):
        self.quote = quote
        self.actual = actual
        self.step_calls = 0

    def reset(self, seed=None):
        self.step_calls = 0
        return {"seed": seed}

    def action_space(self):
        return [{"type": "act"}]

    def action_cost(self, action):
        return self.quote

    def step(self, action):
        self.step_calls += 1
        return {"accepted": True, "step": self.step_calls}, self.actual


def _first_policy(env, belief, menu):
    return menu[0]


class BudgetContractTest(unittest.TestCase):
    def test_unaffordable_action_is_never_executed(self):
        env = _CostEnvironment(quote=5.0, actual=5.0)
        log = interact(env, _NoopBelief(), policy=_first_policy, budget=4.0)
        self.assertEqual(env.step_calls, 0)
        self.assertEqual(log.total_cost, 0.0)
        self.assertEqual(log.stop_reason, "unaffordable_action")

    def test_invalid_quote_is_rejected_before_mutation(self):
        for quote in (-1.0, float("nan"), float("inf")):
            with self.subTest(quote=repr(quote)):
                env = _CostEnvironment(quote=quote)
                with self.assertRaisesRegex(ValueError, "quoted action cost"):
                    interact(env, _NoopBelief(), policy=_first_policy, budget=2.0)
                self.assertEqual(env.step_calls, 0)

    def test_invalid_actual_cost_retains_the_reservation(self):
        env = _CostEnvironment(quote=1.0, actual=float("nan"))
        with self.assertRaisesRegex(ValueError, "reported action cost"):
            interact(env, _NoopBelief(), policy=_first_policy, budget=2.0)
        self.assertEqual(env.step_calls, 1)

    def test_actual_cost_cannot_exceed_reserved_quote(self):
        env = _CostEnvironment(quote=1.0, actual=2.0)
        with self.assertRaisesRegex(RuntimeError, "above its reserved upper bound"):
            interact(env, _NoopBelief(), policy=_first_policy, budget=2.0)
        self.assertEqual(env.step_calls, 1)

    def test_zero_cost_nonterminal_environment_stops_at_step_limit(self):
        env = _CostEnvironment(quote=0.0, actual=0.0)
        log = interact(env, _NoopBelief(), policy=_first_policy, budget=2.0, max_steps=3)
        self.assertEqual(env.step_calls, 3)
        self.assertEqual(log.n_attempts, 3)
        self.assertEqual(log.stop_reason, "step_limit")

    def test_budget_and_step_limit_are_validated(self):
        env = _CostEnvironment()
        for budget in (-1.0, float("nan"), float("inf")):
            with self.subTest(budget=repr(budget)), self.assertRaisesRegex(ValueError, "budget"):
                interact(env, _NoopBelief(), policy=_first_policy, budget=budget)
        for max_steps in (0, -1, 1.5, True):
            with self.subTest(max_steps=repr(max_steps)), self.assertRaisesRegex(ValueError, "max_steps"):
                interact(env, _NoopBelief(), policy=_first_policy, budget=1.0, max_steps=max_steps)


class StreamingBeliefDomainTest(unittest.TestCase):
    """MXR-080-1665: invalid probability/prior domains cannot survive into a published bound."""

    def test_credible_level_must_be_in_the_open_unit_interval(self):
        belief = GaussianStreamingBelief()
        for level in (-0.9, 0.0, 1.0, 1.1, float("nan"), float("inf")):
            with self.subTest(level=repr(level)), self.assertRaisesRegex(ValueError, "credible level"):
                belief.credible_interval(0, level=level)

    def test_valid_levels_return_finite_ordered_intervals(self):
        belief = GaussianStreamingBelief()
        for level in (0.5, 0.8, 0.9, 0.95, 0.99, 0.123):
            lo, hi = belief.credible_interval(0, level=level)
            with self.subTest(level=repr(level)):
                self.assertTrue(np.isfinite(lo) and np.isfinite(hi))
                self.assertLessEqual(lo, hi)

    def test_wider_levels_give_wider_intervals(self):
        belief = GaussianStreamingBelief()
        narrow = belief.credible_interval(0, level=0.5)
        wide = belief.credible_interval(0, level=0.99)
        self.assertGreater(wide[1] - wide[0], narrow[1] - narrow[0])

    def test_negative_prior_and_pseudo_count_are_rejected_at_construction(self):
        with self.assertRaisesRegex(ValueError, "prior_sigma2"):
            GaussianStreamingBelief(prior_sigma2=-1.0)
        with self.assertRaisesRegex(ValueError, "belief_pseudo_count"):
            GaussianStreamingBelief(belief_pseudo_count=-1.0)
        with self.assertRaisesRegex(ValueError, "min_covar"):
            GaussianStreamingBelief(min_covar=-0.5)
        for bad in (float("nan"), float("inf")):
            with self.subTest(bad=repr(bad)), self.assertRaisesRegex(ValueError, "finite"):
                GaussianStreamingBelief(prior_mu=bad)

    def test_a_rejected_configuration_never_folds_in_evidence(self):
        # the old failure mode: construction succeeded, an observation mutated the belief, and only
        # then did the interval divide by zero (k0 + n == 0) or take sqrt of a negative variance.
        with self.assertRaises(ValueError):
            belief = GaussianStreamingBelief(belief_pseudo_count=-1.0)
            belief.update({"type": "survey", "accepted": True, "cell": 0, "prospectivity": 1.0})
            belief.credible_interval(0)

    def test_zero_pseudo_count_remains_a_legal_calibration_point(self):
        belief = GaussianStreamingBelief(belief_pseudo_count=0.0)
        belief.update({"type": "survey", "accepted": True, "cell": 0, "prospectivity": 1.0})
        lo, hi = belief.credible_interval(0)
        self.assertTrue(np.isfinite(lo) and np.isfinite(hi))
        self.assertLessEqual(lo, hi)


if __name__ == "__main__":
    unittest.main()
