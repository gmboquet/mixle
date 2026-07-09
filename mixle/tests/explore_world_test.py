"""ExplorationWorld: the sequential-exploration world with synthetic ground truth (EXPLORE-a's
functional substance, built in mixle core -- see the module docstring for why). Per the card's own
acceptance: determinism given seed, exact budget accounting, and the greedy baseline beats random on
average over 20 seeds (the sanity check that the world has learnable signal at all)."""

import unittest

from mixle.task.explore_world import (
    DRILL_COST,
    SURVEY_COST,
    ExplorationWorld,
    greedy_prospectivity_policy,
    random_policy,
    run_episode,
)


class DeterminismTest(unittest.TestCase):
    def test_same_seed_gives_bit_identical_episodes(self):
        r1 = run_episode(greedy_prospectivity_policy, n_cells=20, n_targets=3, budget=30, seed=7)
        r2 = run_episode(greedy_prospectivity_policy, n_cells=20, n_targets=3, budget=30, seed=7)
        self.assertEqual(r1.score, r2.score)
        self.assertEqual(r1.trace, r2.trace)

    def test_different_seeds_can_differ(self):
        scores = {run_episode(random_policy, n_cells=20, n_targets=3, budget=20, seed=s).score for s in range(5)}
        self.assertGreater(len(scores), 1)


class BudgetAccountingTest(unittest.TestCase):
    def test_survey_and_drill_costs_are_exact(self):
        world = ExplorationWorld(n_cells=10, n_targets=2, budget=10, seed=0)
        world.step({"type": "survey", "cell": 0})
        self.assertEqual(world.remaining_budget, 10 - SURVEY_COST)
        world.step({"type": "drill", "cell": 1})
        self.assertEqual(world.remaining_budget, 10 - SURVEY_COST - DRILL_COST)

    def test_episode_ends_exactly_when_budget_cannot_afford_the_cheapest_action(self):
        world = ExplorationWorld(n_cells=10, n_targets=2, budget=SURVEY_COST, seed=0)
        self.assertFalse(world.done)
        world.step({"type": "survey", "cell": 0})
        self.assertEqual(world.remaining_budget, 0)
        self.assertTrue(world.done)

    def test_actions_past_done_are_refused_not_crashing(self):
        world = ExplorationWorld(n_cells=10, n_targets=2, budget=0, seed=0)
        obs = world.step({"type": "drill", "cell": 0})
        self.assertFalse(obs["accepted"])
        self.assertEqual(world.remaining_budget, 0)


class LearnableSignalTest(unittest.TestCase):
    def test_greedy_beats_random_on_average_over_20_seeds(self):
        greedy_scores = [
            run_episode(greedy_prospectivity_policy, n_cells=30, n_targets=4, budget=60, seed=s).score
            for s in range(20)
        ]
        random_scores = [
            run_episode(random_policy, n_cells=30, n_targets=4, budget=60, seed=s).score for s in range(20)
        ]
        self.assertGreater(sum(greedy_scores) / len(greedy_scores), sum(random_scores) / len(random_scores))


class ActionMenuAndPlainDictTest(unittest.TestCase):
    def test_actions_and_observations_are_plain_dicts(self):
        world = ExplorationWorld(n_cells=5, n_targets=1, budget=20, seed=0)
        menu = world.action_menu()
        self.assertTrue(all(isinstance(a, dict) for a in menu))
        obs = world.step(menu[0])
        self.assertIsInstance(obs, dict)

    def test_score_counts_only_true_targets_correctly_drilled(self):
        world = ExplorationWorld(n_cells=5, n_targets=5, budget=100, seed=0)  # every cell IS a target
        self.assertEqual(world.score(), 0)
        world.step({"type": "drill", "cell": 0})
        self.assertEqual(world.score(), 1)
        world.step({"type": "drill", "cell": 0})  # re-drilling the same cell is not double-counted
        self.assertEqual(world.score(), 1)


if __name__ == "__main__":
    unittest.main()
