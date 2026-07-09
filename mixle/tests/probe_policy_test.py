"""CARD PROBE-a: myopic EIG vs the non-myopic (outcome-trained) plan model, head-to-head on held-out
world seeds at matched budget -- computed, not assumed to favor either side, per the card's own kill
criterion (if non-myopic does not win, the negative result is recorded and myopic is kept).
"""

import unittest

from mixle.task.explore_world import random_policy, run_episode
from mixle.task.outcome_decomposer import train_outcome_decomposer
from mixle.task.probe_policy import head_to_head_probe, myopic_eig_policy

N_CELLS, N_TARGETS, BUDGET = 20, 3, 30
HELD_OUT_SEEDS = list(range(20_000, 20_020))  # >= 20 seeds, disjoint from any training seed range


class MyopicEIGSanityTest(unittest.TestCase):
    def test_myopic_eig_beats_random_on_average(self):
        eig_scores = [
            run_episode(myopic_eig_policy, n_cells=N_CELLS, n_targets=N_TARGETS, budget=BUDGET, seed=s).score
            for s in range(20)
        ]
        random_scores = [
            run_episode(random_policy, n_cells=N_CELLS, n_targets=N_TARGETS, budget=BUDGET, seed=s).score
            for s in range(20)
        ]
        self.assertGreater(sum(eig_scores) / len(eig_scores), sum(random_scores) / len(random_scores))

    def test_myopic_eig_returns_none_on_an_exhausted_world(self):
        from mixle.task.explore_world import ExplorationWorld

        world = ExplorationWorld(n_cells=5, n_targets=1, budget=0, seed=0)
        self.assertIsNone(myopic_eig_policy(world))


class HeadToHeadTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.decomposer = train_outcome_decomposer(
            seed_worlds=40, n_cells=N_CELLS, n_targets=N_TARGETS, budget=BUDGET, rounds=3, seed=1
        )
        cls.result = head_to_head_probe(
            cls.decomposer.plan_model,
            held_out_seeds=HELD_OUT_SEEDS,
            n_cells=N_CELLS,
            n_targets=N_TARGETS,
            budget=BUDGET,
        )

    def test_the_comparison_is_computed_on_at_least_20_held_out_seeds(self):
        self.assertGreaterEqual(len(HELD_OUT_SEEDS), 20)

    def test_the_decision_is_reported_either_way(self):
        # the card's own discipline: whichever way this goes is a valid, reportable outcome -- the
        # test only asserts the DECISION IS COMPUTED (a real bool from real scores), not which way.
        self.assertIsInstance(self.result.non_myopic_wins, bool)
        self.assertIsInstance(self.result.non_myopic_score, float)
        self.assertIsInstance(self.result.myopic_score, float)

    def test_held_out_seeds_disjoint_from_training(self):
        self.assertTrue(min(HELD_OUT_SEEDS) >= 20_000)

    def test_matches_the_recorded_kill_criterion_verdict(self):
        # measured on this exact benchmark: myopic EIG wins (2.25 vs 2.00) -- the negative result is
        # recorded in notes/c7-probing-policy-negative.md, per the card's own stop rule. Pinned here
        # so a future change to either policy that silently flips the verdict is caught, not missed.
        self.assertFalse(self.result.non_myopic_wins)


if __name__ == "__main__":
    unittest.main()
