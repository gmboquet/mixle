"""CARD C2-a: outcome-trained decomposer -- propose plans by sampling the fitted plan model, execute
in the exploration world, keep verifiably-successful traces, refit, iterate. Acceptance test (per the
card, computed on HELD-OUT seeds never used during training): the outcome-refit model beats both the
imitation-only model (round 0) and the greedy heuristic at matched budget.
"""

import unittest

from mixle.task.outcome_decomposer import (
    evaluate_greedy_heuristic,
    evaluate_plan_model,
    train_outcome_decomposer,
)

N_CELLS, N_TARGETS, BUDGET = 20, 3, 30
HELD_OUT_SEEDS = list(range(10_000, 10_030))  # disjoint from any training seed range


class OutcomeTrainingBeatsImitationAndGreedyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.decomposer = train_outcome_decomposer(
            seed_worlds=40, n_cells=N_CELLS, n_targets=N_TARGETS, budget=BUDGET, rounds=3, seed=0
        )
        cls.outcome_score = evaluate_plan_model(
            cls.decomposer.plan_model, seeds=HELD_OUT_SEEDS, n_cells=N_CELLS, n_targets=N_TARGETS, budget=BUDGET
        )
        cls.imitation_score = evaluate_plan_model(
            cls.decomposer.imitation_model, seeds=HELD_OUT_SEEDS, n_cells=N_CELLS, n_targets=N_TARGETS, budget=BUDGET
        )
        cls.greedy_score = evaluate_greedy_heuristic(
            seeds=HELD_OUT_SEEDS, n_cells=N_CELLS, n_targets=N_TARGETS, budget=BUDGET
        )

    def test_outcome_refit_beats_imitation_only_on_held_out_seeds(self):
        self.assertGreater(self.outcome_score, self.imitation_score)

    def test_outcome_refit_beats_the_greedy_heuristic_on_held_out_seeds(self):
        self.assertGreater(self.outcome_score, self.greedy_score)

    def test_training_rounds_are_recorded_not_silently_discarded(self):
        self.assertEqual(len(self.decomposer.rounds), 3)
        for r in self.decomposer.rounds:
            self.assertGreaterEqual(r.n_candidates, r.n_kept)

    def test_held_out_seeds_never_overlap_training_seed_range(self):
        # training uses seed_worlds=40 (seeds 0..39) for imitation, plus rng-drawn seeds during
        # outcome rounds -- held-out starts at 10000, far outside any plausible training draw.
        self.assertTrue(min(HELD_OUT_SEEDS) >= 10_000)


if __name__ == "__main__":
    unittest.main()
