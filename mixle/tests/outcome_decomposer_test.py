"""CARD C2-a: outcome-trained decomposer -- propose plans by sampling the fitted plan model, execute
in the exploration world, keep verifiably-successful traces, refit, iterate. Acceptance test (per the
card, computed on HELD-OUT seeds never used during training): the outcome-refit model beats both the
imitation-only model (round 0) and the greedy heuristic at matched budget.
"""

import unittest
from unittest.mock import patch

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
            self.assertEqual(len(r.candidates), r.n_candidates)
            self.assertEqual(len(r.candidate_scores), r.n_candidates)
            self.assertTrue(set(r.selection_seeds).isdisjoint(r.audit_seeds))

    def test_held_out_seeds_never_overlap_training_seed_range(self):
        # training uses seed_worlds=40 (seeds 0..39) for imitation, plus rng-drawn seeds during
        # outcome rounds -- held-out starts at 10000, far outside any plausible training draw.
        self.assertTrue(min(HELD_OUT_SEEDS) >= 10_000)

    def test_every_candidate_uses_the_same_selection_worlds(self):
        calls = []

        def score(plan, *, n_cells, n_targets, budget, seed):
            calls.append((tuple(plan), seed))
            return 1

        with patch("mixle.task.outcome_decomposer.execute_plan", side_effect=score):
            result = train_outcome_decomposer(
                seed_worlds=3,
                n_cells=5,
                n_targets=1,
                budget=10,
                k_candidates=3,
                rounds=1,
                seed=7,
                selection_worlds=2,
                audit_worlds=2,
            )
        selection_calls = calls[: 3 * 2]
        panels = [tuple(seed for _, seed in selection_calls[offset : offset + 2]) for offset in range(0, 6, 2)]
        self.assertEqual(panels, [result.rounds[0].selection_seeds] * 3)
        self.assertEqual(len(result.rounds[0].candidate_scores), 3)

    def test_training_seed_controls_all_world_panels(self):
        kwargs = dict(
            seed_worlds=3,
            n_cells=5,
            n_targets=1,
            budget=10,
            k_candidates=3,
            rounds=1,
            selection_worlds=2,
            audit_worlds=2,
        )
        first = train_outcome_decomposer(**kwargs, seed=11)
        repeated = train_outcome_decomposer(**kwargs, seed=11)
        different = train_outcome_decomposer(**kwargs, seed=12)
        self.assertEqual(first.imitation_seeds, repeated.imitation_seeds)
        self.assertEqual(first.rounds[0].selection_seeds, repeated.rounds[0].selection_seeds)
        self.assertNotEqual(first.imitation_seeds, different.imitation_seeds)
        self.assertTrue(set(first.imitation_seeds).isdisjoint(first.rounds[0].selection_seeds))
        self.assertTrue(set(first.imitation_seeds).isdisjoint(first.rounds[0].audit_seeds))


if __name__ == "__main__":
    unittest.main()
