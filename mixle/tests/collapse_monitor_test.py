"""collapse_monitor: the shared collapse-detection utility for self-improvement loops (CARD COLLAPSE-a).

A self-improvement round claims to be getting better; this is the check that claim must pass: held-out
verified score non-decreasing AND proposal diversity not shrinking, across every round.
"""

import unittest

from mixle.task.collapse import (
    CollapseVerdict,
    collapse_monitor,
    distinct_count_diversity,
    entropy_diversity,
)


def _round(score, candidates):
    return {"score": score, "candidates": candidates}


class ImprovingTrajectoryTest(unittest.TestCase):
    """A synthetic genuinely-improving trajectory: score rises, diversity holds -- verdict is ok."""

    def test_improving_and_diverse_trajectory_is_ok(self):
        history = [
            _round(0.50, ["a", "b", "c", "d"]),
            _round(0.62, ["a", "e", "f", "g"]),
            _round(0.71, ["b", "h", "i", "j"]),
            _round(0.80, ["c", "k", "l", "m"]),
        ]
        verdict = collapse_monitor(history)
        self.assertIsInstance(verdict, CollapseVerdict)
        self.assertTrue(verdict.ok)
        self.assertIsNone(verdict.reason)
        self.assertIsNone(verdict.failed_round)
        self.assertEqual(verdict.scores, [0.50, 0.62, 0.71, 0.80])
        self.assertEqual(verdict.diversities, [4.0, 4.0, 4.0, 4.0])

    def test_flat_score_within_tolerance_is_ok(self):
        history = [_round(0.80, ["a", "b"]), _round(0.799, ["c", "d"]), _round(0.801, ["e", "f"])]
        verdict = collapse_monitor(history, score_tol=0.01)
        self.assertTrue(verdict.ok)


class ModeCollapseTest(unittest.TestCase):
    """A synthetic mode-collapsing trajectory: score rises only because the pool shrank to a few winners."""

    def test_shrinking_diversity_is_flagged_even_as_score_rises(self):
        history = [
            _round(0.50, ["a", "b", "c", "d", "e", "f"]),
            _round(0.65, ["a", "a", "a", "b", "c", "d"]),  # pool narrowing
            _round(0.90, ["a", "a", "a", "a", "a", "a"]),  # fully collapsed onto one candidate
        ]
        verdict = collapse_monitor(history)
        self.assertFalse(verdict.ok)
        self.assertEqual(verdict.reason, "diversity_shrunk")
        self.assertEqual(verdict.failed_round, 1)  # the FIRST round the shrink is visible

    def test_score_decrease_is_flagged_regardless_of_diversity(self):
        history = [
            _round(0.80, ["a", "b", "c"]),
            _round(0.60, ["d", "e", "f"]),  # genuine regression, diversity fine
        ]
        verdict = collapse_monitor(history)
        self.assertFalse(verdict.ok)
        self.assertEqual(verdict.reason, "score_decreased")
        self.assertEqual(verdict.failed_round, 1)

    def test_score_check_is_reported_before_diversity_check_in_the_same_round(self):
        """When both regress in the same round, score_decreased is the named cause (checked first)."""
        history = [
            _round(0.80, ["a", "b", "c", "d"]),
            _round(0.50, ["a", "a", "a", "a"]),  # both score down AND diversity down
        ]
        verdict = collapse_monitor(history)
        self.assertEqual(verdict.reason, "score_decreased")


class PrecomputedDiversityTest(unittest.TestCase):
    def test_accepts_a_precomputed_diversity_value_without_a_candidates_key(self):
        history = [
            {"score": 0.5, "diversity": 5.0},
            {"score": 0.6, "diversity": 5.0},
            {"score": 0.7, "diversity": 2.0},  # shrinks
        ]
        verdict = collapse_monitor(history)
        self.assertFalse(verdict.ok)
        self.assertEqual(verdict.reason, "diversity_shrunk")


class DiversityFunctionsTest(unittest.TestCase):
    def test_distinct_count_diversity(self):
        self.assertEqual(distinct_count_diversity(["a", "a", "b", "c"]), 3.0)
        self.assertEqual(distinct_count_diversity([]), 0.0)

    def test_entropy_diversity_is_zero_for_a_single_repeated_candidate(self):
        self.assertEqual(entropy_diversity(["a", "a", "a"]), 0.0)

    def test_entropy_diversity_is_higher_for_a_uniform_spread(self):
        uniform = entropy_diversity(["a", "b", "c", "d"])
        skewed = entropy_diversity(["a", "a", "a", "b"])
        self.assertGreater(uniform, skewed)

    def test_collapse_monitor_can_use_entropy_diversity(self):
        history = [
            _round(0.5, ["a", "b", "c", "d"]),
            _round(0.6, ["a", "a", "a", "b"]),  # entropy drops even though count stays 2
        ]
        verdict = collapse_monitor(history, diversity_fn=entropy_diversity)
        self.assertFalse(verdict.ok)
        self.assertEqual(verdict.reason, "diversity_shrunk")


class EdgeCasesTest(unittest.TestCase):
    def test_single_round_history_is_trivially_ok(self):
        verdict = collapse_monitor([_round(0.5, ["a"])])
        self.assertTrue(verdict.ok)

    def test_empty_history_is_trivially_ok(self):
        verdict = collapse_monitor([])
        self.assertTrue(verdict.ok)
        self.assertEqual(verdict.scores, [])


if __name__ == "__main__":
    unittest.main()
