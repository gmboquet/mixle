"""Bound the automatic model-search cost (worklist I6.5).

``mixle.propose`` scores a frontier of candidate models by fitting each one; without a budget that cost is
unbounded in the number of proposers. ``max_candidates`` and ``timeout`` bound it -- and a candidate skipped
for budget is *recorded* in the frontier and notes, never silently dropped, so a bounded search reports
exactly what it did not evaluate. The defaults (``None``) leave the search unbounded, so existing behavior
is unchanged.
"""

import unittest

import numpy as np

import mixle


def _records(n=300, seed=0):
    rng = np.random.RandomState(seed)
    return [("free" if rng.rand() < 0.5 else "paid", float(rng.randn()), int(rng.poisson(3))) for _ in range(n)]


class ProposeBudgetTest(unittest.TestCase):
    def test_unbounded_by_default_evaluates_all_candidates(self):
        m = mixle.propose(_records())
        self.assertGreaterEqual(len(m.frontier), 2)  # heterogeneous records -> >=2 proposers
        self.assertEqual([f for f in m.frontier if "skipped" in f], [])  # nothing skipped
        self.assertTrue(any("heldout_mean_log_density" in f for f in m.frontier))

    def test_max_candidates_bounds_the_search_and_records_skips(self):
        m = mixle.propose(_records(), max_candidates=1)
        evaluated = [f for f in m.frontier if "heldout_mean_log_density" in f]
        skipped = [f for f in m.frontier if "skipped" in f]
        self.assertEqual(len(evaluated), 1)  # only the first (recommended) candidate is fit
        self.assertGreaterEqual(len(skipped), 1)  # the rest are recorded as skipped
        self.assertIsNotNone(m.spec)  # a winner is still returned
        self.assertTrue(any("search budget" in n for n in m.notes))  # and it is surfaced in notes
        self.assertIn("max_candidates", skipped[0]["skipped"])

    def test_generous_timeout_evaluates_all(self):
        m = mixle.propose(_records(), timeout=1e6)
        self.assertEqual([f for f in m.frontier if "skipped" in f], [])

    def test_zero_timeout_skips_and_falls_back_to_recommendation(self):
        m = mixle.propose(_records(), timeout=0.0)
        self.assertTrue(all("heldout_mean_log_density" not in f for f in m.frontier))  # nothing scored
        self.assertIsNotNone(m.spec)  # still returns the heuristic recommendation as the winner


if __name__ == "__main__":
    unittest.main()
