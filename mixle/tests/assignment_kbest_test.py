"""Tests for Murty k-best assignment and the lazy MatchingEnumerator built on it."""

import itertools
import unittest

import numpy as np

from mixle.enumeration.assignment import best_assignment, k_best_assignments
from mixle.stats.rankings.matching import MatchingDistribution


def _brute(cost):
    n = cost.shape[0]
    return sorted(float(cost[range(n), p].sum()) for p in itertools.permutations(range(n)))


class KBestAssignmentTestCase(unittest.TestCase):
    def test_matches_brute_force(self):
        rng = np.random.RandomState(0)
        for _ in range(15):
            n = rng.randint(2, 7)
            cost = rng.rand(n, n)
            murty = [c for c, _, _ in k_best_assignments(cost)]
            brute = _brute(cost)
            self.assertEqual(len(murty), len(brute))
            np.testing.assert_allclose(murty, brute, atol=1e-9)
            self.assertTrue(all(murty[i] <= murty[i + 1] + 1e-12 for i in range(len(murty) - 1)))

    def test_optimum_matches_hungarian(self):
        rng = np.random.RandomState(1)
        cost = rng.rand(10, 10)
        first = next(k_best_assignments(cost))
        opt_cost, _, _ = best_assignment(cost)
        self.assertAlmostEqual(first[0], opt_cost, places=9)

    def test_lazy_top_k_large(self):
        rng = np.random.RandomState(2)
        cost = rng.rand(50, 50)  # 50! assignments; brute force impossible
        top = list(k_best_assignments(cost, k=5))
        self.assertEqual(len(top), 5)
        self.assertTrue(all(top[i][0] <= top[i + 1][0] for i in range(4)))
        self.assertAlmostEqual(top[0][0], best_assignment(cost)[0], places=9)

    def test_forbidden_inf_edges(self):
        rng = np.random.RandomState(3)
        cost = rng.rand(4, 4)
        cost[0, 0] = np.inf
        for _, rows, cols in k_best_assignments(cost, k=6):
            self.assertFalse(any(r == 0 and c == 0 for r, c in zip(rows, cols)))

    def test_forbidden_negative_inf_edge_still_finds_finite_assignment(self):
        # MXR-080-0201: a -inf edge used to be passed straight to scipy, which raises on raw -inf/nan: that
        # raise was caught and misreported as infeasible even though a perfectly good finite assignment exists
        # using the other edges. -inf must normalize to the same "forbidden" sentinel as +inf.
        cost = np.array([[-np.inf, 2.0], [3.0, 1.0]])
        results = list(k_best_assignments(cost, k=5))
        self.assertEqual(len(results), 1)  # only one feasible complete assignment avoids the forbidden edge
        total, rows, cols = results[0]
        self.assertEqual(total, 5.0)
        self.assertFalse(any(r == 0 and c == 0 for r, c in zip(rows, cols)))

    def test_forbidden_inf_edge_before_maximize_flip_still_finds_finite_assignment(self):
        # A +inf edge negated by the internal maximize sign flip becomes -inf, hitting the same scipy failure
        # mode as a raw -inf input unless normalized before the flip.
        cost = np.array([[np.inf, 2.0], [3.0, 1.0]])
        results = list(k_best_assignments(cost, k=5, maximize=True))
        self.assertEqual(len(results), 1)
        total, rows, cols = results[0]
        self.assertEqual(total, 5.0)
        self.assertFalse(any(r == 0 and c == 0 for r, c in zip(rows, cols)))

    def test_forbidden_nan_edge_still_finds_finite_assignment(self):
        cost = np.array([[np.nan, 2.0], [3.0, 1.0]])
        results = list(k_best_assignments(cost, k=5))
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], 5.0)

    def test_genuinely_infeasible_problem_still_reports_no_assignments(self):
        # Every complete assignment of this 2x2 matrix must use at least one forbidden edge -- this must stay
        # infeasible (empty), not be conflated with the invalid-numeric-input case fixed above.
        cost = np.array([[-np.inf, np.inf], [np.inf, -np.inf]])
        self.assertEqual(list(k_best_assignments(cost, k=5)), [])

    def test_negative_control_all_finite_matrix_unaffected(self):
        # Ordinary all-finite problems (no forbidden edges at all) must be untouched by the normalization.
        rng = np.random.RandomState(6)
        cost = rng.rand(5, 5)
        results = list(k_best_assignments(cost, k=5))
        opt_cost, _, _ = best_assignment(cost)
        self.assertAlmostEqual(results[0][0], opt_cost, places=9)
        self.assertTrue(all(results[i][0] <= results[i + 1][0] + 1e-12 for i in range(4)))

    def test_best_assignment_normalizes_negative_inf_forbidden_edge(self):
        total, rows, cols = best_assignment(np.array([[-np.inf, 2.0], [3.0, 1.0]]))
        self.assertEqual(total, 5.0)
        self.assertFalse(any(r == 0 and c == 0 for r, c in zip(rows, cols)))

    def test_best_assignment_genuinely_infeasible_raises(self):
        with self.assertRaises(ValueError):
            best_assignment(np.array([[-np.inf, np.inf], [np.inf, -np.inf]]))

    def test_maximize(self):
        rng = np.random.RandomState(4)
        w = rng.rand(5, 5)
        got = list(k_best_assignments(w, k=4, maximize=True))
        self.assertTrue(all(got[i][0] >= got[i + 1][0] for i in range(3)))

    def test_matching_enumerator_exact_and_lazy(self):
        rng = np.random.RandomState(5)
        d = MatchingDistribution(rng.rand(5, 5) + 0.1)
        items = list(d.enumerator())
        lds = [lp for _, lp in items]
        self.assertEqual(len(items), 120)
        self.assertTrue(all(lds[i] >= lds[i + 1] - 1e-12 for i in range(len(lds) - 1)))
        self.assertAlmostEqual(float(np.sum(np.exp(lds))), 1.0, places=6)
        for sig, lp in items[:10]:
            self.assertAlmostEqual(lp, d.log_density(sig), places=9)
        # lazy top-3 on a larger matching (full support 9! = 362880)
        big = MatchingDistribution(rng.rand(9, 9) + 0.1)
        top = list(itertools.islice(big.enumerator(), 3))
        self.assertEqual(len(top), 3)
        self.assertTrue(top[0][1] >= top[1][1] >= top[2][1])


if __name__ == "__main__":
    unittest.main()
