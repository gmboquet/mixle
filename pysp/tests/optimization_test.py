"""Optimization-problem enumeration API: shared surface + each problem against a reference."""

import itertools
import unittest

import numpy as np
from scipy.optimize import linear_sum_assignment

from pysp.optimize import (
    Assignment,
    BestSubsetRegression,
    EditDistance,
    OptimizationProblem,
    ShortestPath,
    Solution,
    SpanningTree,
    ViterbiPath,
    best_first_paths,
)


def _levenshtein(a, b):
    m, n = len(a), len(b)
    d = np.zeros((m + 1, n + 1))
    d[:, 0] = np.arange(m + 1)
    d[0, :] = np.arange(n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (0 if a[i - 1] == b[j - 1] else 1))
    return d[m, n]


def _viterbi_best(log_init, log_trans, log_obs):
    t, s = len(log_obs), len(log_init)
    dp = np.full((t, s), -np.inf)
    bp = np.zeros((t, s), dtype=int)
    dp[0] = np.asarray(log_init) + np.asarray(log_obs[0])
    for ti in range(1, t):
        for st in range(s):
            sc = dp[ti - 1] + np.asarray(log_trans)[:, st]
            bp[ti, st] = int(np.argmax(sc))
            dp[ti, st] = sc[bp[ti, st]] + log_obs[ti][st]
    last = int(np.argmax(dp[t - 1]))
    path = [last]
    for ti in range(t - 1, 0, -1):
        last = bp[ti, last]
        path.append(last)
    return path[::-1], float(dp[t - 1].max())


class SharedSurfaceTest(unittest.TestCase):
    def test_solve_top_iter_consistent(self):
        prob = Assignment(np.array([[1.0, 9.0], [9.0, 1.0]]))
        self.assertIsInstance(prob, OptimizationProblem)
        sol = prob.solve()
        # Solution is a named tuple: attribute access AND unpacking both work
        self.assertIsInstance(sol, Solution)
        value, objective = sol
        self.assertTrue(np.array_equal(sol.value, value))
        self.assertEqual(sol.objective, objective)
        top2 = prob.top(2)
        self.assertTrue(np.array_equal(sol.value, top2[0].value))
        self.assertEqual(sol.objective, top2[0].objective)
        # iterating the problem yields the same as enumerator()
        self.assertEqual([s.value.tolist() for s in itertools.islice(iter(prob), 2)], [s.value.tolist() for s in top2])
        self.assertAlmostEqual(sol.objective, 2.0)  # the 1+1 diagonal

    def test_shortest_path_sink_goal_default(self):
        # a tiny weighted DAG; ShortestPath needs no is_goal -- the sink "t" is the goal by default
        edges = {"s": [("a", 1.0), ("b", 4.0)], "a": [("t", 5.0), ("b", 1.0)], "b": [("t", 1.0)], "t": []}
        sol = ShortestPath("s", lambda n: edges[n]).solve()
        self.assertEqual(sol.value, ["s", "a", "b", "t"])
        self.assertAlmostEqual(sol.objective, 3.0)


class EngineTest(unittest.TestCase):
    def test_min_and_max_senses(self):
        rng = np.random.RandomState(0)
        w = rng.rand(4, 4)
        goal = (3, 3)

        def succ(node):
            i, j = node
            out = []
            if i < 3:
                out.append(((i + 1, j), float(w[i + 1, j])))
            if j < 3:
                out.append(((i, j + 1), float(w[i, j + 1])))
            return out

        got = [c for _p, c in best_first_paths((0, 0), succ, lambda n: n == goal, sense="min")]
        brute = sorted(sum(w[i, j] for i, j in _grid_path(moves)) for moves in set(itertools.permutations("DDDRRR")))
        np.testing.assert_allclose(got, brute, atol=1e-12)


def _grid_path(moves):
    i = j = 0
    cells = []
    for mv in moves:
        i, j = (i + 1, j) if mv == "D" else (i, j + 1)
        cells.append((i, j))
    return cells


class AssignmentTest(unittest.TestCase):
    def test_best_matches_scipy(self):
        rng = np.random.RandomState(1)
        cost = rng.rand(5, 5)
        cols, total = Assignment(cost).solve()
        r, c = linear_sum_assignment(cost)
        self.assertAlmostEqual(total, cost[r, c].sum())
        self.assertEqual(list(cols), list(c))

    def test_maximize_and_k_best_increasing(self):
        rng = np.random.RandomState(2)
        cost = rng.rand(4, 4)
        items = Assignment(cost, maximize=True).top(6)
        scores = [s for _c, s in items]
        self.assertTrue(all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1)))


class SpanningTreeTest(unittest.TestCase):
    def test_best_is_mst_and_increasing(self):
        w = np.array([[0, 1, 5, 4], [1, 0, 3, 2], [5, 3, 0, 6], [4, 2, 6, 0]], dtype=float)
        items = SpanningTree(w).top(5)
        costs = [c for _e, c in items]
        self.assertTrue(all(costs[i] <= costs[i + 1] for i in range(len(costs) - 1)))
        edges, total = items[0]
        self.assertEqual(len(edges), 3)  # n-1 edges
        self.assertAlmostEqual(total, sum(w[i, j] for i, j in edges))


class EditDistanceTest(unittest.TestCase):
    def test_best_equals_levenshtein(self):
        for a, b in [("kitten", "sitting"), ("flaw", "lawn"), ("", "abc")]:
            self.assertAlmostEqual(EditDistance(list(a), list(b)).solve()[1], _levenshtein(a, b), places=9)

    def test_non_uniform_costs_and_ops_rebuild(self):
        # expensive substitution -> prefer delete + insert
        ops, cost = EditDistance(list("a"), list("b"), sub_cost=lambda x, y: 0.0 if x == y else 5.0).solve()
        self.assertAlmostEqual(cost, 2.0)
        self.assertNotIn("sub", [o[0] for o in ops])
        # applying the optimal script to the source rebuilds the target
        a, b = list("abc"), list("axc")
        ops = EditDistance(a, b).solve()[0]
        self.assertEqual([t for kind, _s, t in ops if kind in ("match", "sub", "ins")], b)


class ViterbiPathTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(1)
        self.s, self.t = 3, 5
        self.li = np.log(rng.dirichlet(np.ones(self.s)))
        self.lt = np.log([rng.dirichlet(np.ones(self.s)) for _ in range(self.s)])
        self.lo = np.log([rng.dirichlet(np.ones(self.s)) for _ in range(self.t)])

    def test_best_matches_viterbi_dp(self):
        path, score = ViterbiPath(self.li, self.lt, self.lo).solve()
        ref_path, ref_score = _viterbi_best(self.li, self.lt, self.lo)
        self.assertEqual(path, ref_path)
        self.assertAlmostEqual(score, ref_score, places=9)

    def test_k_best_descending_and_exhaustive(self):
        allp = list(ViterbiPath(self.li, self.lt, self.lo).enumerator())
        self.assertEqual(len(allp), self.s**self.t)
        scores = [s for _p, s in allp]
        self.assertTrue(all(scores[i] >= scores[i + 1] - 1e-12 for i in range(len(scores) - 1)))
        brute = sum(
            np.exp(
                self.li[st[0]]
                + self.lo[0][st[0]]
                + sum(self.lt[st[i - 1]][st[i]] + self.lo[i][st[i]] for i in range(1, self.t))
            )
            for st in itertools.product(range(self.s), repeat=self.t)
        )
        self.assertAlmostEqual(sum(np.exp(scores)), brute, places=9)


class BestSubsetRegressionTest(unittest.TestCase):
    def test_recovers_true_support(self):
        rng = np.random.RandomState(0)
        n, p = 200, 6
        X = rng.randn(n, p)
        y = 3.0 * X[:, 1] - 2.0 * X[:, 4] + 0.1 * rng.randn(n)  # only features 1 and 4 matter
        best_subset, _crit = BestSubsetRegression(X, y, criterion="bic").solve()
        self.assertEqual(set(best_subset), {1, 4})

    def test_criterion_ordering_and_rss_monotone(self):
        rng = np.random.RandomState(1)
        X = rng.randn(80, 4)
        y = X[:, 0] + rng.randn(80)
        items = BestSubsetRegression(X, y, criterion="aic").enumerator()
        crits = [c for _s, c in items]
        self.assertTrue(all(crits[i] <= crits[i + 1] for i in range(len(crits) - 1)))
        # rss of a subset >= rss of any superset (adding features cannot increase residual)
        prob = BestSubsetRegression(X, y, criterion="rss")
        self.assertGreaterEqual(prob._score((0,)), prob._score((0, 1, 2, 3)))

    def test_max_size_limits_subsets(self):
        rng = np.random.RandomState(2)
        X = rng.randn(50, 5)
        y = rng.randn(50)
        items = BestSubsetRegression(X, y, max_size=2).top(1000)
        self.assertTrue(all(len(s) <= 2 for s, _c in items))


if __name__ == "__main__":
    unittest.main()
