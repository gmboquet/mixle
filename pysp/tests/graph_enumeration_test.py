"""Unified best-first enumeration core + its edit-distance / k-best-Viterbi reductions."""

import itertools
import unittest

import numpy as np

from pysp.utils.graph_enumeration import best_first_paths, k_best_edit_scripts, k_best_viterbi_paths


def _levenshtein(a, b):
    """Reference uniform edit distance via the standard DP."""
    m, n = len(a), len(b)
    d = np.zeros((m + 1, n + 1))
    d[:, 0] = np.arange(m + 1)
    d[0, :] = np.arange(n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            d[i, j] = min(
                d[i - 1, j] + 1,
                d[i, j - 1] + 1,
                d[i - 1, j - 1] + (0 if a[i - 1] == b[j - 1] else 1),
            )
    return d[m, n]


def _viterbi_best(log_init, log_trans, log_obs):
    """Reference single-best Viterbi (max-product) returning (path, log_prob)."""
    T, S = len(log_obs), len(log_init)
    dp = np.full((T, S), -np.inf)
    bp = np.zeros((T, S), dtype=int)
    dp[0] = np.asarray(log_init) + np.asarray(log_obs[0])
    for t in range(1, T):
        for s in range(S):
            scores = dp[t - 1] + np.asarray(log_trans)[:, s]
            bp[t, s] = int(np.argmax(scores))
            dp[t, s] = scores[bp[t, s]] + log_obs[t][s]
    last = int(np.argmax(dp[T - 1]))
    path = [last]
    for t in range(T - 1, 0, -1):
        last = bp[t, last]
        path.append(last)
    return path[::-1], float(dp[T - 1].max())


class BestFirstCoreTest(unittest.TestCase):
    def test_min_enumerates_increasing_cost_grid(self):
        # weighted DAG on a 4x4 grid; brute-force all monotone start->goal paths.
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
        # brute force: every monotone path is a permutation of 3 downs / 3 rights
        brute = []
        for moves in set(itertools.permutations("DDDRRR")):
            i = j = 0
            cost = 0.0
            for mv in moves:
                if mv == "D":
                    i += 1
                else:
                    j += 1
                cost += w[i, j]
            brute.append(cost)
        brute.sort()
        self.assertEqual(len(got), len(brute))
        np.testing.assert_allclose(got, brute, atol=1e-12)
        self.assertTrue(all(got[i] <= got[i + 1] + 1e-12 for i in range(len(got) - 1)))

    def test_max_sense_descending(self):
        # tiny scoring tree; scores are negative (log-probs)
        logp = {None: np.log([0.5, 0.5]), 0: np.log([0.7, 0.3]), 1: np.log([0.4, 0.6])}

        def succ(node):
            depth, last = node
            if depth == 2:
                return []
            return [((depth + 1, t), float(logp[last][t])) for t in (0, 1)]

        got = [s for _p, s in best_first_paths((0, None), succ, lambda n: n[0] == 2, sense="max")]
        self.assertTrue(all(got[i] >= got[i + 1] - 1e-12 for i in range(len(got) - 1)))
        # 4 complete length-2 paths, probabilities sum to 1
        self.assertEqual(len(got), 4)
        self.assertAlmostEqual(sum(np.exp(got)), 1.0, places=9)


class EditDistanceTest(unittest.TestCase):
    def test_top_cost_equals_levenshtein(self):
        for a, b in [("kitten", "sitting"), ("flaw", "lawn"), ("", "abc"), ("abc", "abc")]:
            best = next(k_best_edit_scripts(list(a), list(b)), (None, None))
            self.assertAlmostEqual(best[1], _levenshtein(a, b), places=9, msg=f"{a!r}->{b!r}")

    def test_k_best_increasing_and_ops_reconstruct(self):
        a, b = list("abc"), list("axc")
        scripts = list(k_best_edit_scripts(a, b, k=5))
        costs = [c for _o, c in scripts]
        self.assertTrue(all(costs[i] <= costs[i + 1] + 1e-12 for i in range(len(costs) - 1)))
        # the best script applied to `a` yields `b`
        ops = scripts[0][0]
        rebuilt = []
        for kind, _src, tgt in ops:
            if kind in ("match", "sub", "ins"):
                rebuilt.append(tgt)
        self.assertEqual(rebuilt, b)

    def test_non_uniform_costs(self):
        # make substitution very expensive -> prefer delete+insert over substitute
        a, b = list("a"), list("b")
        sub_expensive = next(k_best_edit_scripts(a, b, sub_cost=lambda x, y: 0.0 if x == y else 5.0))
        self.assertAlmostEqual(sub_expensive[1], 2.0)  # del 'a' + ins 'b'
        self.assertNotIn("sub", [op[0] for op in sub_expensive[0]])


class KBestViterbiTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(1)
        self.S, self.T = 3, 6
        self.log_init = np.log(rng.dirichlet(np.ones(self.S)))
        self.log_trans = np.log([rng.dirichlet(np.ones(self.S)) for _ in range(self.S)])
        self.log_obs = np.log([rng.dirichlet(np.ones(self.S)) for _ in range(self.T)])

    def test_top1_matches_viterbi_dp(self):
        best_path, best_score = next(k_best_viterbi_paths(self.log_init, self.log_trans, self.log_obs))
        ref_path, ref_score = _viterbi_best(self.log_init, self.log_trans, self.log_obs)
        self.assertAlmostEqual(best_score, ref_score, places=9)
        self.assertEqual(best_path, ref_path)

    def test_k_best_descending_and_exhaustive(self):
        all_paths = list(k_best_viterbi_paths(self.log_init, self.log_trans, self.log_obs))
        self.assertEqual(len(all_paths), self.S**self.T)  # every state sequence
        scores = [s for _p, s in all_paths]
        self.assertTrue(all(scores[i] >= scores[i + 1] - 1e-12 for i in range(len(scores) - 1)))
        # scores are joint log-probs of (states, fixed obs); exp-sum equals total obs likelihood
        # computed independently by the forward sum over all paths.
        brute = 0.0
        for states in itertools.product(range(self.S), repeat=self.T):
            lp = self.log_init[states[0]] + self.log_obs[0][states[0]]
            for t in range(1, self.T):
                lp += self.log_trans[states[t - 1]][states[t]] + self.log_obs[t][states[t]]
            brute += np.exp(lp)
        self.assertAlmostEqual(sum(np.exp(scores)), brute, places=9)

    def test_lazy_top_k(self):
        top3 = list(k_best_viterbi_paths(self.log_init, self.log_trans, self.log_obs, k=3))
        self.assertEqual(len(top3), 3)
        full = list(k_best_viterbi_paths(self.log_init, self.log_trans, self.log_obs))
        np.testing.assert_allclose([s for _p, s in top3], [s for _p, s in full[:3]], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
