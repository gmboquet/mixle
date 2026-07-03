"""HMMPathIndex: quantized random-access unranking of HMM state paths (the count-DP companion to A*).

Contract verified against :func:`hmm_best_paths` (exact list-Viterbi) on fully-enumerable models: every
path is unranked exactly once with its exact joint log-probability (completeness), rank order follows the
quantized bucket order up to the documented T-floor smear, rank 0 is Viterbi up to that smear, and counts /
mass brackets bound the brute truth -- including with UNNORMALIZED (positive) emission log-likelihoods,
which the per-position score shift makes well-defined. Deep random access -- rank 1e9 of 6^12 paths -- is
one O(T*K) table walk, no enumeration.
"""

import math
import unittest

import numpy as np

from mixle.enumeration import HMMPathIndex, hmm_best_paths


def _norm_rows(a):
    e = np.exp(a)
    return np.log(e / e.sum(axis=-1, keepdims=True))


def _model(K, T, seed=0, emission_scale=1.5):
    rng = np.random.RandomState(seed)
    return (
        _norm_rows(rng.randn(K)),
        _norm_rows(rng.randn(K, K)),
        rng.randn(T, K) * emission_scale,
    )


def _joint(log_pi, log_A, log_b, path):
    lp = log_pi[path[0]] + log_b[0, path[0]]
    for t in range(1, len(path)):
        lp += log_A[path[t - 1], path[t]] + log_b[t, path[t]]
    return float(lp)


class AgreementWithAStarTest(unittest.TestCase):
    def setUp(self):
        self.K, self.T = 3, 5
        self.log_pi, self.log_A, self.log_b = _model(self.K, self.T, seed=0)
        self.idx = HMMPathIndex(self.log_pi, self.log_A, self.log_b, oversample=64)
        self.astar = list(hmm_best_paths(self.log_pi, self.log_A, self.log_b))

    def test_total_covers_every_path(self):
        self.assertEqual(int(self.idx.total()), self.K**self.T)
        self.assertEqual(len(self.astar), self.K**self.T)
        self.assertFalse(self.idx.truncated)

    def test_every_path_unranked_exactly_once_with_exact_logprob(self):
        mine = sorted((self.idx.unrank(i) for i in range(int(self.idx.total()))), key=lambda u: u[0])
        exact = sorted(self.astar, key=lambda u: u[0])
        self.assertEqual([p for p, _ in mine], [p for p, _ in exact])
        for (_, lp_m), (_, lp_e) in zip(mine, exact):
            self.assertAlmostEqual(lp_m, lp_e, places=12)

    def test_unranked_logprobs_are_exact(self):
        for i in (0, 37, 121, 242):
            path, lp = self.idx.unrank(i)
            self.assertAlmostEqual(lp, _joint(self.log_pi, self.log_A, self.log_b, path), places=12)

    def test_buckets_nondecreasing_in_rank_up_to_smear(self):
        # the internal walk is ascending in the STRUCTURAL bucket; a true score's bucket_of sits within
        # T-1 fine buckets of it (sum-of-floors vs floor-of-sum), so rank order dips by at most that
        buckets = [self.idx.bucket_of(self.idx.unrank(i)[1]) for i in range(int(self.idx.total()))]
        slack = self.T - 1
        self.assertTrue(all(buckets[i + 1] >= buckets[i] - slack for i in range(len(buckets) - 1)))

    def test_rank0_is_viterbi_up_to_smear(self):
        _vp, vlp = self.astar[0]
        _p0, lp0 = self.idx.unrank(0)
        self.assertLessEqual(lp0, vlp + 1e-12)
        self.assertLessEqual(self.idx.bucket_of(lp0) - self.idx.bucket_of(vlp), self.T - 1)

    def test_count_brackets_brute_truth(self):
        # structural buckets under-estimate bits by up to T floors -> count(thr) over-counts by at most
        # the paths within that smear band; verify both sides against the exact enumeration
        thr = self.astar[60][1]
        n = self.idx.count(thr)
        brute = sum(1 for _p, lp in self.astar if lp >= thr)
        smear = ((self.T + 2) / self.idx.quantizer.fine_per_bit()) * math.log(2.0)
        brute_hi = sum(1 for _p, lp in self.astar if lp >= thr - smear)
        self.assertGreaterEqual(n, brute)
        self.assertLessEqual(n, brute_hi)

    def test_mass_above_brackets_true_mass(self):
        thr = self.astar[40][1]
        true_mass = sum(math.exp(lp) for _p, lp in self.astar if lp >= thr)
        lo, hi = self.idx.mass_above(thr)
        self.assertLessEqual(lo, true_mass + 1e-12)
        # hi covers at least the true mass of the counted set (over-count only inflates it)
        self.assertGreaterEqual(hi, true_mass - 1e-12)

    def test_threshold_and_iter(self):
        lp5 = self.idx.threshold(5)
        self.assertAlmostEqual(lp5, self.idx.unrank(4)[1])
        head = list(self.idx.iter_paths())[:4]
        self.assertEqual([p for p, _ in head], [self.idx.unrank(i)[0] for i in range(4)])


class DeepAccessTest(unittest.TestCase):
    def test_unrank_1e9_is_one_table_walk(self):
        K, T = 6, 12  # 6**12 ~ 2.18e9 paths: A* would need 1e9 expansions to reach this rank
        log_pi, log_A, log_b = _model(K, T, seed=1, emission_scale=1.0)
        idx = HMMPathIndex(log_pi, log_A, log_b, oversample=16)
        self.assertEqual(int(idx.total()), K**T)
        path, lp = idx.unrank(10**9)
        self.assertEqual(len(path), T)
        self.assertAlmostEqual(lp, _joint(log_pi, log_A, log_b, path), places=10)
        # deeper rank -> deeper bucket (up to the T-floor smear, negligible at these separations)
        probes = [0, 10**3, 10**6, 10**9, 2 * 10**9]
        buckets = [idx.bucket_of(idx.unrank(i)[1]) for i in probes]
        self.assertTrue(all(buckets[i + 1] >= buckets[i] - (T - 1) for i in range(len(buckets) - 1)))

    def test_out_of_range(self):
        log_pi, log_A, log_b = _model(2, 3, seed=2)
        idx = HMMPathIndex(log_pi, log_A, log_b)
        with self.assertRaises(IndexError):
            idx.unrank(2**3)
        with self.assertRaises(IndexError):
            idx.unrank(-1)


class StructureEdgeCasesTest(unittest.TestCase):
    def test_impossible_transitions_are_excluded(self):
        K, T = 3, 4
        log_pi, log_A, log_b = _model(K, T, seed=3)
        log_A = log_A.copy()
        log_A[0, 1] = -np.inf  # forbid 0 -> 1
        idx = HMMPathIndex(log_pi, log_A, log_b, oversample=32)
        n = int(idx.total())
        self.assertLess(n, K**T)
        for i in range(n):
            path, lp = idx.unrank(i)
            self.assertTrue(np.isfinite(lp))
            for t in range(1, T):
                self.assertFalse(path[t - 1] == 0 and path[t] == 1)

    def test_budget_truncation_flag(self):
        log_pi, log_A, log_b = _model(3, 6, seed=4)
        idx = HMMPathIndex(log_pi, log_A, log_b, budget_bits=6.0)  # far too shallow for 3**6 paths
        self.assertTrue(idx.truncated)
        self.assertLess(idx.total(), 3**6)
        full = HMMPathIndex(log_pi, log_A, log_b)  # default budget covers everything
        self.assertFalse(full.truncated)
        self.assertEqual(int(full.total()), 3**6)

    def test_single_position_model(self):
        log_pi, log_A, log_b = _model(4, 1, seed=5)
        idx = HMMPathIndex(log_pi, log_A, log_b, oversample=64)
        self.assertEqual(int(idx.total()), 4)
        paths = {idx.unrank(i)[0] for i in range(4)}
        self.assertEqual(paths, {(s,) for s in range(4)})


if __name__ == "__main__":
    unittest.main()
