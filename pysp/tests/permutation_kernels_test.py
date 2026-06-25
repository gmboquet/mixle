"""Numba permutation-distance kernels -- verified against independent brute-force references."""

import itertools
import unittest

import numpy as np

import pysp.stats.rankings._permutation_kernels as K


def _rank(p):
    r = np.empty(len(p), dtype=int)
    r[np.asarray(p)] = np.arange(len(p))
    return r


def _bf_kendall(a, b):
    ra, rb, n, c = _rank(a), _rank(b), len(a), 0
    for i in range(n):
        for j in range(i + 1, n):
            if (ra[i] - ra[j]) * (rb[i] - rb[j]) < 0:
                c += 1
    return c


def _bf_footrule(a, b):
    return int(np.abs(_rank(a) - _rank(b)).sum())


def _bf_spearman(a, b):
    return int(((_rank(a) - _rank(b)) ** 2).sum())


def _bf_hamming(a, b):
    return int(np.sum(np.asarray(a) != np.asarray(b)))


def _bf_cayley(a, b):
    r, n = _rank(b)[np.asarray(a)], len(a)
    seen, cyc = [False] * n, 0
    for i in range(n):
        if not seen[i]:
            cyc += 1
            j = i
            while not seen[j]:
                seen[j], j = True, r[j]
    return n - cyc


def _bf_ulam(a, b):
    r, n = _rank(b)[np.asarray(a)], len(a)
    dp = [1] * n
    for i in range(n):
        for j in range(i):
            if r[j] < r[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return n - (max(dp) if n else 0)


_REFS = {
    "kendall": _bf_kendall,
    "footrule": _bf_footrule,
    "spearman": _bf_spearman,
    "hamming": _bf_hamming,
    "cayley": _bf_cayley,
    "ulam": _bf_ulam,
}


class PermutationKernelTest(unittest.TestCase):
    def test_all_metrics_match_brute_force_small_n(self):
        rng = np.random.RandomState(0)
        for n in range(2, 7):
            for a in itertools.permutations(range(n)):
                for _ in range(6):
                    b = tuple(rng.permutation(n))
                    for m, f in _REFS.items():
                        self.assertEqual(K.permutation_distance(np.array(a), np.array(b), m), f(a, b), msg=(m, a, b))

    def test_batched_matches_per_pair_large_n(self):
        rng = np.random.RandomState(1)
        n = 40
        X = np.array([rng.permutation(n) for _ in range(150)])
        center = rng.permutation(n)
        for m in K.METRICS:
            batched = K.seq_distance_to_center(X, _rank(center), m)
            per_pair = np.array([K.permutation_distance(x, center, m) for x in X])
            np.testing.assert_array_equal(batched, per_pair)

    def test_metric_axioms(self):
        rng = np.random.RandomState(2)
        n = 12
        for m in K.METRICS:
            a, b = rng.permutation(n), rng.permutation(n)
            self.assertEqual(K.permutation_distance(a, a, m), 0)  # identity of indiscernibles (=0)
            self.assertEqual(  # symmetry
                K.permutation_distance(a, b, m), K.permutation_distance(b, a, m)
            )

    def test_unknown_metric_raises(self):
        with self.assertRaises(ValueError):
            K.permutation_distance(np.arange(3), np.arange(3), "manhattan")


if __name__ == "__main__":
    unittest.main()
