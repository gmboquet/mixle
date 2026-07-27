"""Numba permutation-distance kernels -- verified against independent brute-force references."""

import itertools
import math
import unittest

import numpy as np

import mixle.stats.rankings._permutation_kernels as K


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

    def test_public_distance_helpers_reject_malformed_permutations(self):
        malformed = (
            np.asarray([0, 0, 2]),
            np.asarray([0, 1.5, 2]),
            np.asarray([-1, 1, 2]),
            np.asarray([0, 1, 3]),
        )
        one_argument = (
            K.kendall_perm,
            K.cayley_perm,
            K.hamming_perm,
            K.footrule_perm,
            K.spearman_perm,
            K.ulam_perm,
        )
        for value in malformed:
            for function in one_argument:
                with self.subTest(value=value, function=function.__name__):
                    with self.assertRaises((TypeError, ValueError)):
                        function(value)
            for metric in K.METRICS:
                with self.subTest(value=value, metric=metric):
                    with self.assertRaises((TypeError, ValueError)):
                        K.permutation_distance(value, np.arange(3), metric)

    def test_batched_and_rim_helpers_validate_width_center_and_rows(self):
        with self.assertRaises(ValueError):
            K.seq_distance_to_center([[0, 1]], np.arange(3), "kendall")
        with self.assertRaises(ValueError):
            K.seq_distance_to_center([[0, 0, 2]], np.arange(3), "kendall")
        with self.assertRaises(ValueError):
            K.seq_distance_to_center([[0, 1, 2]], [0, 0, 2], "kendall")
        with self.assertRaises(ValueError):
            K.seq_rim_code([[0, 1, 1]], np.arange(3))
        with self.assertRaises(ValueError):
            K.seq_rim_code([[0, 1]], np.arange(3))
        with self.assertRaises(ValueError):
            K.permutation_distance(np.arange(2), np.arange(3))


class AssignmentKernelContractTest(unittest.TestCase):
    def test_permanent_rejects_invalid_matrix_contracts(self):
        invalid = (
            np.ones((2, 3)),
            np.asarray([[1.0, -1.0], [1.0, 1.0]]),
            np.asarray([[1.0, np.nan], [1.0, 1.0]]),
            np.asarray([[1.0, np.inf], [1.0, 1.0]]),
        )
        for matrix in invalid:
            with self.subTest(matrix=matrix):
                with self.assertRaises(ValueError):
                    K.ryser_log_permanent(matrix)

    def test_log_domain_permanent_handles_extreme_finite_scales(self):
        for scale in (1.0e-200, 1.0e200):
            matrix = np.full((3, 3), scale)
            expected = math.log(math.factorial(3)) + 3.0 * math.log(scale)
            with self.subTest(scale=scale):
                self.assertAlmostEqual(K.ryser_log_permanent(matrix), expected, places=10)
        self.assertEqual(K.ryser_log_permanent(np.zeros((2, 2))), -np.inf)

    def test_log_weight_permanent_preserves_extreme_dynamic_range(self):
        log_weights = np.asarray([[700.0, -700.0], [-700.0, 700.0]])
        self.assertAlmostEqual(K.log_matrix_permanent(log_weights), 1400.0, places=12)
        log_weights.setflags(write=False)
        self.assertAlmostEqual(K.log_matrix_permanent(log_weights), 1400.0, places=12)
        with self.assertRaises(ValueError):
            K.log_matrix_permanent(np.ones((2, 3)))
        with self.assertRaises(ValueError):
            K.log_matrix_permanent(np.asarray([[0.0, np.nan], [0.0, 0.0]]))

    def test_sinkhorn_rejects_invalid_controls_and_infeasible_support(self):
        invalid_scores = (
            np.ones((2, 3)),
            np.asarray([[0.0, np.nan], [0.0, 0.0]]),
            np.asarray([[0.0, np.inf], [0.0, 0.0]]),
            np.full((2, 2), -np.inf),
            np.asarray([[0.0, -np.inf], [0.0, -np.inf]]),
        )
        for scores in invalid_scores:
            with self.subTest(scores=scores):
                with self.assertRaises((ValueError, FloatingPointError)):
                    K.sinkhorn_bethe(scores, 10)
        for iterations in (0, -1, 1.5, np.nan, True):
            with self.subTest(iterations=iterations):
                with self.assertRaises((TypeError, ValueError)):
                    K.sinkhorn_bethe(np.zeros((2, 2)), iterations)

    def test_sinkhorn_handles_feasible_exclusions_and_extreme_scores(self):
        for scores in (
            np.asarray([[0.0, -np.inf], [-np.inf, 0.0]]),
            np.asarray([[-1.0e300, 0.0], [0.0, -1.0e300]]),
        ):
            plan, logz = K.sinkhorn_bethe(scores, 20)
            with self.subTest(scores=scores):
                self.assertTrue(np.all(np.isfinite(plan)))
                self.assertTrue(np.isfinite(logz))
                np.testing.assert_allclose(plan.sum(axis=0), 1.0)
                np.testing.assert_allclose(plan.sum(axis=1), 1.0)


if __name__ == "__main__":
    unittest.main()
