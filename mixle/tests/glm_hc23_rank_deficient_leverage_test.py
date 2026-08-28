# Adversarial review of T1-02's own fix (glm_hc23_ill_conditioned_leverage_test.py) found a
# regression the fix introduced: reading leverage straight off U (diag(U @ U.T)) is correct only
# for a full-rank design. xtwx_inv masks out insignificant singular directions via
# inv_sq_singular[significant] = 1/s**2 (zero elsewhere), so the OLD quadratic-form leverage
# implicitly summed over only the `rank` significant directions. The unmasked `sum(u*u, axis=1)`
# sums over all min(n,p) directions regardless of significance -- for a genuinely rank-deficient
# design (collinear or duplicated columns, not merely ill-conditioned) it sums to p instead of
# rank, reintroducing the exact false-positive "leverage 1 ... e.g. a dummy level with a single
# observation" refusal T1-02 was written to eliminate, on a design with no dummy coding at all.

import unittest

import numpy as np

from mixle.inference.glm import glm


class Hc23RankDeficientLeverageTest(unittest.TestCase):
    def test_a_duplicated_predictor_block_does_not_trigger_false_positive_singleton_refusal(self):
        # p=10 columns (intercept + 3 predictors, duplicated 3x -- rank 4, no dummy coding at
        # all), n=9. Pre-fix this raised a false-positive "leverage 1" ValueError.
        rng = np.random.default_rng(3)
        n = 9
        x = rng.normal(size=(n, 3))
        X = np.column_stack([np.ones(n), x, x, x])
        beta = np.zeros(10)
        beta[:4] = [1.0, 2.0, -1.0, 0.5]
        y = X[:, :4] @ beta[:4] + rng.normal(scale=0.1, size=n)
        self.assertEqual(int(np.linalg.matrix_rank(X)), 4)

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # the (correct, expected) rank-deficiency warning
            res = glm(X, y, family="gaussian", link="identity", robust="HC3")
        self.assertTrue(np.all(np.isfinite(res.se)))

    def test_leverage_sums_to_rank_not_to_column_count_for_a_rank_deficient_design(self):
        rng = np.random.default_rng(5)
        n = 12
        x = rng.normal(size=(n, 2))
        X = np.column_stack([np.ones(n), x, x])  # p=5, rank=3
        rank = int(np.linalg.matrix_rank(X))
        self.assertLess(rank, X.shape[1])

        u, singular, _ = np.linalg.svd(X, full_matrices=False)
        cutoff = np.finfo(float).eps * max(*X.shape) * singular[0]
        significant = singular > cutoff
        masked_hat_diag = np.sum((u * significant[None, :]) ** 2, axis=1)
        self.assertAlmostEqual(float(np.sum(masked_hat_diag)), float(rank), places=6)

        unmasked_hat_diag = np.sum(u * u, axis=1)
        self.assertAlmostEqual(float(np.sum(unmasked_hat_diag)), float(X.shape[1]), places=6)
        self.assertGreater(np.sum(unmasked_hat_diag), np.sum(masked_hat_diag) + 1.0)

    def test_a_genuine_singleton_dummy_level_still_correctly_refuses(self):
        # regression guard: the mask fix must not widen the refusal away for the case it exists
        # to catch -- an actual dummy level fitted by exactly one observation.
        X = np.array([[1.0, 0.0]] * 4 + [[1.0, 1.0]])
        y = np.array([1.0, 1.1, 0.9, 1.05, 5.0])
        with self.assertRaises(ValueError):
            glm(X, y, family="gaussian", link="identity", robust="HC3")

    def test_a_full_rank_ill_conditioned_design_is_unaffected_by_the_mask(self):
        # the mask is a no-op whenever rank == p (every column significant) -- the case T1-02's
        # own test suite covers -- so this must match unmasked behavior exactly.
        rng = np.random.default_rng(7)
        n = 200
        x1 = rng.normal(size=n)
        x2 = x1 + rng.normal(scale=1e-9, size=n)
        x3 = rng.normal(size=n)
        X = np.column_stack([np.ones(n), x1, x2, x3])
        self.assertEqual(int(np.linalg.matrix_rank(X)), 4)
        y = X @ np.array([1.0, 0.5, 0.5, -1.0]) + rng.normal(scale=0.1, size=n)

        u, singular, _ = np.linalg.svd(X, full_matrices=False)
        cutoff = np.finfo(float).eps * max(*X.shape) * singular[0]
        significant = singular > cutoff
        self.assertTrue(np.all(significant))

        res = glm(X, y, family="gaussian", link="identity", robust="HC3")
        self.assertTrue(np.all(np.isfinite(res.se)))


if __name__ == "__main__":
    unittest.main()
