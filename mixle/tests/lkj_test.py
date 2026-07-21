"""WS-2: LKJ correlation-matrix distribution; normalizer verified exactly via mpmath."""

import unittest

import numpy as np
import scipy.stats as ss

import mixle
from mixle.capability import Fittable
from mixle.stats import LKJDistribution as LKJ


class LKJTest(unittest.TestCase):
    def test_normalizer_integrates_to_one_mpmath(self):
        # arbitrary-precision check that c_d(eta) * integral det(R)^(eta-1) dR == 1
        import mpmath as mp

        mp.mp.dps = 16
        for eta in (0.7, 2.0, 3.5):  # incl. eta < 1 and non-integer
            d = LKJ(2, eta)
            integ = mp.e ** mp.mpf(d._log_c) * mp.quad(lambda r, e=eta: (1 - r * r) ** (e - 1), [-1, 1])
            with self.subTest(d=2, eta=eta):
                self.assertTrue(mp.almosteq(integ, 1, 1e-12))

        def z3(eta):
            def inner(a, b):
                rad = mp.sqrt((1 - a * a) * (1 - b * b))
                return mp.quad(
                    lambda c: (1 - a * a - b * b - c * c + 2 * a * b * c) ** (eta - 1), [a * b - rad, a * b + rad]
                )

            return mp.quad(lambda a: mp.quad(lambda b: inner(a, b), [-1, 1]), [-1, 1])

        # This triple-nested quadrature dominates the test's runtime. Its assertion tolerance
        # (1e-8) is far looser than the d=2 check above (1e-12), so it doesn't need the same
        # working precision: at dps=10 the true numerical error is ~7e-12 (verified against a
        # correct formula), a >1000x margin below the 1e-8 tolerance, and an injected 0.1%
        # normalization bug is still caught cleanly (fails almosteq) at this precision -- while
        # cutting this quadrature's runtime by roughly 4x versus dps=16.
        mp.mp.dps = 10
        self.assertTrue(mp.almosteq(mp.e ** mp.mpf(LKJ(3, 2.0)._log_c) * z3(2.0), 1, 1e-8))

    def test_sampler_valid_and_marginal_beta(self):
        for d, eta in [(3, 1.5), (4, 3.0), (5, 1.0)]:
            dist = LKJ(d, eta)
            samples = dist.sampler(seed=0).sample(8000)
            offs = np.array([R[0, 1] for R in samples])
            with self.subTest(d=d, eta=eta):
                self.assertTrue(all(np.allclose(np.diag(R), 1.0) for R in samples[:200]))
                self.assertTrue(all(np.all(np.linalg.eigvalsh(R) > -1e-9) for R in samples[:200]))
                # exact marginal: (r+1)/2 ~ Beta(eta+(d-2)/2, eta+(d-2)/2)
                a = eta + (d - 2) / 2.0
                self.assertAlmostEqual(((offs + 1) / 2).mean(), ss.beta(a, a).mean(), delta=0.01)
                self.assertAlmostEqual(((offs + 1) / 2).var(), ss.beta(a, a).var(), delta=0.005)

    def test_seq_matches_scalar(self):
        d = LKJ(4, 2.5)
        rs = d.sampler(seed=0).sample(6)
        scalar = np.array([d.log_density(R) for R in rs])
        self.assertTrue(np.allclose(scalar, d.seq_log_density(d.dist_to_encoder().seq_encode(rs))))

    def test_non_pd_is_neg_inf(self):
        bad = np.array([[1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]])  # not positive definite
        self.assertEqual(LKJ(3, 2.0).log_density(bad), -np.inf)

    def test_not_pd_with_positive_determinant_is_neg_inf(self):
        # symmetric, unit diagonal, determinant > 0 (a determinant-sign check would accept this),
        # but eigenvalues [-0.6, -0.2, 3.8] -- not positive definite.
        bad = np.array([[1.0, -1.5, -1.5], [-1.5, 1.0, 1.2], [-1.5, 1.2, 1.0]])
        self.assertGreater(np.linalg.det(bad), 0.0)
        self.assertEqual(LKJ(3, 2.0).log_density(bad), -np.inf)
        # and for eta < 1, where (eta - 1) is negative -- a naive sign-propagation through the
        # arithmetic would flip -inf to +inf instead of keeping it -inf.
        self.assertEqual(LKJ(3, 0.5).log_density(bad), -np.inf)

    def test_wrong_size_is_neg_inf(self):
        wrong_size = np.eye(2)  # a valid 2x2 correlation matrix, but this LKJ is dim=3
        self.assertEqual(LKJ(3, 2.0).log_density(wrong_size), -np.inf)

    def test_non_symmetric_is_neg_inf(self):
        non_symmetric = np.array([[1.0, 0.5, 0.0], [0.9, 1.0, 0.0], [0.0, 0.0, 1.0]])
        self.assertEqual(LKJ(3, 2.0).log_density(non_symmetric), -np.inf)

    def test_non_unit_diagonal_is_neg_inf(self):
        non_unit_diag = np.array([[2.0, 0.5, 0.0], [0.5, 2.0, 0.0], [0.0, 0.0, 2.0]])  # scaled covariance, not corr
        self.assertEqual(LKJ(3, 2.0).log_density(non_unit_diag), -np.inf)

    def test_seq_encode_flags_invalid_rows_without_corrupting_valid_ones(self):
        from mixle.stats.matrix.lkj import LKJDataEncoder

        d = LKJ(3, 2.0)
        good = d.sampler(seed=3).sample(2)
        bad = np.array([[1.0, -1.5, -1.5], [-1.5, 1.0, 1.2], [-1.5, 1.2, 1.0]])
        batch = np.array([good[0], bad, good[1]])
        encoded = LKJDataEncoder().seq_encode(batch)
        self.assertEqual(encoded[1], -np.inf)
        np.testing.assert_allclose(encoded[[0, 2]], [np.linalg.slogdet(good[0])[1], np.linalg.slogdet(good[1])[1]])
        seq_ll = d.seq_log_density(encoded)
        self.assertEqual(seq_ll[1], -np.inf)
        np.testing.assert_allclose(seq_ll[[0, 2]], [d.log_density(good[0]), d.log_density(good[1])])

    def test_accumulator_poisons_sum_log_det_for_invalid_matrix(self):
        from mixle.stats.matrix.lkj import LKJAccumulator

        bad = np.array([[1.0, -1.5, -1.5], [-1.5, 1.0, 1.2], [-1.5, 1.2, 1.0]])
        acc = LKJAccumulator()
        acc.update(np.eye(3), 1.0, None)
        acc.update(bad, 1.0, None)
        count, sum_log_det = acc.value()
        self.assertEqual(count, 2.0)
        self.assertEqual(sum_log_det, -np.inf)

    def test_accumulator_zero_weight_on_invalid_matrix_does_not_poison(self):
        from mixle.stats.matrix.lkj import LKJAccumulator

        bad = np.array([[1.0, -1.5, -1.5], [-1.5, 1.0, 1.2], [-1.5, 1.2, 1.0]])
        acc = LKJAccumulator()
        acc.update(np.eye(3), 1.0, None)  # log det = 0
        acc.update(bad, 0.0, None)  # zero weight: must contribute exactly 0, not nan (0 * -inf)
        count, sum_log_det = acc.value()
        self.assertEqual(count, 1.0)
        self.assertEqual(sum_log_det, 0.0)

    def test_seq_update_zero_weight_on_invalid_row_does_not_poison(self):
        from mixle.stats.matrix.lkj import LKJAccumulator, LKJDataEncoder

        bad = np.array([[1.0, -1.5, -1.5], [-1.5, 1.0, 1.2], [-1.5, 1.2, 1.0]])
        encoded = LKJDataEncoder().seq_encode(np.array([np.eye(3), bad]))
        acc = LKJAccumulator()
        acc.seq_update(encoded, np.array([1.0, 0.0]), None)
        count, sum_log_det = acc.value()
        self.assertEqual(count, 1.0)
        self.assertEqual(sum_log_det, 0.0)

    def test_mle_recovers_eta(self):
        for d, eta in [(3, 2.0), (4, 4.0)]:
            true = LKJ(d, eta)
            data = true.sampler(seed=1).sample(8000)
            est = true.estimator()
            acc = est.accumulator_factory().make()
            acc.seq_update(true.dist_to_encoder().seq_encode(data), np.ones(len(data)), None)
            with self.subTest(d=d, eta=eta):
                self.assertAlmostEqual(est.estimate(len(data), acc.value()).eta, eta, delta=0.2)

    def test_capabilities(self):
        self.assertTrue(mixle.supports(LKJ(3, 2.0), Fittable))


if __name__ == "__main__":
    unittest.main()
