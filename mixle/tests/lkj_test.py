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
