"""Zero-inflated count models: analytic density, sampling, and EM recovery of pi + the base."""

import unittest

import numpy as np
from scipy.stats import poisson

from pysp.stats import (
    BinomialDistribution,
    BinomialEstimator,
    PoissonDistribution,
    PoissonEstimator,
    ZeroInflatedDataEncoder,
    ZeroInflatedDistribution,
    ZeroInflatedEstimator,
)
from pysp.utils.estimation import optimize


class ZeroInflatedDensityTest(unittest.TestCase):
    def setUp(self):
        self.dist = ZeroInflatedDistribution(PoissonDistribution(4.0), pi=0.3)

    def test_density_matches_definition(self):
        d = self.dist
        self.assertAlmostEqual(d.density(0), 0.3 + 0.7 * poisson.pmf(0, 4), places=10)
        for k in (1, 2, 5):
            self.assertAlmostEqual(d.density(k), 0.7 * poisson.pmf(k, 4), places=10)

    def test_normalizes(self):
        mass = self.dist.density(0) + sum(self.dist.density(k) for k in range(1, 80))
        self.assertAlmostEqual(mass, 1.0, places=9)

    def test_seq_matches_scalar(self):
        data = [0, 0, 1, 0, 3, 2, 0, 5, 0, 1, 7]
        enc = self.dist.dist_to_encoder().seq_encode(data)
        np.testing.assert_allclose(self.dist.seq_log_density(enc), [self.dist.log_density(x) for x in data], atol=1e-12)

    def test_pi_zero_reduces_to_base(self):
        base = PoissonDistribution(4.0)
        zi = ZeroInflatedDistribution(base, pi=0.0)
        for k in (0, 1, 5):
            self.assertAlmostEqual(zi.log_density(k), base.log_density(k), places=12)

    def test_invalid_pi_raises(self):
        with self.assertRaises(ValueError):
            ZeroInflatedDistribution(PoissonDistribution(1.0), pi=1.0)
        with self.assertRaises(ValueError):
            ZeroInflatedDistribution(PoissonDistribution(1.0), pi=-0.1)

    def test_string_round_trip(self):
        d = ZeroInflatedDistribution(PoissonDistribution(4.0), pi=0.3, name="zi", keys="k")
        self.assertEqual(str(eval(str(d))), str(d))

    def test_encoder_equality(self):
        e1 = self.dist.dist_to_encoder()
        e2 = ZeroInflatedDataEncoder(PoissonDistribution(4.0).dist_to_encoder())
        self.assertEqual(e1, e2)


class ZeroInflatedSamplerTest(unittest.TestCase):
    def test_excess_zero_fraction(self):
        # observed zero fraction ~ pi + (1-pi) P_base(0)
        d = ZeroInflatedDistribution(PoissonDistribution(4.0), pi=0.3)
        x = np.array(d.sampler(seed=0).sample(20000))
        expected_zero = 0.3 + 0.7 * float(poisson.pmf(0, 4))
        self.assertAlmostEqual(np.mean(x == 0), expected_zero, delta=0.02)
        self.assertTrue(np.all(x >= 0))


class ZeroInflatedEMTest(unittest.TestCase):
    def test_zip_recovers_pi_and_rate(self):
        truth = ZeroInflatedDistribution(PoissonDistribution(4.0), pi=0.3)
        x = truth.sampler(seed=1).sample(8000)
        fit = optimize(
            x, ZeroInflatedEstimator(PoissonEstimator()), max_its=60, rng=np.random.RandomState(0), print_iter=0
        )
        self.assertAlmostEqual(fit.pi, 0.3, delta=0.03)
        self.assertAlmostEqual(fit.base.lam, 4.0, delta=0.2)

    def test_zi_binomial_recovers(self):
        truth = ZeroInflatedDistribution(BinomialDistribution(0.4, 10), pi=0.25)
        x = truth.sampler(seed=1).sample(8000)
        fit = optimize(
            x, ZeroInflatedEstimator(BinomialEstimator()), max_its=60, rng=np.random.RandomState(0), print_iter=0
        )
        self.assertAlmostEqual(fit.pi, 0.25, delta=0.03)
        self.assertAlmostEqual(fit.base.p, 0.4, delta=0.03)

    def test_em_is_monotone(self):
        truth = ZeroInflatedDistribution(PoissonDistribution(3.0), pi=0.4)
        x = truth.sampler(seed=2).sample(4000)
        enc = truth.dist_to_encoder().seq_encode(x)
        lls = [
            float(
                np.sum(
                    optimize(
                        x,
                        ZeroInflatedEstimator(PoissonEstimator()),
                        max_its=k,
                        rng=np.random.RandomState(0),
                        print_iter=0,
                    ).seq_log_density(enc)
                )
            )
            for k in range(1, 8)
        ]
        self.assertTrue(all(lls[i] <= lls[i + 1] + 1e-4 for i in range(len(lls) - 1)))


if __name__ == "__main__":
    unittest.main()
