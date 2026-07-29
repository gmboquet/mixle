"""Tests for the Tweedie (compound Poisson-Gamma, 1<p<2) distribution.

No external reference is available, so correctness is checked by self-consistency of the series
density (numerical normalization + moments matching mu / phi*mu**p) and against the *exact*
compound-Poisson-Gamma sampler.
"""

import math
import unittest

import numpy as np
from scipy import integrate

from mixle.inference.estimation import optimize
from mixle.stats import TweedieDistribution, TweedieEstimator


class TweedieDistributionTest(unittest.TestCase):
    def test_density_normalizes_and_matches_moments(self):
        for mu, phi, p in [(2.0, 1.0, 1.5), (5.0, 0.5, 1.3), (1.0, 2.0, 1.7)]:
            with self.subTest(mu=repr(mu), phi=repr(phi), p=repr(p)):
                d = TweedieDistribution(mu, phi, p)
                p0 = math.exp(-d.lam)
                hi = mu * 60.0
                mass, _ = integrate.quad(lambda y, d=d: math.exp(d.log_density(y)), 0.0, hi, limit=400)
                mean_int, _ = integrate.quad(lambda y, d=d: y * math.exp(d.log_density(y)), 0.0, hi, limit=400)
                m2_int, _ = integrate.quad(lambda y, d=d: y * y * math.exp(d.log_density(y)), 0.0, hi, limit=400)
                self.assertAlmostEqual(p0 + mass, 1.0, places=3)  # total probability
                self.assertAlmostEqual(mean_int, mu, places=2)  # E[Y] = mu
                self.assertAlmostEqual(m2_int - mu * mu, phi * mu**p, places=2)  # Var = phi*mu**p

    def test_point_mass_at_zero(self):
        d = TweedieDistribution(3.0, 1.0, 1.5)
        self.assertAlmostEqual(d.log_density(0.0), -d.lam, places=12)
        self.assertEqual(d.log_density(-1.0), -np.inf)

    def test_seq_log_density_matches_scalar(self):
        d = TweedieDistribution(2.5, 0.8, 1.4)
        xs = np.array([0.0, 0.3, 1.0, 2.5, 7.0])
        enc = d.dist_to_encoder().seq_encode(list(xs))
        seq = np.asarray(d.seq_log_density(enc))
        scalar = np.array([d.log_density(float(x)) for x in xs])
        np.testing.assert_allclose(seq, scalar, atol=1e-9)

    def test_sampler_moments_and_zero_fraction(self):
        d = TweedieDistribution(2.0, 1.0, 1.5)
        y = d.sampler(0).sample(40000)
        self.assertGreaterEqual(float(np.min(y)), 0.0)
        self.assertAlmostEqual(float(np.mean(y)), 2.0, delta=0.1)  # E[Y] = mu
        self.assertAlmostEqual(float(np.var(y)), 1.0 * 2.0**1.5, delta=0.4)  # Var = phi*mu**p
        self.assertAlmostEqual(float(np.mean(y == 0.0)), math.exp(-d.lam), delta=0.02)  # P(Y=0)

    def test_estimator_recovers_parameters(self):
        true = TweedieDistribution(4.0, 0.7, 1.5)
        data = list(true.sampler(1).sample(30000))
        fit = optimize(data, TweedieEstimator(p=1.5), max_its=1, rng=np.random.RandomState(0), out=None)
        self.assertAlmostEqual(fit.mu, 4.0, delta=0.15)
        self.assertAlmostEqual(fit.phi, 0.7, delta=0.2)

    def test_pseudo_count_smooths_toward_prior(self):
        # estimator(pseudo_count=...) previously ignored pseudo_count entirely -- a silent no-op.
        d = TweedieDistribution(3.0, 2.0, 1.5)
        est = d.estimator(pseudo_count=1.0e6)
        self.assertIsNotNone(est.suff_stat)
        fitted = est.estimate(None, (1.0, 0.01, 0.0001))  # one observation far from mu=3.0
        self.assertAlmostEqual(fitted.mu, d.mu, places=2)
        self.assertAlmostEqual(fitted.phi, d.phi, places=2)

        fitted_plain = d.estimator().estimate(None, (1.0, 0.01, 0.0001))
        self.assertNotAlmostEqual(fitted_plain.mu, d.mu, places=1)

    def test_pseudo_count_recovers_exact_prior_moments(self):
        # feeding the stored suff_stat back through with zero real data (pure pseudo-count blend)
        # must reproduce this distribution's own (mu, phi) almost exactly.
        d = TweedieDistribution(1.5, 0.4, 1.2)
        est = d.estimator(pseudo_count=1.0)
        fitted = est.estimate(None, (0.0, 0.0, 0.0))
        self.assertAlmostEqual(fitted.mu, d.mu, places=6)
        self.assertAlmostEqual(fitted.phi, d.phi, places=6)


class TweedieKeyedSharingTest(unittest.TestCase):
    def test_keyed_accumulators_carry_pooled_statistics(self):
        """Three keyed accumulators over {1,2,3} must all carry the POOLED stats.

        Regression for the key_merge/key_replace bug where a tuple was stored under the
        key instead of the accumulator object, so only the first batch survived.
        """
        from mixle.stats import TweedieEstimator

        est = TweedieEstimator(p=1.5, keys="shared")
        accs = [est.accumulator_factory().make() for _ in range(3)]
        for acc, x in zip(accs, (1.0, 2.0, 3.0)):
            acc.update(x, 1.0, None)

        stats_dict: dict = {}
        for acc in accs:
            acc.key_merge(stats_dict)
        for acc in accs:
            acc.key_replace(stats_dict)

        # Pooled: count=3, sum=1+2+3=6, sum2=1+4+9=14.
        for acc in accs:
            self.assertEqual(acc.value(), (3.0, 6.0, 14.0))


if __name__ == "__main__":
    unittest.main()
