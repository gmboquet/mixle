"""Tests for pysp.stats symdirichlet, catdirichlet, int_range, bernoulli, and beta.

Migrated from the original wave-2 suite. Each test class preserves the regression
coverage of runtime-confirmed math bugs, asserted now against the folded pysp.stats
implementations:
  - SymmetricDirichletDistribution.log_density returns +nc (gammaln(3) = +log(2) for n = 3) in the
    alpha == 1 branch, and its sampler reads dim,
  - DictDirichletDistribution.log_density uses the correctly-signed gammaln terms (Dir(2, 2) at
    (.5, .5) gives +log(1.5)), poisoning no DictDirichlet-prior model_log_density, and its sampler
    requires a dict alpha,
  - IntegerCategoricalDistribution normalizes the density over its support, and the conjugate
    estimate clamps counts + (alpha - 1) at the simplex boundary with a posterior-mean fallback,
  - BernoulliEstimator estimates a conjugate Beta posterior and a no-prior MLE, and
    BernoulliDistribution.expected_log_density uses the digamma closed form,
  - BetaDistribution is a pysp.stats ProbabilityDistribution.

The legacy estimator API was estimate(suff_stat); the stats API is estimate(nobs, suff_stat), so a
local fit() helper supplies the leading None.
"""

import unittest

import numpy as np
import scipy.integrate
import scipy.stats

from pysp.stats.bayes.dict_dirichlet import DictDirichletDistribution
from pysp.stats.bayes.dirichlet import DirichletDistribution
from pysp.stats.bayes.symmetric_dirichlet import SymmetricDirichletDistribution
from pysp.stats.compute.pdist import ProbabilityDistribution
from pysp.stats.leaf.bernoulli import (
    BernoulliAccumulatorFactory,
    BernoulliDistribution,
    BernoulliEstimator,
)
from pysp.stats.leaf.beta import BetaDistribution, BetaSampler
from pysp.stats.leaf.categorical import CategoricalDistribution, CategoricalEstimator
from pysp.stats.leaf.integer_categorical import (
    IntegerCategoricalAccumulator,
    IntegerCategoricalAccumulatorFactory,
    IntegerCategoricalDistribution,
    IntegerCategoricalEstimator,
)


def fit(data, est):
    acc = est.accumulator_factory().make()
    for x in data:
        acc.update(x, 1.0, None)
    return acc, est.estimate(None, acc.value())


class SymmetricDirichletTestCase(unittest.TestCase):
    def test_log_density_unit_alpha(self):
        # regression: returned -log(2) where the correct value is
        # log Gamma(3) = +log(2) (Dirichlet(1,..,1) density is (n-1)!)
        d = SymmetricDirichletDistribution(1.0)
        x = np.array([0.2, 0.3, 0.5])
        self.assertAlmostEqual(d.log_density(x), np.log(2.0), places=12)
        self.assertAlmostEqual(d.log_density(x), scipy.stats.dirichlet.logpdf(x, np.ones(3)), places=12)

    def test_log_density_matches_scipy(self):
        d = SymmetricDirichletDistribution(2.5)
        x = np.array([0.2, 0.3, 0.5])
        self.assertAlmostEqual(d.log_density(x), scipy.stats.dirichlet.logpdf(x, np.ones(3) * 2.5), places=12)

    def test_parameter_roundtrip(self):
        d = SymmetricDirichletDistribution(2.5)
        d.set_parameters(3.5)
        self.assertEqual(d.get_parameters(), 3.5)

    def test_sampler_with_dim(self):
        # regression: sample() used to raise AttributeError (dist.ndim unset)
        d = SymmetricDirichletDistribution(2.0, dim=3)
        s = d.sampler(seed=1)
        x = s.sample()
        self.assertEqual(np.shape(x), (3,))
        self.assertAlmostEqual(np.sum(x), 1.0, places=12)
        xs = s.sample(size=4)
        self.assertEqual(np.shape(xs), (4, 3))
        self.assertTrue(np.allclose(np.sum(xs, axis=1), 1.0))

    def test_sampler_without_dim_raises(self):
        d = SymmetricDirichletDistribution(2.0)
        with self.assertRaises(ValueError):
            d.sampler(seed=1).sample()


class DictDirichletTestCase(unittest.TestCase):
    def test_log_density_matches_scipy(self):
        # regression: Dir(2, 2) at (.5, .5) used to give -3.178 instead of
        # log(1.5) = +0.405 (both gammaln terms were sign-flipped)
        d = DictDirichletDistribution({"a": 2.0, "b": 2.0})
        x = {"a": 0.5, "b": 0.5}
        self.assertAlmostEqual(d.log_density(x), np.log(1.5), places=12)

        d = DictDirichletDistribution({"a": 2.0, "b": 1.5, "c": 1.0})
        x = {"a": 0.2, "b": 0.3, "c": 0.5}
        ref = scipy.stats.dirichlet.logpdf([0.2, 0.3, 0.5], [2.0, 1.5, 1.0])
        self.assertAlmostEqual(d.log_density(x), ref, places=12)

    def test_log_density_scalar_alpha(self):
        x = {"a": 0.2, "b": 0.3, "c": 0.5}
        # alpha == 1 branch: density of the uniform Dirichlet is Gamma(n)
        d = DictDirichletDistribution(1.0)
        self.assertAlmostEqual(d.log_density(x), np.log(2.0), places=12)
        # alpha != 1 branch matches scipy
        d = DictDirichletDistribution(2.5)
        ref = scipy.stats.dirichlet.logpdf([0.2, 0.3, 0.5], np.ones(3) * 2.5)
        self.assertAlmostEqual(d.log_density(x), ref, places=12)

    def test_model_log_density_through_categorical(self):
        # regression: the sign flips poisoned CategoricalEstimator's
        # model_log_density for every DictDirichlet prior
        est = CategoricalEstimator(prior=DictDirichletDistribution({"a": 2.0, "b": 2.0}))
        model = CategoricalDistribution({"a": 0.5, "b": 0.5})
        self.assertAlmostEqual(est.model_log_density(model), np.log(1.5), places=12)

    def test_sampler_dict_alpha(self):
        # regression: sampler read a nonexistent dist.dist attribute and had
        # no sample method
        d = DictDirichletDistribution({"a": 2.0, "b": 3.0})
        s = d.sampler(seed=2)
        x = s.sample()
        self.assertEqual(set(x.keys()), {"a", "b"})
        self.assertAlmostEqual(sum(x.values()), 1.0, places=12)
        xs = s.sample(size=3)
        self.assertEqual(len(xs), 3)
        for u in xs:
            self.assertEqual(set(u.keys()), {"a", "b"})
            self.assertAlmostEqual(sum(u.values()), 1.0, places=12)

    def test_sampler_scalar_alpha_raises(self):
        d = DictDirichletDistribution(2.0)
        with self.assertRaises(ValueError):
            d.sampler(seed=2).sample()


class IntegerCategoricalTestCase(unittest.TestCase):
    def test_density_normalizes_over_support(self):
        # regression: log_const was log(2 + default_value), so every density
        # was off by a factor of 1/2 with default_value = 0
        pv = [0.2, 0.3, 0.5]
        d = IntegerCategoricalDistribution(0, pv)
        for x, p in enumerate(pv):
            self.assertAlmostEqual(d.log_density(x), np.log(p), places=12)
        total = sum(d.density(x) for x in range(3))
        self.assertAlmostEqual(total, 1.0, places=12)

    def test_density_normalizes_with_min_index(self):
        d = IntegerCategoricalDistribution(2, [0.4, 0.6])
        self.assertAlmostEqual(d.density(2) + d.density(3), 1.0, places=12)
        self.assertEqual(d.density(1), 0.0)

    def test_seq_log_density_matches_scalar(self):
        d = IntegerCategoricalDistribution(1, [0.2, 0.3, 0.5])
        data = d.sampler(seed=3).sample(size=50)
        enc = d.dist_to_encoder().seq_encode(data)
        self.assertTrue(np.allclose(d.seq_log_density(enc), [d.log_density(u) for u in data]))

    def test_estimate_roundtrip(self):
        pv = np.array([0.2, 0.3, 0.5])
        d = IntegerCategoricalDistribution(0, pv)
        data = d.sampler(seed=4).sample(size=400)
        est = IntegerCategoricalEstimator(min_val=0, max_val=2)
        acc, fitted = fit(data, est)
        self.assertTrue(np.allclose(fitted.p_vec, pv, atol=0.08))
        self.assertAlmostEqual(np.sum(fitted.p_vec), 1.0, places=10)

    def test_conjugate_estimate_clamps_small_alpha(self):
        # regression: counts + (alpha - 1) went negative for alpha < 1
        est = IntegerCategoricalEstimator(min_val=0, max_val=2, prior=SymmetricDirichletDistribution(0.5))
        acc, fitted = fit([0, 0, 1], est)
        self.assertTrue(np.all(fitted.p_vec >= 0.0))
        self.assertAlmostEqual(np.sum(fitted.p_vec), 1.0, places=12)
        self.assertTrue(np.allclose(fitted.p_vec, [0.75, 0.25, 0.0]))
        # posterior carries counts + alpha
        self.assertTrue(np.allclose(fitted.get_prior().get_parameters(), [2.5, 1.5, 0.5]))

    def test_conjugate_estimate_posterior_mean_fallback(self):
        # MAP is degenerate with no data and alpha < 1; fall back to the
        # posterior mean (uniform here)
        est = IntegerCategoricalEstimator(min_val=0, max_val=2, prior=SymmetricDirichletDistribution(0.5))
        acc = est.accumulator_factory().make()
        acc.seq_update(np.asarray([0, 1, 2]), np.zeros(3), None)
        fitted = est.estimate(None, acc.value())
        self.assertTrue(np.allclose(fitted.p_vec, np.ones(3) / 3.0))

    def test_accumulator_factory_make(self):
        factory = IntegerCategoricalAccumulatorFactory(0, 3)
        acc = factory.make()
        self.assertIsInstance(acc, IntegerCategoricalAccumulator)
        acc.update(1, 1.0, None)
        min_val, counts = acc.value()
        self.assertEqual(min_val, 0)
        self.assertEqual(counts[1], 1.0)

        est = IntegerCategoricalEstimator(min_val=0, max_val=3)
        self.assertIsInstance(est.accumulator_factory(), IntegerCategoricalAccumulatorFactory)

    def test_expected_log_density_finite(self):
        d = IntegerCategoricalDistribution(0, [0.2, 0.3, 0.5], prior=DirichletDistribution([2.0, 3.0, 5.0]))
        for x in range(3):
            self.assertTrue(np.isfinite(d.expected_log_density(x)))
        enc = d.dist_to_encoder().seq_encode([0, 1, 2])
        self.assertTrue(np.allclose(d.seq_expected_log_density(enc), [d.expected_log_density(u) for u in range(3)]))


class BernoulliTestCase(unittest.TestCase):
    def test_density_normalizes(self):
        d = BernoulliDistribution(0.3)
        self.assertAlmostEqual(d.density(True) + d.density(False), 1.0, places=12)
        self.assertAlmostEqual(d.log_density(True), np.log(0.3), places=12)
        self.assertAlmostEqual(d.log_density(False), np.log(0.7), places=12)

    def test_seq_log_density_matches_scalar(self):
        d = BernoulliDistribution(0.3)
        data = d.sampler(seed=5).sample(size=50)
        enc = d.dist_to_encoder().seq_encode(data)
        self.assertTrue(np.allclose(d.seq_log_density(enc), [d.log_density(u) for u in data]))

    def test_conjugate_estimate_roundtrip(self):
        d = BernoulliDistribution(0.3)
        data = d.sampler(seed=6).sample(size=400)
        est = BernoulliEstimator(prior=BetaDistribution(2.0, 2.0))
        acc, fitted = fit(data, est)
        count, psum = acc.value()
        nsum = count - psum
        self.assertAlmostEqual(count, 400.0, places=12)
        self.assertLess(abs(fitted.p - 0.3), 0.07)
        # MAP under Beta(a, b) and posterior carried as the new prior
        self.assertAlmostEqual(fitted.p, (psum + 1.0) / (psum + nsum + 2.0), places=12)
        self.assertEqual(fitted.get_prior().get_parameters(), (2.0 + psum, 2.0 + nsum))

    def test_estimate_without_prior(self):
        # no conjugate prior -> plain relative-frequency MLE
        est = BernoulliEstimator()
        fitted = est.estimate(None, (100.0, 30.0))
        self.assertAlmostEqual(fitted.p, 0.3, places=12)
        self.assertIsNone(fitted.get_prior())
        self.assertFalse(fitted.has_conj_prior)

    def test_no_prior_expected_log_density(self):
        # regression: a non-conjugate / absent prior degenerates to the plug-in log_density
        d = BernoulliDistribution(0.3)
        self.assertAlmostEqual(d.expected_log_density(True), d.log_density(True), places=12)

    def test_expected_log_density_digamma_terms(self):
        from scipy.special import digamma

        d = BernoulliDistribution(0.3, prior=BetaDistribution(2.0, 3.0))
        self.assertAlmostEqual(d.expected_log_density(True), digamma(2.0) - digamma(5.0), places=12)
        self.assertAlmostEqual(d.expected_log_density(False), digamma(3.0) - digamma(5.0), places=12)

    def test_accumulator_factory(self):
        est = BernoulliEstimator()
        self.assertIsInstance(est.accumulator_factory(), BernoulliAccumulatorFactory)

    def test_seq_update_matches_update(self):
        est = BernoulliEstimator()
        data = [True, False, True, True, False]
        acc1 = est.accumulator_factory().make()
        for x in data:
            acc1.update(x, 1.0, None)
        acc2 = est.accumulator_factory().make()
        acc2.seq_update(np.asarray(data, dtype=bool), np.ones(len(data)), None)
        self.assertEqual(acc1.value(), acc2.value())


class BetaTestCase(unittest.TestCase):
    def test_uses_stats_base_class(self):
        # regression: the legacy Beta inherited from the wrong base; the stats Beta is a
        # pysp.stats ProbabilityDistribution.
        self.assertIsInstance(BetaDistribution(2.0, 3.0), ProbabilityDistribution)

    def test_log_density_matches_scipy(self):
        d = BetaDistribution(2.5, 3.5)
        for x in (0.1, 0.4, 0.9):
            self.assertAlmostEqual(d.log_density(x), scipy.stats.beta.logpdf(x, 2.5, 3.5), places=12)
            self.assertAlmostEqual(d.density(x), scipy.stats.beta.pdf(x, 2.5, 3.5), places=12)

    def test_density_normalizes(self):
        d = BetaDistribution(2.5, 3.5)
        total = scipy.integrate.quad(d.density, 0, 1)[0]
        self.assertAlmostEqual(total, 1.0, places=8)

    def test_sampler(self):
        d = BetaDistribution(2.0, 5.0)
        s = d.sampler(seed=7)
        self.assertIsInstance(s, BetaSampler)
        x = s.sample()
        self.assertTrue(0.0 < x < 1.0)
        xs = s.sample(size=100)
        self.assertEqual(len(xs), 100)
        self.assertTrue(np.all((xs > 0.0) & (xs < 1.0)))


if __name__ == "__main__":
    unittest.main()
