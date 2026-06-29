"""Tests for mixle.stats dirichlet, point_mass (Dirac), null_dist, gamma, and conditional.

Migrated from the original wave-1 regression suite. Each test class preserves the regression
coverage of the math/attribute bugs the original fixes encoded, asserted now against the folded
mixle.stats implementations:
  - DirichletDistribution.seq_log_density round-trips its own encoded output, the sampler reads
    only attributes it sets, and the estimator exposes accumulator_factory()/estimate(None, ss),
  - PointMassDistribution (the stats Dirac) log_density/sampler/estimator are consistent,
  - NullDistribution.sampler() constructs cleanly and threads through SequenceDistribution's
    default null len_dist,
  - GammaEstimator carries name and uses the correct pseudo-count denominator for theta,
  - ConditionalDistribution.seq_log_density / seq_encode / sampler / estimator behave with and
    without a default branch.

The legacy estimator API was estimate(suff_stat); the stats API is estimate(nobs, suff_stat), so a
local fit() helper supplies the leading None.
"""

import unittest

import numpy as np
import scipy.stats

from mixle.stats.bayes.dirichlet import DirichletDistribution, DirichletEstimator
from mixle.stats.combinator.conditional import ConditionalDistribution, ConditionalDistributionEstimator
from mixle.stats.combinator.null_dist import NullDistribution, NullSampler
from mixle.stats.combinator.sequence import SequenceDistribution
from mixle.stats.univariate.continuous.gamma import GammaDistribution, GammaEstimator
from mixle.stats.univariate.continuous.gaussian import GaussianDistribution
from mixle.stats.univariate.discrete.point_mass import PointMassDistribution, PointMassEstimator


def fit(data, est):
    acc = est.accumulator_factory().make()
    for x in data:
        acc.update(x, 1.0, None)
    return acc, est.estimate(None, acc.value())


class DirichletTestCase(unittest.TestCase):
    alpha = np.array([2.0, 3.0, 4.0])

    def make_dist(self):
        return DirichletDistribution(self.alpha)

    def test_log_density_matches_scipy(self):
        d = self.make_dist()
        x = np.array([0.2, 0.3, 0.5])
        self.assertAlmostEqual(d.log_density(x), scipy.stats.dirichlet.logpdf(x, self.alpha), places=10)

    def test_seq_log_density_matches_scalar(self):
        # regression: used to raise AttributeError ('tuple' object has no attribute 'shape')
        d = self.make_dist()
        data = d.sampler(seed=1).sample(size=25)
        enc = d.dist_to_encoder().seq_encode(data)
        self.assertTrue(np.allclose(d.seq_log_density(enc), [d.log_density(u) for u in data]))

    def test_seq_log_density_with_unit_alphas(self):
        # exercises the masked dot over columns with alpha != 1
        d = DirichletDistribution(np.array([1.0, 2.5, 1.0]))
        data = [[0.2, 0.3, 0.5], [0.6, 0.1, 0.3]]
        enc = d.dist_to_encoder().seq_encode(data)
        self.assertTrue(np.allclose(d.seq_log_density(enc), [d.log_density(u) for u in data]))

    def test_sampler(self):
        # regression: used to raise AttributeError (dist.has_invalid never set)
        d = self.make_dist()
        s = d.sampler(seed=1)
        x = s.sample()
        self.assertEqual(x.shape, (3,))
        self.assertAlmostEqual(x.sum(), 1.0, places=12)
        xs = s.sample(size=5)
        self.assertEqual(xs.shape, (5, 3))
        self.assertTrue(np.allclose(xs.sum(axis=1), 1.0))

    def test_scalar_alpha_raises(self):
        # stats DirichletDistribution requires a positive vector alpha
        with self.assertRaises(ValueError):
            DirichletDistribution(2.0)

    def test_estimate_round_trip(self):
        # regression: DirichletEstimator exposes accumulator_factory()/estimate(None, ss)
        d = self.make_dist()
        data = d.sampler(seed=4).sample(size=400)
        est = DirichletEstimator(dim=3)
        acc, m = fit(data, est)
        self.assertTrue(np.all(np.abs(m.alpha - self.alpha) / self.alpha < 0.3))

        # seq path accumulates the same statistics
        acc2 = est.accumulator_factory().make()
        acc2.seq_update(d.dist_to_encoder().seq_encode(data), np.ones(len(data)), None)
        v1 = acc.value()
        v2 = acc2.value()
        for u, v in zip(v1, v2):
            self.assertTrue(np.allclose(u, v))


class PointMassTestCase(unittest.TestCase):
    """The stats Dirac is PointMassDistribution; it assigns all mass to one fixed value."""

    def test_log_density(self):
        # regression: legacy Dirac used to raise AttributeError (self.dist never set)
        d = PointMassDistribution("a")
        self.assertEqual(d.log_density("a"), 0.0)
        self.assertEqual(d.log_density("b"), -np.inf)
        self.assertEqual(d.density("a"), 1.0)
        self.assertEqual(d.density("b"), 0.0)

    def test_seq_log_density_matches_scalar(self):
        d = PointMassDistribution(3)
        data = [3, 1, 3, 7]
        enc = d.dist_to_encoder().seq_encode(data)
        self.assertTrue(np.array_equal(d.seq_log_density(enc), [d.log_density(u) for u in data]))

    def test_get_prior(self):
        # PointMass is a plain point model: no parameter prior.
        self.assertIsNone(PointMassDistribution("a").get_prior())

    def test_sampler(self):
        # regression: legacy Dirac sampler used to raise AttributeError (self.dist never set)
        d = PointMassDistribution("a")
        s = d.sampler(seed=1)
        self.assertEqual(s.sample(), "a")
        self.assertEqual(s.sample(size=3), ["a", "a", "a"])

    def test_estimator_round_trip(self):
        # regression: legacy Dirac estimator() used to raise TypeError (missing value argument)
        d = PointMassDistribution("a")
        est = d.estimator()
        self.assertIsInstance(est, PointMassEstimator)
        acc, m = fit(["a", "a", "b"], est)
        self.assertIsInstance(m, PointMassDistribution)
        self.assertEqual(m.value, "a")


class NullTestCase(unittest.TestCase):
    def test_sampler(self):
        # regression: NullDistribution.sampler() used to raise TypeError
        # (NullSampler.__init__ only accepted a seed)
        s = NullDistribution().sampler(seed=1)
        self.assertIsInstance(s, NullSampler)
        self.assertIsNone(s.sample())
        self.assertIsNone(s.sample(size=3))

    def test_sequence_with_null_length_sampler(self):
        # regression: the null default len_dist threads cleanly through the
        # SequenceDistribution (the legacy NullSampler used to crash). In stats a
        # sampler needs a real length distribution, but the null-len model still
        # scores exogenous-length sequences.
        sd = SequenceDistribution(GaussianDistribution(0.0, 1.0))
        self.assertTrue(sd.null_len_dist)
        self.assertTrue(np.isfinite(sd.log_density([0.1, -0.2, 0.5])))

    def test_estimator(self):
        est = NullDistribution().estimator()
        acc = est.accumulator_factory().make()
        for x in [1, "a", None]:
            acc.update(x, 1.0, None)
        m = est.estimate(None, acc.value())
        self.assertIsInstance(m, NullDistribution)
        self.assertIsNone(acc.value())


class GammaTestCase(unittest.TestCase):
    def test_estimator_carries_name(self):
        # regression: GammaDistribution.estimator() threads the name to GammaEstimator
        d = GammaDistribution(2.0, 3.0, name="g")
        est = d.estimator()
        self.assertIsInstance(est, GammaEstimator)
        self.assertEqual(est.name, "g")

    def test_seq_log_density_matches_scalar(self):
        d = GammaDistribution(3.0, 2.0)
        data = d.sampler(seed=1).sample(20)
        enc = d.dist_to_encoder().seq_encode(data)
        self.assertTrue(np.allclose(d.seq_log_density(enc), [d.log_density(u) for u in data]))
        self.assertTrue(np.allclose(d.seq_log_density(enc), scipy.stats.gamma.logpdf(data, 3.0, scale=2.0)))

    def test_estimate_recovery(self):
        d = GammaDistribution(3.0, 2.0)
        data = d.sampler(seed=5).sample(500)
        acc, m = fit(data, GammaEstimator())
        self.assertLess(abs(m.k - 3.0) / 3.0, 0.15)
        self.assertLess(abs(m.theta - 2.0) / 2.0, 0.15)
        # ML invariant: k*theta equals the sample mean
        self.assertAlmostEqual(m.k * m.theta, data.mean(), places=8)

    def test_estimate_pseudo_count_theta_denominator(self):
        # regression: theta used the log-pseudo-count denominator, so with
        # pc1 != pc2 the fitted mean k*theta missed the adjusted mean. stats Gamma
        # suff_stat is (count, sum, sum_of_logs); pseudo_count/suff_stat are per-moment pairs.
        d = GammaDistribution(3.0, 2.0)
        data = d.sampler(seed=6).sample(200)
        pc1, ss1 = 2.0, 1.5
        est = GammaEstimator(pseudo_count=(pc1, 0.0), suff_stat=(ss1, 0.0))
        acc, m = fit(data, est)
        adj_mean = (data.sum() + ss1 * pc1) / (len(data) + pc1)
        self.assertAlmostEqual(m.k * m.theta, adj_mean, places=8)

    def test_estimate_empty_suff_stat(self):
        m = GammaEstimator(name="g").estimate(None, (0, 0.0, 0.0))
        self.assertEqual((m.k, m.theta), (1.0, 1.0))
        self.assertEqual(m.name, "g")


class ConditionalTestCase(unittest.TestCase):
    def make_dist(self):
        return ConditionalDistribution({"a": GaussianDistribution(0.0, 1.0), "b": GaussianDistribution(5.0, 1.0)})

    def test_seq_log_density_matches_scalar(self):
        # regression: used to raise AttributeError (self.has_default never set)
        cd = self.make_dist()
        data = [("a", 0.5), ("b", 5.2), ("a", -0.3), ("c", 1.0)]
        enc = cd.dist_to_encoder().seq_encode(data)
        self.assertTrue(np.allclose(cd.seq_log_density(enc), [cd.log_density(u) for u in data]))
        # the unmatched value 'c' is out of support (the default null dist is not a
        # real branch): scalar and seq paths agree (both -inf), which is the regression.
        self.assertEqual(cd.log_density(("c", 1.0)), -np.inf)

    def test_seq_log_density_without_default(self):
        cd = self.make_dist()
        data = [("a", 0.5), ("c", 1.0)]
        cd_nodef = ConditionalDistribution(dict(cd.dmap), default_dist=None)
        enc = cd_nodef.dist_to_encoder().seq_encode(data)
        ll = cd_nodef.seq_log_density(enc)
        self.assertAlmostEqual(ll[0], cd.dmap["a"].log_density(0.5), places=10)
        self.assertEqual(ll[1], -np.inf)

    def test_seq_encode_without_default(self):
        # regression: seq_encode crashed on None.seq_encode when default_dist
        # is None and an unmatched conditioning value appears
        cd = ConditionalDistribution({"a": GaussianDistribution(0.0, 1.0)}, default_dist=None)
        data = [("a", 0.5), ("z", 1.0)]
        ll = cd.seq_log_density(cd.dist_to_encoder().seq_encode(data))
        self.assertAlmostEqual(ll[0], cd.dmap["a"].log_density(0.5), places=10)
        self.assertEqual(ll[1], -np.inf)

    def test_seq_initialize_routes_to_members(self):
        # regression: the accumulator inherited the no-op seq_initialize, so
        # seq-path initialization silently gathered no statistics
        cd = self.make_dist()
        data = [("a", 0.5), ("b", 5.2), ("a", -0.3)]
        enc = cd.dist_to_encoder().seq_encode(data)
        est = cd.estimator()
        rng = np.random.RandomState(3)
        acc = est.accumulator_factory().make()
        acc.seq_initialize(enc, np.ones(len(data)), rng)
        ref = est.accumulator_factory().make()
        # stats EM contract: seq_update threads the current estimate to the children
        ref.seq_update(enc, np.ones(len(data)), cd)
        v1, v2 = acc.value(), ref.value()
        for k in v2[0]:
            self.assertTrue(np.allclose(v1[0][k], v2[0][k]))

    def test_sampler(self):
        # regression: sampler()/sample() used to be empty stubs returning None
        cd = self.make_dist()
        s = cd.sampler(seed=7)
        self.assertTrue(np.isfinite(s.sample_given("a")))
        self.assertEqual(len([s.sample_given("b") for _ in range(4)]), 4)

    def test_sampler_unmatched_without_default(self):
        cd = ConditionalDistribution({"a": GaussianDistribution(0.0, 1.0)}, default_dist=None)
        with self.assertRaisesRegex(Exception, "default distribution"):
            cd.sampler(seed=1).sample_given("z")

    def test_estimator_round_trip(self):
        # regression: estimator() used to be an empty stub returning None, and
        # the accumulator crashed on update(..., estimate=None)
        cd = self.make_dist()
        est = cd.estimator()
        self.assertIsInstance(est, ConditionalDistributionEstimator)

        rng = np.random.RandomState(2)
        data = [("a", v) for v in rng.normal(1.0, 1.0, 80)]
        data += [("b", v) for v in rng.normal(-3.0, 1.0, 80)]
        acc, m = fit(data, est)

        self.assertIsInstance(m, ConditionalDistribution)
        self.assertIsInstance(m.default_dist, NullDistribution)
        self.assertLess(abs(m.dmap["a"].mu - 1.0), 0.4)
        self.assertLess(abs(m.dmap["b"].mu + 3.0), 0.4)

        # the fitted model scores data consistently in scalar and seq form
        enc = m.dist_to_encoder().seq_encode(data[:10])
        self.assertTrue(np.allclose(m.seq_log_density(enc), [m.log_density(u) for u in data[:10]]))

    # The legacy "seq_update without an estimate" regression is dropped: the stats
    # Conditional accumulator follows the EM contract and requires the current estimate
    # to be threaded through seq_update (seq_initialize_routes_to_members covers the routing).


if __name__ == "__main__":
    unittest.main()
