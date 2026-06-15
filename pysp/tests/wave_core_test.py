"""Tests for the wave-core fixes: weighted accumulator combine, WeightedSampler round-trip,
int_spike and dirac_length enumerators, DiagonalGaussianSampler export, and the
DataSequenceEncoder.__str__ recursion fix.
"""

import unittest

import numpy as np

from pysp.stats import DiagonalGaussianSampler, DistributionSampler
from pysp.stats.combinator.weighted import WeightedDistribution, WeightedSampler
from pysp.stats.compute.pdist import DataSequenceEncoder, EnumerationError
from pysp.stats.latent.dirac_length import DiracLengthMixtureDistribution
from pysp.stats.leaf.gaussian import GaussianDistribution
from pysp.stats.leaf.int_range import IntegerCategoricalDistribution
from pysp.stats.leaf.int_spike import IntegerUniformSpikeDistribution
from pysp.stats.leaf.poisson import PoissonDistribution
from pysp.utils.enumeration import freeze

TOL = 1e-9


def check_enumeration_invariants(test, dist, items):
    """Assert non-increasing order, exact dedup, and log_prob == log_density for items."""
    lps = [lp for _, lp in items]
    for i in range(len(lps) - 1):
        test.assertGreaterEqual(lps[i], lps[i + 1] - TOL, "order violated at %d" % i)
    keys = [freeze(v) for v, _ in items]
    test.assertEqual(len(keys), len(set(keys)), "duplicate values yielded")
    with np.errstate(divide="ignore"):
        for v, lp in items:
            test.assertAlmostEqual(lp, dist.log_density(v), delta=TOL, msg="lp mismatch at %r" % (v,))


class WeightedCombineTestCase(unittest.TestCase):
    def test_combine_adds_sufficient_statistics(self):
        dist = WeightedDistribution(GaussianDistribution(1.0, 2.0))
        factory = dist.estimator().accumulator_factory()
        rng1, rng2, rng3 = (np.random.RandomState(7) for _ in range(3))

        data1 = [(0.5, 1.0), (1.5, 2.0), (-0.25, 0.5)]
        data2 = [(2.0, 1.5), (0.0, 3.0)]

        acc1, acc2, acc_all = factory.make(), factory.make(), factory.make()
        for xw in data1:
            acc1.initialize(xw, 1.0, rng1)
        for xw in data2:
            acc2.initialize(xw, 1.0, rng2)
        for xw in data1 + data2:
            acc_all.initialize(xw, 1.0, rng3)

        acc1.combine(acc2.value())

        np.testing.assert_allclose(
            np.asarray(acc1.value(), dtype=float), np.asarray(acc_all.value(), dtype=float), rtol=0, atol=1e-12
        )


class WeightedSamplerTestCase(unittest.TestCase):
    def test_sampler_type_and_shape(self):
        dist = WeightedDistribution(GaussianDistribution(0.0, 1.0))
        sampler = dist.sampler(seed=11)
        self.assertIsInstance(sampler, WeightedSampler)

        one = sampler.sample()
        self.assertIsInstance(one, tuple)
        self.assertEqual(len(one), 2)
        self.assertEqual(one[1], 1.0)

        many = sampler.sample(size=10)
        self.assertEqual(len(many), 10)
        for v, w in many:
            self.assertEqual(w, 1.0)

    def test_sample_then_accumulate_round_trip(self):
        dist = WeightedDistribution(GaussianDistribution(2.0, 1.5))
        samples = dist.sampler(seed=23).sample(size=200)

        est = dist.estimator()
        acc = est.accumulator_factory().make()

        # Encoder consumes the sampler's (value, weight) tuples directly.
        enc = dist.dist_to_encoder().seq_encode(samples)
        acc.seq_initialize(enc, np.ones(len(samples)), np.random.RandomState(5))

        fitted = est.estimate(None, acc.value())
        self.assertIsInstance(fitted, WeightedDistribution)
        self.assertAlmostEqual(fitted.dist.mu, np.mean([v for v, _ in samples]), places=8)

        # Per-observation path consumes the tuples as well.
        acc2 = est.accumulator_factory().make()
        for xw in samples[:10]:
            acc2.initialize(xw, 1.0, np.random.RandomState(3))
        self.assertAlmostEqual(acc2.value()[2], 10.0, places=10)

        # Log-density of a sampled value matches the base distribution.
        v0 = samples[0][0]
        self.assertAlmostEqual(dist.log_density(v0), dist.dist.log_density(v0), places=12)


class IntegerUniformSpikeEnumeratorTestCase(unittest.TestCase):
    def test_spike_first_full_support(self):
        dist = IntegerUniformSpikeDistribution(k=3, num_vals=10, p=0.6, min_val=0)
        items = list(dist.enumerator())
        self.assertEqual(len(items), 10)
        self.assertEqual(items[0][0], 3)
        check_enumeration_invariants(self, dist, items)
        total = np.logaddexp.reduce([lp for _, lp in items])
        self.assertAlmostEqual(total, 0.0, delta=1e-8)

    def test_spike_last_when_small(self):
        dist = IntegerUniformSpikeDistribution(k=-1, num_vals=4, p=0.05, min_val=-2)
        items = list(dist.enumerator())
        self.assertEqual(len(items), 4)
        self.assertEqual(items[-1][0], -1)
        self.assertEqual([v for v, _ in items[:-1]], [-2, 0, 1])
        check_enumeration_invariants(self, dist, items)
        total = np.logaddexp.reduce([lp for _, lp in items])
        self.assertAlmostEqual(total, 0.0, delta=1e-8)

    def test_top_k(self):
        dist = IntegerUniformSpikeDistribution(k=2, num_vals=5, p=0.5, min_val=0)
        items = dist.enumerator().top_k(3)
        self.assertEqual(len(items), 3)
        self.assertEqual(items[0][0], 2)


class DiracLengthMixtureEnumeratorTestCase(unittest.TestCase):
    def test_scalar_component_log_density_matches_vectorized_path(self):
        len_dist = IntegerCategoricalDistribution(0, [0.2, 0.5, 0.3])
        dist = DiracLengthMixtureDistribution(len_dist=len_dist, p=0.7, v=0)
        data = [0, 1, 2]
        enc = dist.dist_to_encoder().seq_encode(data)

        scalar = np.vstack([dist.component_log_density(x) for x in data])
        vectorized = dist.seq_component_log_density(enc)

        np.testing.assert_allclose(scalar, vectorized, rtol=0.0, atol=1e-12)
        self.assertEqual(scalar[0, 1], 0.0)
        self.assertEqual(scalar[1, 1], -np.inf)

    def test_scalar_posterior_matches_vectorized_path(self):
        len_dist = IntegerCategoricalDistribution(0, [0.2, 0.5, 0.3])
        dist = DiracLengthMixtureDistribution(len_dist=len_dist, p=0.7, v=0)
        data = [0, 1, 2]
        enc = dist.dist_to_encoder().seq_encode(data)

        scalar = np.vstack([dist.posterior(x) for x in data])
        vectorized = dist.seq_posterior(enc)

        np.testing.assert_allclose(scalar, vectorized, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(scalar[0], [0.7 * 0.2 / (0.7 * 0.2 + 0.3), 0.3 / (0.7 * 0.2 + 0.3)])
        np.testing.assert_allclose(scalar[1:], [[1.0, 0.0], [1.0, 0.0]])

    def test_posterior_when_dirac_point_is_outside_length_support(self):
        len_dist = IntegerCategoricalDistribution(1, [0.4, 0.6])
        dist = DiracLengthMixtureDistribution(len_dist=len_dist, p=0.7, v=0)
        data = [0, 1, 2]
        enc = dist.dist_to_encoder().seq_encode(data)

        scalar = np.vstack([dist.posterior(x) for x in data])
        vectorized = dist.seq_posterior(enc)

        np.testing.assert_allclose(scalar, vectorized, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(scalar, [[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]])

    def test_finite_support_with_overlap(self):
        len_dist = IntegerCategoricalDistribution(0, [0.2, 0.5, 0.3])
        dist = DiracLengthMixtureDistribution(len_dist=len_dist, p=0.7, v=0)
        items = list(dist.enumerator())
        # Support is {0, 1, 2}; the dirac point 0 overlaps the length-dist support.
        self.assertEqual(len(items), 3)
        check_enumeration_invariants(self, dist, items)
        total = np.logaddexp.reduce([lp for _, lp in items])
        self.assertAlmostEqual(total, 0.0, delta=1e-8)
        # The overlapping point carries mass from both components.
        lp0 = dict((v, lp) for v, lp in items)[0]
        self.assertAlmostEqual(lp0, np.log(0.7 * 0.2 + 0.3), delta=1e-10)

    def test_infinite_support(self):
        dist = DiracLengthMixtureDistribution(len_dist=PoissonDistribution(lam=2.5), p=0.6, v=0)
        items = dist.enumerator().top_k(15)
        self.assertEqual(len(items), 15)
        check_enumeration_invariants(self, dist, items)

    def test_p_one_degenerates_to_length_dist(self):
        len_dist = IntegerCategoricalDistribution(1, [0.4, 0.6])
        dist = DiracLengthMixtureDistribution(len_dist=len_dist, p=1.0, v=0)
        items = list(dist.enumerator())
        self.assertEqual(sorted(v for v, _ in items), [1, 2])
        check_enumeration_invariants(self, dist, items)

    def test_non_enumerable_length_dist_raises(self):
        dist = DiracLengthMixtureDistribution(len_dist=GaussianDistribution(0.0, 1.0), p=0.5, v=0)
        with self.assertRaises(EnumerationError):
            dist.enumerator()


class ExportsTestCase(unittest.TestCase):
    def test_diagonal_gaussian_sampler_export(self):
        from pysp.stats.multivariate.dmvn import DiagonalGaussianSampler as DmvnSampler

        self.assertIs(DiagonalGaussianSampler, DmvnSampler)

    def test_distribution_sampler_still_exported(self):
        from pysp.stats.compute.pdist import DistributionSampler as PdistSampler

        self.assertIs(DistributionSampler, PdistSampler)

    def test_select_exports(self):
        import pysp.stats as stats

        self.assertTrue(hasattr(stats, "SelectDistribution"))
        self.assertTrue(hasattr(stats, "SelectEstimator"))
        self.assertIn("SelectDistribution", stats.__all__)
        self.assertIn("SelectEstimator", stats.__all__)
        self.assertIn("DiagonalGaussianSampler", stats.__all__)

    def test_all_names_resolve(self):
        import pysp.stats as stats

        missing = [name for name in stats.__all__ if not hasattr(stats, name)]
        self.assertEqual(missing, [])


class DataSequenceEncoderStrTestCase(unittest.TestCase):
    def test_base_str_no_recursion(self):
        self.assertEqual(str(DataSequenceEncoder()), "DataSequenceEncoder")

    def test_subclass_default_str_uses_class_name(self):
        class DummyEncoder(DataSequenceEncoder):
            def __eq__(self, other):
                return isinstance(other, DummyEncoder)

        self.assertEqual(str(DummyEncoder()), "DummyEncoder")


if __name__ == "__main__":
    unittest.main()
