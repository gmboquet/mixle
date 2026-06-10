"""Tests for the latent-structure modules lda, jmixture, and hmixture.

Covers scalar/vectorized estimation parity (Accumulator.update vs seq_update must produce
identical sufficient statistics) and the smart enumeration contract for the mixture-like
models (jmixture, hmixture), plus the fail-fast EnumerationError for LDA.
"""
import itertools
import unittest

import numpy as np

from pysp.stats import *
from pysp.stats.hmixture import HierarchicalMixtureDistribution
from pysp.stats.jmixture import JointMixtureDistribution
from pysp.stats.lda import LDADistribution
from pysp.stats.pdist import EnumerationError
from pysp.utils.enumeration import freeze

TOL = 1e-9


def assert_stats_close(test, a, b, path='', rtol=1e-7, atol=1e-9):
    """Recursively compare two (possibly nested) sufficient statistic values."""
    if isinstance(a, (tuple, list)):
        test.assertEqual(len(a), len(b), 'length mismatch at %s' % path)
        for i, (u, v) in enumerate(zip(a, b)):
            assert_stats_close(test, u, v, path + '[%d]' % i, rtol, atol)
    elif isinstance(a, dict):
        test.assertEqual(set(a.keys()), set(b.keys()), 'key mismatch at %s' % path)
        for k in a:
            assert_stats_close(test, a[k], b[k], path + '[%r]' % k, rtol, atol)
    elif a is None:
        test.assertIsNone(b, 'None mismatch at %s' % path)
    else:
        np.testing.assert_allclose(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64),
                                   rtol=rtol, atol=atol, err_msg='value mismatch at %s' % path)


def scalar_vs_seq(dist, data):
    """Returns (scalar_value, seq_value) accumulator suff-stats for the same data and estimate."""
    factory = dist.estimator().accumulator_factory()

    acc_scalar = factory.make()
    for x in data:
        acc_scalar.update(x, 1.0, dist)

    acc_seq = factory.make()
    enc = dist.dist_to_encoder().seq_encode(data)
    acc_seq.seq_update(enc, np.ones(len(data)), dist)

    return acc_scalar.value(), acc_seq.value()


def check_enumeration_contract(test, dist, brute_values, name):
    """Check ordering, exact re-scoring, uniqueness, and brute-force completeness for a finite support."""
    with np.errstate(divide='ignore'):
        items = list(dist.enumerator())

        # Non-increasing log_prob order.
        lps = [lp for _, lp in items]
        for i in range(len(lps) - 1):
            test.assertGreaterEqual(lps[i], lps[i + 1] - TOL, '%s: order violated at %d' % (name, i))

        # log_prob equals dist.log_density(value).
        for v, lp in items:
            test.assertAlmostEqual(lp, dist.log_density(v), delta=TOL, msg='%s: lp mismatch at %r' % (name, v))

        # Exact dedup.
        keys = [freeze(v) for v, _ in items]
        test.assertEqual(len(keys), len(set(keys)), '%s: duplicate values yielded' % name)

        # Completeness against brute force and total mass.
        brute = {}
        for v in brute_values:
            lp = dist.log_density(v)
            if lp > -np.inf:
                brute[freeze(v)] = lp

        test.assertEqual(len(items), len(brute), '%s: wrong support size' % name)
        for v, lp in items:
            test.assertAlmostEqual(lp, brute[freeze(v)], delta=TOL, msg='%s: brute mismatch at %r' % (name, v))

        total = np.logaddexp.reduce(lps)
        test.assertAlmostEqual(total, 0.0, delta=1e-8, msg='%s: total mass != 1' % name)


class LDAUpdateTestCase(unittest.TestCase):

    def setUp(self):
        self.dist = LDADistribution(
            topics=[IntegerCategoricalDistribution(0, [0.7, 0.2, 0.1]),
                    IntegerCategoricalDistribution(0, [0.1, 0.3, 0.6])],
            alpha=[0.7, 1.3])
        self.docs = [
            [(0, 2.0), (1, 1.0)],
            [(2, 3.0)],
            [(0, 1.0), (2, 1.0)],
            [(1, 2.0), (2, 2.0), (0, 1.0)],
            [(1, 1.0)],
        ]

    def test_update_matches_seq_update(self):
        sv, qv = scalar_vs_seq(self.dist, self.docs)
        assert_stats_close(self, sv, qv)

    def test_update_accumulates(self):
        factory = self.dist.estimator().accumulator_factory()
        acc = factory.make()
        acc.update(self.docs[0], 1.0, self.dist)
        self.assertEqual(acc.value()[2], 1.0)  # doc_counts
        self.assertAlmostEqual(np.sum(acc.value()[3]), 3.0, delta=1e-9)  # topic counts = total word count

    def test_enumerator_fails_fast(self):
        with self.assertRaises(EnumerationError) as ctx:
            self.dist.enumerator()
        self.assertIn('variational', str(ctx.exception))


class JointMixtureUpdateTestCase(unittest.TestCase):

    def setUp(self):
        self.dist = JointMixtureDistribution(
            components1=[GaussianDistribution(0.0, 1.0), GaussianDistribution(3.0, 2.0)],
            components2=[GaussianDistribution(-1.0, 1.0), GaussianDistribution(2.0, 0.5)],
            w1=[0.6, 0.4], w2=[0.5, 0.5],
            taus12=[[0.7, 0.3], [0.2, 0.8]],
            taus21=[[0.7, 0.2], [0.3, 0.8]])
        self.data = self.dist.sampler(seed=7).sample(20)

    def test_update_matches_seq_update(self):
        sv, qv = scalar_vs_seq(self.dist, self.data)
        assert_stats_close(self, sv, qv)

    def test_update_accumulates(self):
        factory = self.dist.estimator().accumulator_factory()
        acc = factory.make()
        for x in self.data[:3]:
            acc.update(x, 1.0, self.dist)
        cc1, cc2, jc, _, _ = acc.value()
        self.assertAlmostEqual(np.sum(cc1), 3.0, delta=1e-9)
        self.assertAlmostEqual(np.sum(cc2), 3.0, delta=1e-9)
        self.assertAlmostEqual(np.sum(jc), 3.0, delta=1e-9)


class JointMixtureEnumeratorTestCase(unittest.TestCase):

    def setUp(self):
        self.dist = JointMixtureDistribution(
            components1=[CategoricalDistribution({'a': 0.7, 'b': 0.3}),
                         CategoricalDistribution({'b': 0.5, 'c': 0.5})],
            components2=[CategoricalDistribution({0: 0.6, 1: 0.4}),
                         CategoricalDistribution({1: 0.2, 2: 0.8})],
            w1=[0.6, 0.4], w2=[0.5, 0.5],
            taus12=[[0.7, 0.3], [0.2, 0.8]],
            taus21=[[0.7, 0.2], [0.3, 0.8]])

    def test_enumeration_contract(self):
        brute = [(x0, x1) for x0 in ['a', 'b', 'c'] for x1 in [0, 1, 2]]
        check_enumeration_contract(self, self.dist, brute, 'jmixture')

    def test_zero_weight_pairs_skipped(self):
        # A zero-weight X1 component (with a continuous distribution) is never asked to enumerate.
        dist = JointMixtureDistribution(
            components1=[CategoricalDistribution({1.0: 0.7, 2.0: 0.3}), GaussianDistribution(0.0, 1.0)],
            components2=[CategoricalDistribution({0: 0.6, 1: 0.4}),
                         CategoricalDistribution({1: 0.2, 2: 0.8})],
            w1=[1.0, 0.0], w2=[0.5, 0.5],
            taus12=[[0.7, 0.3], [0.2, 0.8]],
            taus21=[[0.7, 0.2], [0.3, 0.8]])
        with np.errstate(divide='ignore'):
            items = list(dist.enumerator())
            self.assertEqual(len(items), 6)  # {1.0, 2.0} x {0, 1, 2}
            for v, lp in items:
                self.assertAlmostEqual(lp, dist.log_density(v), delta=TOL)

    def test_continuous_components_fail_fast(self):
        dist = JointMixtureDistribution(
            components1=[GaussianDistribution(0.0, 1.0), GaussianDistribution(3.0, 2.0)],
            components2=[CategoricalDistribution({0: 0.6, 1: 0.4}),
                         CategoricalDistribution({1: 0.2, 2: 0.8})],
            w1=[0.6, 0.4], w2=[0.5, 0.5],
            taus12=[[0.7, 0.3], [0.2, 0.8]],
            taus21=[[0.7, 0.2], [0.3, 0.8]])
        with self.assertRaises(EnumerationError) as ctx:
            dist.enumerator()
        self.assertIn('components1[0]', str(ctx.exception))


class HierarchicalMixtureUpdateTestCase(unittest.TestCase):

    def setUp(self):
        self.dist = HierarchicalMixtureDistribution(
            topics=[GaussianDistribution(0.0, 1.0), GaussianDistribution(4.0, 2.0)],
            mixture_weights=[0.5, 0.5],
            topic_weights=[[0.8, 0.2], [0.3, 0.7]],
            len_dist=IntegerCategoricalDistribution(1, [0.3, 0.5, 0.2]))
        self.data = self.dist.sampler(seed=11).sample(20)

    def test_update_matches_seq_update(self):
        sv, qv = scalar_vs_seq(self.dist, self.data)
        assert_stats_close(self, sv, qv)

    def test_update_accumulates(self):
        factory = self.dist.estimator().accumulator_factory()
        acc = factory.make()
        for x in self.data[:3]:
            acc.update(x, 1.0, self.dist)
        comp_counts = acc.value()[0]
        n_values = sum(len(x) for x in self.data[:3])
        self.assertAlmostEqual(np.sum(comp_counts), float(n_values), delta=1e-9)


class HierarchicalMixtureEnumeratorTestCase(unittest.TestCase):

    def setUp(self):
        self.dist = HierarchicalMixtureDistribution(
            topics=[CategoricalDistribution({'a': 0.7, 'b': 0.3}),
                    CategoricalDistribution({'b': 0.4, 'c': 0.6})],
            mixture_weights=[0.6, 0.4],
            topic_weights=[[0.8, 0.2], [0.3, 0.7]],
            len_dist=IntegerCategoricalDistribution(1, [0.6, 0.4]))

    def test_enumeration_contract(self):
        brute = []
        for length in (1, 2):
            brute.extend([list(seq) for seq in itertools.product('abc', repeat=length)])
        check_enumeration_contract(self, self.dist, brute, 'hmixture')

    def test_no_length_distribution_fails_fast(self):
        dist = HierarchicalMixtureDistribution(
            topics=[CategoricalDistribution({'a': 1.0})],
            mixture_weights=[1.0],
            topic_weights=[[1.0]])
        with self.assertRaises(EnumerationError) as ctx:
            dist.enumerator()
        self.assertIn('component[0]', str(ctx.exception))

    def test_top_k_prefix(self):
        with np.errstate(divide='ignore'):
            full = list(self.dist.enumerator())
            top3 = self.dist.enumerator().top_k(3)
            for (v1, lp1), (v2, lp2) in zip(top3, full[:3]):
                self.assertEqual(freeze(v1), freeze(v2))
                self.assertAlmostEqual(lp1, lp2, delta=TOL)


if __name__ == '__main__':
    unittest.main()
