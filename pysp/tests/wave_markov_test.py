"""Tests for pysp.stats.grammar, pysp.stats.markov_transform, and pysp.stats.sparse_markov_transform.

Covers: clean module imports (grammar must import without the optional 'cnrg' package), the guarded
ImportError raised at first cnrg use, the GrammarDistribution.estimator() fix, markov_transform
sample/estimate smoke on tiny data, and DataSequenceEncoder equality / encode round-trip consistency.
"""
import importlib
import importlib.util
import unittest
import warnings

import numpy as np

CNRG_MISSING = importlib.util.find_spec('cnrg') is None


def _make_markov_transform_dist(alpha=0.05, with_len=True):
    from pysp.stats.markov_transform import MarkovTransformDistribution
    from pysp.stats.composite import CompositeDistribution
    from pysp.stats.categorical import CategoricalDistribution

    nw = 3
    init_prob = np.asarray([0.5, 0.3, 0.2])
    rng = np.random.RandomState(7)
    cond_prob = rng.rand(nw * nw, nw) + 0.1
    cond_prob /= cond_prob.sum(axis=1, keepdims=True)

    if with_len:
        len_dist = CompositeDistribution((CategoricalDistribution({2: 0.5, 3: 0.5}),
                                          CategoricalDistribution({2: 0.5, 3: 0.5}),
                                          CategoricalDistribution({3: 0.6, 4: 0.4})))
    else:
        len_dist = None

    return MarkovTransformDistribution(init_prob, cond_prob, alpha=alpha, len_dist=len_dist)


def _make_sparse_assoc_dist(low_memory=False):
    from pysp.stats.sparse_markov_transform import SparseMarkovAssociationDistribution
    from pysp.stats.composite import CompositeDistribution
    from pysp.stats.categorical import CategoricalDistribution

    nw = 3
    init_prob = np.asarray([0.5, 0.3, 0.2])
    rng = np.random.RandomState(11)
    cond_prob = rng.rand(nw, nw) + 0.1
    cond_prob /= cond_prob.sum(axis=1, keepdims=True)

    len_dist = CompositeDistribution((CategoricalDistribution({2: 0.5, 3: 0.5}),
                                      CategoricalDistribution({3: 0.6, 4: 0.4})))

    return SparseMarkovAssociationDistribution(init_prob, cond_prob, alpha=0.1, len_dist=len_dist,
                                               low_memory=low_memory)


class ImportTestCase(unittest.TestCase):

    def test_markov_transform_imports(self):
        mod = importlib.import_module('pysp.stats.markov_transform')
        for name in ('MarkovTransformDistribution', 'MarkovTransformSampler', 'MarkovTransformAccumulator',
                     'MarkovTransformAccumulatorFactory', 'MarkovTransformEstimator',
                     'MarkovTransformDataEncoder'):
            self.assertTrue(hasattr(mod, name), name)

    def test_sparse_markov_transform_imports(self):
        mod = importlib.import_module('pysp.stats.sparse_markov_transform')
        for name in ('SparseMarkovAssociationDistribution', 'SparseMarkovAssociationSampler',
                     'SparseMarkovAssociationAccumulator', 'SparseMarkovAssociationAccumulatorFactory',
                     'SparseMarkovAssociationEstimator', 'SparseMarkovAssociationDataEncoder'):
            self.assertTrue(hasattr(mod, name), name)

    def test_grammar_imports_without_cnrg(self):
        # The module must import cleanly even when the optional 'cnrg' package is absent.
        mod = importlib.import_module('pysp.stats.grammar')
        for name in ('GrammarDistribution', 'GrammarSampler', 'GrammarEstimatorAccumulator',
                     'GrammarAccumulatorFactory', 'GrammarEstimator', 'GrammarDataEncoder'):
            self.assertTrue(hasattr(mod, name), name)


class GrammarTestCase(unittest.TestCase):

    def test_estimator_does_not_pass_distribution_as_pseudo_count(self):
        from pysp.stats.grammar import GrammarDistribution, GrammarEstimator

        dist = GrammarDistribution(None, 0.01, name='g')
        est = dist.estimator()
        self.assertIsInstance(est, GrammarEstimator)
        self.assertIsNone(est.pseudo_count)
        self.assertEqual(est.name, 'g')

        est2 = dist.estimator(pseudo_count=2.0)
        self.assertEqual(est2.pseudo_count, 2.0)

    def test_accumulator_factory_and_alias(self):
        from pysp.stats.grammar import GrammarEstimator, GrammarAccumulatorFactory

        est = GrammarEstimator()
        self.assertIsInstance(est.accumulator_factory(), GrammarAccumulatorFactory)
        self.assertIsInstance(est.accumulatorFactory(), GrammarAccumulatorFactory)

    def test_encoder_equality(self):
        from pysp.stats.grammar import GrammarDistribution, GrammarDataEncoder

        dist = GrammarDistribution(None, 0.01)
        enc = dist.dist_to_encoder()
        self.assertEqual(enc, GrammarDataEncoder())
        self.assertNotEqual(enc, object())
        data = ['a', 'b']
        self.assertEqual(enc.seq_encode(data), data)
        self.assertEqual(str(enc), 'GrammarDataEncoder')

    @unittest.skipUnless(CNRG_MISSING, "cnrg is installed; guarded ImportError not expected")
    def test_cnrg_required_at_first_use(self):
        from pysp.stats.grammar import GrammarEstimatorAccumulator, GrammarEstimator, GrammarSampler

        with self.assertRaisesRegex(ImportError, 'cnrg'):
            GrammarEstimatorAccumulator()
        with self.assertRaisesRegex(ImportError, 'cnrg'):
            GrammarEstimator().accumulator_factory().make()
        with self.assertRaisesRegex(ImportError, 'cnrg'):
            GrammarSampler(None).sample()
        with self.assertRaisesRegex(ImportError, 'cnrg'):
            GrammarSampler(None).sample_seq([10])


class MarkovTransformTestCase(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore')
        self.dist = _make_markov_transform_dist(with_len=True)
        self.data = self.dist.sampler(seed=11).sample(size=25)

    def test_sample_structure(self):
        single = self.dist.sampler(seed=3).sample()
        self.assertEqual(len(single), 3)
        for part in single:
            for v, c in part:
                self.assertTrue(0 <= int(v) < self.dist.num_vals)
                self.assertGreater(c, 0)
        self.assertEqual(len(self.data), 25)

    def test_seq_log_density_matches_log_density(self):
        enc = self.dist.dist_to_encoder()
        ex = enc.seq_encode(self.data)
        seq_ll = self.dist.seq_log_density(ex)
        single_ll = np.asarray([self.dist.log_density(u) for u in self.data])
        self.assertTrue(np.all(np.isfinite(seq_ll)))
        self.assertTrue(np.allclose(seq_ll, single_ll))

    def test_legacy_seq_encode_matches_encoder(self):
        # Without a length distribution the legacy distribution method and the encoder must agree.
        dist = _make_markov_transform_dist(with_len=False)
        legacy = dist.seq_encode(self.data)
        modern = dist.dist_to_encoder().seq_encode(self.data)
        self.assertIsNone(legacy[1])
        self.assertIsNone(modern[1])
        self.assertTrue(np.allclose(dist.seq_log_density(legacy), dist.seq_log_density(modern)))

    def test_encoder_equality(self):
        from pysp.stats.markov_transform import MarkovTransformDataEncoder

        enc1 = self.dist.dist_to_encoder()
        enc2 = self.dist.dist_to_encoder()
        self.assertEqual(enc1, enc2)
        self.assertNotEqual(enc1, MarkovTransformDataEncoder(len_encoder=None))
        self.assertNotEqual(enc1, object())
        self.assertIn('MarkovTransformDataEncoder', str(enc1))

    def test_estimate_smoke(self):
        from pysp.stats.markov_transform import (MarkovTransformEstimator, MarkovTransformAccumulatorFactory,
                                                 MarkovTransformDistribution)

        dist = _make_markov_transform_dist(with_len=False)
        enc = dist.dist_to_encoder()
        ex = enc.seq_encode(self.data)
        weights = np.ones(len(self.data))

        est = MarkovTransformEstimator(dist.num_vals, alpha=0.05)
        self.assertIsInstance(est.accumulator_factory(), MarkovTransformAccumulatorFactory)
        self.assertIsInstance(est.accumulatorFactory(), MarkovTransformAccumulatorFactory)

        acc = est.accumulator_factory().make()
        acc.seq_initialize(ex, weights, np.random.RandomState(5))
        model0 = est.estimate(None, acc.value())
        self.assertIsInstance(model0, MarkovTransformDistribution)
        self.assertAlmostEqual(float(np.sum(model0.init_prob_vec)), 1.0, places=8)

        acc2 = est.accumulator_factory().make()
        acc2.seq_update(ex, weights, model0)
        model1 = est.estimate(None, acc2.value())
        self.assertTrue(np.all(np.isfinite(model1.seq_log_density(ex))))

        # accumulator encoders must match the distribution encoder
        self.assertEqual(acc.acc_to_encoder(), enc)

    def test_update_and_initialize_single_obs(self):
        dist = _make_markov_transform_dist(with_len=False)
        from pysp.stats.markov_transform import MarkovTransformEstimator

        est = MarkovTransformEstimator(dist.num_vals, alpha=0.05)
        acc = est.accumulator_factory().make()
        acc.initialize(self.data[0], 1.0, np.random.RandomState(2))
        acc.update(self.data[1], 1.0, dist)
        init_count, trans_count, size_val = acc.value()
        self.assertGreater(float(np.sum(init_count)), 0.0)
        self.assertGreater(trans_count.sum(), 0.0)
        self.assertIsNone(size_val)

    def test_str_uses_class_name(self):
        self.assertTrue(str(self.dist).startswith('MarkovTransformDistribution('))


class SparseMarkovAssociationTestCase(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore')
        self.dist = _make_sparse_assoc_dist(low_memory=False)
        self.data = self.dist.sampler(seed=21).sample(size=15)

    def test_seq_log_density_matches_log_density(self):
        enc = self.dist.dist_to_encoder()
        ex = enc.seq_encode(self.data)
        seq_ll = self.dist.seq_log_density(ex)
        single_ll = np.asarray([self.dist.log_density(u) for u in self.data])
        self.assertTrue(np.all(np.isfinite(seq_ll)))
        self.assertTrue(np.allclose(seq_ll, single_ll))

    def test_low_memory_encoding_agrees(self):
        dist_lm = _make_sparse_assoc_dist(low_memory=True)
        enc_lm = dist_lm.dist_to_encoder()
        ex_lm = enc_lm.seq_encode(self.data)
        self.assertIsNone(ex_lm[3])
        seq_ll_lm = dist_lm.seq_log_density(ex_lm)

        ex = self.dist.dist_to_encoder().seq_encode(self.data)
        seq_ll = self.dist.seq_log_density(ex)
        self.assertTrue(np.allclose(seq_ll, seq_ll_lm))

    def test_encoder_equality(self):
        enc1 = self.dist.dist_to_encoder()
        enc2 = self.dist.dist_to_encoder()
        self.assertEqual(enc1, enc2)
        enc_lm = _make_sparse_assoc_dist(low_memory=True).dist_to_encoder()
        self.assertNotEqual(enc1, enc_lm)
        self.assertNotEqual(enc1, object())
        self.assertIn('SparseMarkovAssociationDataEncoder', str(enc1))


if __name__ == '__main__':
    unittest.main()
