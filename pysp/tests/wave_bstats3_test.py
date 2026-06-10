"""Regression tests for the wave-3 pysp.bstats fixes.

Covers:
  - mixture.py: seq_expected_log_density broadcast fix (keepdims) checked
    against per-observation expected_log_density on a 2-component model,
  - categorical.py: df_update groupby-indices iteration fix,
  - sequence.py: SequenceEstimator.set_prior attribute fix and the real
    SequenceEstimatorAccumulatorFactory,
  - exponential.py: expected_log_density/seq_expected_log_density fall back
    to (seq_)log_density when no conjugate prior is set,
  - poisson.py: cross_entropy returns a value, expected_log_density falls
    back to log_density without a conjugate prior, and the accumulator
    update/value round-trip is consistent across the scalar and seq paths,
  - setdist.py: estimate() preserves name/prior; real accumulator factory,
  - composite.py: get_prior/set_prior symmetry with a count check.
"""
import unittest

import numpy as np
import pandas as pd

from pysp.bstats.categorical import CategoricalEstimatorAccumulator
from pysp.bstats.composite import (
    CompositeAccumulatorFactory, CompositeDistribution, CompositeEstimator,
)
from pysp.bstats.exponential import ExponentialDistribution, ExponentialEstimator
from pysp.bstats.gamma import GammaDistribution
from pysp.bstats.beta import BetaDistribution
from pysp.bstats.mixture import MixtureDistribution
from pysp.bstats.nulldist import null_dist
from pysp.bstats.poisson import (
    PoissonDistribution, PoissonEstimator, PoissonEstimatorAccumulator,
)
from pysp.bstats.sequence import (
    SequenceEstimator, SequenceEstimatorAccumulator,
    SequenceEstimatorAccumulatorFactory,
)
from pysp.bstats.setdist import (
    BernoulliSetAccumulator, BernoulliSetAccumulatorFactory,
    BernoulliSetEstimator,
)


class MixtureSeqExpectedLogDensityTestCase(unittest.TestCase):
    """seq_expected_log_density must match expected_log_density per item."""

    def test_two_component_model_more_data_than_components(self):
        m = MixtureDistribution([PoissonDistribution(2.0), PoissonDistribution(9.0)], [0.4, 0.6])

        data = [0, 1, 3, 7, 12]
        enc = m.seq_encode(data)

        seq_vals = m.seq_expected_log_density(enc)
        item_vals = np.asarray([m.expected_log_density(x) for x in data])

        self.assertEqual(seq_vals.shape, (len(data),))
        self.assertTrue(np.allclose(seq_vals, item_vals))

    def test_seq_expected_log_density_without_conjugate_weight_prior(self):
        m = MixtureDistribution([PoissonDistribution(2.0), PoissonDistribution(9.0)],
                                [0.4, 0.6], prior=null_dist)

        data = [0, 2, 5]
        enc = m.seq_encode(data)
        seq_vals = m.seq_expected_log_density(enc)
        item_vals = np.asarray([m.expected_log_density(x) for x in data])

        self.assertEqual(seq_vals.shape, (len(data),))
        self.assertTrue(np.allclose(seq_vals, item_vals))


class CategoricalDataFrameUpdateTestCase(unittest.TestCase):
    """df_update must aggregate weighted counts per category."""

    def test_df_update_counts(self):
        df = pd.DataFrame({'col': ['a', 'b', 'a', 'c', 'b']})
        weights = [1.0, 2.0, 3.0, 4.0, 5.0]

        acc = CategoricalEstimatorAccumulator('col', (None,))
        acc.df_update(df, weights, None)

        self.assertAlmostEqual(acc.count_map['a'], 4.0)
        self.assertAlmostEqual(acc.count_map['b'], 7.0)
        self.assertAlmostEqual(acc.count_map['c'], 4.0)
        self.assertAlmostEqual(acc.count_sum, 15.0)

    def test_df_initialize_delegates(self):
        df = pd.DataFrame({'col': ['x', 'x', 'y']})
        acc = CategoricalEstimatorAccumulator('col', (None,))
        acc.df_initialize(df, np.ones(3), None)

        self.assertAlmostEqual(acc.count_map['x'], 2.0)
        self.assertAlmostEqual(acc.count_map['y'], 1.0)


class SequenceEstimatorPriorTestCase(unittest.TestCase):
    """set_prior must update the entry estimator (was self.dist)."""

    def test_set_prior_round_trip(self):
        est = SequenceEstimator(PoissonEstimator(), PoissonEstimator())

        new_prior = CompositeDistribution((GammaDistribution(2.0, 3.0), GammaDistribution(4.0, 5.0)))
        est.set_prior(new_prior)

        self.assertIs(est.estimator.get_prior(), new_prior.dists[0])
        self.assertIs(est.len_estimator.get_prior(), new_prior.dists[1])

        rt = est.get_prior()
        self.assertIs(rt.dists[0], new_prior.dists[0])
        self.assertIs(rt.dists[1], new_prior.dists[1])

    def test_accumulator_factory_is_real_class(self):
        est = SequenceEstimator(PoissonEstimator(), PoissonEstimator())
        factory = est.accumulator_factory()
        self.assertIsInstance(factory, SequenceEstimatorAccumulatorFactory)

        acc = factory.make()
        self.assertIsInstance(acc, SequenceEstimatorAccumulator)

        data = [[1, 2], [3], [0, 1, 4]]
        for x in data:
            acc.update(x, 1.0, None)

        entry_stats, len_stats = acc.value()
        self.assertAlmostEqual(entry_stats[0], 6.0)   # six entries
        self.assertAlmostEqual(entry_stats[1], 11.0)  # sum of entries
        self.assertAlmostEqual(len_stats[0], 3.0)     # three sequences
        self.assertAlmostEqual(len_stats[1], 6.0)     # sum of lengths

        d = est.estimate(acc.value())
        self.assertGreater(d.dist.lam, 0.0)
        self.assertGreater(d.len_dist.lam, 0.0)

    def test_accumulator_factory_without_len_estimator(self):
        est = SequenceEstimator(PoissonEstimator(), len_estimator=None)
        acc = est.accumulator_factory().make()
        acc.update([2, 2], 1.0, None)

        entry_stats, len_stats = acc.value()
        self.assertAlmostEqual(entry_stats[0], 2.0)
        self.assertAlmostEqual(entry_stats[1], 4.0)
        self.assertIsNone(len_stats)


class ExponentialExpectedLogDensityTestCase(unittest.TestCase):
    """Non-conjugate priors must fall back to (seq_)log_density."""

    def test_conjugate_prior_unchanged(self):
        d = ExponentialDistribution(1.5, prior=GammaDistribution(2.0, 3.0))
        self.assertIsNotNone(d.expected_nparams)
        self.assertTrue(np.isfinite(d.expected_log_density(0.7)))

    def test_non_conjugate_scalar_fallback(self):
        d = ExponentialDistribution(1.5, prior=null_dist)
        self.assertIsNone(d.expected_nparams)
        for x in [0.0, 0.5, 2.0]:
            self.assertAlmostEqual(d.expected_log_density(x), d.log_density(x))

    def test_non_conjugate_seq_fallback(self):
        d = ExponentialDistribution(0.8, prior=null_dist)
        enc = d.seq_encode([0.1, 1.0, 2.5])
        self.assertTrue(np.allclose(d.seq_expected_log_density(enc), d.seq_log_density(enc)))


class PoissonFixesTestCase(unittest.TestCase):

    def test_cross_entropy_returns_value(self):
        d1 = PoissonDistribution(3.0)
        d2 = PoissonDistribution(4.5)

        ce = d1.cross_entropy(d2)
        self.assertIsNotNone(ce)
        # bstats sign convention: entropy() is E[log f(X)] (categorical's
        # entropy is sum p*log p), so cross_entropy(p, q) = E_p[log q] and
        # by Gibbs' inequality E_p[log q] <= E_p[log p], equality iff q == p
        self.assertLess(ce, d1.entropy())
        self.assertAlmostEqual(d1.cross_entropy(d1), d1.entropy(), places=10)

    def test_cross_entropy_rejects_other_distributions(self):
        d = PoissonDistribution(3.0)
        with self.assertRaises(NotImplementedError):
            d.cross_entropy(ExponentialDistribution(1.0))

    def test_expected_log_density_non_conjugate_fallback(self):
        d = PoissonDistribution(2.5, prior=null_dist)
        self.assertFalse(d.has_conj_prior)
        for x in [0, 1, 4]:
            self.assertAlmostEqual(d.expected_log_density(x), d.log_density(x))

        enc = d.seq_encode([0, 1, 4])
        self.assertTrue(np.allclose(d.seq_expected_log_density(enc), d.seq_log_density(enc)))

    def test_accumulator_update_value_round_trip(self):
        data = [1, 0, 3, 2, 4]
        weights = [1.0, 0.5, 2.0, 1.0, 0.25]

        acc = PoissonEstimatorAccumulator('p', None)
        for x, w in zip(data, weights):
            acc.update(x, w, None)

        count, psum = acc.value()
        self.assertAlmostEqual(count, sum(weights))
        self.assertAlmostEqual(psum, sum(x*w for x, w in zip(data, weights)))

        # seq path must produce the same sufficient statistics
        d = PoissonDistribution(2.0)
        acc2 = PoissonEstimatorAccumulator('p', None)
        acc2.seq_update(d.seq_encode(data), np.asarray(weights), None)
        self.assertTrue(np.allclose(acc2.value(), acc.value()))

        # combine/from_value agree with value()
        acc3 = PoissonEstimatorAccumulator('p', None)
        acc3.from_value(acc.value())
        acc3.combine(acc2.value())
        self.assertTrue(np.allclose(acc3.value(), (2*count, 2*psum)))

        # the suff stat feeds estimate() directly
        est = PoissonEstimator(prior=GammaDistribution(2.0, 0.5))
        fit = est.estimate(acc.value())
        k_n = 2.0 + psum
        theta_n = 0.5/(count*0.5 + 1.0)
        self.assertAlmostEqual(fit.lam, (k_n - 1.0)*theta_n, places=10)


class BernoulliSetEstimateTestCase(unittest.TestCase):
    """estimate() must carry the estimator's name and prior."""

    def test_estimate_preserves_name_and_prior(self):
        prior = BetaDistribution(2.0, 3.0)
        est = BernoulliSetEstimator(name='tags', prior=prior)

        factory = est.accumulator_factory()
        self.assertIsInstance(factory, BernoulliSetAccumulatorFactory)

        acc = factory.make()
        self.assertIsInstance(acc, BernoulliSetAccumulator)
        for x in [['a', 'b'], ['a'], ['b', 'c'], []]:
            acc.update(x, 1.0, None)

        d = est.estimate(acc.value())
        self.assertEqual(d.name, 'tags')
        self.assertIs(d.prior, prior)
        for k in 'abc':
            self.assertIn(k, d.pmap)

    def test_estimate_preserves_name_and_prior_without_conjugacy(self):
        est = BernoulliSetEstimator(name='s', prior=null_dist)
        acc = est.accumulator_factory().make()
        acc.update(['a'], 1.0, None)
        acc.update([], 1.0, None)

        d = est.estimate(acc.value())
        self.assertEqual(d.name, 's')
        self.assertIs(d.prior, null_dist)
        self.assertAlmostEqual(d.pmap['a'], 0.5)


class CompositePriorSymmetryTestCase(unittest.TestCase):
    """get_prior/set_prior must round-trip and reject mismatched counts."""

    def test_distribution_prior_round_trip(self):
        cd = CompositeDistribution((PoissonDistribution(2.0), ExponentialDistribution(1.5)))

        prior = cd.get_prior()
        self.assertEqual(len(prior.dists), 2)

        new_prior = CompositeDistribution((GammaDistribution(2.0, 3.0), GammaDistribution(4.0, 5.0)))
        cd.set_prior(new_prior)
        self.assertIs(cd.dists[0].get_prior(), new_prior.dists[0])
        self.assertIs(cd.dists[1].get_prior(), new_prior.dists[1])

        rt = cd.get_prior()
        self.assertIs(rt.dists[0], new_prior.dists[0])
        self.assertIs(rt.dists[1], new_prior.dists[1])

    def test_distribution_set_prior_count_mismatch_raises(self):
        cd = CompositeDistribution((PoissonDistribution(2.0), ExponentialDistribution(1.5)))
        with self.assertRaises(ValueError):
            cd.set_prior(CompositeDistribution((GammaDistribution(2.0, 3.0),)))

    def test_estimator_prior_round_trip_and_mismatch(self):
        ce = CompositeEstimator([PoissonEstimator(), ExponentialEstimator()])

        new_prior = CompositeDistribution((GammaDistribution(2.0, 3.0), GammaDistribution(4.0, 5.0)))
        ce.set_prior(new_prior)
        self.assertIs(ce.estimators[0].get_prior(), new_prior.dists[0])
        self.assertIs(ce.estimators[1].get_prior(), new_prior.dists[1])

        with self.assertRaises(ValueError):
            ce.set_prior(CompositeDistribution((GammaDistribution(2.0, 3.0),)))

    def test_estimator_factory_and_estimate(self):
        ce = CompositeEstimator([PoissonEstimator(), ExponentialEstimator()])
        factory = ce.accumulator_factory()
        self.assertIsInstance(factory, CompositeAccumulatorFactory)

        acc = factory.make()
        rng = np.random.RandomState(7)
        data = [(int(rng.poisson(3.0)), float(rng.exponential(0.5))) for _ in range(50)]
        for x in data:
            acc.update(x, 1.0, None)

        d = ce.estimate(acc.value())
        self.assertIsInstance(d, CompositeDistribution)
        self.assertEqual(d.count, 2)
        self.assertTrue(np.isfinite(d.log_density(data[0])))


if __name__ == '__main__':
    unittest.main()
