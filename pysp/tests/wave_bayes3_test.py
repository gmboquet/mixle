"""Regression tests for the wave-3 fixes, migrated to pysp.stats.

Covers:
  - mixture.py: seq_expected_log_density broadcast checked against per-observation
    expected_log_density on a 2-component model,
  - sequence.py: SequenceEstimator.set_prior threads to the entry/length estimators and the real
    SequenceAccumulator(Factory),
  - exponential.py: expected_log_density/seq_expected_log_density fall back to (seq_)log_density
    when no conjugate prior is set,
  - poisson.py: expected_log_density falls back to log_density without a conjugate prior, and the
    accumulator update/value round-trip is consistent across the scalar and seq paths,
  - setdist.py: estimate() preserves name/prior; real accumulator factory,
  - composite.py: get_prior/set_prior symmetry with a count check.

The legacy estimator API was estimate(suff_stat); the stats API is estimate(nobs, suff_stat).
"""

import unittest

import numpy as np

from pysp.stats.beta import BetaDistribution
from pysp.stats.composite import (
    CompositeAccumulatorFactory,
    CompositeDistribution,
    CompositeEstimator,
)
from pysp.stats.exponential import ExponentialDistribution, ExponentialEstimator
from pysp.stats.gamma import GammaDistribution
from pysp.stats.mixture import MixtureDistribution
from pysp.stats.poisson import (
    PoissonAccumulator,
    PoissonDistribution,
    PoissonEstimator,
)
from pysp.stats.sequence import (
    SequenceAccumulator,
    SequenceAccumulatorFactory,
    SequenceEstimator,
)
from pysp.stats.setdist import (
    BernoulliSetAccumulator,
    BernoulliSetAccumulatorFactory,
    BernoulliSetEstimator,
)


class MixtureSeqExpectedLogDensityTestCase(unittest.TestCase):
    """seq_expected_log_density must match expected_log_density per item."""

    def test_two_component_model_more_data_than_components(self):
        m = MixtureDistribution([PoissonDistribution(2.0), PoissonDistribution(9.0)], [0.4, 0.6])

        data = [0, 1, 3, 7, 12]
        enc = m.dist_to_encoder().seq_encode(data)

        seq_vals = m.seq_expected_log_density(enc)
        item_vals = np.asarray([m.expected_log_density(x) for x in data])

        self.assertEqual(seq_vals.shape, (len(data),))
        self.assertTrue(np.allclose(seq_vals, item_vals))

    def test_seq_expected_log_density_without_conjugate_weight_prior(self):
        # No conjugate weight prior -> the expected log density degenerates to the plug-in path.
        m = MixtureDistribution([PoissonDistribution(2.0), PoissonDistribution(9.0)], [0.4, 0.6])

        data = [0, 2, 5]
        enc = m.dist_to_encoder().seq_encode(data)
        seq_vals = m.seq_expected_log_density(enc)
        item_vals = np.asarray([m.expected_log_density(x) for x in data])

        self.assertEqual(seq_vals.shape, (len(data),))
        self.assertTrue(np.allclose(seq_vals, item_vals))


class SequenceEstimatorPriorTestCase(unittest.TestCase):
    """set_prior must thread to the entry/length estimators (not a fixed dist)."""

    def test_set_prior_round_trip(self):
        est = SequenceEstimator(PoissonEstimator(), PoissonEstimator())

        entry_prior = GammaDistribution(2.0, 3.0)
        len_prior = GammaDistribution(4.0, 5.0)
        est.set_prior((entry_prior, len_prior))

        self.assertIs(est.estimator.get_prior(), entry_prior)
        self.assertIs(est.len_estimator.get_prior(), len_prior)

        rt_entry, rt_len = est.get_prior()
        self.assertIs(rt_entry, entry_prior)
        self.assertIs(rt_len, len_prior)

    def test_accumulator_factory_is_real_class(self):
        est = SequenceEstimator(PoissonEstimator(), PoissonEstimator())
        factory = est.accumulator_factory()
        self.assertIsInstance(factory, SequenceAccumulatorFactory)

        acc = factory.make()
        self.assertIsInstance(acc, SequenceAccumulator)

        data = [[1, 2], [3], [0, 1, 4]]
        for x in data:
            acc.update(x, 1.0, None)

        entry_stats, len_stats = acc.value()
        self.assertAlmostEqual(entry_stats[0], 6.0)  # six entries
        self.assertAlmostEqual(entry_stats[1], 11.0)  # sum of entries
        self.assertAlmostEqual(len_stats[0], 3.0)  # three sequences
        self.assertAlmostEqual(len_stats[1], 6.0)  # sum of lengths

        d = est.estimate(None, acc.value())
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
    """Non-conjugate / absent priors must fall back to (seq_)log_density."""

    def test_conjugate_prior_unchanged(self):
        d = ExponentialDistribution(1.5, prior=GammaDistribution(2.0, 3.0))
        self.assertIsNotNone(d.expected_nparams)
        self.assertTrue(np.isfinite(d.expected_log_density(0.7)))

    def test_non_conjugate_scalar_fallback(self):
        d = ExponentialDistribution(1.5)
        self.assertIsNone(d.expected_nparams)
        for x in [0.5, 2.0]:
            self.assertAlmostEqual(d.expected_log_density(x), d.log_density(x))

    def test_non_conjugate_seq_fallback(self):
        d = ExponentialDistribution(0.8)
        enc = d.dist_to_encoder().seq_encode([0.1, 1.0, 2.5])
        self.assertTrue(np.allclose(d.seq_expected_log_density(enc), d.seq_log_density(enc)))


class PoissonFixesTestCase(unittest.TestCase):
    def test_expected_log_density_non_conjugate_fallback(self):
        d = PoissonDistribution(2.5)
        self.assertFalse(d.has_conj_prior)
        for x in [0, 1, 4]:
            self.assertAlmostEqual(d.expected_log_density(x), d.log_density(x))

        enc = d.dist_to_encoder().seq_encode([0, 1, 4])
        self.assertTrue(np.allclose(d.seq_expected_log_density(enc), d.seq_log_density(enc)))

    def test_accumulator_update_value_round_trip(self):
        data = [1, 0, 3, 2, 4]
        weights = [1.0, 0.5, 2.0, 1.0, 0.25]

        acc = PoissonAccumulator()
        for x, w in zip(data, weights):
            acc.update(x, w, None)

        count, psum = acc.value()
        self.assertAlmostEqual(count, sum(weights))
        self.assertAlmostEqual(psum, sum(x * w for x, w in zip(data, weights)))

        # seq path must produce the same sufficient statistics
        d = PoissonDistribution(2.0)
        acc2 = PoissonAccumulator()
        acc2.seq_update(d.dist_to_encoder().seq_encode(data), np.asarray(weights), None)
        self.assertTrue(np.allclose(acc2.value(), acc.value()))

        # combine/from_value agree with value()
        acc3 = PoissonAccumulator()
        acc3.from_value(acc.value())
        acc3.combine(acc2.value())
        self.assertTrue(np.allclose(acc3.value(), (2 * count, 2 * psum)))

        # the suff stat feeds estimate() directly (conjugate Gamma posterior mode)
        est = PoissonEstimator(prior=GammaDistribution(2.0, 0.5))
        fitted = est.estimate(None, acc.value())
        k_n = 2.0 + psum
        theta_n = 0.5 / (count * 0.5 + 1.0)
        self.assertAlmostEqual(fitted.lam, (k_n - 1.0) * theta_n, places=10)


class BernoulliSetEstimateTestCase(unittest.TestCase):
    """estimate() must carry the estimator's name and prior."""

    def test_estimate_preserves_name_and_prior(self):
        prior = BetaDistribution(2.0, 3.0)
        est = BernoulliSetEstimator(name="tags", prior=prior)

        factory = est.accumulator_factory()
        self.assertIsInstance(factory, BernoulliSetAccumulatorFactory)

        acc = factory.make()
        self.assertIsInstance(acc, BernoulliSetAccumulator)
        for x in [["a", "b"], ["a"], ["b", "c"], []]:
            acc.update(x, 1.0, None)

        d = est.estimate(None, acc.value())
        self.assertEqual(d.name, "tags")
        self.assertIs(d.prior, prior)
        for k in "abc":
            self.assertIn(k, d.pmap)

    def test_estimate_preserves_name_without_conjugacy(self):
        # No conjugate prior -> plain inclusion-frequency estimate; no posterior carried.
        est = BernoulliSetEstimator(name="s")
        acc = est.accumulator_factory().make()
        acc.update(["a"], 1.0, None)
        acc.update([], 1.0, None)

        d = est.estimate(None, acc.value())
        self.assertEqual(d.name, "s")
        self.assertIsNone(d.prior)
        self.assertAlmostEqual(d.pmap["a"], 0.5)


class CompositePriorSymmetryTestCase(unittest.TestCase):
    """get_prior/set_prior must round-trip and reject mismatched counts."""

    def test_distribution_prior_round_trip(self):
        cd = CompositeDistribution((PoissonDistribution(2.0), ExponentialDistribution(1.5)))

        prior = cd.get_prior()
        self.assertEqual(len(prior), 2)

        g1 = GammaDistribution(2.0, 3.0)
        g2 = GammaDistribution(4.0, 5.0)
        cd.set_prior([g1, g2])
        self.assertIs(cd.dists[0].get_prior(), g1)
        self.assertIs(cd.dists[1].get_prior(), g2)

        rt = cd.get_prior()
        self.assertIs(rt[0], g1)
        self.assertIs(rt[1], g2)

    def test_distribution_set_prior_count_mismatch_raises(self):
        cd = CompositeDistribution((PoissonDistribution(2.0), ExponentialDistribution(1.5)))
        with self.assertRaises(ValueError):
            cd.set_prior([GammaDistribution(2.0, 3.0)])

    def test_estimator_prior_round_trip_and_mismatch(self):
        ce = CompositeEstimator([PoissonEstimator(), ExponentialEstimator()])

        g1 = GammaDistribution(2.0, 3.0)
        g2 = GammaDistribution(4.0, 5.0)
        ce.set_prior([g1, g2])
        self.assertIs(ce.estimators[0].get_prior(), g1)
        self.assertIs(ce.estimators[1].get_prior(), g2)

        with self.assertRaises(ValueError):
            ce.set_prior([GammaDistribution(2.0, 3.0)])

    def test_estimator_factory_and_estimate(self):
        ce = CompositeEstimator([PoissonEstimator(), ExponentialEstimator()])
        factory = ce.accumulator_factory()
        self.assertIsInstance(factory, CompositeAccumulatorFactory)

        acc = factory.make()
        rng = np.random.RandomState(7)
        data = [(int(rng.poisson(3.0)), float(rng.exponential(0.5))) for _ in range(50)]
        for x in data:
            acc.update(x, 1.0, None)

        d = ce.estimate(None, acc.value())
        self.assertIsInstance(d, CompositeDistribution)
        self.assertEqual(d.count, 2)
        self.assertTrue(np.isfinite(d.log_density(data[0])))


if __name__ == "__main__":
    unittest.main()
