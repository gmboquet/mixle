"""Tests for the bstats driver/wrapper fixes (wave 4).

Covers:
  - pysp.bstats.estimate / seq_estimate / initialize dispatching to the
    bstats API (accumulator_factory() + estimate(suff_stat)) with a getattr
    fallback to the legacy camelCase / two-argument API. The spark branches
    are exercised at the unit level (dispatch helpers plus source-level
    call-signature checks) rather than by running a SparkContext; the pandas
    branches are run end-to-end with recording estimators.
  - local estimate() still works end-to-end on a small Gaussian mixture,
  - IgnoredEstimator.set_prior accepting a prior argument (protocol fix),
  - OptionalDistribution.get_data_type using the bstats get_data_type
    convention with fallback to legacy get_type.
"""
import inspect
import typing
import unittest

import numpy as np
import pandas as pd

import pysp.bstats as bstats
from pysp.bstats import (
    GaussianDistribution, GaussianEstimator, MixtureDistribution,
    MixtureEstimator, PoissonDistribution, estimate, initialize,
    seq_encode, seq_estimate, seq_log_density_sum,
)
from pysp.bstats.ignored import (
    IgnoredAccumulator, IgnoredDistribution, IgnoredEstimator,
)
from pysp.bstats.optional import OptionalDistribution, OptionalEstimator


class _RecordingAccumulator(object):
    """Accumulator stub recording which driver code paths were used."""

    def __init__(self, log):
        self.log = log
        self.wsum = 0.0

    def update(self, x, weight, estimate=None):
        self.log.append(('update', x))
        self.wsum += weight

    def initialize(self, x, weight, rng):
        self.log.append(('initialize', x))
        self.wsum += weight

    def df_update(self, df, weights, estimate=None):
        self.log.append(('df_update', len(df)))
        self.wsum += float(np.sum(weights))

    def df_initialize(self, df, weights, rng):
        self.log.append(('df_initialize', len(df)))
        self.wsum += float(np.sum(weights))

    def combine(self, suff_stat):
        self.wsum += suff_stat
        return self

    def value(self):
        return self.wsum

    def from_value(self, x):
        self.wsum = x
        return self

    def key_merge(self, stats_dict):
        pass

    def key_replace(self, stats_dict):
        pass


class _RecordingFactory(object):

    def __init__(self, log):
        self.log = log

    def make(self):
        return _RecordingAccumulator(self.log)


class _ModernEstimator(object):
    """bstats-style API: accumulator_factory() and estimate(suff_stat)."""

    def __init__(self):
        self.log = []
        self.estimate_args = None

    def accumulator_factory(self):
        return _RecordingFactory(self.log)

    def estimate(self, suff_stat):
        self.estimate_args = (suff_stat,)
        return ('modern', suff_stat)


class _LegacyEstimator(object):
    """Legacy stats-style API: accumulatorFactory() and estimate(nobs, ss)."""

    def __init__(self):
        self.log = []
        self.estimate_args = None

    def accumulatorFactory(self):
        return _RecordingFactory(self.log)

    def estimate(self, nobs, suff_stat):
        self.estimate_args = (nobs, suff_stat)
        return ('legacy', nobs, suff_stat)


class DispatchHelperTestCase(unittest.TestCase):
    """The getattr/arity fallbacks the spark and pandas branches rely on."""

    def test_accumulator_factory_prefers_snake_case(self):
        est = _ModernEstimator()
        self.assertIsInstance(bstats._accumulator_factory(est), _RecordingFactory)

    def test_accumulator_factory_falls_back_to_camel_case(self):
        est = _LegacyEstimator()
        self.assertIsInstance(bstats._accumulator_factory(est), _RecordingFactory)

    def test_estimate_dispatch_single_argument(self):
        est = _ModernEstimator()
        rv = bstats._estimator_estimate(est, 5.0, 'ss')
        self.assertEqual(rv, ('modern', 'ss'))
        self.assertEqual(est.estimate_args, ('ss',))

    def test_estimate_dispatch_two_argument(self):
        est = _LegacyEstimator()
        rv = bstats._estimator_estimate(est, 5.0, 'ss')
        self.assertEqual(rv, ('legacy', 5.0, 'ss'))
        self.assertEqual(est.estimate_args, (5.0, 'ss'))

    def test_spark_and_pandas_branches_use_dispatch_helpers(self):
        # The RDD/DataFrame branches cannot be run without a SparkContext;
        # check at the source level that they no longer hard-code the legacy
        # camelCase factory or the two-argument estimate call.
        for fn in (bstats.estimate, bstats.seq_estimate, bstats.initialize):
            src = inspect.getsource(fn)
            self.assertNotIn('.accumulatorFactory()', src,
                             '%s still calls camelCase factory' % fn.__name__)
            self.assertNotIn('estimator.estimate(nobs', src,
                             '%s still calls two-argument estimate' % fn.__name__)
            self.assertIn('_accumulator_factory(', src)
            self.assertIn('_estimator_estimate(', src)


class PandasBranchTestCase(unittest.TestCase):
    """End-to-end pandas DataFrame branches with recording estimators."""

    @staticmethod
    def df():
        return pd.DataFrame({'x': [1.0, 2.0, 3.0]})

    def test_estimate_pandas_modern(self):
        est = _ModernEstimator()
        rv = estimate(self.df(), est)
        self.assertEqual(rv, ('modern', 3.0))
        self.assertEqual(est.log, [('df_update', 3)])
        self.assertEqual(est.estimate_args, (3.0,))

    def test_estimate_pandas_legacy(self):
        est = _LegacyEstimator()
        rv = estimate(self.df(), est)
        self.assertEqual(rv, ('legacy', None, 3.0))
        self.assertEqual(est.log, [('df_update', 3)])
        self.assertEqual(est.estimate_args, (None, 3.0))

    def test_initialize_pandas_modern(self):
        est = _ModernEstimator()
        rv = initialize(self.df(), est, np.random.RandomState(1), 0.5)
        self.assertEqual(rv[0], 'modern')
        self.assertEqual(est.log, [('df_initialize', 3)])

    def test_initialize_pandas_legacy(self):
        est = _LegacyEstimator()
        rv = initialize(self.df(), est, np.random.RandomState(1), 0.5)
        self.assertEqual(rv[0], 'legacy')
        self.assertIsNone(rv[1])
        self.assertEqual(est.log, [('df_initialize', 3)])


class LocalEstimateTestCase(unittest.TestCase):
    """Local (iterable) estimation still works end-to-end on a mixture."""

    @staticmethod
    def make_problem(n=400, seed=1):
        truth = MixtureDistribution(
            [GaussianDistribution(-3.0, 1.0), GaussianDistribution(3.0, 1.0)],
            [0.5, 0.5])
        data = truth.sampler(seed=seed).sample(n)
        est = MixtureEstimator([GaussianEstimator(), GaussianEstimator()])
        return data, est

    def test_estimate_end_to_end(self):
        data, est = self.make_problem()
        mm = initialize(data, est, np.random.RandomState(2), 0.5)

        enc = seq_encode(data, mm)
        _, ll0 = seq_log_density_sum(enc, mm)

        for _ in range(8):
            mm = estimate(data, est, prev_estimate=mm)

        _, ll1 = seq_log_density_sum(enc, mm)
        self.assertGreater(ll1, ll0)
        self.assertAlmostEqual(np.sum(mm.w), 1.0, places=8)

    def test_estimate_recovers_components_from_warm_start(self):
        # The bstats mixture initialize() splits each observation's weight
        # with a flat Dirichlet, leaving the components nearly symmetric, so
        # EM symmetry breaking can take arbitrarily many iterations; component
        # recovery is therefore tested from a deterministic warm start.
        data, est = self.make_problem()
        mm = MixtureDistribution(
            [GaussianDistribution(-1.0, 4.0), GaussianDistribution(1.0, 4.0)],
            [0.5, 0.5])

        for _ in range(10):
            mm = estimate(data, est, prev_estimate=mm)

        self.assertAlmostEqual(np.sum(mm.w), 1.0, places=8)
        mus = sorted(c.mu for c in mm.components)
        self.assertAlmostEqual(mus[0], -3.0, delta=0.5)
        self.assertAlmostEqual(mus[1], 3.0, delta=0.5)

    def test_seq_estimate_matches_estimate(self):
        data, est = self.make_problem(n=300, seed=4)
        mm = initialize(data, est, np.random.RandomState(3), 0.5)
        enc = seq_encode(data, mm)

        m_seq = seq_estimate(enc, est, mm)
        m_loc = estimate(data, est, prev_estimate=mm)

        self.assertTrue(np.allclose(np.sort(m_seq.w), np.sort(m_loc.w)))
        self.assertTrue(np.allclose(sorted(c.mu for c in m_seq.components),
                                    sorted(c.mu for c in m_loc.components)))


class IgnoredTestCase(unittest.TestCase):

    class _RecordingDist(object):
        def __init__(self):
            self.prior = None

        def set_prior(self, prior):
            self.prior = prior

        def get_prior(self):
            return self.prior

    def test_estimator_set_prior_signature(self):
        params = list(inspect.signature(IgnoredEstimator.set_prior).parameters)
        self.assertEqual(params, ['self', 'prior'])

    def test_estimator_set_prior_delegates(self):
        inner = self._RecordingDist()
        est = IgnoredEstimator(dist=inner)
        sentinel = object()
        est.set_prior(sentinel)
        self.assertIs(inner.prior, sentinel)
        self.assertIs(est.get_prior(), sentinel)

    def test_distribution_set_prior_delegates(self):
        inner = self._RecordingDist()
        d = IgnoredDistribution(inner)
        sentinel = object()
        d.set_prior(sentinel)
        self.assertIs(d.get_prior(), sentinel)

    def test_estimate_ignores_data(self):
        g = GaussianDistribution(2.5, 1.5)
        est = IgnoredEstimator(dist=g)

        acc = est.accumulator_factory().make()
        self.assertIsInstance(acc, IgnoredAccumulator)
        for x in [1.0, 2.0, 3.0]:
            acc.update(x, 1.0, None)
        self.assertIsNone(acc.value())
        self.assertIs(acc.combine(None), acc)
        self.assertIs(acc.from_value(None), acc)

        d = est.estimate(acc.value())
        self.assertIsInstance(d, IgnoredDistribution)
        self.assertIs(d.dist, g)

    def test_local_estimate_driver_with_ignored(self):
        g = GaussianDistribution(0.5, 2.0)
        d = estimate([10.0, 20.0, 30.0], IgnoredEstimator(dist=g))
        self.assertIsInstance(d, IgnoredDistribution)
        self.assertEqual(d.dist.mu, 0.5)
        self.assertAlmostEqual(d.log_density(0.5), g.log_density(0.5), places=12)

    def test_scoring_and_sampling_delegate(self):
        g = GaussianDistribution(1.0, 2.0)
        d = IgnoredDistribution(g)
        data = [0.0, 1.0, 2.5]
        enc = d.seq_encode(data)
        self.assertTrue(np.allclose(d.seq_log_density(enc),
                                    [g.log_density(x) for x in data]))
        samples = d.sampler(seed=1).sample(size=5)
        self.assertEqual(len(samples), 5)


class OptionalTestCase(unittest.TestCase):

    class _LegacyTypedDist(object):
        """Inner distribution exposing only the legacy get_type() name."""

        def get_type(self):
            return float

    class _UntypedDist(object):
        """Inner distribution exposing neither get_data_type nor get_type."""
        pass

    def test_get_data_type_bstats_convention(self):
        d = OptionalDistribution(PoissonDistribution(3.0))
        t = d.get_data_type()
        self.assertIn(int, typing.get_args(t))
        self.assertIn(type(None), typing.get_args(t))

    def test_get_data_type_legacy_fallback(self):
        d = OptionalDistribution(self._LegacyTypedDist())
        self.assertIn(float, typing.get_args(d.get_data_type()))

    def test_get_data_type_without_type_methods(self):
        # GaussianDistribution declares neither convention; must not raise.
        d1 = OptionalDistribution(GaussianDistribution(0.0, 1.0))
        self.assertIsNotNone(d1.get_data_type())
        d2 = OptionalDistribution(self._UntypedDist())
        self.assertIsNotNone(d2.get_data_type())

    def test_estimate_recovers_missing_rate_and_component(self):
        truth = OptionalDistribution(GaussianDistribution(2.0, 1.0), p=0.3)
        data = truth.sampler(seed=5).sample(400)
        self.assertTrue(any(x is None for x in data))

        model = estimate(data, OptionalEstimator(GaussianEstimator()))

        n_missing = sum(1 for x in data if x is None)
        self.assertAlmostEqual(model.p, n_missing / float(len(data)), delta=0.01)
        self.assertAlmostEqual(model.dist.mu, 2.0, delta=0.3)

    def test_distribution_estimator_roundtrip(self):
        # Regression: estimator() used to omit the required inner estimator
        # argument and raise TypeError.
        d = OptionalDistribution(GaussianDistribution(2.0, 1.0), p=0.3)
        est = d.estimator()
        self.assertIsInstance(est, OptionalEstimator)
        self.assertIsNone(est.missing_value)

        data = d.sampler(seed=5).sample(200)
        model = estimate(data, est)
        self.assertIsInstance(model, OptionalDistribution)
        n_missing = sum(1 for x in data if x is None)
        self.assertAlmostEqual(model.p, n_missing / float(len(data)), delta=0.02)

    def test_seq_log_density_matches_scalar_with_missing(self):
        d = OptionalDistribution(GaussianDistribution(0.0, 1.0), p=0.25)
        data = d.sampler(seed=7).sample(60)
        enc = d.seq_encode(data)
        self.assertTrue(np.allclose(d.seq_log_density(enc),
                                    [d.log_density(x) for x in data]))


if __name__ == '__main__':
    unittest.main()
