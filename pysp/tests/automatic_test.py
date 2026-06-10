"""Tests for automatic estimator detection (pysp.utils.automatic)."""
import unittest

import numpy as np

from pysp.stats import (
    BernoulliSetEstimator, CategoricalEstimator, CompositeEstimator, GaussianEstimator,
    IgnoredEstimator, OptionalEstimator, PoissonEstimator, SequenceEstimator,
    initialize, estimate,
)
from pysp.utils.automatic import get_estimator


class AutomaticDetectionTestCase(unittest.TestCase):

    def test_floats_gaussian(self):
        est = get_estimator([1.2, 3.4, -0.5, 2.2] * 30)
        self.assertIsInstance(est, GaussianEstimator)

    def test_bools_categorical(self):
        # bool is a subclass of int in Python; must not fall into the int path
        est = get_estimator([True, False, True] * 40)
        self.assertIsInstance(est, CategoricalEstimator)

    def test_small_cardinality_ints_categorical(self):
        est = get_estimator([1, 2, 3, 2, 1] * 30)
        self.assertIsInstance(est, CategoricalEstimator)

    def test_counts_poisson(self):
        data = list(np.random.RandomState(0).poisson(50, 300))
        est = get_estimator(data)
        self.assertIsInstance(est, PoissonEstimator)

    def test_signed_ints_gaussian(self):
        data = list(np.random.RandomState(0).randint(-500, 500, 300))
        est = get_estimator(data)
        self.assertIsInstance(est, GaussianEstimator)

    def test_id_strings_ignored(self):
        est = get_estimator(['id_%d' % i for i in range(200)])
        self.assertIsInstance(est, IgnoredEstimator)

    def test_cat_strings_categorical(self):
        est = get_estimator(['a', 'b', 'c'] * 50)
        self.assertIsInstance(est, CategoricalEstimator)

    def test_tuples_composite(self):
        est = get_estimator([('a', 1.5, 3), ('b', 2.5, 1)] * 50)
        self.assertIsInstance(est, CompositeEstimator)
        self.assertEqual(len(est.estimators), 3)

    def test_variable_length_lists_sequence_with_length_model(self):
        est = get_estimator([['a', 'b'], ['a'], ['b', 'c', 'a']] * 40)
        self.assertIsInstance(est, SequenceEstimator)
        self.assertIsInstance(est.estimator, CategoricalEstimator)
        self.assertIsInstance(est.len_estimator, PoissonEstimator)

    def test_fixed_length_word_lists_sequence(self):
        # homogeneous fixed-length lists are sequences, not 3-slot records
        est = get_estimator([['a', 'b', 'c'], ['b', 'a', 'a']] * 50)
        self.assertIsInstance(est, SequenceEstimator)
        self.assertIsInstance(est.len_estimator, CategoricalEstimator)

    def test_fixed_length_float_lists_composite(self):
        # numeric fixed-length lists are vectors: dimensions keep identity
        est = get_estimator([[0.1, 2.0, -1.0], [0.3, 1.9, -1.2]] * 50)
        self.assertIsInstance(est, CompositeEstimator)
        self.assertEqual(len(est.estimators), 3)

    def test_sets_bernoulli_set(self):
        est = get_estimator([{'x', 'y'}, {'y'}, {'x', 'z'}] * 40)
        self.assertIsInstance(est, BernoulliSetEstimator)

    def test_none_optional(self):
        est = get_estimator([1.5, None, 2.5, 3.5] * 30)
        self.assertIsInstance(est, OptionalEstimator)

    def test_dicts_ignored(self):
        est = get_estimator([{'k': 1}] * 20)
        self.assertIsInstance(est, IgnoredEstimator)

    def test_mixed_scalars_and_containers_ignored(self):
        est = get_estimator([1.0, [1.0, 2.0], 'a'] * 20)
        self.assertIsInstance(est, IgnoredEstimator)

    def test_detected_estimators_are_fittable(self):
        # estimators built from detection must round-trip through estimation
        cases = [
            [('a', 1.5, 3), ('b', 2.5, 1), ('a', 0.5, 3)] * 20,
            [['a', 'b'], ['a'], ['b', 'c', 'a']] * 20,
            [{'x', 'y'}, {'y'}, {'x', 'z'}] * 20,
            [1.5, None, 2.5, 3.5] * 20,
        ]
        for data in cases:
            est = get_estimator(data)
            init = initialize(data, est, rng=np.random.RandomState(1), p=1.0)
            model = estimate(data, est, init)
            ll = model.log_density(data[0])
            self.assertTrue(np.isfinite(ll) or ll == -np.inf)

    def test_bstats_estimators_constructible(self):
        for data in ([('a', 1.5, [1.0, 2.0]), ('b', 2.5, [0.5, 1.5])] * 30,
                     [['a', 'b'], ['a'], ['b', 'c', 'a']] * 20,
                     list(np.random.RandomState(0).poisson(50, 200))):
            est = get_estimator(data, use_bstats=True)
            self.assertIsNotNone(est.accumulator_factory() if hasattr(est, 'accumulator_factory')
                                 else est.accumulatorFactory())


if __name__ == '__main__':
    unittest.main()
