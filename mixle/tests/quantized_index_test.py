"""Tests for bounded bit-quantized enumeration indexes."""

import math
import unittest

import numpy as np

from mixle.enumeration.algorithms import (
    BufferedStream,
    QuantizedEnumerationIndex,
    bounded_best_first_union_index,
    freeze,
)
from mixle.stats import (
    BinomialDistribution,
    CategoricalDistribution,
    CompositeDistribution,
    GeometricDistribution,
    IntegerCategoricalDistribution,
    IntegerUniformSpikeDistribution,
    MixtureDistribution,
    NullDistribution,
    PoissonDistribution,
)
from mixle.stats.compute.pdist import EnumerationError
from mixle.tests.enumeration_test import make_cases


def bounded_items(dist, max_bits, bin_width_bits=1.0):
    rv = []
    for value, lp in dist.enumerator():
        bits = max(0.0, -lp / math.log(2.0))
        if bits > max_bits + 1.0e-12:
            break
        rv.append((value, lp))
    return rv


class QuantizedEnumerationIndexTestCase(unittest.TestCase):
    def test_index_matches_exact_bounded_enumeration_cases(self):
        for name, dist, _, _ in make_cases():
            if isinstance(dist, CompositeDistribution):
                continue
            max_bits = 8.0
            index = dist.quantized_index(max_bits=max_bits)
            exact = bounded_items(dist, max_bits=max_bits)

            self.assertEqual(len(index), len(exact), name)
            self.assertEqual(index.total_count, len(exact), name)
            self.assertEqual([freeze(v) for v, _ in index.iter_from()], [freeze(v) for v, _ in exact], name)
            np.testing.assert_allclose(
                [lp for _, lp in index.iter_from()], [lp for _, lp in exact], atol=1.0e-12, err_msg=name
            )

            expected_counts = {}
            for _, lp in exact:
                b = QuantizedEnumerationIndex.bin_for_log_prob(lp)
                expected_counts[b] = expected_counts.get(b, 0) + 1
            self.assertEqual(index.counts, expected_counts, name)

    def test_random_access_and_slice(self):
        dist = CategoricalDistribution({"a": 0.5, "b": 0.25, "c": 0.125, "d": 0.125})
        index = dist.enumerator().quantized_index(max_bits=4)
        exact = list(dist.enumerator())

        self.assertEqual(index.counts, {1: 1, 2: 1, 3: 2})
        self.assertEqual([freeze(index.get(i)[0]) for i in range(len(index))], [freeze(v) for v, _ in exact])
        self.assertEqual([freeze(v) for v, _ in index.slice(1, 2)], [freeze(v) for v, _ in exact[1:3]])
        self.assertEqual(index.bin_for_index(2), (3, 0))

    def test_bounded_union_stops_from_global_frontier(self):
        streams = [
            BufferedStream(iter([("a", math.log(0.6)), ("tail-a", math.log(1.0e-30))])),
            BufferedStream(iter([("b", math.log(0.4)), ("tail-b", math.log(1.0e-30))])),
        ]
        exact_scores = {
            "a": math.log(0.6),
            "b": math.log(0.4),
            "tail-a": math.log(1.0e-30),
            "tail-b": math.log(1.0e-30),
        }
        scored = []

        def exact_log_density(value):
            scored.append(value)
            return exact_scores[value]

        index = bounded_best_first_union_index(streams, [0.0, 0.0], exact_log_density, max_bits=2.0, bin_width_bits=1.0)

        self.assertEqual([value for value, _ in index.iter_from()], ["a", "b"])
        self.assertEqual(scored, ["a", "b"])
        self.assertTrue(index.truncated)

    def test_bounded_union_reuses_known_component_contributions(self):
        streams = [
            BufferedStream(iter([("a", math.log(0.6))])),
            BufferedStream(iter([("tail", math.log(0.1))])),
        ]
        component_calls = []

        def exact_log_density(value):
            raise AssertionError("component-aware bounded union should not use full scoring")

        def component_log_density(k, value):
            component_calls.append((k, value))
            self.assertEqual((k, value), (1, "a"))
            return -np.inf

        index = bounded_best_first_union_index(
            streams,
            [0.0, 0.0],
            exact_log_density,
            max_bits=2.0,
            bin_width_bits=1.0,
            component_log_density=component_log_density,
        )

        self.assertEqual([value for value, _ in index.iter_from()], ["a"])
        self.assertEqual(component_calls, [(1, "a")])
        self.assertTrue(index.truncated)

    def test_bounded_union_collects_duplicate_contributions_before_stopping(self):
        streams = [
            BufferedStream(iter([("a", math.log(0.4))])),
            BufferedStream(iter([("a", math.log(0.3))])),
        ]
        component_calls = []

        def exact_log_density(value):
            raise AssertionError("known duplicate contributions are sufficient")

        def component_log_density(k, value):
            component_calls.append((k, value))
            return -np.inf

        index = bounded_best_first_union_index(
            streams,
            [0.0, 0.0],
            exact_log_density,
            max_bits=1.0,
            bin_width_bits=1.0,
            component_log_density=component_log_density,
        )

        self.assertEqual([value for value, _ in index.iter_from()], ["a"])
        self.assertAlmostEqual(index.get(0)[1], math.log(0.7), places=12)
        self.assertEqual(component_calls, [])
        self.assertFalse(index.truncated)

    def test_infinite_support_truncates_at_bit_bound(self):
        dist = GeometricDistribution(0.5)
        index = dist.enumerator().quantized_index(max_bits=4)

        self.assertTrue(index.truncated)
        self.assertEqual(index.total_count, 4)
        self.assertEqual([v for v, _ in index.iter_from()], [1, 2, 3, 4])
        self.assertEqual(index.counts, {1: 1, 2: 1, 3: 1, 4: 1})

    def test_argument_validation(self):
        dist = CategoricalDistribution({"a": 1.0})
        with self.assertRaises(ValueError):
            dist.enumerator().quantized_index(max_bits=-1)
        with self.assertRaises(ValueError):
            dist.enumerator().quantized_index(max_bits=1, bin_width_bits=0)

        index = dist.enumerator().quantized_index(max_bits=1)
        with self.assertRaises(IndexError):
            index.get(1)
        with self.assertRaises(IndexError):
            index.slice(-1, 1)
        with self.assertRaises(ValueError):
            index.slice(0, -1)

    def test_native_leaf_indexes_match_enumerator_indexes(self):
        cases = [
            ("categorical", CategoricalDistribution({"a": 0.5, "b": 0.3, "c": 0.2})),
            ("integer_categorical", IntegerCategoricalDistribution(3, [0.0, 0.2, 0.5, 0.3])),
            ("binomial", BinomialDistribution(0.35, 12, min_val=4)),
            ("geometric", GeometricDistribution(0.4)),
            ("poisson", PoissonDistribution(3.7)),
            ("integer_uniform_spike", IntegerUniformSpikeDistribution(k=3, num_vals=6, p=0.45, min_val=1)),
            ("null", NullDistribution()),
        ]

        for name, dist in cases:
            native = dist.quantized_index(max_bits=6.0, bin_width_bits=0.5)
            exact = dist.enumerator().quantized_index(max_bits=6.0, bin_width_bits=0.5)
            self.assertEqual(native.summary(), exact.summary(), name)
            self.assertEqual(
                [freeze(v) for v, _ in native.iter_from()], [freeze(v) for v, _ in exact.iter_from()], name
            )
            np.testing.assert_allclose(
                [lp for _, lp in native.iter_from()], [lp for _, lp in exact.iter_from()], atol=1.0e-12, err_msg=name
            )

    def test_native_leaf_indexes_do_not_require_enumerator(self):
        class DirectCategorical(CategoricalDistribution):
            def enumerator(self):
                raise AssertionError("quantized_index should not call enumerator")

        index = DirectCategorical({"a": 0.75, "b": 0.25}).quantized_index(max_bits=3)
        self.assertEqual([v for v, _ in index.iter_from()], ["a", "b"])
        self.assertEqual(index.counts, {0: 1, 2: 1})

    def test_mixture_frontier_index_matches_exact_without_mixture_enumerator(self):
        class DirectIntegerCategorical(IntegerCategoricalDistribution):
            def quantized_index(self, max_bits, bin_width_bits=1.0):
                raise AssertionError("mixture frontier should not call child quantized_index")

        class DirectMixture(MixtureDistribution):
            def enumerator(self):
                raise AssertionError("mixture quantized_index should not call enumerator")

        components = [
            DirectIntegerCategorical(0, [0.7, 0.2, 0.1]),
            DirectIntegerCategorical(1, [0.5, 0.5]),
        ]
        dist = DirectMixture(components, [0.6, 0.4])
        exact_dist = MixtureDistribution(
            [IntegerCategoricalDistribution(0, [0.7, 0.2, 0.1]), IntegerCategoricalDistribution(1, [0.5, 0.5])],
            [0.6, 0.4],
        )

        native = dist.quantized_index(max_bits=4.0, bin_width_bits=0.5)
        exact = exact_dist.enumerator().quantized_index(max_bits=4.0, bin_width_bits=0.5)
        self.assertEqual(native.summary(), exact.summary())
        self.assertEqual([freeze(v) for v, _ in native.iter_from()], [freeze(v) for v, _ in exact.iter_from()])
        np.testing.assert_allclose(
            [lp for _, lp in native.iter_from()], [lp for _, lp in exact.iter_from()], atol=1.0e-12
        )

    def test_three_component_mixture_falls_back_to_multi_cross_index(self):
        class DirectIntegerCategorical(IntegerCategoricalDistribution):
            def enumerator(self):
                raise EnumerationError(self, reason="force cross-index fallback")

            def quantized_index(self, max_bits, bin_width_bits=1.0):
                raise AssertionError("mixture should use multi cross-index, not marginal indexes")

        class DirectMixture(MixtureDistribution):
            def enumerator(self):
                raise AssertionError("mixture enumerator should not be used")

        components = [
            DirectIntegerCategorical(0, [0.7, 0.2, 0.1]),
            DirectIntegerCategorical(1, [0.5, 0.25, 0.25]),
            DirectIntegerCategorical(2, [0.1, 0.9]),
        ]
        dist = DirectMixture(components, [0.4, 0.35, 0.25])
        exact_dist = MixtureDistribution(
            [
                IntegerCategoricalDistribution(0, [0.7, 0.2, 0.1]),
                IntegerCategoricalDistribution(1, [0.5, 0.25, 0.25]),
                IntegerCategoricalDistribution(2, [0.1, 0.9]),
            ],
            [0.4, 0.35, 0.25],
        )

        native = dist.quantized_index(max_bits=5.0, bin_width_bits=0.5)
        exact = exact_dist.enumerator().quantized_index(max_bits=5.0, bin_width_bits=0.5)
        self.assertEqual(native.summary(), exact.summary())
        self.assertEqual([freeze(v) for v, _ in native.iter_from()], [freeze(v) for v, _ in exact.iter_from()])
        np.testing.assert_allclose(
            [lp for _, lp in native.iter_from()], [lp for _, lp in exact.iter_from()], atol=1.0e-12
        )

    def test_mixture_falls_back_to_generic_multi_cross_index_for_other_leaves(self):
        class DirectBinomial(BinomialDistribution):
            def enumerator(self):
                raise EnumerationError(self, reason="force cross-index fallback")

        class DirectMixture(MixtureDistribution):
            def enumerator(self):
                raise AssertionError("mixture enumerator should not be used")

        dist = DirectMixture(
            [DirectBinomial(0.25, 6), DirectBinomial(0.65, 6), DirectBinomial(0.45, 6)], [0.2, 0.5, 0.3]
        )
        exact_dist = MixtureDistribution(
            [BinomialDistribution(0.25, 6), BinomialDistribution(0.65, 6), BinomialDistribution(0.45, 6)],
            [0.2, 0.5, 0.3],
        )

        native = dist.quantized_index(max_bits=5.0, bin_width_bits=0.5)
        exact = exact_dist.enumerator().quantized_index(max_bits=5.0, bin_width_bits=0.5)
        self.assertEqual(native.summary(), exact.summary())
        self.assertEqual([freeze(v) for v, _ in native.iter_from()], [freeze(v) for v, _ in exact.iter_from()])
        np.testing.assert_allclose(
            [lp for _, lp in native.iter_from()], [lp for _, lp in exact.iter_from()], atol=1.0e-12
        )

    def test_mixture_of_composites_falls_back_to_composite_cross_index(self):
        class DirectCategorical(CategoricalDistribution):
            def enumerator(self):
                raise AssertionError("categorical enumerator should not be used")

            def quantized_index(self, max_bits, bin_width_bits=1.0):
                raise AssertionError("mixture should use cross-index, not marginal indexes")

        class DirectComposite(CompositeDistribution):
            def enumerator(self):
                raise EnumerationError(self, reason="force cross-index fallback")

            def quantized_index(self, max_bits, bin_width_bits=1.0):
                raise AssertionError("mixture should use composite cross-index")

        class DirectMixture(MixtureDistribution):
            def enumerator(self):
                raise AssertionError("mixture enumerator should not be used")

        comp0 = DirectComposite(
            (
                DirectCategorical({"a": 0.6, "b": 0.4}),
                DirectCategorical({"x": 0.7, "y": 0.3}),
            )
        )
        comp1 = DirectComposite(
            (
                DirectCategorical({"a": 0.2, "b": 0.8}),
                DirectCategorical({"x": 0.25, "z": 0.75}),
            )
        )
        dist = DirectMixture([comp0, comp1], [0.55, 0.45])

        exact_dist = MixtureDistribution(
            [
                CompositeDistribution(
                    (
                        CategoricalDistribution({"a": 0.6, "b": 0.4}),
                        CategoricalDistribution({"x": 0.7, "y": 0.3}),
                    )
                ),
                CompositeDistribution(
                    (
                        CategoricalDistribution({"a": 0.2, "b": 0.8}),
                        CategoricalDistribution({"x": 0.25, "z": 0.75}),
                    )
                ),
            ],
            [0.55, 0.45],
        )

        native = dist.quantized_index(max_bits=5.0, bin_width_bits=0.5)
        exact = exact_dist.enumerator().quantized_index(max_bits=5.0, bin_width_bits=0.5)
        self.assertEqual(native.summary(), exact.summary())
        self.assertEqual([freeze(v) for v, _ in native.iter_from()], [freeze(v) for v, _ in exact.iter_from()])
        np.testing.assert_allclose(
            [lp for _, lp in native.iter_from()], [lp for _, lp in exact.iter_from()], atol=1.0e-12
        )

    def test_composite_native_index_uses_quantized_dp_without_enumerators(self):
        class DirectCategorical(CategoricalDistribution):
            def enumerator(self):
                raise AssertionError("child quantized_index should not call enumerator")

        class DirectComposite(CompositeDistribution):
            def enumerator(self):
                raise AssertionError("composite quantized_index should not call enumerator")

        dist = DirectComposite(
            (
                DirectCategorical({"a": 0.5, "b": 0.25}),
                DirectCategorical({"x": 0.5, "y": 0.25}),
            )
        )

        index = dist.quantized_index(max_bits=3.0, bin_width_bits=1.0)
        self.assertEqual(index.counts, {2: 1, 3: 2})
        self.assertEqual([freeze(v) for v, _ in index.iter_from()], [("a", "x"), ("a", "y"), ("b", "x")])
        for value, log_prob in index.iter_from():
            self.assertAlmostEqual(log_prob, dist.log_density(value), delta=1.0e-12)
        self.assertEqual(index.get(2)[0], ("b", "x"))
        with self.assertRaises(IndexError):
            index.get(3)


if __name__ == "__main__":
    unittest.main()
