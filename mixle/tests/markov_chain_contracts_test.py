"""Closed-support, generative, and sufficient-statistic contracts for generic Markov chains."""

import copy
import unittest

import numpy as np

from mixle.stats import (
    CategoricalDistribution,
    MarkovChainDistribution,
    MarkovChainEstimator,
    MarkovChainStatistics,
    NonGenerativeMarkovChainError,
)
from mixle.stats.bayes.dirichlet import DirichletDistribution


def _chain(len_dist=None):
    return MarkovChainDistribution(
        {"a": 0.6, "b": 0.4},
        {
            "a": {"a": 0.7, "b": 0.3},
            "b": {"a": 0.2, "b": 0.8},
        },
        len_dist=len_dist,
    )


class MarkovChainContractsTest(unittest.TestCase):
    def test_probability_tables_are_closed_validated_copies(self):
        initial = {"a": 0.6, "b": 0.4}
        transitions = {
            "a": {"a": 0.7, "b": 0.3},
            "b": {"a": 0.2, "b": 0.8},
        }
        dist = MarkovChainDistribution(initial, transitions)
        initial["a"] = 1.0
        transitions["a"]["a"] = 1.0
        self.assertEqual(dist.init_prob_map["a"], 0.6)
        self.assertEqual(dist.transition_map["a"]["a"], 0.7)
        with self.assertRaises(TypeError):
            dist.init_prob_map["a"] = 0.5
        copied = copy.deepcopy(dist)
        self.assertEqual(copied.init_prob_map, dist.init_prob_map)
        self.assertEqual(copied.transition_map, dist.transition_map)
        initial["a"] = 0.6
        transitions["a"]["a"] = 0.7

        invalid = (
            ({"a": 1.1, "b": -0.1}, transitions),
            ({"a": 0.7, "b": 0.4}, transitions),
            (initial, {"a": {"a": 0.7, "b": 0.3}}),
            (initial, {"a": {"a": 0.7, "b": 0.3}, "b": {"a": np.nan, "b": 0.8}}),
        )
        for init_map, trans_map in invalid:
            with self.assertRaises((TypeError, ValueError)):
                MarkovChainDistribution(init_map, trans_map)
        with self.assertRaisesRegex(ValueError, "default_value must be 0"):
            MarkovChainDistribution(initial, transitions, default_value=0.1)

    def test_missing_known_and_unknown_entries_match_scalar_and_batch(self):
        dist = MarkovChainDistribution(
            {"a": 1.0, "b": 0.0},
            {
                "a": {"a": 1.0, "b": 0.0},
                "b": {"a": 0.0, "b": 1.0},
            },
        )
        data = [["b"], ["a", "b"], ["unknown"], ["a", "unknown"]]
        scalar = np.asarray([dist.log_density(row) for row in data])
        batch = dist.seq_log_density(dist.dist_to_encoder().seq_encode(data))
        np.testing.assert_array_equal(np.isneginf(batch), np.isneginf(scalar))
        self.assertTrue(np.all(np.isneginf(batch)))

    def test_lengthless_chain_is_a_factor_with_explicit_path_sampling(self):
        dist = _chain()
        self.assertEqual(dist.density_semantics().value, "likelihood_factor")
        with self.assertRaises(NonGenerativeMarkovChainError):
            dist.sampler(seed=2)
        sampler = dist.path_sampler(seed=2)
        self.assertEqual(len(sampler.sample_seq(5)), 5)
        self.assertEqual([len(row) for row in sampler.sample_paths([0, 2, 4])], [0, 2, 4])
        self.assertFalse(any(value is None for row in sampler.sample_paths([8] * 5) for value in row))
        for bad in (-1, 1.5, np.nan, True):
            with self.assertRaises((TypeError, ValueError)):
                sampler.sample_seq(bad)

    def test_length_law_is_proved_at_construction(self):
        with self.assertRaises((TypeError, ValueError)):
            _chain(CategoricalDistribution({-1: 1.0}))
        with self.assertRaises((TypeError, ValueError)):
            _chain(CategoricalDistribution({1.5: 1.0}))
        proper = _chain(CategoricalDistribution({0: 0.25, 3: 0.75}))
        samples = proper.sampler(seed=5).sample(size=20)
        self.assertTrue(all(len(row) in (0, 3) for row in samples))

    def test_empty_and_all_empty_encodings_are_canonical(self):
        encoder = _chain().dist_to_encoder()
        for data in ([], [[], []]):
            encoded = encoder.seq_encode(data)
            self.assertEqual(encoder.row_count(encoded), len(data))
            self.assertEqual(len(encoded[6]), 0)

    def test_statistics_are_immutable_exact_and_counted(self):
        estimator = MarkovChainEstimator(levels=("a", "b"), pseudo_count=1.0)
        acc = estimator.accumulator_factory().make()
        acc.update(["a", "b"], 2.0, None)
        acc.update(["b", "a", "b"], 3.0, None)
        statistics = acc.value()
        self.assertIsInstance(statistics, MarkovChainStatistics)
        self.assertEqual(statistics.length_nobs, 5.0)
        self.assertEqual(statistics.initial_counts, (2.0, 3.0))
        fitted = estimator.estimate(5.0, statistics)
        self.assertEqual(tuple(fitted.init_prob_map), ("a", "b"))
        self.assertEqual(tuple(fitted.transition_map), ("a", "b"))

        restored = estimator.accumulator_factory().make().from_value(statistics)
        restored.update(["a", "a"], 7.0, None)
        self.assertEqual(statistics.initial_counts, (2.0, 3.0))
        self.assertNotEqual(restored.value().initial_counts, statistics.initial_counts)

    def test_corrupt_statistics_and_unsmoothed_empty_rows_fail(self):
        estimator = MarkovChainEstimator(levels=("a", "b"))
        bad = MarkovChainStatistics(
            1,
            ("a", "b"),
            (1.0, -1.0),
            ((1.0, 0.0), (0.0, 1.0)),
            1.0,
            None,
        )
        with self.assertRaises(ValueError):
            estimator.estimate(1.0, bad)

        acc = estimator.accumulator_factory().make()
        acc.update(["a"], 1.0, None)
        with self.assertRaisesRegex(ValueError, "transition evidence"):
            estimator.estimate(1.0, acc.value())
        fitted = MarkovChainEstimator(levels=("a", "b"), pseudo_count=1.0).estimate(
            1.0,
            acc.value(),
        )
        self.assertTrue(all(np.isclose(sum(row.values()), 1.0) for row in fitted.transition_map.values()))

    def test_empty_accumulator_value_combines_and_reconstructs_without_raising(self):
        # An undeclared-levels accumulator that has combined no data (e.g. a distributed backend's
        # per-partition accumulator for a component that saw zero rows in that partition) must be
        # readable and mergeable -- internal machinery (transactional snapshots, shuffle-merge)
        # calls .value()/.combine()/.from_value() on exactly this state. Regression for a crash
        # reached through mixle.stats.latent.mixture.MixtureAccumulator.from_value's pre-mutation
        # snapshot when running optimize() over a Spark RDD (STAT-NB-estimation_using_spark).
        empty = MarkovChainEstimator().accumulator_factory().make()
        statistics = empty.value()
        self.assertEqual(statistics.states, ())
        self.assertEqual(statistics.initial_counts, ())
        self.assertEqual(statistics.transition_counts, ())

        reconstructed = MarkovChainEstimator().accumulator_factory().make().from_value(statistics)
        self.assertIsNone(reconstructed.levels)
        self.assertEqual(reconstructed.value().states, ())

        merged = MarkovChainEstimator().accumulator_factory().make()
        merged.combine(statistics)
        self.assertIsNone(merged.levels)

        # Merging empty-then-real data in either order recovers the real observation exactly --
        # the empty statistics tuple is a true identity element for combine()/from_value().
        real = MarkovChainEstimator(levels=("a", "b")).accumulator_factory().make()
        real.update(["a", "b", "a"], 1.0, None)
        real_statistics = real.value()

        empty_then_real = MarkovChainEstimator().accumulator_factory().make()
        empty_then_real.from_value(statistics)
        empty_then_real.combine(real_statistics)
        self.assertEqual(empty_then_real.value().states, ("a", "b"))
        self.assertEqual(empty_then_real.value().initial_counts, real_statistics.initial_counts)

        # The guard this replaces still fires, with better context, at the point that actually
        # matters: trying to estimate a distribution from genuinely empty statistics.
        with self.assertRaisesRegex(ValueError, "states cannot be empty"):
            MarkovChainEstimator().estimate(0.0, statistics)

    def test_prior_layout_is_exact_and_evidence_cannot_disappear(self):
        alpha2 = DirichletDistribution([1.0, 1.0])
        valid = (("a", "b"), alpha2, (alpha2, alpha2))
        dist = MarkovChainDistribution(
            {"a": 0.6, "b": 0.4},
            {"a": {"a": 0.7, "b": 0.3}, "b": {"a": 0.2, "b": 0.8}},
            prior=valid,
        )
        self.assertTrue(dist.has_conj_prior)
        with self.assertRaises(ValueError):
            dist.set_prior((("b", "a"), alpha2, (alpha2, alpha2)))
        with self.assertRaises(ValueError):
            dist.set_prior((("a", "b"), alpha2, (alpha2,)))
        with self.assertRaises(ValueError):
            dist.set_prior(
                (
                    ("a", "b"),
                    DirichletDistribution([1.0, 1.0, 1.0]),
                    (alpha2, alpha2),
                )
            )

        acc = dist.estimator().accumulator_factory().make()
        with self.assertRaisesRegex(ValueError, "outside"):
            acc.update(["unknown"], 1.0, None)

    def test_expected_scoring_delegates_to_length_posterior(self):
        alpha2 = DirichletDistribution([1.0, 1.0])
        dist = MarkovChainDistribution(
            {"a": 0.6, "b": 0.4},
            {"a": {"a": 0.7, "b": 0.3}, "b": {"a": 0.2, "b": 0.8}},
            len_dist=CategoricalDistribution({1: 1.0}),
            prior=(("a", "b"), alpha2, (alpha2, alpha2)),
        )
        dist.len_dist.expected_log_density = lambda value: 11.0
        dist.len_dist.seq_expected_log_density = lambda encoded: np.full(2, 13.0)
        expected_state = float(dist.e_log_init[0])
        self.assertAlmostEqual(dist.expected_log_density(["a"]), expected_state + 11.0)
        encoded = dist.dist_to_encoder().seq_encode([["a"], ["a"]])
        np.testing.assert_allclose(
            dist.seq_expected_log_density(encoded),
            np.full(2, expected_state + 13.0),
        )

    def test_mixed_state_diagnostics_and_key_pooling_are_stable(self):
        dist = MarkovChainDistribution(
            {1: 0.5, "1": 0.5},
            {
                1: {1: 0.5, "1": 0.5},
                "1": {1: 0.5, "1": 0.5},
            },
        )
        self.assertIn("MarkovChainDistribution", str(dist))
        estimator = MarkovChainEstimator(levels=(1, "1"), pseudo_count=1.0, keys="shared")
        first = estimator.accumulator_factory().make()
        second = estimator.accumulator_factory().make()
        first.update([1, "1"], 2.0, None)
        pooled = {}
        first.key_merge(pooled)
        first.update([1, 1], 5.0, None)
        second.key_replace(pooled)
        self.assertEqual(second.value().length_nobs, 2.0)


if __name__ == "__main__":
    unittest.main()
