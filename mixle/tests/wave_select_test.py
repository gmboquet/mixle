"""Tests for mixle.stats.combinator.select fixes (routing, factory, value()), the select and conditional
enumerators, and ss_mixture scalar-vs-vectorized accumulation.

The routing tests use two children with very different densities so that any encoder-group /
choice-index mix-up produces wildly wrong numbers.
"""

import unittest

import numpy as np
from numpy.random import RandomState

from mixle.stats.combinator.conditional import ConditionalDistribution
from mixle.stats.combinator.select import SelectDistribution, SelectEstimator
from mixle.stats.compute.pdist import EnumerationError
from mixle.stats.latent.semi_supervised_mixture import SemiSupervisedMixtureDistribution, SemiSupervisedMixtureEstimator
from mixle.stats.univariate.continuous.gaussian import GaussianDistribution, GaussianEstimator
from mixle.stats.univariate.discrete.categorical import CategoricalDistribution

TOL = 1e-9


def numeric_choice(x) -> int:
    """Routes small observations to child 0 and large observations to child 1."""
    return 0 if x < 50.0 else 1


def letter_choice(x) -> int:
    """Routes 'x'/'y' to child 1, everything else to child 0."""
    return 1 if x in ("x", "y") else 0


class SelectRoutingTestCase(unittest.TestCase):
    """Regression tests for the encoder-group vs choice-index routing bug in select.py."""

    def setUp(self) -> None:
        self.dists = [GaussianDistribution(0.0, 1.0), GaussianDistribution(100.0, 1.0)]
        self.dist = SelectDistribution(self.dists, numeric_choice)
        # The first observation routes to child 1, so the encoder's group order (1, 0) differs
        # from the child index order. Old code indexed children by group position and broke here.
        self.data = [100.5, 0.5, 99.5, -0.5, 101.0, 1.0]
        self.low = [0.5, -0.5, 1.0]
        self.high = [100.5, 99.5, 101.0]

    def test_seq_log_density_matches_scalar(self) -> None:
        enc = self.dist.dist_to_encoder().seq_encode(self.data)
        sld = self.dist.seq_log_density(enc)
        expected = np.asarray([self.dist.log_density(x) for x in self.data])

        self.assertEqual(len(sld), len(self.data))
        self.assertTrue(
            np.allclose(sld, expected), "seq_log_density disagrees with log_density: %s vs %s" % (sld, expected)
        )

    def test_seq_update_routes_to_correct_child(self) -> None:
        est = SelectEstimator([GaussianEstimator(), GaussianEstimator()], numeric_choice)
        acc = est.accumulator_factory().make()
        enc = self.dist.dist_to_encoder().seq_encode(self.data)
        acc.seq_update(enc, np.ones(len(self.data)), self.dist)

        fitted = est.estimate(None, acc.value())
        self.assertAlmostEqual(fitted.dists[0].mu, np.mean(self.low), places=8)
        self.assertAlmostEqual(fitted.dists[1].mu, np.mean(self.high), places=8)
        self.assertAlmostEqual(acc.weights[0], 3.0, places=10)
        self.assertAlmostEqual(acc.weights[1], 3.0, places=10)

    def test_seq_initialize_routes_to_correct_child(self) -> None:
        est = SelectEstimator([GaussianEstimator(), GaussianEstimator()], numeric_choice)
        acc = est.accumulator_factory().make()
        enc = acc.acc_to_encoder().seq_encode(self.data)
        acc.seq_initialize(enc, np.ones(len(self.data)), RandomState(3))

        fitted = est.estimate(None, acc.value())
        self.assertAlmostEqual(fitted.dists[0].mu, np.mean(self.low), places=8)
        self.assertAlmostEqual(fitted.dists[1].mu, np.mean(self.high), places=8)

    def test_scalar_estimate_round_trip(self) -> None:
        est = SelectEstimator([GaussianEstimator(), GaussianEstimator()], numeric_choice)
        acc = est.accumulator_factory().make()
        rng = RandomState(1)
        for x in self.data:
            acc.initialize(x, 1.0, rng)

        fitted = est.estimate(None, acc.value())
        self.assertAlmostEqual(fitted.dists[0].mu, np.mean(self.low), places=8)
        self.assertAlmostEqual(fitted.dists[1].mu, np.mean(self.high), places=8)

    def test_accumulator_factory_make_with_snake_case_children(self) -> None:
        # GaussianEstimator only defines accumulator_factory(); the old factory called the
        # legacy camelCase accumulatorFactory() and raised AttributeError.
        child = GaussianEstimator()
        self.assertFalse(hasattr(child, "accumulatorFactory"))

        est = SelectEstimator([GaussianEstimator(), GaussianEstimator()], numeric_choice)
        acc = est.accumulator_factory().make()
        self.assertEqual(len(acc.accumulators), 2)

    def test_value_is_reusable_and_round_trips(self) -> None:
        est = SelectEstimator([GaussianEstimator(), GaussianEstimator()], numeric_choice)
        acc = est.accumulator_factory().make()
        for x in self.data:
            acc.update(x, 1.0, self.dist)

        value = acc.value()
        self.assertEqual(len(value), 2)
        # A zip object would be exhausted after one pass; a list supports repeated iteration.
        first_pass = [w for w, _ in value]
        second_pass = [w for w, _ in value]
        self.assertEqual(first_pass, second_pass)

        other = est.accumulator_factory().make()
        other.from_value(value)
        self.assertTrue(np.allclose(other.weights, acc.weights))

        combined = est.accumulator_factory().make()
        combined.combine(value)
        self.assertTrue(np.allclose(combined.weights, acc.weights))

    def test_encoder_eq_and_str(self) -> None:
        enc_a = self.dist.dist_to_encoder()
        enc_b = self.dist.dist_to_encoder()
        self.assertEqual(enc_a, enc_b)
        self.assertNotEqual(enc_a, "not an encoder")
        self.assertIn("SelectDataEncoder", str(enc_a))


class SelectEnumeratorTestCase(unittest.TestCase):
    """Tests for the select enumerator over the union of child supports."""

    def test_union_sorted_exact_and_deduped(self) -> None:
        d0 = CategoricalDistribution({"a": 0.5, "b": 0.5})
        d1 = CategoricalDistribution({"a": 0.3, "x": 0.7})
        dist = SelectDistribution([d0, d1], letter_choice)

        items = dist.enumerator().top_k(10)
        vals = [v for v, _ in items]
        lps = [lp for _, lp in items]

        # 'a' appears in both child supports but routes to child 0; it must be emitted once with
        # the child-0 score. 'x' routes to child 1.
        self.assertEqual(set(vals), {"a", "b", "x"})
        self.assertEqual(len(vals), len(set(vals)))
        self.assertEqual(vals[0], "x")

        for v, lp in items:
            self.assertAlmostEqual(lp, dist.log_density(v), places=10)
        for i in range(len(lps) - 1):
            self.assertGreaterEqual(lps[i], lps[i + 1] - TOL)

    def test_zero_probability_values_skipped(self) -> None:
        # 'y' is routed to child 1 but only child 0 gives it mass, so p('y') = 0 and it must
        # never be yielded.
        d0 = CategoricalDistribution({"a": 0.6, "y": 0.4})
        d1 = CategoricalDistribution({"x": 1.0})
        dist = SelectDistribution([d0, d1], letter_choice)

        items = dist.enumerator().top_k(10)
        vals = [v for v, _ in items]
        self.assertEqual(set(vals), {"a", "x"})

    def test_non_enumerable_child_fails_fast(self) -> None:
        dist = SelectDistribution([GaussianDistribution(0.0, 1.0), CategoricalDistribution({"x": 1.0})], letter_choice)
        with self.assertRaises(EnumerationError) as ctx:
            dist.enumerator()
        self.assertIn("SelectDistribution.dists[0]", str(ctx.exception))


class ConditionalEnumeratorTestCase(unittest.TestCase):
    """Tests for the conditional enumerator over (given, value) pairs."""

    def test_joint_enumeration_sorted_and_exact(self) -> None:
        given = CategoricalDistribution({"a": 0.6, "b": 0.4})
        dmap = {"a": CategoricalDistribution({"x": 0.7, "y": 0.3}), "b": CategoricalDistribution({"x": 0.2, "z": 0.8})}
        dist = ConditionalDistribution(dmap, given_dist=given)

        items = dist.enumerator().top_k(10)
        expected = {("a", "x"): 0.42, ("a", "y"): 0.18, ("b", "x"): 0.08, ("b", "z"): 0.32}

        self.assertEqual(len(items), 4)
        self.assertEqual({v for v, _ in items}, set(expected))
        for v, lp in items:
            self.assertAlmostEqual(lp, np.log(expected[v]), places=10)
            self.assertAlmostEqual(lp, dist.log_density(v), places=10)

        lps = [lp for _, lp in items]
        for i in range(len(lps) - 1):
            self.assertGreaterEqual(lps[i], lps[i + 1] - TOL)

    def test_default_distribution_covers_missing_keys(self) -> None:
        given = CategoricalDistribution({"a": 0.5, "b": 0.3, "c": 0.2})
        dmap = {"a": CategoricalDistribution({"x": 1.0}), "b": CategoricalDistribution({"y": 1.0})}
        dist = ConditionalDistribution(dmap, given_dist=given, default_dist=CategoricalDistribution({"q": 1.0}))

        items = dist.enumerator().top_k(10)
        as_dict = {v: lp for v, lp in items}
        self.assertEqual(set(as_dict), {("a", "x"), ("b", "y"), ("c", "q")})
        self.assertAlmostEqual(as_dict[("c", "q")], np.log(0.2), places=10)

    def test_missing_key_without_default_contributes_nothing(self) -> None:
        given = CategoricalDistribution({"a": 0.5, "c": 0.5})
        dmap = {"a": CategoricalDistribution({"x": 0.9, "y": 0.1})}
        dist = ConditionalDistribution(dmap, given_dist=given)

        items = dist.enumerator().top_k(10)
        self.assertEqual({v for v, _ in items}, {("a", "x"), ("a", "y")})

    def test_null_given_fails_fast(self) -> None:
        dist = ConditionalDistribution({"a": CategoricalDistribution({"x": 1.0})})
        with self.assertRaises(EnumerationError) as ctx:
            dist.enumerator()
        self.assertIn("given distribution", str(ctx.exception))

    def test_non_enumerable_conditional_fails_fast(self) -> None:
        given = CategoricalDistribution({"a": 1.0})
        dist = ConditionalDistribution({"a": GaussianDistribution(0.0, 1.0)}, given_dist=given)
        with self.assertRaises(EnumerationError) as ctx:
            dist.enumerator()
        self.assertIn("dmap", str(ctx.exception))


class SemiSupervisedMixtureAccumulatorTestCase(unittest.TestCase):
    """Checks that scalar update() matches seq_update() on the same data."""

    def setUp(self) -> None:
        comps = [GaussianDistribution(0.0, 1.0), GaussianDistribution(3.0, 1.0)]
        self.dist = SemiSupervisedMixtureDistribution(comps, [0.5, 0.5])
        self.est = SemiSupervisedMixtureEstimator([GaussianEstimator(), GaussianEstimator()])
        self.data = [
            (0.2, None),
            (2.9, [(1, 1.0)]),
            (1.5, [(0, 0.6), (1, 0.4)]),
            (-0.3, None),
            (3.3, [(1, 0.7), (0, 0.3)]),
        ]

    def test_scalar_update_matches_seq_update(self) -> None:
        scalar_acc = self.est.accumulator_factory().make()
        for xi in self.data:
            scalar_acc.update(xi, 1.0, self.dist)

        seq_acc = self.est.accumulator_factory().make()
        enc = self.dist.dist_to_encoder().seq_encode(self.data)
        seq_acc.seq_update(enc, np.ones(len(self.data)), self.dist)

        v_scalar, v_seq = scalar_acc.value(), seq_acc.value()
        self.assertTrue(np.allclose(v_scalar[0], v_seq[0]), "comp_counts differ: %s vs %s" % (v_scalar[0], v_seq[0]))
        for ss_scalar, ss_seq in zip(v_scalar[1], v_seq[1]):
            self.assertTrue(
                np.allclose(np.asarray(ss_scalar, dtype=float), np.asarray(ss_seq, dtype=float)),
                "component suff stats differ: %s vs %s" % (ss_scalar, ss_seq),
            )

    def test_estimates_agree_between_paths(self) -> None:
        scalar_acc = self.est.accumulator_factory().make()
        for xi in self.data:
            scalar_acc.update(xi, 1.0, self.dist)

        seq_acc = self.est.accumulator_factory().make()
        enc = self.dist.dist_to_encoder().seq_encode(self.data)
        seq_acc.seq_update(enc, np.ones(len(self.data)), self.dist)

        fit_scalar = self.est.estimate(None, scalar_acc.value())
        fit_seq = self.est.estimate(None, seq_acc.value())
        self.assertTrue(np.allclose(fit_scalar.w, fit_seq.w))
        for c_scalar, c_seq in zip(fit_scalar.components, fit_seq.components):
            self.assertAlmostEqual(c_scalar.mu, c_seq.mu, places=10)
            self.assertAlmostEqual(c_scalar.sigma2, c_seq.sigma2, places=10)

    def test_string_representation_keeps_name_intact(self) -> None:
        dist = SemiSupervisedMixtureDistribution(self.dist.components, [0.5, 0.5], name="semi")
        dist_str = str(dist)

        self.assertIn("name='semi'", dist_str)
        self.assertNotIn("name=',s,e,m,i,'", dist_str)

    def test_duplicate_prior_labels_are_aggregated(self) -> None:
        duplicate = (1.5, [(0, 0.2), (0, 0.3), (1, 0.5)])
        combined = (1.5, [(0, 0.5), (1, 0.5)])

        self.assertAlmostEqual(self.dist.log_density(duplicate), self.dist.log_density(combined), places=12)
        np.testing.assert_allclose(self.dist.posterior(duplicate), self.dist.posterior(combined), rtol=1e-12)

        enc = self.dist.dist_to_encoder().seq_encode([duplicate, combined])
        log_density = self.dist.seq_log_density(enc)
        posterior = self.dist.seq_posterior(enc)

        self.assertAlmostEqual(log_density[0], log_density[1], places=12)
        np.testing.assert_allclose(posterior[0], posterior[1], rtol=1e-12)

    def test_enumerator_fails_fast_with_reason(self) -> None:
        with self.assertRaises(EnumerationError) as ctx:
            self.dist.enumerator()
        self.assertIn("prior", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
