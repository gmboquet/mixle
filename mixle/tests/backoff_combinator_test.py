"""Contracts for the backoff combinator: unseen outcomes stay scorable, and the pin actually pins."""

import io
import unittest

import numpy as np

from mixle.inference import optimize
from mixle.stats import (
    BackoffDataEncoder,
    BackoffDistribution,
    BackoffEstimator,
    CategoricalDistribution,
    IntegerCategoricalDistribution,
    IntegerCategoricalEstimator,
    PoissonDistribution,
    PoissonEstimator,
)
from mixle.stats.compute.pdist import DensitySemantics


def _sparse_base() -> IntegerCategoricalDistribution:
    """A support with a hole at every value except 11 and 21, both inside [1, 21]."""
    return IntegerCategoricalDistribution(1, [0.0] * 10 + [0.5] + [0.0] * 9 + [0.5])


class BackoffScoringTest(unittest.TestCase):
    def test_unseen_outcomes_are_finite_where_the_base_alone_is_not(self):
        base = _sparse_base()
        dist = BackoffDistribution(base, PoissonDistribution(16.0), escape_weight=0.01)
        for unseen in (17, 84):  # a hole inside the support, and a value past its maximum
            with self.subTest(x=unseen):
                self.assertEqual(base.log_density(unseen), -np.inf)
                self.assertTrue(np.isfinite(dist.log_density(unseen)))
        # An observed value is barely moved. It is bracketed: above the retained base alone, because
        # the fallback puts a little mass there too, and below the unmixed base, because some mass left.
        observed = dist.log_density(11)
        self.assertGreater(observed, float(np.log(0.5) + np.log1p(-0.01)))
        self.assertLess(observed, float(np.log(0.5)))
        self.assertLess(float(np.log(0.5)) - observed, 0.01)

    def test_zero_escape_weight_is_inert_rather_than_wrong(self):
        base = _sparse_base()
        dist = BackoffDistribution(base, PoissonDistribution(16.0), escape_weight=0.0)
        self.assertEqual(dist.log_density(11), base.log_density(11))
        self.assertEqual(dist.log_density(17), -np.inf)

    def test_sequence_and_scalar_paths_agree(self):
        dist = BackoffDistribution(_sparse_base(), PoissonDistribution(16.0), escape_weight=0.01)
        xs = [11, 21, 17, 84]
        encoded = dist.dist_to_encoder().seq_encode(xs)
        scalar = np.asarray([dist.log_density(x) for x in xs], dtype=np.float64)
        np.testing.assert_allclose(np.asarray(dist.seq_log_density(encoded), dtype=np.float64), scalar)

    def test_encoder_equality_follows_both_children(self):
        dist = BackoffDistribution(_sparse_base(), PoissonDistribution(16.0), escape_weight=0.01)
        self.assertEqual(dist.dist_to_encoder(), dist.dist_to_encoder())
        self.assertNotEqual(dist.dist_to_encoder(), BackoffDataEncoder(object(), object()))

    def test_invalid_escape_weight_is_rejected(self):
        base = _sparse_base()
        for bad in (1.5, -0.1, float("nan")):
            with self.subTest(escape_weight=repr(bad)), self.assertRaisesRegex(ValueError, "escape_weight"):
                BackoffDistribution(base, PoissonDistribution(16.0), escape_weight=bad)


class BackoffFitTest(unittest.TestCase):
    """The floor is the load-bearing part; without it EM collapses the escape branch on step one."""

    def setUp(self) -> None:
        self.data = list(np.random.RandomState(0).poisson(16, size=400).astype(int))

    def _fit(self, **kwargs) -> BackoffDistribution:
        est = BackoffEstimator(IntegerCategoricalEstimator(min_val=1, max_val=30), PoissonEstimator(), **kwargs)
        return optimize(self.data, est, max_its=25, out=io.StringIO())

    def test_em_never_drives_the_escape_weight_below_its_floor(self):
        # Zero is absorbing: at w == 0 the fallback's responsibility is 0 for every row, so
        # escape_count can never recover and the fit silently loses the ability to score anything new.
        fitted = self._fit(escape_weight=0.01, max_escape_weight=0.05)
        self.assertGreaterEqual(fitted.escape_weight, 0.01)
        self.assertLessEqual(fitted.escape_weight, 0.05)

    def test_a_fitted_model_still_scores_values_absent_from_its_training_sample(self):
        fitted = self._fit(escape_weight=0.01, max_escape_weight=0.05)
        absent = sorted(set(range(1, 61)) - set(self.data))
        self.assertTrue(absent, "the sample was expected to leave some values unobserved")
        for x in absent[:5]:
            with self.subTest(x=x):
                self.assertTrue(np.isfinite(fitted.log_density(x)))

    def test_freezing_the_weight_returns_it_exactly(self):
        self.assertAlmostEqual(self._fit(escape_weight=0.02, max_escape_weight=0.02).escape_weight, 0.02)

    def test_an_unbounded_weight_becomes_model_selection_not_smoothing(self):
        """Documents why the ceiling exists rather than asserting the ceiling is optional."""
        capped = self._fit(escape_weight=0.01, max_escape_weight=0.05)
        free = self._fit(escape_weight=0.01, max_escape_weight=None)
        self.assertGreater(free.escape_weight, capped.escape_weight * 2.0)

    def test_a_starting_weight_outside_its_own_bound_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "exceeds max_escape_weight"):
            BackoffEstimator(
                IntegerCategoricalEstimator(min_val=1, max_val=30),
                PoissonEstimator(),
                escape_weight=0.5,
                max_escape_weight=0.05,
            )


class DensitySemanticsTest(unittest.TestCase):
    """A backoff inherits its children's status rather than conferring one (MXR-080-1844)."""

    def test_two_exact_children_stay_exact(self):
        backoff = BackoffDistribution(PoissonDistribution(1.0), PoissonDistribution(2.0))
        self.assertIs(backoff.density_semantics(), DensitySemantics.EXACT)

    def test_a_likelihood_factor_child_makes_the_backoff_a_factor(self):
        factor = CategoricalDistribution({"a": 0.5, "b": 0.5}, scoring_only=True)
        self.assertIs(factor.density_semantics(), DensitySemantics.LIKELIHOOD_FACTOR)
        backoff = BackoffDistribution(factor, CategoricalDistribution({"a": 0.5, "b": 0.5}))
        # The escape-weighted sum of an exact law and an unnormalized factor is itself unnormalized.
        self.assertIs(backoff.density_semantics(), DensitySemantics.LIKELIHOOD_FACTOR)

    def test_the_factor_may_be_either_child(self):
        factor = CategoricalDistribution({"a": 0.5, "b": 0.5}, scoring_only=True)
        backoff = BackoffDistribution(CategoricalDistribution({"a": 0.5, "b": 0.5}), factor)
        self.assertIs(backoff.density_semantics(), DensitySemantics.LIKELIHOOD_FACTOR)


class SampleCountTest(unittest.TestCase):
    """``sample(size)`` takes a count, not anything ``int()`` will swallow (MXR-080-1845)."""

    def _sampler(self):
        return BackoffDistribution(
            CategoricalDistribution({"a": 0.5, "b": 0.5}), CategoricalDistribution({"a": 0.9, "b": 0.1})
        ).sampler(seed=0)

    def test_an_exact_count_draws_that_many(self):
        self.assertEqual(len(self._sampler().sample(3)), 3)

    def test_none_still_draws_a_single_value(self):
        self.assertIsInstance(self._sampler().sample(), str)

    def test_a_fractional_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "exact non-negative integer"):
            self._sampler().sample(2.7)  # silently became two draws

    def test_a_boolean_count_is_rejected(self):
        with self.assertRaisesRegex(TypeError, "exact non-negative integer"):
            self._sampler().sample(True)  # silently became one draw

    def test_a_negative_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "exact non-negative integer"):
            self._sampler().sample(-3)  # silently became an empty "successful" sample


class ChildScoreRankTest(unittest.TestCase):
    """A batch score is one log-density per observation, whatever the children agree on."""

    class _TwoDimensional:
        """A child that returns a matrix instead of a vector."""

        def log_density(self, x):
            return 0.0

        def seq_log_density(self, enc):
            return np.zeros((2, 2))

        def sampler(self, seed=None):
            return None

        def estimator(self, pseudo_count=None):
            return None

        def dist_to_encoder(self):
            return None

        def density_semantics(self):
            return DensitySemantics.EXACT

    def test_two_children_agreeing_on_a_matrix_are_still_rejected(self):
        # Equal shapes were the whole check, so (2, 2) against (2, 2) passed and produced a
        # two-dimensional "log-density" (MXR-080-1843).
        backoff = BackoffDistribution(self._TwoDimensional(), self._TwoDimensional())
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            backoff.seq_log_density(("enc", "enc"))

    class _DropsARow:
        """A child that silently returns one score for a two-row batch."""

        def log_density(self, x):
            return 0.0

        def seq_log_density(self, enc):
            return np.zeros(1)

        def sampler(self, seed=None):
            return None

        def estimator(self, pseudo_count=None):
            return None

        def dist_to_encoder(self):
            return None

        def density_semantics(self):
            return DensitySemantics.EXACT

    def test_both_children_dropping_the_same_row_is_still_rejected(self):
        # Agreement between the children is not enough: both can drop the SAME row and return a
        # matching shorter vector, which re-aligns every score with the wrong observation. The count
        # recorded at encode time ties the answer to the question (MXR-080-1843).
        backoff = BackoffDistribution(self._DropsARow(), self._DropsARow())
        with self.assertRaisesRegex(ValueError, "score.* for 2 encoded observation"):
            backoff.seq_log_density(("enc", "enc", 2))

    def test_the_row_count_travels_with_a_real_encoding(self):
        dist = BackoffDistribution(_sparse_base(), PoissonDistribution(16.0), escape_weight=0.01)
        xs = [11, 21, 17]
        encoded = dist.dist_to_encoder().seq_encode(xs)
        self.assertEqual(encoded[2], len(xs))
        self.assertEqual(len(dist.seq_log_density(encoded)), len(xs))

    def test_the_degenerate_escape_weights_check_too(self):
        # w == 0 and w == 1 short-circuit past the two-child comparison entirely.
        for weight in (0.0, 1.0):
            backoff = BackoffDistribution(self._TwoDimensional(), self._TwoDimensional(), escape_weight=weight)
            with self.assertRaisesRegex(ValueError, "one-dimensional"):
                backoff.seq_log_density(("enc", "enc"))


if __name__ == "__main__":
    unittest.main()
