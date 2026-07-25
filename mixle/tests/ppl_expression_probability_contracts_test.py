"""Release contracts for expression dependency, conditioning, and event probabilities."""

import unittest

import numpy as np

from mixle.ppl import MVN, Gamma, Normal, constrain, free
from mixle.ppl.core import Constraint, ProbabilityEstimate


class ExpressionDependencyContractTest(unittest.TestCase):
    def test_repeated_handle_reuses_one_draw_and_copy_is_explicit(self):
        x = Normal(0.0, 1.0, name="x")
        np.testing.assert_array_equal((x - x).sample(200, seed=3), np.zeros(200))
        doubled = (x + x).sample(6000, seed=4)
        independent = (x + x.independent()).sample(6000, seed=4)
        self.assertAlmostEqual(float(np.var(doubled)), 4.0, delta=0.25)
        self.assertAlmostEqual(float(np.var(independent)), 2.0, delta=0.2)
        self.assertAlmostEqual((x + x).log_prob(0.3), Normal(0.0, 2.0).log_prob(0.3))

    def test_has_free_recurses_through_priors_composites_and_expressions(self):
        nested = Normal(Normal(free, 1.0), 1.0)
        expression = nested + Normal(0.0, 1.0)
        self.assertTrue(nested.has_free)
        self.assertTrue(expression.has_free)
        self.assertFalse(Normal(0.0, 1.0).has_free)

    def test_unsupported_derived_density_never_substitutes_kde(self):
        with self.assertRaisesRegex(NotImplementedError, "no registered exact convolution"):
            (Normal(0.0, 1.0) + Gamma(2.0, 1.0)).log_prob(1.0)
        with self.assertRaisesRegex(NotImplementedError, "exact log-density"):
            (Normal(0.0, 1.0) * Normal(0.0, 1.0)).log_prob(1.0)


class ConditioningContractTest(unittest.TestCase):
    def test_impossible_rejection_is_bounded_and_diagnostic(self):
        x = Normal(0.0, 1.0)
        impossible = x > 1.0e9
        with self.assertRaisesRegex(RuntimeError, "accepted=0"):
            x.given(impossible).sample(1, seed=1, max_attempts=2)
        with self.assertRaisesRegex(RuntimeError, "accepted=0"):
            constrain(impossible).sample(1, seed=1, max_attempts=2)

    def test_vector_event_reduction_is_explicit_per_row(self):
        x = MVN(2, mean=np.zeros(2), cov=np.eye(2))
        all_positive = x.given(x > 0.0)
        any_positive = x.given((x > 0.0).with_reduction("any"))
        all_estimate = all_positive.prob_of_event(samples=6000, seed=2)
        any_estimate = any_positive.prob_of_event(samples=6000, seed=2)
        self.assertAlmostEqual(float(all_estimate), 0.25, delta=0.04)
        self.assertAlmostEqual(float(any_estimate), 0.75, delta=0.04)
        self.assertTrue(np.all(all_positive.sample(100, seed=3) > 0.0))
        self.assertTrue(np.isfinite(all_positive.log_prob(np.array([1.0, 1.0]))))
        self.assertEqual(all_positive.log_prob(np.array([-1.0, 1.0])), -np.inf)

    def test_malformed_constraint_masks_are_rejected(self):
        x = Normal(0.0, 1.0)
        numeric = Constraint([x], lambda env: np.ones(len(env[x])), "numeric")
        wrong_rows = Constraint([x], lambda env: np.ones(1, dtype=bool), "wrong rows")
        with self.assertRaisesRegex(TypeError, "boolean"):
            x.given(numeric).sample(1, max_attempts=1)
        with self.assertRaisesRegex(ValueError, "expected"):
            x.given(wrong_rows).sample(1, max_attempts=1)

    def test_zero_hits_remain_zero_with_an_uncertainty_receipt_and_keyed_cache(self):
        x = Normal(0.0, 1.0)
        conditioned = x.given(x > 1.0e9)
        first = conditioned.prob_of_event(samples=1000, seed=5)
        second = conditioned.prob_of_event(samples=2000, seed=5)
        self.assertIsInstance(first, ProbabilityEstimate)
        self.assertEqual(float(first), 0.0)
        self.assertEqual(first.hits, 0)
        self.assertGreater(first.upper, 0.0)
        self.assertNotEqual(first.trials, second.trials)
        with self.assertRaisesRegex(RuntimeError, "unresolved"):
            conditioned.log_prob(0.0)


class PredictiveRngContractTest(unittest.TestCase):
    def test_point_prediction_consumes_the_caller_rng(self):
        model = Normal(0.0, 1.0)
        first = model.predict(20, rng=np.random.RandomState(9))
        second = model.predict(20, rng=np.random.RandomState(9))
        third = model.predict(20, rng=np.random.RandomState(10))
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, third))


if __name__ == "__main__":
    unittest.main()
