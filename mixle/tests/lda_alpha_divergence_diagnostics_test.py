"""Pins T4-02: LDAEstimator's default alpha solver diverges to infinity for realistic small
corpora, and raised a message ("...iteration_budget_exhausted...") that implied raising
max_alpha_iter would fix it, without naming either of the working escape hatches
(fixed_alpha=, or a loosened alpha_threshold=). update_alpha() now detects the case where the
corpus's mean expected log-topic-proportions cannot be matched by any finite Dirichlet alpha
(sum_k exp(mean_log_p_k) >= 1, a genuine non-existence condition, not a slow-convergence one)
and reports it distinctly, naming the remedies that actually work.
"""

import unittest

import numpy as np

from mixle.inference import optimize
from mixle.stats.latent.lda import LDAConvergenceError, LDAEstimator, update_alpha
from mixle.stats.univariate.discrete.categorical import CategoricalEstimator


class LDAAlphaDivergenceDiagnosticsTest(unittest.TestCase):
    def _estimator(self, **kwargs):
        return LDAEstimator([CategoricalEstimator(), CategoricalEstimator()], **kwargs)

    def test_default_two_word_two_topic_corpus_reports_divergence_not_slow_convergence(self):
        # The tester's minimal reproduction: a single short, skewed two-word document is enough
        # to make the E-step's gamma symmetric across topics, which pins mean_log_p exactly on
        # (or past) the boundary where no finite alpha exists.
        doc_a = [("a", 3.0), ("b", 1.0)]

        with self.assertRaises(LDAConvergenceError) as caught:
            optimize([doc_a], self._estimator(), out=None)

        diagnostics = caught.exception.diagnostics
        self.assertFalse(diagnostics.converged)
        self.assertEqual(diagnostics.termination_reason, "alpha_diverging")

        message = str(caught.exception)
        # The message must not leave "raise max_alpha_iter" as the implied remedy, and must name
        # both real escape hatches by their actual constructor keyword.
        self.assertIn("max_alpha_iter will not help", message)
        self.assertIn("fixed_alpha=", message)
        self.assertIn("alpha_threshold", message)

    def test_raising_max_alpha_iter_does_not_resolve_the_divergence(self):
        # Guards the message's own claim: a caller who (reasonably, given the old message) tries
        # a much larger budget must still fail, and still be told it's divergence, not a slow fit.
        doc_a = [("a", 3.0), ("b", 1.0)]
        est = self._estimator(max_alpha_iter=50_000)

        with self.assertRaises(LDAConvergenceError) as caught:
            optimize([doc_a], est, out=None)

        self.assertEqual(caught.exception.diagnostics.termination_reason, "alpha_diverging")

    def test_fixed_alpha_escape_hatch_named_in_the_message_actually_works(self):
        doc_a = [("a", 3.0), ("b", 1.0)]
        est = self._estimator(fixed_alpha=np.array([1.0, 1.0]))
        fitted = optimize([doc_a], est, out=None)
        np.testing.assert_array_equal(fitted.alpha, np.array([1.0, 1.0]))

    def test_loosened_alpha_threshold_escape_hatch_named_in_the_message_actually_works(self):
        doc_a = [("a", 3.0), ("b", 1.0)]
        est = self._estimator(alpha_threshold=1.0e-3)
        fitted = optimize([doc_a], est, out=None)
        self.assertTrue(np.all(np.isfinite(fitted.alpha)))
        self.assertTrue(np.all(fitted.alpha > 0.0))

    def test_well_separated_corpus_still_converges_normally_and_is_unaffected(self):
        # Regression guard: the new divergence check must not misclassify (or otherwise change
        # the numeric outcome of) an ordinary corpus whose alpha fixed point genuinely converges.
        # mean_log_p = [-1, -1] gives sum_k exp(mean_log_p_k) = 2/e ~= 0.736 < 1, comfortably
        # inside the feasible region -- the same convergent case already covered by
        # lda_contract_test.py's test_alpha_update_is_bounded_monotone_and_receipted.
        alpha, iterations, diagnostics = update_alpha(
            np.array([1.0, 1.0]),
            np.array([-1.0, -1.0]),
            1.0e-10,
            max_iter=100,
            return_diagnostics=True,
        )
        self.assertTrue(diagnostics.converged)
        self.assertEqual(diagnostics.termination_reason, "converged")
        self.assertEqual(iterations, diagnostics.iterations)
        self.assertTrue(np.all(alpha > 0.0))


if __name__ == "__main__":
    unittest.main()
