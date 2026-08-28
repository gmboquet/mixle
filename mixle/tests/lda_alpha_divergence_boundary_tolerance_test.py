"""Pins T4-01: update_alpha()'s alpha-non-existence check (logsumexp(mean_log_p) >= 0.0, the
T4-02 fix) is exact at the idealized boundary but the full E-step's floating-point mean_log_p for
a genuinely-degenerate corpus lands a few ULPs to either side of that boundary depending on
incidental details like topic count. On an all-empty-document corpus (symmetric zero-evidence,
mean_log_p mathematically == [-ln(K)] * K exactly), K in {2, 4, 8} landed on the ">= 0" side and
were correctly reported as 'alpha_diverging', while K in {3, 5, 6} landed a hair below 0 and were
misclassified as 'iteration_budget_exhausted' -- silently dropping the two working escape hatches
(fixed_alpha=, alpha_threshold=) and implying more iterations would help, which is false: the
guidance's own probe (max_alpha_iter raised 1000 -> 200,000 for K=5) left the residual essentially
unchanged. update_alpha() now widens the boundary check by a small absolute tolerance
(_ALPHA_BOUNDARY_TOL) that absorbs this floating-point noise.
"""

import unittest

import numpy as np

from mixle.inference import optimize
from mixle.stats.latent.lda import LDAConvergenceError, LDAEstimator, update_alpha
from mixle.stats.univariate.discrete.categorical import CategoricalEstimator


class LDAAlphaDivergenceBoundaryToleranceTest(unittest.TestCase):
    def _estimator(self, num_topics, **kwargs):
        return LDAEstimator([CategoricalEstimator() for _ in range(num_topics)], **kwargs)

    def test_all_empty_document_corpus_reports_alpha_diverging_across_topic_count_sweep(self):
        # The tester's exact reproduction: an all-empty-document corpus is the symmetric
        # zero-evidence case that pins mean_log_p exactly on the non-existence boundary. Every one
        # of these topic counts is the same degenerate case mathematically and must be classified
        # identically -- pre-fix, only K in {2, 4, 8} were.
        for num_topics in (2, 3, 4, 5, 6, 8):
            with self.subTest(num_topics=num_topics):
                est = self._estimator(num_topics)
                with self.assertRaises(LDAConvergenceError) as caught:
                    optimize([[], [], []], est, out=None)

                diagnostics = caught.exception.diagnostics
                self.assertFalse(diagnostics.converged)
                self.assertEqual(diagnostics.termination_reason, "alpha_diverging")

                message = str(caught.exception)
                self.assertIn("max_alpha_iter will not help", message)
                self.assertIn("fixed_alpha=", message)
                self.assertIn("alpha_threshold", message)

    def test_boundary_noise_just_below_zero_is_still_classified_as_diverging(self):
        # Direct unit check on the detector itself: mean_log_p landing a few ULPs below the exact
        # boundary (as the real E-step pipeline does for K in {3, 5, 6}) must still be classified
        # as unreachable, not as ordinary slow convergence.
        for num_topics, epsilon in ((3, 2.78e-17), (5, 2.22e-16), (6, 1.11e-16)):
            with self.subTest(num_topics=num_topics):
                exact_boundary = np.full(num_topics, -np.log(num_topics))
                # Perturb one entry down slightly so logsumexp(mean_log_p) drops just under 0,
                # mirroring the sign of the observed floating-point noise.
                mean_log_p = exact_boundary.copy()
                mean_log_p[0] -= epsilon
                with self.assertRaises(LDAConvergenceError) as caught:
                    update_alpha(
                        np.ones(num_topics),
                        mean_log_p,
                        1.0e-8,
                        max_iter=1000,
                        return_diagnostics=True,
                    )
                diagnostics = caught.exception.diagnostics
                self.assertFalse(diagnostics.converged)
                self.assertEqual(diagnostics.termination_reason, "alpha_diverging")

    def test_well_separated_corpus_still_converges_normally_and_is_unaffected(self):
        # Regression guard (step 3): the widened boundary must not change the numeric outcome for
        # an ordinary corpus whose alpha fixed point genuinely converges, comfortably clear of the
        # non-existence boundary. mean_log_p = [-1, -1] gives sum_k exp(mean_log_p_k) = 2/e ~=
        # 0.736 < 1, ~0.31 away from the boundary in logsumexp terms -- five orders of magnitude
        # larger than _ALPHA_BOUNDARY_TOL (1e-9).
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
