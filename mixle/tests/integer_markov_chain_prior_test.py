"""IntegerMarkovChainEstimator.pseudo_count is an implicit conjugate prior, now exposed.

Regression test: pseudo_count-based additive smoothing (cond_mat += pseudo_count before
row-normalizing in estimate()) is the exact MAP point estimate under an independent
SymmetricDirichletDistribution(pseudo_count + 1) prior on each row of the conditional matrix, but
the estimator had no get_prior()/model_log_density() at all -- so
mixle.inference.estimation.optimize()'s objective auto-detection always resolved a pseudo_count-
regularized fit to plain 'mle', silently tracking the wrong (unpenalized) convergence quantity.
"""

import unittest

import numpy as np
from numpy.random import RandomState

from mixle.inference import seq_estimate, seq_initialize
from mixle.inference.estimation import _resolve_objective
from mixle.stats import IntegerMarkovChainDistribution, IntegerMarkovChainEstimator, seq_encode
from mixle.stats.bayes.symmetric_dirichlet import SymmetricDirichletDistribution


def _fit(pseudo_count, num_values=3, lag=1, seed=0, n=40):
    start = IntegerMarkovChainDistribution(
        num_values, np.full((num_values**lag, num_values), 1.0 / num_values), lag=lag
    )
    rng = RandomState(seed)
    data = [[int(v) for v in rng.randint(0, num_values, size=rng.randint(3, 8))] for _ in range(n)]
    est = IntegerMarkovChainEstimator(num_values, lag=lag, pseudo_count=pseudo_count)
    enc = seq_encode(data, model=start)
    init_model = seq_initialize(enc, est, RandomState(seed + 1), p=1.0)
    fitted = seq_estimate(enc, est, init_model)
    return est, fitted


class ImplicitDirichletPriorTestCase(unittest.TestCase):
    def test_no_pseudo_count_has_no_prior(self):
        est, fitted = _fit(pseudo_count=None)
        self.assertIsNone(est.get_prior())
        self.assertEqual(est.model_log_density(fitted), 0.0)
        self.assertEqual(_resolve_objective("auto", est, fitted), "mle")

    def test_pseudo_count_exposes_a_nonzero_implicit_prior(self):
        est, fitted = _fit(pseudo_count=0.1)
        self.assertIsInstance(est.get_prior(), SymmetricDirichletDistribution)
        self.assertAlmostEqual(est.get_prior().get_parameters(), 1.1)
        self.assertEqual(_resolve_objective("auto", est, fitted), "map")

    def test_model_log_density_matches_scipy_dirichlet_independently(self):
        """Cross-check against scipy.stats.dirichlet directly (not mixle's own DirichletDistribution/
        SymmetricDirichletDistribution) so this doesn't just assert self-consistency with whatever
        formula the implementation itself happens to use."""
        from scipy.stats import dirichlet

        pseudo_count = 0.25
        num_values, lag = 3, 1
        est, fitted = _fit(pseudo_count=pseudo_count, num_values=num_values, lag=lag)
        alpha = np.full(num_values, pseudo_count + 1.0)
        # Re-normalize each row to float64 exactly on the simplex because scipy's
        # dirichlet.logpdf rejects even small accumulated sum error.
        rows = np.asarray(fitted.cond_dist, dtype=np.float64)
        rows = rows / rows.sum(axis=1, keepdims=True)
        expected = float(sum(dirichlet.logpdf(row, alpha=alpha) for row in rows))
        self.assertAlmostEqual(est.model_log_density(fitted), expected, places=4)

    def test_larger_pseudo_count_scores_a_flatter_matrix_higher(self):
        """A bigger pseudo_count means a stronger pull toward uniform rows -- model_log_density
        should favor a flatter fitted matrix more as pseudo_count grows."""
        num_values, lag = 3, 1
        flat = IntegerMarkovChainDistribution(
            num_values, np.full((num_values**lag, num_values), 1.0 / num_values), lag=lag
        )
        peaked_row = np.array([0.9, 0.05, 0.05])
        peaked = IntegerMarkovChainDistribution(num_values, np.tile(peaked_row, (num_values**lag, 1)), lag=lag)

        small = IntegerMarkovChainEstimator(num_values, lag=lag, pseudo_count=0.1)
        large = IntegerMarkovChainEstimator(num_values, lag=lag, pseudo_count=5.0)

        # under a weak prior the peaked matrix isn't penalized much relative to flat; under a
        # strong prior pulling toward uniform, the gap in favor of flat must grow.
        gap_small = small.model_log_density(flat) - small.model_log_density(peaked)
        gap_large = large.model_log_density(flat) - large.model_log_density(peaked)
        self.assertGreater(gap_large, gap_small)


if __name__ == "__main__":
    unittest.main()
