import unittest

import numpy as np

from pysp.bstats.dirichlet import DirichletDistribution as BayesianDirichletDistribution
from pysp.bstats.dirichlet import DirichletEstimator as BayesianDirichletEstimator
from pysp.bstats.gamma import GammaEstimator as BayesianGammaEstimator
from pysp.stats.dirichlet import DirichletDistribution, DirichletEstimator
from pysp.stats.gamma import GammaDistribution, GammaEstimator


class GammaEstimatorStabilityTestCase(unittest.TestCase):
    def assert_valid_gamma(self, dist, mean):
        self.assertTrue(np.isfinite(dist.k))
        self.assertTrue(np.isfinite(dist.theta))
        self.assertGreater(dist.k, 0.0)
        self.assertGreater(dist.theta, 0.0)
        self.assertAlmostEqual(dist.k * dist.theta, mean, delta=max(1.0e-12, abs(mean) * 1.0e-10))

    def test_stats_gamma_nearly_degenerate_data_stays_finite(self):
        for value in (1.0e-9, 2.0, 1.0e9):
            data = np.full(40, value)
            ss = (len(data), float(data.sum()), float(np.log(data).sum()))
            self.assert_valid_gamma(GammaEstimator().estimate(None, ss), value)

    def test_stats_gamma_low_variance_sample_stays_finite(self):
        data = 3.0 + np.linspace(-1.0e-10, 1.0e-10, 50)
        ss = (len(data), float(data.sum()), float(np.log(data).sum()))
        self.assert_valid_gamma(GammaEstimator().estimate(None, ss), float(data.mean()))

    def test_bstats_gamma_nearly_degenerate_data_stays_finite(self):
        for value in (1.0e-9, 2.0, 1.0e9):
            data = np.full(40, value)
            ss = (len(data), float(data.sum()), float(np.log(data).sum()))
            self.assert_valid_gamma(BayesianGammaEstimator().estimate(ss), value)

    def test_gamma_nonpositive_density_is_zero_not_nan(self):
        dist = GammaDistribution(2.0, 3.0)
        self.assertEqual(dist.density(0.0), 0.0)
        self.assertEqual(dist.log_density(0.0), -np.inf)


class DirichletEstimatorStabilityTestCase(unittest.TestCase):
    def assert_valid_dirichlet(self, dist):
        self.assertTrue(np.all(np.isfinite(dist.alpha)))
        self.assertTrue(np.all(dist.alpha > 0.0))

    def dirichlet_suff_stat(self, data, bayesian=False):
        if bayesian:
            enc = BayesianDirichletDistribution(np.ones(len(data[0]))).seq_encode(data)
        else:
            enc = DirichletDistribution(np.ones(len(data[0]))).dist_to_encoder().seq_encode(data)
        return len(data), enc[0].sum(axis=0), enc[1].sum(axis=0), enc[2].sum(axis=0)

    def test_stats_dirichlet_zero_count_returns_valid_default(self):
        ss = (0.0, np.zeros(3), np.zeros(3), np.zeros(3))
        self.assert_valid_dirichlet(DirichletEstimator(dim=3).estimate(None, ss))

    def test_stats_dirichlet_zero_entries_stay_finite(self):
        data = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.999999, 1.0e-6, 0.0],
                [0.2, 0.3, 0.5],
            ]
        )
        self.assert_valid_dirichlet(DirichletEstimator(dim=3).estimate(None, self.dirichlet_suff_stat(data)))
        self.assert_valid_dirichlet(
            DirichletEstimator(dim=3, use_mpe=True).estimate(None, self.dirichlet_suff_stat(data))
        )

    def test_stats_dirichlet_pseudo_count_without_data_stays_finite(self):
        dist = DirichletDistribution([2.0, 3.0, 4.0])
        est = dist.estimator(pseudo_count=2.0)
        ss = (0.0, np.zeros(3), np.zeros(3), np.zeros(3))
        self.assert_valid_dirichlet(est.estimate(None, ss))

    def test_stats_dirichlet_low_variance_data_stays_finite(self):
        base = np.asarray([0.2, 0.3, 0.5])
        data = np.vstack([base + [0.0, 0.0, 0.0], base + [1.0e-12, -1.0e-12, 0.0]] * 20)
        self.assert_valid_dirichlet(DirichletEstimator(dim=3).estimate(None, self.dirichlet_suff_stat(data)))

    def test_bstats_dirichlet_zero_count_and_zero_entries_stay_finite(self):
        ss0 = (0.0, np.zeros(3), np.zeros(3), np.zeros(3))
        self.assert_valid_dirichlet(BayesianDirichletEstimator(dim=3).estimate(ss0))

        data = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.999999, 1.0e-6, 0.0],
                [0.2, 0.3, 0.5],
            ]
        )
        ss = self.dirichlet_suff_stat(data, bayesian=True)
        self.assert_valid_dirichlet(BayesianDirichletEstimator(dim=3).estimate(ss))
        self.assert_valid_dirichlet(BayesianDirichletEstimator(dim=3, use_mpe=True).estimate(ss))


if __name__ == "__main__":
    unittest.main()
