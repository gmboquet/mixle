"""Likelihood comparability contracts for automatic continuous-family selection."""

import math
import unittest

import numpy as np
from scipy import stats

from mixle.utils.automatic.factories import get_gamma_estimator, get_student_t_estimator
from mixle.utils.automatic.profiling import (
    _bic_penalty_bits,
    _gamma_bic_bits,
    _gamma_mle,
    _gamma_nll_bits,
    _student_t_bic_bits,
    _student_t_mle,
    _student_t_nll_bits,
)


class MaximumLikelihoodBicTest(unittest.TestCase):
    def test_gamma_bic_uses_likelihood_fit_not_moments(self):
        data = stats.gamma.rvs(0.7, scale=4.0, size=600, random_state=np.random.RandomState(12))
        shape, scale = _gamma_mle(data)
        mean = float(data.mean())
        variance = float(data.var())
        moment_scale = variance / mean
        moment_shape = mean / moment_scale

        mle_bits = _gamma_nll_bits(data, shape, scale)
        moment_bits = _gamma_nll_bits(data, moment_shape, moment_scale)
        self.assertLessEqual(mle_bits, moment_bits + 1.0e-10)
        self.assertAlmostEqual(
            _gamma_bic_bits(data, data.size),
            mle_bits + _bic_penalty_bits(2, data.size),
            places=10,
        )

    def test_student_t_bic_uses_likelihood_fit_not_kurtosis_moments(self):
        data = stats.t.rvs(4.5, loc=3.0, scale=1.7, size=800, random_state=np.random.RandomState(17))
        params = _student_t_mle(data)
        mean = float(data.mean())
        variance = float(data.var())
        excess = float(np.mean((data - mean) ** 4)) / (variance * variance) - 3.0
        moment_df = min(max(4.0 + 6.0 / excess, 2.5), 100.0)
        moment_scale = math.sqrt(variance * (moment_df - 2.0) / moment_df)

        mle_bits = _student_t_nll_bits(data, params)
        moment_bits = _student_t_nll_bits(data, (moment_df, mean, moment_scale))
        self.assertLessEqual(mle_bits, moment_bits + 1.0e-8)
        self.assertAlmostEqual(
            _student_t_bic_bits(data, data.size),
            mle_bits + _bic_penalty_bits(3, data.size),
            places=8,
        )


class FactoryFitSeedTest(unittest.TestCase):
    def test_gamma_factory_passes_correct_sufficient_statistics_and_mass(self):
        estimator = get_gamma_estimator({1.0: 1.0, 4.0: 3.0}, pseudo_count=2.5)
        self.assertEqual(estimator.pseudo_count, (2.5, 2.5))
        self.assertAlmostEqual(estimator.suff_stat[0], 3.25)
        self.assertAlmostEqual(estimator.suff_stat[1], 0.75 * math.log(4.0))

    def test_student_t_factory_preserves_requested_smoothing_mass(self):
        data = stats.t.rvs(5.0, size=200, random_state=np.random.RandomState(5))
        counts = {float(value): 1.0 for value in data}
        estimator = get_student_t_estimator(counts, pseudo_count=3.0)
        self.assertEqual(estimator.pseudo_count, 3.0)
        self.assertIsNotNone(estimator.suff_stat)
        self.assertTrue(math.isfinite(estimator.df))
        self.assertGreater(estimator.df, 0.0)


if __name__ == "__main__":
    unittest.main()
