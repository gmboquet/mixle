"""Public fused kernels enforce distribution support on encoded payloads."""

import unittest

import numpy as np
from scipy.special import gammaln

from mixle.stats import (
    BernoulliDistribution,
    BetaDistribution,
    CompositeDistribution,
    ExponentialDistribution,
    GammaDistribution,
    GaussianDistribution,
    GeometricDistribution,
    HalfNormalDistribution,
    InverseGammaDistribution,
    InverseGaussianDistribution,
    LogSeriesDistribution,
    MixtureDistribution,
    PoissonDistribution,
    RayleighDistribution,
)
from mixle.stats.compute.fused_codegen import fused_accumulate, fused_seq_log_density
from mixle.stats.compute.fused_nested import fused_nested_seq_log_density
from mixle.utils.optional_deps import HAS_NUMBA


def _positive_log(values):
    return np.log(np.where(values > 0.0, values, 1.0))


def _flatten(value):
    if isinstance(value, (tuple, list)):
        parts = [_flatten(child) for child in value]
        return np.concatenate(parts) if parts else np.empty(0)
    return np.asarray(value, dtype=np.float64).reshape(-1)


@unittest.skipUnless(HAS_NUMBA, "fused kernels require numba")
class FusedSupportValidationTest(unittest.TestCase):
    def test_scalar_templates_reject_every_off_support_row(self):
        dists = (
            ExponentialDistribution(2.0),
            GeometricDistribution(0.4),
            BernoulliDistribution(0.3),
            PoissonDistribution(2.0),
            GammaDistribution(2.0, 1.0),
            HalfNormalDistribution(1.0),
            RayleighDistribution(1.0),
            InverseGaussianDistribution(1.0, 2.0),
            BetaDistribution(2.0, 3.0),
            InverseGammaDistribution(3.0, 2.0),
            LogSeriesDistribution(0.4),
        )
        valid = [1.0, 2.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 2.0]
        invalid = [-1.0, 1.5, 0.5, -1.0, 0.0, -1.0, -1.0, -1.0, 0.0, -1.0, 0.0]
        rows = [valid.copy() for _ in range(len(dists) + 1)]
        for index, value in enumerate(invalid):
            rows[index][index] = value
        columns = [np.asarray(column, dtype=np.float64) for column in zip(*rows)]
        poisson = columns[3]
        gamma = columns[4]
        half_normal = columns[5]
        rayleigh = columns[6]
        inverse_gaussian = columns[7]
        beta = columns[8]
        inverse_gamma = columns[9]
        log_series = columns[10]
        encoded = (
            columns[0],
            columns[1],
            columns[2],
            (poisson, gammaln(poisson + 1.0)),
            (gamma, _positive_log(gamma)),
            (half_normal, half_normal * half_normal),
            (rayleigh, rayleigh * rayleigh, _positive_log(rayleigh)),
            (
                inverse_gaussian,
                np.where(inverse_gaussian > 0.0, 1.0 / inverse_gaussian, -1.0),
                _positive_log(inverse_gaussian),
            ),
            (_positive_log(beta), np.log1p(-beta), beta, beta * beta),
            (
                _positive_log(inverse_gamma),
                np.where(inverse_gamma > 0.0, 1.0 / inverse_gamma, -1.0),
            ),
            (log_series, _positive_log(log_series)),
        )
        model = CompositeDistribution(dists)

        actual = fused_seq_log_density(model, encoded)
        self.assertTrue(np.all(np.isneginf(actual[:-1])))
        expected_valid = sum(dist.log_density(value) for dist, value in zip(dists, valid))
        self.assertAlmostEqual(actual[-1], expected_valid, places=12)

        suff = fused_accumulate(model, encoded, np.ones(len(rows)), parallel=False)
        accumulator = model.estimator().accumulator_factory().make()
        valid_encoded = model.dist_to_encoder().seq_encode([tuple(valid)])
        accumulator.seq_update(valid_encoded, np.ones(1), model)
        np.testing.assert_allclose(_flatten(suff), _flatten(accumulator.value()), rtol=1.0e-12, atol=1.0e-12)

    def test_geometric_boundary_parameter_avoids_zero_times_infinity(self):
        model = GeometricDistribution(1.0)
        encoded = np.asarray([1.0, 2.0])
        expected = np.asarray([model.log_density(value) for value in encoded])
        actual = fused_seq_log_density(model, encoded)
        np.testing.assert_array_equal(actual, expected)

    def test_nested_fusion_applies_the_same_support_guards(self):
        inner = MixtureDistribution(
            [ExponentialDistribution(1.0), ExponentialDistribution(2.0)],
            [0.5, 0.5],
        )
        model = CompositeDistribution((GaussianDistribution(0.0, 1.0), inner))
        encoded = (np.asarray([0.0, 1.0]), np.asarray([-1.0, 0.5]))
        actual = fused_nested_seq_log_density(model, encoded)
        self.assertTrue(np.isneginf(actual[0]))
        self.assertTrue(np.isfinite(actual[1]))


if __name__ == "__main__":
    unittest.main()
