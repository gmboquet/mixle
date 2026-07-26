"""Backend capability decline must stay distinct from scorer execution failure."""

import unittest
from unittest import mock

import numpy as np

from mixle.engines import NUMPY_ENGINE
from mixle.stats.compute.backend import (
    BackendCapabilityUnavailableError,
    backend_seq_log_density,
)
from mixle.stats.compute.kernel import GenericKernel


class _LegacyOnlyDistribution:
    def __init__(self):
        self.legacy_calls = 0

    def seq_log_density(self, enc):
        self.legacy_calls += 1
        return np.asarray(enc, dtype=float) + 100.0


class BackendFailureContractTest(unittest.TestCase):
    def test_missing_generated_capability_uses_typed_decline_and_numpy_fallback(self):
        dist = _LegacyOnlyDistribution()
        with mock.patch(
            "mixle.stats.compute.declarations.generated_log_density_available",
            return_value=False,
        ):
            with self.assertRaises(BackendCapabilityUnavailableError):
                backend_seq_log_density(dist, [1.0], NUMPY_ENGINE)
            result = GenericKernel(dist, NUMPY_ENGINE).score([1.0])
        np.testing.assert_array_equal(result, [101.0])
        self.assertEqual(dist.legacy_calls, 1)

    def test_declared_scorer_execution_error_is_not_wrapped_or_fallen_back(self):
        dist = _LegacyOnlyDistribution()
        failure = ValueError("declared formula rejected input shape")
        with (
            mock.patch(
                "mixle.stats.compute.declarations.generated_log_density_available",
                return_value=True,
            ),
            mock.patch(
                "mixle.stats.compute.declarations.generated_log_density",
                side_effect=failure,
            ),
        ):
            with self.assertRaises(ValueError) as direct:
                backend_seq_log_density(dist, [[1.0, 2.0]], NUMPY_ENGINE)
            self.assertIs(direct.exception, failure)
            with self.assertRaises(ValueError) as through_kernel:
                GenericKernel(dist, NUMPY_ENGINE).score([[1.0, 2.0]])
            self.assertIs(through_kernel.exception, failure)
        self.assertEqual(dist.legacy_calls, 0)


if __name__ == "__main__":
    unittest.main()
