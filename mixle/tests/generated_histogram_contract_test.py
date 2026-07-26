import unittest

import numpy as np

from mixle.engines import NUMPY_ENGINE
from mixle.stats import NegativeBinomialDistribution
from mixle.stats.compute.declarations import _weighted_histogram, generated_sufficient_statistics


class GeneratedHistogramContractTest(unittest.TestCase):
    def test_valid_exact_counts_match_legacy_histogram(self):
        result = _weighted_histogram(
            np.asarray([0, 1, 1, 4]),
            np.asarray([0.5, 1.0, 2.0, 0.25]),
            NUMPY_ENGINE,
        )
        self.assertEqual(result, {0: 0.5, 1: 3.0, 4: 0.25})

        dist = NegativeBinomialDistribution(3.0, 0.4)
        data = np.asarray([0, 1, 1, 4])
        enc = dist.dist_to_encoder().seq_encode(data)
        generated = generated_sufficient_statistics(
            dist,
            enc,
            np.asarray([0.5, 1.0, 2.0, 0.25]),
            NUMPY_ENGINE,
        )
        self.assertEqual(generated[2], result)

    def test_fractional_nonfinite_negative_and_overflow_counts_are_rejected(self):
        invalid_values = (
            np.asarray([0.0, 1.25]),
            np.asarray([0.0, np.nan]),
            np.asarray([0.0, np.inf]),
            np.asarray([0.0, -1.0]),
            np.asarray([0.0, float(2**63)]),
        )
        for values in invalid_values:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    _weighted_histogram(values, np.ones(2), NUMPY_ENGINE)

    def test_weight_geometry_and_domain_are_validated_before_aggregation(self):
        invalid_weights = (
            np.ones(1),
            np.ones((2, 1)),
            np.asarray([1.0, np.nan]),
            np.asarray([1.0, np.inf]),
            np.asarray([1.0, -0.5]),
        )
        for weights in invalid_weights:
            with self.subTest(weights=weights):
                with self.assertRaises(ValueError):
                    _weighted_histogram(np.asarray([0, 1]), weights, NUMPY_ENGINE)


if __name__ == "__main__":
    unittest.main()
