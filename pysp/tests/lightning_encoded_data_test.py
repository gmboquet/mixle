"""Tests for the Lightning encoded-data backend / mini-batch stochastic EM (WS-C2).

Lightning is optional; these tests skip when it (or torch) is not installed. The backend is always
*registered* (the factory lazy-imports Lightning only when used), which is checked unconditionally.
"""

import importlib.util
import unittest

import numpy as np
from numpy.random import RandomState

from pysp.planner import available_encoded_data_backends, encoded_data
from pysp.stats import GaussianDistribution, GaussianEstimator

HAS_LIGHTNING = importlib.util.find_spec("lightning") is not None and importlib.util.find_spec("torch") is not None


class LightningBackendRegistrationTest(unittest.TestCase):
    def test_backend_is_registered(self):
        # Registration must not require Lightning to be importable.
        self.assertIn("lightning", available_encoded_data_backends())


@unittest.skipUnless(HAS_LIGHTNING, "lightning/torch not installed")
class LightningEncodedDataTest(unittest.TestCase):
    def setUp(self):
        rng = RandomState(0)
        self.data = list(rng.normal(3.0, 2.0, size=500))
        self.estimator = GaussianEstimator()

    def test_full_em_matches_local_backend(self):
        prev = GaussianDistribution(0.0, 1.0)
        local = encoded_data(self.data, estimator=self.estimator, backend="local")
        light = encoded_data(self.data, estimator=self.estimator, backend="lightning")
        m_local = local.pysp_seq_estimate(self.estimator, prev)
        m_light = light.pysp_seq_estimate(self.estimator, prev)
        self.assertAlmostEqual(m_local.mu, m_light.mu, places=10)
        self.assertAlmostEqual(m_local.sigma2, m_light.sigma2, places=10)

    def test_minibatches_partition_the_data(self):
        handle = encoded_data(self.data, estimator=self.estimator, backend="lightning")
        batches = list(handle.minibatches())
        self.assertTrue(len(batches) >= 2)  # default batch_size = size//10
        self.assertEqual(sum(len(b) for b in batches), len(self.data))  # one epoch covers all rows

    def test_stochastic_em_recovers_gaussian(self):
        handle = encoded_data(self.data, estimator=self.estimator, backend="lightning")
        model = handle.stochastic_em(self.estimator, epochs=8, init_p=0.5, seed=1)
        self.assertAlmostEqual(float(model.mu), float(np.mean(self.data)), delta=0.5)
        self.assertGreater(float(model.sigma2), 0.0)


if __name__ == "__main__":
    unittest.main()
