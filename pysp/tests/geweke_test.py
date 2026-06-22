"""WS-7: Geweke single-chain convergence z-score."""

import unittest

import numpy as np

from pysp.inference import geweke_z


class GewekeTest(unittest.TestCase):
    def test_stationary_chains_pass(self):
        rng = np.random.RandomState(0)
        self.assertLess(abs(geweke_z(rng.randn(5000))[0]), 2.0)  # iid
        ar = np.zeros(5000)
        for t in range(1, 5000):
            ar[t] = 0.7 * ar[t - 1] + rng.randn()
        self.assertLess(abs(geweke_z(ar)[0]), 2.0)  # stationary AR(1)

    def test_trending_chain_flagged(self):
        rng = np.random.RandomState(1)
        trend = 0.01 * np.arange(5000) + rng.randn(5000)
        self.assertGreater(abs(geweke_z(trend)[0]), 3.0)  # drift -> large |z|

    def test_multidim(self):
        self.assertEqual(geweke_z(np.random.RandomState(2).randn(3000, 4)).shape, (4,))


if __name__ == "__main__":
    unittest.main()
