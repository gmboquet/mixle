"""WS-7: mcmc_summary -- the per-parameter posterior + convergence table."""

import unittest

import numpy as np

from pysp.inference import ess_bulk, mcmc_summary, rhat_max


class MCMCSummaryTest(unittest.TestCase):
    def test_fields_and_values(self):
        rng = np.random.RandomState(0)
        offsets = np.array([0.0, 5.0, -2.0])
        chains = rng.randn(4, 2000, 3) + offsets
        s = mcmc_summary(chains)
        self.assertEqual(len(s), 3)
        flat = chains.reshape(8000, 3)
        rh, eb = rhat_max(chains), ess_bulk(chains)
        for k, row in enumerate(s):
            self.assertAlmostEqual(row["mean"], float(flat[:, k].mean()), places=9)
            self.assertAlmostEqual(row["sd"], float(flat[:, k].std(ddof=1)), places=9)
            self.assertAlmostEqual(row["q50"], float(np.quantile(flat[:, k], 0.5)), places=9)
            self.assertAlmostEqual(row["r_hat"], float(rh[k]), places=9)  # = rhat_max
            self.assertAlmostEqual(row["ess_bulk"], float(eb[k]), places=6)
            self.assertLess(row["q05"], row["q50"])
            self.assertLess(row["q50"], row["q95"])

    def test_converged_chains_report_good_diagnostics(self):
        s = mcmc_summary(np.random.RandomState(1).randn(4, 3000, 1))
        self.assertLess(s[0]["r_hat"], 1.01)
        self.assertGreater(s[0]["ess_bulk"], 5000)

    def test_one_dim_input(self):
        s = mcmc_summary(np.random.RandomState(2).randn(4, 1000))  # (chains, draws) -> 1 param
        self.assertEqual(len(s), 1)


if __name__ == "__main__":
    unittest.main()
