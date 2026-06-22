"""WS-7: rank-normalized split-R-hat, folded-R-hat, MCSE, and bulk/tail ESS (Vehtari et al. 2021)."""

import unittest

import numpy as np

from pysp.inference import ess_bulk, ess_tail, folded_split_rhat, mcse_mean, rhat_max, split_rhat


class MCMCDiagnosticsTest(unittest.TestCase):
    def test_well_mixed_iid_chains(self):
        iid = np.random.RandomState(0).randn(4, 3000)
        self.assertLess(split_rhat(iid)[0], 1.01)  # converged
        self.assertGreater(ess_bulk(iid)[0], 9000)  # near the 12000 total
        self.assertGreater(ess_tail(iid)[0], 5000)

    def test_autocorrelation_reduces_ess(self):
        rng = np.random.RandomState(1)
        ar = np.zeros((4, 3000))
        for c in range(4):
            for t in range(1, 3000):
                ar[c, t] = 0.9 * ar[c, t - 1] + rng.randn()
        # AR(1) rho=0.9 -> ESS ~ total*(1-rho)/(1+rho) ~ 632, far below 12000
        self.assertLess(ess_bulk(ar)[0], 2000)
        self.assertLess(split_rhat(ar)[0], 1.05)  # stationary, so still "converged"

    def test_non_convergence_is_flagged(self):
        rng = np.random.RandomState(2)
        bad = np.concatenate([rng.randn(2, 3000), rng.randn(2, 3000) + 5.0], axis=0)
        self.assertGreater(split_rhat(bad)[0], 1.2)  # chains at different means

    def test_folded_rhat_catches_scale_mismatch(self):
        rng = np.random.RandomState(4)
        # same mean (0) but different scales: plain split-R-hat misses it, folded catches it
        mixed = np.concatenate([rng.randn(2, 3000), 3.0 * rng.randn(2, 3000)], axis=0)
        self.assertLess(split_rhat(mixed)[0], 1.05)
        self.assertGreater(folded_split_rhat(mixed)[0], 1.1)
        self.assertEqual(rhat_max(mixed)[0], max(split_rhat(mixed)[0], folded_split_rhat(mixed)[0]))

    def test_mcse_mean(self):
        rng = np.random.RandomState(5)
        iid = rng.randn(4, 3000)
        self.assertAlmostEqual(mcse_mean(iid)[0], 1.0 / np.sqrt(12000), delta=0.003)  # sd/sqrt(ess)
        ar = np.zeros((4, 3000))
        for c in range(4):
            for t in range(1, 3000):
                ar[c, t] = 0.9 * ar[c, t - 1] + rng.randn()
        self.assertGreater(mcse_mean(ar)[0], mcse_mean(iid)[0])  # autocorrelation inflates MCSE

    def test_multidimensional(self):
        chains = np.random.RandomState(3).randn(4, 2000, 3)
        for fn in (split_rhat, ess_bulk, ess_tail, folded_split_rhat, rhat_max, mcse_mean):
            self.assertEqual(fn(chains).shape, (3,))


if __name__ == "__main__":
    unittest.main()
