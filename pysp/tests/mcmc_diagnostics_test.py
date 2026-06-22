"""WS-7: rank-normalized split-R-hat and bulk/tail ESS (Vehtari et al. 2021)."""

import unittest

import numpy as np

from pysp.inference import ess_bulk, ess_tail, split_rhat


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

    def test_multidimensional(self):
        chains = np.random.RandomState(3).randn(4, 2000, 3)
        for fn in (split_rhat, ess_bulk, ess_tail):
            self.assertEqual(fn(chains).shape, (3,))


if __name__ == "__main__":
    unittest.main()
