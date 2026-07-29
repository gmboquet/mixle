"""G: the physics-inverse flagship — Bayesian parameter recovery with honest UQ + downgraded certificate."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
from flagship_physics_inverse import K_TRUE, infer_k, observe  # noqa: E402


class PhysicsInverseTest(unittest.TestCase):
    def test_posterior_recovers_the_rate(self):
        draws, _cert = infer_k(observe(0), 0, draws=800)
        self.assertLess(abs(draws.mean() - K_TRUE), 0.15)  # near the truth (up to noise-draw MLE shift)

    def test_certificate_downgrades_under_the_physics_potential(self):
        """The potential caps what the certificate may claim, and says so by name.

        The claim under test is the honest downgrade, and the certificate expresses it with two
        fields, not one: ``candidate_guarantee`` is what the block WOULD earn (STATIONARY, from its
        exponential-family shape) and ``guarantee`` is what it actually earns without a receipt.
        Asserting ``cert.guarantee == STATIONARY`` asked the aggregate to report the uncapped value
        -- i.e. asked for exactly the false claim the downgrade exists to prevent -- and the reason
        is worded "CANDIDATE CAPPED", never "DOWNGRADED". The mechanism was working the whole time.
        """
        _draws, cert = infer_k(observe(0), 0, draws=400)
        block = cert.blocks[0]
        self.assertEqual(block.candidate_guarantee.name, "STATIONARY")  # what it would earn
        self.assertEqual(block.guarantee.name, "UNVERIFIED")  # what it earns unreceipted
        self.assertEqual(cert.guarantee.name, "UNVERIFIED")  # the aggregate never exceeds a block
        self.assertIn("CANDIDATE CAPPED", block.reason)
        self.assertIn("custom potential", block.reason)  # the potential is named, never a false claim

    def test_interval_coverage_over_noise_draws(self):
        hits = 0
        for s in range(5):
            d, _ = infer_k(observe(s), s, draws=600)
            lo, hi = np.quantile(d, [0.05, 0.95])
            hits += int(lo <= K_TRUE <= hi)
        self.assertGreaterEqual(hits, 3)  # a 90% interval must bracket most of the time


if __name__ == "__main__":
    unittest.main()
