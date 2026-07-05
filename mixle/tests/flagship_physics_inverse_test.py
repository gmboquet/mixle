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
        _draws, cert = infer_k(observe(0), 0, draws=400)
        self.assertEqual(cert.guarantee.name, "STATIONARY")
        self.assertIn("DOWNGRADED", cert.blocks[0].reason)  # the potential is named, never a false claim

    def test_interval_coverage_over_noise_draws(self):
        hits = 0
        for s in range(5):
            d, _ = infer_k(observe(s), s, draws=600)
            lo, hi = np.quantile(d, [0.05, 0.95])
            hits += int(lo <= K_TRUE <= hi)
        self.assertGreaterEqual(hits, 3)  # a 90% interval must bracket most of the time


if __name__ == "__main__":
    unittest.main()
