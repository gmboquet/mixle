"""Re-represent a NEURAL model as a tiny STRUCTURED one -- the capability the projections were built for.

Separated from ``project_test.py`` because this trains a torch normalizing flow (the exact numpy
projections stay in the fast gate; this joins the slow gate via conftest). The teacher is a real RealNVP
flow; the test asserts the projected Gaussian-mixture student is an order of magnitude smaller and keeps
the teacher's held-out likelihood (the target is well-specified for a mixture).
"""

import unittest

import numpy as np

from mixle.stats.latent.gaussian_mixture import GaussianMixtureDistribution


class NeuralToStructuredTest(unittest.TestCase):
    def test_projecting_a_flow_onto_a_gmm_retains_likelihood_and_shrinks(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        from mixle.inference import fit, moment_project
        from mixle.models.neural_density import NeuralDensity, build_coupling_flow

        rng = np.random.RandomState(0)
        torch.manual_seed(0)

        def blobs(n):  # a 4-blob ring: a GMM is well-specified, so projection should lose ~nothing
            ang = rng.choice(4, n) * (np.pi / 2)
            c = np.stack([3 * np.cos(ang), 3 * np.sin(ang)], axis=1)
            return list(c + rng.randn(n, 2) * 0.5)

        train, test = blobs(1500), blobs(600)
        module = build_coupling_flow(2, hidden=16, layers=4)
        flow_params = sum(p.numel() for p in module.parameters())
        teacher = fit(train, NeuralDensity(module).estimator(), max_its=15)

        # best of a few restarts, scored on held-out teacher samples (no test peeking)
        val = list(teacher.sampler(7).sample(800))
        best = None
        for s in range(3):
            g = moment_project(
                teacher,
                GaussianMixtureDistribution(np.zeros((8, 2)), np.stack([np.eye(2)] * 8), np.ones(8) / 8).estimator(),
                exact=False,
                n_samples=4000,
                seed=s,
                max_its=60,
            )
            v = float(-np.mean([g.log_density(x) for x in val]))
            if best is None or v < best[0]:
                best = (v, g)
        student = best[1]

        def nll(d):
            return float(-np.mean([d.log_density(x) for x in test]))

        gmm_params = 8 * (2 + 3) + 7  # means + full-cov uppers + free weights
        self.assertLess(gmm_params * 10, flow_params)  # the structured student is an order of magnitude smaller
        self.assertLess(nll(student), nll(teacher) + 0.4)  # and keeps the teacher's likelihood (well-specified)


if __name__ == "__main__":
    unittest.main()
