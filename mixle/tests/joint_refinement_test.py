"""C3: responsibility-weighted encoder fine-tuning inside EM — the canonical pool block.

A mixture of neural experts: the E-step computes responsibilities, each expert's M-step is a
responsibility-weighted gradient step. The certificate marks exactly those gradient M-steps as the
pool-eligible blocks (HEURISTIC), while the mixture's own EM step is STATIONARY -- so the graph shows
which piece a GPU pool would take and which stays local.
"""

import unittest

import numpy as np

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

if _HAS_TORCH:
    import mixle.stats as st
    from mixle.inference import certify, optimize
    from mixle.models.neural_leaf import NeuralGaussian


def _mlp():
    return torch.nn.Sequential(torch.nn.Linear(1, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1))


def _two_regime(seed, n=600):
    """y | x has two branches: +3x and -3x, mixed 50/50 -- a single expert cannot fit both."""
    r = np.random.RandomState(seed)
    rows = []
    for _ in range(n):
        x = r.uniform(-1, 1)
        y = (3 * x if r.rand() < 0.5 else -3 * x) + 0.3 * r.randn()
        rows.append(((float(x),), (float(y),)))
    return rows


def _ll(model, data):
    return float(np.sum(model.seq_log_density(model.dist_to_encoder().seq_encode(data))))


@unittest.skipUnless(_HAS_TORCH, "requires torch")
class JointRefinementTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        np.random.seed(0)
        self.train = _two_regime(0)
        self.test = _two_regime(1)
        est = st.MixtureEstimator(
            [
                NeuralGaussian(_mlp(), lr=5e-3, m_steps=30).estimator(),
                NeuralGaussian(_mlp(), lr=5e-3, m_steps=30).estimator(),
            ]
        )
        init = st.MixtureDistribution([NeuralGaussian(_mlp()), NeuralGaussian(_mlp())], [0.5, 0.5])
        self.mixture = optimize(self.train, est, prev_estimate=init, max_its=6, out=None)

    def test_certificate_is_heuristic_with_pool_eligible_gradient_blocks(self):
        cert = certify(self.mixture)
        self.assertEqual(cert.guarantee.name, "HEURISTIC")  # capped by the neural M-steps
        self.assertEqual(len(cert.gradient_blocks), 2)  # one per expert
        self.assertTrue(all(b.placement == "pool_eligible" for b in cert.gradient_blocks))
        self.assertIn("required gradient descent", cert.why_not_adam())

    def test_two_experts_beat_a_single_expert_on_held_out(self):
        # responsibility-weighted specialization pays off: two experts capture both branches
        torch.manual_seed(1)
        single = optimize(self.train, NeuralGaussian(_mlp(), lr=5e-3, m_steps=180).estimator(), max_its=1, out=None)
        self.assertGreater(_ll(self.mixture, self.test), _ll(single, self.test))


if __name__ == "__main__":
    unittest.main()
