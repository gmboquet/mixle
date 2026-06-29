"""NeuralLeaf: a neural net as a mixle conditional-density leaf, composing into a mixture of experts (EM+grad)."""

import unittest

import numpy as np

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def _mlp(dims):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(torch.nn.Tanh())
    return torch.nn.Sequential(*layers)


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class NeuralLeafTest(unittest.TestCase):
    def test_fits_via_the_estimator_contract(self):
        from mixle.models.neural_leaf import NeuralLeaf

        rng = np.random.RandomState(0)
        x = rng.uniform(-2, 2, 200).astype("float32")
        y = (2 * x + 0.1 * rng.randn(200)).astype("float32")
        data = list(zip(x[:, None], y[:, None]))
        leaf = NeuralLeaf(_mlp([1, 16, 1]), noise=1.0, m_steps=150, lr=0.02)
        est = leaf.estimator()
        acc = est.accumulator_factory().make()
        enc = leaf.dist_to_encoder().seq_encode(data)
        acc.seq_update(enc, np.ones(len(data)), leaf)
        fitted = est.estimate(None, acc.value())
        self.assertLess(((fitted._forward(x[:, None])[:, 0] - 2 * x) ** 2).mean(), 0.05)
        self.assertLess(fitted.noise, 0.5)  # learned a small observation noise

    def test_mixture_of_neural_experts_specializes(self):
        from mixle.inference import estimate
        from mixle.models.neural_leaf import NeuralLeaf
        from mixle.stats import MixtureDistribution, MixtureEstimator

        rng = np.random.RandomState(0)
        z = rng.randint(0, 2, 400)
        x = rng.uniform(-2, 2, 400).astype("float32")
        y = (np.where(z == 0, 2 * x, -2 * x) + 0.1 * rng.randn(400)).astype("float32")  # two latent regimes
        data = list(zip(x[:, None], y[:, None]))
        la = NeuralLeaf(_mlp([1, 16, 1]), noise=1.0, m_steps=30, lr=0.03)
        lb = NeuralLeaf(_mlp([1, 16, 1]), noise=1.0, m_steps=30, lr=0.03)
        est = MixtureEstimator([la.estimator(), lb.estimator()])
        model = MixtureDistribution([la, lb], [0.5, 0.5])
        for _ in range(15):  # EM: responsibilities (E) + per-expert weighted-NLL gradient (M)
            model = estimate(data, est, model)
        pa = model.components[0]._forward(x[:, None])[:, 0]
        pb = model.components[1]._forward(x[:, None])[:, 0]
        err = min(  # experts specialize to +2x and -2x (either assignment)
            ((pa - 2 * x) ** 2).mean() + ((pb + 2 * x) ** 2).mean(),
            ((pa + 2 * x) ** 2).mean() + ((pb - 2 * x) ** 2).mean(),
        )
        self.assertLess(err, 0.2)


if __name__ == "__main__":
    unittest.main()
