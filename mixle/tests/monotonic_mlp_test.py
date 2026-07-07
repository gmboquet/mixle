"""make_monotonic_mlp (mixle.models.neural): a Torch MLP monotonic in every input by construction, and its
composition with the existing NeuralGaussian regression wrapper."""

import unittest

import numpy as np

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class MakeMonotonicMlpTest(unittest.TestCase):
    def test_increasing_by_construction_at_random_init(self):
        # the monotonicity guarantee comes from the non-negative-weight architecture, not from training --
        # checked here at several random, untrained initializations.
        from mixle.models.neural import make_monotonic_mlp

        for seed in range(5):
            torch.manual_seed(seed)
            module = make_monotonic_mlp(3, [16, 16], 1)
            x = torch.randn(100, 3, requires_grad=True)
            y = module(x)
            (grad,) = torch.autograd.grad(y.sum(), x)
            self.assertGreaterEqual(float(grad.min()), 0.0)

    def test_decreasing_flag_negates_the_gradient_sign(self):
        from mixle.models.neural import make_monotonic_mlp

        torch.manual_seed(0)
        module = make_monotonic_mlp(2, [16], 1, increasing=False)
        x = torch.randn(100, 2, requires_grad=True)
        y = module(x)
        (grad,) = torch.autograd.grad(y.sum(), x)
        self.assertLessEqual(float(grad.max()), 0.0)

    def test_monotone_1d_curve_is_actually_nondecreasing_pointwise(self):
        from mixle.models.neural import make_monotonic_mlp

        torch.manual_seed(1)
        module = make_monotonic_mlp(1, [8, 8], 1)
        grid = torch.linspace(-5.0, 5.0, 200).reshape(-1, 1)
        with torch.no_grad():
            y = module(grid).flatten().numpy()
        self.assertTrue(np.all(np.diff(y) >= -1e-6))

    def test_invalid_dims_raise(self):
        from mixle.models.neural import make_monotonic_mlp

        with self.assertRaises(ValueError):
            make_monotonic_mlp(0, [8], 1)
        with self.assertRaises(ValueError):
            make_monotonic_mlp(2, [0], 1)

    def test_fits_a_monotone_regression_target_through_neural_gaussian(self):
        from mixle.models.neural import make_monotonic_mlp
        from mixle.models.neural_leaf import NeuralGaussian

        torch.manual_seed(0)
        rng = np.random.RandomState(0)
        x = rng.uniform(-2, 2, 200).astype("float32")
        y = (2.0 * x + 0.05 * rng.randn(200)).astype("float32")  # a monotone increasing target
        data = list(zip(x[:, None], y[:, None]))

        model = NeuralGaussian(make_monotonic_mlp(1, [32, 32], 1), noise=1.0, m_steps=800, lr=0.03)
        est = model.estimator()
        acc = est.accumulator_factory().make()
        enc = model.dist_to_encoder().seq_encode(data)
        acc.seq_update(enc, np.ones(len(data)), model)
        fitted = est.estimate(None, acc.value())

        pred = fitted._forward(x[:, None])[:, 0]
        self.assertLess(float(np.mean((pred - y) ** 2)), 1.0)  # tracks y = 2x well (data variance itself ~0.0025)
        # the fit is monotone by construction even though the fitting loop never checked for it
        order = np.argsort(x)
        self.assertTrue(np.all(np.diff(pred[order]) >= -1e-4))


if __name__ == "__main__":
    unittest.main()
