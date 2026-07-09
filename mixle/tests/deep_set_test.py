"""make_deep_set (mixle.models.neural): a set-input network invariant to permutation of the set axis, by
construction, and its composition with the existing NeuralGaussian regression wrapper."""

import unittest

import numpy as np

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class MakeDeepSetTest(unittest.TestCase):
    def test_invariant_to_permutation_of_the_set_axis_at_random_init(self):
        # the invariance comes from the shared per-element phi + symmetric pool, not from training --
        # checked here at several random, untrained initializations and permutations.
        from mixle.models.neural import make_deep_set

        for seed in range(5):
            torch.manual_seed(seed)
            net = make_deep_set(3, [16], 8, [16], 2, pooling="mean")
            x = torch.randn(6, 9, 3)
            y = net(x)
            perm = torch.randperm(9)
            y_perm = net(x[:, perm, :])
            self.assertTrue(torch.allclose(y, y_perm, atol=1e-5))

    def test_invariant_under_sum_and_max_pooling_too(self):
        from mixle.models.neural import make_deep_set

        torch.manual_seed(0)
        x = torch.randn(4, 5, 2)
        for pooling in ("sum", "max"):
            net = make_deep_set(2, [8], 4, [8], 1, pooling=pooling)
            perm = torch.randperm(5)
            self.assertTrue(torch.allclose(net(x), net(x[:, perm, :]), atol=1e-5))

    def test_not_a_degenerate_constant_function(self):
        from mixle.models.neural import make_deep_set

        torch.manual_seed(0)
        net = make_deep_set(3, [16], 8, [16], 1)
        x = torch.randn(20, 5, 3)
        y = net(x)
        self.assertGreater(float(y.detach().std()), 1e-4)

    def test_invalid_args_raise(self):
        from mixle.models.neural import make_deep_set

        with self.assertRaises(ValueError):
            make_deep_set(0, [8], 4, [8], 1)
        with self.assertRaises(ValueError):
            make_deep_set(3, [8], 4, [8], 1, pooling="bogus")

    def test_fits_a_set_summary_target_with_an_ordinary_training_loop(self):
        # a permutation-invariant target (the set's element-wise mean) -- the network should learn to
        # reproduce it regardless of the order the elements arrive in. Trained directly (not through
        # NeuralGaussian's accumulator, which flattens the set axis -- see make_deep_set's docstring).
        from mixle.models.neural import make_deep_set

        torch.manual_seed(0)
        rng = np.random.RandomState(0)
        sets = rng.uniform(-2, 2, (300, 4, 2)).astype("float32")
        targets = sets.mean(axis=1)  # (300, 2): permutation-invariant by construction
        xt = torch.as_tensor(sets)
        yt = torch.as_tensor(targets)

        net = make_deep_set(2, [32], 16, [32], 2)
        opt = torch.optim.Adam(net.parameters(), lr=0.01)
        for _ in range(500):
            opt.zero_grad()
            loss = ((net(xt) - yt) ** 2).mean()
            loss.backward()
            opt.step()

        with torch.no_grad():
            pred = net(xt).numpy()
        self.assertLess(float(np.mean((pred - targets) ** 2)), 0.2)

        # predictions are unchanged by shuffling each set's element order at inference time
        shuffled = xt[:, torch.randperm(4), :]
        with torch.no_grad():
            pred_shuffled = net(shuffled).numpy()
        np.testing.assert_allclose(pred, pred_shuffled, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
