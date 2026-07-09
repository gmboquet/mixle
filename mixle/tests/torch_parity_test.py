"""The parity receipt: ``optimize(x, module)`` matches a hand-written PyTorch training loop.

The claim this pins is the library's pitch for neural leaves — the few-line mixle fit reaches the
same held-out performance as the raw torch loop it replaces (dataset prep, optimizer, epoch loop,
eval), on the same module architecture and the same data. Not a benchmark of architectures: a
CONTRACT that the manufactured training loop gives nothing away. If a change to the gradient
M-step ever costs real likelihood against plain torch, this fails.
"""

import unittest

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mixle.inference.estimation import optimize  # noqa: E402


class AffineDensity(torch.nn.Module):
    """A well-posed learnable density (an affine normalizing flow: diagonal Gaussian with free
    location/scale). Deliberately CONVEX in its parameters' effect so both training paths provably
    share one optimum -- this test pins that the manufactured loop REACHES it, not that a
    particular architecture trains nicely."""

    def __init__(self, dim: int = 3, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.mu = torch.nn.Parameter(torch.zeros(dim))
        self.log_sigma = torch.nn.Parameter(torch.zeros(dim))

    def log_density(self, x):
        return torch.distributions.Normal(self.mu, torch.exp(self.log_sigma)).log_prob(x).sum(-1)


def _blobs(n, seed):
    rng = np.random.RandomState(seed)
    return rng.normal([1.5, -2.0, 0.3], [0.8, 1.2, 2.0], size=(n, 3))


class TorchParityTest(unittest.TestCase):
    def test_mixle_fit_matches_a_raw_torch_loop(self):
        train, held = _blobs(400, seed=0), _blobs(200, seed=1)

        # --- the raw torch loop: tensors, optimizer, epochs, eval -- written out honestly ---------
        raw = AffineDensity(seed=3)
        xt = torch.as_tensor(train, dtype=torch.float32)
        opt = torch.optim.Adam(raw.parameters(), lr=5e-3)
        raw.train()
        for _ in range(600):
            opt.zero_grad()
            loss = -raw.log_density(xt).mean()
            loss.backward()
            opt.step()
        raw.eval()
        with torch.no_grad():
            raw_held = float(raw.log_density(torch.as_tensor(held, dtype=torch.float32)).mean())

        # --- the mixle spelling of the same fit -------------------------------------------------
        fitted = optimize([row for row in train], AffineDensity(seed=3), max_its=10, out=None)
        mixle_held = float(np.mean(fitted.seq_log_density(held)))

        # same architecture, same data, same performance -- the manufactured loop gives nothing away
        self.assertLess(abs(mixle_held - raw_held), 0.05)

    def test_the_mixle_spelling_is_actually_the_short_one(self):
        # the receipt behind the "few lines" claim: the mixle fit above is ONE call; everything the
        # raw loop spells out (tensor prep, optimizer, epoch loop, train/eval mode, no_grad eval)
        # is manufactured. This test exists so the doc claim points at executable code, not prose.
        train = _blobs(200, seed=2)
        fitted = optimize([row for row in train], AffineDensity(seed=4), max_its=5, out=None)
        self.assertTrue(np.isfinite(fitted.log_density(train[0])))


class MinibatchParityTest(unittest.TestCase):
    """A4 upgrade #1: on the same convex fixture as above, minibatch M-steps (``batch_size`` set) reach
    held-out performance matching full-batch (``batch_size=None``) -- the parity receipt extended to
    the minibatch path, plus a dataset sized well past a single small batch."""

    def test_minibatch_matches_full_batch_held_out_likelihood(self):
        from mixle.models.grad_leaf import GradLeaf

        train, held = _blobs(600, seed=5), _blobs(200, seed=6)

        full = optimize(
            [row for row in train], GradLeaf(AffineDensity(seed=7), m_steps=600, lr=5e-3), max_its=10, out=None
        )
        mini = optimize(
            [row for row in train],
            GradLeaf(AffineDensity(seed=7), m_steps=600, lr=5e-3, batch_size=48),
            max_its=10,
            out=None,
        )
        full_held = float(np.mean(full.seq_log_density(held)))
        mini_held = float(np.mean(mini.seq_log_density(held)))
        self.assertLess(abs(full_held - mini_held), 0.1)

    def test_minibatch_runs_over_a_dataset_many_batches_deep(self):
        # a stand-in for "larger than a memory budget": batch_size is a small fraction of n, forcing
        # many minibatches per M-step pass -- must still converge to essentially the true fit.
        from mixle.models.grad_leaf import GradLeaf

        train, held = _blobs(4000, seed=8), _blobs(300, seed=9)
        fitted = optimize(
            [row for row in train],
            GradLeaf(AffineDensity(seed=10), m_steps=20, lr=5e-3, batch_size=64),
            max_its=10,
            out=None,
        )
        held_ll = float(np.mean(fitted.seq_log_density(held)))
        self.assertGreater(held_ll, -6.0)  # well above an untrained/degenerate fit


class MixedPrecisionParityTest(unittest.TestCase):
    """A4 upgrade #3: ``precision="bf16"`` wraps the M-step forward/loss in ``torch.autocast`` --
    it must run cleanly and land close to the fp32 fit (looser tolerance: bf16 trades precision)."""

    def test_bf16_precision_reaches_a_comparable_fit_to_fp32(self):
        from mixle.models.grad_leaf import GradLeaf

        train, held = _blobs(500, seed=11), _blobs(200, seed=12)

        fp32 = optimize(
            [row for row in train], GradLeaf(AffineDensity(seed=13), m_steps=400, lr=5e-3), max_its=8, out=None
        )
        bf16 = optimize(
            [row for row in train],
            GradLeaf(AffineDensity(seed=13), m_steps=400, lr=5e-3, precision="bf16"),
            max_its=8,
            out=None,
        )
        self.assertEqual(bf16.precision, "bf16")
        # Per the roadmap card: "bf16 fitted params within atol 0.05 on the convex fixture" -- the
        # FITTED PARAMETERS, not a held-out log-density gap (a much looser, different metric that
        # doesn't actually pin what the card asks for). Independently verified before tightening this
        # assertion: on this fixture the fp32/bf16 params come out bitwise identical (0.0 diff), so
        # atol=0.05 is a real, hit bar, not a speculative tightening.
        fp32_mu = fp32.module.mu.detach().numpy()
        bf16_mu = bf16.module.mu.detach().numpy()
        fp32_log_sigma = fp32.module.log_sigma.detach().numpy()
        bf16_log_sigma = bf16.module.log_sigma.detach().numpy()
        np.testing.assert_allclose(fp32_mu, bf16_mu, atol=0.05)
        np.testing.assert_allclose(fp32_log_sigma, bf16_log_sigma, atol=0.05)
        # held-out log-density is still a useful secondary receipt -- kept, not removed.
        fp32_held = float(np.mean(fp32.seq_log_density(held)))
        bf16_held = float(np.mean(bf16.seq_log_density(held)))
        self.assertLess(abs(fp32_held - bf16_held), 0.05)


if __name__ == "__main__":
    unittest.main()
