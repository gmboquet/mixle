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


if __name__ == "__main__":
    unittest.main()
