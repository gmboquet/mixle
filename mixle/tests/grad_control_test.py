"""The two remaining control receipts: gradient checkpointing and LoRA-style adapters.

Checkpointing is a memory/compute trade, not a model change: the claim worth pinning is that the
recompute path produces IDENTICAL gradients to the plain path on the same weights and batch.
The adapter test pins the deeper claim behind "peft just works": an adapter-wrapped module is
still just a module -- drop it into the bridge, the frozen base never moves, only the low-rank
deltas train, and the fit genuinely improves the objective.
"""

import unittest

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mixle.inference.estimation import optimize  # noqa: E402
from mixle.models import GradLeaf  # noqa: E402
from mixle.models.transformer import build_causal_lm  # noqa: E402


class GradientCheckpointingTest(unittest.TestCase):
    def test_checkpointing_gradients_are_identical(self):
        torch.manual_seed(0)
        plain = build_causal_lm(vocab=17, d_model=16, n_layer=2, n_head=2, block=8)
        ckpt = build_causal_lm(vocab=17, d_model=16, n_layer=2, n_head=2, block=8, gradient_checkpointing=True)
        ckpt.load_state_dict(plain.state_dict())

        x = torch.randint(0, 17, (5, 8)).float()
        y = torch.randint(0, 17, (5,))

        def grads(model):
            model.train()
            model.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(x), y)
            loss.backward()
            return loss, {k: p.grad.clone() for k, p in model.named_parameters() if p.grad is not None}

        loss_a, ga = grads(plain)
        loss_b, gb = grads(ckpt)
        self.assertTrue(ckpt.gradient_checkpointing)
        np.testing.assert_allclose(float(loss_a), float(loss_b), rtol=0, atol=0)
        self.assertEqual(set(ga), set(gb))
        for k in ga:
            np.testing.assert_allclose(ga[k].numpy(), gb[k].numpy(), atol=1.0e-6)

    def test_flag_toggles_on_an_existing_model(self):
        model = build_causal_lm(vocab=11, d_model=16, n_layer=1, n_head=2, block=4)
        self.assertFalse(model.gradient_checkpointing)
        model.gradient_checkpointing = True  # a plain attribute: no rebuild, no ctor round-trip
        model.train()
        out = model(torch.randint(0, 11, (3, 4)).float())
        self.assertEqual(tuple(out.shape), (3, 11))


class LoRAStyleAdapter(torch.nn.Module):
    """The peft pattern, hand-rolled: a FROZEN base map plus a trainable low-rank delta. peft's
    wrapped modules are exactly this shape -- torch modules with frozen base weights -- which is
    why they drop into the bridge unchanged."""

    def __init__(self, dim: int = 2, rank: int = 1, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.base = torch.nn.Linear(dim, dim)
        self.base.requires_grad_(False)  # the frozen backbone
        self.lora_a = torch.nn.Parameter(torch.randn(dim, rank) * 0.1)
        self.lora_b = torch.nn.Parameter(torch.zeros(rank, dim))
        self.log_sigma = torch.nn.Parameter(torch.zeros(dim))

    def log_density(self, x):
        center = self.base(x) * 0.0 + self.base.bias + (self.lora_a @ self.lora_b).diag() + x * 0.0
        # a learnable location built ONLY from frozen-base offsets + the low-rank delta
        return torch.distributions.Normal(center, torch.exp(self.log_sigma)).log_prob(x).sum(-1)


class AdapterThroughTheBridgeTest(unittest.TestCase):
    def test_only_the_adapter_trains(self):
        rng = np.random.RandomState(0)
        data = rng.normal([2.0, -1.0], 0.5, size=(300, 2))

        module = LoRAStyleAdapter(dim=2)
        base_before = {k: v.clone() for k, v in module.base.state_dict().items()}
        before_ll = float(np.mean(GradLeaf(module).seq_log_density(data)))

        fitted = optimize([row for row in data], GradLeaf(module, m_steps=200, lr=0.05), max_its=3, out=None)

        for k, v in fitted.module.base.state_dict().items():  # the backbone never moved
            np.testing.assert_array_equal(v.numpy(), base_before[k].numpy())
        self.assertGreater(float(fitted.module.lora_b.abs().sum()), 0.0)  # the delta did
        after_ll = float(np.mean(fitted.seq_log_density(data)))
        self.assertGreater(after_ll, before_ll + 1.0)  # and the fit is real


if __name__ == "__main__":
    unittest.main()
