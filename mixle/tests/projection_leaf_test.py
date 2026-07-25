"""Contrastive projection objective/ranker contracts.

The stage-1 multimodal pattern -- frozen encoder -> trainable projection -> frozen encoder -- as a family with
no domain nouns. Three claims to check, matching the roadmap's acceptance criteria:

  1. with both backbones frozen, fitting moves ONLY the projection's parameters (bitwise check on backbones);
  2. the fitted projection is retrieval-useful: a true (x, y) pair scores higher than a shuffled/mismatched
     pair, on average, over enough held-out pairs for the check to be meaningful;
  3. it is explicitly not a probability-distribution leaf: its batch objective cannot enter mixture/HMM EM.
"""

import unittest

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mixle.models.mixture_density import (  # noqa: E402
    NeuralConditionalDensity,
    build_contrastive_projection,
    build_projection_leaf,
)
from mixle.models.neural import make_mlp  # noqa: E402


def _seed(s: int = 0) -> None:
    """Torch-model tests must be order-independent: pin the global RNG that drives module init and Adam."""
    torch.manual_seed(s)
    np.random.seed(s)


def _paired_embeddings(n=400, d_x=6, d_y=5, seed=0):
    """y is a fixed nonlinear function of x plus noise -- a genuine (x, y) correspondence to recover."""
    rng = np.random.RandomState(seed)
    x = rng.randn(n, d_x).astype("float32")
    mix = rng.randn(d_x, d_y).astype("float32")
    y = np.tanh(x @ mix) + 0.05 * rng.randn(n, d_y).astype("float32")
    return x, y


def _rows(x, y):
    return [(x[i], y[i]) for i in range(len(x))]


def _params_snapshot(module):
    return {name: p.detach().clone() for name, p in module.named_parameters()}


class FrozenBackboneTest(unittest.TestCase):
    def test_only_projection_params_move(self):
        _seed()
        d_x, d_y, raw_x, raw_y = 4, 4, 8, 7
        encoder_x = make_mlp(raw_x, [16], d_x)
        encoder_y = make_mlp(raw_y, [16], d_y)

        module = build_contrastive_projection(d_x, d_y, encoder_x=encoder_x, encoder_y=encoder_y, proj_dim=3, hidden=16)
        before = _params_snapshot(module)

        rng = np.random.RandomState(1)
        n = 200
        raw = rng.randn(n, raw_x).astype("float32")
        rx = raw
        mix = rng.randn(raw_x, raw_y).astype("float32")
        ry = np.tanh(raw @ mix).astype("float32")
        # x, y here are the RAW items the frozen encoders consume, not precomputed embeddings.
        x_tensor = torch.as_tensor(rx)
        y_tensor = torch.as_tensor(ry)
        optimizer = torch.optim.Adam(
            (parameter for parameter in module.parameters() if parameter.requires_grad), lr=1e-2
        )
        module.train()
        for _ in range(20):
            optimizer.zero_grad()
            module.contrastive_loss(x_tensor, y_tensor).backward()
            optimizer.step()
        after = _params_snapshot(module)

        backbone_names = [n for n in before if n.startswith("encoder_x.") or n.startswith("encoder_y.")]
        proj_names = [n for n in before if n.startswith("proj_x.") or n.startswith("proj_y.") or n == "log_tau"]
        self.assertTrue(backbone_names, "expected the frozen encoders to contribute named parameters")
        self.assertTrue(proj_names, "expected the projection to contribute named parameters")

        for name in backbone_names:
            self.assertTrue(
                torch.equal(before[name], after[name]), f"frozen backbone parameter {name!r} changed during fit"
            )

        moved = [name for name in proj_names if not torch.equal(before[name], after[name])]
        self.assertTrue(moved, "expected at least one projection parameter to change during fit")

    def test_encoders_are_not_in_the_optimizer_train_mode(self):
        # a frozen backbone stays in eval() through the M-step's train() call, regardless of dropout/batchnorm
        _seed()
        module = build_contrastive_projection(3, 3, hidden=8, proj_dim=2)
        module.train()
        self.assertFalse(module.encoder_x.training)
        self.assertFalse(module.encoder_y.training)


class RetrievalSanityTest(unittest.TestCase):
    def test_true_pairs_score_above_shuffled_pairs(self):
        _seed()
        d_x, d_y = 6, 5
        x, y = _paired_embeddings(n=400, d_x=d_x, d_y=d_y, seed=0)
        x_train, y_train = torch.as_tensor(x[:300]), torch.as_tensor(y[:300])
        x_test, y_test = torch.as_tensor(x[300:]), torch.as_tensor(y[300:])

        module = build_contrastive_projection(d_x, d_y, proj_dim=4, hidden=32)
        optimizer = torch.optim.Adam(module.parameters(), lr=5e-3)
        for _ in range(80):
            optimizer.zero_grad()
            module.contrastive_loss(x_train, y_train).backward()
            optimizer.step()

        with torch.no_grad():
            true_score = module.score_pairs(x_test, y_test).numpy()
            shuffled = torch.as_tensor(np.random.RandomState(2).permutation(len(y_test)))
            shuffled_score = module.score_pairs(x_test, y_test[shuffled]).numpy()

        self.assertGreater(true_score.mean(), shuffled_score.mean())
        # a meaningful, not marginal, margin over enough held-out pairs
        self.assertGreater(true_score.mean() - shuffled_score.mean(), 0.15)

    def test_pair_scores_do_not_depend_on_unrelated_batch_rows(self):
        _seed()
        module = build_contrastive_projection(3, 2, proj_dim=2, hidden=8)
        x = torch.randn(5, 3)
        y = torch.randn(5, 2)
        single = module.score_pairs(x[:1], y[:1])
        batched = module.score_pairs(x, y)
        torch.testing.assert_close(single[0], batched[0])


class ObjectiveBoundaryTest(unittest.TestCase):
    def test_weighted_objective_has_explicit_validated_measures(self):
        _seed()
        module = build_contrastive_projection(3, 3, hidden=8)
        x, y = torch.randn(4, 3), torch.randn(4, 3)
        uniform = module.contrastive_loss(x, y)
        weighted = module.contrastive_loss(
            x,
            y,
            anchor_weights=torch.tensor([2.0, 1.0, 0.0, 1.0]),
            x_candidate_weights=torch.tensor([2.0, 1.0, 0.0, 1.0]),
            y_candidate_weights=torch.tensor([1.0, 2.0, 0.0, 1.0]),
        )
        self.assertTrue(torch.isfinite(uniform))
        self.assertTrue(torch.isfinite(weighted))
        with self.assertRaises(ValueError):
            module.contrastive_loss(x, y, anchor_weights=torch.tensor([1.0, -1.0, 1.0, 1.0]))
        with self.assertRaises(ValueError):
            module.contrastive_loss(x[:1], y[:1])

    def test_contrastive_ranker_cannot_masquerade_as_a_probability_leaf(self):
        module = build_projection_leaf(3, 3)
        self.assertFalse(hasattr(module, "log_density"))
        with self.assertRaisesRegex(TypeError, "one independent score per row"):
            NeuralConditionalDensity(module)


if __name__ == "__main__":
    unittest.main()
