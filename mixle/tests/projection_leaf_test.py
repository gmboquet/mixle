"""``build_projection_leaf`` (mixle.models.mixture_density): a contrastive/InfoNCE conditional leaf.

The stage-1 multimodal pattern -- frozen encoder -> trainable projection -> frozen encoder -- as a family with
no domain nouns. Three claims to check, matching the roadmap's acceptance criteria:

  1. with both backbones frozen, fitting moves ONLY the projection's parameters (bitwise check on backbones);
  2. the fitted projection is retrieval-useful: a true (x, y) pair scores higher than a shuffled/mismatched
     pair, on average, over enough held-out pairs for the check to be meaningful;
  3. it rides the same ``NeuralConditionalDensity`` adapter as ``build_mdn``, so it composes inside a
     heterogeneous model next to a classical leaf (here: a ``CompositeDistribution`` field, the pattern this
     repo already uses for a conditional neural leaf beside a classical family -- see
     ``neural_composition_grid_test.py``).
"""

import unittest

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import mixle.stats as st  # noqa: E402
from mixle.inference import optimize  # noqa: E402
from mixle.models.mixture_density import NeuralConditionalDensity, build_projection_leaf  # noqa: E402
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

        module = build_projection_leaf(d_x, d_y, encoder_x=encoder_x, encoder_y=encoder_y, proj_dim=3, hidden=16)
        before = _params_snapshot(module)

        rng = np.random.RandomState(1)
        n = 200
        raw = rng.randn(n, raw_x).astype("float32")
        rx = raw
        mix = rng.randn(raw_x, raw_y).astype("float32")
        ry = np.tanh(raw @ mix).astype("float32")
        # x, y here are the RAW items the frozen encoders consume, not precomputed embeddings.
        rows = _rows(rx, ry)

        leaf = NeuralConditionalDensity(module, m_steps=60, lr=1e-2)
        fit = optimize(rows, leaf.estimator(), prev_estimate=leaf, max_its=3, out=None)
        after = _params_snapshot(fit.module)

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
        module = build_projection_leaf(3, 3, hidden=8, proj_dim=2)
        module.train()
        self.assertFalse(module.encoder_x.training)
        self.assertFalse(module.encoder_y.training)


class RetrievalSanityTest(unittest.TestCase):
    def test_true_pairs_score_above_shuffled_pairs(self):
        _seed()
        d_x, d_y = 6, 5
        x_train, y_train = _paired_embeddings(n=500, d_x=d_x, d_y=d_y, seed=0)
        x_test, y_test = _paired_embeddings(n=200, d_x=d_x, d_y=d_y, seed=1)

        module = build_projection_leaf(d_x, d_y, proj_dim=4, hidden=32)
        leaf = NeuralConditionalDensity(module, m_steps=300, lr=5e-3)
        fit = optimize(_rows(x_train, y_train), leaf.estimator(), prev_estimate=leaf, max_its=8, out=None)

        with torch.no_grad():
            px = fit.module.embed_x(torch.as_tensor(x_test, dtype=torch.float32))
            py = fit.module.embed_y(torch.as_tensor(y_test, dtype=torch.float32))
            true_score = (px * py).sum(dim=-1).numpy()

            shuffled = np.random.RandomState(2).permutation(len(y_test))
            py_shuffled = fit.module.embed_y(torch.as_tensor(y_test[shuffled], dtype=torch.float32))
            shuffled_score = (px * py_shuffled).sum(dim=-1).numpy()

        self.assertGreater(true_score.mean(), shuffled_score.mean())
        # a meaningful, not marginal, margin over enough held-out pairs
        self.assertGreater(true_score.mean() - shuffled_score.mean(), 0.15)


class MixtureCompositionTest(unittest.TestCase):
    def test_composes_with_a_classical_leaf_in_a_composite(self):
        # the established pattern in this repo for a conditional neural leaf beside a classical family
        # (see neural_composition_grid_test.py's NeuralGaussianCompositeAndHmmTest) -- one record, two
        # heterogeneous fields, fit jointly by EM.
        _seed()
        d_x, d_y = 4, 3
        module = build_projection_leaf(d_x, d_y, proj_dim=2, hidden=16)
        proj = NeuralConditionalDensity(module, m_steps=20, lr=1e-2)
        gauss = st.GaussianDistribution(0.0, 1.0)
        comp = st.CompositeDistribution((proj, gauss))

        rng = np.random.RandomState(0)
        n = 60
        x = rng.randn(n, d_x).astype("float32")
        mix = rng.randn(d_x, d_y).astype("float32")
        y = np.tanh(x @ mix).astype("float32")
        scalar = rng.randn(n)
        rows = [((x[i], y[i]), float(scalar[i])) for i in range(n)]

        self.assertTrue(np.isfinite(comp.log_density(rows[0])))

        est = st.CompositeEstimator((proj.estimator(), gauss.estimator()))
        fitted = optimize(rows, est, max_its=2, out=None)
        self.assertIsInstance(fitted, st.CompositeDistribution)
        ll = fitted.log_density(rows[0])
        self.assertTrue(np.isfinite(ll))


if __name__ == "__main__":
    unittest.main()
