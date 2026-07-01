"""Tests for the rate-adaptive common embedding (mixle.reason.embedding)."""

import unittest

import numpy as np

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _low_rank(rng, n, ambient, rank, noise=0.02):
    """n points on a `rank`-dim subspace of R^ambient (intrinsic dimension = rank)."""
    A = rng.normal(size=(ambient, rank))
    z = rng.normal(size=(n, rank))
    return z @ A.T + rng.normal(0, noise, size=(n, ambient))


@unittest.skipUnless(HAS_TORCH, "scaled embedding needs torch")
class ScaledEmbeddingTest(unittest.TestCase):
    def test_shapes_and_bounds(self):
        from mixle.reason import ScaledEmbedding

        rng = np.random.RandomState(0)
        X = _low_rank(rng, 300, ambient=8, rank=3)
        emb = ScaledEmbedding(in_dim=8, max_dim=10, seed=0).fit(X, epochs=200)
        code = emb.encode(X[:5])
        self.assertEqual(code.shape, (5, 10))
        ad = emb.active_dim(X[:5])
        self.assertTrue(np.all(ad >= 0) and np.all(ad <= 10))
        self.assertTrue(np.all(emb.rate_nats(X[:5]) >= 0))

    def test_active_dim_tracks_intrinsic_dimension(self):
        # The central claim: data of higher intrinsic dimension uses more active coordinates.
        from mixle.reason import ScaledEmbedding

        rng = np.random.RandomState(1)
        ambient, n = 12, 800
        X_low = _low_rank(rng, n, ambient, rank=1)
        X_high = _low_rank(rng, n, ambient, rank=6)
        emb_low = ScaledEmbedding(ambient, max_dim=10, beta=1.0, seed=1).fit(X_low, epochs=700)
        emb_high = ScaledEmbedding(ambient, max_dim=10, beta=1.0, seed=1).fit(X_high, epochs=700)
        ad_low = emb_low.active_dim(X_low).mean()
        ad_high = emb_high.active_dim(X_high).mean()
        self.assertGreater(ad_high, ad_low)  # more information content -> more active dimensions
        self.assertGreater(ad_low, 0)  # but still uses some

    def test_larger_beta_tightens_rate_budget(self):
        # A larger rate weight spends fewer bits -> fewer active dimensions on the same data.
        from mixle.reason import ScaledEmbedding

        rng = np.random.RandomState(2)
        X = _low_rank(rng, 800, ambient=12, rank=6)
        loose = ScaledEmbedding(12, max_dim=10, beta=0.2, seed=2).fit(X, epochs=700)
        tight = ScaledEmbedding(12, max_dim=10, beta=8.0, seed=2).fit(X, epochs=700)
        self.assertGreater(loose.active_dim(X).mean(), tight.active_dim(X).mean())

    def test_reconstructs_and_code_is_shared(self):
        # Similar inputs get nearby codes (a usable common coordinate system for retrieval).
        from mixle.reason import ScaledEmbedding

        rng = np.random.RandomState(3)
        X = _low_rank(rng, 500, ambient=6, rank=2)
        emb = ScaledEmbedding(6, max_dim=8, seed=3).fit(X, epochs=500)
        x = X[0]
        x_near = x + rng.normal(0, 0.01, size=x.shape)
        x_far = X[250]
        code = emb.encode(np.stack([x, x_near, x_far]))
        d_near = np.linalg.norm(code[0] - code[1])
        d_far = np.linalg.norm(code[0] - code[2])
        self.assertLess(d_near, d_far)


if __name__ == "__main__":
    unittest.main()
