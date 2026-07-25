"""Tests for the rate-adaptive embedding (mixle.reason.embedding)."""

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

    def test_encode_before_fit_raises_instead_of_returning_untrained_output(self):
        # _fitted was set but never checked -- encode/coordinate_kl/active_dim/rate_nats on a fresh
        # (randomly initialized, untrained) network used to silently return a code as if it meant
        # something.
        from mixle.reason import ScaledEmbedding

        emb = ScaledEmbedding(in_dim=4, max_dim=6, seed=0)
        with self.assertRaises(RuntimeError):
            emb.encode(np.zeros((1, 4)))
        with self.assertRaises(RuntimeError):
            emb.coordinate_kl(np.zeros((1, 4)))
        with self.assertRaises(RuntimeError):
            emb.active_dim(np.zeros((1, 4)))
        with self.assertRaises(RuntimeError):
            emb.rate_nats(np.zeros((1, 4)))

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

    # -- MXR-080-0280: empty/malformed/untrained fits must raise, not be silently certified --------

    def test_zero_or_negative_epochs_raises_instead_of_certifying_random_weights(self):
        # epochs<=0 used to still set _fitted=True with the network at its random initialization --
        # the training loop (`for _ in range(int(epochs))`) simply never ran. fit() must now require
        # a genuine positive epoch count, so at least one optimizer step always runs before a network
        # is certified fitted.
        from mixle.reason import ScaledEmbedding

        rng = np.random.RandomState(10)
        X = rng.normal(size=(20, 4))
        for bad_epochs in (0, -5):
            emb = ScaledEmbedding(in_dim=4, max_dim=5, seed=0)
            with self.assertRaises(ValueError):
                emb.fit(X, epochs=bad_epochs)
            # must NOT be silently certified fitted -- encode still refuses to run on an untrained net.
            with self.assertRaises(RuntimeError):
                emb.encode(X[:1])

    def test_empty_data_raises_instead_of_storing_nan_stats(self):
        # Zero rows used to silently store NaN standardization stats (mean/std of an empty slice)
        # via a bare np.atleast_2d(...) and still mark the embedding fitted.
        from mixle.reason import ScaledEmbedding

        emb = ScaledEmbedding(in_dim=4, max_dim=5, seed=0)
        with self.assertRaises(ValueError):
            emb.fit(np.empty((0, 4)), epochs=5)
        with self.assertRaises(RuntimeError):
            emb.encode(np.zeros((1, 4)))

    def test_fractional_dimension_raises_instead_of_truncating(self):
        # A bare int(in_dim) used to silently truncate 4.9 -> 4 instead of rejecting the fractional
        # value outright.
        from mixle.reason import ScaledEmbedding

        with self.assertRaises(TypeError):
            ScaledEmbedding(in_dim=4.9, max_dim=6, seed=0)
        with self.assertRaises(TypeError):
            ScaledEmbedding(in_dim=4, max_dim=6.2, seed=0)
        with self.assertRaises(TypeError):
            ScaledEmbedding(in_dim=4, max_dim=True, seed=0)  # bool masquerading as int

    def test_feature_width_mismatch_raises(self):
        from mixle.reason import ScaledEmbedding

        rng = np.random.RandomState(11)
        emb = ScaledEmbedding(in_dim=6, max_dim=4, seed=0)
        with self.assertRaises(ValueError):
            emb.fit(rng.normal(size=(20, 3)), epochs=5)

    def test_non_finite_data_raises(self):
        from mixle.reason import ScaledEmbedding

        rng = np.random.RandomState(12)
        X = rng.normal(size=(20, 3))
        X[0, 0] = np.nan
        with self.assertRaises(ValueError):
            ScaledEmbedding(in_dim=3, max_dim=4, seed=0).fit(X, epochs=5)
        X2 = rng.normal(size=(20, 3))
        X2[1, 1] = np.inf
        with self.assertRaises(ValueError):
            ScaledEmbedding(in_dim=3, max_dim=4, seed=0).fit(X2, epochs=5)

    def test_invalid_beta_or_kl_tau_raises(self):
        # beta weights the rate term and kl_tau thresholds a quantity (KL) that is always >= 0;
        # non-finite or negative values used to be accepted unchecked.
        from mixle.reason import ScaledEmbedding

        with self.assertRaises(ValueError):
            ScaledEmbedding(in_dim=3, max_dim=4, beta=-1.0, seed=0)
        with self.assertRaises(ValueError):
            ScaledEmbedding(in_dim=3, max_dim=4, beta=float("nan"), seed=0)
        with self.assertRaises(ValueError):
            ScaledEmbedding(in_dim=3, max_dim=4, kl_tau=-1e-2, seed=0)
        with self.assertRaises(ValueError):
            ScaledEmbedding(in_dim=3, max_dim=4, kl_tau=float("inf"), seed=0)

    def test_invalid_lr_and_weight_decay_raise(self):
        from mixle.reason import ScaledEmbedding

        rng = np.random.RandomState(13)
        X = rng.normal(size=(20, 3))
        with self.assertRaises(ValueError):
            ScaledEmbedding(in_dim=3, max_dim=4, seed=0).fit(X, epochs=5, lr=0.0)
        with self.assertRaises(ValueError):
            ScaledEmbedding(in_dim=3, max_dim=4, seed=0).fit(X, epochs=5, lr=-0.1)
        with self.assertRaises(ValueError):
            ScaledEmbedding(in_dim=3, max_dim=4, seed=0).fit(X, epochs=5, weight_decay=-0.1)

    # -- MXR-080-0282: independent instances have no cross-modal alignment mechanism ----------------

    def test_independently_fit_instances_have_no_cross_modal_alignment(self):
        # The documented claim (pre-fix) was that codes "are comparable across modalities (a common
        # embedding)". Demonstrate that's false: two ScaledEmbedding instances, each fit on its own
        # linear rendering of the SAME shared underlying factor u (like an "image view" and a "text
        # view" of the same concept), converge to independent, arbitrary latent rotations. A genuine
        # common embedding would let a truly-paired sample (same row of u) retrieve as each other's
        # nearest neighbor far above chance; an unaligned one should not do reliably better than
        # chance, because nothing ties the two instances' coordinate systems together.
        from mixle.reason import ScaledEmbedding

        rng = np.random.RandomState(42)
        n, k, n_test = 600, 3, 100
        u = rng.normal(size=(n, k))  # the shared underlying factor, paired across modalities by row
        A = rng.normal(size=(7, k))
        B = rng.normal(size=(9, k))
        xA = u @ A.T + rng.normal(0, 0.05, size=(n, 7))  # "modality A" -- 7 raw features
        xB = u @ B.T + rng.normal(0, 0.05, size=(n, 9))  # "modality B" -- 9 raw features

        emb_a = ScaledEmbedding(in_dim=7, max_dim=5, seed=1).fit(xA, epochs=500)
        emb_b = ScaledEmbedding(in_dim=9, max_dim=5, seed=2).fit(xB, epochs=500)

        code_a = emb_a.encode(xA[:n_test])
        code_b = emb_b.encode(xB[:n_test])

        a_norm = code_a / (np.linalg.norm(code_a, axis=1, keepdims=True) + 1e-12)
        b_norm = code_b / (np.linalg.norm(code_b, axis=1, keepdims=True) + 1e-12)
        sim = a_norm @ b_norm.T  # (n_test, n_test) cosine similarity, true pairs on the diagonal

        nn_idx = np.argmax(sim, axis=1)
        top1_acc = float(np.mean(nn_idx == np.arange(n_test)))
        chance = 1.0 / n_test
        # A genuine common embedding retrieves true pairs far above chance; an arbitrary independent
        # rotation should not clear a generous multiple of chance.
        self.assertLess(top1_acc, 10 * chance)

        matched_sim = np.diag(sim).mean()
        unmatched_sim = (sim.sum() - np.diag(sim).sum()) / (n_test * n_test - n_test)
        # No meaningful similarity advantage for genuinely-paired samples over mismatched ones.
        self.assertLess(matched_sim - unmatched_sim, 0.15)


if __name__ == "__main__":
    unittest.main()
