"""StructuredAdapter contract (fast, synthetic, no CLIP).

Locks the robust properties the adapter guarantees; the empirical "structured preserves transfer where a
full matrix overfits" claim is demonstrated on REAL CLIP in examples/adapt_vlm_structured.py (3 splits),
which is where it is stable -- small synthetic anchors lack the shared cross-class structure that makes a
global adaptation transfer, so that comparison is not asserted here.
"""

import importlib.util
import unittest

import numpy as np

_HAS_TORCH = importlib.util.find_spec("torch") is not None


def _separable(seed, dim=32, nc=12, k=30, noise=0.05):
    rng = np.random.RandomState(seed)
    anchors = rng.randn(nc, dim)
    anchors /= np.linalg.norm(anchors, axis=1, keepdims=True)
    distort = np.ones(dim)
    distort[: dim // 2] = 0.3  # a diagonal distortion the encoder applies
    y = np.repeat(np.arange(nc), k)
    x = (anchors[y] * distort + rng.randn(len(y), dim) * noise).astype(np.float32)
    return anchors.astype(np.float32), x, y


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class StructuredAdapterTest(unittest.TestCase):
    def test_fit_optimizes_and_scores_over_unseen_anchors(self):
        import torch

        from mixle.reason import StructuredAdapter

        torch.manual_seed(0)
        anchors, x, y = _separable(0)
        seen = np.arange(8)
        m = np.isin(y, seen)
        ad = StructuredAdapter(x.shape[1], rank=4).fit(x[m], y[m], anchors[seen], epochs=200)
        # fit optimizes: the adapted model classifies its training classes well (labels 0..7 == their
        # positions in anchors[seen], so a direct compare is valid here)
        self.assertGreater((ad.predict(x[m], anchors[seen]) == y[m]).mean(), 0.9)
        # class-agnostic: it can score images against anchors for classes it NEVER trained on.
        # predict returns POSITIONS into the anchor array it is given, so map back through `unseen`.
        unseen = np.arange(8, 12)
        mu = np.isin(y, unseen)
        pos = ad.predict(x[mu], anchors[unseen])
        self.assertEqual(pos.shape, y[mu].shape)
        self.assertTrue(((pos >= 0) & (pos < len(unseen))).all())
        self.assertGreater((unseen[pos] == y[mu]).mean(), 0.5)  # and does better than chance on unseen

    def test_structured_is_far_smaller_than_the_full_matrix(self):
        from mixle.reason import StructuredAdapter

        dim = 512
        structured = StructuredAdapter(dim, rank=8)  # diagonal + rank-8 residual
        full = StructuredAdapter(dim, rank=8, full=True)
        self.assertEqual(structured.n_params(), dim + 2 * dim * 8)  # 8704
        self.assertEqual(full.n_params(), dim * dim)  # 262144
        self.assertLess(structured.n_params() * 25, full.n_params())

    def test_transform_before_fit_raises_instead_of_a_bare_attributeerror(self):
        # self._built was only ever assigned inside fit(), never in __init__ -- transform()/scores()/
        # predict() before fit() used to crash with an unguarded, unhelpful AttributeError.
        from mixle.reason import StructuredAdapter

        ad = StructuredAdapter(8, rank=2)
        with self.assertRaises(RuntimeError):
            ad.transform(np.zeros((1, 8)))

    def test_strong_weight_decay_keeps_the_map_near_identity(self):
        import torch

        from mixle.reason import StructuredAdapter

        torch.manual_seed(0)
        anchors, x, y = _separable(1)
        # heavy weight decay pulls the residual to zero -> transform(x) == normalize(x); this is the
        # mechanism that lets it adapt without moving the encoder's geometry (so transfer is preserved)
        ad = StructuredAdapter(x.shape[1], rank=4, weight_decay=1e4).fit(x, y, anchors, epochs=80)
        g = ad.transform(x)
        xn = x / np.linalg.norm(x, axis=1, keepdims=True)
        self.assertLess(np.abs(g - xn).max(), 1e-2)

    def test_anchor_rescaling_leaves_training_and_scores_unchanged(self):
        # MXR-080-0281: fit() used to normalize the adapted embedding but not the anchors, while
        # scores() normalized both -- so anchor magnitude silently changed what fit() optimized for
        # without changing what scores() ever reported. Rescaling one class's anchor by 50x, with an
        # identical seed and everything else held fixed, used to move every trained parameter (checked
        # against the pre-fix code: an L1 diff of ~0.88 after a single step). Now both paths agree on
        # the same cosine geometry, so it moves nothing.
        import torch

        from mixle.reason import StructuredAdapter

        anchors, x, y = _separable(2, dim=16, nc=3, k=30)
        rescale = np.array([1.0, 50.0, 1.0], dtype=np.float32)[:, None]  # only class 1's anchor grows

        ad_plain = StructuredAdapter(x.shape[1], rank=4).fit(x, y, anchors, epochs=40, seed=0)
        ad_rescaled = StructuredAdapter(x.shape[1], rank=4).fit(x, y, anchors * rescale, epochs=40, seed=0)

        for p_plain, p_rescaled in zip(ad_plain._params, ad_rescaled._params):
            self.assertTrue(torch.allclose(p_plain, p_rescaled, atol=1e-6))

        scores_plain = ad_plain.scores(x, anchors)
        scores_rescaled = ad_rescaled.scores(x, anchors)
        self.assertLess(np.abs(scores_plain - scores_rescaled).max(), 1e-5)

    def test_zero_vector_embedding_raises_instead_of_nan(self):
        from mixle.reason import StructuredAdapter

        anchors, x, y = _separable(3, dim=16, nc=3, k=10)
        x_bad = x.copy()
        x_bad[0] = 0.0
        ad = StructuredAdapter(x.shape[1], rank=4)
        with self.assertRaisesRegex(ValueError, "zero"):
            ad.fit(x_bad, y, anchors, epochs=3)
        # A failed fit() must not leave the adapter half-usable.
        self.assertIsNone(ad._built)

    def test_zero_vector_anchor_raises_instead_of_nan(self):
        from mixle.reason import StructuredAdapter

        anchors, x, y = _separable(4, dim=16, nc=3, k=10)
        ad = StructuredAdapter(x.shape[1], rank=4).fit(x, y, anchors, epochs=3)
        bad_anchors = anchors.copy()
        bad_anchors[0] = 0.0
        with self.assertRaisesRegex(ValueError, "zero"):
            ad.scores(x, bad_anchors)

    def test_fractional_labels_raise_instead_of_silent_truncation(self):
        # int64(0.5) truncated silently to 0 pre-fix; labels must be exact class indices.
        from mixle.reason import StructuredAdapter

        anchors, x, y = _separable(5, dim=16, nc=3, k=10)
        ad = StructuredAdapter(x.shape[1], rank=4)
        with self.assertRaisesRegex(ValueError, "fractional"):
            ad.fit(x, y.astype(np.float64) + 0.5, anchors, epochs=1)

    def test_out_of_range_label_raises_a_clear_error(self):
        # Pre-fix this reached torch.nn.functional.cross_entropy and raised a late, low-level
        # "Target 99 is out of bounds" IndexError instead of an input-validation error.
        from mixle.reason import StructuredAdapter

        anchors, x, y = _separable(6, dim=16, nc=3, k=10)
        y_bad = y.copy()
        y_bad[0] = 99
        ad = StructuredAdapter(x.shape[1], rank=4)
        with self.assertRaisesRegex(ValueError, "index into anchors"):
            ad.fit(x, y_bad, anchors, epochs=1)

    def test_mismatched_embedding_or_anchor_width_raises(self):
        # Pre-fix this reached the `g @ a.T` matmul and raised a cryptic shape-mismatch RuntimeError.
        from mixle.reason import StructuredAdapter

        anchors, x, y = _separable(7, dim=16, nc=3, k=10)
        ad = StructuredAdapter(x.shape[1], rank=4)
        with self.assertRaisesRegex(ValueError, "width"):
            ad.fit(x, y, anchors[:, :10], epochs=1)
        with self.assertRaisesRegex(ValueError, "width"):
            ad.fit(x[:, :10], y, anchors, epochs=1)

    def test_labels_row_count_mismatch_raises(self):
        from mixle.reason import StructuredAdapter

        anchors, x, y = _separable(8, dim=16, nc=3, k=10)
        ad = StructuredAdapter(x.shape[1], rank=4)
        with self.assertRaisesRegex(ValueError, "labels has"):
            ad.fit(x, y[:-5], anchors, epochs=1)

    def test_zero_epochs_raises_and_adapter_stays_unfitted(self):
        # Pre-fix, epochs=0 skipped the training loop entirely but still set self._params, so
        # scores()/transform() happily served a random, never-trained map as if it were fitted.
        from mixle.reason import StructuredAdapter

        anchors, x, y = _separable(9, dim=16, nc=3, k=10)
        ad = StructuredAdapter(x.shape[1], rank=4)
        with self.assertRaises(ValueError):
            ad.fit(x, y, anchors, epochs=0)
        self.assertIsNone(ad._built)
        self.assertIsNone(ad._params)
        with self.assertRaises(RuntimeError):
            ad.transform(x[:1])

    def test_non_positive_or_non_integer_epochs_rejected(self):
        from mixle.reason import StructuredAdapter

        anchors, x, y = _separable(10, dim=16, nc=3, k=10)
        for bad_epochs in (-1, 10.0, True):
            with self.assertRaises((ValueError, TypeError)):
                StructuredAdapter(x.shape[1], rank=4).fit(x, y, anchors, epochs=bad_epochs)

    def test_seed_makes_fit_reproducible(self):
        # Only the low-rank V is randomly initialized (diag/U/full-w start at zero); before this fix
        # fit() had no seed parameter at all, so repeated calls used whatever the global torch RNG
        # state happened to be.
        import torch

        from mixle.reason import StructuredAdapter

        anchors, x, y = _separable(11, dim=16, nc=3, k=10)
        ad_a = StructuredAdapter(x.shape[1], rank=4).fit(x, y, anchors, epochs=5, seed=123)
        ad_b = StructuredAdapter(x.shape[1], rank=4).fit(x, y, anchors, epochs=5, seed=123)
        for p_a, p_b in zip(ad_a._params, ad_b._params):
            self.assertTrue(torch.equal(p_a, p_b))

        ad_c = StructuredAdapter(x.shape[1], rank=4).fit(x, y, anchors, epochs=5, seed=999)
        self.assertFalse(all(torch.equal(p_a, p_c) for p_a, p_c in zip(ad_a._params, ad_c._params)))


if __name__ == "__main__":
    unittest.main()
