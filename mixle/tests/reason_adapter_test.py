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


if __name__ == "__main__":
    unittest.main()
