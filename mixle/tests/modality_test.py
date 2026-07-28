"""Modality leaves (C2): image/signal -> fixed-dim vectors that participate in the cross-modal graph."""

import unittest

import numpy as np

from mixle.represent import image_features, signal_features, vectorize, vectorize_all

try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class ImageFeatureTest(unittest.TestCase):
    def test_fixed_dim_from_2d_and_3d(self):
        self.assertEqual(image_features(np.random.rand(32, 32), dim=16).shape, (16,))
        self.assertEqual(image_features(np.random.rand(20, 20, 3), dim=9).shape, (9,))  # channels averaged

    def test_captures_brightness(self):
        dark = image_features(np.full((16, 16), 0.1), dim=9)
        light = image_features(np.full((16, 16), 0.9), dim=9)
        self.assertLess(dark.mean(), light.mean())  # brighter image -> higher descriptor

    def test_captures_spatial_layout(self):
        left = np.zeros((16, 16))
        left[:, :8] = 1.0  # bright on the left
        right = np.zeros((16, 16))
        right[:, 8:] = 1.0  # bright on the right
        self.assertFalse(np.allclose(image_features(left, dim=16), image_features(right, dim=16)))

    def test_no_spatial_blind_spot_for_non_square_dim(self):
        # MXR-080-1662: a ceil(sqrt(dim)) square grid truncated row-major to dim did not pool to a
        # coarser resolution -- it deleted the bottom/right cells outright. At dim=2 the entire bottom
        # half of this image was invisible (descriptor stayed exactly [0, 0]); at dim=3 the whole
        # bottom-right quadrant was.
        base = np.zeros((4, 4))
        for dim in (2, 3, 5, 7):
            ref = image_features(base, dim=dim)
            self.assertEqual(ref.shape, (dim,))
            bottom = base.copy()
            bottom[2:, :] = 100.0
            self.assertFalse(np.allclose(ref, image_features(bottom, dim=dim)), f"bottom half invisible at dim={dim}")
            quadrant = base.copy()
            quadrant[2:, 2:] = 100.0
            self.assertFalse(
                np.allclose(ref, image_features(quadrant, dim=dim)), f"bottom-right quadrant invisible at dim={dim}"
            )

    def test_every_pixel_influences_the_descriptor(self):
        # The cells partition the image, so each pixel lands in exactly one output coordinate; no
        # location may be a documented-nonzero-influence hole.
        zero = np.zeros((4, 4))
        for dim in (1, 2, 3, 5, 8, 16):
            ref = image_features(zero, dim=dim)
            for i in range(4):
                for j in range(4):
                    lit = zero.copy()
                    lit[i, j] = 1.0
                    self.assertFalse(
                        np.allclose(ref, image_features(lit, dim=dim)), f"pixel ({i},{j}) invisible at dim={dim}"
                    )

    def test_small_axis_is_reread_not_zero_padded(self):
        # Fewer image rows than grid rows must not append fabricated all-zero cells that are
        # indistinguishable from measured black regions.
        v = image_features(np.array([[1.0, 2.0, 3.0, 4.0]]), dim=16)
        self.assertEqual(v.shape, (16,))
        self.assertTrue((v > 0).all())

    def test_explicit_grid_must_partition_the_requested_dim(self):
        with self.assertRaises(ValueError):
            image_features(np.zeros((4, 4)), dim=5, grid=3)
        self.assertEqual(image_features(np.zeros((4, 4)), dim=4, grid=2).shape, (4,))


class SignalFeatureTest(unittest.TestCase):
    def test_fixed_dim(self):
        self.assertEqual(signal_features(np.sin(np.linspace(0, 10, 500)), dim=12).shape, (12,))

    def test_energy_separates_loud_from_quiet(self):
        quiet = signal_features(0.1 * np.random.RandomState(0).randn(300), dim=9)
        loud = signal_features(3.0 * np.random.RandomState(0).randn(300), dim=9)
        self.assertGreater(np.abs(loud).sum(), np.abs(quiet).sum())


class DispatchTest(unittest.TestCase):
    def test_vectorize_dispatch(self):
        self.assertEqual(vectorize(np.random.rand(16, 16), "image", dim=8).shape, (8,))
        self.assertEqual(vectorize(np.random.randn(200), "signal", dim=8).shape, (8,))

    def test_unknown_modality_raises(self):
        with self.assertRaises(ValueError):
            vectorize("x", "hologram")

    def test_vectorize_all_image_batch(self):
        imgs = [np.random.rand(12, 12) for _ in range(5)]
        out = vectorize_all(imgs, "image", dim=9)
        self.assertEqual(out.shape, (5, 9))


class CrossModalGraphTest(unittest.TestCase):
    def test_image_vector_participates_in_the_discovered_graph(self):
        from mixle.inference import Guarantee, certify, learn_bayesian_network

        # image detail (a top-left region) drives price, independent of the category (global brightness)
        def make(n, seed):
            r = np.random.RandomState(seed)
            rows = []
            for _ in range(n):
                cat = ["dark", "light"][r.randint(2)]
                base = 0.3 if cat == "dark" else 0.6
                q = r.uniform(0, 1)
                img = np.clip(base + 0.05 * r.randn(16, 16), 0, 1)
                img[:5, :5] = np.clip(q + 0.05 * r.randn(5, 5), 0, 1)  # region encodes q
                vec = image_features(img, dim=9)
                price = float(40.0 * q + 2.0 * r.randn())
                rows.append((cat, vec, price))
            return rows

        train, test = make(500, 0), make(200, 1)
        net = learn_bayesian_network(train, max_parents=2)
        kinds = {f.child: type(f).__name__ for f in net.factors}
        self.assertEqual(kinds[1], "_VectorCLGFactor")  # the image field is a multivariate vector node
        # the image vector and price are connected (orientation is non-identifiable for linear-Gaussian)
        edges = net.edges()
        self.assertTrue((1, 2) in edges or (2, 1) in edges, edges)
        # scores + certifies as the closed-form / convex graph it is (no gradient descent)
        enc = net.dist_to_encoder().seq_encode(test)
        self.assertTrue(np.isfinite(net.seq_log_density(enc)).all())
        cert = certify(net, data=train)
        self.assertEqual(cert.guarantee, Guarantee.UNVERIFIED)
        self.assertTrue(all(block.candidate_guarantee >= Guarantee.GLOBAL for block in cert.blocks))
        self.assertEqual(cert.gradient_blocks, [])


if __name__ == "__main__":
    unittest.main()
