"""Generative objective (mixle.represent.generative): train the embedding to MODEL the data, tokens inferred.

The plain autoencoder must drop reconstruction loss (the embedding becomes a generative representation, no
collapse), and the VQ-VAE variant must reconstruct through a learned codebook that uses several codes -- the
vocabulary inferred to preserve information.
"""

import unittest

import numpy as np
import pytest

pytest.importorskip("torch")

from mixle.represent import VectorQuantizer, fit_autoencoder  # noqa: E402


def _clustered_units(seed=0, per=80, dim_in=6, k=3):
    # data on k low-dim clusters -> reconstructable by a small autoencoder
    rng = np.random.RandomState(seed)
    centers = rng.randn(k, dim_in) * 4
    return np.vstack([centers[j] + 0.3 * rng.randn(per, dim_in) for j in range(k)]).astype(np.float32)


class AutoencoderTest(unittest.TestCase):
    def test_reconstruction_loss_drops(self):
        units = _clustered_units()
        res = fit_autoencoder(units, dim=4, hidden=(16,), epochs=200, lr=5e-3, seed=0)
        self.assertLess(res.losses[-1], 0.5 * res.losses[0])  # the encoder learned to model the data
        self.assertLess(res.losses[-1], 0.5)

    def test_encoder_is_usable_after_fit(self):
        units = _clustered_units(1)
        res = fit_autoencoder(units, dim=4, hidden=(16,), epochs=120, seed=0)
        z = res.encode(units[:5])
        self.assertEqual(z.shape, (5, 4))


class VQVAETest(unittest.TestCase):
    def test_learned_codebook_reconstructs(self):
        units = _clustered_units(2, k=4)
        vq = VectorQuantizer(num_codes=4, dim=4, seed=0)
        res = fit_autoencoder(units, dim=4, hidden=(16,), quantizer=vq, epochs=250, lr=5e-3, seed=0)
        # reconstruction through the discrete bottleneck still improves a lot from the start
        self.assertLess(res.losses[-1], 0.6 * res.losses[0])
        # the learned vocabulary uses several codes (didn't collapse to one)
        codes = vq.quantize(res.encode(units))
        self.assertGreaterEqual(len(set(codes.tolist())), 3)


if __name__ == "__main__":
    unittest.main()
