"""Generative objective (mixle.represent.generative): train the embedding to MODEL the data, tokens inferred.

The plain autoencoder must drop reconstruction loss (the embedding becomes a generative representation, no
collapse), and the VQ-VAE variant must reconstruct through a learned codebook that uses several codes -- the
vocabulary inferred to preserve information.
"""

import unittest

import numpy as np
import pytest

pytest.importorskip("torch")

from mixle.represent import AutoencoderFitError, VectorQuantizer, fit_autoencoder  # noqa: E402
from mixle.represent.embed import FeatureEmbedding  # noqa: E402


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


class AutoencoderControlValidationTest(unittest.TestCase):
    """MXR-080-1661: an untrained or diverged encoder is never returned as a successful fit."""

    def setUp(self):
        self.units = _clustered_units(3, per=20, dim_in=4, k=2)

    def test_non_positive_or_fractional_epochs_are_rejected(self):
        for bad in (0, -2, 0.9, True):
            with self.subTest(epochs=repr(bad)), self.assertRaisesRegex(ValueError, "epochs"):
                fit_autoencoder(self.units, dim=3, epochs=bad)

    def test_non_positive_or_fractional_dims_are_rejected(self):
        for bad in (0, -1, 2.5, True):
            with self.subTest(dim=repr(bad)), self.assertRaisesRegex(ValueError, "dim"):
                fit_autoencoder(self.units, dim=bad, epochs=2)

    def test_hidden_widths_are_validated_too(self):
        with self.assertRaises(ValueError):
            fit_autoencoder(self.units, dim=3, hidden=(0,), epochs=2)
        with self.assertRaises(ValueError):
            FeatureEmbedding(4, 3, hidden=(-1,))

    def test_non_finite_units_are_rejected_before_any_training(self):
        bad = self.units.copy()
        bad[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            fit_autoencoder(bad, dim=3, epochs=1)

    def test_empty_and_ragged_units_are_rejected(self):
        with self.assertRaises(ValueError):
            fit_autoencoder(np.zeros((0, 4), dtype=np.float32), dim=3, epochs=2)
        with self.assertRaises(ValueError):
            fit_autoencoder(np.zeros(4, dtype=np.float32), dim=3, epochs=2)  # 1-D, not (N, f)

    def test_learning_rate_and_commitment_domains(self):
        with self.assertRaisesRegex(ValueError, "lr"):
            fit_autoencoder(self.units, dim=3, epochs=2, lr=0.0)
        with self.assertRaisesRegex(ValueError, "commitment"):
            fit_autoencoder(self.units, dim=3, epochs=2, commitment=-1.0)

    def test_divergent_training_raises_instead_of_returning_a_result(self):
        # a learning rate far past the stability limit drives the loss to inf/nan
        with self.assertRaises(AutoencoderFitError) as ctx:
            fit_autoencoder(self.units * 1e18, dim=3, hidden=(16,), epochs=200, lr=1e10, seed=0)
        self.assertIsInstance(ctx.exception.losses, list)
        self.assertGreaterEqual(ctx.exception.epoch, 0)

    def test_a_valid_fit_records_one_finite_loss_per_epoch(self):
        res = fit_autoencoder(self.units, dim=3, hidden=(8,), epochs=7, seed=0)
        self.assertEqual(len(res.losses), 7)
        self.assertTrue(all(np.isfinite(loss) for loss in res.losses))
        self.assertTrue(np.isfinite(res.encode(self.units)).all())

    def test_fitting_does_not_reseed_the_callers_torch_rng(self):
        import torch

        torch.manual_seed(1234)
        expected = torch.randn(3)
        torch.manual_seed(1234)
        fit_autoencoder(self.units, dim=3, epochs=3, seed=99)
        np.testing.assert_allclose(torch.randn(3).numpy(), expected.numpy())


if __name__ == "__main__":
    unittest.main()
