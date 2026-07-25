"""Tests for the amortized modality encoder (mixle.reason.encoder)."""

import unittest

import numpy as np

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from mixle.reason import Latent, reason


@unittest.skipUnless(HAS_TORCH, "amortized encoder needs torch")
class AmortizedEncoderTest(unittest.TestCase):
    def test_recovers_a_linear_mapping(self):
        from mixle.reason import AmortizedEncoder

        rng = np.random.RandomState(0)
        X = rng.normal(size=(400, 3))
        W = np.array([[1.0, -2.0, 0.5], [0.0, 1.0, 1.0]])  # 2-d latent = W x
        Z = X @ W.T + rng.normal(0, 0.05, size=(400, 2))
        enc = AmortizedEncoder(in_dim=3, latent_dim=2, hidden=(32,), seed=0).fit(X, Z, epochs=400)
        # held-out recovery
        Xt = rng.normal(size=(100, 3))
        Zt = Xt @ W.T
        mu, _ = enc.encode_batch(Xt)
        rmse = np.sqrt(((mu - Zt) ** 2).mean())
        self.assertLess(rmse, 0.2)

    def test_encode_returns_gaussian_belief(self):
        from mixle.reason import AmortizedEncoder

        rng = np.random.RandomState(1)
        X = rng.normal(size=(200, 2))
        Z = (X[:, :1] * 2.0) + rng.normal(0, 0.1, size=(200, 1))
        enc = AmortizedEncoder(in_dim=2, latent_dim=1, seed=1).fit(X, Z, epochs=200)
        b = enc.encode(X[0])
        self.assertEqual(np.size(b.mean()), 1)
        self.assertGreater(b.var()[0], 0.0)

    def test_heteroscedastic_variance_tracks_noise(self):
        # Region A (x<0) is noisy; region B (x>=0) is clean. The encoder should report a LARGER
        # predicted sd in A than in B -- the whole point of a heteroscedastic PoE expert.
        from mixle.reason import AmortizedEncoder

        rng = np.random.RandomState(2)
        x = rng.uniform(-1, 1, size=(1500, 1))
        noise = np.where(x < 0, 0.6, 0.02)
        z = np.sin(3 * x) + rng.normal(0, 1, size=(1500, 1)) * noise
        enc = AmortizedEncoder(in_dim=1, latent_dim=1, hidden=(64, 64), seed=2).fit(x, z, epochs=600)
        _, var_noisy = enc.encode_batch(np.array([[-0.5]]))
        _, var_clean = enc.encode_batch(np.array([[0.5]]))
        self.assertGreater(np.sqrt(var_noisy[0, 0]), np.sqrt(var_clean[0, 0]))

    def test_evidence_plugs_into_reason_and_fuses(self):
        # Two encoders (two "modalities") of the same 1-d latent with different noise levels; fusing
        # their evidence beats either alone, and the cleaner modality earns more attribution.
        from mixle.reason import AmortizedEncoder

        rng = np.random.RandomState(3)
        n = 800
        # each modality is a noisy view of the SAME latent z: clean view (X1) vs noisy view (X2).
        z_true = rng.normal(size=(n, 1))
        X1 = z_true + rng.normal(0, 0.1, size=(n, 2))  # clean view (2 features)
        X2 = z_true + rng.normal(0, 0.8, size=(n, 2))  # noisy view (2 features)
        e1 = AmortizedEncoder(2, 1, seed=4).fit(X1, z_true, epochs=300)
        e2 = AmortizedEncoder(2, 1, seed=5).fit(X2, z_true, epochs=300)

        zt = 1.3
        x1 = np.full((1, 2), zt + 0.05)
        x2 = np.full((1, 2), zt - 0.3)
        prior = Latent.vector(1, var=100.0)
        fused = reason(prior, [e1.evidence(x1, name="clean"), e2.evidence(x2, name="noisy")])
        clean_only = reason(prior, [e1.evidence(x1, name="clean")])
        self.assertLess(fused.entropy(), clean_only.entropy())  # fusing adds information
        attr = fused.attribution()
        self.assertGreater(attr["clean"], attr["noisy"])  # cleaner modality contributes more

    def test_encode_before_fit_raises_instead_of_returning_untrained_output(self):
        # _fitted was set but never checked -- encode/encode_batch/evidence on a fresh (randomly
        # initialized, untrained) network used to silently return a belief as if it meant something.
        from mixle.reason import AmortizedEncoder

        enc = AmortizedEncoder(in_dim=2, latent_dim=1, seed=0)
        with self.assertRaises(RuntimeError):
            enc.encode(np.zeros(2))
        with self.assertRaises(RuntimeError):
            enc.encode_batch(np.zeros((3, 2)))
        with self.assertRaises(RuntimeError):
            enc.evidence(np.zeros(2))

    def test_evidence_onto_selects_sublatent(self):
        # An encoder targeting a 1-d property can inform coordinate 1 of a 3-d shared latent via onto.
        from mixle.reason import AmortizedEncoder

        rng = np.random.RandomState(6)
        X = rng.normal(size=(200, 2))
        Z = X[:, :1] * 1.5 + rng.normal(0, 0.05, size=(200, 1))
        enc = AmortizedEncoder(2, 1, seed=6).fit(X, Z, epochs=200)
        onto = np.array([[0.0, 1.0, 0.0]])  # reads coordinate 1 of a 3-vector latent
        ev = enc.evidence(X[0], onto=onto, name="prop")
        self.assertEqual(np.shape(ev.H), (1, 3))
        ans = reason(Latent.vector(3, var=10.0), [ev])
        self.assertEqual(np.size(ans.mean), 3)

    # -- MXR-080-0280: empty/malformed/untrained fits must raise, not be silently certified --------

    def test_zero_or_negative_epochs_raises_instead_of_certifying_random_weights(self):
        # epochs<=0 used to still set _fitted=True with the network at its random initialization --
        # the training loop (`for _ in range(int(epochs))`) simply never ran. fit() must now require
        # a genuine positive epoch count, so at least one optimizer step always runs before a network
        # is certified fitted.
        from mixle.reason import AmortizedEncoder

        rng = np.random.RandomState(10)
        X = rng.normal(size=(20, 3))
        Z = rng.normal(size=(20, 2))
        for bad_epochs in (0, -5):
            enc = AmortizedEncoder(in_dim=3, latent_dim=2, seed=0)
            with self.assertRaises(ValueError):
                enc.fit(X, Z, epochs=bad_epochs)
            # must NOT be silently certified fitted -- encode still refuses to run on an untrained net.
            with self.assertRaises(RuntimeError):
                enc.encode(X[0])

    def test_empty_data_raises_instead_of_storing_nan_stats(self):
        # Zero rows used to silently store NaN standardization stats (mean/std of an empty slice)
        # via a bare np.atleast_2d(...) and still mark the encoder fitted.
        from mixle.reason import AmortizedEncoder

        enc = AmortizedEncoder(in_dim=3, latent_dim=2, seed=0)
        with self.assertRaises(ValueError):
            enc.fit(np.empty((0, 3)), np.empty((0, 2)), epochs=5)
        with self.assertRaises(RuntimeError):
            enc.encode(np.zeros(3))

    def test_fractional_dimension_raises_instead_of_truncating(self):
        # A bare int(in_dim) used to silently truncate 2.7 -> 2 instead of rejecting the fractional
        # value outright.
        from mixle.reason import AmortizedEncoder

        with self.assertRaises(TypeError):
            AmortizedEncoder(in_dim=2.7, latent_dim=2, seed=0)
        with self.assertRaises(TypeError):
            AmortizedEncoder(in_dim=2, latent_dim=3.9, seed=0)
        with self.assertRaises(TypeError):
            AmortizedEncoder(in_dim=True, latent_dim=2, seed=0)  # bool masquerading as int

    def test_feature_width_mismatch_raises(self):
        from mixle.reason import AmortizedEncoder

        rng = np.random.RandomState(11)
        # X narrower than the declared in_dim.
        enc = AmortizedEncoder(in_dim=5, latent_dim=2, seed=0)
        with self.assertRaises(ValueError):
            enc.fit(rng.normal(size=(20, 3)), rng.normal(size=(20, 2)), epochs=5)
        # Z narrower than the declared latent_dim: this used to broadcast silently (a width-1 Z
        # against a latent_dim=3 target) instead of raising -- the encoder trained against the wrong
        # target shape without complaint.
        enc2 = AmortizedEncoder(in_dim=3, latent_dim=3, seed=0)
        with self.assertRaises(ValueError):
            enc2.fit(rng.normal(size=(20, 3)), rng.normal(size=(20, 1)), epochs=5)

    def test_non_finite_data_raises(self):
        from mixle.reason import AmortizedEncoder

        rng = np.random.RandomState(12)
        X = rng.normal(size=(20, 2))
        Z = rng.normal(size=(20, 1))
        X_bad = X.copy()
        X_bad[0, 0] = np.nan
        with self.assertRaises(ValueError):
            AmortizedEncoder(in_dim=2, latent_dim=1, seed=0).fit(X_bad, Z, epochs=5)
        Z_bad = Z.copy()
        Z_bad[0, 0] = np.inf
        with self.assertRaises(ValueError):
            AmortizedEncoder(in_dim=2, latent_dim=1, seed=0).fit(X, Z_bad, epochs=5)

    def test_invalid_min_sd_raises(self):
        # min_sd floors the predicted sd (see _forward_std); zero or negative defeats its documented
        # purpose of preventing an over-confident zero-variance expert.
        from mixle.reason import AmortizedEncoder

        with self.assertRaises(ValueError):
            AmortizedEncoder(in_dim=2, latent_dim=1, min_sd=0.0, seed=0)
        with self.assertRaises(ValueError):
            AmortizedEncoder(in_dim=2, latent_dim=1, min_sd=-1e-3, seed=0)

    def test_invalid_lr_and_weight_decay_raise(self):
        from mixle.reason import AmortizedEncoder

        rng = np.random.RandomState(13)
        X = rng.normal(size=(20, 2))
        Z = rng.normal(size=(20, 1))
        with self.assertRaises(ValueError):
            AmortizedEncoder(in_dim=2, latent_dim=1, seed=0).fit(X, Z, epochs=5, lr=0.0)
        with self.assertRaises(ValueError):
            AmortizedEncoder(in_dim=2, latent_dim=1, seed=0).fit(X, Z, epochs=5, lr=-0.1)
        with self.assertRaises(ValueError):
            AmortizedEncoder(in_dim=2, latent_dim=1, seed=0).fit(X, Z, epochs=5, weight_decay=-0.1)

    def test_onto_shape_inconsistent_with_latent_dim_raises(self):
        # onto's row count must equal latent_dim (H's rows must match mu's length for y = H z to be
        # well-formed evidence); this used to be accepted unchecked and only surface, if at all, deep
        # inside belief assimilation.
        from mixle.reason import AmortizedEncoder

        rng = np.random.RandomState(14)
        X = rng.normal(size=(30, 2))
        Z = rng.normal(size=(30, 1))
        enc = AmortizedEncoder(in_dim=2, latent_dim=1, seed=0).fit(X, Z, epochs=20)
        with self.assertRaises(ValueError):
            enc.evidence(X[0], onto=np.zeros((7, 4)))  # 7 rows != latent_dim=1
        with self.assertRaises(ValueError):
            enc.evidence(X[0], onto=np.full((1, 3), np.nan))  # non-finite


if __name__ == "__main__":
    unittest.main()
