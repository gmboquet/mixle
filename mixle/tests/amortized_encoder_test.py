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


if __name__ == "__main__":
    unittest.main()
