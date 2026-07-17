"""Tests for the trained multimodal PoE-VAE (mixle.reason.model.CrossModalModel).

The point of these: the model learns a shared latent from multimodal data with NO access to the
true latent -- unsupervised joint training through the shared latent, the thing supervised encoders
cannot do.
"""

import unittest

import numpy as np

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _two_view_data(rng, n, k, dA, dB, noise=0.05):
    """Two modalities that are different noisy linear views of the SAME latent factor s (unobserved)."""
    s = rng.normal(size=(n, k))
    WA = rng.normal(size=(k, dA))
    WB = rng.normal(size=(k, dB))
    xA = s @ WA + rng.normal(0, noise, size=(n, dA))
    xB = s @ WB + rng.normal(0, noise, size=(n, dB))
    return s, xA, xB


def _linear_r2(z, s):
    """R^2 of the best linear map z -> s (VAE recovers the factor up to a linear transform)."""
    Z = np.hstack([z, np.ones((len(z), 1))])
    coef, *_ = np.linalg.lstsq(Z, s, rcond=None)
    pred = Z @ coef
    ss_res = ((s - pred) ** 2).sum()
    ss_tot = ((s - s.mean(0)) ** 2).sum()
    return 1.0 - ss_res / ss_tot


@unittest.skipUnless(HAS_TORCH, "cross-modal model needs torch")
class CrossModalModelTest(unittest.TestCase):
    def test_recovers_shared_factor_unsupervised(self):
        # THE test: after training with no access to s, the inferred latent linearly predicts s.
        from mixle.reason import CrossModalModel

        rng = np.random.RandomState(0)
        s, xA, xB = _two_view_data(rng, 1200, k=2, dA=6, dB=5)
        m = CrossModalModel(latent_dim=4, seed=0)
        m.add_modality("A", 6).add_modality("B", 5)
        m.fit({"A": xA, "B": xB}, epochs=700, beta=0.3)

        # the inferred latent (from both modalities) linearly recovers the never-seen factor s
        z = np.array([m.encode({"A": xA[i], "B": xB[i]}) for i in range(300)])
        r2 = _linear_r2(z, s[:300])
        self.assertGreater(r2, 0.7)  # learned the shared factor from reconstruction alone

    def test_infers_from_a_single_modality(self):
        from mixle.reason import CrossModalModel

        rng = np.random.RandomState(2)
        s, xA, xB = _two_view_data(rng, 1200, k=2, dA=6, dB=5)
        m = CrossModalModel(latent_dim=4, seed=1)
        m.add_modality("A", 6).add_modality("B", 5)
        m.fit({"A": xA, "B": xB}, epochs=700, beta=0.3)
        # inference from modality A ALONE still recovers the shared factor (subset training worked)
        zA = np.array([m.encode({"A": xA[i]}) for i in range(300)])
        self.assertGreater(_linear_r2(zA, s[:300]), 0.55)

    def test_both_modalities_sharpen_the_belief(self):
        from mixle.reason import CrossModalModel

        rng = np.random.RandomState(3)
        s, xA, xB = _two_view_data(rng, 1000, k=2, dA=6, dB=5)
        m = CrossModalModel(latent_dim=4, seed=2)
        m.add_modality("A", 6).add_modality("B", 5)
        m.fit({"A": xA, "B": xB}, epochs=600, beta=0.3)
        b_both = m.belief({"A": xA[0], "B": xB[0]})
        b_one = m.belief({"A": xA[0]})
        # product-of-experts: two modalities give a tighter (lower-entropy) belief than one
        self.assertLess(b_both.entropy(), b_one.entropy())

    def test_cross_modal_generation(self):
        # Predict modality B from modality A alone -- generate a missing modality.
        from mixle.reason import CrossModalModel

        rng = np.random.RandomState(4)
        s, xA, xB = _two_view_data(rng, 1200, k=2, dA=6, dB=5, noise=0.03)
        m = CrossModalModel(latent_dim=4, seed=3)
        m.add_modality("A", 6).add_modality("B", 5)
        m.fit({"A": xA, "B": xB}, epochs=800, beta=0.2)
        pred_B = np.array([m.predict({"A": xA[i]}, target="B") for i in range(200)])
        # generated B correlates with the true B (information flowed A -> z -> B)
        corr = np.corrcoef(pred_B.ravel(), xB[:200].ravel())[0, 1]
        self.assertGreater(corr, 0.6)

    def test_belief_flows_into_reasoning_stack(self):
        # The trained belief is a GaussianBelief -> usable by reason()/decompose/conformal.
        from mixle.inference import decompose_variance
        from mixle.reason import CrossModalModel

        rng = np.random.RandomState(5)
        s, xA, xB = _two_view_data(rng, 800, k=2, dA=6, dB=5)
        m = CrossModalModel(latent_dim=3, seed=4)
        m.add_modality("A", 6).add_modality("B", 5)
        m.fit({"A": xA, "B": xB}, epochs=400, beta=0.3)
        b = m.belief({"A": xA[0], "B": xB[0]})
        self.assertEqual(np.size(b.mean()), 3)
        self.assertTrue(np.all(b.sd() > 0))
        # an ensemble of beliefs across records -> epistemic variance decomposition
        means = np.array([m.belief({"A": xA[i]}).mean() for i in range(20)])
        dec = decompose_variance(means)
        self.assertEqual(dec.kind, "variance")

    def test_conformal_prediction_intervals_have_coverage(self):
        # The honest-UQ claim, verified: conformal intervals cover the truth at ~1-alpha on held-out
        # data -- a finite-sample, distribution-free guarantee (not a Gaussian-posterior hope).
        from mixle.reason import CrossModalModel

        rng = np.random.RandomState(7)
        s, xA, xB = _two_view_data(rng, 1500, k=2, dA=6, dB=4, noise=0.1)
        m = CrossModalModel(latent_dim=4, seed=6)
        m.add_modality("A", 6).add_modality("B", 4)
        m.fit({"A": xA[:900], "B": xB[:900]}, epochs=700, beta=0.2)

        alpha = 0.1
        m.calibrate({"A": xA[900:1200], "B": xB[900:1200]}, target="B", alpha=alpha)  # calibration split
        # test split: empirical coverage of the interval predicting B from A
        covered = []
        for i in range(1200, 1500):
            lo, hi = m.predict_interval({"A": xA[i]}, target="B")
            covered.append(np.all((xB[i] >= lo) & (xB[i] <= hi)))
        coverage = np.mean(covered)
        # SIMULTANEOUS coverage over the whole target vector should hold near/above 1-alpha
        # (finite-sample conformal guarantee), give or take sampling slack on 300 test points.
        self.assertGreater(coverage, 1 - alpha - 0.06)

    def test_predict_interval_needs_calibration(self):
        from mixle.reason import CrossModalModel

        rng = np.random.RandomState(8)
        s, xA, xB = _two_view_data(rng, 400, k=2, dA=4, dB=3)
        m = CrossModalModel(latent_dim=3, seed=7)
        m.add_modality("A", 4).add_modality("B", 3)
        m.fit({"A": xA, "B": xB}, epochs=200)
        with self.assertRaises(RuntimeError):
            m.predict_interval({"A": xA[0]}, target="B")

    def test_inference_before_fit_raises_instead_of_scoring_random_weights(self):
        # _fitted was tracked but never checked: a freshly constructed model's encoders/decoder
        # carry their random init weights, so belief()/encode()/predict() could silently return a
        # meaningless "result" that looks like real output. All four inference entry points route
        # through belief(), so one check there covers them.
        from mixle.reason import CrossModalModel

        rng = np.random.RandomState(9)
        _, xA, xB = _two_view_data(rng, 10, k=2, dA=4, dB=3)
        m = CrossModalModel(latent_dim=3, seed=7)
        m.add_modality("A", 4).add_modality("B", 3)

        with self.assertRaises(RuntimeError):
            m.belief({"A": xA[0]})
        with self.assertRaises(RuntimeError):
            m.encode({"A": xA[0]})
        with self.assertRaises(RuntimeError):
            m.predict({"A": xA[0]}, target="B")
        with self.assertRaises(RuntimeError):
            m.calibrate({"A": xA, "B": xB}, target="B")

        # fit() clears the block; the model is usable afterward exactly as before
        m.fit({"A": xA, "B": xB}, epochs=5)
        m.belief({"A": xA[0]})  # no longer raises


if __name__ == "__main__":
    unittest.main()
