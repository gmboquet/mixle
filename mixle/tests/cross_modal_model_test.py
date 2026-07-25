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

    def test_calibrate_widens_to_unbounded_radius_when_k_exceeds_n_cal(self):
        # split-conformal edge case: when ceil((n_cal+1)(1-alpha)) > n_cal, NO calibration score can
        # certify the stated level. calibrate() must widen to an infinite (unbounded) radius instead
        # of silently substituting scores.max() -- the substitution regime measures coverage at the
        # exact theoretical shortfall n_cal/(n_cal+1) (e.g. 0.8889 for n_cal=8), violating the
        # documented ">= 1 - alpha" simultaneous-coverage guarantee. Mirrors the k >= len(scores)
        # handling in mixle.scientist.study().
        from mixle.reason import CrossModalModel

        rng = np.random.RandomState(11)
        s, xA, xB = _two_view_data(rng, 300, k=2, dA=4, dB=3)
        m = CrossModalModel(latent_dim=3, seed=10)
        m.add_modality("A", 4).add_modality("B", 3)
        m.fit({"A": xA, "B": xB}, epochs=150, beta=0.3)

        alpha, n_cal = 0.1, 8
        k = int(np.ceil((n_cal + 1) * (1.0 - alpha)))
        self.assertGreater(k, n_cal)  # confirms this exercises the k > n regime, not a typo

        m.calibrate({"A": xA[:n_cal], "B": xB[:n_cal]}, target="B", alpha=alpha)
        _, _, q = m._conformal["B"]
        self.assertTrue(np.isinf(q))  # unbounded radius, not the finite scores.max()

        lo, hi = m.predict_interval({"A": xA[n_cal]}, target="B")
        self.assertTrue(np.all(np.isinf(hi - lo)))  # the box itself is genuinely unbounded

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

    # -- MXR-080-0276: modality replacement must not strand stale fitted/calibration state --------
    def test_add_modality_rejects_duplicate_name(self):
        # Silently replacing a registered modality's encoder/decoder (the pre-fix behavior) let a
        # fresh, untrained replacement hide behind calibration computed against the OLD one.
        # add_modality() now rejects the duplicate outright; replace_modality() is the deliberate,
        # differently-named escape hatch, and it refuses a name that ISN'T already registered.
        from mixle.reason import CrossModalModel

        m = CrossModalModel(latent_dim=3, seed=0)
        m.add_modality("A", 4)
        with self.assertRaises(ValueError):
            m.add_modality("A", 4)
        with self.assertRaises(KeyError):
            m.replace_modality("nope", 4)

    def test_replace_modality_invalidates_fitted_and_calibration_state(self):
        # MXR-080-0276's own reproduction: fit, calibrate, then deliberately replace a modality's
        # encoder/decoder with a fresh, untrained pair. Pre-fix, _fitted and the stored conformal
        # radius survived the replacement, so predict_interval() kept advertising calibrated
        # coverage for a target whose predictions now depended on an untrained "A".
        from mixle.reason import CrossModalModel

        rng = np.random.RandomState(20)
        _, xA, xB = _two_view_data(rng, 300, k=2, dA=4, dB=3)
        m = CrossModalModel(latent_dim=3, seed=0)
        m.add_modality("A", 4).add_modality("B", 3)
        m.fit({"A": xA, "B": xB}, epochs=40, beta=0.3)
        m.calibrate({"A": xA[:150], "B": xB[:150]}, target="B", alpha=0.1)
        self.assertTrue(m._fitted)
        self.assertIn("B", m._conformal)

        m.replace_modality("A", 4)  # fresh, untrained encoder/decoder for "A"

        # none of the stale state may survive a deliberate structural replacement
        self.assertFalse(m._fitted)
        self.assertEqual(m._conformal, {})
        self.assertIsNone(m._n_train)
        with self.assertRaises(RuntimeError):
            m.belief({"A": xA[0]})
        with self.assertRaises(RuntimeError):
            m.predict_interval({"A": xA[0]}, target="B")

        # re-fitting and re-calibrating from here works normally again
        m.fit({"A": xA, "B": xB}, epochs=40, beta=0.3)
        m.calibrate({"A": xA[:150], "B": xB[:150]}, target="B", alpha=0.1)
        m.belief({"A": xA[0]})
        m.predict_interval({"A": xA[0]}, target="B")

    def test_add_modality_after_fit_also_invalidates_state(self):
        # _fitted describes the WHOLE model -- a brand-new (non-duplicate) modality registered
        # after fit() was never part of that joint training either, so adding structure invalidates
        # fitted/calibration state exactly like an explicit replace_modality() call does.
        from mixle.reason import CrossModalModel

        rng = np.random.RandomState(21)
        _, xA, xB = _two_view_data(rng, 200, k=2, dA=4, dB=3)
        m = CrossModalModel(latent_dim=3, seed=0)
        m.add_modality("A", 4).add_modality("B", 3)
        m.fit({"A": xA, "B": xB}, epochs=40, beta=0.3)
        m.calibrate({"A": xA[:100], "B": xB[:100]}, target="B", alpha=0.1)
        self.assertTrue(m._fitted)

        m.add_modality("C", 2)  # a genuinely new modality, not a duplicate

        self.assertFalse(m._fitted)
        self.assertEqual(m._conformal, {})
        with self.assertRaises(RuntimeError):
            m.belief({"A": xA[0]})

    # -- MXR-080-0277: fit() must validate the complete aligned training table --------------------
    def test_fit_rejects_misaligned_training_table_and_bad_epochs(self):
        from mixle.reason import CrossModalModel

        rng = np.random.RandomState(22)
        _, xA, xB = _two_view_data(rng, 200, k=2, dA=4, dB=3)

        def fresh():
            m = CrossModalModel(latent_dim=3, seed=0)
            return m.add_modality("A", 4).add_modality("B", 3)

        with self.assertRaises(ValueError):  # unequal row counts: A has 200, B has 50
            fresh().fit({"A": xA, "B": xB[:50]}, epochs=5)
        with self.assertRaises(ValueError):  # declared in_dim=4 for A but data has width 6
            fresh().fit({"A": rng.normal(size=(200, 6)), "B": xB}, epochs=5)
        with self.assertRaises(ValueError):  # empty training data
            fresh().fit({"A": xA[:0], "B": xB[:0]}, epochs=5)
        with self.assertRaises(ValueError):  # non-finite training data
            bad = xA.copy()
            bad[0, 0] = np.nan
            fresh().fit({"A": bad, "B": xB}, epochs=5)
        with self.assertRaises(ValueError):  # epochs=0 must not mark a random-init model fitted
            fresh().fit({"A": xA, "B": xB}, epochs=0)
        with self.assertRaises(ValueError):  # negative epochs
            fresh().fit({"A": xA, "B": xB}, epochs=-1)
        with self.assertRaises(TypeError):  # epochs must be a genuine int, not a float
            fresh().fit({"A": xA, "B": xB}, epochs=3.0)
        with self.assertRaises(TypeError):  # epochs must not be a bool (an int subclass)
            fresh().fit({"A": xA, "B": xB}, epochs=True)

    def test_fit_with_valid_aligned_table_still_succeeds(self):
        from mixle.reason import CrossModalModel

        rng = np.random.RandomState(23)
        _, xA, xB = _two_view_data(rng, 200, k=2, dA=4, dB=3)
        m = CrossModalModel(latent_dim=3, seed=0)
        m.add_modality("A", 4).add_modality("B", 3)
        m.fit({"A": xA, "B": xB}, epochs=5)
        self.assertTrue(m._fitted)
        self.assertEqual(m._n_train, 200)


if __name__ == "__main__":
    unittest.main()
