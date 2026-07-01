"""Tests for the mixle.reason front door — assimilation, attribution, prediction UQ."""

import unittest

import numpy as np

from mixle.reason import Evidence, Latent, reason


class ReasonBasicsTest(unittest.TestCase):
    def test_single_evidence_matches_kalman(self):
        prior = Latent.vector(1, var=4.0)
        ans = reason(prior, [Evidence(H=[[1.0]], y=[3.0], R=[[1.0]], name="obs")])
        self.assertAlmostEqual(float(ans.mean[0]), 4.0 / 5.0 * 3.0, places=10)
        self.assertAlmostEqual(float(ans.sd()[0]) ** 2, 4.0 / 5.0, places=10)
        self.assertGreater(ans.information_gain(), 0.0)

    def test_fusion_beats_either_modality_alone(self):
        # Two noisy linear views of a 2-vector latent; fusing both is tighter than each alone.
        prior = Latent.vector(2, var=10.0)
        e1 = Evidence(H=[[1.0, 0.0]], y=[2.0], R=[[1.0]], name="a")
        e2 = Evidence(H=[[0.0, 1.0]], y=[-1.0], R=[[1.0]], name="b")
        both = reason(prior, [e1, e2])
        just_a = reason(prior, [e1])
        self.assertLess(both.entropy(), just_a.entropy())
        # each coordinate is pinned by its own view
        np.testing.assert_allclose(both.mean, [2.0 * 10 / 11, -1.0 * 10 / 11], rtol=1e-6)

    def test_order_independence(self):
        prior = Latent.vector(2, var=5.0)
        e1 = Evidence([[1.0, 0.5]], [1.0], [[0.4]], "a")
        e2 = Evidence([[0.2, 1.0]], [2.0], [[0.6]], "b")
        ab = reason(prior, [e1, e2])
        ba = reason(prior, [e2, e1])
        np.testing.assert_allclose(ab.mean, ba.mean, atol=1e-10)
        np.testing.assert_allclose(ab.cov(), ba.cov(), atol=1e-10)


class AttributionTest(unittest.TestCase):
    def test_attribution_sums_to_total_gain(self):
        prior = Latent.vector(2, var=8.0)
        ev = [
            Evidence([[1.0, 0.0]], [1.0], [[0.5]], "gravity"),
            Evidence([[0.0, 1.0]], [2.0], [[2.0]], "magnetic"),
        ]
        ans = reason(prior, ev)
        attr = ans.attribution()
        self.assertEqual(set(attr), {"gravity", "magnetic"})
        self.assertAlmostEqual(sum(attr.values()), ans.information_gain(), places=10)
        # the lower-noise modality removes more uncertainty about its coordinate
        self.assertGreater(attr["gravity"], attr["magnetic"])

    def test_attribution_normalized(self):
        prior = Latent.vector(2, var=6.0)
        ans = reason(prior, [Evidence([[1.0, 0.0]], [1.0], [[0.5]], "a"), Evidence([[0.0, 1.0]], [1.0], [[0.5]], "b")])
        frac = ans.attribution(normalize=True)
        self.assertAlmostEqual(sum(frac.values()), 1.0, places=10)


class PredictionUQTest(unittest.TestCase):
    def test_predict_splits_epistemic_and_aleatoric(self):
        # After some evidence, predict a new readout y* = z0 + noise(R). Epistemic = posterior var of z0.
        prior = Latent.vector(1, var=9.0)
        ans = reason(prior, [Evidence([[1.0]], [4.0], [[1.0]], "obs")])
        post_var = float(ans.sd()[0]) ** 2  # posterior variance of the latent
        dec = ans.predict(H=[[1.0]], R=0.25)
        self.assertEqual(dec.kind, "variance")
        self.assertAlmostEqual(float(np.reshape(dec.epistemic, -1)[0]), post_var, places=10)
        self.assertAlmostEqual(float(np.reshape(dec.aleatoric, -1)[0]), 0.25, places=10)
        self.assertAlmostEqual(float(np.reshape(dec.total, -1)[0]), post_var + 0.25, places=10)

    def test_more_evidence_shrinks_epistemic_prediction_variance(self):
        prior = Latent.vector(1, var=9.0)
        one = reason(prior, [Evidence([[1.0]], [4.0], [[1.0]], "a")])
        two = reason(prior, [Evidence([[1.0]], [4.0], [[1.0]], "a"), Evidence([[1.0]], [4.2], [[1.0]], "b")])
        epi1 = float(np.reshape(one.predict([[1.0]], 0.5).epistemic, -1)[0])
        epi2 = float(np.reshape(two.predict([[1.0]], 0.5).epistemic, -1)[0])
        self.assertLess(epi2, epi1)


class QueryTest(unittest.TestCase):
    def test_query_restricts_to_subset(self):
        prior = Latent.vector(3, var=4.0)
        ev = [Evidence(np.eye(3), [1.0, 2.0, 3.0], np.eye(3) * 0.5, "full")]
        ans = reason(prior, ev, query=[1])
        self.assertEqual(np.size(ans.mean), 1)
        self.assertAlmostEqual(float(ans.mean[0]), 2.0 * 4.0 / 4.5, places=6)


if __name__ == "__main__":
    unittest.main()
