"""B7: sense -> simulate -> invert -> report -- the track-M full-loop demo (M0/M2/M3/M5/A1)."""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

HAS_TORCH = importlib.util.find_spec("torch") is not None

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
from geoscience_inversion_report import (  # noqa: E402
    SENSOR_NOISE,
    TRUE_DEPTH,
    TRUE_FORMATION,
    _amplitude,
    build_calibration_set,
    fit_joint,
    invert_new_observation,
    main,
    sense,
    what_if_salt,
)

from mixle.stats.univariate.continuous.gaussian import GaussianDistribution  # noqa: E402


class SenseSimulateInvertReportTest(unittest.TestCase):
    def setUp(self):
        self.records = sense(600, seed=0)
        self.net = fit_joint(self.records)

    def test_fit_joint_recovers_the_shared_latent_structure(self):
        # field 0 (formation) is the root latent driving both field 1 (amplitude) and field 2 (depth)
        by_child = {f.child: f for f in self.net.factors}
        self.assertEqual(len(self.net.factors), 3)
        self.assertEqual(list(by_child[0].parents), [])
        self.assertIn(0, by_child[1].parents)  # amplitude conditions on formation
        self.assertIn(0, by_child[2].parents)  # depth conditions on formation

    def test_m2_what_if_rolls_out_the_salt_regime(self):
        sim, depths, amps = what_if_salt(self.net, seed=1)
        salt_mean = 4.5  # FORMATION_PARAMS["salt"]'s generative depth mean
        # the do(formation="salt") rollout should land near salt's own generative depth law
        self.assertLess(abs(depths.mean() - salt_mean), 0.3)
        self.assertGreater(depths.std(), 0.0)
        self.assertEqual(amps.shape[1], 3)
        self.assertEqual(sim.receipt.method, "none")  # intervention only, no evidence to condition on

    @unittest.skipUnless(HAS_TORCH, "invert_new_observation's learn_inverse requires torch")
    def test_m3_inverts_a_new_observation_close_to_the_true_depth(self):
        sim, wi_depths, _wi_amps = what_if_salt(self.net, seed=1)
        depth_prior = GaussianDistribution(mu=float(wi_depths.mean()), sigma2=float(wi_depths.var()))
        obs_rng = np.random.RandomState(123)
        y_obs = np.asarray(_amplitude(TRUE_DEPTH, TRUE_FORMATION), dtype=float) + SENSOR_NOISE * obs_rng.randn(3)

        inv_model = invert_new_observation(depth_prior, y_obs, seed=9)
        post_samples = inv_model.posterior(y_obs).sample(2000, seed=5)
        self.assertLess(abs(float(post_samples.mean()) - TRUE_DEPTH), 0.3)
        # calibration receipts are always computed, whether or not they pass
        self.assertIsInstance(inv_model.receipts.sbc_pvalue, float)
        self.assertIn(0.9, inv_model.receipts.coverage)

    @unittest.skipUnless(HAS_TORCH, "invert_new_observation's learn_inverse requires torch")
    def test_m5_report_serves_a_claim_that_brackets_the_truth(self):
        sim, wi_depths, _wi_amps = what_if_salt(self.net, seed=1)
        depth_prior = GaussianDistribution(mu=float(wi_depths.mean()), sigma2=float(wi_depths.var()))
        obs_rng = np.random.RandomState(123)
        y_obs = np.asarray(_amplitude(TRUE_DEPTH, TRUE_FORMATION), dtype=float) + SENSOR_NOISE * obs_rng.randn(3)

        inv_model = invert_new_observation(depth_prior, y_obs, seed=9)
        calibration_set = build_calibration_set(inv_model, depth_prior, n=60, seed=999)
        self.assertEqual(len(calibration_set), 60)

        from mixle.reason.language_bridge import PosteriorDescriber

        describer = PosteriorDescriber(
            "depth_km", tol=0.1, k=3, alpha=0.2, width_multiples=(1.0, 3.0, 10.0), n_probe=300, seed=0
        )
        describer.calibrate(calibration_set, seed=0)
        claim = describer.describe(inv_model.posterior(y_obs), seed=0)
        self.assertIsNotNone(claim)  # a calibrated candidate clears the threshold for this seeded run
        self.assertTrue(claim.contains(TRUE_DEPTH))

    @unittest.skipUnless(HAS_TORCH, "main() calls invert_new_observation, which requires torch")
    def test_main_runs_end_to_end(self):
        main()  # exercises the full sense -> simulate -> invert -> report loop; asserts internally


if __name__ == "__main__":
    unittest.main()
