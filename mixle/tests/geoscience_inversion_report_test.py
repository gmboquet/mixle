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

        from mixle.reason.language_bridge import PosteriorDescriber
        from mixle.task.calibrated_generator import smallest_certifiable_calibration_set

        # ALPHA=0.1 is only reachable above a calibration size fixed by the certificate itself, not by the
        # model: calibrate() certifies on half the set with a Bonferroni-corrected Clopper-Pearson bound
        # whose zero-error floor is 1 - tail**(1/c) for c certification rows. This test used n=60 -> c=30,
        # where that floor is 0.1157 > 0.1, so qhat was +inf and describe() abstained on EVERY input at
        # EVERY tol, regardless of the inversion's quality. A previous revision read that as a scoring
        # tie and tuned tol and alpha against it across seed sweeps; no setting could have served a claim.
        ALPHA = 0.1
        n_cal = max(160, smallest_certifiable_calibration_set(ALPHA))
        calibration_set = build_calibration_set(inv_model, depth_prior, n=n_cal, seed=999)
        self.assertEqual(len(calibration_set), n_cal)

        def describer_at(tol):
            d = PosteriorDescriber(
                "depth_km", tol=tol, k=3, alpha=ALPHA, width_multiples=(1.0, 3.0, 10.0), n_probe=300, seed=0
            )
            d.calibrate(calibration_set, seed=0)
            return d

        # The describer is selective, so the receipt has to show BOTH directions or it shows nothing.
        #
        # Abstain at a precision the inversion cannot deliver. tol=0.2 asks for +/-0.2 km; measured on this
        # calibration set the served claim misses the truth on 0.275 of rows (median |center - truth| is
        # 0.136 with a tail to 0.688), so a 10%-risk gate must refuse. That the posterior is sharp does not
        # help: claim_score is coverage-per-unit-width against the posterior's OWN draws, so at std ~0.001
        # it saturates to exactly 1.0/width -- an identical 2.5 on every row -- and carries no information
        # about whether the posterior is centered on the truth. A confidently mis-centered inversion is
        # indistinguishable from a correct one by that statistic, which is precisely why the risk
        # certificate, not the score, is what decides here.
        tight = describer_at(0.2)
        # no per-call seed: certified describers refuse schedule overrides (STAT-RR17-07)
        self.assertIsNone(tight.describe(inv_model.posterior(y_obs)))
        self.assertFalse(tight._gen.risk_receipt["target_attainable"] is False)  # size is not the blocker now

        # Serve at a precision it can. tol=0.5 misses on 0.006 of rows and certifies at 0.077 <= 0.1.
        loose = describer_at(0.5)
        claim = loose.describe(inv_model.posterior(y_obs))
        self.assertIsNotNone(claim, msg=str(loose._gen.risk_receipt))
        self.assertTrue(claim.contains(TRUE_DEPTH))
        self.assertLessEqual(loose._gen.risk_receipt["error_upper"], ALPHA)  # the served claim is receipted

    @unittest.skipUnless(HAS_TORCH, "main() calls invert_new_observation, which requires torch")
    def test_main_runs_end_to_end(self):
        main()  # exercises the full sense -> simulate -> invert -> report loop; asserts internally


if __name__ == "__main__":
    unittest.main()
