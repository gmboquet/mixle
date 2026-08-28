"""T1-04: GammaEstimator's hard shape-parameter ceiling (_MAX_GAMMA_SHAPE = 1e12) was never
disclosed through numerical_repairs()/fit_provenance(), unlike GaussianEstimator's analogous
variance floor. A near-zero-coefficient-of-variation sample silently clamped k at the ceiling
with an empty numerical_repairs() tuple, so a caller reading only (k, theta) had no way to tell
the fit hit a hard-coded wall rather than a converged estimate.
"""

import unittest

import numpy as np

from mixle.inference import estimate, optimize
from mixle.stats import GammaDistribution, GammaEstimator
from mixle.stats.univariate.continuous.gamma import _MAX_GAMMA_SHAPE


class GammaShapeCeilingDisclosureTest(unittest.TestCase):
    def test_single_observation_clamp_is_disclosed(self):
        # Exact repro from the filed finding: a single draw has zero sample coefficient of
        # variation (sum_of_logs = log(mean) exactly), driving the bisection solver's degenerate
        # "s <= 0" branch, which used to return the hard ceiling with no disclosure at all.
        true_dist = GammaDistribution(3.0, 2.0)
        data = true_dist.sampler(seed=1).sample(1).tolist()

        fit = estimate(data, GammaEstimator())
        self.assertEqual(fit.k, _MAX_GAMMA_SHAPE)
        # k*theta must still match the single data point exactly -- the clamp's numeric behavior
        # is unchanged by this fix.
        self.assertAlmostEqual(fit.k * fit.theta, data[0], places=6)

        repairs = fit.numerical_repairs()
        self.assertTrue(repairs, "shape-ceiling clamp must be disclosed via numerical_repairs()")
        self.assertTrue(any("shape" in note and "1e+12" in note for note in repairs), repairs)

    def test_optimize_fit_provenance_also_carries_the_repair(self):
        # optimize()/fit_provenance() reads numerical_repairs() off the fitted model -- confirm the
        # disclosure survives the full EM path, not just the bare estimator.estimate() call.
        true_dist = GammaDistribution(3.0, 2.0)
        data = true_dist.sampler(seed=1).sample(1).tolist()

        fitted = optimize(data, GammaEstimator(), max_its=3, out=None)
        provenance = fitted.fit_provenance()
        self.assertTrue(provenance.repairs, "FitProvenance.repairs must carry the shape-ceiling note")
        self.assertTrue(any("shape" in note for note in provenance.repairs), provenance.repairs)

    def test_near_zero_cv_clamp_from_a_realistic_low_noise_sample_is_disclosed(self):
        # Not an n=1-only artifact: a realistic low-noise-instrument sample (CV ~1e-5) also drives
        # the shape solver to exhaust its doubling loop at the ceiling with s > 0, taking the
        # "shape-ceiling-clamped" branch rather than the "shape-unresolvable" (s <= 0) branch.
        rs = np.random.RandomState(2)
        data = (100.0 + rs.normal(0, 1e-7, 30)).tolist()

        fit = estimate(data, GammaEstimator())
        self.assertEqual(fit.k, _MAX_GAMMA_SHAPE)
        self.assertTrue(
            any(note.startswith("shape-ceiling-clamped(") for note in fit.numerical_repairs()),
            fit.numerical_repairs(),
        )

    def test_ordinary_well_scaled_fit_is_unaffected_and_undisclosed(self):
        # A well-scaled, non-degenerate fit must recover the generating parameters exactly as
        # before this fix, with no repair recorded (the guard is scoped to the actual ceiling).
        true_k, true_theta = 3.0, 2.0
        true_dist = GammaDistribution(true_k, true_theta)
        data = true_dist.sampler(seed=7).sample(500).tolist()

        fit = estimate(data, GammaEstimator())
        self.assertLess(fit.k, _MAX_GAMMA_SHAPE)
        self.assertEqual(fit.numerical_repairs(), ())
        self.assertAlmostEqual(fit.k, true_k, delta=0.5)
        self.assertAlmostEqual(fit.theta, true_theta, delta=0.5)

    def test_moderately_large_but_unclamped_shape_stays_undisclosed(self):
        # A genuinely converged large shape (below the ceiling) is not a "repair" -- it is the
        # solver's real answer for a low-but-nonzero-CV sample, and must not be flagged.
        rs = np.random.RandomState(2)
        data = (100.0 + rs.normal(0, 0.001, 30)).tolist()

        fit = estimate(data, GammaEstimator())
        self.assertLess(fit.k, _MAX_GAMMA_SHAPE)
        self.assertGreater(fit.k, 1.0e9)
        self.assertEqual(fit.numerical_repairs(), ())


if __name__ == "__main__":
    unittest.main()
