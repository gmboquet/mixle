"""T1-01: optimize()/fit() on GeneralizedParetoEstimator crashed on realistic negative-shape data.

A shape<0 method-of-moments fit only matches the first two moments, so its implied support upper
endpoint (loc - scale/shape) could legitimately sit below the training data's own max -- estimate()
itself succeeded with such a fit. The next optimize()/fit() EM iteration then re-validated the SAME
training data against that self-inconsistent model and raised ValueError on ordinary, realistic
peaks-over-threshold data (100% crash rate whenever the precondition occurred, per the campaign
sweep). GeneralizedParetoEstimator.estimate() now floors scale so its own fit is always consistent
with the data it was fit from, disclosed via numerical_repairs().
"""

import unittest
import warnings

import numpy as np

from mixle.inference import estimate, fit, optimize
from mixle.stats import GeneralizedParetoDistribution, GeneralizedParetoEstimator


class GeneralizedParetoSelfInconsistentRefitTest(unittest.TestCase):
    def test_optimize_does_not_crash_on_the_filed_repro(self):
        # Exact repro from the filed finding: a threshold-1.0 exceedance sample whose raw
        # method-of-moments fit has an implied upper endpoint below the sample's own max.
        rs = np.random.RandomState(9)
        data = (1.0 + rs.uniform(0.0, 1.0e-4, 3000)).tolist()
        est = GeneralizedParetoEstimator(loc=1.0)

        m1 = estimate(data, est)
        self.assertLess(m1.shape, 0.0)
        self.assertGreaterEqual(m1._upper(), max(data))  # estimate() itself is now self-consistent

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # unrelated unconverged-fit note at max_its=1
            m2 = optimize(data, est, max_its=1)  # used to raise ValueError on the very first E-step
        self.assertGreaterEqual(m2._upper(), max(data))

    def test_optimize_does_not_crash_on_a_genuine_peaks_over_threshold_draw(self):
        # A real draw from a bounded (xi<0) GPD, not the uniform-data example above, at an ordinary
        # sample size -- seed=2, n=20 from the finding's own 60-seed sweep, which crashed with the
        # identical ValueError before this fix.
        true_dist = GeneralizedParetoDistribution(1.0, -0.3, loc=0.0)
        data = true_dist.sampler(seed=2).sample(20).tolist()
        est = GeneralizedParetoEstimator(loc=0.0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = optimize(data, est, max_its=3)
        self.assertGreaterEqual(m._upper(), max(data))

    def test_fit_discloses_the_scale_floor_that_avoided_the_crash(self):
        rs = np.random.RandomState(9)
        data = (1.0 + rs.uniform(0.0, 1.0e-4, 3000)).tolist()
        est = GeneralizedParetoEstimator(loc=1.0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = fit(data, est, max_its=5)
        self.assertTrue(any("scale-floored-for-support" in note for note in m.numerical_repairs()))

    def test_a_pseudo_count_prior_that_pushes_the_blended_shape_negative_does_not_crash(self):
        # A gap an adversarial re-review found in the fix above: the accumulator used to decide
        # whether to carry the observation max forward from the RAW moments alone, dropping it
        # whenever raw data implied shape>=0 -- invisible to a pseudo_count prior blended in later
        # by estimate(). Raw data here is ordinary and heavy-tailed (implied shape > 0 on its own),
        # but a strong, tightly-bounded-tail prior blends the FINAL shape negative, reproducing the
        # identical crash class T1-01 was written to eliminate, just reached through the prior
        # instead of the raw data.
        raw_dist = GeneralizedParetoDistribution(1.0, 0.5, loc=0.0)  # genuinely heavy-tailed (xi>0)
        data = raw_dist.sampler(seed=3).sample(40).tolist()
        raw_only_fit = estimate(data, GeneralizedParetoEstimator(loc=0.0))
        self.assertGreater(raw_only_fit.shape, 0.0)  # raw data alone: no clamp would ever be needed

        prior = GeneralizedParetoDistribution(1.0, -0.9, loc=0.0)
        est = prior.estimator(pseudo_count=2000.0)

        m1 = estimate(data, est)
        self.assertLess(m1.shape, 0.0)  # the prior, not the data, drives shape negative
        self.assertGreaterEqual(m1._upper(), max(data))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m2 = optimize(data, est, max_its=3)  # used to crash with the prior blend before this fix
        self.assertGreaterEqual(m2._upper(), max(data))

    def test_ordinary_positive_shape_fit_is_unaffected(self):
        # A heavy-tailed (xi>0) fit has infinite support and can never trip the clamp; optimize()
        # must recover the generating parameters just as it did before this fix.
        true_dist = GeneralizedParetoDistribution(2.0, 0.3, loc=0.0)
        data = true_dist.sampler(seed=11).sample(5000).tolist()
        est = GeneralizedParetoEstimator(loc=0.0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = optimize(data, est, max_its=10)
        self.assertEqual(m.numerical_repairs(), ())
        self.assertAlmostEqual(m.scale, 2.0, delta=0.2)
        self.assertAlmostEqual(m.shape, 0.3, delta=0.08)


if __name__ == "__main__":
    unittest.main()
