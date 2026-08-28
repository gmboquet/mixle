"""T1-01/T4-01: two compounding gaps where "anchored is None" silently discards a usable signal.

T1-01: `dist.estimator(pseudo_count=...)` with ZERO raw observations reaches `estimate()`'s raw
(non-anchored) branch, which used to recompute the prior's variance by differencing the
prior-blended sums (`sum_x2/count - mean_x**2`) at threshold magnitude, ignoring the exact,
cancellation-free `prior_variance` already sitting in scope. At an ordinary threshold the
recomputation happens to survive; at an epoch-scale threshold (~1.7e9) it does not, and a prior of
(scale=2.0, shape=-0.3) came back sign-flipped and clamped near the shape ceiling with zero
observations blended in and zero raw signal to blame it on. This is the exact `estimator.estimate
(nobs, accumulator.value())` call convention used by every EM M-step, so a mixture/HMM component
that receives zero responsibility in a batch (or a prior-only cold start) hits this path directly.

T4-01: `GeneralizedParetoAccumulator.combine()` on a foreign/plain-tuple partial (no `.max_x`
payload) at a large-magnitude loc taints the tracked max, which is supposed to make `estimate()`
disclose the ambiguity via `numerical_repairs()` (T1-02/D-0206). That disclosure only fired for
`xi < -_XI_TOL`; the very same raw-moment cancellation that makes the fit untrustworthy can just as
easily push the computed shape to the *positive* boundary instead, or collapse `var` to non-positive
and hit the *earlier* degenerate short-circuit -- both bypassed the disclosure entirely and returned
`numerical_repairs() == ()` for a badly wrong fit, with only a transient RuntimeWarning as any signal.
"""

import unittest
import warnings

from mixle.stats import GeneralizedParetoDistribution


class GeneralizedParetoPriorZeroObservationsTest(unittest.TestCase):
    def test_pseudo_count_prior_with_zero_observations_matches_across_threshold_magnitude(self):
        # Exact repro from the filed finding: a prior-carrying estimator, an untouched accumulator
        # (acc.value() == (0.0, 0.0, 0.0), a plain tuple, zero raw observations), at loc=0 vs an
        # epoch-scale loc. Both must reproduce the prior almost exactly, matching each other.
        def fit_at(loc):
            dist = GeneralizedParetoDistribution(scale=2.0, shape=-0.3, loc=loc)
            est = dist.estimator(pseudo_count=50.0)
            acc = est.accumulator_factory().make()
            val = acc.value()
            self.assertEqual(val, (0.0, 0.0, 0.0))
            return est.estimate(None, val)

        with warnings.catch_warnings():
            warnings.simplefilter("error")  # the zero-observation, prior-only path must not warn
            fitted_near = fit_at(0.0)
            fitted_far = fit_at(1_700_000_000.0)

        self.assertAlmostEqual(fitted_near.scale, 2.0, places=6)
        self.assertAlmostEqual(fitted_near.shape, -0.3, places=6)
        # Before the fix: scale=0.771..., shape=+0.4988... (sign-flipped, clamped near xi_max).
        self.assertAlmostEqual(fitted_far.scale, 2.0, places=4)
        self.assertAlmostEqual(fitted_far.shape, -0.3, places=4)
        self.assertLess(fitted_far.shape, 0.0)

    def test_blended_real_observations_are_unaffected_by_the_zero_observation_fix(self):
        # n=1,2,3 real observations blended with the same prior already matched across threshold
        # magnitude before this fix (they go through the anchored branch, not the touched raw
        # branch) -- must keep matching, not regress.
        base = GeneralizedParetoDistribution(scale=1.5, shape=-0.2, loc=0.0)
        offsets = [float(base.sampler(seed=7).sample()) for _ in range(3)]

        def fit_with_n(n, loc):
            prior = GeneralizedParetoDistribution(scale=2.0, shape=-0.3, loc=loc)
            est = prior.estimator(pseudo_count=50.0)
            acc = est.accumulator_factory().make()
            for off in offsets[:n]:
                acc.update(loc + off, 1.0, None)
            return est.estimate(float(n), acc.value())

        for n in (1, 2, 3):
            near = fit_with_n(n, 0.0)
            far = fit_with_n(n, 1_700_000_000.0)
            self.assertAlmostEqual(near.scale, far.scale, places=3)
            self.assertAlmostEqual(near.shape, far.shape, places=3)


class GeneralizedParetoForeignCombineCancellationDisclosureTest(unittest.TestCase):
    def _foreign_combine_at(self, loc, seed, prior_scale=2.0, prior_shape=-0.3, pseudo_count=50.0):
        import random

        rng = random.Random(seed)
        data = [loc + rng.gammavariate(2.0, 1.5) for _ in range(40)]
        dist = GeneralizedParetoDistribution(scale=prior_scale, shape=prior_shape, loc=loc)
        est = dist.estimator(pseudo_count=pseudo_count)
        acc = est.accumulator_factory().make()
        s = sum(data)
        s2 = sum(x * x for x in data)
        n = float(len(data))
        acc.combine((s, s2, n))  # foreign plain tuple: no .max_x, no .anchored payload
        val = acc.value()
        self.assertTrue(getattr(val, "max_unverified", False))
        self.assertIsNone(getattr(val, "max_x", "missing"))
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            return est.estimate(n, val)

    def test_foreign_combine_cancellation_is_disclosed_even_when_shape_lands_non_negative(self):
        # Exact repro from the filed finding: cancellation at loc~1.7e9 pushed the fitted shape to
        # the positive clamp boundary instead of negative, which the pre-fix `xi < -_XI_TOL` gate on
        # the disclosure branch let through with numerical_repairs() == ().
        m = self._foreign_combine_at(1_700_000_000.0, seed=0)
        self.assertGreaterEqual(m.shape, 0.0)
        self.assertTrue(any("support-consistency-unverified" in note for note in m.numerical_repairs()))

    def test_foreign_combine_cancellation_is_disclosed_through_the_degenerate_shortcircuit(self):
        # A seed where raw-moment cancellation collapses var to non-positive and hits the EARLIER
        # `m <= 0.0 or var <= 0.0` degenerate return, which used to bypass the disclosure entirely.
        m = self._foreign_combine_at(1_700_000_000.0, seed=95)
        self.assertEqual(m.shape, 0.0)
        self.assertTrue(any("support-consistency-unverified" in note for note in m.numerical_repairs()))

    def test_ordinary_small_magnitude_foreign_combine_stays_quiet(self):
        # Regression guard: the broadened disclosure must not fire just because max_unverified is
        # True -- only when raw-moment cancellation actually occurred. At loc=0, with a weak prior
        # that does not pull the blended shape negative, there is neither cancellation nor a
        # negative-shape clamp to disclose (mirrors the pinned
        # test_foreign_combine_that_never_needs_the_clamp_stays_quiet expectation).
        m = self._foreign_combine_at(0.0, seed=0, prior_scale=1.0, prior_shape=0.5, pseudo_count=1.0)
        self.assertGreaterEqual(m.shape, 0.0)
        self.assertEqual(m.numerical_repairs(), ())

    def test_ordinary_small_magnitude_foreign_combine_with_negative_pull_still_only_discloses_via_shape(self):
        # At loc=0 (no cancellation possible) a prior that DOES pull the blended shape negative must
        # still disclose -- via the pre-existing T1-02 xi<0 path, not the new cancellation path --
        # confirming the broadened `or raw_ill_conditioned` clause did not change this pinned case.
        m = self._foreign_combine_at(0.0, seed=0, prior_scale=2.0, prior_shape=-0.3, pseudo_count=50.0)
        self.assertLess(m.shape, 0.0)
        self.assertTrue(any("support-consistency-unverified" in note for note in m.numerical_repairs()))


if __name__ == "__main__":
    unittest.main()
