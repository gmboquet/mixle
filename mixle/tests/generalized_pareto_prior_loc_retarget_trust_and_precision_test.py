"""Two more findings in the GeneralizedPareto prior-moments mechanism D-0207/T1-01 already patched
twice, both confirmed independently against the built wheel (campaign eight).

FINDING A: the ``loc`` setter permanently silences the prior-variance cancellation warning after one
retarget, without fixing the underlying corruption. A manually-supplied plain ``suff_stat`` 2-tuple
at a large-magnitude ``loc`` correctly triggers ``_prior_variance()``'s cancellation ``RuntimeWarning``
on every ``.estimate()`` call -- until ``.loc`` is reassigned even once (the documented "threshold
stability" re-thresholding workflow), at which point the setter used to unconditionally re-wrap
whatever ``_prior_variance()`` returned as a permanently "carried"/trusted value, laundering a value
that had just triggered a warning into one that would never warn again. The fit was exactly as wrong
after the retarget as before it; only the diagnostic disappeared. Fixed by only marking the new
payload's variance as carried/trusted when :func:`_prior_variance_is_carried` says the SOURCE value
already could be vouched for -- otherwise the new payload is left un-carried, so every subsequent
``.estimate()`` call keeps re-deriving (and, since the pair still cannot support the variance it
implies, keeps warning), matching the untouched-``loc`` control's behavior.

FINDING B: the same mechanism loses the prior's MEAN to precision loss at extreme ``loc`` magnitude,
silently. D-0207/T1-01 protected the prior's VARIANCE from cancellation at large ``loc`` by carrying
an exact payload; the MEAN half of the same ``(mean, second_moment)`` pair was never given the same
protection -- ``mean0`` was stored as an absolute float (``loc`` plus a small ``O(1)`` exceedance
offset), which rounds the offset away to ``loc``'s own ULP once ``loc`` is large enough (negligible
through ~1e12, meaningfully wrong by 1e14-1e16, a full silent collapse to the pre-T1-01 degenerate
default -- shape flattened to 0, scale pinned to the ``min_scale`` floor -- from about 1e17 upward,
which covers real nanosecond Unix epoch timestamps). The pinned T1-01 regression test only exercises
``loc~1.7e9`` (epoch SECONDS), safely inside the unaffected range. Fixed by capturing the exceedance
offset (``mean - loc``) exactly, at construction (directly as ``scale/(1-xi)``, with no dependence on
``loc`` at all) or via a single Sterbenz-exact subtraction of the current pair, carrying it forward
UNCHANGED across any number of ``loc`` retargets, and having ``estimate()`` consume that preserved
offset directly wherever it used to recompute ``mean_x - self.loc`` (or, in the anchored branch,
``prior_mean - anchor``) from an already-corrupted absolute value.
"""

import unittest
import warnings

from mixle.stats import GeneralizedParetoDistribution, GeneralizedParetoEstimator


def _warned_about_second_moment(est, suff_stat=(0.0, 0.0, 0.0)):
    """Whether one ``.estimate()`` call raises the ``_prior_variance`` cancellation RuntimeWarning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        est.estimate(None, suff_stat)
    return any("second_moment" in str(w.message) for w in caught)


class PriorVarianceTrustSurvivesRetargetTest(unittest.TestCase):
    """FINDING A: a retarget must not launder an unproven fallback variance into a trusted one."""

    BIG_LOC = 1_700_000_000.0

    def _bad_pair_estimator(self, loc=BIG_LOC):
        # A manually-supplied plain 2-tuple whose second moment implies exactly zero variance
        # (`second0 == mean0**2`) -- a prior no generalized Pareto with this exceedance mean and the
        # default `xi_min=-10.0` can actually have, so `_prior_variance`'s floor check must reject it.
        mean0 = loc + 2.5
        second0 = mean0 * mean0
        return GeneralizedParetoEstimator(loc=loc, pseudo_count=50.0, suff_stat=(mean0, second0))

    def test_control_untouched_estimator_warns_on_every_call(self):
        # Baseline: with no retarget at all, the plain-tuple prior must warn on every independent
        # `.estimate()` call -- this is the behavior a retarget must not be able to switch off.
        est = self._bad_pair_estimator()
        for _ in range(3):
            self.assertTrue(_warned_about_second_moment(est))

    def test_warns_on_every_call_before_any_retarget(self):
        est = self._bad_pair_estimator()
        self.assertTrue(_warned_about_second_moment(est))
        self.assertTrue(_warned_about_second_moment(est))

    def test_retargeting_to_a_small_well_conditioned_loc_does_not_extinguish_the_warning(self):
        # Exact repro from the filed finding: retarget once, to a threshold small enough that a
        # FRESH construction there would be exact -- the pre-fix bug baked the fallback variance in
        # as permanently "trusted" regardless, silencing the warning even though this pair is exactly
        # as wrong as it was before the retarget.
        est = self._bad_pair_estimator()
        self.assertTrue(_warned_about_second_moment(est))  # before: warns (sanity check)

        est.loc = 0.0  # retarget to a small, well-conditioned threshold

        self.assertTrue(_warned_about_second_moment(est), "warning must survive a retarget to a good loc")
        self.assertTrue(_warned_about_second_moment(est), "warning must keep firing on every later call, not just once")

    def test_genuine_noop_retarget_does_not_extinguish_the_warning(self):
        # `est.loc = est.loc` changes nothing about the threshold, yet the pre-fix setter still
        # unconditionally rebuilt (and re-trusted) the payload on every assignment, retarget or not.
        est = self._bad_pair_estimator()
        self.assertTrue(_warned_about_second_moment(est))

        est.loc = est.loc  # no-op re-target: same value back in

        self.assertTrue(_warned_about_second_moment(est), "a no-op `.loc` round trip must not launder the warning away")
        self.assertTrue(_warned_about_second_moment(est))

    def test_double_retarget_bad_good_bad_keeps_warning_throughout(self):
        # bad -> good -> bad: two retargets must not accumulate into a trusted value either.
        est = self._bad_pair_estimator()
        self.assertTrue(_warned_about_second_moment(est))

        est.loc = 0.0
        self.assertTrue(_warned_about_second_moment(est))

        est.loc = self.BIG_LOC
        self.assertTrue(_warned_about_second_moment(est), "a second retarget back to a bad loc must still warn")
        self.assertTrue(_warned_about_second_moment(est))

    def test_a_library_built_prior_still_never_warns_across_retargets(self):
        # Regression guard on the fix itself: a prior built via `.estimator(pseudo_count)` carries an
        # EXACT variance throughout, at any loc -- retargeting it must not start spuriously warning.
        dist = GeneralizedParetoDistribution(scale=2.0, shape=-0.3, loc=0.0)
        est = dist.estimator(pseudo_count=50.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            est.estimate(None, (0.0, 0.0, 0.0))
            est.loc = self.BIG_LOC
            est.estimate(None, (0.0, 0.0, 0.0))
            est.loc = 3.0
            est.estimate(None, (0.0, 0.0, 0.0))


class PriorMeanPrecisionAcrossMagnitudeTest(unittest.TestCase):
    """FINDING B: the prior's exceedance mean must survive threshold magnitude, like its variance."""

    SWEEP_LOCS = (1.0e9, 1.0e10, 1.0e11, 1.0e12, 1.0e13, 1.0e14, 1.0e15, 1.0e16, 1.0e17, 1.0e18)

    def test_pure_prior_sweep_stays_accurate_from_1e9_through_1e18(self):
        # Exact repro from the filed finding: `.estimator(pseudo_count=...)` with ZERO raw
        # observations, swept far past the pinned T1-01 test's single ~1.7e9 point. Before the fix
        # this degrades steadily (measurably wrong by 1e14-1e16) and fully collapses to the
        # pre-T1-01 degenerate default (shape=0.0, scale=min_scale=1e-12) from about 1e17 upward.
        for scale, shape in ((2.0, -0.3), (1.5, 0.2)):
            for loc in self.SWEEP_LOCS:
                with self.subTest(scale=scale, shape=shape, loc=loc):
                    dist = GeneralizedParetoDistribution(scale=scale, shape=shape, loc=loc)
                    est = dist.estimator(pseudo_count=50.0)
                    with warnings.catch_warnings():
                        warnings.simplefilter("error")  # zero-observation, prior-only: must stay quiet
                        fitted = est.estimate(None, (0.0, 0.0, 0.0))
                    self.assertAlmostEqual(fitted.scale, scale, places=8, msg="loc=%.0e" % loc)
                    self.assertAlmostEqual(fitted.shape, shape, places=8, msg="loc=%.0e" % loc)
                    # The mean half of the fix, stated directly: the fitted exceedance mean (mean()
                    # minus the threshold) must match the prior's own, computed the same way.
                    self.assertAlmostEqual(
                        fitted.mean() - loc, dist.mean() - loc, places=6, msg="exceedance mean at loc=%.0e" % loc
                    )

    def test_loc_setter_retarget_with_one_real_observation_stays_accurate(self):
        # "Also test via the .loc setter re-anchoring path specifically (build at a small, exact loc;
        # retarget to an extreme one), and confirm one real observation drawn from the prior does not
        # mask the corruption." A single real observation activates GeneralizedParetoAccumulator's
        # anchored track unconditionally (T4-01's own weight-gating aside), so this exercises the
        # ANCHORED branch of estimate() -- not just the pure-prior raw branch above -- with the
        # prior's own (now precision-protected) contribution blended in.
        #
        # Ground truth is a baseline built AND fit entirely at loc=0.0, with the SAME relative
        # exceedance for the one real observation: the method-of-moments fit is threshold-equivariant
        # (see test_a_library_built_prior_is_threshold_equivariant in campaign4_shift_sweep_test.py),
        # so a correct retarget must reproduce it, while the pre-fix corruption does not.
        scale, shape, pseudo_count, exceedance = 2.0, -0.3, 2000.0, 3.0

        baseline_est = GeneralizedParetoDistribution(scale, shape, loc=0.0).estimator(pseudo_count)
        baseline_acc = baseline_est.accumulator_factory().make()
        baseline_acc.update(exceedance, 1.0, None)
        baseline = baseline_est.estimate(1.0, baseline_acc.value())

        for loc in (1.0e12, 1.0e13, 1.0e14, 1.0e15, 1.0e16):
            with self.subTest(loc=loc):
                est = GeneralizedParetoDistribution(scale, shape, loc=0.0).estimator(pseudo_count)
                est.loc = loc  # retarget the SAME estimator to an extreme threshold

                acc = est.accumulator_factory().make()
                acc.update(loc + exceedance, 1.0, None)  # one real observation drawn from the prior
                fitted = est.estimate(1.0, acc.value())

                # Before the fix this drifts steadily off the threshold-equivariant baseline (percent-
                # to-order-of-magnitude scale error by loc=1e14-1e16, e.g. shape off by 184% at
                # 1e16); a RELATIVE tolerance is used, loose enough that a single real observation's
                # own contribution to the anchored branch's pooled variance -- itself limited by the
                # same loc-scale ULP floor everywhere else in this mechanism accepts, see
                # anchored_pooled_variance's MEAN_ROUNDING_BOUND, and not the corruption under test --
                # does not itself trip it, while the pre-fix error is 10-1000x past it throughout.
                rel_tol = 0.01
                self.assertLess(
                    abs(fitted.scale - baseline.scale) / abs(baseline.scale), rel_tol, msg="scale at loc=%.0e" % loc
                )
                self.assertLess(
                    abs(fitted.shape - baseline.shape) / abs(baseline.shape), rel_tol, msg="shape at loc=%.0e" % loc
                )

    def test_manually_supplied_pair_offset_survives_a_retarget_even_though_variance_cannot(self):
        # A plain 2-tuple has no exact side channel for either quantity, but the mean OFFSET is
        # always recoverable post-hoc by a single Sterbenz-exact subtraction (no plausibility floor
        # needed, unlike the variance) -- so a retarget must not lose it even when the pair's
        # variance is simultaneously destroyed and correctly warned about (Finding A above).
        big_loc = 1_700_000_000.0
        mean0 = big_loc + 2.5
        second0 = mean0 * mean0  # variance destroyed by construction; Finding A's own scenario
        est = GeneralizedParetoEstimator(loc=big_loc, pseudo_count=50.0, suff_stat=(mean0, second0))

        est.loc = 0.0  # retarget to a small, exact loc

        # The offset (2.5) must come through the retarget essentially exactly, independent of the
        # variance warning firing alongside it.
        self.assertAlmostEqual(float(est.suff_stat[0]), 2.5, places=6)


if __name__ == "__main__":
    unittest.main()
