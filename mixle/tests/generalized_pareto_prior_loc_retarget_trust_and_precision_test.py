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

An independent adversarial re-review of the Finding A/B fix above confirmed both are genuinely fixed
for the specific mechanism they named, but found two further gaps in the same area:

GAP A: Finding B only protected the PURE-PRIOR passthrough (zero real observations). The ANCHORED
branch -- real data blended with the prior -- computes the prior's mean displacement from the pooled
location (``displacement``) correctly, entirely at small/exceedance scale, but then (pre-fix) folded
it into an ABSOLUTE, loc-scale ``prior_mean_for_variance = mean_x + displacement`` before handing it
to the shared ``anchored_pooled_variance``, which differences it back out internally. Forming that
absolute float is itself the disease this whole mechanism exists to cure -- adding a small
``displacement`` onto the huge ``mean_x`` -- and once ``displacement`` is smaller than ``mean_x``'s
own ULP the addition rounds it away, not to a negligible sub-ULP residue but potentially to nothing
at all, corrupting the pooled variance's prior contribution by 50-100% in the case reproduced below.
Fixed by threading ``displacement`` straight through to ``anchored_pooled_variance`` as a new
``prior_mean_offset`` keyword, which squares it directly instead of re-deriving it by differencing
two absolute values -- additive to the shared function's signature, so every other family that calls
it (Nakagami, Student-t, Gumbel, log-Gaussian, Rician) is byte-for-byte unaffected.

GAP B: ``_prior_mean_offset()`` (added by the Finding B fix) had no cancellation warning analogous to
its sibling ``_prior_variance()``. A manually-supplied plain ``suff_stat`` 2-tuple -- "a first-class,
still-supported encoding" per this module's own docstring -- can arrive with ``mean0`` ALREADY
collapsed to bit-identical with ``loc``, because the CALLER computed ``mean0 = loc + true_offset``
externally (outside this module) at a large ``loc`` before this module ever saw the pair.
Differencing ``mean0 - loc`` is exact given what it receives, but "exact" here means "exactly 0.0",
not "correct" -- and there was no diagnostic. It also happens to make ``_prior_variance()``'s own
floor check degenerate (its floor is ``(mean - loc) ** 2 / ...``, so a zero displacement trivially
satisfies it), silencing THAT warning too, collapsing silently to the pre-existing degenerate default
(shape=0.0, scale=min_scale) with zero disclosure. Fixed by warning when the recovered offset is
indistinguishable from zero at ``loc``'s own float64 resolution: every generalized-Pareto prior's
mean is ``loc`` plus a STRICTLY POSITIVE exceedance amount, so such a reading is not one any valid
prior produces.

A further independent adversarial review of the GAP B fix (commit c40a5fe6) found GAP C: the
``mean_offset`` warning GAP B added does not survive a SECOND ``.loc`` retarget, reproducing the exact
"silent collapse, zero disclosure" failure GAP B's own commit message says it fixes. The variance half
of this mechanism (Finding A above) already had the right discipline -- the ``loc`` setter checks
``_prior_variance_is_carried()`` BEFORE re-deriving and only forwards the variance as trusted when the
SOURCE already was -- but GAP B's own fix never gave ``mean_offset`` the analogous
``_prior_mean_offset_is_carried()`` check: the setter unconditionally set ``mean_offset=offset``
regardless of whether ``offset`` came from ``_prior_mean_offset``'s trusted "carried" fast path or from
its GAP-B fallback path that had just emitted the implausibility warning. Once a collapsed offset was
detected once and carried forward on a new payload, ``_prior_mean_offset``'s carried fast path returned
it unconditionally on every subsequent call -- another ``.loc`` retarget, or a plain ``estimate()`` --
with zero re-validation. In the pure-prior case this silently reproduced the degenerate default
(shape=0.0, scale=min_scale) a second time with no warning; composed with a foreign-combine()'d real
partial (see ``GeneralizedParetoForeignCombineTest``), it instead blended the permanently-"trusted",
still-wrong offset with well-formed real data, producing a fit inside the ordinary parameter bounds --
worse than the pure-prior case, since there is no safe degenerate placeholder hinting anything is
wrong. Fixed the same way Finding A fixed the variance: ``_prior_mean_offset_is_carried()`` mirrors
``_prior_variance_is_carried()``, and the ``loc`` setter checks it BEFORE re-deriving, forwarding
``mean_offset`` as trusted only when the source already was -- otherwise the new payload is left
un-carried, so every subsequent retarget or ``estimate()`` call keeps re-deriving (and, since the pair
still cannot support a valid offset, keeps warning).
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


class AnchoredBranchPriorMeanDisplacementCancellationTest(unittest.TestCase):
    """GAP A: the ANCHORED branch (real data blended with the prior) still round-trips the prior's
    mean displacement through an absolute, loc-scale intermediate. See the module docstring above.

    The parameters below are not arbitrary: they thread a narrow numerical needle so the corruption
    is detectable in the end-to-end fitted (scale, shape) at all, rather than being masked by either
    of two OTHER, unrelated effects that dominate more "obvious" choices:

    * If the real exceedances don't individually keep several ULPs of headroom over ``loc``'s own
      ULP, the observations THEMSELVES lose precision being represented as ``loc + e`` -- a separate,
      unavoidable float64 floor this test is not about.
    * The prior's contribution to the pooled variance is ``pseudo_count * (prior_variance +
      displacement**2)``, pooled alongside the real data's own ``observed_scatter`` (roughly
      ``raw_count * exceedance_spread**2``). A real generalized Pareto's variance and mean are linked
      (``variance = mean_offset**2 / (1 - 2*xi)``), so a prior whose mean sits close enough to the raw
      sample's own mean for ``displacement`` to be small (needed for the corruption to bite --
      ``displacement`` must be comparable to ``loc``'s ULP to be lost) inherits a variance on the same
      huge order as that mean squared UNLESS ``xi`` is pushed very negative -- otherwise
      ``observed_scatter`` swamps the prior's contribution entirely and the corruption, though total,
      becomes numerically invisible in the final fit. This is presumably why ce2e6614's OWN test
      exercising the anchored branch with a real observation
      (``PriorMeanPrecisionAcrossMagnitudeTest.test_loc_setter_retarget_with_one_real_observation_stays_accurate``
      above) did not catch this: its pseudo_count (2000) dominates its single real observation (1),
      routing almost all of the prior/data mean gap into ``anchored_pooled_variance``'s already-
      accepted ``shift`` term rather than into ``displacement``.

    ``_prior_for_loc`` picks ``(scale, xi)`` to hit an exact ``(mean_offset, variance)`` target that
    keeps both effects out of the way -- still an entirely ordinary, valid
    ``GeneralizedParetoDistribution``, just an unusually thin-tailed one.
    """

    JITTER = 180.0
    BASE_EXCEEDANCES = (
        4.0,
        9.0,
        21.0,
        13.0,
        27.0,
        6.0,
        18.0,
        11.0,
        24.0,
        8.0,
        15.0,
        19.0,
        5.0,
        22.0,
        10.0,
        17.0,
        7.0,
        25.0,
        12.0,
        20.0,
    )
    PSEUDO_COUNT = 35.0
    MEAN_OFFSET_DELTA = 180.0  # how far the prior's exceedance mean sits from the raw sample's own

    @classmethod
    def _exceedances(cls):
        return [b * cls.JITTER for b in cls.BASE_EXCEEDANCES]

    @classmethod
    def _prior_for_loc(cls, loc):
        exceedances = cls._exceedances()
        n = float(len(exceedances))
        raw_mean = sum(exceedances) / n
        mean_offset = raw_mean + cls.MEAN_OFFSET_DELTA
        # Target variance comparable to (displacement)**2 at this pseudo_count/raw_count blend, NOT
        # to mean_offset**2 -- see the class docstring's second bullet.
        displacement_estimate = cls.MEAN_OFFSET_DELTA * n / (n + cls.PSEUDO_COUNT)
        variance = 0.5 * displacement_estimate * displacement_estimate
        xi = (1.0 - mean_offset * mean_offset / variance) / 2.0
        scale = mean_offset * (1.0 - xi)
        return GeneralizedParetoDistribution(scale=scale, shape=xi, loc=loc)

    def _fit(self, loc):
        dist = self._prior_for_loc(loc)
        est = dist.estimator(pseudo_count=self.PSEUDO_COUNT)
        acc = est.accumulator_factory().make()
        for e in self._exceedances():
            acc.update(loc + e, 1.0, None)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fitted = est.estimate(float(len(self.BASE_EXCEEDANCES)) + self.PSEUDO_COUNT, acc.value())
        self.assertEqual(caught, [], "this well-conditioned scenario should not itself warn")
        return fitted

    def test_anchored_branch_matches_loc_zero_baseline_at_extreme_loc(self):
        # Threshold-equivariant baseline: the same data pattern (offsets from loc), fit at loc=0,
        # where no term in this computation is anywhere near a precision cliff.
        baseline = self._fit(0.0)
        fitted = self._fit(1.0e18)

        # Before the fix this drifts off the baseline by ~1.3% (scale) / ~1.4% (shape) at loc=1e18 --
        # the prior's mean-displacement contribution to the pooled variance is rounded to exactly 0
        # by the `mean_x + displacement` addition. After the fix both stay within a few tenths of a
        # percent, matching the residual float64 floor for representing `raw_count` real observations
        # at this magnitude (unrelated to this bug -- see the class docstring's first bullet).
        rel_tol = 0.005
        rel_scale = abs(fitted.scale - baseline.scale) / abs(baseline.scale)
        rel_shape = abs(fitted.shape - baseline.shape) / abs(baseline.shape)
        self.assertLess(rel_scale, rel_tol, msg="scale at loc=1e18 vs the loc=0 baseline (%.6g)" % rel_scale)
        self.assertLess(rel_shape, rel_tol, msg="shape at loc=1e18 vs the loc=0 baseline (%.6g)" % rel_shape)


class ManuallySuppliedPairExternallyCancelledMeanWarnsTest(unittest.TestCase):
    """GAP B: a plain ``suff_stat`` 2-tuple whose ``mean0`` was already collapsed to bit-identical
    with ``loc`` by an addition performed OUTSIDE this module must now be disclosed, not silently
    returned as an offset of 0.0. See the module docstring above.
    """

    def test_externally_precomputed_mean0_that_collapsed_to_loc_now_warns(self):
        big_loc = 1.0e18
        true_offset = 50.0
        # Simulates a caller who computed `mean0 = loc + true_offset` themselves, outside this
        # module, before ever constructing the estimator -- exactly the scenario the finding
        # describes. Asserted explicitly so this repro does not silently stop reproducing if float64
        # semantics or the chosen constants ever change.
        mean0 = big_loc + true_offset
        self.assertEqual(mean0, big_loc, "setup invariant: the addition must already have collapsed")
        second0 = mean0 * mean0
        est = GeneralizedParetoEstimator(loc=big_loc, pseudo_count=50.0, suff_stat=(mean0, second0))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fitted = est.estimate(None, (0.0, 0.0, 0.0))  # zero real observations: pure-prior path

        # Before the fix: zero warnings, silent collapse to the pre-T1-01 degenerate default.
        self.assertTrue(
            any("offset" in str(w.message).lower() for w in caught),
            "a collapsed mean0 must now be disclosed instead of silently returned",
        )
        # The collapse itself still happens -- by construction mean0 is bit-identical to loc, so no
        # reconstruction is possible after the fact. This test is about the disclosure, not recovery.
        self.assertEqual(fitted.shape, 0.0)
        self.assertEqual(fitted.scale, 1.0e-12)

    def test_loc_setter_retarget_path_also_warns_on_a_collapsed_pair(self):
        # _prior_mean_offset is also called from the `.loc` setter's re-anchoring, not just
        # estimate() -- a collapsed pair must be disclosed there too.
        source_loc = 1.0e18
        mean0 = source_loc + 50.0
        self.assertEqual(mean0, source_loc)
        second0 = mean0 * mean0
        est = GeneralizedParetoEstimator(loc=source_loc, pseudo_count=50.0, suff_stat=(mean0, second0))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            est.loc = 2.0e18  # retarget to a different (still large) threshold

        self.assertTrue(
            any("offset" in str(w.message).lower() for w in caught),
            "the .loc setter's re-anchoring must also disclose a collapsed offset",
        )

    def test_ordinary_well_resolved_offset_stays_quiet(self):
        # Regression guard mirroring _prior_variance's own "stays deliberately quiet" guarantee: an
        # offset that survives its own magnitude comfortably must not trigger the NEW warning (the
        # pinned T1-01 scenario's own second_moment==mean0**2 still warns about the VARIANCE
        # separately -- Finding A's own test above -- which is unrelated and must keep firing).
        loc = 1_700_000_000.0  # epoch seconds -- the pinned T1-01 test's own magnitude
        mean0 = loc + 2.5
        second0 = mean0 * mean0
        est = GeneralizedParetoEstimator(loc=loc, pseudo_count=50.0, suff_stat=(mean0, second0))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            est.estimate(None, (0.0, 0.0, 0.0))

        self.assertFalse(
            any("offset" in str(w.message).lower() for w in caught),
            "a comfortably-resolved offset must not trigger the new plausibility warning",
        )


def _warned_about_offset(fn, *args, **kwargs):
    """Whether one call raises an ``_prior_mean_offset`` implausibility ``RuntimeWarning``."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fn(*args, **kwargs)
    return result, any("offset" in str(w.message).lower() for w in caught)


class PriorMeanOffsetTrustSurvivesRetargetTest(unittest.TestCase):
    """GAP C: mirrors ``PriorVarianceTrustSurvivesRetargetTest`` (Finding A) above, but for
    ``_prior_mean_offset``/``mean_offset`` -- the field GAP B added without giving it the same
    trust-tracking discipline the variance already had. See the module docstring's GAP C entry.
    """

    BIG_LOC = 1.0e18
    TRUE_OFFSET = 50.0

    def _collapsed_pair_estimator(self, pseudo_count=50.0):
        # Same construction as GAP B's own test: a manually-supplied plain 2-tuple whose mean0 was
        # already collapsed to bit-identical with loc by an addition performed OUTSIDE this module.
        mean0 = self.BIG_LOC + self.TRUE_OFFSET
        self.assertEqual(mean0, self.BIG_LOC, "setup invariant: the addition must already have collapsed")
        second0 = mean0 * mean0
        return GeneralizedParetoEstimator(loc=self.BIG_LOC, pseudo_count=pseudo_count, suff_stat=(mean0, second0))

    def test_double_retarget_of_a_collapsed_pair_warns_on_every_retarget_not_just_the_first(self):
        # Exact repro from the adversarial review: first .estimate() and first .loc retarget both
        # correctly warn (GAP B); the finding is that a SECOND retarget used to produce zero warnings.
        est = self._collapsed_pair_estimator()

        _, warned = _warned_about_offset(est.estimate, None, (0.0, 0.0, 0.0))
        self.assertTrue(warned, "the first .estimate() must warn about the collapsed offset")

        _, warned = _warned_about_offset(lambda: setattr(est, "loc", 2.0e18))
        self.assertTrue(warned, "the first .loc retarget must warn")

        # THE FIX: a second retarget of the SAME (still-untrustworthy) prior must warn again -- this
        # is exactly where c40a5fe6's own mean_offset warning failed to survive, unlike its sibling
        # variance warning (Finding A). Pre-fix this assertion fails: the setter had already baked
        # the fallback's collapsed offset in as permanently "carried", so this second retarget read
        # it back through the trusted fast path with zero re-validation.
        _, warned = _warned_about_offset(lambda: setattr(est, "loc", 3.0e18))
        self.assertTrue(warned, "a SECOND .loc retarget must still warn -- the pre-fix bug permanently silenced this")

        # `_prior_mean_offset` is also consulted directly from `estimate()` (its own `prior_offset`
        # line), independent of the setter -- a plain further call must keep warning too.
        _, warned = _warned_about_offset(est.estimate, None, (0.0, 0.0, 0.0))
        self.assertTrue(warned, "a plain .estimate() call after the second retarget must still warn")

        # The collapse itself is still unrecoverable -- by construction the true offset is gone, so
        # the pure-prior fit is still the safe degenerate default. The bug was the missing
        # disclosure, not a recoverable numeric error; this test is about the warning surviving, not
        # about resurrecting a number that was never there to recover (mirrors GAP B's own test).
        fitted, _ = _warned_about_offset(est.estimate, None, (0.0, 0.0, 0.0))
        self.assertEqual(fitted.shape, 0.0)
        self.assertEqual(fitted.scale, est.min_scale)

    def test_warning_survives_four_further_retargets_not_just_the_second(self):
        # Belt-and-suspenders: the trust bit must not get laundered back in on a THIRD, FOURTH, ...
        # retarget either -- confirms the fix is a genuine per-generation check, not an off-by-one
        # patch that merely pushes the leak one retarget later.
        est = self._collapsed_pair_estimator()
        for generation, target_loc in enumerate((2.0e18, 3.0e18, 4.0e18, 5.0e18), start=1):
            _, warned = _warned_about_offset(lambda loc=target_loc: setattr(est, "loc", loc))
            self.assertTrue(warned, "retarget #%d of an untrustworthy pair must warn, not just the first" % generation)

    def test_a_library_built_prior_still_never_warns_across_retargets(self):
        # Regression guard mirroring Finding A's own control: a prior built via
        # `.estimator(pseudo_count)` carries an EXACT mean_offset throughout (captured directly as
        # `scale/(1-xi)`, no dependence on loc), at any loc -- retargeting it must not start
        # spuriously warning just because the trust check now exists.
        dist = GeneralizedParetoDistribution(scale=2.0, shape=-0.3, loc=0.0)
        est = dist.estimator(pseudo_count=50.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            est.estimate(None, (0.0, 0.0, 0.0))
            est.loc = self.BIG_LOC
            est.estimate(None, (0.0, 0.0, 0.0))
            est.loc = 2.0 * self.BIG_LOC
            est.estimate(None, (0.0, 0.0, 0.0))
            est.loc = 3.0
            est.estimate(None, (0.0, 0.0, 0.0))


class ForeignCombineThenDoubleRetargetDisclosureTest(unittest.TestCase):
    """GAP C's harsher composition, from the same adversarial review: once real data has been
    combined in via a foreign/plain-tuple partial (:class:`GeneralizedParetoForeignCombineTest`'s own
    scenario, in ``generalized_pareto_prior_retarget_and_foreign_combine_test.py``), the collapsed-
    offset silence does not just reproduce a red-flag degenerate default (as in the pure-prior case
    above) -- it blends the corrupted, permanently-"trusted" offset with well-formed real data into a
    fit that lands inside ordinary parameter bounds, with the offset-specific disclosure gone.

    (The fit's ``max_unverified``/``support-consistency-unverified`` note is a SEPARATE, pre-existing
    disclosure this same foreign-combine shape already triggers regardless of this fix -- present on
    both sides of it, confirmed below -- so it is not by itself evidence the offset bug was disclosed;
    the assertions here specifically target the offset warning, the one this fix adds.)
    """

    BIG_LOC = 1.0e18
    TRUE_OFFSET = 50.0

    def _setup(self, pseudo_count=50.0):
        mean0 = self.BIG_LOC + self.TRUE_OFFSET
        self.assertEqual(mean0, self.BIG_LOC, "setup invariant: the addition must already have collapsed")
        second0 = mean0 * mean0
        est = GeneralizedParetoEstimator(loc=self.BIG_LOC, pseudo_count=pseudo_count, suff_stat=(mean0, second0))
        acc = est.accumulator_factory().make()
        self.assertTrue(acc.has_prior)

        # Real data, combined in via a foreign/plain-tuple partial (no `.max_x`/`.anchored` payload)
        # -- the reviewer's "foreign-combine()'d" real data -- BEFORE either retarget, matching the
        # report's ordering.
        exceedances = [2000.0 + 100.0 * i for i in range(50)]
        data = [self.BIG_LOC + e for e in exceedances]
        foreign_tuple = (sum(data), sum(x * x for x in data), float(len(data)))
        acc.combine(foreign_tuple)
        self.assertTrue(acc._max_tainted, "setup invariant: this must be the foreign-combine max-tainted shape")
        return est, acc, float(len(data))

    def test_second_retarget_after_a_foreign_combine_still_discloses_the_offset(self):
        est, acc, n = self._setup()

        _, warned = _warned_about_offset(lambda: setattr(est, "loc", self.BIG_LOC + 100.0))
        self.assertTrue(warned, "the first retarget must warn, same as the pure-prior case")

        # THE FIX, in the harsher composition: pre-fix this second retarget was completely silent --
        # zero warnings of any kind -- because the first retarget had already laundered the collapsed
        # offset into "carried". The real data combined in beforehand does not itself affect the
        # `.loc` setter at all (it only ever reads/writes `self.suff_stat`, the prior's own pair), so
        # this is the same trust-bit leak as the pure-prior test, just with a foreign-combined
        # accumulator sitting alongside it, as the reviewer's report combined them.
        _, warned = _warned_about_offset(lambda: setattr(est, "loc", self.BIG_LOC + 200.0))
        self.assertTrue(
            warned,
            "the second retarget must still disclose the collapsed offset even with real data already "
            "combined into the accumulator -- pre-fix this was silent",
        )

        # And the composition the reviewer actually flagged as worse: fitting with the still-
        # untrustworthy prior blended against the real data must also disclose the offset (`estimate`
        # reads `_prior_mean_offset` directly too), rather than quietly returning an in-bounds fit
        # with the offset-specific warning gone -- rather than the plainly-abnormal degenerate default
        # the pure-prior case falls back to when there is no real data to mask the collapse.
        suff_stat = acc.value()
        fitted, estimate_warned_offset = _warned_about_offset(est.estimate, n, suff_stat)
        self.assertTrue(
            estimate_warned_offset,
            "estimate() after the double retarget must disclose the still-untrustworthy prior offset, "
            "not silently blend it into a fit with no offset-specific diagnostic at all",
        )
        # Sanity check on the "worse than the pure-prior case" framing: the fit is NOT the plainly-
        # abnormal (min_scale, shape=0.0) degenerate default here -- confirming this composition
        # really does land inside ordinary parameter bounds rather than tripping an obvious red flag.
        self.assertNotEqual(fitted.scale, est.min_scale)

    def test_max_taint_note_alone_is_not_evidence_of_offset_disclosure(self):
        # Guards the class docstring's caveat: the `support-consistency-unverified` note from the
        # foreign combine's own max-tainting fires on BOTH sides of this fix (it is unrelated to
        # mean_offset trust-tracking), so it must not be mistaken for the fix having done anything.
        est, acc, n = self._setup()
        est.loc = self.BIG_LOC + 100.0
        est.loc = self.BIG_LOC + 200.0
        fitted = est.estimate(n, acc.value())
        self.assertTrue(any("support-consistency-unverified" in note for note in fitted.numerical_repairs()))


if __name__ == "__main__":
    unittest.main()
