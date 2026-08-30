"""Campaign eight (release-checklists/0.8.0-candidate-31d8e1fd-campaign.md), numerical/statistical
correctness lens: robust_regression()'s IRLS scale estimate (median(|r - median(r)|) / 0.6745) is
EXACTLY zero by construction whenever half or more of the rows share (near enough) the same
residual under the current fit -- an ordinary shape for zero-inflated counts or capped/rounded
measurements, not a contrived one -- and floors at the hardcoded 1e-8. For method='tukey' the
redescending biweight then hard-zeros essentially every row, either tripping the existing "assigned
zero weight to every observation" crash outright, or -- on a design with genuine minority signal --
settling at a self-consistent degenerate fixed point that discards the real signal with
converged=True and no warning. The verifier's extension found method='huber' is not a safe
alternative either: given enough iterations it reaches the same silent degenerate answer, just more
slowly.

The >50% breakdown point itself is real and inherent to any MAD-scaled M-estimator -- not something
this fix attempts to overcome. What was missing, and what these tests pin down, is disclosure,
matching the convention this module already uses elsewhere (GLM separation / saturation / rank
deficiency): a UserWarning plus a boolean field (RegressionFit.degenerate_scale) on a fit that
still returns, and a named-cause RuntimeError on a fit that cannot.

Both reproduction scenarios below are the campaign's own two findings:
  * a single-column (intercept-only) design where most responses are exactly zero and a minority
    carries the only real signal -- crashes under tukey with, pre-fix, no explanation of why.
  * a two-column zero-inflated design with a genuine, design-dependent minority signal -- tukey
    silently returns coefficients of [0, 0] with converged=True and no warning, pre-fix.

Every assertion below that checks for a warning, a RuntimeError substring, or
`degenerate_scale` fails against the pre-fix code (no such field existed; no warning was ever
issued; the crash message named no cause) and passes after it.

---

A second, independent adversarial review of that fix (commit 04ef6ae1) then executed the fixed code
directly and found the detection heuristic itself (scale bit-exactly AT its hard floor, AND >=5% of
weights near zero) had three material gaps, plus two lower-severity items. The classes below
(GapOneLowMinorityFractionTest through HeteroskedasticNoiselessMajorityHonestDisclosureTest -- named
HeteroskedasticNoiselessMajorityMustNotWarnTest at the time this paragraph was written; see the
audit R-3 section further down for the rename) reproduce the reviewer's own scenarios and are proven
to FAIL against the pre-redesign 04ef6ae1 code (verified by
direct execution against that exact commit, restored temporarily) and PASS after the redesign, which
replaces the single combined check with three independent signals in mixle/inference/glm.py:
`_robust_weight_collapse` (signal 1, a repair of the original check), `_robust_response_point_mass`
(signal 2, new), and `_robust_diverges_from_ols` (signal 3, new, an OLS-agreement SUPPRESSOR only --
see the module comment above `_robust_weight_collapse` in glm.py for the full three-signal design and
why signal 3 is deliberately not a fourth detector):

  * gap 1 (blocking): any minority fraction below the old 5% weight-crushed cutoff went undetected
    however strong the discarded signal -- reproduced down to a 1% minority, 10 true SDs from the
    majority, across two design shapes. Fixed two ways at once: signal 1's own crushed-weight cutoff
    drops to 1% (justified in glm.py: false positives here are prevented by the near-floor
    requirement, not by this fraction), and signal 2 reads the response's majority tie directly,
    independent of any weight-fraction threshold -- needed because the two fractions are not always
    the same number (a two-column 1%-minority design was found to crush as little as 0.67% of
    weights, under even signal 1's new 1% cutoff, while the response itself sits 99.33% tied).
  * gap 2 (blocking): on the fix's own 35%-minority target scenario, method='huber' missed the
    degenerate case in 22 of 60 seeds (default max_iter=100) even though the true coefficients were
    discarded in all 60 -- the old bit-exact `scale == floor` comparison raced IRLS's convergence
    tolerance (both defaulting to 1e-8); the 22 missed seeds had converged scales 0.04-0.86 times
    `tol` above the floor. Fixed by keying signal 1's near-floor comparison to `tol` itself (with an
    8x margin) instead of bit-exact equality. Signal 2 also fires unconditionally on this exact
    design (65% response tied at zero, comfortably past its own 50% threshold), so this fixture ends
    up doubly covered -- gap 2's OWN mechanism (the tol race) is what the 8x-margin change in signal
    1 specifically targets and what the test below isolates via the `scale > _ROBUST_SCALE_FLOOR`
    count.
  * gap 3 (major): capped/censored measurements are a qualitatively different breakdown geometry --
    a device ceiling ties rows' raw Y values to each other but NOT, in general, to a common residual
    once the fit carries any slope. This is NOT uniformly missed by the old code, which is itself a
    useful, verified distinction: at cap_frac=0.70 (30% of rows uncapped), method='huber' partially
    collapses the slope while keeping the scale nowhere near its floor (confirmed stable under a
    5000x max_iter raise -- a genuine fixed point, not slow convergence) and crushes zero rows to
    near-zero weight, so the old code missed all 8 reproduced seeds; method='tukey' on the SAME data
    collapses the slope completely, which incidentally re-creates a residual tie as a side effect
    and DOES trip the old scale+weight check, so the old code already caught all 8 tukey seeds there
    (by an accident of how far tukey happens to collapse, not because the old signal is sensitive to
    this breakdown shape in general). At the more severe cap_frac=0.96, both methods collapse fully,
    yet the old code missed all 16 (seed, method) combinations anyway, because the crushed-weight
    fraction there (~4%) sits just under the OLD 5% cutoff -- so this specific sub-case is also a
    second, independent demonstration of gap 1's fraction-threshold fix, not only of gap 3's.
    Fixed for the cases signal 1 cannot reach at any threshold (cap_frac=0.70 under huber, confirmed
    in test_huber_scale_never_approaches_the_floor_here below) by signal 2, which reads the raw
    response directly and does not depend on the fit's residuals tying at all.
  * gap 4 (minor, false positive): a benign, correctly-recovered heteroskedastic fit (a noiseless
    majority, a noisier minority of the SAME relationship) tripped the old alarmist wording ("may be
    discarding a real minority signal") even though nothing was discarded -- and, checked directly,
    the old scale+weight condition itself (not just the wording) was genuinely satisfied (scale
    exactly at the floor, ~20% of rows crushed), so this is a real false-positive case, not only a
    wording complaint. Fixed for real (not just reworded) by signal 3, which suppresses signal 1's
    reading when the robust fit lands close to plain OLS (~1-2.5% disagreement here, versus ~76-77%
    on the legitimate outlier-rejection design this module already exists to handle well) -- see
    HeteroskedasticNoiselessMajorityMustNotWarnTest. The remaining, now-honest wording no longer
    asserts that a real signal was discarded; it states the observed weight/response condition.
  * gap 5 (reporting accuracy): 04ef6ae1's own commit message misstated two numbers -- see this
    fix's commit message for the corrected, actually-run figures.

Documented, not silently uncovered (see also the module-level comment above _robust_weight_collapse
in glm.py, which this paragraph summarizes): a response point mass affecting LESS than half the rows
(e.g. 30%-uncapped or 96%-uncapped censoring, which attenuate the fitted slope without reaching the
>50% mark) trips neither signal, matching this module's existing choice to disclose clear breakdown
rather than quantify ordinary estimation imprecision; and no fixed threshold can, from the weight or
response pattern alone, distinguish a real minority signal from a genuine minority of contamination
correctly discarded. Signal 3 (OLS agreement) was investigated and REJECTED as a general gap-3
detector, for a reason stronger than "it agrees with a censoring-biased fit too often": on the
cap_frac=0.70 design above, its reading for one and the same breakdown swings from ~14-22% (huber) to
~47-73% (tukey) depending only on which method the caller chose, and at cap_frac=0.96 it reads
~1-2% disagreement with OLS under EITHER method -- statistically indistinguishable from the ~1-2.5%
disagreement measured on the benign gap-4 fixture, where nothing is wrong at all. It was kept, in
that commit (743a2185), only as a narrow suppressor of signal 1, never applied to signal 2 -- see the
next section for why even that narrow role did not survive a third review.

---

A THIRD, independent adversarial review -- of 743a2185, the R-2 redesign summarized above -- again
executed the fixed code directly and found two further problems. This is addressed as a genuine
third design pass, not another patch on the second: signal 3 (_robust_diverges_from_ols) is REMOVED
entirely rather than re-tuned. Full reasoning and the actual reproduction numbers live in the module
comment above `_robust_weight_collapse` in glm.py (audit R-3); this section summarizes what the
classes below verify, and is honest about what is a real fix versus what remains a documented,
unclosed limitation:

  * GAP A (blocking): signal 3 turned out to make the SAME structural blindness worse, not better.
    Every signal in this file reasons about a majority tied to one VALUE or one near-exact residual;
    none can see a majority following one real (e.g. sloped) relationship while a minority follows a
    genuinely different one -- an entirely ordinary reason to reach for robust regression in the
    first place (pooling across instruments/conditions/regimes with different slopes), and
    qualitatively different from a value tie. Reproduced directly against 743a2185: a two-column
    design with a majority at one slope and a minority at 2x-4x that slope, at ORDINARY majority
    noise, is never flagged at minority fractions 1%-20% under either method -- signal 1 never even
    reaches its scale condition, so signal 3 (a pure suppressor) was never even the proximate cause
    of THIS half of the gap. It IS the proximate cause of the other half: in the narrow window where
    signal 1's absolute floor condition DOES fire on its own (majority noise small enough to floor
    the scale, e.g. 1e-9), signal 3 suppressed that correct detection at minority_frac=5%/slope
    ratio=2x and minority_frac=10%/slope ratio=2x (OLS disagreement 4.81% and 9.77%, both under its
    10% bar) -- ordinary parameters, not an edge case -- turning a correct signal-1-alone detection
    back into silent, undisclosed complete discard. SlopedMajorityMinorityMixtureTest reproduces both
    halves directly against the actual (post-redesign) module, and is explicit that the fix is a
    REMOVAL of signal 3 plus a new, only PARTIALLY-closing scale-relative extension to signal 1
    (_ROBUST_RELATIVE_SCALE_FLOOR in glm.py) -- not a complete closure of the sloped-mixture blind
    spot, which remains real and is asserted as such (not silently skipped) at ordinary,
    non-floor-adjacent majority noise levels, and for method='huber' specifically at majority noise
    around 1e-4 and coarser. Audit R-2's original motivation for signal 3 -- not warning on a benign
    heteroskedastic fixture where nothing is actually discarded -- is real, and is now handled by
    HONEST WORDING instead of suppression: see HeteroskedasticNoiselessMajorityHonestDisclosureTest
    below (renamed from HeteroskedasticNoiselessMajorityMustNotWarnTest, because that fixture DOES
    now warn -- correctly, since the underlying weight/scale condition genuinely is satisfied -- and
    the fix is that the wording no longer overclaims what firing means).
  * GAP B (major): _RESPONSE_POINT_MASS_FRACTION's hard 50% cutoff did not track where fit quality
    for signal 2's own target shape (a tied response value) actually degrades. An independent sweep
    of this module's own two-column zero-inflated fixture family (noise 0.02-1.0, n=600-6000,
    single- and two-column shapes, 2-10 true SD minority separations) found a roughly symmetric
    severe-degradation band from ~44% through ~56% response-tie-fraction, essentially independent of
    noise level or effect size -- i.e. a property of the estimator's own breakdown dynamics near 50%,
    confirmed directly against 743a2185, not merely re-quoted from the earlier review.
    ResponseTieFractionDegradationBandTest reproduces the band and confirms the effect of lowering
    the constant to 0.44 (at one representative point in that band, 29 of 60 seeds are newly
    disclosed that would NOT have been under the old 0.5 cutoff), while documenting -- not hiding --
    that a coin-flip-like uncertainty persists at whatever cutoff is current: it moved from 50% to
    44%, it did not disappear.

GapOneLowMinorityFractionTest, GapTwoHuberSixtySeedSweepTest, and GapThreeCappedCensoredDesignTest
below are UNCHANGED from the R-2 redesign (signal 3 was never load-bearing for any of them) and are
re-run here specifically to confirm audit R-3 -- the suppressor's removal, the new relative-scale
check, and the lowered point-mass threshold -- does not regress any of them.
"""

import unittest
import warnings

import numpy as np

from mixle.inference import robust_regression
from mixle.inference.glm import (
    _RESPONSE_POINT_MASS_FRACTION,
    _ROBUST_SCALE_FLOOR,
    _response_point_mass_fraction,
)


class PointMassSingleColumnTest(unittest.TestCase):
    """Scenario 1: an intercept-only design where >50% of responses are exactly zero."""

    def setUp(self):
        rng = np.random.RandomState(0)
        n = 200
        self.X = np.ones((n, 1))
        self.y = np.zeros(n)
        n_nonzero = int(0.3 * n)  # minority carries the only real signal (true mean ~10)
        self.y[:n_nonzero] = rng.normal(10.0, 1.0, n_nonzero)

    def test_tukey_crash_now_names_the_breakdown(self):
        # pre-fix: raises RuntimeError("robust regression assigned zero weight to every
        # observation") with no indication of why. The crash itself is legitimate (this is past
        # the estimator's real breakdown point, not a bug to paper over) -- only the message
        # changes, to actually name the mechanism instead of leaving the caller to guess.
        with self.assertRaises(RuntimeError) as ctx:
            robust_regression(self.X, self.y, method="tukey")
        message = str(ctx.exception)
        self.assertIn("breakdown", message)
        self.assertIn("zero-inflated", message)
        self.assertIn("RegressionFit.degenerate_scale", message)

    def test_huber_silently_degenerates_and_now_warns(self):
        # huber's weight (c / |u|) is never exactly zero, so it never trips the crash guard -- it
        # instead settles on a near-zero-coefficient fit (true signal ~10, fitted ~0) and, pre-fix,
        # reports converged=True with no warning at all. The verifier's extension is that this
        # persists past the default max_iter=100, so exercise a raised max_iter here too.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = robust_regression(self.X, self.y, method="huber", max_iter=500)
        self.assertTrue(fit.converged)
        self.assertLess(abs(fit.coef[0]), 1e-4)  # true signal (~10) discarded entirely
        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        self.assertEqual(len(user_warnings), 1, caught)
        self.assertIn("breakdown", str(user_warnings[0].message))
        self.assertTrue(fit.degenerate_scale)


class ZeroInflatedTwoColumnTest(unittest.TestCase):
    """Scenario 2: a two-column zero-inflated design with genuine, design-dependent signal."""

    def setUp(self):
        rng = np.random.RandomState(7)
        n = 600
        predictor = rng.normal(0, 1, n)
        self.X = np.column_stack([np.ones(n), predictor])
        self.true_beta = np.array([5.0, 2.0])
        is_nonzero = rng.rand(n) < 0.35  # minority carries the design-dependent real signal
        self.y = np.zeros(n)
        self.y[is_nonzero] = self.X[is_nonzero] @ self.true_beta + rng.normal(
            0.0, 0.3, int(np.count_nonzero(is_nonzero))
        )

    def test_scenario_is_a_genuine_majority_point_mass(self):
        # documents the fixture's shape so a future edit that accidentally drops it below the
        # >50% breakdown threshold fails here, with a clear reason, instead of surfacing as a
        # confusing failure in the disclosure tests below
        self.assertGreater(np.mean(self.y == 0.0), 0.5)
        self.assertGreater(np.count_nonzero(self.y != 0.0), 20)

    def test_tukey_silently_returns_near_zero_and_now_warns(self):
        # the campaign's own "tukey currently silently returns [0, 0]" reproduction
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = robust_regression(self.X, self.y, method="tukey")
        self.assertTrue(fit.converged)
        np.testing.assert_allclose(fit.coef, [0.0, 0.0], atol=1e-6)
        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        self.assertEqual(len(user_warnings), 1, caught)
        self.assertIn("breakdown", str(user_warnings[0].message))
        self.assertTrue(fit.degenerate_scale)

    def test_huber_at_raised_max_iter_also_degenerates_and_now_warns(self):
        # the verifier's extension: huber reaches the SAME degenerate fixed point, just more
        # slowly, so this raises max_iter well past the 100 default to give it room to get there
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = robust_regression(self.X, self.y, method="huber", max_iter=2000)
        self.assertTrue(fit.converged)
        self.assertLess(np.max(np.abs(fit.coef)), 1e-4)  # true_beta = [5.0, 2.0] discarded entirely
        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        self.assertEqual(len(user_warnings), 1, caught)
        self.assertIn("breakdown", str(user_warnings[0].message))
        self.assertTrue(fit.degenerate_scale)


class OrdinaryContaminationStaysUndisclosedTest(unittest.TestCase):
    """The existing RobustQuantileTest.test_robust_ignores_outliers shape must not newly warn.

    8% gross contamination on otherwise continuous, non-tied noise never floors the scale estimate
    (the bulk's residuals keep a real, non-degenerate spread), so this must behave exactly as
    before this fix: no warning, no `degenerate_scale`, and the true coefficients recovered.
    """

    def setUp(self):
        self.rng = np.random.RandomState(2)
        self.n = 1000
        self.X = np.column_stack([np.ones(self.n), self.rng.normal(0, 1, self.n)])
        self.beta = np.array([1.0, 2.0])

    def test_smooth_contamination_produces_no_new_warning(self):
        y = self.X @ self.beta + self.rng.normal(0, 0.3, self.n)
        y[:80] += 50.0  # gross contamination, but continuous underlying noise -- no point mass
        for method in ("huber", "tukey"):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fit = robust_regression(self.X, y, method=method)
            user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
            self.assertEqual(user_warnings, [], f"unexpected warning(s) for method={method!r}: {user_warnings}")
            self.assertFalse(fit.degenerate_scale, f"method={method!r} incorrectly flagged as degenerate")
            np.testing.assert_allclose(fit.coef, self.beta, atol=0.2)


def _single_col_point_mass_design(seed: int, frac: float, n: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """Intercept-only design: ``1 - frac`` of rows exactly 0.0, ``frac`` minority ~ N(10, 1) --
    10 true SDs from the majority. Mirrors PointMassSingleColumnTest, parameterized by fraction."""
    rng = np.random.RandomState(seed)
    X = np.ones((n, 1))
    y = np.zeros(n)
    n_minor = max(1, int(round(frac * n)))
    y[:n_minor] = rng.normal(10.0, 1.0, n_minor)
    return X, y


def _two_col_zero_inflated_design(seed: int, frac: float, n: int = 600) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two-column zero-inflated design: ``1 - frac`` of rows exactly 0.0, ``frac`` minority carries
    ``X @ [5.0, 2.0] + noise``. Mirrors ZeroInflatedTwoColumnTest, parameterized by fraction."""
    rng = np.random.RandomState(seed)
    predictor = rng.normal(0, 1, n)
    X = np.column_stack([np.ones(n), predictor])
    true_beta = np.array([5.0, 2.0])
    is_nonzero = rng.rand(n) < frac
    y = np.zeros(n)
    y[is_nonzero] = X[is_nonzero] @ true_beta + rng.normal(0.0, 0.3, int(np.count_nonzero(is_nonzero)))
    return X, y, true_beta


class GapOneLowMinorityFractionTest(unittest.TestCase):
    """Reviewer's gap 1: a minority fraction below the old 5% weight-fraction cutoff went completely
    undetected, however strong its discarded signal -- reproduced by the reviewer down to a 1%
    minority sitting 10 true SDs from the majority, across two design shapes. Swept here at 1%, 3%,
    4.5%, and 4.99% (below the old cutoff IN EXPECTATION) x 3 seeds x both design shapes -- 24
    combinations, matching the reviewer's own count. Verified by direct execution against the
    pre-redesign 04ef6ae1 code: 19 of the 24 fail there outright (coef collapses fully to the
    majority value, degenerate_scale stays False, no warning). The other 5 already passed
    pre-redesign, for an unglamorous but honest reason rather than a silent omission: a hard
    percentage cutoff is only ever applied to a REALIZED count, and the single-column design's exact
    rounding (0.0499 x 2000 rounds to precisely 100, i.e. exactly 5.00%, for every seed) or the
    two-column design's random Bernoulli draw (frac=0.045/0.0499 occasionally realizes 5%+ by chance
    at a given seed) can land AT or above the old cutoff even when the requested fraction sits below
    it. Every one of the 24 passes below regardless, including those 5: signal 2 (response point
    mass) does not depend on the realized weight-crushed count at all, and signal 1's own lowered 1%
    cutoff covers the rest with room to spare.
    """

    FRACTIONS = (0.01, 0.03, 0.045, 0.0499)
    SEEDS = (0, 1, 2)

    def test_single_column_low_fraction_now_detected(self):
        for frac in self.FRACTIONS:
            for seed in self.SEEDS:
                with self.subTest(frac=frac, seed=seed):
                    X, y = _single_col_point_mass_design(seed, frac)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        fit = robust_regression(X, y, method="huber")
                    self.assertTrue(fit.converged)
                    self.assertLess(abs(fit.coef[0]), 1e-4)  # true minority signal (~10) discarded
                    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
                    self.assertEqual(len(user_warnings), 1, caught)
                    self.assertIn("breakdown", str(user_warnings[0].message))
                    self.assertTrue(fit.degenerate_scale)

    def test_two_column_low_fraction_now_detected(self):
        for frac in self.FRACTIONS:
            for seed in self.SEEDS:
                with self.subTest(frac=frac, seed=seed):
                    X, y, true_beta = _two_col_zero_inflated_design(seed, frac)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        fit = robust_regression(X, y, method="huber")
                    self.assertTrue(fit.converged)
                    self.assertLess(np.max(np.abs(fit.coef)), 1e-4)  # true_beta discarded entirely
                    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
                    self.assertEqual(len(user_warnings), 1, caught)
                    self.assertIn("breakdown", str(user_warnings[0].message))
                    self.assertTrue(fit.degenerate_scale)

    def test_single_column_low_fraction_tukey_crash_now_named(self):
        # tukey hard-zeros the same low-fraction minority outright rather than settling near it;
        # the crash message should still name the (now response-point-mass-driven) cause.
        for frac in self.FRACTIONS:
            with self.subTest(frac=frac):
                X, y = _single_col_point_mass_design(0, frac)
                with self.assertRaises(RuntimeError) as ctx:
                    robust_regression(X, y, method="tukey")
                message = str(ctx.exception)
                self.assertIn("breakdown", message)
                self.assertIn("zero-inflated", message)
                self.assertIn("RegressionFit.degenerate_scale", message)


class GapTwoHuberSixtySeedSweepTest(unittest.TestCase):
    """Reviewer's gap 2: on the fix's OWN 35%-minority zero-inflated design (the exact shape
    ZeroInflatedTwoColumnTest hard-codes at seed=7), method='huber' at the DEFAULT max_iter=100
    missed the degenerate case in 22 of 60 seeds tried (reviewer's own count) even though the true
    coefficients were discarded in all 60 of them -- because huber's weight never hits exactly zero,
    IRLS's convergence tolerance could stop the loop with the raw MAD sitting marginally ABOVE the
    hard-coded floor, and the old check compared scale to that floor bit-exactly. Verified by direct
    execution against the pre-redesign 04ef6ae1 code: 22 of the 60 seeds below fail every assertion
    in this class.
    """

    def test_all_sixty_seeds_discard_the_signal_and_are_now_flagged(self):
        true_beta = np.array([5.0, 2.0])
        n_scale_strictly_above_floor_and_flagged = 0
        for seed in range(60):
            with self.subTest(seed=seed):
                X, y, _ = _two_col_zero_inflated_design(seed, 0.35)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    fit = robust_regression(X, y, method="huber")  # default max_iter=100
                # the reviewer's "discarded in all 60/60" premise, checked directly rather than assumed
                self.assertGreater(np.max(np.abs(fit.coef - true_beta)), 1.0)
                self.assertTrue(fit.degenerate_scale, f"seed={seed} discarded the signal but was not flagged")
                user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
                self.assertEqual(len(user_warnings), 1, caught)
                self.assertIn("breakdown", str(user_warnings[0].message))
                if fit.scale > _ROBUST_SCALE_FLOOR:
                    n_scale_strictly_above_floor_and_flagged += 1
        # gap 2's actual mechanism, not just a backstop: a real, substantial share of these seeds
        # converge with scale sitting STRICTLY above the exact hard floor (the old bit-exact
        # comparison's blind spot) yet are still correctly flagged here. The reviewer counted 22/60;
        # a generous lower bound is asserted instead of that exact count to stay robust to any
        # platform/BLAS-dependent shift in exactly which seeds land where relative to the floor.
        self.assertGreaterEqual(
            n_scale_strictly_above_floor_and_flagged,
            10,
            "expected a substantial share of seeds to be flagged via the near-floor TOLERANCE "
            "specifically (scale above the exact floor), not only via bit-exact equality or the "
            "response-point-mass signal",
        )


class GapThreeCappedCensoredDesignTest(unittest.TestCase):
    """Reviewer's gap 3: a qualitatively different breakdown geometry. A continuous
    covariate-dependent relationship, right-censored at a device ceiling, ties many rows' raw
    response values to each other but NOT to a common residual once the fit carries any slope.

    This is NOT uniformly missed by the pre-redesign 04ef6ae1 code -- verified by direct execution,
    which is itself a useful, honest distinction to draw rather than round off. At cap_frac=0.70,
    method='huber' partially collapses the slope while keeping the scale nowhere near its floor
    (confirmed stable under a 5000x max_iter raise for seed 0 -- a genuine fixed point, not slow
    convergence) and crushes zero rows to near-zero weight, so the old code missed all 8 reproduced
    seeds; method='tukey' on the SAME data instead collapses the slope completely, which incidentally
    re-creates a residual tie as a side effect and DOES trip the old scale+weight check, so the old
    code already caught all 8 tukey seeds there -- by an accident of how far tukey happens to
    collapse on this particular design, not because the old signal is sensitive to this breakdown
    shape in general. At the more severe cap_frac=0.96, both methods collapse fully, yet the old code
    missed all 16 (seed, method) combinations anyway, because the crushed-weight fraction there
    (~4%) sits just under the OLD 5% cutoff -- a second, independent demonstration of gap 1's
    fraction-threshold fix, not only of gap 3's own response-point-mass fix.
    """

    N_SEEDS = 8
    TRUE_SLOPE = 3.0

    @staticmethod
    def _design(seed: int, cap_frac: float, n: int = 800) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.RandomState(seed)
        x = rng.normal(0, 2.0, n)
        y_star = 2.0 + GapThreeCappedCensoredDesignTest.TRUE_SLOPE * x + rng.normal(0, 1.0, n)
        ceiling = np.quantile(y_star, 1.0 - cap_frac)  # cap_frac of rows land AT the ceiling
        y = np.minimum(y_star, ceiling)
        X = np.column_stack([np.ones(n), x])
        return X, y

    def test_severe_capping_30pct_uncapped_now_detected(self):
        # the reviewer's "30%-uncapped" sweep (cap_frac=0.70): under huber, the slope collapses
        # substantially (>50% relative here) at a genuinely non-degenerate scale (never near the
        # floor) and zero near-zero-weight rows -- the response-point-mass signal (signal 2) is the
        # only one that can catch it, and is what newly fixes this half of the sub-case. Under tukey,
        # the full collapse also happens to trip the old scale+weight signal on its own (see the
        # class docstring), so that half was already passing pre-redesign -- both halves are
        # asserted identically below because both are correctly flagged post-redesign either way.
        for seed in range(self.N_SEEDS):
            for method in ("huber", "tukey"):
                with self.subTest(seed=seed, method=method):
                    X, y = self._design(seed, cap_frac=0.70)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        fit = robust_regression(X, y, method=method)
                    relative_collapse = 1.0 - fit.coef[1] / self.TRUE_SLOPE
                    self.assertGreater(relative_collapse, 0.5, "fixture stopped producing a severe collapse")
                    self.assertTrue(fit.degenerate_scale, f"seed={seed} method={method}")
                    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
                    self.assertEqual(len(user_warnings), 1, caught)

    def test_huber_scale_never_approaches_the_floor_here(self):
        # documents the reviewer's specific claim for this sub-case: raw_mad stays far from the
        # floor and no weight is ever crushed, so the scale/weight signal alone (however tuned)
        # structurally cannot fire -- only the response-point-mass signal does.
        X, y = self._design(0, cap_frac=0.70)
        fit = robust_regression(X, y, method="huber")
        self.assertGreater(fit.scale, 0.1)  # nowhere near _ROBUST_SCALE_FLOOR (1e-8)
        r = y - X @ fit.coef
        w = np.where(np.abs(r / fit.scale) <= 1.345, 1.0, 1.345 / np.maximum(np.abs(r / fit.scale), 1e-12))
        self.assertEqual(np.mean(w <= 1e-6), 0.0)
        self.assertTrue(fit.degenerate_scale)  # still caught, via the response, not the weights

    def test_severe_capping_4pct_uncapped_now_detected(self):
        # the reviewer's "4%-uncapped" sweep (cap_frac=0.96): total collapse for both methods, and
        # (unlike cap_frac=0.70 above) missed by the pre-redesign code for BOTH methods -- the
        # crushed-weight fraction here (~4%) sits just under the old 5% cutoff, so this sub-case
        # doubles as a demonstration of gap 1's fraction-threshold fix, not only of gap 3's own.
        for seed in range(self.N_SEEDS):
            for method in ("huber", "tukey"):
                with self.subTest(seed=seed, method=method):
                    X, y = self._design(seed, cap_frac=0.96)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        fit = robust_regression(X, y, method=method)
                    self.assertLess(fit.coef[1], self.TRUE_SLOPE * 0.5)
                    self.assertTrue(fit.degenerate_scale, f"seed={seed} method={method}")
                    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
                    self.assertEqual(len(user_warnings), 1, caught)

    def test_mild_capping_stays_undisclosed(self):
        # 30% and 4% capped (NOT the severe 70%/96% sub-cases above) bias the slope only modestly
        # (well under the >50% collapse bar) and should NOT warn -- documents that this module still
        # discloses clear breakdown rather than every degree of ordinary estimation imprecision.
        for cap_frac in (0.30, 0.04):
            for method in ("huber", "tukey"):
                with self.subTest(cap_frac=cap_frac, method=method):
                    X, y = self._design(0, cap_frac=cap_frac)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        fit = robust_regression(X, y, method=method)
                    relative_collapse = 1.0 - fit.coef[1] / self.TRUE_SLOPE
                    self.assertLess(relative_collapse, 0.3)
                    self.assertFalse(fit.degenerate_scale)
                    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
                    self.assertEqual(user_warnings, [])


class HeteroskedasticNoiselessMajorityHonestDisclosureTest(unittest.TestCase):
    """Reviewer's (R-2's own) gap 4 fixture: a benign, correctly-recovered heteroskedastic fit --
    80% of rows noiseless (lying exactly on the true line), 20% noisier draws of the SAME true
    relationship. Both methods correctly recover the exact true coefficients, so nothing is actually
    discarded, even though (checked directly below) the scale still collapses to its floor and ~20%
    of rows still carry near-zero weight -- a real, genuine satisfaction of signal 1's condition, not
    a false trigger of it.

    RENAMED from HeteroskedasticNoiselessMajorityMustNotWarnTest: that name asserted the fix should
    make this fixture silent, which was audit R-2's answer (_robust_diverges_from_ols, a narrow
    suppressor of signal 1). Audit R-3 removed that suppressor entirely (module comment above
    _robust_weight_collapse in glm.py, GAP A) because, everywhere else, it did more harm than good --
    so this fixture now DOES warn, and degenerate_scale IS True here, and that is the INTENDED,
    correct behavior post-redesign, not a regression to paper over. What R-3 actually fixes for this
    fixture is the WORDING, not the firing: the message must state the observed weight/scale
    condition honestly, without asserting that a real minority signal was discarded (false here --
    nothing was), and must give a caller who already expects an exact-by-design majority an explicit
    "this is not news" reading. That is what this class actually tests.
    """

    @staticmethod
    def _design(seed: int, n: int = 500, frac_noiseless: float = 0.8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.RandomState(seed)
        x = rng.normal(0, 1, n)
        X = np.column_stack([np.ones(n), x])
        beta = np.array([1.0, 2.0])
        y = X @ beta
        n_noisy = int(round((1 - frac_noiseless) * n))
        noisy_idx = rng.choice(n, n_noisy, replace=False)
        y = y.copy()
        y[noisy_idx] += rng.normal(0, 1.5, n_noisy)
        return X, y, beta

    def test_recovers_truth_exactly_despite_now_being_flagged(self):
        # audit R-3: no longer suppressed. degenerate_scale=True and a single UserWarning are now
        # the EXPECTED, correct outcome here -- the point of this test (and the next two) is that
        # this is handled honestly, not that it is silenced.
        for seed in range(4):
            for method in ("huber", "tukey"):
                with self.subTest(seed=seed, method=method):
                    X, y, beta = self._design(seed)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        fit = robust_regression(X, y, method=method)
                    np.testing.assert_allclose(fit.coef, beta, atol=1e-6)
                    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
                    self.assertEqual(len(user_warnings), 1, f"seed={seed} method={method}: {user_warnings}")
                    self.assertTrue(fit.degenerate_scale, f"seed={seed} method={method}")

    def test_the_weight_signal_condition_is_genuinely_present(self):
        # documents WHY this fixture is a real test of the wording fix rather than a case nothing
        # ever fired on: scale really does sit at the floor and a real 20% of rows really do carry
        # near-zero weight, matching the shape of a genuine collapse by weight pattern alone -- so
        # firing here is correct, and the old suppression was hiding a real, if harmless, signal.
        X, y, _ = self._design(0)
        fit = robust_regression(X, y, method="huber")
        self.assertEqual(fit.scale, _ROBUST_SCALE_FLOOR)
        r = y - X @ fit.coef
        w = np.where(np.abs(r / fit.scale) <= 1.345, 1.0, 1.345 / np.maximum(np.abs(r / fit.scale), 1e-12))
        self.assertGreater(np.mean(w <= 1e-6), 0.15)
        self.assertTrue(fit.degenerate_scale)  # audit R-3: no longer suppressed

    def test_wording_is_honest_and_non_alarmist_about_the_benign_case(self):
        # the actual substance of audit R-3's GAP A wording fix: states the observed weight/scale
        # condition plainly, never asserts the old (now-removed) diagnosis that a real minority
        # signal was discarded -- false for this fixture -- and explicitly tells an
        # already-informed caller this is not a new alarm.
        X, y, beta = self._design(0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = robust_regression(X, y, method="huber")
        np.testing.assert_allclose(fit.coef, beta, atol=1e-6)
        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        self.assertEqual(len(user_warnings), 1)
        message = str(user_warnings[0].message)
        self.assertIn("assigned (near) zero weight", message)
        self.assertIn("breakdown", message)
        self.assertNotIn("discarding a real minority signal", message)
        self.assertNotIn("may be discarding", message)
        self.assertIn("naming that shape, not flagging a new problem", message)


def _sloped_mixture_design(
    seed: int, minority_frac: float, slope_ratio: float, majority_sigma: float, n: int = 2000
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Two-column design: a majority (``1 - minority_frac``) following ``y = 1 + 3*x +
    N(0, majority_sigma)``, and a minority (``minority_frac``) following the SAME intercept but a
    ``slope_ratio``-times-different slope, with ordinary (not tiny) noise 0.3. This is audit R-3's
    GAP A scenario: an entirely ordinary reason to reach for robust regression (pooling across two
    regimes/instruments with different slopes) that ties no VALUE, only a relationship -- signal 2
    cannot see this by construction, at any minority fraction or noise level.
    """
    rng = np.random.RandomState(seed)
    x = rng.normal(0, 2.0, n)
    is_minor = rng.rand(n) < minority_frac
    beta_major = np.array([1.0, 3.0])
    beta_minor = np.array([1.0, 3.0 * slope_ratio])
    y = np.empty(n)
    n_major = int(np.sum(~is_minor))
    n_minor = int(np.sum(is_minor))
    y[~is_minor] = beta_major[0] + beta_major[1] * x[~is_minor] + rng.normal(0, majority_sigma, n_major)
    y[is_minor] = beta_minor[0] + beta_minor[1] * x[is_minor] + rng.normal(0, 0.3, n_minor)
    X = np.column_stack([np.ones(n), x])
    return X, y, beta_major, beta_minor


class SlopedMajorityMinorityMixtureTest(unittest.TestCase):
    """A third reviewer's GAP A: a majority sub-population following one real (sloped) relationship,
    a minority following a genuinely different one -- pooling data across two conditions/instruments/
    regimes with different slopes, a completely ordinary reason to reach for robust regression in the
    first place. Ties no VALUE (signal 2 cannot see this at all, by construction) and, at ordinary
    noise levels, never floors the scale either (signal 1's condition is never even reached). This
    class is deliberately honest about BOTH what audit R-3 fixed and what remains a real, documented
    limitation -- see the module comment above _robust_weight_collapse in glm.py for the full account
    this class's test names summarize.
    """

    def test_ordinary_noise_majority_relationship_discard_remains_undetectable_by_design(self):
        # DOCUMENTED LIMITATION, not a bug: at an ORDINARY (not tiny) majority noise level, the
        # scale never comes close to collapsed, absolutely or relative to the response's own spread,
        # so NEITHER signal can see this -- exactly as the module comment says plainly. This pins
        # down that the gap is a stable, understood one, not a silent regression: if a future change
        # makes this fire, that is a welcome surprise, not a broken assumption to "fix" back.
        for minority_frac in (0.01, 0.05, 0.10, 0.20):
            for slope_ratio in (2.0, 4.0):
                with self.subTest(minority_frac=minority_frac, slope_ratio=slope_ratio):
                    X, y, beta_major, _ = _sloped_mixture_design(0, minority_frac, slope_ratio, majority_sigma=0.05)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        fit = robust_regression(X, y, method="tukey", max_iter=500)
                    # the discard is real and complete ...
                    self.assertLess(np.max(np.abs(fit.coef - beta_major)), 1e-2)
                    # ... and undisclosed, which is the documented gap, not an oversight
                    self.assertFalse(fit.degenerate_scale)
                    self.assertEqual([w for w in caught if issubclass(w.category, UserWarning)], [])

    def test_previously_suppressed_tiny_noise_cases_are_now_correctly_flagged(self):
        # audit R-2's suppressor (_robust_diverges_from_ols, now removed) used to silence EXACTLY
        # this case at ordinary minority fractions/slope differences once majority noise was small
        # enough to floor the scale -- a direct regression this redesign fixes by REMOVING the
        # suppressor rather than re-tuning it. Confirmed by direct execution against 743a2185 with
        # the suppressor still active: (5%, 2x) and (10%, 2x) were both suppressed at
        # majority_sigma=1e-9 (OLS disagreement 4.81% and 9.77%, both under the old 10% bar) despite
        # signal 1 alone correctly firing on both, for both methods.
        for minority_frac, slope_ratio in ((0.05, 2.0), (0.10, 2.0)):
            for method in ("huber", "tukey"):
                with self.subTest(minority_frac=minority_frac, slope_ratio=slope_ratio, method=method):
                    X, y, beta_major, _ = _sloped_mixture_design(0, minority_frac, slope_ratio, majority_sigma=1e-9)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        fit = robust_regression(X, y, method=method, max_iter=500)
                    self.assertLess(np.max(np.abs(fit.coef - beta_major)), 1e-2)
                    self.assertTrue(
                        fit.degenerate_scale,
                        f"minority_frac={minority_frac} slope_ratio={slope_ratio} method={method}",
                    )
                    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
                    self.assertEqual(len(user_warnings), 1)

    def test_relative_scale_check_closes_the_gap_down_to_1e6_majority_noise_both_methods(self):
        # audit R-3's new relative floor-band check (_ROBUST_RELATIVE_SCALE_FLOOR in glm.py):
        # extends detection to a majority noise level far too coarse for the absolute 1e-8 floor to
        # ever reach (1e-6 against an O(1) response here), for BOTH methods.
        for minority_frac, slope_ratio in ((0.05, 2.0), (0.10, 4.0)):
            for method in ("huber", "tukey"):
                with self.subTest(minority_frac=minority_frac, slope_ratio=slope_ratio, method=method):
                    X, y, beta_major, _ = _sloped_mixture_design(1, minority_frac, slope_ratio, majority_sigma=1e-6)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        fit = robust_regression(X, y, method=method, max_iter=1000)
                    self.assertLess(np.max(np.abs(fit.coef - beta_major)), 1e-2)
                    self.assertTrue(
                        fit.degenerate_scale,
                        f"minority_frac={minority_frac} slope_ratio={slope_ratio} method={method}",
                    )

    def test_relative_scale_check_covers_tukey_but_not_huber_at_1e4_majority_noise(self):
        # DOCUMENTED, HONEST LIMITATION (module comment above _robust_weight_collapse in glm.py):
        # at a coarser majority noise (1e-4), tukey's hard redescending weight still crushes the
        # minority below the negligible-weight bar and is correctly flagged; huber's soft weight
        # settles the minority around 1e-5 to 1e-4 -- a genuine fixed point (confirmed stable to
        # 50,000 IRLS iterations during development, not a slow-convergence artifact) that never
        # crosses the FIXED _ROBUST_NEGLIGIBLE_WEIGHT bar (1e-6), so degenerate_scale stays False
        # even though the coefficients already match the majority. Widening the negligible-weight
        # bar itself was investigated and deliberately deferred as a bigger, riskier change than the
        # scale check alone; this test pins the boundary down honestly rather than papering over it.
        for minority_frac, slope_ratio in ((0.05, 2.0), (0.10, 4.0)):
            with self.subTest(minority_frac=minority_frac, slope_ratio=slope_ratio):
                X, y, beta_major, _ = _sloped_mixture_design(1, minority_frac, slope_ratio, majority_sigma=1e-4)
                with warnings.catch_warnings(record=True):
                    warnings.simplefilter("always")
                    fit_tukey = robust_regression(X, y, method="tukey", max_iter=1000)
                with warnings.catch_warnings(record=True):
                    warnings.simplefilter("always")
                    fit_huber = robust_regression(X, y, method="huber", max_iter=1000)
                self.assertLess(np.max(np.abs(fit_tukey.coef - beta_major)), 1e-2)
                self.assertLess(np.max(np.abs(fit_huber.coef - beta_major)), 1e-2)
                self.assertTrue(fit_tukey.degenerate_scale, "tukey should be caught at 1e-4 majority noise")
                self.assertFalse(fit_huber.degenerate_scale, "huber at 1e-4 majority noise is a documented gap")


def _two_col_zero_inflated_design_v2(seed: int, frac: float, n: int = 600) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same fixture shape as ZeroInflatedTwoColumnTest / _two_col_zero_inflated_design above,
    duplicated locally (rather than imported) so this class's fixture is self-contained and its own
    docstring's numbers stay tied to exactly this function."""
    rng = np.random.RandomState(seed)
    predictor = rng.normal(0, 1, n)
    X = np.column_stack([np.ones(n), predictor])
    true_beta = np.array([5.0, 2.0])
    is_nonzero = rng.rand(n) < frac
    y = np.zeros(n)
    y[is_nonzero] = X[is_nonzero] @ true_beta + rng.normal(0.0, 0.3, int(np.count_nonzero(is_nonzero)))
    return X, y, true_beta


class ResponseTieFractionDegradationBandTest(unittest.TestCase):
    """Reviewer's GAP B: fit quality for the shape signal 2 targets (a tied response value) degrades
    well before the theoretical >50% breakdown point, in a roughly symmetric band from ~44% through
    ~56% response-tie-fraction -- confirmed here with an independent sweep of this module's own
    two-column zero-inflated fixture family, matching the reviewer's own account. Honest about what
    lowering _RESPONSE_POINT_MASS_FRACTION to 0.44 does and does not do: it pulls more of the
    clearly-severe part of that band into disclosure; it does not, and structurally cannot, make the
    boundary itself sharp -- that uncertainty moved from 50% to 44%, it did not disappear.
    """

    N_SEEDS = 60  # smaller than the 200-seed sweep used to pick the threshold, kept fast for CI

    @staticmethod
    def _sweep(frac: float, n_seeds: int, max_iter: int = 100) -> tuple[np.ndarray, np.ndarray]:
        """Returns (relative_coef_errors, flagged) arrays, one entry per seed. A non-convergence at
        this max_iter (only seen at the most severe, majority-far-past-50%-tied fracs) counts as a
        full discard (error 1.0) and a flag (the RuntimeError itself names the breakdown -- a crash
        with a named cause is this module's OTHER form of disclosure, not an absence of one)."""
        rel_errs = []
        flags = []
        for seed in range(n_seeds):
            X, y, true_beta = _two_col_zero_inflated_design_v2(seed, frac)
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                try:
                    fit = robust_regression(X, y, method="huber", max_iter=max_iter)
                except RuntimeError:
                    rel_errs.append(1.0)
                    flags.append(1.0)
                    continue
            rel_errs.append(float(np.linalg.norm(fit.coef - true_beta) / np.linalg.norm(true_beta)))
            flags.append(1.0 if fit.degenerate_scale else 0.0)
        return np.array(rel_errs), np.array(flags)

    def test_degradation_band_is_severe_on_both_sides_of_the_old_50pct_cutoff(self):
        # frac is P(nonzero); tie-fraction is 1 - frac. frac=0.56/0.54 -> ~44%/46% tied (BELOW the
        # old 50% cutoff, previously almost never disclosed); frac=0.46/0.44 -> ~54%/56% tied (above
        # it, always disclosed even under the old rule). All four are already severely degraded.
        for frac in (0.44, 0.46, 0.54, 0.56):
            with self.subTest(frac=frac):
                rel_errs, _ = self._sweep(frac, self.N_SEEDS)
                self.assertGreater(
                    float(np.mean(rel_errs)),
                    0.25,
                    f"frac={frac} (tie~{1 - frac:.0%}) expected already-severe error near the old 50% line",
                )

    def test_new_44pct_cutoff_discloses_most_of_the_just_below_50pct_severe_region(self):
        # frac=0.56 targets ~44% tied: severe error (previous test), and only 1 of 60 seeds here
        # realizes a tie-fraction >= 50% by chance -- so the OLD 0.5 cutoff would have left this
        # region almost entirely undisclosed via signal 2. The new 0.44 cutoff discloses a real
        # majority of it instead (confirmed directly: 29/60 seeds flagged here that would NOT have
        # been under the old rule) -- a genuine, substantial improvement, asserted with margin below
        # what was actually measured so this does not become flaky on unrelated changes.
        _, flags = self._sweep(0.56, self.N_SEEDS)
        self.assertGreater(
            float(np.mean(flags)),
            0.3,
            "the just-below-the-old-cutoff severe region (~44% tied) should now be disclosed for a "
            "real share of seeds -- it was almost NEVER disclosed under the old 0.5 cutoff",
        )

    def test_boundary_uncertainty_persists_at_the_new_cutoff_documented_not_hidden(self):
        # DOCUMENTED LIMITATION, not a bug: exactly near whatever the current cutoff is, disclosure
        # is inherently close to a coin flip on the realized per-seed tie-fraction, not a clean
        # function of fit quality -- this pins that down at the NEW 0.44 cutoff so a future change to
        # the constant is forced to re-examine this comment rather than silently invalidating it.
        rel_errs, flags = self._sweep(0.56, self.N_SEEDS)  # targets ~44% realized tie, straddling 0.44
        # both flagged and unflagged seeds exist at this frac (i.e. a real straddle, not one-sided) ...
        self.assertGreater(int(np.sum(flags)), 0)
        self.assertGreater(int(np.sum(1 - flags)), 0)
        # ... and error is already meaningfully bad pretty much regardless of which side a seed lands
        # on: the UNflagged group's error is NOT reliably "fine" just because it escaped disclosure
        unflagged_median = float(np.median(rel_errs[flags == 0]))
        self.assertGreater(
            unflagged_median,
            0.10,
            "even seeds that land on the UNflagged side of the new cutoff should still show "
            "meaningfully degraded error here -- that is the point of GAP B, and of this test",
        )

    def test_response_point_mass_fraction_constant_matches_the_module(self):
        # pins the actual constant this whole class validates, so a future edit that changes
        # _RESPONSE_POINT_MASS_FRACTION without re-reading this class's reasoning fails loudly here
        # instead of silently invalidating the margins chosen above.
        self.assertEqual(_RESPONSE_POINT_MASS_FRACTION, 0.44)
        # and a sanity check on the tie-detector itself, independent of robust_regression: a response
        # that is exactly 44% tied at one value is (correctly) at the boundary, not below it.
        y = np.concatenate([np.zeros(44), np.arange(1, 57)])
        self.assertGreaterEqual(_response_point_mass_fraction(y), _RESPONSE_POINT_MASS_FRACTION)


if __name__ == "__main__":
    unittest.main()
