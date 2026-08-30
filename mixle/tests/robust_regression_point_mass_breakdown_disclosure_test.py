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
(GapOneLowMinorityFractionTest through HeteroskedasticNoiselessMajorityMustNotWarnTest) reproduce the
reviewer's own scenarios and are proven to FAIL against the pre-redesign 04ef6ae1 code (verified by
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
disagreement measured on the benign gap-4 fixture, where nothing is wrong at all. It is kept only as
a narrow suppressor of signal 1, never applied to signal 2, precisely because the cap_frac=0.96 case
above is flagged exclusively through signal 2 in this implementation and would be silently un-flagged
by the same suppressor that correctly resolves gap 4.
"""

import unittest
import warnings

import numpy as np

from mixle.inference import robust_regression
from mixle.inference.glm import _ROBUST_SCALE_FLOOR


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


class HeteroskedasticNoiselessMajorityMustNotWarnTest(unittest.TestCase):
    """Reviewer's gap 4: a benign, correctly-recovered heteroskedastic fit -- 80% of rows noiseless
    (lying exactly on the true line), 20% noisier draws of the SAME true relationship -- must not
    warn. Both methods correctly recover the exact true coefficients, so nothing is actually
    discarded, even though (checked directly below) the scale still collapses to its floor and ~20%
    of rows still carry near-zero weight -- confirmed, by direct execution against the pre-redesign
    04ef6ae1 code, to be a real false positive there and not only a wording complaint: every
    assertion below except the exact-recovery ones fails against it (degenerate_scale is True and a
    UserWarning naming a "real minority signal" is raised, which this fit does not have). This is
    resolved for real (not merely by softened wording) by _robust_diverges_from_ols: the robust fit
    lands within ~0.9-2.5% of plain OLS here (across the 4 seeds x 2 methods below), comfortably
    under the suppression threshold and nowhere near the ~76-77% disagreement on the legitimate
    outlier-rejection design this module exists to handle well. This particular OLS-disagreement
    figure is NOT, on its own, a general stand-in for "is this fit actually fine" -- the capped/
    censored fixtures elsewhere in this file show a genuine collapse can read anywhere from ~1% to
    ~73% depending on cap severity and method (see the module comment above
    _robust_weight_collapse in glm.py) -- which is exactly why this check is scoped as a narrow
    suppressor of the weight-collapse signal alone, never a detector in its own right and never
    applied to the response-point-mass signal.
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

    def test_recovers_truth_with_no_warning(self):
        for seed in range(4):
            for method in ("huber", "tukey"):
                with self.subTest(seed=seed, method=method):
                    X, y, beta = self._design(seed)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        fit = robust_regression(X, y, method=method)
                    np.testing.assert_allclose(fit.coef, beta, atol=1e-6)
                    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
                    self.assertEqual(user_warnings, [], f"seed={seed} method={method}: {user_warnings}")
                    self.assertFalse(fit.degenerate_scale, f"seed={seed} method={method}")

    def test_the_old_weight_signal_alone_would_have_fired_here(self):
        # documents WHY this fixture is a genuine test of the OLS-divergence refinement rather than
        # a case the weight signal was already silent on: scale really does sit at the floor and a
        # real 20% of rows really do carry near-zero weight, matching the shape of a genuine
        # collapse by weight pattern alone.
        X, y, _ = self._design(0)
        fit = robust_regression(X, y, method="huber")
        self.assertEqual(fit.scale, _ROBUST_SCALE_FLOOR)
        r = y - X @ fit.coef
        w = np.where(np.abs(r / fit.scale) <= 1.345, 1.0, 1.345 / np.maximum(np.abs(r / fit.scale), 1e-12))
        self.assertGreater(np.mean(w <= 1e-6), 0.15)
        self.assertFalse(fit.degenerate_scale)  # suppressed anyway, by the OLS cross-check


if __name__ == "__main__":
    unittest.main()
