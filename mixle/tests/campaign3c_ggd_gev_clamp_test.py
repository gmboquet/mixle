"""Regression tests for the GGD/GEV anchored-moment clamp repair (campaign 3c, ggd-gev-clamp).

``GeneralizedGaussianEstimator``/``GeneralizedExtremeValueEstimator`` grew a shift-anchored moment
track this release wave (mirroring ``gaussian.py``'s ``_anchored_pooled_variance``), but both still
carried the PRE-split single-sum clamp: the whole reduced second central moment was compared against
a threshold built from ``(4*eps*|mean|)**2`` -- a threshold that scales with the (possibly huge)
*location*, not with the spread being tested. At a mean of ~1e15 with a genuine, representable spread
of sd ~0.5, that threshold (~0.79) exceeded the true variance (~0.24) and the fit silently collapsed
to the degenerate floor (``alpha=1e-6`` / ``scale=min_scale``) -- see
``test_alpha_recovered_at_extreme_magnitude`` / ``test_scale_recovered_at_extreme_magnitude``, which
FAIL before this fix and PASS after.

The fix splits the second (and, since these families carry moments through order four and three
respectively, the higher) central moment into a data-only "core" about the sample's own mean (gated
by the pre-existing RELATIVE 1e-12 cancellation clamp) plus the displacement of the reported location
from that sample mean (gated by the ulp-scale clamp, and recentering the core moments onto the
reported location via the single-group parallel-axis expansion, never a further cancellation). See
``_anchored_central_moments`` in each module.
"""

import unittest

import numpy as np
from scipy.special import gamma

import mixle
from mixle.stats.univariate.continuous.generalized_extreme_value import (
    GeneralizedExtremeValueAccumulator,
    GeneralizedExtremeValueDistribution,
    GeneralizedExtremeValueEstimator,
)
from mixle.stats.univariate.continuous.generalized_gaussian import (
    GeneralizedGaussianAccumulator,
    GeneralizedGaussianDistribution,
    GeneralizedGaussianEstimator,
)


def _grid(rng: np.random.RandomState, draw: np.ndarray) -> np.ndarray:
    """Round draws to a 2**-16 grid, so adding an offset up to ~2**36 is EXACT in float64.

    Keeps the shift-invariance checks below from being confounded by rounding in the addition
    itself -- any residual mismatch then reflects the estimator's own numerics, not float64's
    inability to represent ``base + c`` exactly.
    """
    del rng
    return np.round(draw * 65536.0) / 65536.0


class GeneralizedGaussianClampTestCase(unittest.TestCase):
    """The fitted scale (``alpha``) must not depend on a constant offset of the data."""

    def test_alpha_recovered_at_extreme_magnitude(self):
        # This is the reported defect: genuine data with a large mean and a small but fully
        # representable spread read as constant under the old single-sum clamp. FAILS before this
        # fix (alpha collapses to the 1e-6 floor), PASSES after.
        rng = np.random.RandomState(0)
        sd = 0.5
        x = 1.0e15 + rng.randn(5000) * sd
        acc = GeneralizedGaussianAccumulator()
        acc.seq_update(x, np.ones(len(x)), None)
        self.assertIsNotNone(getattr(acc.value(), "anchored", None))
        dist = GeneralizedGaussianEstimator().estimate(None, acc.value())

        self.assertGreater(dist.alpha, 0.3, "alpha collapsed toward the degenerate floor")
        implied_var = dist.variance()
        # Ground truth: shift by one ACTUAL data point (exact in float64, Sterbenz's lemma) rather
        # than by the reported mean, then compute the central moments directly at small magnitude.
        y = x - x[0]
        self.assertLessEqual(abs(implied_var - np.var(y)) / np.var(y), 1.0e-6)

    def test_shift_invariant_up_to_1_7e9(self):
        rng = np.random.RandomState(7)
        base = _grid(rng, rng.randn(400) * 0.81 + 13.0)
        reference = mixle.inference.fit(base.tolist(), GeneralizedGaussianEstimator())
        for c in (1.0e7, 1.0e9, 1.7e9):
            shifted = mixle.inference.fit((base + c).tolist(), GeneralizedGaussianEstimator())
            self.assertLessEqual(
                abs(shifted.alpha - reference.alpha) / reference.alpha,
                1.0e-9,
                "alpha not shift-invariant at offset %g" % c,
            )
            self.assertLessEqual(abs(shifted.beta - reference.beta), 1.0e-6)
            self.assertAlmostEqual(shifted.mu - c, reference.mu, places=6)
            self.assertEqual(shifted.numerical_repairs(), ())

    def test_well_conditioned_fit_bit_identical_to_historical_path(self):
        # The conditioning gate keeps ordinary data on the exact historical single-pass
        # accumulation: raw reduced moments, no anchored payload, bit-identical estimate.
        rng = np.random.RandomState(3)
        base = _grid(rng, rng.randn(250) * 0.81 + 13.0)
        acc = GeneralizedGaussianAccumulator()
        acc.seq_update(base, np.ones(len(base)), None)
        self.assertIsInstance(acc.value(), tuple)
        self.assertIsNone(getattr(acc.value(), "anchored", None))
        fitted = GeneralizedGaussianEstimator().estimate(None, acc.value())

        # Read the raw sums straight off the accumulator (rather than recomputing them separately)
        # so this only tests the ESTIMATION formula, not whether numpy's summation order matches.
        n, s1, s2, s3, s4 = acc.value()
        mu = s1 / n
        r2, r3, r4 = s2 / n, s3 / n, s4 / n
        m2 = r2 - mu * mu
        m4 = r4 - 4.0 * mu * r3 + 6.0 * mu * mu * r2 - 3.0 * mu**4
        k = m4 / (m2 * m2) - 3.0
        self.assertEqual(fitted.mu, mu)
        self.assertAlmostEqual(
            gamma(5.0 / fitted.beta) * gamma(1.0 / fitted.beta) / gamma(3.0 / fitted.beta) ** 2 - 3.0, k, places=8
        )
        self.assertEqual(fitted.alpha, np.sqrt(m2 * gamma(1.0 / fitted.beta) / gamma(3.0 / fitted.beta)))

    def test_degenerate_data_clamps_to_exactly_the_alpha_floor(self):
        floor = 1.0e-6
        for c in (0.0, 1.7e9):
            seq_fitted = mixle.inference.fit((np.full(50, 3.0) + c).tolist(), GeneralizedGaussianEstimator())
            self.assertEqual(seq_fitted.alpha, floor)
            self.assertEqual(seq_fitted.mu, 3.0 + c)

            acc = GeneralizedGaussianAccumulator()
            for _ in range(50):
                acc.update(3.0 + c, 1.0, None)
            scalar_fitted = GeneralizedGaussianEstimator().estimate(None, acc.value())
            self.assertEqual(scalar_fitted.alpha, floor)

    def test_accumulator_scale_matches_reweighted_seq_update_at_large_offset(self):
        # The accumulator/reweighted-seq_update parity invariant (see compute_metadata_test.py),
        # exercised specifically on data that forces the anchored track live.
        x = np.asarray([-1.0, 0.0, 2.0, 3.5]) + 1.7e9
        weights = np.linspace(0.5, 1.5, 4)
        c = 0.37
        est = GeneralizedGaussianEstimator()
        acc = est.accumulator_factory().make()
        acc.seq_update(x, weights, None)
        scaled = acc.scale(c)
        self.assertIs(scaled, acc)
        self.assertIsNotNone(getattr(scaled.value(), "anchored", None))

        expected = est.accumulator_factory().make()
        expected.seq_update(x, weights * c, None)

        nobs = float(weights.sum() * c)
        scaled_model = est.estimate(nobs, scaled.value())
        expected_model = est.estimate(nobs, expected.value())
        self.assertAlmostEqual(scaled_model.mu, expected_model.mu, places=6)
        self.assertAlmostEqual(scaled_model.alpha, expected_model.alpha, places=6)
        self.assertAlmostEqual(scaled_model.beta, expected_model.beta, places=6)

    def test_pseudo_count_prior_blend_shift_stable(self):
        rng = np.random.RandomState(17)
        base = _grid(rng, rng.randn(200) * 1.2)
        results = []
        for c in (0.0, 2.0**30):
            prior = GeneralizedGaussianDistribution(0.5 + c, 0.9, 2.2)
            est = prior.estimator(pseudo_count=5.0)
            model = mixle.inference.fit((base + c).tolist(), est)
            results.append((model.mu - c, model.alpha, model.beta))
        self.assertLessEqual(abs(results[1][0] - results[0][0]), 1.0e-5)
        self.assertLessEqual(abs(results[1][1] - results[0][1]) / results[0][1], 1.0e-8)
        self.assertLessEqual(abs(results[1][2] - results[0][2]) / results[0][2], 1.0e-6)

    def test_combine_across_anchors_matches_expected_variance(self):
        rng = np.random.RandomState(5)
        base = _grid(rng, rng.randn(200) * 0.6)
        x1 = base[:100] + 1.7e9
        x2 = base[100:] + 1.7e9 + 5.0
        a1, a2 = GeneralizedGaussianAccumulator(), GeneralizedGaussianAccumulator()
        a1.seq_update(x1, np.ones(100), None)
        a2.seq_update(x2, np.ones(100), None)
        a1.combine(a2.value())
        fitted = GeneralizedGaussianEstimator().estimate(None, a1.value())

        pooled = np.concatenate([x1, x2])
        y = pooled - pooled[0]
        expected_var = float(np.var(y))
        implied_var = fitted.variance()
        self.assertLessEqual(abs(implied_var - expected_var) / expected_var, 1.0e-8)


class GeneralizedExtremeValueClampTestCase(unittest.TestCase):
    """The fitted scale must not depend on a constant offset of the data."""

    def test_scale_recovered_at_extreme_magnitude(self):
        # This is the reported defect, on the GEV's own moment track. FAILS before this fix (scale
        # collapses to the min_scale floor), PASSES after.
        rng = np.random.RandomState(0)
        sd = 0.5
        x = 1.0e15 + rng.standard_gamma(3.0, size=5000) * (sd / np.sqrt(3.0))
        acc = GeneralizedExtremeValueAccumulator()
        acc.seq_update(x, np.ones(len(x)), None)
        self.assertIsNotNone(getattr(acc.value(), "anchored", None))
        dist = GeneralizedExtremeValueEstimator().estimate(None, acc.value())

        self.assertGreater(dist.scale, 0.1, "scale collapsed toward the degenerate floor")
        # Ground truth: shift by one ACTUAL data point (exact in float64, Sterbenz's lemma) rather
        # than by the reported loc, then compare the theoretical sd implied by (scale, shape)
        # against the sample sd computed directly at small magnitude.
        y = x - x[0]
        self.assertLessEqual(abs(np.sqrt(dist.variance()) - np.std(y)) / np.std(y), 1.0e-6)

    def test_shift_invariant_up_to_1_7e9(self):
        rng = np.random.RandomState(7)
        base = _grid(rng, rng.standard_gamma(3.0, size=400) * 0.4)
        reference = mixle.inference.fit(base.tolist(), GeneralizedExtremeValueEstimator())
        for c in (1.0e7, 1.0e9, 1.7e9):
            shifted = mixle.inference.fit((base + c).tolist(), GeneralizedExtremeValueEstimator())
            self.assertLessEqual(
                abs(shifted.scale - reference.scale) / reference.scale,
                1.0e-9,
                "scale not shift-invariant at offset %g" % c,
            )
            self.assertLessEqual(abs(shifted.shape - reference.shape), 1.0e-6)
            self.assertAlmostEqual(shifted.loc - c, reference.loc, places=6)
            self.assertEqual(shifted.numerical_repairs(), ())

    def test_well_conditioned_fit_bit_identical_to_historical_path(self):
        rng = np.random.RandomState(3)
        base = _grid(rng, rng.standard_gamma(3.0, size=250) * 0.4)
        acc = GeneralizedExtremeValueAccumulator()
        acc.seq_update(base, np.ones(len(base)), None)
        self.assertIsInstance(acc.value(), tuple)
        self.assertIsNone(getattr(acc.value(), "anchored", None))
        fitted = GeneralizedExtremeValueEstimator().estimate(None, acc.value())

        # Read the raw sums straight off the accumulator (rather than recomputing them separately)
        # so this only tests the ESTIMATION formula, not whether numpy's summation order matches.
        # ``loc`` is the GEV location PARAMETER, not the sample mean (the law is skewed) -- but
        # method-of-moments matches the mean and variance exactly regardless of whether the
        # support-covering clamp later moves the shape, so the fitted distribution's OWN mean/
        # variance accessors must reproduce the sample's raw moments.
        s1, s2, s3, n = acc.value()[:4]
        mean, var = s1 / n, s2 / n - (s1 / n) ** 2
        self.assertLessEqual(abs(fitted.mean() - mean) / max(abs(mean), 1.0), 1.0e-9)
        self.assertLessEqual(abs(fitted.variance() - var) / var, 1.0e-9)
        # Calling estimate() twice on the identical (unchanged, ``anchored is None``) input is
        # deterministic -- confirms this restructure introduced no incidental state.
        again = GeneralizedExtremeValueEstimator().estimate(None, acc.value())
        self.assertEqual((fitted.loc, fitted.scale, fitted.shape), (again.loc, again.scale, again.shape))

    def test_degenerate_data_clamps_to_exactly_the_scale_floor(self):
        floor = GeneralizedExtremeValueEstimator().min_scale
        for c in (0.0, 1.7e9):
            seq_fitted = mixle.inference.fit((np.full(50, 3.0) + c).tolist(), GeneralizedExtremeValueEstimator())
            self.assertEqual(seq_fitted.scale, floor)
            self.assertEqual(seq_fitted.loc, 3.0 + c)

            acc = GeneralizedExtremeValueAccumulator()
            for _ in range(50):
                acc.update(3.0 + c, 1.0, None)
            scalar_fitted = GeneralizedExtremeValueEstimator().estimate(None, acc.value())
            self.assertEqual(scalar_fitted.scale, floor)

    def test_accumulator_scale_matches_reweighted_seq_update_at_large_offset(self):
        x = np.asarray([-1.0, 0.0, 2.0]) + 1.7e9
        weights = np.linspace(0.5, 1.5, 3)
        c = 0.37
        est = GeneralizedExtremeValueEstimator()
        acc = est.accumulator_factory().make()
        acc.seq_update(x, weights, None)
        scaled = acc.scale(c)
        self.assertIs(scaled, acc)
        self.assertIsNotNone(getattr(scaled.value(), "anchored", None))

        expected = est.accumulator_factory().make()
        expected.seq_update(x, weights * c, None)

        nobs = float(weights.sum() * c)
        scaled_model = est.estimate(nobs, scaled.value())
        expected_model = est.estimate(nobs, expected.value())
        self.assertAlmostEqual(scaled_model.loc, expected_model.loc, places=6)
        self.assertAlmostEqual(scaled_model.scale, expected_model.scale, places=6)
        self.assertAlmostEqual(scaled_model.shape, expected_model.shape, places=6)

    def test_pseudo_count_prior_blend_shift_stable(self):
        rng = np.random.RandomState(17)
        base = _grid(rng, rng.standard_gamma(3.0, size=200) * 0.4)
        results = []
        for c in (0.0, 2.0**30):
            prior = GeneralizedExtremeValueDistribution(0.5 + c, 0.9, 0.1)
            est = prior.estimator(pseudo_count=5.0)
            model = mixle.inference.fit((base + c).tolist(), est)
            results.append((model.loc - c, model.scale, model.shape))
        self.assertLessEqual(abs(results[1][0] - results[0][0]), 1.0e-5)
        self.assertLessEqual(abs(results[1][1] - results[0][1]) / results[0][1], 1.0e-6)
        self.assertLessEqual(abs(results[1][2] - results[0][2]), 1.0e-6)

    def test_combine_across_anchors_matches_expected_variance(self):
        rng = np.random.RandomState(5)
        base = _grid(rng, rng.standard_gamma(3.0, size=200) * 0.4)
        x1 = base[:100] + 1.7e9
        x2 = base[100:] + 1.7e9 + 5.0
        a1, a2 = GeneralizedExtremeValueAccumulator(), GeneralizedExtremeValueAccumulator()
        a1.seq_update(x1, np.ones(100), None)
        a2.seq_update(x2, np.ones(100), None)
        a1.combine(a2.value())
        fitted = GeneralizedExtremeValueEstimator().estimate(None, a1.value())

        pooled = np.concatenate([x1, x2])
        y = pooled - pooled[0]
        expected_var = float(np.var(y))
        # var, not scale directly, since scale also folds in the shape -- but a variance match this
        # tight already rules out the cancellation the old clamp let through.
        g1, g2 = gamma(1.0 - fitted.shape), gamma(1.0 - 2.0 * fitted.shape)
        implied_var = (
            fitted.scale**2 * (g2 - g1 * g1) / (fitted.shape * fitted.shape)
            if abs(fitted.shape) > 1.0e-8
            else (fitted.scale * np.pi) ** 2 / 6.0
        )
        self.assertLessEqual(abs(implied_var - expected_var) / expected_var, 1.0e-6)


if __name__ == "__main__":
    unittest.main()
