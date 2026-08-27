"""Campaign-3 T1-F1: the multivariate/diagonal Gaussians must not lose their covariance to a data offset.

Before this repair both estimators computed their covariance from raw reduced moments -- the classic
cancellation-prone ``E[xx^T] - mu mu^T`` -- so ordinary data with a large constant offset (epoch
seconds are ~1.7e9; UTM coordinates, prices in minor units and row ids are the same shape) fitted a
covariance that was silently wrong and then stopped being computable at all. Measured on the
pre-repair tree, n=1500 bivariate normal, sd 1, corr 0.9:

    full covariance   offset 1e7   -> var00 rel err 0.083, corr 0.883 vs exact 0.897, SILENT
    full covariance   offset 1e8   -> ValueError "covar is not positive semi-definite within
                                      tolerance ... refusing to self-heal"
    diagonal          offset 1e7   -> var rel err 0.32, SILENT
    diagonal          offset 1.7e9 -> one coordinate collapsed onto the 1e-8 floor, the other
                                      3466x too large, with no repair naming the real cause

The univariate GaussianAccumulator had already been repaired for exactly this failure mode with a
conditioning-gated shift-anchored moment track; these tests pin the same treatment on the vector
families, plus the two consequences that repair exposed: the diagonal scorer's expanded quadratic
form (which made EM select the WORSE of its own iterates at offset 1.7e9), and the fact that raw
statistics arriving WITHOUT an anchor cannot be corrected and must therefore say so.

Tolerances are stated against an exact-Fraction reference computed here on the same float64 array
that is fitted, so nothing incidental to a seed or a platform is pinned.
"""

import pickle
import unittest
import warnings
from fractions import Fraction

import numpy as np

from mixle.stats.bayes.multivariate_normal_gamma import MultivariateNormalGammaDistribution
from mixle.stats.bayes.normal_wishart import NormalWishartDistribution
from mixle.stats.multivariate.diagonal_gaussian import (
    DiagonalGaussianDistribution,
    DiagonalGaussianEstimator,
)
from mixle.stats.multivariate.multivariate_gaussian import (
    MultivariateGaussianDistribution,
    MultivariateGaussianEstimator,
)
from mixle.utils.serialization import from_json, to_json

# Offsets a user reaches by accident: a million, ten million, a hundred million, and epoch seconds.
OFFSETS = (0.0, 1.0e6, 1.0e7, 1.0e8, 1.7e9)

# ``ridge=0.0`` asks for the exact MLE; the estimator documents that it then falls back to the
# absolute ``min_covar`` as the last-resort jitter, adding exactly this to every diagonal entry.
MIN_COVAR = 1.0e-8


def exact_covariance(x: np.ndarray) -> np.ndarray:
    """Population covariance of the rows of ``x`` in exact rational arithmetic."""
    n, d = x.shape
    columns = [[Fraction(float(v)) for v in x[:, j]] for j in range(d)]
    means = [sum(c) / n for c in columns]
    out = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            total = sum((a - means[i]) * (b - means[j]) for a, b in zip(columns[i], columns[j]))
            out[i, j] = float(total / n)
    return out


def sample(seed: int, n: int, cov: list[list[float]]) -> np.ndarray:
    return np.random.default_rng(seed).multivariate_normal(np.zeros(len(cov)), cov, size=n)


def at_both_offsets(rows: np.ndarray, offset: float) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(x + c, (x + c) - c)`` -- the SAME float64 values, seen at two offsets.

    Adding ``c`` re-quantizes the data (``ulp(1.7e9)`` is 2.4e-7 against a unit spread), so
    ``fit(x + c)`` and ``fit(x)`` describe genuinely different samples and cannot agree closer than
    ~1e-8 however exact the estimator is. Subtracting ``c`` straight back is exact for operands of
    the same magnitude, so this pair isolates the estimator's shift equivariance from float64's own
    representation limit. Tests that want an absolute yardstick instead use
    :func:`exact_covariance` on the shifted array.
    """
    shifted = rows + offset
    return shifted, shifted - offset


def fit_full(x: np.ndarray, **kwargs) -> MultivariateGaussianDistribution:
    est = MultivariateGaussianEstimator(dim=x.shape[1], **kwargs)
    acc = est.accumulator_factory().make()
    acc.seq_update(np.asarray(x, dtype=float), np.ones(len(x)), None)
    return est.estimate(float(len(x)), acc.value())


def fit_diagonal(x: np.ndarray, **kwargs) -> DiagonalGaussianDistribution:
    est = DiagonalGaussianEstimator(dim=x.shape[1], **kwargs)
    acc = est.accumulator_factory().make()
    acc.seq_update(np.asarray(x, dtype=float), np.ones(len(x)), None)
    return est.estimate(float(len(x)), acc.value())


class ShiftEquivarianceTest(unittest.TestCase):
    """fit(x + c) must recover the covariance of ``x + c``, at every offset a user can hit."""

    def test_full_covariance_matches_the_exact_covariance_at_every_offset(self):
        base = sample(20260827, 1500, [[1.0, 0.9], [0.9, 1.0]])
        for offset in OFFSETS:
            with self.subTest(offset=offset):
                x = base + offset
                reference = exact_covariance(x)
                # ridge=0.0 is the exact MLE up to the documented absolute min_covar jitter.
                fitted = np.asarray(fit_full(x, ridge=0.0).covar) - MIN_COVAR * np.eye(2)
                relative = np.max(np.abs(fitted - reference) / np.abs(reference))
                # Pre-repair: 0.083 at 1e7 and a ValueError at 1e8 and beyond.
                self.assertLess(relative, 1.0e-12, "offset %g covariance rel err %g" % (offset, relative))

    def test_diagonal_variances_match_the_exact_variances_at_every_offset(self):
        base = np.random.default_rng(99).standard_normal((1500, 3)) * np.asarray([1.0, 0.5, 2.0])
        for offset in OFFSETS:
            with self.subTest(offset=offset):
                x = base + offset
                reference = np.diag(exact_covariance(x))
                fitted = np.asarray(fit_diagonal(x, ridge=0.0).covar)
                relative = np.max(np.abs(fitted - reference) / reference)
                # Pre-repair: 0.32 at 1e7, and thousands-fold wrong (or floored) at 1.7e9.
                self.assertLess(relative, 1.0e-12, "offset %g variance rel err %g" % (offset, relative))

    def test_fitted_correlation_no_longer_drifts_at_offset_1e7(self):
        # The worst kind of pre-repair failure: no error, no repair note, a plausible-looking number.
        base = sample(20260827, 1500, [[1.0, 0.9], [0.9, 1.0]])
        x = base + 1.0e7
        reference = exact_covariance(x)
        exact_corr = reference[0, 1] / np.sqrt(reference[0, 0] * reference[1, 1])
        cov = np.asarray(fit_full(x, ridge=0.0).covar) - MIN_COVAR * np.eye(2)
        corr = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
        # Pre-repair this read 0.883452 against an exact 0.897187, with nothing said about it.
        self.assertAlmostEqual(corr, exact_corr, places=12)

    def test_offset_data_is_no_longer_refused_as_non_positive_semidefinite(self):
        # Pre-repair these two offsets raised out of _robust_cho_factor, blaming an input whose
        # two-pass covariance is exact.
        base = sample(20260827, 1500, [[1.0, 0.9], [0.9, 1.0]])
        for offset in (1.0e8, 1.7e9):
            with self.subTest(offset=offset):
                model = fit_full(base + offset, ridge=0.0)
                self.assertTrue(np.all(np.linalg.eigvalsh(np.asarray(model.covar)) > 0.0))

    def test_scalar_update_path_agrees_with_the_vectorized_one_at_offset(self):
        x = np.random.default_rng(11).standard_normal((200, 2)) + 1.7e9
        est = MultivariateGaussianEstimator(dim=2, ridge=0.0)
        by_row = est.accumulator_factory().make()
        for row in x:
            by_row.update(row, 1.0, None)
        vectorized = est.accumulator_factory().make()
        vectorized.seq_update(x, np.ones(len(x)), None)
        a = np.asarray(est.estimate(float(len(x)), by_row.value()).covar)
        b = np.asarray(est.estimate(float(len(x)), vectorized.value()).covar)
        np.testing.assert_allclose(a, b, rtol=1.0e-12, atol=0.0)


class AnchoredAccumulationProtocolTest(unittest.TestCase):
    """The anchored track has to survive every path an accumulator's statistics take."""

    def test_combine_pools_accumulators_carrying_different_anchors(self):
        x = sample(424242, 1200, [[1.0, 0.7], [0.7, 2.0]]) + 1.7e9
        est = MultivariateGaussianEstimator(dim=2, ridge=0.0)
        pool = est.accumulator_factory().make()
        for lo in range(0, len(x), 250):
            chunk = x[lo : lo + 250]
            part = est.accumulator_factory().make()
            part.seq_update(chunk, np.ones(len(chunk)), None)
            # Each chunk anchors on its own first row, so this exercises the Chan-style merge
            # across genuinely different anchors, not a shared one.
            self.assertIsNotNone(part._anchor)
            pool.combine(part.value())
        pooled = np.asarray(est.estimate(float(len(x)), pool.value()).covar) - MIN_COVAR * np.eye(2)
        reference = exact_covariance(x)
        self.assertLess(np.max(np.abs(pooled - reference) / np.abs(reference)), 1.0e-12)

    def test_diagonal_combine_pools_accumulators_carrying_different_anchors(self):
        x = np.random.default_rng(3).standard_normal((1200, 3)) + 1.7e9
        est = DiagonalGaussianEstimator(dim=3, ridge=0.0)
        pool = est.accumulator_factory().make()
        for lo in range(0, len(x), 300):
            chunk = x[lo : lo + 300]
            part = est.accumulator_factory().make()
            part.seq_update(chunk, np.ones(len(chunk)), None)
            pool.combine(part.value())
        pooled = np.asarray(est.estimate(float(len(x)), pool.value()).covar)
        reference = np.diag(exact_covariance(x))
        self.assertLess(np.max(np.abs(pooled - reference) / reference), 1.0e-12)

    def test_value_round_trips_through_pickle_with_its_anchored_payload(self):
        # The Spark/multiprocessing reducers move accumulator values through pickle; a tuple
        # subclass with a payload-bearing __new__ does not pickle without help.
        for est, x in (
            (MultivariateGaussianEstimator(dim=2, ridge=0.0), sample(7, 400, [[1.0, 0.5], [0.5, 1.0]]) + 1.7e9),
            (DiagonalGaussianEstimator(dim=3, ridge=0.0), np.random.default_rng(7).standard_normal((400, 3)) + 1.7e9),
        ):
            with self.subTest(estimator=type(est).__name__):
                acc = est.accumulator_factory().make()
                acc.seq_update(x, np.ones(len(x)), None)
                value = acc.value()
                self.assertIsNotNone(getattr(value, "anchored", None))
                restored = pickle.loads(pickle.dumps(value))
                self.assertIsNotNone(getattr(restored, "anchored", None))
                direct = np.asarray(est.estimate(float(len(x)), value).covar)
                through_pickle = np.asarray(est.estimate(float(len(x)), restored).covar)
                np.testing.assert_array_equal(direct, through_pickle)
                # ... and through from_value, which is how a restored value re-enters an accumulator.
                revived = est.accumulator_factory().make().from_value(restored)
                np.testing.assert_array_equal(direct, np.asarray(est.estimate(float(len(x)), revived.value()).covar))

    def test_accumulator_itself_pickles_with_its_anchored_state(self):
        x = sample(7, 300, [[1.0, 0.5], [0.5, 1.0]]) + 1.7e9
        est = MultivariateGaussianEstimator(dim=2, ridge=0.0)
        acc = est.accumulator_factory().make()
        acc.seq_update(x, np.ones(len(x)), None)
        revived = pickle.loads(pickle.dumps(acc))
        np.testing.assert_array_equal(
            np.asarray(est.estimate(float(len(x)), acc.value()).covar),
            np.asarray(est.estimate(float(len(x)), revived.value()).covar),
        )

    def test_fitted_models_still_round_trip_through_json_and_pickle(self):
        # The anchored track is accumulator state, never distribution state: an offset-fitted model
        # must serialize exactly as any other one does.
        models = [
            fit_full(sample(7, 300, [[1.0, 0.5], [0.5, 1.0]]) + 1.7e9),
            fit_diagonal(np.random.default_rng(7).standard_normal((300, 3)) + 1.7e9),
        ]
        for model in models:
            with self.subTest(model=type(model).__name__):
                for revived in (from_json(to_json(model)), pickle.loads(pickle.dumps(model))):
                    np.testing.assert_array_equal(np.asarray(revived.covar), np.asarray(model.covar))
                    np.testing.assert_array_equal(np.asarray(revived.mu), np.asarray(model.mu))

    def test_scale_matches_a_reweighted_seq_update(self):
        """The accumulator/reweighted-seq_update invariant, at the offsets that need the anchor.

        ``compute_metadata_test`` enforces this for the declared families at rtol 1e-10; a previous
        repair in this release broke it by letting a degenerate component's scatter come out
        ``+1e-30`` on one path and ``0.0`` on the other. The degenerate cases below therefore demand
        EXACT equality, not closeness.
        """
        offset_rows = sample(11, 60, [[1.0, 0.6], [0.6, 1.5]]) + 1.7e9
        diag_rows = np.random.default_rng(11).standard_normal((60, 3)) + 1.7e9
        cases = (
            ("full/offset", MultivariateGaussianEstimator(dim=2), offset_rows, False),
            ("full/degenerate", MultivariateGaussianEstimator(dim=2), np.repeat(offset_rows[:1], 4, axis=0), True),
            ("diagonal/offset", DiagonalGaussianEstimator(dim=3), diag_rows, False),
            ("diagonal/degenerate", DiagonalGaussianEstimator(dim=3), np.repeat(diag_rows[:1], 4, axis=0), True),
        )
        c = 0.37
        for label, est, rows, exact in cases:
            with self.subTest(case=label):
                weights = np.linspace(0.5, 1.5, len(rows))
                scaled = est.accumulator_factory().make()
                scaled.seq_update(rows, weights, None)
                self.assertIs(scaled.scale(c), scaled)
                reweighted = est.accumulator_factory().make()
                reweighted.seq_update(rows, weights * c, None)
                nobs = float(weights.sum() * c)
                a = np.asarray(est.estimate(nobs, scaled.value()).covar)
                b = np.asarray(est.estimate(nobs, reweighted.value()).covar)
                if exact:
                    np.testing.assert_array_equal(a, b)
                else:
                    np.testing.assert_allclose(a, b, rtol=1.0e-10, atol=0.0)

    def test_a_degenerate_component_gets_an_exactly_zero_scatter(self):
        # Identical observations carry no spread at all. The scatter has to be exactly 0.0 -- an
        # O(eps) residue would be read as a real spread by the scale-relative ridge/floor and make
        # two algebraically equivalent accumulation paths disagree.
        row = np.asarray([1.7e9, -3.2e8])
        est = MultivariateGaussianEstimator(dim=2, ridge=0.0)
        acc = est.accumulator_factory().make()
        acc.seq_update(np.repeat(row[None, :], 5, axis=0), np.ones(5), None)
        model = est.estimate(5.0, acc.value())
        np.testing.assert_array_equal(np.asarray(model.covar), MIN_COVAR * np.eye(2))

    def test_an_inconsistent_anchored_payload_is_ignored_rather_than_trusted_or_refused(self):
        """A payload that contradicts its own tuple must change nothing, and must not raise.

        The anchored moments are a payload on a tuple subclass, so a caller can hand-build one that
        disagrees with the ``(sum, sum2, count)`` it rides on. The estimate that tuple alone implies
        is the contract; the payload may only make it more accurate, never different -- and a
        payload the estimator declines to use is not an error, it is the historical raw path.
        """
        x = sample(5, 200, [[1.0, 0.4], [0.4, 1.0]])
        est = MultivariateGaussianEstimator(dim=2, ridge=0.0)
        acc = est.accumulator_factory().make()
        acc.seq_update(x, np.ones(len(x)), None)
        raw = tuple(acc.value())
        baseline = np.asarray(est.estimate(float(len(x)), raw).covar)

        from mixle.stats.multivariate.multivariate_gaussian import MultivariateGaussianSuffStat

        tampered = MultivariateGaussianSuffStat(
            raw[0],
            raw[1],
            raw[2],
            # An anchor and moments that describe completely different data.
            anchored=(np.asarray([500.0, -500.0]), np.asarray([1.0, 1.0]), np.eye(2)),
        )
        np.testing.assert_array_equal(baseline, np.asarray(est.estimate(float(len(x)), tampered).covar))


class WellConditionedFitsAreUnchangedTest(unittest.TestCase):
    """The gate must not change, slow, or annotate ordinary work."""

    def test_well_conditioned_seq_update_keeps_the_historical_plain_tuple(self):
        x = sample(3, 500, [[1.0, 0.3], [0.3, 1.0]])
        est = MultivariateGaussianEstimator(dim=2)
        acc = est.accumulator_factory().make()
        acc.seq_update(x, np.ones(len(x)), None)
        value = acc.value()
        # No anchor activated, so the value is the exact 3-tuple this family always returned.
        self.assertIsNone(acc._anchor)
        self.assertIs(type(value), tuple)
        # ... and it is BIT-identical to the historical single-pass reduction, which is exactly
        # these two expressions -- the gate must cost accuracy nowhere on well-conditioned data.
        weighted = np.multiply(x.T, np.ones(len(x)))
        np.testing.assert_array_equal(value[0], weighted.sum(axis=1))
        np.testing.assert_array_equal(value[1], weighted @ x)
        self.assertEqual(value[2], float(len(x)))

    def test_ordinary_fits_report_no_repair_and_emit_no_warning(self):
        full_rows = sample(3, 500, [[1.0, 0.3], [0.3, 1.0]])
        diag_rows = np.random.default_rng(3).standard_normal((500, 3))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            models = [
                fit_full(full_rows),
                fit_diagonal(diag_rows),
                fit_full(full_rows + 1.7e9),
                fit_diagonal(diag_rows + 1.7e9),
            ]
        self.assertEqual([], [str(w.message) for w in caught])
        for model in models:
            self.assertEqual((), model.numerical_repairs())

    def test_a_rank_deficient_fit_still_discloses_its_ridge(self):
        # The disclosure the repair must not silence: fewer points than dimensions.
        model = fit_full(np.random.default_rng(4).standard_normal((3, 5)))
        repairs = model.numerical_repairs()
        self.assertTrue(repairs, "a rank-deficient covariance must still report its ridge")
        self.assertTrue(any("covariance-ridged" in note for note in repairs), repairs)

    def test_the_conditioning_receipt_now_describes_the_data_not_the_cancellation(self):
        # track_conditioning used to compute its eigenspectrum from the already-corrupted scatter,
        # so it certified a 15%-wrong fit as near_degenerate=False with plausible eigenvalues.
        base = sample(3, 500, [[1.0, 0.3], [0.3, 1.0]])
        plain = fit_full(base, track_conditioning=True).conditioning_receipt
        shifted = fit_full(base + 1.7e9, track_conditioning=True).conditioning_receipt
        np.testing.assert_allclose(shifted.eigenvalues, plain.eigenvalues, rtol=1.0e-8, atol=0.0)
        self.assertEqual(shifted.near_degenerate, plain.near_degenerate)


class RawOnlyStatisticsAreNamedTest(unittest.TestCase):
    """Statistics that arrive already reduced and unanchored cannot be repaired -- so they must speak."""

    def test_ill_conditioned_raw_statistics_warn_with_the_real_reason(self):
        x = np.random.default_rng(13).standard_normal((500, 2)) + 1.7e9
        cases = (
            (MultivariateGaussianEstimator(dim=2), MultivariateGaussianEstimator(dim=2), "E[xx^T]"),
            (DiagonalGaussianEstimator(dim=2), DiagonalGaussianEstimator(dim=2), "E[x^2]"),
        )
        for accumulating, estimating, marker in cases:
            with self.subTest(estimator=type(estimating).__name__):
                acc = accumulating.accumulator_factory().make()
                acc.seq_update(x, np.ones(len(x)), None)
                stripped = tuple(acc.value())  # a plain tuple: the anchored payload is gone
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    try:
                        estimating.estimate(float(len(x)), stripped)
                    except ValueError:
                        # The full-covariance estimator may still refuse the corrupted scatter; the
                        # point of this test is that the warning explains it either way.
                        pass
                messages = [str(w.message) for w in caught]
                self.assertTrue(messages, "raw ill-conditioned statistics must not be silent")
                self.assertTrue(any(marker in m for m in messages), messages)
                self.assertTrue(any("shift-anchored" in m for m in messages), messages)

    def test_well_scaled_and_degenerate_raw_statistics_stay_quiet(self):
        """The gate for the warning must not also fire on the states the library legitimately produces.

        A degenerate/single-point EM component has no spread by construction; that is what the ridge
        and the variance floor exist for, and they already disclose it. Warning there would fire on
        every starved component of every mixture fit.
        """
        well_scaled = np.random.default_rng(17).standard_normal((300, 3))
        single_point = np.repeat(well_scaled[:1], 3, axis=0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for rows in (well_scaled, single_point):
                for est in (DiagonalGaussianEstimator(dim=3), MultivariateGaussianEstimator(dim=3)):
                    acc = est.accumulator_factory().make()
                    acc.seq_update(rows, np.ones(len(rows)), None)
                    est.estimate(float(len(rows)), acc.value())
        self.assertEqual([], [str(w.message) for w in caught])


class DiagonalScoringSurvivesOffsetsTest(unittest.TestCase):
    """The diagonal scorer's expanded quadratic form cancelled away every digit at large offsets."""

    def test_log_density_is_shift_invariant(self):
        rows = np.random.default_rng(23).standard_normal((50, 3))
        for offset in OFFSETS:
            with self.subTest(offset=offset):
                shifted, recovered = at_both_offsets(rows, offset)
                model = DiagonalGaussianDistribution([offset] * 3, [1.0, 0.5, 2.0])
                plain = DiagonalGaussianDistribution([0.0] * 3, [1.0, 0.5, 2.0])
                enc = model.dist_to_encoder().seq_encode(list(shifted))
                enc0 = plain.dist_to_encoder().seq_encode(list(recovered))
                # Pre-repair, offset 1.7e9 returned round-number garbage here (whole multiples of
                # 1024, from three ~1e18 terms cancelling): the answer had no significant digits.
                np.testing.assert_allclose(model.seq_log_density(enc), plain.seq_log_density(enc0), rtol=1.0e-12)
                np.testing.assert_allclose(
                    float(model.log_density(shifted[0])), float(plain.log_density(recovered[0])), rtol=1.0e-12
                )

    def test_log_density_matches_the_documented_centered_formula(self):
        model = DiagonalGaussianDistribution([1.7e9, 1.7e9], [1.0, 4.0])
        x = np.asarray([1.7e9 + 0.5, 1.7e9 - 1.25])
        mu = np.asarray(model.mu)
        covar = np.asarray(model.covar)
        expected = -0.5 * float(np.sum((x - mu) ** 2 / covar)) - 0.5 * float(np.sum(np.log(2.0 * np.pi * covar)))
        self.assertAlmostEqual(float(model.log_density(x)), expected, places=12)

    def test_em_no_longer_selects_the_worse_iterate_at_large_offset(self):
        # Pre-repair the objective at offset 1.7e9 was round-number garbage (66560.0 and -135168.0
        # on this data), so optimize() returned its own initialization instead of the fitted model.
        from mixle.inference import optimize

        rows = np.random.default_rng(1).standard_normal((200, 2))
        shifted, recovered = at_both_offsets(rows, 1.7e9)
        fits = []
        for data in (recovered, shifted):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = optimize(list(data), DiagonalGaussianEstimator(dim=2), rng=np.random.RandomState(5), out=None)
            fits.append(np.asarray(model.covar))
        # Pre-repair the offset run returned [2.533, 0.765] -- its own random initialization --
        # because the garbage objective scored that above the fitted [0.832, 0.818].
        np.testing.assert_allclose(fits[1], fits[0], rtol=1.0e-12, atol=0.0)


class ConjugatePosteriorsSurviveOffsetsTest(unittest.TestCase):
    """The conjugate M-steps read the same scatter and were wrong in the same way.

    These assert 1e-9 rather than the 1e-12 the maximum-likelihood paths reach, and the gap is the
    PRIOR's own representation limit, not the estimator's: a conjugate prior mean sited at 1.7e9 is
    itself only representable to ``ulp(1.7e9)`` = 2.4e-7, and the posterior scale reads it through
    ``(sample_mean - prior_mean)^2``. Pre-repair these differed by whole percent, and the
    full-covariance posterior could not be computed at all past offset 1e8.
    """

    def test_normal_wishart_posterior_covariance_is_shift_equivariant(self):
        rows = np.random.default_rng(31).standard_normal((400, 2))
        shifted, recovered = at_both_offsets(rows, 1.7e9)
        covars = []
        for data, centre in ((recovered, 0.0), (shifted, 1.7e9)):
            prior = NormalWishartDistribution(np.asarray([centre, centre]), 1.0, np.eye(2), 4.0)
            est = MultivariateGaussianEstimator(dim=2, prior=prior)
            acc = est.accumulator_factory().make()
            acc.seq_update(data, np.ones(len(data)), None)
            covars.append(np.asarray(est.estimate(float(len(data)), acc.value()).covar))
        np.testing.assert_allclose(covars[1], covars[0], rtol=1.0e-9, atol=0.0)

    def test_multivariate_normal_gamma_posterior_variances_are_shift_equivariant(self):
        rows = np.random.default_rng(37).standard_normal((400, 2))
        shifted, recovered = at_both_offsets(rows, 1.7e9)
        covars = []
        for data, centre in ((recovered, 0.0), (shifted, 1.7e9)):
            prior = MultivariateNormalGammaDistribution(
                np.asarray([centre, centre]),
                np.asarray([1.0, 1.0]),
                np.asarray([2.0, 2.0]),
                np.asarray([1.0, 1.0]),
            )
            est = DiagonalGaussianEstimator(dim=2, prior=prior)
            acc = est.accumulator_factory().make()
            acc.seq_update(data, np.ones(len(data)), None)
            covars.append(np.asarray(est.estimate(float(len(data)), acc.value()).covar))
        np.testing.assert_allclose(covars[1], covars[0], rtol=1.0e-9, atol=0.0)


class PseudoCountSmoothingSurvivesOffsetsTest(unittest.TestCase):
    """Pseudo-count smoothing pools a prior moment in; the anchored path must pool the same one."""

    def test_pseudo_count_pooling_is_shift_equivariant_for_both_families(self):
        rows = np.random.default_rng(41).standard_normal((300, 2))
        shifted, recovered = at_both_offsets(rows, 1.7e9)
        shifted_mu0 = np.asarray([1.7e9 + 0.25, 1.7e9 - 0.5])
        full, diagonal = [], []
        for data, mu0 in ((recovered, shifted_mu0 - 1.7e9), (shifted, shifted_mu0)):
            est = MultivariateGaussianEstimator(
                dim=2, pseudo_count=(5.0, 5.0), suff_stat=(mu0, np.asarray([[2.0, 0.3], [0.3, 1.5]])), ridge=0.0
            )
            acc = est.accumulator_factory().make()
            acc.seq_update(data, np.ones(len(data)), None)
            full.append(np.asarray(est.estimate(float(len(data)), acc.value()).covar))

            dest = DiagonalGaussianEstimator(
                dim=2, pseudo_count=(5.0, 5.0), suff_stat=(mu0, np.asarray([2.0, 1.5])), ridge=0.0
            )
            dacc = dest.accumulator_factory().make()
            dacc.seq_update(data, np.ones(len(data)), None)
            diagonal.append(np.asarray(dest.estimate(float(len(data)), dacc.value()).covar))
        np.testing.assert_allclose(full[1], full[0], rtol=1.0e-11, atol=0.0)
        np.testing.assert_allclose(diagonal[1], diagonal[0], rtol=1.0e-11, atol=0.0)


if __name__ == "__main__":
    unittest.main()
