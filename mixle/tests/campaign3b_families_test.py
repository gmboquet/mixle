"""Shift-equivariance of the GeneralizedGaussian and GEV method-of-moments M-steps.

Both families fit by matching reduced moments differenced out of raw power sums. That form loses
roughly ``k*log2(|mean|/sd)`` bits for a ``k``-th moment, so before the shift-anchored moment track
was added both estimators degenerated *silently* on data carried at a large offset:

  * ``GeneralizedGaussianEstimator`` on sd~0.70 data at offset 1.7e9 returned ``alpha = 8.77e-3``
    (118x too small) with ``beta`` pinned to its lower bound 0.25, costing 1468 nats of
    log-likelihood on its own training data -- no warning, no ``numerical_repairs()`` entry.
  * ``GeneralizedExtremeValueEstimator`` on sd~0.61 data at offset 1.7e9 returned ``scale = 22.6``
    (58x too large) with ``shape`` pinned to its lower bound -1.0 -- i.e. a *bounded-support* model
    for unbounded data -- costing 13621 nats, and leaking a raw numpy ``overflow encountered in exp``
    RuntimeWarning as the only hint.

The tests below pin the property that rules both out: the fit must be shift-equivariant.
"""

import copy
import math
import pickle
import unittest
import warnings

import numpy as np

from mixle.inference.estimation import optimize
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

# 1.7e9 with base values quantized onto a 1/256 grid: ``offset + value`` then needs mantissa bits
# from 2^30 down to 2^-8, comfortably inside float64's 53, so every shifted sample is represented
# EXACTLY and the anchored differences ``x_i - x_0`` come out exactly equal to ``v_i - v_0`` at every
# offset. That turns "shift-equivariant to within the data's own rounding" into "bit-identical" for
# any two offsets that both take the anchored path, which is what these tests assert. A separate test
# uses un-quantized data and the looser tolerance the input rounding actually permits.
LARGE_OFFSET = 1.7e9
# All three sit far enough above the samples' spread to trip the conditioning gate, so all three take
# the anchored path and must agree bit for bit.
ANCHORED_OFFSETS = (1.0e3, 1.0e6, LARGE_OFFSET)


def _grid(values):
    """Quantize onto a 1/256 grid so ``LARGE_OFFSET + v`` is exact in float64."""
    return [float(np.round(np.float64(v) * 256.0) / 256.0) for v in values]


def _gg_sample(n=2000, seed=11):
    return _grid(GeneralizedGaussianDistribution(0.0, 1.0, 2.0).sampler(seed).sample(n))


def _gev_sample(n=4000, seed=3):
    # The reproduction's base law: exp(N(0, 0.5)) - 1, sd ~0.52-0.61, right-skewed.
    return _grid(np.exp(np.random.RandomState(seed).normal(0.0, 0.5, size=n)) - 1.0)


class GeneralizedGaussianShiftEquivarianceTest(unittest.TestCase):
    def test_explicit_prototype_fit_is_shift_equivariant_at_1e9(self):
        base = _gg_sample()
        fit0 = optimize(list(base), estimator=GeneralizedGaussianDistribution(0.0, 1.0, 2.0))
        reference = None
        for offset in ANCHORED_OFFSETS:
            with self.subTest(offset=offset):
                shifted = [offset + v for v in base]
                fit = optimize(shifted, estimator=GeneralizedGaussianDistribution(offset, 1.0, 2.0))
                # Against the un-shifted fit: that one takes the raw path (its |mean|/spread ratio is
                # tiny), so the two evaluate different but algebraically equal expressions and may
                # differ in the last couple of ulps. Before the fix alpha differed by 118x.
                self.assertAlmostEqual(fit.alpha / fit0.alpha, 1.0, places=13)
                self.assertAlmostEqual(fit.beta / fit0.beta, 1.0, places=13)
                self.assertEqual(fit.mu, offset + fit0.mu)
                # Between two anchored offsets the anchored differences are bit-identical, so the
                # scale and shape must be too -- no drift with the size of the offset at all.
                if reference is None:
                    reference = fit
                else:
                    self.assertEqual(fit.alpha, reference.alpha)
                    self.assertEqual(fit.beta, reference.beta)

    def test_large_offset_fit_no_longer_collapses_onto_the_shape_bound(self):
        # The pre-fix failure mode, stated in the units a user would notice: alpha two orders of
        # magnitude small and beta pinned to beta_bounds[0].
        base = _gg_sample()
        fit0 = optimize(list(base), estimator=GeneralizedGaussianDistribution(0.0, 1.0, 2.0))
        data = [LARGE_OFFSET + v for v in base]
        fit = optimize(data, estimator=GeneralizedGaussianDistribution(LARGE_OFFSET, 1.0, 2.0))
        self.assertGreater(fit.beta, 0.25 + 1.0e-6, "beta collapsed onto its lower bound")
        self.assertAlmostEqual(fit.alpha, fit0.alpha, places=10)
        # And it must not have declared its own training data implausible: the pre-fix fit lost 1468
        # nats of log-likelihood against the shift of the un-shifted fit.
        scores = fit.seq_log_density(np.asarray(data))
        self.assertTrue(np.all(np.isfinite(scores)))
        ideal = GeneralizedGaussianDistribution(fit0.mu + LARGE_OFFSET, fit0.alpha, fit0.beta)
        self.assertLess(abs(float(np.sum(scores)) - float(np.sum(ideal.seq_log_density(np.asarray(data))))), 1.0e-4)

    def test_unquantized_data_fit_matches_within_input_rounding(self):
        # Realistic (un-quantized) data: shifting to 1.7e9 rounds each observation by up to half an
        # ulp (1.2e-7) of the offset, so the fit can only agree to ~1e-7 relative. Before the fix the
        # disagreement was a factor of 118.
        base = list(GeneralizedGaussianDistribution(0.0, 1.0, 2.0).sampler(11).sample(2000))
        fit0 = optimize(list(base), estimator=GeneralizedGaussianDistribution(0.0, 1.0, 2.0))
        data = [LARGE_OFFSET + v for v in base]
        fit = optimize(data, estimator=GeneralizedGaussianDistribution(LARGE_OFFSET, 1.0, 2.0))
        self.assertAlmostEqual(fit.alpha / fit0.alpha, 1.0, places=6)
        self.assertAlmostEqual(fit.beta / fit0.beta, 1.0, places=6)

    def test_well_conditioned_data_keeps_the_historical_raw_estimate(self):
        # The conditioning gate must leave ordinary data alone: no anchor, and the estimate the raw
        # power sums alone produce.
        base = _gg_sample(n=500, seed=5)
        acc = GeneralizedGaussianAccumulator()
        acc.seq_update(np.asarray(base), np.ones(len(base)), None)
        stat = acc.value()
        self.assertIsNone(getattr(stat, "anchored", None))
        est = GeneralizedGaussianEstimator()
        raw = est.estimate(None, tuple(stat))
        fit = est.estimate(None, stat)
        self.assertEqual((fit.mu, fit.alpha, fit.beta), (raw.mu, raw.alpha, raw.beta))

    def test_dropping_the_anchored_payload_falls_back_instead_of_raising(self):
        # A consumer that knows nothing about the payload (an older serializer, scale_suff_stat, ...)
        # must still get an estimate -- the historical, less accurate one -- not an exception.
        data = [LARGE_OFFSET + v for v in _gg_sample(n=400, seed=2)]
        acc = GeneralizedGaussianAccumulator()
        acc.seq_update(np.asarray(data), np.ones(len(data)), None)
        stat = acc.value()
        self.assertIsNotNone(getattr(stat, "anchored", None))
        est = GeneralizedGaussianEstimator()
        degraded = est.estimate(None, tuple(stat))  # payload stripped
        self.assertTrue(math.isfinite(degraded.alpha) and degraded.alpha > 0.0)

    def test_inconsistent_payload_is_ignored_rather_than_trusted(self):
        data = [LARGE_OFFSET + v for v in _gg_sample(n=400, seed=2)]
        acc = GeneralizedGaussianAccumulator()
        acc.seq_update(np.asarray(data), np.ones(len(data)), None)
        stat = acc.value()
        est = GeneralizedGaussianEstimator()
        anchor, a1, a2, a3, a4 = stat.anchored
        stat.anchored = (anchor + 1.0e6, a1, a2, a3, a4)  # first moment no longer implies stat[1]
        self.assertEqual(
            est.estimate(None, stat).alpha,
            est.estimate(None, tuple(stat)).alpha,
        )

    def test_chunked_and_single_pass_accumulation_agree(self):
        data = np.asarray([LARGE_OFFSET + v for v in _gg_sample(n=600, seed=8)])
        w = np.ones(len(data))
        whole = GeneralizedGaussianAccumulator()
        whole.seq_update(data, w, None)
        parts = GeneralizedGaussianAccumulator()
        for lo in range(0, len(data), 137):
            side = GeneralizedGaussianAccumulator()
            side.seq_update(data[lo : lo + 137], w[lo : lo + 137], None)
            parts.combine(side.value())
        est = GeneralizedGaussianEstimator()
        a, b = est.estimate(None, whole.value()), est.estimate(None, parts.value())
        self.assertAlmostEqual(a.alpha / b.alpha, 1.0, places=10)
        self.assertAlmostEqual(a.beta / b.beta, 1.0, places=10)

    def test_ill_conditioned_raw_merge_degrades_to_the_historical_answer_loudly(self):
        # Raw power sums that have already lost their central moments cannot be rescued by a change
        # of reference point. Converting them onto an anchor anyway seeds the track with an error
        # larger than the spread, giving a pooled estimate WORSE than the historical raw one; the
        # accumulator must instead withhold the payload (so the estimate is exactly the historical
        # one) and say so.
        data = np.asarray([LARGE_OFFSET + v for v in _gg_sample(n=600, seed=15)])
        w = np.ones(len(data))
        left, right = GeneralizedGaussianAccumulator(), GeneralizedGaussianAccumulator()
        left.seq_update(data[:300], w[:300], None)
        right.seq_update(data[300:], w[300:], None)
        est = GeneralizedGaussianEstimator()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mixed = GeneralizedGaussianAccumulator().combine(tuple(left.value())).combine(right.value())
        messages = [str(x.message) for x in caught if issubclass(x.category, RuntimeWarning)]
        self.assertTrue(messages, "an unrecoverable raw merge must warn")
        self.assertIn("seq_update", messages[0])
        self.assertIsNone(getattr(mixed.value(), "anchored", None))
        historical = GeneralizedGaussianAccumulator().combine(tuple(left.value())).combine(tuple(right.value()))
        got, expected = est.estimate(None, mixed.value()), est.estimate(None, tuple(historical.value()))
        self.assertEqual((got.mu, got.alpha, got.beta), (expected.mu, expected.alpha, expected.beta))
        # And the mirror order -- raw statistics arriving at an already-anchored pool -- must behave
        # the same way; an anchored pool that swallows lost content is just as wrong.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            other = GeneralizedGaussianAccumulator().combine(left.value()).combine(tuple(right.value()))
        self.assertTrue([str(x.message) for x in caught if issubclass(x.category, RuntimeWarning)])
        self.assertIsNone(getattr(other.value(), "anchored", None))

    def test_well_conditioned_raw_merge_still_anchors_without_warning(self):
        # The fallback above must not fire for a raw seed the gate certifies: that content converts
        # onto the anchor accurately, so the pool keeps its shift-equivariance.
        seed_data = np.asarray(_gg_sample(n=300, seed=16))
        anchored_data = np.asarray([LARGE_OFFSET + v for v in _gg_sample(n=300, seed=17)])
        raw = GeneralizedGaussianAccumulator()
        raw.seq_update(seed_data, np.ones(300), None)
        far = GeneralizedGaussianAccumulator()
        far.seq_update(anchored_data, np.ones(300), None)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mixed = GeneralizedGaussianAccumulator().combine(tuple(raw.value())).combine(far.value())
        self.assertEqual([str(x.message) for x in caught if issubclass(x.category, RuntimeWarning)], [])
        self.assertIsNotNone(getattr(mixed.value(), "anchored", None))

    def test_scalar_update_path_is_shift_equivariant(self):
        base = _gg_sample(n=300, seed=4)
        est = GeneralizedGaussianEstimator()
        fits = []
        for offset in (0.0, LARGE_OFFSET):
            acc = GeneralizedGaussianAccumulator()
            for v in base:
                acc.update(offset + v, 1.0, None)
            fits.append(est.estimate(None, acc.value()))
        self.assertAlmostEqual(fits[0].alpha, fits[1].alpha, delta=1e-10 * fits[0].alpha)
        self.assertAlmostEqual(fits[0].beta, fits[1].beta, delta=1e-10 * fits[0].beta)

    def test_constant_data_at_a_large_offset_stays_degenerate(self):
        # The sub-noise clamp must send the scatter to EXACTLY zero, so the degenerate branch fires
        # deterministically instead of reading a rounding residue as spread.
        acc = GeneralizedGaussianAccumulator()
        acc.seq_update(np.full(200, LARGE_OFFSET), np.ones(200), None)
        fit = GeneralizedGaussianEstimator().estimate(None, acc.value())
        self.assertEqual(fit.alpha, 1.0e-6)
        self.assertEqual(fit.mu, LARGE_OFFSET)

    def test_suff_stat_payload_survives_pickle(self):
        data = [LARGE_OFFSET + v for v in _gg_sample(n=200, seed=6)]
        acc = GeneralizedGaussianAccumulator()
        acc.seq_update(np.asarray(data), np.ones(len(data)), None)
        stat = pickle.loads(pickle.dumps(acc.value()))
        self.assertEqual(tuple(stat), tuple(acc.value()))
        self.assertEqual(stat.anchored, acc.value().anchored)
        est = GeneralizedGaussianEstimator()
        self.assertEqual(est.estimate(None, stat).alpha, est.estimate(None, acc.value()).alpha)

    def test_scale_keeps_the_anchored_track(self):
        data = [LARGE_OFFSET + v for v in _gg_sample(n=200, seed=6)]
        acc = GeneralizedGaussianAccumulator()
        acc.seq_update(np.asarray(data), np.ones(len(data)), None)
        est = GeneralizedGaussianEstimator()
        before = est.estimate(None, acc.value())
        after = est.estimate(None, acc.scale(0.5).value())
        self.assertAlmostEqual(after.alpha / before.alpha, 1.0, places=12)

    def test_pseudo_count_prior_at_a_large_location_is_blended_exactly(self):
        # The prototype-built prior carries its central moments, so a prior sitting at 1.7e9 blends
        # into anchored data without losing its own spread to E[X^2]'s ulp: the whole blend is
        # shift-equivariant, prior and data together.
        base = _gg_sample(n=400, seed=9)
        fits = []
        for offset in (0.0, LARGE_OFFSET):
            proto = GeneralizedGaussianDistribution(offset + 0.25, 1.0347728, 2.2162343)
            acc = GeneralizedGaussianAccumulator()
            acc.seq_update(np.asarray([offset + v for v in base]), np.ones(len(base)), None)
            fits.append(proto.estimator(pseudo_count=50.0).estimate(None, acc.value()))
        # Not bit-identical, and cannot be: the prototype's own location at 1.7e9 is representable
        # only to an ulp of 3.8e-7. What matters is that the prior's SPREAD survives the blend --
        # pre-fix, ``E[X^2] = var + mu^2`` had erased it completely.
        self.assertAlmostEqual(fits[1].alpha / fits[0].alpha, 1.0, places=7)
        self.assertAlmostEqual(fits[1].beta / fits[0].beta, 1.0, places=7)
        self.assertGreater(fits[1].beta, 0.25 + 1.0e-6)

    def test_raw_prior_that_lost_its_spread_warns_with_a_remedy(self):
        # No central payload and a location that dominates the spread: the blend cannot be done
        # correctly, so it must be LOUD rather than silently wrong.
        est = GeneralizedGaussianEstimator(
            pseudo_count=10.0,
            suff_stat=(LARGE_OFFSET, LARGE_OFFSET**2 + 1.0, LARGE_OFFSET**3, LARGE_OFFSET**4),
        )
        acc = GeneralizedGaussianAccumulator()
        base = _gg_sample(n=200, seed=12)
        acc.seq_update(np.asarray([LARGE_OFFSET + v for v in base]), np.ones(len(base)), None)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = est.estimate(None, acc.value())
        messages = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
        self.assertTrue(messages, "an unrecoverable prior blend must warn")
        self.assertIn("prior_central", messages[0])
        self.assertIn("prior-moments-ill-conditioned", fit.numerical_repairs())


class GeneralizedExtremeValueShiftEquivarianceTest(unittest.TestCase):
    def test_explicit_prototype_fit_is_shift_equivariant_at_1e9(self):
        base = _gev_sample()
        fit0 = optimize(list(base), estimator=GeneralizedExtremeValueDistribution(0.0, 1.0, 0.0))
        reference = None
        for offset in ANCHORED_OFFSETS:
            with self.subTest(offset=offset):
                shifted = [offset + v for v in base]
                fit = optimize(shifted, estimator=GeneralizedExtremeValueDistribution(offset, 1.0, 0.0))
                # Against the un-shifted fit: that one takes the raw path, so the two evaluate
                # different but algebraically equal expressions and may differ in the last couple of
                # ulps. Before the fix scale differed by 58x and shape sat on its bound.
                self.assertAlmostEqual(fit.scale / fit0.scale, 1.0, places=13)
                self.assertAlmostEqual(fit.shape / fit0.shape, 1.0, places=13)
                self.assertEqual(fit.loc, offset + fit0.loc)
                # Between two anchored offsets the anchored differences are bit-identical, so the
                # scale and shape must be too -- no drift with the size of the offset at all.
                if reference is None:
                    reference = fit
                else:
                    self.assertEqual(fit.scale, reference.scale)
                    self.assertEqual(fit.shape, reference.shape)

    def test_large_offset_fit_no_longer_collapses_onto_the_shape_bound(self):
        base = _gev_sample()
        fit0 = optimize(list(base), estimator=GeneralizedExtremeValueDistribution(0.0, 1.0, 0.0))
        data = [LARGE_OFFSET + v for v in base]
        fit = optimize(data, estimator=GeneralizedExtremeValueDistribution(LARGE_OFFSET, 1.0, 0.0))
        self.assertGreater(fit.shape, -1.0 + 1.0e-6, "shape collapsed onto its lower bound")
        # sd of the base law is ~0.52-0.61; the pre-fix scale was 22.6.
        self.assertLess(fit.scale, 1.0)
        self.assertGreater(fit.scale, 0.1)
        scores = fit.seq_log_density(np.asarray(data))
        self.assertTrue(np.all(np.isfinite(scores)))
        # The pre-fix fit lost 13621 nats against the shift of the un-shifted fit.
        ideal = GeneralizedExtremeValueDistribution(fit0.loc + LARGE_OFFSET, fit0.scale, fit0.shape)
        self.assertLess(abs(float(np.sum(scores)) - float(np.sum(ideal.seq_log_density(np.asarray(data))))), 1.0e-4)

    def test_large_offset_fit_emits_no_overflow_warning(self):
        # The pre-fix fit's only signal was numpy's "overflow encountered in exp" leaking out of
        # seq_log_density; a correct fit is silent.
        data = [LARGE_OFFSET + v for v in _gev_sample()]
        with warnings.catch_warnings(record=True) as caught, np.errstate(over="warn"):
            warnings.simplefilter("always")
            fit = optimize(data, estimator=GeneralizedExtremeValueDistribution(LARGE_OFFSET, 1.0, 0.0))
            fit.seq_log_density(np.asarray(data))
        self.assertEqual([str(w.message) for w in caught if "overflow" in str(w.message)], [])

    def test_unquantized_data_fit_matches_within_input_rounding(self):
        base = list(np.exp(np.random.RandomState(3).normal(0.0, 0.5, size=4000)) - 1.0)
        fit0 = optimize(list(base), estimator=GeneralizedExtremeValueDistribution(0.0, 1.0, 0.0))
        data = [LARGE_OFFSET + v for v in base]
        fit = optimize(data, estimator=GeneralizedExtremeValueDistribution(LARGE_OFFSET, 1.0, 0.0))
        self.assertAlmostEqual(fit.scale / fit0.scale, 1.0, places=6)
        self.assertAlmostEqual(fit.shape, fit0.shape, places=6)

    def test_well_conditioned_data_keeps_the_historical_raw_estimate(self):
        base = _gev_sample(n=500, seed=5)
        acc = GeneralizedExtremeValueAccumulator()
        acc.seq_update(np.asarray(base), np.ones(len(base)), None)
        stat = acc.value()
        self.assertIsNone(getattr(stat, "anchored", None))
        est = GeneralizedExtremeValueEstimator()
        raw = est.estimate(None, tuple(stat))
        fit = est.estimate(None, stat)
        self.assertEqual((fit.loc, fit.scale, fit.shape), (raw.loc, raw.scale, raw.shape))

    def test_dropping_the_anchored_payload_falls_back_instead_of_raising(self):
        data = [LARGE_OFFSET + v for v in _gev_sample(n=400, seed=2)]
        acc = GeneralizedExtremeValueAccumulator()
        acc.seq_update(np.asarray(data), np.ones(len(data)), None)
        stat = acc.value()
        self.assertIsNotNone(getattr(stat, "anchored", None))
        degraded = GeneralizedExtremeValueEstimator().estimate(None, tuple(stat))
        self.assertTrue(math.isfinite(degraded.scale) and degraded.scale > 0.0)

    def test_inconsistent_payload_is_ignored_rather_than_trusted(self):
        data = [LARGE_OFFSET + v for v in _gev_sample(n=400, seed=2)]
        acc = GeneralizedExtremeValueAccumulator()
        acc.seq_update(np.asarray(data), np.ones(len(data)), None)
        stat = acc.value()
        est = GeneralizedExtremeValueEstimator()
        anchor, a1, a2, a3 = stat.anchored
        stat.anchored = (anchor + 1.0e6, a1, a2, a3)
        self.assertEqual(est.estimate(None, stat).scale, est.estimate(None, tuple(stat)).scale)

    def test_chunked_and_single_pass_accumulation_agree(self):
        data = np.asarray([LARGE_OFFSET + v for v in _gev_sample(n=900, seed=8)])
        w = np.ones(len(data))
        whole = GeneralizedExtremeValueAccumulator()
        whole.seq_update(data, w, None)
        parts = GeneralizedExtremeValueAccumulator()
        for lo in range(0, len(data), 211):
            side = GeneralizedExtremeValueAccumulator()
            side.seq_update(data[lo : lo + 211], w[lo : lo + 211], None)
            parts.combine(side.value())
        est = GeneralizedExtremeValueEstimator()
        a, b = est.estimate(None, whole.value()), est.estimate(None, parts.value())
        self.assertAlmostEqual(a.scale / b.scale, 1.0, places=10)
        self.assertAlmostEqual(a.shape, b.shape, places=10)

    def test_ill_conditioned_raw_merge_degrades_to_the_historical_answer_loudly(self):
        data = np.asarray([LARGE_OFFSET + v for v in _gev_sample(n=900, seed=15)])
        w = np.ones(len(data))
        left, right = GeneralizedExtremeValueAccumulator(), GeneralizedExtremeValueAccumulator()
        left.seq_update(data[:450], w[:450], None)
        right.seq_update(data[450:], w[450:], None)
        est = GeneralizedExtremeValueEstimator()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mixed = GeneralizedExtremeValueAccumulator().combine(tuple(left.value())).combine(right.value())
        messages = [str(x.message) for x in caught if issubclass(x.category, RuntimeWarning)]
        self.assertTrue(messages, "an unrecoverable raw merge must warn")
        self.assertIn("seq_update", messages[0])
        self.assertIsNone(getattr(mixed.value(), "anchored", None))
        historical = GeneralizedExtremeValueAccumulator().combine(tuple(left.value())).combine(tuple(right.value()))
        got, expected = est.estimate(None, mixed.value()), est.estimate(None, tuple(historical.value()))
        self.assertEqual((got.loc, got.scale, got.shape), (expected.loc, expected.scale, expected.shape))
        # And the mirror order -- raw statistics arriving at an already-anchored pool.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            other = GeneralizedExtremeValueAccumulator().combine(left.value()).combine(tuple(right.value()))
        self.assertTrue([str(x.message) for x in caught if issubclass(x.category, RuntimeWarning)])
        self.assertIsNone(getattr(other.value(), "anchored", None))

    def test_well_conditioned_raw_merge_still_anchors_without_warning(self):
        raw = GeneralizedExtremeValueAccumulator()
        raw.seq_update(np.asarray(_gev_sample(n=400, seed=16)), np.ones(400), None)
        far = GeneralizedExtremeValueAccumulator()
        far.seq_update(np.asarray([LARGE_OFFSET + v for v in _gev_sample(n=400, seed=17)]), np.ones(400), None)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mixed = GeneralizedExtremeValueAccumulator().combine(tuple(raw.value())).combine(far.value())
        self.assertEqual([str(x.message) for x in caught if issubclass(x.category, RuntimeWarning)], [])
        self.assertIsNotNone(getattr(mixed.value(), "anchored", None))

    def test_scalar_update_path_is_shift_equivariant(self):
        base = _gev_sample(n=400, seed=4)
        est = GeneralizedExtremeValueEstimator()
        fits = []
        for offset in (0.0, LARGE_OFFSET):
            acc = GeneralizedExtremeValueAccumulator()
            for v in base:
                acc.update(offset + v, 1.0, None)
            fits.append(est.estimate(None, acc.value()))
        self.assertAlmostEqual(fits[0].scale, fits[1].scale, delta=1e-10 * fits[0].scale)
        self.assertAlmostEqual(fits[0].shape, fits[1].shape, delta=1e-10)

    def test_constant_data_at_a_large_offset_stays_degenerate(self):
        acc = GeneralizedExtremeValueAccumulator()
        acc.seq_update(np.full(200, LARGE_OFFSET), np.ones(200), None)
        est = GeneralizedExtremeValueEstimator()
        fit = est.estimate(None, acc.value())
        self.assertEqual(fit.scale, est.min_scale)
        self.assertEqual(fit.shape, 0.0)
        self.assertEqual(fit.loc, LARGE_OFFSET)

    def test_suff_stat_payload_survives_pickle(self):
        data = [LARGE_OFFSET + v for v in _gev_sample(n=200, seed=6)]
        acc = GeneralizedExtremeValueAccumulator()
        acc.seq_update(np.asarray(data), np.ones(len(data)), None)
        stat = pickle.loads(pickle.dumps(acc.value()))
        self.assertEqual(tuple(stat), tuple(acc.value()))
        self.assertEqual(stat.anchored, acc.value().anchored)
        est = GeneralizedExtremeValueEstimator()
        self.assertEqual(est.estimate(None, stat).scale, est.estimate(None, acc.value()).scale)

    def test_scale_keeps_the_anchored_track(self):
        data = [LARGE_OFFSET + v for v in _gev_sample(n=300, seed=6)]
        acc = GeneralizedExtremeValueAccumulator()
        acc.seq_update(np.asarray(data), np.ones(len(data)), None)
        est = GeneralizedExtremeValueEstimator()
        before = est.estimate(None, acc.value())
        after = est.estimate(None, acc.scale(0.5).value())
        self.assertAlmostEqual(after.scale / before.scale, 1.0, places=12)

    def test_pseudo_count_prior_at_a_large_location_is_blended_exactly(self):
        base = _gev_sample(n=600, seed=9)
        fits = []
        for offset in (0.0, LARGE_OFFSET):
            proto = GeneralizedExtremeValueDistribution(offset - 0.16, 0.39, 0.12)
            acc = GeneralizedExtremeValueAccumulator()
            acc.seq_update(np.asarray([offset + v for v in base]), np.ones(len(base)), None)
            fits.append(proto.estimator(pseudo_count=50.0).estimate(None, acc.value()))
        # Not bit-identical, and cannot be: the prototype's own ``mean()`` at loc = 1.7e9 is
        # ``loc + scale*(g1-1)/xi`` rounded to an ulp of 3.8e-7, so the two prototypes differ by
        # 1.05e-7 in their location before the estimator sees them. What matters is that the prior's
        # SPREAD survives the blend -- pre-fix, ``E[X^2] = var + mean^2`` had erased it completely.
        self.assertAlmostEqual(fits[1].scale / fits[0].scale, 1.0, places=7)
        self.assertAlmostEqual(fits[1].shape / fits[0].shape, 1.0, places=7)
        self.assertGreater(fits[1].shape, -1.0 + 1.0e-6)

    def test_raw_prior_that_lost_its_spread_warns_with_a_remedy(self):
        est = GeneralizedExtremeValueEstimator(
            pseudo_count=10.0,
            suff_stat=(LARGE_OFFSET, LARGE_OFFSET**2 + 1.0, LARGE_OFFSET**3),
        )
        acc = GeneralizedExtremeValueAccumulator()
        base = _gev_sample(n=300, seed=12)
        acc.seq_update(np.asarray([LARGE_OFFSET + v for v in base]), np.ones(len(base)), None)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = est.estimate(None, acc.value())
        messages = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
        self.assertTrue(messages, "an unrecoverable prior blend must warn")
        self.assertIn("prior_central", messages[0])
        self.assertIn("prior-moments-ill-conditioned", fit.numerical_repairs())

    def test_small_location_prior_is_unaffected_by_the_anchored_path(self):
        # A prior that is itself well-conditioned must blend without warning, whether or not the data
        # needed the anchor -- the warning is about the prior's own conditioning, nothing else.
        est = GeneralizedExtremeValueEstimator(pseudo_count=1.0, suff_stat=(0.1, 1.0, 0.5))
        acc = GeneralizedExtremeValueAccumulator()
        base = _gev_sample(n=300, seed=13)
        acc.seq_update(np.asarray([LARGE_OFFSET + v for v in base]), np.ones(len(base)), None)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = est.estimate(None, acc.value())
        self.assertEqual([str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)], [])
        self.assertTrue(math.isfinite(fit.scale))


class LegacyEstimatorStateTest(unittest.TestCase):
    """An estimator unpickled from a release without ``prior_central`` must still estimate."""

    def _strip(self, estimator):
        clone = copy.copy(estimator)
        del clone.prior_central
        return clone

    def test_generalized_gaussian_estimator_without_prior_central(self):
        est = self._strip(GeneralizedGaussianDistribution(LARGE_OFFSET, 1.0, 2.0).estimator(pseudo_count=5.0))
        acc = GeneralizedGaussianAccumulator()
        acc.seq_update(np.asarray([LARGE_OFFSET + v for v in _gg_sample(n=200, seed=21)]), np.ones(200), None)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = est.estimate(None, acc.value())  # anchored ref, no central payload
        self.assertTrue(math.isfinite(fit.alpha) and fit.alpha > 0.0)

    def test_gev_estimator_without_prior_central(self):
        est = self._strip(GeneralizedExtremeValueDistribution(LARGE_OFFSET, 0.39, 0.12).estimator(pseudo_count=5.0))
        acc = GeneralizedExtremeValueAccumulator()
        acc.seq_update(np.asarray([LARGE_OFFSET + v for v in _gev_sample(n=300, seed=21)]), np.ones(300), None)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = est.estimate(None, acc.value())
        self.assertTrue(math.isfinite(fit.scale) and fit.scale > 0.0)


class MixtureEmShiftEquivarianceTest(unittest.TestCase):
    """The downstream shape of the defect: a full EM run over a two-component mixture.

    Pre-fix, at offset 1.7e9 the two-component generalized-Gaussian mixture came back with component
    scales ``1e-6`` (the degenerate floor) and ``0.0124`` against the correct ``0.451`` and ``1.016``,
    and the GEV mixture with ``22.6`` and ``45.3`` against ``0.531`` and ``1.032`` -- component
    locations scrambled along with them. Every EM iteration re-derives the same broken moments, so
    the run converges confidently to the wrong answer.
    """

    def _run(self, offset, proto_factory, sample, loc_of, scale_of):
        fitted = optimize([float(offset + v) for v in sample], estimator=proto_factory(offset), max_its=40)
        pairs = sorted((loc_of(c) - offset, scale_of(c)) for c in fitted.components)
        return pairs

    def test_generalized_gaussian_mixture_em_is_shift_equivariant(self):
        from mixle.stats import MixtureDistribution

        sample = _grid(
            np.concatenate(
                [
                    GeneralizedGaussianDistribution(0.0, 1.0, 2.0).sampler(1).sample(1500),
                    GeneralizedGaussianDistribution(6.0, 0.5, 2.0).sampler(2).sample(1500),
                ]
            )
        )

        def proto(offset):
            return MixtureDistribution(
                [
                    GeneralizedGaussianDistribution(offset, 1.0, 2.0),
                    GeneralizedGaussianDistribution(offset + 5.0, 1.0, 2.0),
                ],
                [0.5, 0.5],
            )

        loc, scale = (lambda c: c.mu), (lambda c: c.alpha)
        at0 = self._run(0.0, proto, sample, loc, scale)
        at_large = self._run(LARGE_OFFSET, proto, sample, loc, scale)
        for (loc0, scale0), (loc1, scale1) in zip(at0, at_large, strict=True):
            self.assertAlmostEqual(loc1, loc0, places=6)
            self.assertAlmostEqual(scale1 / scale0, 1.0, places=6)
        self.assertGreater(min(s for _, s in at_large), 1.0e-3, "a component collapsed to the scale floor")

    def test_gev_mixture_em_is_shift_equivariant(self):
        from mixle.stats import MixtureDistribution

        sample = _grid(
            np.concatenate(
                [
                    GeneralizedExtremeValueDistribution(0.0, 1.0, 0.1).sampler(1).sample(1500),
                    GeneralizedExtremeValueDistribution(6.0, 0.5, 0.1).sampler(2).sample(1500),
                ]
            )
        )

        def proto(offset):
            return MixtureDistribution(
                [
                    GeneralizedExtremeValueDistribution(offset, 1.0, 0.1),
                    GeneralizedExtremeValueDistribution(offset + 5.0, 1.0, 0.1),
                ],
                [0.5, 0.5],
            )

        loc, scale = (lambda c: c.loc), (lambda c: c.scale)
        at0 = self._run(0.0, proto, sample, loc, scale)
        at_large = self._run(LARGE_OFFSET, proto, sample, loc, scale)
        for (loc0, scale0), (loc1, scale1) in zip(at0, at_large, strict=True):
            self.assertAlmostEqual(loc1, loc0, places=6)
            self.assertAlmostEqual(scale1 / scale0, 1.0, places=6)
        self.assertLess(max(s for _, s in at_large), 5.0, "a component scale blew up")


class EngineParityTest(unittest.TestCase):
    """The anchored track must not desynchronize the compute-engine scoring paths."""

    def test_torch_backend_scoring_matches_the_shift_equivariant_gev_fit(self):
        pytest = __import__("pytest")
        torch = pytest.importorskip("torch")
        from mixle.engines import TorchEngine

        data = np.asarray([LARGE_OFFSET + v for v in _gev_sample(n=500, seed=14)])
        fit = optimize(list(data), estimator=GeneralizedExtremeValueDistribution(LARGE_OFFSET, 1.0, 0.0))
        engine = TorchEngine(dtype=torch.float64)
        got = fit.backend_seq_log_density(data, engine).detach().cpu().numpy()
        np.testing.assert_allclose(got, fit.seq_log_density(data), rtol=1e-9, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
