"""Regression tests for the logistic-manifest shift-instability repair (campaign 3c).

``LogisticEstimator`` had the SAME shift-instability class as the Gaussian/GGD/GEV families fixed
elsewhere in this release wave: the M-step variance is the classic ``E[x^2] - E[x]^2`` reduced-moment
form, which loses ~2*log2(|mean|/sd) bits of the true spread as ``|mean|`` grows. Concretely (see
``test_variance_matches_expected_at_1_7e9_offset``), fitting ``LogisticDistribution(0, 1)``-shaped
data at offset 0 gave scale ~0.99; the SAME data shifted to offset 1.7e9 gave a scale off by an order
of magnitude, silently, with an empty ``numerical_repairs()``.

The fix mirrors ``gaussian.py``'s ``_anchored_pooled_variance``: ``LogisticAccumulator`` keeps a
CONDITIONING-GATED shift-anchored moment track alongside the raw ``(sum, sum2, count)``, and
``LogisticEstimator.estimate`` splits the scatter into (1) the scatter about the sample's own mean
(all O(count * spread^2), carries the data) and (2) the displacement of the reported mean from the
sample mean (the only place large magnitude enters). Ordinary well-conditioned data never activates
the track and stays on the exact historical code path.
"""

import pickle
import unittest

import numpy as np

import mixle
from mixle.stats.univariate.continuous.logistic import (
    LogisticAccumulator,
    LogisticDistribution,
    LogisticEstimator,
    LogisticSuffStat,
)


def _grid_logistic(seed: int, n: int, loc: float = 0.0, scale: float = 1.0) -> np.ndarray:
    """Logistic draws rounded to a 2^-16 grid, so adding a power-of-two offset is EXACT in float64."""
    rng = np.random.RandomState(seed)
    return np.round(rng.logistic(loc, scale, size=n) * 65536.0) / 65536.0


class ShiftInvariantLogisticVarianceTestCase(unittest.TestCase):
    """The fitted scale must not depend on a constant offset of the data."""

    def test_variance_shift_invariant_up_to_1e9_exact_shift(self):
        base = _grid_logistic(7, 400)
        reference = mixle.inference.fit(base.tolist(), LogisticEstimator())
        for c in (2.0**10, 2.0**20, 2.0**30):  # up to ~1.07e9
            shifted = mixle.inference.fit((base + c).tolist(), LogisticEstimator())
            self.assertLessEqual(
                abs(shifted.scale - reference.scale) / reference.scale,
                1.0e-9,
                "scale not shift-invariant at offset %g" % c,
            )
            self.assertAlmostEqual(shifted.loc - c, reference.loc, places=6)

    def test_variance_matches_expected_at_1_7e9_offset(self):
        # This is the reported finding: LogisticDistribution(0, 1)-shaped data at offset 0 fit a
        # scale of ~0.99; the SAME spread shifted to 1.7e9 fit a scale wrong by more than an order
        # of magnitude before this fix (and, on more extreme draws, collapsed to the scale floor).
        # FAILS before the fix, PASSES after.
        base = _grid_logistic(11, 300, loc=13.0, scale=0.81)
        reference_scale = mixle.inference.fit(base.tolist(), LogisticEstimator()).scale
        for c in (1.0e7, 1.0e9, 1.7e9):
            data = (base + c).tolist()
            fitted = mixle.inference.fit(data, LogisticEstimator())
            self.assertLessEqual(
                abs(fitted.scale - reference_scale) / reference_scale,
                1.0e-9,
                "scale not shift-invariant at offset %g (got %.6f, expected ~%.6f)"
                % (c, fitted.scale, reference_scale),
            )
            self.assertEqual(fitted.numerical_repairs(), ())

    def test_well_conditioned_fit_bit_identical_to_historical_path(self):
        # The conditioning gate keeps ordinary data on the exact historical single-pass
        # accumulation: raw reduced moments, no anchored payload, bit-identical estimate.
        base = _grid_logistic(3, 250, loc=13.0, scale=0.81)
        acc = LogisticAccumulator()
        acc.seq_update(base, np.ones(len(base)), None)
        self.assertIsInstance(acc.value(), tuple)
        self.assertIsNone(getattr(acc.value(), "anchored", None))
        fitted = LogisticEstimator().estimate(None, acc.value())

        sum_x = float(np.dot(base, np.ones(len(base))))
        sum_x2 = float(np.dot(base * base, np.ones(len(base))))
        n = float(len(base))
        loc = sum_x / n
        legacy_var = max(sum_x2 / n - loc * loc, 0.0)
        legacy_scale = (3.0 * legacy_var / (np.pi**2)) ** 0.5

        self.assertEqual(fitted.loc, loc)
        self.assertEqual(fitted.scale, legacy_scale)

    def test_combine_across_anchors_matches_expected_variance(self):
        # Chan-style parallel merge of two accumulators anchored at different values.
        base = _grid_logistic(5, 200)
        x1 = base[:100] + 1.7e9
        x2 = base[100:] + 1.7e9 + 5.0
        a1, a2 = LogisticAccumulator(), LogisticAccumulator()
        a1.seq_update(x1, np.ones(100), None)
        a2.seq_update(x2, np.ones(100), None)
        a1.combine(a2.value())
        fitted = LogisticEstimator().estimate(None, a1.value())

        pooled = np.concatenate([x1, x2])
        loc = float(np.mean(pooled))
        expected_scale = (3.0 * float(np.var(pooled)) / (np.pi**2)) ** 0.5
        self.assertLessEqual(abs(fitted.loc - loc) / max(abs(loc), 1.0), 1.0e-12)
        self.assertLessEqual(abs(fitted.scale - expected_scale) / expected_scale, 1.0e-9)

    def test_value_pickle_and_from_value_round_trip_keep_the_anchored_track(self):
        base = _grid_logistic(9, 150) + 1.7e9
        acc = LogisticAccumulator()
        acc.seq_update(base, np.ones(150), None)
        value = acc.value()
        self.assertIsNotNone(getattr(value, "anchored", None))
        revived = pickle.loads(pickle.dumps(value))
        self.assertEqual(tuple(revived), tuple(value))
        self.assertEqual(revived.anchored, value.anchored)
        restored = LogisticAccumulator().from_value(revived)
        direct = LogisticEstimator().estimate(None, acc.value())
        via_round_trip = LogisticEstimator().estimate(None, restored.value())
        self.assertEqual(via_round_trip.scale, direct.scale)
        self.assertEqual(via_round_trip.loc, direct.loc)

    def test_scalar_update_path_is_shift_stable(self):
        base = _grid_logistic(13, 120)
        results = []
        for c in (0.0, 1.7e9):
            acc = LogisticAccumulator()
            for value in base + c:
                acc.update(float(value), 1.0, None)
            fitted = LogisticEstimator().estimate(None, acc.value())
            results.append(fitted.scale)
        self.assertLessEqual(abs(results[1] - results[0]) / results[0], 1.0e-9)

    def test_degenerate_data_clamps_to_exactly_the_scale_floor(self):
        # Truly constant data must clamp to exactly the min_scale floor on every path -- offset,
        # seq_update vs scalar update -- never a positive-but-tiny cancellation residue.
        floor = LogisticEstimator().min_scale
        for c in (0.0, 1.7e9):
            seq_fitted = mixle.inference.fit((np.full(50, 3.0) + c).tolist(), LogisticEstimator())
            self.assertEqual(seq_fitted.scale, floor)
            self.assertEqual(seq_fitted.loc, 3.0 + c)

            acc = LogisticAccumulator()
            for _ in range(50):
                acc.update(3.0 + c, 1.0, None)
            scalar_fitted = LogisticEstimator().estimate(None, acc.value())
            self.assertEqual(scalar_fitted.scale, floor)

    def test_pseudo_count_prior_blend_shift_stable(self):
        base = _grid_logistic(17, 200, loc=0.0, scale=1.2)
        results = []
        for c in (0.0, 2.0**30):
            prior = LogisticDistribution(0.5 + c, 0.9)
            est = LogisticEstimator(pseudo_count=5.0, suff_stat=(prior.loc, prior.scale))
            model = mixle.inference.fit((base + c).tolist(), est)
            results.append((model.loc - c, model.scale))
        self.assertLessEqual(abs(results[1][0] - results[0][0]), 1.0e-6)
        self.assertLessEqual(abs(results[1][1] - results[0][1]) / results[0][1], 1.0e-9)

    def test_inconsistent_hand_built_payload_falls_back_to_the_tuple(self):
        # A payload contradicting its own tuple must not silently change the estimate.
        bogus = LogisticSuffStat(10.0, 30.0, 5.0)
        bogus.anchored = (100.0, 50.0, 25.0)
        from_bogus = LogisticEstimator().estimate(None, bogus)
        from_plain = LogisticEstimator().estimate(None, (10.0, 30.0, 5.0))
        self.assertEqual(from_bogus.loc, from_plain.loc)
        self.assertEqual(from_bogus.scale, from_plain.scale)

    def test_accumulator_scale_matches_reweighted_seq_update_at_large_offset(self):
        # The accumulator/reweighted-seq_update parity invariant (see compute_metadata_test.py),
        # exercised specifically on data that forces the anchored track live.
        x = np.asarray([-1.0, 0.0, 2.0]) + 1.7e9
        weights = np.linspace(0.5, 1.5, 3)
        c = 0.37
        est = LogisticEstimator()
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
        # ``places=9`` on a value of 1.7e9 was a ONE-ulp check (the grid step there is 2.4e-7): the two
        # paths form ``(sum w dx) * c`` and ``sum (w c) dx`` at spread scale and round each onto the
        # anchor independently, so their locations may legitimately land one grid step apart. It
        # passed on x86 OpenBLAS and failed by exactly one step on the arm64 kernel. Parity here means
        # agreement to within a few grid steps of the magnitude, and exactness on the scale.
        self.assertAlmostEqual(scaled_model.loc, expected_model.loc, delta=4.0 * np.spacing(abs(expected_model.loc)))
        self.assertAlmostEqual(scaled_model.scale, expected_model.scale, places=9)


if __name__ == "__main__":
    unittest.main()
