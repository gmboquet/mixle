"""Campaign 3b: the univariate Gaussian sub-noise clamp at extreme magnitude.

The shift-anchored M-step clamps a sub-noise scatter to exactly zero so that algebraically
equivalent accumulation paths agree on degenerate components. The clamp was applied to the WHOLE
scatter against ``count * (4 eps |mean|)**2``, which at mean 1e15 is ~0.79 per observation -- so
genuine data with sd 0.5 there (four float64 grid steps of real spread) was declared constant and
handed to the variance floor, silently reporting sigma2 = 1e22 for a sample whose variance is 0.245.

These tests pin:

  * recovery of real spread right up to the representational limit (the magnitude at which the
    observations genuinely round to a single float), measured against exact rational arithmetic;
  * that the limit itself is still respected -- unrepresentable spread reports the floor and says so;
  * that the invariants the clamp exists for survive: degenerate components collapse to EXACTLY zero
    on every equivalent path, and the fit stays shift-invariant;
  * that a clamp binding on apparent spread the magnitude COULD have represented is disclosed
    through ``numerical_repairs()`` instead of being reported as a bare zero variance;
  * that a prior-driven mean shift still contributes scatter (the ulp clamp is on mean ROUNDING).
"""

import unittest
from fractions import Fraction

import numpy as np

import mixle
from mixle.stats.latent.mixture import MixtureEstimator
from mixle.stats.univariate.continuous.gaussian import (
    GaussianAccumulator,
    GaussianEstimator,
    GaussianSuffStat,
)


def _grid_normal(seed: int, n: int, sd: float) -> np.ndarray:
    """Normal draws on a 2^-16 grid, so adding a power-of-two offset is exact in float64."""
    rng = np.random.RandomState(seed)
    return np.round(rng.normal(0.0, sd, size=n) * 65536.0) / 65536.0


def _exact_variance(data) -> float:
    """Population variance of the float64 values actually stored, in exact rational arithmetic.

    ``np.var`` is not a usable reference at these magnitudes: it centers on the float64 mean, so it
    carries ``(mean rounding)**2`` of its own -- 0.2% at 1e15 -- and would hide the very error the
    anchored track exists to remove.
    """
    values = [Fraction(float(x)) for x in data]
    n = Fraction(len(values))
    mean = sum(values) / n
    return float(sum((v - mean) ** 2 for v in values) / n)


def _fit(data) -> "mixle.stats.univariate.continuous.gaussian.GaussianDistribution":
    acc = GaussianAccumulator()
    acc.seq_update(np.asarray(data, dtype=float), np.ones(len(data)), None)
    return GaussianEstimator().estimate(None, acc.value())


class ExtremeMagnitudeSpreadTestCase(unittest.TestCase):
    """Genuine spread must survive the sub-noise clamp wherever float64 can carry it."""

    def test_sd_half_at_1e15_is_data_not_noise(self):
        # The headline case. Pre-fix: sigma2 = 1e+22 with numerical_repairs()
        # ('variance-floored(0 -> 1e+22)',) -- the fit claimed the data implied zero variance.
        data = _grid_normal(7, 400, 0.5) + 1.0e15
        exact = _exact_variance(data)
        self.assertGreater(exact, 0.24)  # the spread really is there in the stored floats
        fitted = _fit(data)
        self.assertLessEqual(abs(fitted.sigma2 - exact) / exact, 1.0e-12)
        self.assertEqual(fitted.numerical_repairs(), ())

    def test_spread_recovered_up_to_the_representational_limit(self):
        # Every (sd, magnitude) whose stored data still carries a nonzero spread must come back to
        # near machine precision -- not merely "not floored".
        for sd in (1.0, 0.5, 0.1):
            for magnitude in (1.0e13, 1.0e14, 1.0e15, 3.0e15):
                with self.subTest(sd=sd, magnitude=magnitude):
                    data = _grid_normal(7, 400, sd) + magnitude
                    exact = _exact_variance(data)
                    if exact == 0.0:
                        continue  # genuinely unrepresentable at this magnitude; covered below
                    fitted = _fit(data)
                    self.assertLessEqual(abs(fitted.sigma2 - exact) / exact, 1.0e-12)
                    self.assertEqual(fitted.numerical_repairs(), ())

    def test_unrepresentable_spread_reports_the_floor_and_discloses_it(self):
        # The limit is real: at 1e17 a sd-1.0 sample rounds to a single float (ulp is 16), so the
        # stored data ARE constant. The fit must not invent a spread -- it floors, and says so.
        data = _grid_normal(7, 400, 1.0) + 1.0e17
        self.assertEqual(len(np.unique(data)), 1)
        self.assertEqual(_exact_variance(data), 0.0)
        fitted = _fit(data)
        self.assertGreater(fitted.sigma2, 0.0)
        self.assertTrue(any(r.startswith("variance-floored") for r in fitted.numerical_repairs()))

    def test_mixture_em_recovers_components_at_extreme_magnitude(self):
        # Reachable by ordinary use: a plain two-component fit of offset data, no opt-in anywhere.
        rng = np.random.RandomState(21)
        base = np.round(np.concatenate([rng.normal(0.0, 0.5, 300), rng.normal(40.0, 0.5, 300)]) * 65536.0) / 65536.0
        model = mixle.inference.fit(
            (base + 1.0e15).tolist(),
            MixtureEstimator(estimators=[GaussianEstimator() for _ in range(2)]),
            rng=np.random.RandomState(3),
            max_its=50,
        )
        for component in model.components:
            self.assertLess(component.sigma2, 1.0)  # pre-fix: 1e+22, the scale-relative floor
            self.assertGreater(component.sigma2, 0.1)


class ClampInvariantsTestCase(unittest.TestCase):
    """What the clamp exists for must still hold."""

    def test_constant_data_scatter_is_exactly_zero_on_equivalent_paths(self):
        # The accumulator/reweighted-seq_update parity invariant, at the magnitude where the mean's
        # own rounding is the only nonzero term. "Close" is not enough: the scale-relative variance
        # floor reads any positive residue as a genuine spread, and the two fits then disagree.
        data = np.full(200, 1.0e15)
        weights = np.linspace(0.5, 1.5, 200)
        c = 0.37
        scaled = GaussianAccumulator()
        scaled.seq_update(data, weights, None)
        scaled.scale(c)
        reweighted = GaussianAccumulator()
        reweighted.seq_update(data, weights * c, None)
        left = GaussianEstimator().estimate(None, scaled.value())
        right = GaussianEstimator().estimate(None, reweighted.value())
        # The observed scatter must be zero on BOTH paths, not merely close: the scale-relative
        # variance floor reads any positive residue as a genuine spread. What remains is the floor
        # itself, which tracks each path's own last-ulp mean.
        self.assertTrue(all(r.startswith("variance-floored(0 ->") for r in left.numerical_repairs()))
        self.assertTrue(all(r.startswith("variance-floored(0 ->") for r in right.numerical_repairs()))
        self.assertLessEqual(abs(left.sigma2 - right.sigma2) / right.sigma2, 1.0e-10)
        self.assertLessEqual(abs(left.mu - right.mu) / abs(right.mu), 1.0e-15)

    def test_constant_data_combine_matches_single_pass(self):
        data = np.full(200, 1.0e15)
        single = GaussianAccumulator()
        single.seq_update(data, np.ones(200), None)
        halves = GaussianAccumulator()
        halves.seq_update(data[:100], np.ones(100), None)
        other = GaussianAccumulator()
        other.seq_update(data[100:], np.ones(100), None)
        halves.combine(other.value())
        self.assertEqual(
            GaussianEstimator().estimate(None, single.value()).sigma2,
            GaussianEstimator().estimate(None, halves.value()).sigma2,
        )

    def test_variance_is_shift_invariant_at_extreme_exact_offsets(self):
        # The sample must be quantized coarsely enough that adding 2**k is EXACT in float64 -- a
        # value carrying bits from 2**k down to 2**(k-52) already fills the mantissa -- otherwise
        # the shifted data is a different sample and the comparison measures quantization, not the
        # estimator.
        for exponent in (40, 45, 50):  # offsets up to ~1.1e15
            step = 2.0 ** (exponent - 52)
            base = np.round(_grid_normal(7, 400, 0.5) / step) * step
            offset = 2.0**exponent
            shifted_data = base + offset
            with self.subTest(offset=offset):
                self.assertTrue(np.all((shifted_data - offset) == base))  # exactly representable
                reference = _fit(base)
                shifted = _fit(shifted_data)
                self.assertLessEqual(
                    abs(shifted.sigma2 - reference.sigma2) / reference.sigma2,
                    1.0e-12,
                    "sigma2 not shift-invariant at offset %g" % offset,
                )

    def test_scale_keeps_the_anchored_track(self):
        # scale() round-trips through value()/from_value(), and the structural scale_suff_stat
        # rebuilds the payload as a plain tuple -- which used to drop the anchored moments and send
        # the scaled accumulator back through the cancellation-prone raw form (fitted 1e+22 for a
        # sample whose variance is 0.246). Reachable from HMM/LDA child accumulators and streaming
        # EM, which scale every batch.
        data = _grid_normal(7, 200, 0.5) + 1.0e15
        acc = GaussianAccumulator()
        acc.seq_update(data, np.ones(len(data)), None)
        acc.scale(0.37)
        self.assertIsNotNone(getattr(acc.value(), "anchored", None))
        scaled_fit = GaussianEstimator().estimate(None, acc.value())
        reweighted = GaussianAccumulator()
        reweighted.seq_update(data, np.full(len(data), 0.37), None)
        reweighted_fit = GaussianEstimator().estimate(None, reweighted.value())
        exact = _exact_variance(data)
        self.assertLessEqual(abs(scaled_fit.sigma2 - exact) / exact, 1.0e-12)
        self.assertLessEqual(abs(scaled_fit.sigma2 - reweighted_fit.sigma2) / reweighted_fit.sigma2, 1.0e-12)

    def test_well_conditioned_fit_still_takes_the_raw_path_unchanged(self):
        # The conditioning gate must be untouched: ordinary data accumulates raw-only, with no
        # anchored payload for the split scatter to read.
        base = _grid_normal(3, 250, 0.81) + 13.0
        acc = GaussianAccumulator()
        acc.seq_update(base, np.ones(len(base)), None)
        self.assertIsNone(getattr(acc.value(), "anchored", None))
        fitted = GaussianEstimator().estimate(None, acc.value())
        n = float(len(base))
        sum_x = float(np.dot(base, np.ones(len(base))))
        sum_xx = float(np.dot(base * base, np.ones(len(base))))
        mu = sum_x / n
        self.assertEqual(fitted.sigma2, (sum_xx - 2.0 * mu * sum_x + n * mu * mu) / n)


class ClampDisclosureTestCase(unittest.TestCase):
    """A clamp binding on resolvable apparent spread has to be reported, not reported as zero."""

    @staticmethod
    def _payload(anchor: float, a_sum: float, a_sum2: float, count: float) -> GaussianSuffStat:
        """An anchored payload whose anchor sits far from its data (a restored/merged pool)."""
        sum_x = a_sum + count * anchor
        stat = GaussianSuffStat(sum_x, sum_x * sum_x / count, count, count)
        stat.anchored = (anchor, a_sum, a_sum2)
        return stat

    def test_resolvable_spread_lost_to_cancellation_is_disclosed(self):
        # anchor 1e7 below the data at 1e15: differencing 1e16-scale quantities cannot resolve a
        # scatter of 8.0, so the fit reports zero -- but sd 0.28 IS representable at 1e15 (ulp
        # 0.125), so the caller is told the zero is a repair rather than a measurement.
        count = 100.0
        a_sum = 1.0e9  # data sit 1e7 above the anchor
        core = 8.0
        stat = self._payload(1.0e15, a_sum, a_sum * a_sum / count + core, count)
        fitted = GaussianEstimator().estimate(None, stat)
        repairs = fitted.numerical_repairs()
        self.assertTrue(any(r.startswith("spread-below-noise") for r in repairs), repairs)
        self.assertTrue(any(r.startswith("variance-floored") for r in repairs), repairs)
        # Both facts are kept: the note does not overwrite the floor note or vice versa.
        self.assertEqual(len(repairs), 2)

    def test_sub_grid_residue_is_not_disclosed(self):
        # Same shape, but the apparent spread is far below one grid step at this magnitude: that is
        # the arithmetic the clamp exists to absorb, and flagging it would put a platform-dependent
        # note on ordinary degenerate components.
        count = 100.0
        a_sum = 1.0e9
        core = 1.0e-6  # per-observation variance 1e-8, vs a half-grid-step of 0.0625 at 1e15
        stat = self._payload(1.0e15, a_sum, a_sum * a_sum / count + core, count)
        fitted = GaussianEstimator().estimate(None, stat)
        repairs = fitted.numerical_repairs()
        self.assertFalse(any(r.startswith("spread-below-noise") for r in repairs), repairs)

    def test_ordinary_degenerate_component_carries_only_the_floor_note(self):
        fitted = _fit(np.full(200, 1.0e15))
        self.assertEqual(fitted.numerical_repairs(), ("variance-floored(0 -> 1e+22)",))


class PriorShiftTestCase(unittest.TestCase):
    """The ulp clamp is on mean ROUNDING; a prior that really moves the mean must still count."""

    def test_prior_mean_shift_contributes_scatter_at_extreme_magnitude(self):
        data = _grid_normal(7, 400, 0.5) + 1.0e15
        count = float(len(data))
        acc = GaussianAccumulator()
        acc.seq_update(data, np.ones(len(data)), None)
        stat = acc.value()
        sample_mean = float(sum(Fraction(float(x)) for x in data) / Fraction(len(data)))
        shift = 1000.0
        estimator = GaussianEstimator(
            pseudo_count=(count, None),  # a mean pseudo-count only: the variance stays maximum-likelihood
            suff_stat=(sample_mean + 2.0 * shift, None),
        )
        fitted = estimator.estimate(None, stat)
        # mu is pulled halfway to the prior mean, so the scatter about it grows by shift**2. A shift
        # of 1000 at magnitude 1e15 is ~8000 grid steps: real displacement, not mean rounding.
        self.assertLessEqual(abs(fitted.mu - (sample_mean + shift)), 1.0)
        expected = _exact_variance(data) + shift * shift
        # 1e-3, not 1e-12: the residual is the reported mean's OWN resolution at 1e15 (a grid step
        # of 0.125 on a displacement of 1000 is 1.25e-4, doubled by the squaring). The point is that
        # the displacement survives at all -- clamping it would give back the bare 0.2449.
        self.assertLessEqual(abs(fitted.sigma2 - expected) / expected, 1.0e-3)
        self.assertGreater(fitted.sigma2, 0.9e6)


if __name__ == "__main__":
    unittest.main()
