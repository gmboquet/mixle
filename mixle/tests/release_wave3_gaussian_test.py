"""Regression tests for the 0.8.0 release-wave Gaussian/estimation fixes.

Covers, against the public ``mixle.inference`` entry points and the scalar Gaussian family:

  * B2  -- the M-step variance is shift-invariant (the naive ``E[x^2]-E[x]^2`` reduced-moment form
           lost to catastrophic cancellation: sd-0.81 data at offset 1.7e9 collapsed the variance
           to the 1e-8 floor), including the accumulator combine/from_value/pickle contract and the
           conditioning gate that keeps well-conditioned fits bit-identical to the historical path;
  * B1  -- numpy masked arrays with a nontrivial mask are rejected with a remedial error instead of
           silently fitting the fill values; trivial (all-False) masks still fit;
  * B3  -- USER-SUPPLIED all-zero observation weights are rejected at the entry point, while zero
           component weight inside EM (dead mixture components) keeps its no-evidence default;
  * t1  -- ``np.random.default_rng`` Generators are accepted by the fit verbs (no internal
           AttributeError), and empty-input errors name the called entry point and say
           "no observations".
"""

import pickle
import unittest

import numpy as np

import mixle
from mixle.stats.combinator.weighted import WeightedEstimator, WeightedObservation
from mixle.stats.latent.mixture import MixtureEstimator
from mixle.stats.univariate.continuous.gaussian import (
    GaussianAccumulator,
    GaussianEstimator,
    GaussianSuffStat,
)


def _grid_normal(seed: int, n: int, mean: float = 0.0, sd: float = 1.0) -> np.ndarray:
    """Normal draws rounded to a 2^-16 grid, so adding a power-of-two offset is EXACT in float64."""
    rng = np.random.RandomState(seed)
    return np.round(rng.normal(mean, sd, size=n) * 65536.0) / 65536.0


class ShiftInvariantVarianceTestCase(unittest.TestCase):
    """B2: fitted variance must not depend on a constant offset of the data."""

    def test_variance_shift_invariant_up_to_1e9_exact_shift(self):
        # Offsets are powers of two and the data sit on a 2^-16 grid, so the shifted data are
        # exactly representable: fit(x) and fit(x + c) describe literally the same spread.
        base = _grid_normal(7, 400)
        reference = mixle.inference.fit(base.tolist(), GaussianEstimator())
        for c in (2.0**10, 2.0**20, 2.0**30):  # up to ~1.07e9
            shifted = mixle.inference.fit((base + c).tolist(), GaussianEstimator())
            self.assertLessEqual(
                abs(shifted.sigma2 - reference.sigma2) / reference.sigma2,
                1.0e-9,
                "sigma2 not shift-invariant at offset %g" % c,
            )
            self.assertAlmostEqual(shifted.mu - c, reference.mu, places=6)

    def test_variance_matches_numpy_at_extreme_offsets(self):
        # Non-representable offsets: the estimate must match np.var of the data actually stored
        # (pre-fix: rel err 6.3% at 1e7, 280x at 1e9, floor collapse at 1.7e9).
        base = _grid_normal(11, 300, mean=13.0, sd=0.81)
        for c in (1.0e7, 1.0e9, 1.7e9):
            data = (base + c).tolist()
            fitted = mixle.inference.fit(data, GaussianEstimator())
            expected = float(np.var(data))
            self.assertLessEqual(abs(fitted.sigma2 - expected) / expected, 1.0e-12)
            self.assertEqual(fitted.numerical_repairs(), ())

    def test_well_conditioned_fit_bit_identical_to_historical_path(self):
        # The conditioning gate keeps ordinary data on the exact historical single-pass
        # accumulation: raw reduced moments, no anchored payload, bit-identical estimate.
        base = _grid_normal(3, 250, mean=13.0, sd=0.81)
        acc = GaussianAccumulator()
        acc.seq_update(base, np.ones(len(base)), None)
        self.assertIsInstance(acc.value(), tuple)
        self.assertIsNone(getattr(acc.value(), "anchored", None))
        fitted = GaussianEstimator().estimate(None, acc.value())
        sum_x = float(np.dot(base, np.ones(len(base))))
        sum_xx = float(np.dot(base * base, np.ones(len(base))))
        n = float(len(base))
        mu = sum_x / n
        legacy_sigma2 = (sum_xx - 2.0 * mu * sum_x + n * mu * mu) / n
        self.assertEqual(fitted.mu, mu)
        self.assertEqual(fitted.sigma2, legacy_sigma2)

    def test_combine_across_anchors_matches_pooled_variance(self):
        # Chan-style parallel merge of two accumulators anchored at different values.
        base = _grid_normal(5, 200)
        x1 = base[:100] + 1.7e9
        x2 = base[100:] + 1.7e9 + 5.0
        a1, a2 = GaussianAccumulator(), GaussianAccumulator()
        a1.seq_update(x1, np.ones(100), None)
        a2.seq_update(x2, np.ones(100), None)
        a1.combine(a2.value())
        fitted = GaussianEstimator().estimate(None, a1.value())
        pooled = np.concatenate([x1, x2])
        expected = float(np.var(pooled))
        self.assertLessEqual(abs(fitted.sigma2 - expected) / expected, 1.0e-12)

    def test_value_pickle_and_from_value_round_trip_keep_the_anchored_track(self):
        base = _grid_normal(9, 150) + 1.7e9
        acc = GaussianAccumulator()
        acc.seq_update(base, np.ones(150), None)
        value = acc.value()
        self.assertIsNotNone(getattr(value, "anchored", None))
        revived = pickle.loads(pickle.dumps(value))
        self.assertEqual(tuple(revived), tuple(value))
        self.assertEqual(revived.anchored, value.anchored)
        restored = GaussianAccumulator().from_value(revived)
        direct = GaussianEstimator().estimate(None, acc.value())
        via_round_trip = GaussianEstimator().estimate(None, restored.value())
        self.assertEqual(via_round_trip.sigma2, direct.sigma2)
        self.assertEqual(via_round_trip.mu, direct.mu)

    def test_scalar_update_path_is_shift_stable(self):
        base = _grid_normal(13, 120)
        for c in (0.0, 1.7e9):
            acc = GaussianAccumulator()
            for value in base + c:
                acc.update(float(value), 1.0, None)
            fitted = GaussianEstimator().estimate(None, acc.value())
            expected = float(np.var(base + c))
            self.assertLessEqual(abs(fitted.sigma2 - expected) / max(expected, 1e-300), 1.0e-9)

    def test_mixture_fit_shift_invariant(self):
        # The full EM loop (initialization, responsibilities, per-component accumulators).
        rng = np.random.RandomState(21)
        mix = np.round(np.concatenate([rng.normal(0.0, 0.5, 200), rng.normal(8.0, 0.5, 200)]) * 65536.0) / 65536.0
        results = []
        for c in (0.0, 2.0**30):
            model = mixle.inference.fit(
                (mix + c).tolist(),
                MixtureEstimator(estimators=[GaussianEstimator() for _ in range(2)]),
                rng=np.random.RandomState(3),
                max_its=50,
            )
            results.append(sorted((comp.mu - c, comp.sigma2) for comp in model.components))
        for (mu0, s0), (mu1, s1) in zip(results[0], results[1]):
            self.assertLessEqual(abs(mu1 - mu0), 1.0e-6)
            self.assertLessEqual(abs(s1 - s0) / s0, 1.0e-6)

    def test_conjugate_prior_variance_shift_stable(self):
        from mixle.stats.bayes.normal_gamma import NormalGammaDistribution

        base = _grid_normal(17, 200)
        results = []
        for c in (0.0, 2.0**30):
            prior = NormalGammaDistribution(c, 1.0, 2.0, 1.0)
            model = mixle.inference.fit((base + c).tolist(), GaussianEstimator(prior=prior))
            results.append((model.mu - c, model.sigma2))
        self.assertLessEqual(abs(results[1][0] - results[0][0]), 1.0e-6)
        self.assertLessEqual(abs(results[1][1] - results[0][1]) / results[0][1], 1.0e-9)

    def test_inconsistent_hand_built_payload_falls_back_to_the_tuple(self):
        # A payload contradicting its own tuple must not silently change the estimate.
        bogus = GaussianSuffStat(10.0, 30.0, 5.0, 5.0)
        bogus.anchored = (100.0, 50.0, 25.0)
        from_bogus = GaussianEstimator().estimate(None, bogus)
        from_plain = GaussianEstimator().estimate(None, (10.0, 30.0, 5.0, 5.0))
        self.assertEqual(from_bogus.mu, from_plain.mu)
        self.assertEqual(from_bogus.sigma2, from_plain.sigma2)


class MaskedArrayGuardTestCase(unittest.TestCase):
    """B1: a nontrivial mask must be rejected, a trivial one must not."""

    def _masked(self):
        x = np.array([14.23, 13.2, 13.16, 14.37, 999.0])
        mask = np.zeros(5, bool)
        mask[4] = True
        return np.ma.masked_array(x, mask=mask)

    def test_fit_rejects_masked_values_with_remedy(self):
        with self.assertRaises(ValueError) as ctx:
            mixle.inference.fit(self._masked(), GaussianEstimator())
        message = str(ctx.exception)
        self.assertIn("fit() received a numpy masked array", message)
        self.assertIn("compressed()", message)
        self.assertIn("OptionalEstimator", message)

    def test_optimize_rejects_masked_values(self):
        with self.assertRaises(ValueError) as ctx:
            mixle.inference.optimize(self._masked(), GaussianEstimator())
        self.assertIn("optimize() received a numpy masked array", str(ctx.exception))

    def test_encoder_rejects_masked_values(self):
        # The family-level coercion path is guarded too (enc paths that bypass the fit verbs).
        from mixle.stats.univariate.continuous.gaussian import GaussianDataEncoder

        with self.assertRaises(ValueError) as ctx:
            GaussianDataEncoder().seq_encode(self._masked())
        self.assertIn("masked array", str(ctx.exception))

    def test_trivial_mask_is_not_rejected(self):
        x = np.array([14.23, 13.2, 13.16, 14.37])
        trivial = np.ma.masked_array(x, mask=False)
        fitted = mixle.inference.fit(trivial, GaussianEstimator())
        self.assertAlmostEqual(fitted.mu, float(np.mean(x)), places=12)

    def test_compressed_data_fits_the_unmasked_values(self):
        masked = self._masked()
        fitted = mixle.inference.fit(masked.compressed(), GaussianEstimator())
        self.assertAlmostEqual(fitted.mu, float(np.mean(masked.compressed())), places=12)


class AllZeroWeightGuardTestCase(unittest.TestCase):
    """B3: all-zero user weights are rejected at the entry; EM-internal zero counts are not."""

    def _weighted(self, weights):
        values = [13.16, 13.2, 14.23, 14.37, 13.9]
        return [WeightedObservation(v, w) for v, w in zip(values, weights)]

    def test_fit_rejects_all_zero_weights(self):
        with self.assertRaises(ValueError) as ctx:
            mixle.inference.fit(self._weighted([0.0] * 5), WeightedEstimator(GaussianEstimator()))
        message = str(ctx.exception)
        self.assertIn("fit() received observation weights that sum to zero", message)
        self.assertIn("positive weight", message)

    def test_optimize_rejects_all_zero_weights(self):
        with self.assertRaises(ValueError) as ctx:
            mixle.inference.optimize(self._weighted([0.0] * 5), WeightedEstimator(GaussianEstimator()))
        self.assertIn("optimize() received observation weights that sum to zero", str(ctx.exception))

    def test_partial_zero_weights_fit_the_surviving_rows(self):
        weights = [1.0, 1.0, 1.0, 0.0, 0.0]
        fitted = mixle.inference.fit(self._weighted(weights), WeightedEstimator(GaussianEstimator()))
        survivors = np.array([13.16, 13.2, 14.23])
        self.assertAlmostEqual(fitted.dist.mu, float(np.mean(survivors)), places=12)

    def test_em_internal_zero_count_still_returns_no_evidence_default(self):
        # The routine dead-component M-step (zero responsibility) must keep working: the guard
        # lives at the entry point only, never inside estimate().
        fitted = GaussianEstimator().estimate(0.0, (0.0, 0.0, 0.0, 0.0))
        self.assertEqual(fitted.mu, 0.0)
        self.assertEqual(fitted.sigma2, 1e-08)

    def test_unweighted_data_untouched_by_the_guard(self):
        fitted = mixle.inference.fit([13.16, 13.2, 14.23], GaussianEstimator())
        self.assertAlmostEqual(fitted.mu, float(np.mean([13.16, 13.2, 14.23])), places=12)


class GeneratorRngTestCase(unittest.TestCase):
    """t1-major: np.random.default_rng must be accepted by the fit verbs."""

    def _data(self):
        return np.random.RandomState(1).normal(5.0, 2.0, 200).tolist()

    def _estimator(self):
        return MixtureEstimator(estimators=[GaussianEstimator() for _ in range(2)])

    def test_fit_accepts_generator(self):
        model = mixle.inference.fit(self._data(), self._estimator(), rng=np.random.default_rng(0))
        self.assertEqual(len(model.components), 2)

    def test_optimize_accepts_generator(self):
        model = mixle.inference.optimize(self._data(), self._estimator(), rng=np.random.default_rng(0))
        self.assertEqual(len(model.components), 2)

    def test_generator_fits_are_deterministic_given_the_seed(self):
        one = mixle.inference.fit(self._data(), self._estimator(), rng=np.random.default_rng(42))
        two = mixle.inference.fit(self._data(), self._estimator(), rng=np.random.default_rng(42))
        self.assertEqual(str(one), str(two))

    def test_randomstate_and_seed_alias_unchanged(self):
        one = mixle.inference.fit(self._data(), self._estimator(), rng=np.random.RandomState(3))
        two = mixle.inference.fit(self._data(), self._estimator(), seed=3)
        self.assertEqual(str(one), str(two))


class EmptyDataMessageTestCase(unittest.TestCase):
    """t1-minor: empty-input errors name the called entry point and say "no observations"."""

    def test_fit_empty_and_none_messages(self):
        with self.assertRaises(ValueError) as ctx:
            mixle.inference.fit([], GaussianEstimator())
        self.assertIn("fit() received no observations", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            mixle.inference.fit(None, GaussianEstimator())
        self.assertIn("fit() received no observations", str(ctx.exception))

    def test_optimize_empty_and_none_messages(self):
        with self.assertRaises(ValueError) as ctx:
            mixle.inference.optimize([], GaussianEstimator())
        self.assertIn("optimize() received no observations", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            mixle.inference.optimize(None, GaussianEstimator())
        self.assertIn("optimize() received no observations", str(ctx.exception))

    def test_best_of_none_message(self):
        with self.assertRaises(ValueError) as ctx:
            mixle.inference.best_of(None, None, GaussianEstimator(), 1, 5, 0.1, 1e-6)
        self.assertIn("best_of() received no observations", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()


class ZeroWeightPriorExemptionTest(unittest.TestCase):
    """The all-zero-weight guard must not reject estimators that carry a prior.

    Found by the wave-3 adversarial check, this codebase's historical defect class: with
    ``suff_stat``/``pseudo_count`` present, zero total evidence has DEFINED semantics -- the
    posterior is the prior -- and the first version of the guard rejected exactly that state.
    Bare estimators (whose prior attributes are None or tuples of Nones) stay refused.
    """

    def _zeros(self):
        from mixle.stats.combinator.weighted import WeightedObservation

        return [WeightedObservation(v, 0.0) for v in (13.5, 14.0, 14.5, 13.16)]

    def test_prior_carrying_estimator_returns_the_prior_on_zero_evidence(self):
        from mixle.inference import fit
        from mixle.stats import GaussianEstimator
        from mixle.stats.combinator.weighted import WeightedEstimator

        m = fit(self._zeros(), WeightedEstimator(GaussianEstimator(suff_stat=(13.5, 0.25), pseudo_count=(2.0, 2.0))))
        self.assertEqual((m.dist.mu, m.dist.sigma2), (13.5, 0.25))

    def test_suff_stat_alone_is_a_prior(self):
        from mixle.inference import fit
        from mixle.stats import GaussianEstimator
        from mixle.stats.combinator.weighted import WeightedEstimator

        m = fit(self._zeros(), WeightedEstimator(GaussianEstimator(suff_stat=(13.5, 0.25))))
        self.assertEqual((m.dist.mu, m.dist.sigma2), (13.5, 0.25))

    def test_bare_estimator_is_still_refused(self):
        from mixle.inference import fit
        from mixle.stats import GaussianEstimator
        from mixle.stats.combinator.weighted import WeightedEstimator

        with self.assertRaises(ValueError) as ctx:
            fit(self._zeros(), WeightedEstimator(GaussianEstimator()))
        self.assertIn("weights that sum to zero", str(ctx.exception))

    def test_tuple_of_nones_is_not_a_prior(self):
        from mixle.inference.estimation import _estimator_carries_prior
        from mixle.stats import GaussianEstimator
        from mixle.stats.combinator.weighted import WeightedEstimator

        self.assertFalse(_estimator_carries_prior(WeightedEstimator(GaussianEstimator())))
        self.assertTrue(_estimator_carries_prior(GaussianEstimator(suff_stat=(13.5, 0.25))))
