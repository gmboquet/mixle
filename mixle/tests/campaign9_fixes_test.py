"""Campaign nine: regression tests for the six confirmed findings' fixes (D-0209).

Each class pins one finding's exact reproduction (or the closest deterministic equivalent) so a
future change cannot silently reopen it. Finding 1's milder (non-total-collapse) sub-case is
deliberately NOT pinned here -- see D-0209 for why an attempted fix there was reverted as unsound
(it broke ExtremeMagnitudeSpreadTestCase's already-pinned precision-recovery guarantee) and is
recorded as a known limit rather than closed.
"""

import unittest

import numpy as np

from mixle.stats.rankings.generalized_mallows import (
    GeneralizedMallowsAccumulatorFactory,
    GeneralizedMallowsDistribution,
    GeneralizedMallowsEstimator,
)
from mixle.stats.rankings.low_rank_permutation import LowRankPermutationDistribution
from mixle.stats.rankings.spearman_rho import SpearmanRankingDistribution, _validate_location
from mixle.stats.univariate.continuous.gaussian import GaussianAccumulator
from mixle.stats.univariate.continuous.generalized_extreme_value import (
    GeneralizedExtremeValueAccumulator,
    GeneralizedExtremeValueDataEncoder,
)
from mixle.stats.univariate.continuous.generalized_gaussian import (
    GeneralizedGaussianAccumulator,
    GeneralizedGaussianDataEncoder,
)
from mixle.stats.univariate.continuous.generalized_pareto import (
    GeneralizedParetoAccumulatorFactory,
    GeneralizedParetoDistribution,
    GeneralizedParetoEstimator,
)
from mixle.stats.univariate.continuous.gumbel import (
    GumbelAccumulator,
    GumbelDataEncoder,
    GumbelEstimator,
)
from mixle.stats.univariate.continuous.logistic import (
    LogisticAccumulatorFactory,
    LogisticDataEncoder,
    LogisticEstimator,
)
from mixle.stats.univariate.continuous.student_t import (
    StudentTAccumulator,
    StudentTDataEncoder,
    StudentTEstimator,
)
from mixle.stats.univariate.continuous.weibull import WeibullAccumulator, WeibullDataEncoder
from mixle.utils.automatic.detectors.generalized_pareto import _applies as gpd_detector_applies
from mixle.utils.serialization import to_serializable


class ZeroWeightOverflowDoesNotPoisonRawFoldTest(unittest.TestCase):
    """Finding 2: a zero-weight observation at extreme magnitude must not poison sum2 via nan."""

    OFFSET = 1.7e9
    N = 200
    SENTINEL = 1.0e200

    def _poisoned_chunk(self):
        rng = np.random.RandomState(1)
        real = self.OFFSET + rng.normal(0, 2.0, size=self.N)
        x = np.concatenate([[self.SENTINEL], real])
        w = np.concatenate([[0.0], np.ones(self.N)])
        return x, w

    def test_gumbel_seq_update_stays_finite(self):
        x, w = self._poisoned_chunk()
        acc = GumbelAccumulator()
        acc.seq_update(GumbelDataEncoder().seq_encode(x), w, None)
        self.assertTrue(np.isfinite(acc.sum2), acc.sum2)
        dist = GumbelEstimator().estimate(None, acc.value())
        self.assertTrue(np.isfinite(dist.loc) and np.isfinite(dist.scale) and dist.scale > 0.0)

    def test_gaussian_seq_update_stays_finite(self):
        x, w = self._poisoned_chunk()
        acc = GaussianAccumulator()
        acc.seq_update(x, w, None)
        self.assertTrue(np.isfinite(acc.sum2), acc.sum2)

    def test_logistic_seq_update_stays_finite(self):
        x, w = self._poisoned_chunk()
        acc = LogisticAccumulatorFactory().make()
        acc.seq_update(LogisticDataEncoder().seq_encode(x), w, None)
        self.assertTrue(np.isfinite(acc.sum2), acc.sum2)
        dist = LogisticEstimator().estimate(None, acc.value())
        self.assertTrue(np.isfinite(dist.scale) and dist.scale > 0.0)

    def test_weibull_seq_update_stays_finite(self):
        rng = np.random.RandomState(1)
        real = self.OFFSET + rng.uniform(1.0, 5.0, size=self.N)  # Weibull requires x >= 0
        x = np.concatenate([[self.SENTINEL], real])
        w = np.concatenate([[0.0], np.ones(self.N)])
        acc = WeibullAccumulator()
        acc.seq_update(WeibullDataEncoder().seq_encode(x), w, None)
        self.assertTrue(np.isfinite(acc.sum2), acc.sum2)

    def test_generalized_extreme_value_seq_update_stays_finite(self):
        x, w = self._poisoned_chunk()
        acc = GeneralizedExtremeValueAccumulator()
        acc.seq_update(GeneralizedExtremeValueDataEncoder().seq_encode(x), w, None)
        self.assertTrue(np.isfinite(acc.sum2) and np.isfinite(acc.sum3), (acc.sum2, acc.sum3))

    def test_generalized_gaussian_seq_update_stays_finite(self):
        x, w = self._poisoned_chunk()
        acc = GeneralizedGaussianAccumulator()
        acc.seq_update(GeneralizedGaussianDataEncoder().seq_encode(x), w, None)
        self.assertTrue(np.isfinite(acc.s2) and np.isfinite(acc.s3) and np.isfinite(acc.s4), (acc.s2, acc.s3, acc.s4))

    def test_reordered_chunk_first_no_anchor_yet_does_not_crash(self):
        # The finding's second variant: the poisoned chunk arrives BEFORE any anchor exists, so
        # needs_anchor's own decision must not be corrupted by the overflow either.
        rng = np.random.RandomState(1)
        real = self.OFFSET + rng.normal(0, 2.0, size=self.N)
        x = np.concatenate([[self.SENTINEL], real])
        w = np.concatenate([[0.0], np.ones(self.N)])
        acc = GumbelAccumulator()
        acc.seq_update(GumbelDataEncoder().seq_encode(x), w, None)
        dist = GumbelEstimator().estimate(None, acc.value())
        self.assertTrue(np.isfinite(dist.loc) and np.isfinite(dist.scale) and dist.scale > 0.0)


class ScaleFloorCollapseIsDisclosedTest(unittest.TestCase):
    """Finding 3: a total anchored collapse at nanosecond-epoch magnitude must disclose, not be
    silent -- matching GaussianEstimator's own existing variance-floor precedent."""

    LOC = 1.0e18

    def test_generalized_pareto_discloses(self):
        gen = GeneralizedParetoDistribution(2.0, 0.15, loc=self.LOC)
        x = gen.sampler(seed=0).sample(size=2000)
        acc = GeneralizedParetoAccumulatorFactory(loc=self.LOC).make()
        acc.seq_update(gen.dist_to_encoder().seq_encode(x), np.ones_like(x), None)
        fitted = GeneralizedParetoEstimator(loc=self.LOC).estimate(2000.0, acc.value())
        repairs = fitted.numerical_repairs()
        self.assertTrue(any(r.startswith("variance-floored") for r in repairs), repairs)

    def test_gumbel_discloses(self):
        rng = np.random.RandomState(0)
        x = self.LOC + rng.gumbel(0.0, 2.0, size=2000)
        acc = GumbelAccumulator()
        acc.seq_update(GumbelDataEncoder().seq_encode(x), np.ones_like(x), None)
        fitted = GumbelEstimator().estimate(2000.0, acc.value())
        repairs = fitted.numerical_repairs()
        self.assertTrue(any(r.startswith("scale-floored") for r in repairs), repairs)

    def test_student_t_discloses(self):
        rng = np.random.RandomState(0)
        x = self.LOC + rng.standard_t(5.0, size=2000) * 2.0
        acc = StudentTAccumulator()
        acc.seq_update(StudentTDataEncoder().seq_encode(x), np.ones_like(x), None)
        fitted = StudentTEstimator(df=5.0).estimate(2000.0, acc.value())
        repairs = fitted.numerical_repairs()
        self.assertTrue(any(r.startswith("scale-floored") for r in repairs), repairs)

    def test_ordinary_magnitude_stays_clean(self):
        # No regression: ordinary-magnitude fits must not pick up a spurious floor note.
        gen = GeneralizedParetoDistribution(2.0, 0.15, loc=0.0)
        x = gen.sampler(seed=0).sample(size=2000)
        acc = GeneralizedParetoAccumulatorFactory(loc=0.0).make()
        acc.seq_update(gen.dist_to_encoder().seq_encode(x), np.ones_like(x), None)
        fitted = GeneralizedParetoEstimator(loc=0.0).estimate(2000.0, acc.value())
        self.assertEqual(fitted.numerical_repairs(), ())


class SpearmanOrdinaryConstructionToleranceTest(unittest.TestCase):
    """Finding 5: ordinary (non-restore) construction must not spuriously reject a legitimate
    mean-rank vector carrying an unremarkable common offset."""

    def test_offset_mean_rank_vector_constructs(self):
        raw = np.array([94472424.81325343, 94472425.81325343, 94472426.81325343])
        dist = SpearmanRankingDistribution(sigma=raw, rho=0.5)
        self.assertEqual(len(dist.sigma), 3)

    def test_batch_offsets_rarely_reject(self):
        rng = np.random.RandomState(42)
        n_fail = 0
        for _ in range(200):
            dim = rng.randint(2, 10)
            offset = rng.uniform(1e6, 1e7)
            raw = np.arange(dim, dtype=np.float64) + offset
            try:
                _validate_location(raw, already_centered=False)
            except ValueError:
                n_fail += 1
        self.assertEqual(n_fail, 0)

    def test_genuine_corruption_still_rejected(self):
        bad = np.array([1.0e6, 1.0e6 + 1000.0, 1.0e6 + 2000.0])
        with self.assertRaises(ValueError):
            SpearmanRankingDistribution(sigma=bad, rho=0.5)


class RankingDiagnosticsSerializeTest(unittest.TestCase):
    """Finding 6: GeneralizedMallows/LowRankPermutation instances must serialize and hash."""

    def test_generalized_mallows_construction_serializes(self):
        dist = GeneralizedMallowsDistribution([0, 1, 2, 3], theta=1.0, n_mc=500)
        to_serializable(dist)  # must not raise SerializationError

    def test_generalized_mallows_fitted_serializes(self):
        dist = GeneralizedMallowsDistribution([0, 1, 2, 3], theta=1.0, n_mc=500)
        orderings = np.array([np.random.RandomState(i).permutation(4) for i in range(30)])
        enc = dist.dist_to_encoder()
        acc = GeneralizedMallowsAccumulatorFactory(4).make()
        acc.seq_update(enc.seq_encode(orderings), np.ones(30), None)
        fitted = GeneralizedMallowsEstimator(4).estimate(30.0, acc.value())
        to_serializable(fitted)  # must not raise SerializationError

    def test_low_rank_permutation_construction_serializes(self):
        dist = LowRankPermutationDistribution(
            np.random.RandomState(0).randn(4, 2), np.random.RandomState(1).randn(4, 2)
        )
        to_serializable(dist)  # must not raise SerializationError


class AutomaticGeneralizedParetoDetectorSanityTest(unittest.TestCase):
    """Finding 7: the auto-inference GPD gate must not admit a candidate whose real MLE fit is a
    magnitude-collapsed nonsense corner, while still admitting legitimate heavy-tailed data."""

    def test_nanosecond_epoch_collapse_is_refused(self):
        from scipy import stats

        rng = np.random.RandomState(0)
        exceedances = stats.genpareto.rvs(0.3, loc=0, scale=300.0, size=2000, random_state=rng)
        arr = 1.7e18 + exceedances
        self.assertFalse(gpd_detector_applies(arr))

    def test_ordinary_heavy_tail_still_applies(self):
        from scipy import stats

        rng = np.random.RandomState(1)
        arr = stats.genpareto.rvs(0.3, loc=0, scale=50.0, size=2000, random_state=rng) + 1.0
        self.assertTrue(gpd_detector_applies(arr))

    def test_exponential_data_still_screened_out(self):
        rng = np.random.RandomState(3)
        arr = rng.exponential(scale=5.0, size=2000) + 1.0
        self.assertFalse(gpd_detector_applies(arr))

    def test_near_dirac_ordinary_magnitude_is_refused(self):
        # A DIFFERENT failure mode than the magnitude collapse above: a heavy point mass at
        # ordinary magnitude can also send scipy's MLE to a nonsense corner (GPD's density is
        # 1/scale at the fitted threshold for any shape, so ties at the minimum drive scale -> 0
        # regardless of magnitude). This test originally asserted the OPPOSITE -- that this case
        # should stay admitted -- reasoning that mixle.lifecycle's _degenerate_likelihood_spike
        # safety net downstream would catch it. Round-2 review (D-0209) found that reasoning
        # itself flawed: that safety net only protects the full propose()/EM pipeline, not a
        # direct caller like get_estimator()/analyze_structure() that queries this family
        # directly. The gate now refuses ANY genuinely-degenerate MLE case -- magnitude-driven or
        # point-mass-driven -- via the independent _boundary_point_mass_is_suspect check, matching
        # release_wave3_lifecycle_test.test_degenerate_near_dirac_fit_does_not_win's own updated
        # expectation that a DIFFERENT family wins this data outright rather than relying on a
        # downstream rejection of an admitted-but-degenerate GPD fit.
        rng = np.random.default_rng(0)
        arr = rng.geometric(0.4, size=2000).astype(float)
        self.assertFalse(gpd_detector_applies(arr))


if __name__ == "__main__":
    unittest.main()
