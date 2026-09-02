"""Campaign nine, fix-wave review round: regression tests for the 12 findings an independent
adversarial review of campaign9_fixes_test.py's own fix wave confirmed (D-0209).

The review found the same defect classes recurring far beyond the six files/paths the first pass
touched: hand-rolled stacked-mixture-kernel hooks, the exp-family generated-statistics acceleration
path, several more accumulators' scalar update() methods, and two more disclosure/serialization
siblings. Each class below pins one confirmed finding.
"""

import unittest

import numpy as np

from mixle.engines import NUMPY_ENGINE
from mixle.stats import MixtureEstimator
from mixle.stats.compute.declarations import generated_sufficient_statistics
from mixle.stats.compute.stacked import stacked_component_params, stacked_component_sufficient_statistics
from mixle.stats.latent.hierarchical import HierarchicalNormalDistribution, HierarchicalNormalEstimator
from mixle.stats.multivariate.diagonal_gaussian import DiagonalGaussianAccumulator
from mixle.stats.multivariate.multivariate_student_t import MultivariateStudentTAccumulator
from mixle.stats.rankings.bradley_terry import BradleyTerryEstimator
from mixle.stats.rankings.generalized_mallows_model import GeneralizedMallowsModelDistribution
from mixle.stats.rankings.mallows import MallowsDistribution
from mixle.stats.univariate.continuous.gaussian import GaussianDistribution, GaussianEstimator
from mixle.stats.univariate.continuous.generalized_pareto import (
    GeneralizedParetoAccumulator,
    GeneralizedParetoEstimator,
)
from mixle.stats.univariate.continuous.gumbel import GumbelAccumulator, GumbelDistribution, GumbelEstimator
from mixle.stats.univariate.continuous.half_normal import HalfNormalAccumulator, HalfNormalEstimator
from mixle.stats.univariate.continuous.log_gaussian import LogGaussianAccumulator, LogGaussianEstimator
from mixle.stats.univariate.continuous.logistic import LogisticAccumulator, LogisticEstimator
from mixle.stats.univariate.continuous.rayleigh import RayleighAccumulator, RayleighEstimator
from mixle.stats.univariate.continuous.rician import RicianDistribution
from mixle.stats.univariate.continuous.student_t import StudentTAccumulator, StudentTEstimator
from mixle.stats.univariate.continuous.tweedie import TweedieAccumulator, TweedieEstimator
from mixle.utils.automatic.detectors.generalized_pareto import _applies as gpd_detector_applies
from mixle.utils.serialization import to_serializable


class StackedKernelSquareBeforeWeightTest(unittest.TestCase):
    """Finding #1: backend_stacked_sufficient_statistics hooks must not poison to NaN."""

    def test_gumbel_stacked_stats_stay_finite(self):
        components = [GumbelDistribution(loc=-2.0, scale=0.7), GumbelDistribution(loc=2.0, scale=1.1)]
        route = stacked_component_params(components, NUMPY_ENGINE)
        rng = np.random.RandomState(3)
        n = 40
        x = rng.normal(size=n)
        x[5] = 2.0e154
        gamma = np.column_stack([np.full(n, 0.5), np.full(n, 0.5)])
        gamma[5, :] = 0.0
        est = MixtureEstimator([GumbelEstimator(), GumbelEstimator()])
        stats = stacked_component_sufficient_statistics(x, gamma, route, NUMPY_ENGINE, est)
        for s in stats:
            self.assertTrue(np.all(np.isfinite(np.asarray(s))), s)

    def test_rician_backend_stacked_stays_finite(self):
        xx = np.array([1.0, 2.0, 1e100, 3.0])
        ww = np.array([[1.0, 0.5], [1.0, 0.5], [0.0, 0.0], [1.0, 0.5]])
        result = RicianDistribution.backend_stacked_sufficient_statistics(xx, ww, {}, NUMPY_ENGINE)
        for r in result:
            self.assertTrue(np.all(np.isfinite(np.asarray(r))), r)


class SerializationSiblingsTest(unittest.TestCase):
    """Finding #2: four more FitDiagnostics siblings must serialize."""

    def test_mallows_serializes(self):
        dist = MallowsDistribution([2, 0, 1], theta=1.0)
        fitted = dist.estimator(pseudo_count=2.0).estimate(None, (0.0, np.zeros((3, 3))))
        to_serializable(fitted)

    def test_bradley_terry_serializes(self):
        m = np.array([[0, 3, 1], [1, 0, 2], [2, 1, 0]], dtype=float)
        fitted = BradleyTerryEstimator(3).estimate(None, (float(m.sum()), m))
        to_serializable(fitted)

    def test_hierarchical_normal_serializes(self):
        groups = [np.array([-1.0, -0.5, 0.0]), np.array([1.0, 1.5, 2.0])]
        dist = HierarchicalNormalDistribution(0.0, 1.0, 1.0)
        enc = dist.dist_to_encoder().seq_encode(groups)
        acc = HierarchicalNormalEstimator().accumulator_factory().make()
        acc.seq_update(enc, np.ones(len(groups)), None)
        fitted = HierarchicalNormalEstimator().estimate(None, acc.value())
        to_serializable(fitted)

    def test_generalized_mallows_model_serializes(self):
        gmm = GeneralizedMallowsModelDistribution([0, 1, 2], [1.5, 0.8])
        x = gmm.sampler(seed=0).sample(size=200)
        enc = gmm.dist_to_encoder().seq_encode(x)
        acc = gmm.estimator().accumulator_factory().make()
        acc.seq_update(enc, np.ones(200), None)
        fitted = gmm.estimator().estimate(200.0, acc.value())
        to_serializable(fitted)


class FloorDisclosureSiblingsTest(unittest.TestCase):
    """Finding #3: Logistic/LogGaussian must disclose a floor bind, matching GPD/Gumbel/Student-t."""

    def test_logistic_discloses(self):
        n = 50
        x = np.full(n, 1_700_000_000_000.0)
        acc = LogisticAccumulator()
        acc.seq_update(x, np.ones(n), None)
        dist = LogisticEstimator().estimate(None, acc.value())
        self.assertTrue(any(r.startswith("scale-floored") for r in dist.numerical_repairs()))

    def test_log_gaussian_discloses(self):
        x = np.full(50, np.exp(40))
        acc = LogGaussianAccumulator()
        acc.seq_update(x, np.ones(50), None)
        dist = LogGaussianEstimator().estimate(None, acc.value())
        self.assertTrue(any(r.startswith("variance-floored") for r in dist.numerical_repairs()))


class GeneratedStatisticsZeroWeightTest(unittest.TestCase):
    """Finding #4: the exp-family acceleration path must not crash on a zero-weight extreme row."""

    def test_generated_sufficient_statistics_matches_legacy(self):
        dist = GaussianDistribution(0.0, 1.0)
        enc = dist.dist_to_encoder().seq_encode(np.array([1.0, 2.0, 1e200, 3.0]))
        weights = np.array([1.0, 1.0, 0.0, 1.0])
        result = generated_sufficient_statistics(dist, enc, weights, NUMPY_ENGINE)
        legacy = GaussianEstimator().accumulator_factory().make()
        legacy.seq_update(enc, weights, dist)
        legacy_value = legacy.value()
        for got, want in zip(result, legacy_value):
            self.assertAlmostEqual(float(got), float(want), places=6)


class LogGaussianDomainBoundaryTest(unittest.TestCase):
    """Finding #5: a zero-weight x=0 observation must not poison LogGaussianAccumulator."""

    def test_boundary_zero_weight_does_not_poison(self):
        acc = LogGaussianAccumulator()
        acc.update(0.0, 0.0, None)
        acc.update(5.0, 1.0, None)
        value = acc.value()
        self.assertTrue(all(np.isfinite(v) for v in value), value)
        LogGaussianEstimator().estimate(None, value)  # must not raise


class ScalarUpdateRawFoldSiblingsTest(unittest.TestCase):
    """Findings #6/#8: Gumbel/Student-t/GeneralizedPareto's own scalar update() raw fold."""

    def test_gumbel_scalar_update_stays_finite(self):
        acc = GumbelAccumulator()
        acc.update(1.0e200, 0.0, None)
        acc.update(5.0, 1.0, None)
        self.assertTrue(np.isfinite(acc.sum2))
        dist = GumbelEstimator().estimate(1.0, acc.value())
        self.assertTrue(np.isfinite(dist.scale) and dist.scale > 0.0)

    def test_gumbel_scalar_update_poison_first_ordering(self):
        acc = GumbelAccumulator()
        acc.update(1.0e250, 0.0, None)
        acc.update(10.0, 1.0, None)
        acc.update(11.0, 1.0, None)
        acc.update(9.5, 1.0, None)
        self.assertTrue(np.isfinite(acc.sum2) and np.isfinite(acc._anchored_sum2))
        dist = GumbelEstimator().estimate(acc.count, acc.value())
        self.assertTrue(np.isfinite(dist.scale) and dist.scale > 0.0)

    def test_student_t_scalar_update_stays_finite(self):
        offset, n, sentinel = 1.7e9, 50, 1.0e200
        rng = np.random.RandomState(1)
        real = offset + rng.normal(0, 2.0, size=n)
        acc = StudentTAccumulator()
        acc.update(sentinel, 0.0, None)
        for v in real:
            acc.update(float(v), 1.0, None)
        dist = StudentTEstimator(df=5.0).estimate(float(n), acc.value())
        self.assertTrue(np.isfinite(dist.scale) and dist.scale > 0.0)

    def test_generalized_pareto_scalar_update_stays_finite(self):
        acc = GeneralizedParetoAccumulator(loc=0.0)
        acc.update(1.0e200, 0.0, None)
        acc.update(5.0, 1.0, None)
        acc.update(6.0, 1.0, None)
        acc.update(4.5, 1.0, None)
        self.assertTrue(np.isfinite(acc.sum2))
        GeneralizedParetoEstimator(loc=0.0).estimate(acc.count, acc.value())  # must not raise


class UnanchoredAccumulatorSiblingsTest(unittest.TestCase):
    """Finding #9: HalfNormal/Rayleigh's non-anchored scalar and chunk folds."""

    def test_half_normal_update_undisclosed_wrong_fit_is_fixed(self):
        acc = HalfNormalAccumulator()
        acc.update(3.0, 1.0, None)
        acc.update(1.0e250, 0.0, None)
        acc.update(4.0, 1.0, None)
        acc.update(2.5, 1.0, None)
        fitted = HalfNormalEstimator().estimate(acc.count, acc.value())
        self.assertAlmostEqual(fitted.sigma, float(np.sqrt(np.mean([9.0, 16.0, 6.25]))), places=6)

    def test_rayleigh_update_does_not_crash(self):
        acc = RayleighAccumulator()
        acc.update(3.0, 1.0, None)
        acc.update(1.0e250, 0.0, None)
        acc.update(4.0, 1.0, None)
        acc.update(2.5, 1.0, None)
        RayleighEstimator().estimate(acc.count, acc.value())  # must not raise


class TweedieSeqUpdateTest(unittest.TestCase):
    """Finding #10: TweedieAccumulator.seq_update's chunked square-before-weight overflow."""

    def test_seq_update_stays_finite(self):
        acc = TweedieAccumulator()
        x = np.array([3.0, 1.0e250, 4.0, 2.5])
        w = np.array([1.0, 0.0, 1.0, 1.0])
        acc.seq_update(x, w, None)
        self.assertTrue(np.isfinite(acc.sum2))
        TweedieEstimator().estimate(acc.count, acc.value())  # must not raise


class MultivariateSquareBeforeWeightTest(unittest.TestCase):
    """Findings #11/#12: DiagonalGaussian/MultivariateStudentT's hand-rolled anchored folds."""

    def test_diagonal_gaussian_update_stays_finite(self):
        acc = DiagonalGaussianAccumulator(dim=2)
        acc.update([3.0, 4.0], 1.0, None)
        acc.update([1.0e250, 5.0], 0.0, None)
        acc.update([3.5, 4.5], 1.0, None)
        self.assertFalse(np.any(np.isnan(acc._anchored_sum2)))

    def test_multivariate_student_t_update_stays_finite(self):
        acc = MultivariateStudentTAccumulator(dof=5.0, dim=2)
        acc.update(np.array([3.0, 4.0]), 1.0, None)
        acc.update(np.array([1.0e250, 5.0]), 0.0, None)
        acc.update(np.array([3.5, 4.5]), 1.0, None)
        self.assertFalse(np.any(np.isnan(acc.sum_uxx)))


class MagnitudeGateOrdinaryMagnitudeTest(unittest.TestCase):
    """Finding #7 (round 1): the magnitude-precision gate must not fire at ordinary real-world
    magnitudes -- an ordinary heavy tail must stay admitted regardless of offset.

    This class's own near-Dirac case was rewritten after round-2 review (D-0209, finding #1):
    the round-1 replacement gate (``_typical_adjacent_gap``) is blind to boundary point mass (GPD's
    density is ``1/scale`` at the fitted threshold for any shape, so ties at the minimum drive
    ``scale -> 0`` independent of magnitude), so a near-Dirac sample -- which ties heavily at its
    own minimum by construction -- was being admitted for the wrong reason: not because the gate
    correctly judged it non-degenerate, but because the gate could not see point-mass degeneracy at
    all. The correct invariant was never "near-Dirac data is always admitted"; it is "the verdict
    tracks genuine degeneracy, consistently across every offset" -- round 2 added
    ``_boundary_point_mass_is_suspect`` precisely so near-Dirac data is refused at every offset,
    the same way an ordinary heavy tail is admitted at every offset.
    """

    def test_ordinary_heavy_tail_stays_admitted_across_realistic_offsets(self):
        from scipy import stats

        rng = np.random.RandomState(0)
        exceedances = stats.genpareto.rvs(0.3, loc=0, scale=300.0, size=2000, random_state=rng)
        for offset in (0.0, 1e10, 1e12, 1.7e12):
            with self.subTest(offset=offset):
                self.assertTrue(gpd_detector_applies(offset + exceedances))

    def test_near_dirac_is_refused_across_realistic_offsets(self):
        rng = np.random.default_rng(0)
        base = rng.geometric(0.4, size=2000).astype(float)
        for offset in (0.0, 1e10, 1e12, 1.7e12, 1e15):
            with self.subTest(offset=offset):
                self.assertFalse(gpd_detector_applies(base + offset))

    def test_magnitude_collapse_still_refused(self):
        from scipy import stats

        rng = np.random.RandomState(0)
        exceedances = stats.genpareto.rvs(0.3, loc=0, scale=300.0, size=2000, random_state=rng)
        self.assertFalse(gpd_detector_applies(1.7e18 + exceedances))


if __name__ == "__main__":
    unittest.main()
