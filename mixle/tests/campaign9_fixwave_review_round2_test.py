"""Campaign nine, fix-wave review round 2: regression tests for a further batch of findings an
independent adversarial review confirmed on top of ``campaign9_fixwave_review_test.py``'s own fix
wave (D-0209).

This round found the same defect classes recurring in yet more places: another accumulator's
``seq_update`` with a square-before-weight fast path (Skellam), five more ``FitDiagnostics``
dataclasses that were never registered as ``__pysp_serializable__`` (Matching, Thurstone-Mosteller,
the shared tie-model base Davidson/RaoKupper use, LDA/LabeledLDA, and the quantized HMM), a
non-idempotent centering step on restore (Bradley-Terry, mirroring the Thurstone
``_mu_already_centered`` precedent), an outer-product-before-weight overflow in the probabilistic
PCA accumulator, a silent conjugate-posterior variance-floor bind with no disclosure (LogGaussian),
and a missing histogram-kind branch in the generated stacked sufficient-statistics path that
misapplied a moment-shaped reducer to a count-histogram statistic. Each class below pins one
confirmed finding.
"""

import unittest

import numpy as np
from numpy.random import RandomState

from mixle.engines import NUMPY_ENGINE
from mixle.inference import seq_estimate, seq_initialize
from mixle.stats import (
    CategoricalDistribution,
    CategoricalEstimator,
    DavidsonDistribution,
    NegativeBinomialDistribution,
    ThurstoneMostellerDistribution,
)
from mixle.stats.bayes.normal_gamma import NormalGammaDistribution
from mixle.stats.compute.declarations import (
    generated_stacked_params,
    generated_stacked_sufficient_statistics,
    generated_sufficient_statistics,
)
from mixle.stats.latent.lda import LDADistribution, LDAEstimator
from mixle.stats.latent.probabilistic_pca import ProbabilisticPCAAccumulator
from mixle.stats.latent.quantized_hidden_markov_model import (
    QuantizedHiddenMarkovEstimator,
    QuantizedHiddenMarkovModelDistribution,
)
from mixle.stats.rankings.bradley_terry import BradleyTerryEstimator
from mixle.stats.rankings.matching import MatchingEstimator
from mixle.stats.univariate.continuous.log_gaussian import LogGaussianAccumulator, LogGaussianEstimator
from mixle.stats.univariate.discrete.skellam import SkellamAccumulator, SkellamEstimator
from mixle.utils.optsutil import count_by_value
from mixle.utils.serialization import from_json, to_json, to_serializable


class SkellamSeqUpdateTest(unittest.TestCase):
    """Finding #5: SkellamAccumulator.seq_update's chunked square-before-weight overflow.

    Mirrors ``TweedieSeqUpdateTest`` in campaign9_fixwave_review_test.py: a weight of exactly 0.0
    must contribute exactly zero to ``chunk_sum2`` regardless of the paired observation's
    magnitude, but squaring before weighting can overflow to a non-finite value for an ordinary
    finite row, and ``inf * 0.0`` is ``nan``. The fix recomputes with the zero-weight row masked to
    0.0, only on this rare already-broken path.
    """

    def test_seq_update_stays_finite(self):
        acc = SkellamAccumulator()
        x = np.array([3.0, 1.0e200, 4.0, 2.5])
        w = np.array([1.0, 0.0, 1.0, 1.0])
        acc.seq_update(x, w, None)
        self.assertTrue(np.isfinite(acc.sum2))
        SkellamEstimator().estimate(acc.count, acc.value())  # must not raise


class SerializationSiblingsTest(unittest.TestCase):
    """Findings #6/#11/#12/#13/#14: five more FitDiagnostics siblings must serialize.

    Mirrors ``SerializationSiblingsTest`` in campaign9_fixwave_review_test.py: every one of these
    ``FitDiagnostics``/``OptimizationDiagnostics`` dataclasses is attached unconditionally by its
    estimator's ``estimate()``, so without ``__pysp_serializable__ = True`` a fitted instance of any
    of these five families raised an unhandled ``SerializationError`` from ``to_serializable``.
    """

    def test_matching_serializes(self):
        acc = MatchingEstimator(3).accumulator_factory().make()
        # Three permutations whose stacked assignment counts are already uniform (each cell gets
        # exactly 1), so the default pseudo_count=1.0 target is met at log_w=0 and the fixed point
        # converges in a single iteration -- keeps this a serialization-plumbing test, not a slow
        # convergence exercise.
        acc.update([0, 1, 2], 1.0, None)
        acc.update([1, 2, 0], 1.0, None)
        acc.update([2, 0, 1], 1.0, None)
        fitted = MatchingEstimator(3).estimate(3.0, acc.value())
        to_serializable(fitted)

    def test_thurstone_moseller_serializes(self):
        pair_dist = ThurstoneMostellerDistribution(np.zeros(3))
        pair_accumulator = pair_dist.estimator().accumulator_factory().make()
        pair_accumulator.update((0, 1), 1.0, None)
        # Node 2 is never compared, so the graph is disconnected; regularizing with a pseudo_count
        # makes the fit well-posed anyway, which is the cheapest way to reach a real
        # ThurstoneMostellerFitDiagnostics-bearing fit.
        fitted = pair_dist.estimator(0.5).estimate(1.0, pair_accumulator.value())
        to_serializable(fitted)

    def test_tie_model_serializes(self):
        # DavidsonDistribution and RaoKupperDistribution share the same TieModelFitDiagnostics via
        # _BaseTieDistribution, so exercising one exercises the sibling registration too.
        tie_dist = DavidsonDistribution(np.zeros(3))
        tie_accumulator = tie_dist.estimator().accumulator_factory().make()
        tie_accumulator.update((0, 1, 2), 1.0, None)
        fitted = tie_dist.estimator(0.5).estimate(1.0, tie_accumulator.value())
        to_serializable(fitted)

    def test_lda_serializes(self):
        topics = [
            CategoricalDistribution({0: 0.6, 1: 0.2, 2: 0.1, 3: 0.1}),
            CategoricalDistribution({0: 0.1, 1: 0.1, 2: 0.4, 3: 0.4}),
        ]
        dist = LDADistribution(topics, alpha=[1.0, 1.0], len_dist=CategoricalDistribution({5: 0.5, 6: 0.5}))
        raw = dist.sampler(seed=3).sample(20)
        data = [sorted(count_by_value(u).items()) for u in raw]
        estimator = LDAEstimator([CategoricalEstimator() for _ in range(2)])
        enc = dist.dist_to_encoder().seq_encode(data)
        accumulator = estimator.accumulator_factory().make()
        accumulator.seq_update(enc, np.ones(len(data)), dist)
        fitted = estimator.estimate(float(len(data)), accumulator.value())
        self.assertTrue(fitted.fit_diagnostics.converged)
        to_serializable(fitted)  # LabeledLDAEstimator reuses this same LDAOptimizationDiagnostics.

    def test_quantized_hmm_serializes(self):
        gen = QuantizedHiddenMarkovModelDistribution(
            0.5,
            ["a", "b", "c"],
            [[0, 1], [2, 0]],
            [[0, 1, 2], [2, 1, 0]],
            initial_exponents=[0, 1],
            init_mode="quantized",
            len_dist=CategoricalDistribution({3: 0.5, 4: 0.5}),
        )
        data = gen.sampler(seed=11).sample(20)
        # fixed_theta skips the scalar optimizer entirely, the cheapest deterministic way to reach a
        # real QuantizedHMMFitDiagnostics-bearing fit.
        estimator = QuantizedHiddenMarkovEstimator(
            2, pseudo_count=0.5, k_max=12, fixed_theta=0.5, len_estimator=CategoricalEstimator()
        )
        encoder = estimator.accumulator_factory().make().acc_to_encoder()
        enc_data = [(len(data), encoder.seq_encode(data))]
        model = seq_initialize(enc_data, estimator, RandomState(7), p=1.0)
        model = seq_estimate(enc_data, estimator, model)
        self.assertTrue(model.fit_diagnostics.converged)
        to_serializable(model)


class BradleyTerryIdempotentCenteringTest(unittest.TestCase):
    """Findings #4/#9/#16: Bradley-Terry's log_w centering must round-trip exactly through JSON.

    ``BradleyTerryDistribution.__init__`` centers ``log_w`` (mean zero, worths are identified only
    up to scale) via a longdouble computation. ``__pysp_getstate__`` serializes that already-centered
    array verbatim, and a naive ``__pysp_setstate__`` would feed it back through the SAME
    unconditional centering step -- centering an already-centered array is not exactly idempotent,
    so a generic fraction of fits would drift by 1-few ULP on restore. The fix mirrors
    ThurstoneDistribution's ``_mu_already_centered`` precedent with a ``_log_w_already_centered``
    keyword-only constructor flag that ``__pysp_setstate__`` alone sets, restoring ``log_w`` as given
    rather than re-deriving it.
    """

    _DIMS = (2, 3, 4, 5, 6, 8)
    _SEEDS = range(5)

    @staticmethod
    def _fit(dim: int, seed: int):
        # A hand-built, all-positive win matrix (rather than sampled comparisons) keeps every case
        # trivially connected and strongly connected by construction -- no regularization needed --
        # while the MM fixed point still produces "generic" float64 log-worths, exactly the kind of
        # bit pattern whose post-centering residual mean generically lands a few ULP from zero.
        rng = np.random.RandomState(seed * 100 + dim)
        wins = rng.uniform(1.0, 20.0, size=(dim, dim))
        np.fill_diagonal(wins, 0.0)
        return BradleyTerryEstimator(dim).estimate(float(wins.sum()), (float(wins.sum()), wins))

    def test_json_round_trip_is_bit_identical_across_a_seeded_sweep(self):
        checked = 0
        for dim in self._DIMS:
            for seed in self._SEEDS:
                with self.subTest(dim=dim, seed=seed):
                    fitted = self._fit(dim, seed)
                    restored = from_json(to_json(fitted))
                    self.assertTrue(
                        np.array_equal(fitted.log_w, restored.log_w),
                        f"log_w drifted by {restored.log_w - fitted.log_w!r} across a JSON round trip "
                        f"(dim={dim!r} seed={seed!r})",
                    )
                    # Round-trip the round-tripped object again: restoring state must be a fixed
                    # point, not merely correct once.
                    restored_again = from_json(to_json(restored))
                    self.assertTrue(np.array_equal(restored.log_w, restored_again.log_w))
                    checked += 1
        # Large enough that a reintroduced regression cannot hide behind one lucky exact-zero
        # residual (measured: about three-quarters of these combinations reproduce the mismatch
        # under the pre-fix re-centering behavior).
        self.assertGreater(checked, 20)


class ProbabilisticPCAOuterProductBeforeWeightTest(unittest.TestCase):
    """Finding #10: ProbabilisticPCAAccumulator.update's outer-product-before-weight overflow.

    Mirrors ``MultivariateSquareBeforeWeightTest`` in campaign9_fixwave_review_test.py: a weight of
    exactly 0.0 must contribute exactly zero to ``sum2`` regardless of the observation's magnitude,
    but ``np.outer(xx, xx)`` squares before weighting can be applied, and ``inf * 0.0`` is ``nan``.
    """

    def test_update_stays_finite(self):
        acc = ProbabilisticPCAAccumulator(dim=2)
        acc.update(np.array([3.0, 4.0]), 1.0, None)
        acc.update(np.array([1.0e200, 5.0]), 0.0, None)
        acc.update(np.array([3.5, 4.5]), 1.0, None)
        self.assertFalse(np.any(np.isnan(acc.sum2)))


class LogGaussianConjugateFloorDisclosureTest(unittest.TestCase):
    """Finding #17: LogGaussianEstimator._estimate_conjugate must disclose a variance-floor bind.

    Matches the sibling MLE branch's own "variance-floored" disclosure a few dozen lines below it
    (and GaussianEstimator._estimate_conjugate's precedent): when the closed-form conjugate
    posterior variance collapses to (or below) ``min_covar``, the fitted distribution must carry a
    ``_numerical_repairs`` note saying so, not silently substitute the floor.
    """

    def test_conjugate_posterior_discloses_variance_floor(self):
        # A tight NormalGamma prior (tiny b, so the prior itself carries almost no variance mass)
        # combined with perfectly degenerate log-space data (constant log(x) == the prior mean, so
        # both scatter terms new_b0/new_b1 are exactly zero) drives the unfloored posterior variance
        # to ~1.8e-13, far below the default min_covar=1e-8.
        prior = NormalGammaDistribution(0.0, 1.0, 1.0, 1.0e-12)
        estimator = LogGaussianEstimator(prior=prior)
        acc = LogGaussianAccumulator()
        # seq_update takes ALREADY-log-space values (the real encoder applies np.log before
        # seq_update ever sees them), unlike the scalar update() path, which logs raw x itself --
        # zeros here means the underlying raw x is exp(0) == 1.0 for all ten observations.
        acc.seq_update(np.zeros(10), np.ones(10), None)
        dist = estimator.estimate(None, acc.value())
        repairs = dist.numerical_repairs()
        self.assertIsInstance(repairs, tuple)
        self.assertTrue(repairs)
        self.assertTrue(any("variance-floored" in note for note in repairs), repairs)


class StackedHistogramSufficientStatisticsTest(unittest.TestCase):
    """Finding #3: generated_stacked_sufficient_statistics must branch on histogram-kind statistics.

    NegativeBinomialDistribution is the only histogram-kind distribution in the codebase (its
    dispersion MLE needs the raw count histogram, not a fixed-width moment). Before the fix, the
    stacked route misapplied ``_weighted_component_sum`` -- built for moment-shaped statistics -- to
    the per-row count array instead of folding it into a per-component ``{value: weight}`` histogram
    via the new ``_weighted_stacked_histogram`` helper. This pins the fixed dict-of-dicts shape and
    checks each component's histogram against the single-distribution ground truth computed with
    that component's own weight column.
    """

    def test_stacked_histogram_matches_per_component_ground_truth(self):
        dist = NegativeBinomialDistribution(r=5.0, p=0.4)
        counts = dist.sampler(seed=0).sample(200)
        enc = dist.dist_to_encoder().seq_encode(counts)
        enc_payload = getattr(enc, "engine_payload", enc)

        rng = np.random.default_rng(0)
        n_components = 3
        weights = rng.dirichlet(np.ones(n_components), size=200)

        params = generated_stacked_params([dist, dist, dist], NUMPY_ENGINE)
        stacked_stats = generated_stacked_sufficient_statistics(enc_payload, weights, params, NUMPY_ENGINE)

        histograms = stacked_stats[2]
        self.assertIsInstance(histograms, tuple)
        self.assertEqual(len(histograms), n_components)

        for k in range(n_components):
            with self.subTest(component=k):
                self.assertIsInstance(histograms[k], dict)
                ground_truth = generated_sufficient_statistics(dist, enc_payload, weights[:, k], NUMPY_ENGINE)
                ground_truth_histogram = ground_truth[2]
                self.assertIsInstance(ground_truth_histogram, dict)
                self.assertEqual(set(histograms[k].keys()), set(ground_truth_histogram.keys()))
                for value, expected_weight in ground_truth_histogram.items():
                    self.assertAlmostEqual(histograms[k][value], expected_weight, places=10)


if __name__ == "__main__":
    unittest.main()
