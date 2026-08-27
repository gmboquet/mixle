"""Auto-inference on constant and large-offset numeric columns (campaign-three T4-01).

``optimize(data)`` / ``fit(data)`` with ``estimator=None`` -- the documented one-argument entry point
-- returned grossly wrong parameters while its ``fit_provenance()`` asserted a converged MLE fit with
an empty ``numerical_repairs()``:

* ``optimize([300.0] * 10)`` returned ``GaussianDistribution(272.7272728181818, 7438.016479429752)``
  for a column whose MLE is ``(300, 0)``, and ``optimize([5.0] * 100)`` returned
  ``(4.950495059405941, 0.24507392422312518)``.
* ``optimize(1e12 + N(0, 1))`` returned mean ``998003992015.9713`` (exactly 500/501 of the truth) and
  variance ``1.9920239361596297e+21``; at 1e9 the variance came back ``1.890656264769758`` against a
  sample variance of ``0.8704375772992959``; at 1e15 the selected family flipped to
  ``ParetoDistribution(999999999999997.0, 501000000000000.0)``.

Three causes, all in the auto-inference layer:

1. ``get_gaussian_estimator`` / ``get_lognormal_estimator`` seeded their prior from moments taken in
   the cancelling ``E[x^2] - E[x]^2`` form, which returns noise at an offset of 1e9 and a negative
   number at 1e12 -- even though the Gaussian accumulator downstream had already been repaired to
   compute its own scatter shift-invariantly.
2. A non-positive computed spread fell back to a pseudo-observation at ``(1e-6, 1e-6)``. That is not
   "no information": it is an observation at the ORIGIN, which dragged the mean toward zero by
   n/(n+1) and inflated the variance by the squared distance from the data to zero. The estimator's
   own spelling for "no pseudo-observations" is ``suff_stat=(None, None)``, under which it takes the
   plain MLE and DISCLOSES its variance floor.
3. Family selection compared code lengths computed at wildly different numerical conditioning. The
   Pareto's NLL subtracted two terms of size ``alpha*log(xm)`` ~ 3e18 from each other and kept the
   residue, which came out negative; other origin-anchored families lose the same way through their
   scipy-backed fits (plain ``N(1e7, 1)`` drew a gamma "win", a 200-row ramp at 1.7e9 an
   inverse-gamma one).

``_integer_moments`` carried the same cancelling form over integers, where ``k*k`` for an epoch
timestamp is past 2**53: the spread of an ordinary column came out wrong by thousands at second
resolution and collapsed to exactly 0.0 at millisecond resolution.

Guard-overreach watch: the last test pins that ordinary well-scaled columns still select exactly the
families they selected before this repair.
"""

import math
import unittest

import numpy as np

import mixle.inference as inf
from mixle.stats.univariate import GaussianDistribution
from mixle.utils.automatic import get_estimator
from mixle.utils.automatic.factories import _anchored_moments, get_gaussian_estimator
from mixle.utils.automatic.profiling import (
    DatumNode,
    _numeric_candidate_bics,
    _numeric_model_recommendation,
    _origin_anchored_scores_unmeasurable,
)

# The explicit-prototype path was correct on every one of these inputs, so it is the reference the
# auto path has to meet -- not a hand-copied constant.
_EXPLICIT = GaussianDistribution(0.0, 1.0)


def _offset_sample(offset, seed=21, n=500):
    base = np.random.default_rng(seed).normal(0.0, 1.0, n)
    return [float(offset + value) for value in base]


def _exact_moments(values):
    """Mean and (population) variance in exact rational arithmetic."""
    from fractions import Fraction

    exact = [Fraction(value) for value in values]
    mean = sum(exact) / len(exact)
    var = sum((value - mean) ** 2 for value in exact) / len(exact)
    return float(mean), float(var)


class ConstantColumnTest(unittest.TestCase):
    """A constant column gets the MLE with a DISCLOSED floor, not a confident wrong parameter."""

    def test_constant_column_returns_the_constant_and_discloses_the_floor(self):
        for value, count in ((300.0, 10), (5.0, 100), (-12.5, 10), (0.0, 20)):
            with self.subTest(value=value, count=count):
                model = inf.optimize([value] * count, max_its=3, out=None)
                # Before: 272.7272728181818 for 300.0, 4.950495059405941 for 5.0 -- the fitted mean
                # was pulled toward the origin by exactly count/(count+1).
                self.assertEqual(model.mu, value)
                self.assertGreater(model.sigma2, 0.0)
                # Before: repairs=() while the variance was 7438.016479429752 for a constant column.
                self.assertEqual(len(model.numerical_repairs()), 1)
                self.assertIn("variance-floored", model.numerical_repairs()[0])
                self.assertEqual(model.fit_provenance().repairs, model.numerical_repairs())

    def test_constant_column_matches_the_explicit_prototype_path(self):
        for value, count in ((300.0, 10), (5.0, 100)):
            with self.subTest(value=value, count=count):
                auto = inf.optimize([value] * count, max_its=3, out=None)
                explicit = inf.optimize([value] * count, max_its=3, estimator=_EXPLICIT, out=None)
                self.assertEqual(auto.mu, explicit.mu)
                self.assertEqual(auto.sigma2, explicit.sigma2)
                self.assertEqual(auto.numerical_repairs(), explicit.numerical_repairs())

    def test_constant_column_prior_carries_no_pseudo_observation_at_the_origin(self):
        # The factory-level statement of the same repair: no scale to seed a prior from means no
        # prior, not a pseudo-observation at 1e-6.
        estimator = get_gaussian_estimator({300.0: 10.0}, pseudo_count=1.0)
        self.assertEqual(estimator.suff_stat, (None, None))
        self.assertEqual(estimator.pseudo_count, (1.0, 1.0))


class LargeOffsetTest(unittest.TestCase):
    """Epoch- and price-scale offsets keep the fit and the family."""

    def test_offset_fit_matches_exact_moments(self):
        for offset in (1.0e9, 1.0e12):
            for seed in (21, 99, 12345):
                with self.subTest(offset=offset, seed=seed):
                    data = _offset_sample(offset, seed=seed)
                    exact_mean, exact_var = _exact_moments(data)
                    model = inf.optimize(data, max_its=3, out=None)
                    self.assertEqual(type(model).__name__, "GaussianDistribution")
                    # Before at 1e12: mean 998003992015.9713, variance 1.9920239361596297e+21.
                    self.assertLessEqual(abs(model.mu - exact_mean) / abs(exact_mean), 1.0e-15)
                    # Before at 1e9: variance 1.890656264769758 against an exact 0.870437577299296.
                    self.assertLessEqual(abs(model.sigma2 - exact_var) / exact_var, 1.0e-7)

    def test_offset_fit_tracks_the_explicit_prototype_path(self):
        for offset in (1.0e9, 1.0e12, 1.0e15):
            with self.subTest(offset=offset):
                data = _offset_sample(offset)
                auto = inf.optimize(data, max_its=3, out=None)
                explicit = inf.optimize(data, max_its=3, estimator=_EXPLICIT, out=None)
                self.assertEqual(type(auto).__name__, type(explicit).__name__)
                self.assertEqual(auto.mu, explicit.mu)
                # The auto path applies get_estimator's pseudo_count=1.0 with the sample's own
                # moments as the prior, which reproduces the MLE up to the rounding of recomputing
                # them; at 1e15 the float grid itself costs the variance ~2e-6 relative on both paths.
                self.assertLessEqual(abs(auto.sigma2 - explicit.sigma2) / explicit.sigma2, 1.0e-8)

    def test_large_offset_gaussian_data_stays_gaussian(self):
        # Before at 1e15: ParetoDistribution(999999999999997.0, 501000000000000.0).
        for offset in (1.0e7, 1.0e9, 1.0e12, 1.0e15):
            with self.subTest(offset=offset):
                data = _offset_sample(offset)
                self.assertEqual(type(get_estimator(data)).__name__, "GaussianEstimator")

    def test_anchored_moments_are_shift_invariant(self):
        # The comparison is against the EXACT moments of the shifted values, not against the
        # unshifted sample's: adding 1e12 moves the data onto a coarser float grid, so the sample
        # genuinely changes. What must not change is that the computed spread is the spread of the
        # values actually handed over.
        base = np.random.default_rng(21).normal(0.0, 1.0, 500)
        for offset in (0.0, 1.0e6, 1.0e9, 1.0e12, 1.0e15):
            with self.subTest(offset=offset):
                values = [float(offset + value) for value in base]
                vdict = {}
                for value in values:
                    vdict[value] = vdict.get(value, 0.0) + 1.0
                total, mean, var = _anchored_moments(list(vdict.items()))
                exact_mean, exact_var = _exact_moments(values)
                self.assertEqual(total, 500.0)
                self.assertLessEqual(abs(mean - exact_mean) / max(abs(exact_mean), 1.0), 1.0e-15)
                # Before, the same quantity at 1e12 was NEGATIVE, which is what sent the factory
                # into its origin-seeded fallback; at 1e9 it was off by ~500 in absolute terms.
                self.assertGreater(var, 0.0)
                self.assertLessEqual(abs(var - exact_var) / exact_var, 1.0e-12)

    def test_integer_moments_are_shift_invariant(self):
        base = np.random.default_rng(5).integers(0, 86400, 400)
        reference = None
        for offset in (0, 1_700_000_000, 1_700_000_000_000):
            values = [int(offset + value) for value in base]
            _, mean, var, min_val, max_val, width = DatumNode(data=values)._integer_moments()
            self.assertEqual(min_val, offset + int(base.min()))
            self.assertEqual(max_val, offset + int(base.max()))
            self.assertEqual(width, int(base.max() - base.min()) + 1)
            self.assertLessEqual(abs(mean - offset - float(base.mean())) / max(offset, 1.0), 1.0e-15)
            if reference is None:
                reference = var
            else:
                # Before: 612999168.0 at 1.7e9 and exactly 0.0 at 1.7e12, against 612997114.3186002.
                self.assertEqual(var, reference)


class OriginAnchoredScoringTest(unittest.TestCase):
    """Origin-anchored families are scored honestly or not offered."""

    def test_pareto_code_length_is_offset_stable(self):
        from mixle.utils.automatic.detectors import get_detector

        pareto = get_detector("pareto")
        base = np.random.default_rng(11).exponential(1.0, 300) - 1.0
        reference = None
        for offset in (1.0e2, 1.0e6, 1.0e9, 1.0e12):
            arr = np.asarray([float(offset + value) for value in base])
            score = pareto.score(arr, arr.size)
            self.assertIsNotNone(score)
            if reference is None:
                reference = score
            else:
                # Before at 1e15 the same computation returned -2.867458513208603 bits/obs -- below
                # the Gaussian's 1.967 on plain N(1e15, 1) data -- because two ~3e18 terms cancelled.
                self.assertLessEqual(abs(score - reference), 1.0e-3)

    def test_offset_dominated_data_does_not_pick_an_origin_anchored_family(self):
        rng = np.random.default_rng(11)
        cases = {
            "normal@1e7": [float(1.0e7 + v) for v in rng.normal(0.0, 1.0, 300)],
            "ramp@1.7e9": [float(1.7e9 + i) for i in range(200)],
            "ramp@1e10": [float(1.0e10 + i) for i in range(200)],
        }
        origin_anchored = {"lognormal", "gamma", "inverse_gamma", "inverse_gaussian", "weibull", "rayleigh"}
        for name, data in cases.items():
            with self.subTest(case=name):
                arr = np.asarray(data)
                self.assertTrue(_origin_anchored_scores_unmeasurable(arr))
                chosen = _numeric_model_recommendation(_numeric_candidate_bics(arr, arr.size))
                # Before: "gamma" for normal@1e7 and "inverse_gamma" for both ramps, each producing
                # a fit whose spread was 10x-30x the data's with numerical_repairs() empty.
                self.assertNotIn(chosen, origin_anchored)

    def test_offset_gate_keeps_a_family_that_can_represent_the_data(self):
        data = [float(1.0e7 + v) for v in np.random.default_rng(11).normal(0.0, 1.0, 300)]
        arr = np.asarray(data)
        model = inf.optimize(data, max_its=10, out=None)
        self.assertEqual(type(model).__name__, "GaussianDistribution")
        self.assertLessEqual(abs(model.mu - float(arr.mean())) / float(arr.mean()), 1.0e-12)
        self.assertLessEqual(abs(math.sqrt(model.sigma2) - float(arr.std())) / float(arr.std()), 0.05)

    def test_gate_spares_shifted_exponential_data_that_genuinely_wants_the_pareto(self):
        # The Pareto declares its score offset-stable, so the conditioning gate does not remove it:
        # shifted-exponential data far from the origin still selects the family that fits it best.
        base = np.random.default_rng(11).exponential(1.0, 300) - 1.0
        for offset in (1.0e2, 1.0e6, 1.0e9, 1.0e12):
            with self.subTest(offset=offset):
                arr = np.asarray([float(offset + value) for value in base])
                bics = _numeric_candidate_bics(arr, arr.size)
                self.assertEqual(_numeric_model_recommendation(bics), "pareto")

    def test_gate_is_inert_on_well_scaled_data(self):
        rng = np.random.default_rng(20260827)
        for name, arr in (
            ("normal", rng.normal(0.0, 1.0, 400)),
            ("lognormal", np.exp(rng.normal(0.0, 1.0, 400))),
            ("gamma", rng.gamma(2.0, 3.0, 400)),
            ("prices", rng.uniform(1.0, 500.0, 400)),
            ("offset_1e6", 1.0e6 + rng.normal(0.0, 1.0, 400)),
        ):
            with self.subTest(case=name):
                self.assertFalse(_origin_anchored_scores_unmeasurable(np.asarray(arr)))


class OrdinaryColumnSelectionTest(unittest.TestCase):
    """Guard-overreach watch: ordinary columns select exactly the families they selected before.

    The expectations below were recorded by running this corpus against the tree as it stood before
    the T4-01 repair; every one of them is unchanged by it.
    """

    EXPECTED = {
        "beta22": "BetaEstimator",
        "bimodal": "MixtureEstimator",
        "binom_ints": "BinomialEstimator",
        "bools": "CategoricalEstimator",
        "exponential": "GammaEstimator",
        "gamma_k2": "WeibullEstimator",
        "geom_ints": "GeometricEstimator",
        "gumbel": "GumbelEstimator",
        "halfnormal": "HalfNormalEstimator",
        "id_ints": "IgnoredEstimator",
        "inv_gauss": "LogGaussianEstimator",
        "laplace": "LaplaceEstimator",
        "logistic": "LogisticEstimator",
        "lognormal": "LogGaussianEstimator",
        "mid_offset_1e4": "GaussianEstimator",
        "mid_offset_1e6": "GaussianEstimator",
        "nbinom_ints": "NegativeBinomialEstimator",
        "normal_n30": "GaussianEstimator",
        "normal_shift": "GaussianEstimator",
        "normal_small": "GaussianEstimator",
        "normal_std": "GaussianEstimator",
        "pareto_true": "ParetoEstimator",
        "percent": "GeneralizedGaussianEstimator",
        "poisson_ints": "PoissonEstimator",
        "prices": "GeneralizedGaussianEstimator",
        "rayleigh": "RayleighEstimator",
        "skewnormal": "ExponentiallyModifiedGaussianEstimator",
        "small_card_ints": "BetaBinomialEstimator",
        "str_ids": "IgnoredEstimator",
        "strings": "CategoricalEstimator",
        "student_t3": "StudentTEstimator",
        "uniform01": "BetaEstimator",
        "weibull": "GammaEstimator",
        "wide_ints": "IgnoredEstimator",
    }

    @staticmethod
    def corpus():
        rng = np.random.default_rng(20260827)
        return {
            "normal_std": list(rng.normal(0.0, 1.0, 400)),
            "normal_shift": list(rng.normal(10.0, 2.0, 400)),
            "normal_small": list(rng.normal(0.0, 1.0e-4, 400)),
            "normal_n30": list(rng.normal(5.0, 1.0, 30)),
            "lognormal": list(np.exp(rng.normal(0.0, 1.0, 400))),
            "gamma_k2": list(rng.gamma(2.0, 3.0, 400)),
            "exponential": list(rng.exponential(2.0, 400)),
            "student_t3": list(rng.standard_t(3, 400)),
            "laplace": list(rng.laplace(0.0, 1.0, 400)),
            "logistic": list(rng.logistic(0.0, 1.0, 400)),
            "uniform01": list(rng.uniform(0.0, 1.0, 400)),
            "beta22": list(rng.beta(2.0, 2.0, 400)),
            "weibull": list(rng.weibull(1.5, 400)),
            "rayleigh": list(rng.rayleigh(2.0, 400)),
            "gumbel": list(rng.gumbel(0.0, 1.0, 400)),
            "pareto_true": list((rng.pareto(2.5, 400) + 1.0) * 3.0),
            "bimodal": list(np.concatenate([rng.normal(-4.0, 0.7, 200), rng.normal(4.0, 0.7, 200)])),
            "skewnormal": list(np.abs(rng.normal(0.0, 1.0, 400)) + rng.normal(0.0, 0.4, 400)),
            "halfnormal": list(np.abs(rng.normal(0.0, 2.0, 400))),
            "inv_gauss": list(rng.wald(1.0, 3.0, 400)),
            "prices": [round(float(v), 2) for v in rng.uniform(1.0, 500.0, 400)],
            "percent": [float(v) for v in rng.uniform(0.0, 100.0, 400)],
            "poisson_ints": [int(v) for v in rng.poisson(4.0, 400)],
            "nbinom_ints": [int(v) for v in rng.negative_binomial(5, 0.4, 400)],
            "geom_ints": [int(v) for v in rng.geometric(0.3, 400)],
            "binom_ints": [int(v) for v in rng.binomial(20, 0.3, 400)],
            "small_card_ints": [int(v) for v in rng.integers(0, 6, 400)],
            "wide_ints": [int(v) for v in rng.integers(0, 100000, 400)],
            "id_ints": list(range(1000, 1000 + 400 * 30, 30)),
            "bools": [bool(v) for v in rng.integers(0, 2, 400)],
            "strings": [["a", "b", "c", "d"][int(v)] for v in rng.integers(0, 4, 400)],
            "str_ids": ["id-%d" % i for i in range(400)],
            "mid_offset_1e6": [float(1.0e6 + v) for v in rng.normal(0.0, 1.0, 400)],
            "mid_offset_1e4": [float(1.0e4 + v) for v in rng.normal(0.0, 1.0, 400)],
        }

    def test_ordinary_columns_select_the_same_families(self):
        corpus = self.corpus()
        self.assertEqual(sorted(corpus), sorted(self.EXPECTED))
        for name, data in corpus.items():
            with self.subTest(column=name):
                self.assertEqual(type(get_estimator(data)).__name__, self.EXPECTED[name])


if __name__ == "__main__":
    unittest.main()
