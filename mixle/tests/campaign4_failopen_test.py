"""Auto-inference no longer returns a silently-wrong model for a dirty cell or an infinity.

Two campaign-four findings, one shape: ``optimize(data)`` with no estimator produced a model that was
wrong rather than refusing or disclosing, and every diagnostic the library offers reported health.

T4-01 -- ONE non-numeric cell retyped the whole column. 300 finite floats plus a single ``"N/A"``
resolved to a categorical over the 301 observed values (inside ``IgnoredDistribution`` once the
identifier thresholds were crossed): a memorization table that scored the sample mean at ``-inf``,
along with 50/50 held-out points, while ``fit_provenance()`` reported ``converged=True``,
``repairs=()`` and a plausible finite objective. The mechanism was broader than a single reproduction
-- 50 rows plus ``"N/A"`` produced a bare ``CategoricalDistribution`` with the same ``-inf``, and 300
rows over 109 distinct values scored a SEEN value finitely while still giving ``-inf`` at 123.456.
The column now fits a typed dispatch mixture: the numbers get whatever family those same numbers get
on their own, the strings get their categorical, and the branch weights are the observed type
proportions. On the reproduction the result is bit-identical to spelling the dirty cell ``nan``,
which is the answer the library already called correct.

T4-02 -- a float ``inf`` was absorbed as a ZERO-COST missing sentinel. The wrapper reported
``.p == 0.0`` ("an infinity never occurs") while scoring ``log_density(inf) == 0.0`` (probability
one) on top of an unscaled base: total mass 2.0, and the sentinel row free in the objective, so 300
draws plus one inf produced a ``final_objective`` bit-identical to the clean 300-point fit at
``n_observations=301``. The inflation grows without bound in the number of infinities, and the paired
comparison tests the ``optimize`` docstring points callers to consume plain arrays, so they cannot
see the ``LIKELIHOOD_FACTOR`` flag that would have warned them. The sentinel now carries a fitted
rate, exactly as ``None`` and ``nan`` already did.

The overreach direction is tested too: a genuinely categorical column that mixes strings and small
integers still fits as one categorical, an identifier-like numeric side is not split, and ``None`` /
``nan`` are pinned to the exact values they produced before this repair.
"""

import io
import math
import unittest

import numpy as np
import pytest

import mixle
from mixle.inference import optimize
from mixle.stats import GaussianEstimator, OptionalEstimator
from mixle.stats.combinator.select import SelectDistribution
from mixle.stats.compute.pdist import DensitySemantics
from mixle.utils.automatic import analyze_structure, get_dpm_mixture, get_estimator
from mixle.utils.serialization import from_json, to_json, trusted_deserialization


def _draws(seed=11, n=300, loc=50.0, scale=10.0):
    return [round(float(x), 4) for x in np.random.RandomState(seed).normal(loc, scale, n)]


class DirtyCellDoesNotRetypeTheColumnTest(unittest.TestCase):
    """T4-01: a stray non-numeric value no longer collapses a continuous column into a lookup table."""

    def setUp(self):
        self.train = _draws()
        self.test = [round(float(x), 4) for x in np.random.RandomState(11).normal(50, 10, 350)][300:]

    def test_held_out_numbers_score_finitely(self):
        # Before: 50/50 held-out points scored -inf, the sample mean included.
        model = optimize(self.train + ["N/A"], out=None)
        self.assertIsInstance(model, SelectDistribution)
        self.assertTrue(np.isfinite(model.log_density(50.0)))
        self.assertEqual(sum(1 for x in self.test if model.log_density(x) == -np.inf), 0)

    def test_the_fit_equals_the_nan_spelling_of_the_same_dirty_cell(self):
        """The strongest statement available: the repaired model IS the already-correct one.

        ``optimize(train + [nan])`` was always right -- a Gaussian over the 300 numbers beside a
        sentinel atom carrying rate 1/301. Routing the string by type produces that same object with
        a categorical in place of the atom, so the numeric density and the training objective agree
        to the bit rather than merely to a tolerance.
        """
        dirty = optimize(self.train + ["N/A"], out=None)
        spelled_missing = optimize(self.train + [float("nan")], out=None)
        self.assertEqual(dirty.log_density(50.0), spelled_missing.log_density(50.0))
        self.assertEqual(dirty.log_density("N/A"), spelled_missing.log_density(float("nan")))
        self.assertEqual(
            dirty.fit_provenance().final_objective,
            spelled_missing.fit_provenance().final_objective,
        )
        clean = optimize(self.train, out=None)
        self.assertEqual(dirty.dists[1].mu, clean.mu)
        self.assertEqual(dirty.dists[1].sigma2, clean.sigma2)

    def test_string_branch_density_matches_the_merged_categorical(self):
        """log(n_str/n) + log(count/n_str) == log(count/n): nothing that scored finitely moved.

        The identity is exact in real arithmetic and correct to within a ULP in floating point --
        the mixture evaluates it as two logs and an addition where the merged categorical evaluated
        one log -- so this pins the value, not the rounding.
        """
        model = optimize(self.train + ["N/A"], out=None)
        self.assertAlmostEqual(model.log_density("N/A"), math.log(1.0 / 301.0), places=12)
        self.assertEqual(model.log_density("never-observed"), -np.inf)

    def test_low_cardinality_column_without_the_ignored_wrapper(self):
        """The verifier's demolition attempt: the defect is not the Ignored wrapper.

        At 50 rows the identifier thresholds are not crossed and the old result was a bare
        ``CategoricalDistribution``, with the same ``-inf`` at 50.0. Fixing only the wrapper would
        have left this case broken.
        """
        model = optimize(_draws(seed=3, n=50) + ["N/A"], out=None)
        self.assertTrue(np.isfinite(model.log_density(50.0)))

    def test_mid_cardinality_column_scores_unseen_numbers(self):
        """300 rows over 109 distinct values: a SEEN value scored finitely before, 123.456 did not."""
        values = [float(v) for v in np.random.RandomState(5).randint(0, 109, 300)]
        model = optimize(values + ["N/A"], out=None)
        self.assertTrue(np.isfinite(model.log_density(values[0])))
        self.assertTrue(np.isfinite(model.log_density(123.456)))

    def test_a_stray_number_in_a_string_column_splits_the_same_way(self):
        """Symmetric direction, deliberately not gated on which type is the majority."""
        model = optimize(["cat", "dog", "bird"] * 100 + [7.5], out=None)
        self.assertIsInstance(model, SelectDistribution)
        self.assertTrue(np.isfinite(model.log_density(9.9)))
        self.assertAlmostEqual(model.log_density("cat"), math.log(100.0 / 301.0), places=12)

    def test_several_distinct_dirty_markers_in_one_column(self):
        markers = ["N/A", "NULL", "", "?", "-", "n/a"]
        model = optimize(self.train + markers, out=None)
        self.assertTrue(np.isfinite(model.log_density(50.0)))
        for marker in markers:
            self.assertAlmostEqual(model.log_density(marker), math.log(1.0 / 306.0), places=12)

    def test_the_model_is_a_normalized_law_that_samples(self):
        model = optimize(self.train + ["N/A"], out=None)
        self.assertEqual(model.density_semantics(), DensitySemantics.EXACT)
        mass = float(
            np.trapezoid(
                np.exp([model.log_density(float(x)) for x in np.linspace(-150.0, 250.0, 40001)]),
                np.linspace(-150.0, 250.0, 40001),
            )
        )
        self.assertAlmostEqual(mass + math.exp(model.log_density("N/A")), 1.0, places=6)
        draws = model.sampler(seed=1).sample(50)
        self.assertEqual(len(draws), 50)

    def test_the_fit_round_trips_through_the_safe_json_codec(self):
        """The router is a serializable TypeDispatch, not a lambda: deploy/load still works."""
        model = optimize(self.train + ["N/A"], out=None)
        with trusted_deserialization():
            restored = from_json(to_json(model))
        self.assertEqual(restored.log_density(50.0), model.log_density(50.0))
        self.assertEqual(restored.log_density("N/A"), model.log_density("N/A"))

    def test_propose_and_optimize_no_longer_disagree(self):
        """propose() used to refuse this data with a non-finite-objective error while optimize()
        returned the broken model. Both now build the same family."""
        proposed = mixle.propose(self.train + ["N/A"], fit=True)
        self.assertTrue(proposed.fitted)
        self.assertIn("SelectDistribution", repr(proposed))

    def test_dirichlet_process_mixture_fits_a_dirty_column(self):
        """Routing splits the evidence, so a component can be initialized with no string rows at all.

        Under the Bayesian path that hands an empty count map to the string branch, which the
        symmetric Dirichlet cannot widen ("empty categorical fitting requires a prior with an
        explicit finite support"). ``get_typed_mixture_estimator`` pins the observed labels in the
        prior so the empty branch estimates to the uniform over them instead.
        """
        model = get_dpm_mixture(self.train + ["N/A"], max_components=3, max_its=3, out=io.StringIO())
        self.assertTrue(np.isfinite(model.log_density(50.0)))
        self.assertTrue(np.isfinite(model.log_density("N/A")))

    def test_a_dirty_column_inside_a_record_scores_fresh_rows(self):
        rows = [(float(i % 17) + 0.5, "N/A" if i == 3 else float(i)) for i in range(200)]
        model = optimize(rows, structure="off", out=None)
        self.assertTrue(np.isfinite(model.log_density((1.5, 99.5))))

    def test_a_dirty_column_in_a_multi_column_table_scores_fresh_rows(self):
        """The headline path: load a CSV with one bad cell and call optimize().

        The leaf sits inside a CompositeDistribution, so before this repair the memorization table
        was one factor of the row density and drove the WHOLE row to -inf -- every fresh row, not
        just ones containing a value from the dirty column.
        """
        rng = np.random.RandomState(2)
        rows = [
            ("N/A" if i == 7 else float(rng.normal(0, 1)), float(rng.normal(5, 2)), float(rng.normal(1, 1)))
            for i in range(300)
        ]
        model = optimize(rows, out=None)
        self.assertTrue(np.isfinite(model.log_density((0.5, 5.5, 1.5))))

    def test_object_dtype_pandas_column_with_a_dirty_cell(self):
        pd = pytest.importorskip("pandas")  # optional dep: base CI envs ship only numpy+scipy

        series = pd.Series(self.train + ["N/A"], dtype=object)
        model = optimize(list(series), out=None)
        self.assertTrue(np.isfinite(model.log_density(50.0)))

    def test_profile_reports_the_typed_mixture_instead_of_ignored(self):
        """The report is load-bearing: 'ignored' is what skips predictive validation and refuses the
        column for pairwise encoding, so it must name what the estimator really does."""
        profile = analyze_structure(self.train + ["N/A"], pairwise=False).fields[0]
        self.assertEqual(profile.kind, "mixed_scalar")
        self.assertEqual(profile.recommendation, "typed_mixture")
        self.assertTrue(any("typed dispatch mixture" in note for note in profile.notes))
        self.assertFalse(any("left unmodeled" in note for note in profile.notes))


class GenuinelyCategoricalColumnsAreUntouchedTest(unittest.TestCase):
    """The overreach direction: the repair must not restructure columns that were already right."""

    def test_string_and_small_integer_codes_stay_one_categorical(self):
        """["low", 2, "high", 3] is a coding column, not a continuous measurement with dirt in it.

        The gate is ``_numeric_side_is_memorizing``: split only when the numbers want a family with
        support past the observed values. Two repeated small integers do not.
        """
        data = ["low", 2, "high", 3] * 40
        model = optimize(data, out=None)
        self.assertIn("Categorical", type(model).__name__)
        profile = analyze_structure(data, pairwise=False).fields[0]
        self.assertEqual(profile.kind, "mixed_categorical")
        self.assertEqual(profile.recommendation, "categorical")

    def test_identifier_like_numeric_side_behaves_as_it_does_without_the_strings(self):
        """A wide sparse integer ID column is a memorization table with or without a dirty cell; the
        invariant is that the mixed column agrees with the same column minus its strings."""
        ids = [i * 1000 for i in range(300)]
        with_dirt = optimize(ids + ["N/A"], out=None)
        without = optimize(ids, out=None)
        self.assertEqual(type(with_dirt).__name__, type(without).__name__)
        self.assertEqual(with_dirt.log_density(123), -np.inf)

    def test_pure_columns_are_unchanged(self):
        self.assertIn("Categorical", type(optimize(["a", "b", "c"] * 40, out=None)).__name__)
        self.assertEqual(type(optimize(_draws(seed=9), out=None)).__name__, "GaussianDistribution")
        self.assertEqual(type(get_estimator(_draws(seed=9))).__name__, "GaussianEstimator")

    def test_bool_numeric_string_mix_stays_frozen(self):
        """A bool/number ambiguity is still refused a guess -- True == 1 under Python equality."""
        model = optimize([True, 1, "a"] * 40, out=None)
        self.assertEqual(type(model).__name__, "IgnoredDistribution")

    def test_an_unrecognized_scalar_type_is_still_frozen_not_claimed_as_a_mixture(self):
        import datetime

        data = [1.0, 2.0, datetime.datetime(2020, 1, 1)] * 40
        self.assertEqual(type(get_estimator(data)).__name__, "IgnoredEstimator")
        profile = analyze_structure(data, pairwise=False).fields[0]
        self.assertEqual(profile.recommendation, "ignored")


class InfinityIsNotAFreePassTest(unittest.TestCase):
    """T4-02: a non-finite sentinel carries a fitted rate instead of costing zero nats."""

    def setUp(self):
        self.train = _draws()

    def test_the_sentinel_carries_the_fitted_rate(self):
        model = optimize(self.train + [float("inf")], out=None)
        self.assertTrue(model.has_p)
        self.assertEqual(model.p, 1.0 / 301.0)
        # Before: log_density(inf) == 0.0, i.e. probability ONE on the sentinel.
        self.assertEqual(model.log_density(float("inf")), math.log(1.0 / 301.0))

    def test_the_density_is_normalized_rather_than_mass_two(self):
        model = optimize(self.train + [float("inf")], out=None)
        clean = optimize(self.train, out=None)
        # Before: log_density(50.0) was bit-equal to the clean fit -- the base was passed through
        # unscaled while the sentinel also carried mass 1.
        self.assertNotEqual(model.log_density(50.0), clean.log_density(50.0))
        self.assertAlmostEqual(model.log_density(50.0), clean.log_density(50.0) + math.log(1.0 - model.p))
        self.assertEqual(model.density_semantics(), DensitySemantics.EXACT)

    def test_the_objective_no_longer_ignores_the_sentinel_rows(self):
        """Before: the inf variant's final_objective was bit-identical to the clean 300-point fit at
        n_observations=301. It now equals the nan spelling of the same data."""
        model = optimize(self.train + [float("inf")], out=None)
        clean = optimize(self.train, out=None)
        spelled_missing = optimize(self.train + [float("nan")], out=None)
        self.assertNotEqual(model.fit_provenance().final_objective, clean.fit_provenance().final_objective)
        self.assertEqual(
            model.fit_provenance().final_objective,
            spelled_missing.fit_provenance().final_objective,
        )

    def test_the_likelihood_inflation_is_gone_at_scale(self):
        """With 30 infinities the free gain measured 100.5 nats against the proper competitor."""
        data = self.train + [float("inf")] * 30
        auto = optimize(data, out=None)
        proper = optimize(
            data,
            OptionalEstimator(GaussianEstimator(), missing_value=float("inf"), est_prob=True),
            out=None,
        )
        auto_scores = np.array([auto.log_density(x) for x in data])
        proper_scores = np.array([proper.log_density(x) for x in data])
        self.assertAlmostEqual(float(auto_scores.sum()), float(proper_scores.sum()), places=9)
        self.assertAlmostEqual(auto.p, 30.0 / 330.0)

    def test_paired_comparison_no_longer_decides_for_the_improper_model(self):
        """The named harm vector: clarke_test preferred the unnormalized model on all 301 points
        (p = 4.9e-91) because those helpers take plain arrays and cannot see density_semantics()).

        clarke_test is a SIGN test: it counts how many of 301 points favor A, which makes its
        p-value a measure of DIRECTIONAL CONSISTENCY, not magnitude. Two independently-fitted
        models computed via separate optimize() calls virtually always carry a consistent
        ulp-level bias in one direction (BLAS operation ordering is deterministic, just not
        identical across implementations), so at n=301 the sign test reports an astronomically
        small p-value REGARDLESS of whether the actual numerical agreement is meaningful --
        measured on this exact pair on ubuntu/OpenBLAS: p=3.6e-12 with the log-density arrays
        agreeing to 1e-9 relative, a difference many orders of magnitude below anything a sign
        test can distinguish from a real modeling difference. A p-value threshold on clarke_test
        is therefore not a usable regression check here, at any threshold: the original pathology
        (p=4.9e-91, a ~30+ nat systematic gap) and ordinary BLAS noise (p~1e-12, a <1e-9 relative
        gap) are both "significant" by the same test, for entirely different reasons. The actual
        claim -- that the fix closes the numerical gap, not that a sign test can no longer tell
        the two models apart -- is what test_the_likelihood_inflation_is_gone_at_scale already
        asserts the right way for the n=30 case; assert it directly here too, and drop clarke_test
        from this test entirely rather than parametrize a threshold that cannot mean anything.
        """
        data = self.train + [float("inf")]
        auto = optimize(data, out=None)
        proper = optimize(
            data,
            OptionalEstimator(GaussianEstimator(), missing_value=float("inf"), est_prob=True),
            out=None,
        )
        auto_scores = np.array([auto.log_density(x) for x in data])
        proper_scores = np.array([proper.log_density(x) for x in data])
        self.assertAlmostEqual(float(auto_scores.sum()), float(proper_scores.sum()), places=9)
        np.testing.assert_allclose(auto_scores, proper_scores, rtol=1e-9, atol=1e-9)

    def test_both_signs_and_the_nested_case(self):
        data = self.train + [float("inf"), float("-inf")]
        model = optimize(data, out=None)
        self.assertTrue(np.isfinite(model.log_density(50.0)))
        self.assertTrue(np.isfinite(model.log_density(float("inf"))))
        self.assertTrue(np.isfinite(model.log_density(float("-inf"))))
        total = (
            math.exp(model.log_density(float("inf")))
            + math.exp(model.log_density(float("-inf")))
            + math.exp(model.log_density(float("inf"))) * 0.0
        )
        self.assertLess(total, 1.0)

    def test_an_all_infinite_column_still_puts_all_its_mass_there(self):
        model = optimize([float("inf")] * 5, out=None)
        self.assertEqual(model.log_density(float("inf")), 0.0)  # p == 1.0, so log(p) == 0.0


class MissingSpellingsAreUnchangedTest(unittest.TestCase):
    """The other overreach direction: None and nan must keep working exactly as they did."""

    def setUp(self):
        self.train = _draws()

    def test_nan_and_none_produce_the_documented_values(self):
        for sentinel in (float("nan"), None):
            with self.subTest(sentinel=sentinel):
                model = optimize(self.train + [sentinel], out=None)
                self.assertEqual(model.p, 1.0 / 301.0)
                self.assertEqual(model.log_density(sentinel), math.log(1.0 / 301.0))
                self.assertEqual(model.density_semantics(), DensitySemantics.EXACT)

    def test_every_sentinel_spelling_now_agrees(self):
        """None, nan and inf are three spellings that arrive interchangeably (pandas coercion,
        overflow, json.loads('[1e999]')); after this repair they cost the same."""
        scores = {
            name: optimize(self.train + [value], out=None).log_density(value)
            for name, value in (("none", None), ("nan", float("nan")), ("inf", float("inf")))
        }
        self.assertEqual(len(set(scores.values())), 1)


if __name__ == "__main__":
    unittest.main()
